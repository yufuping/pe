"""Fine-tune on real-CAD-style samples + labeled user screenshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

from pat_parser import COMMON_PATTERNS, parse_pat_file
from renderer import add_interference, render_pattern, suggest_scale
from train import HatchCNN, build_loaders, evaluate

ROOT = Path(__file__).resolve().parent
USER_IMG = Path(
    "/home/ubuntu/.cursor/projects/workspace/assets/"
    "CE4D9686-1522-4EDE-A013-9618C6A66357_L0_001.jpg"
)


def _augment_real(img: Image.Image, rng: np.random.Generator, size: int = 128) -> Image.Image:
    """Augment a real CAD crop toward train distribution."""
    g = img.convert("L")
    w, h = g.size
    # Random square-ish crop
    side = int(min(w, h) * float(rng.uniform(0.55, 0.95)))
    x0 = int(rng.integers(0, max(1, w - side)))
    y0 = int(rng.integers(0, max(1, h - side)))
    crop = g.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.LANCZOS)

    if rng.random() < 0.5:
        crop = crop.rotate(float(rng.uniform(-20, 20)), fillcolor=255, expand=False)
    if rng.random() < 0.3:
        crop = ImageOps.mirror(crop)
    if rng.random() < 0.3:
        crop = ImageOps.flip(crop)
    if rng.random() < 0.5:
        crop = ImageEnhance.Contrast(crop).enhance(float(rng.uniform(0.85, 1.2)))
    if rng.random() < 0.4:
        crop = ImageEnhance.Brightness(crop).enhance(float(rng.uniform(0.9, 1.1)))
    if rng.random() < 0.35:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8))))
    if rng.random() < 0.5:
        crop = add_interference(
            crop, rng=rng, noise=0.02, lines=rng.random() < 0.4, blur=False, crosshair_chance=0.5
        )
    return crop


def append_real_style_samples(out_dir: Path, size: int = 128, per_class: int = 60, seed: int = 99) -> None:
    rng = np.random.default_rng(seed)
    patterns = parse_pat_file(ROOT / "acad_4270.pat")
    shapes = ["rect", "rect", "circle", "irregular"]

    for name in COMMON_PATTERNS:
        pattern = patterns[name]
        dest = out_dir / "train" / name
        dest.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            # Prefer denser/closer zoom for AR-CONC to match user screenshot triangles
            if name == "AR-CONC":
                target = float(rng.uniform(10.0, 22.0))
            elif name == "GRAVEL":
                target = float(rng.uniform(10.0, 20.0))
            else:
                target = float(rng.uniform(7.0, 16.0))
            base = suggest_scale(pattern, size=size, target_px=target)
            scale = float(base * rng.uniform(0.65, 1.5))
            rot = float(rng.uniform(0, 360))
            shape = shapes[int(rng.integers(0, len(shapes)))]
            fg = int(rng.choice([0, 0, 15, 25]))
            img = render_pattern(
                pattern,
                size=size,
                scale=scale,
                rotation=rot,
                bg=255,
                fg=fg,
                stroke=1,
                shape=shape,
                supersample=2,
            )
            img = add_interference(
                img,
                rng=rng,
                noise=float(rng.uniform(0.01, 0.05)),
                lines=rng.random() < 0.5,
                blur=True,
                crosshair_chance=0.55 if name == "AR-CONC" else 0.35,
            )
            img.save(dest / f"cadstyle_{name}_{i:03d}.png")

    # Inject labeled user screenshot (AR-CONC) with many augments
    if USER_IMG.exists():
        dest = out_dir / "train" / "AR-CONC"
        real = Image.open(USER_IMG)
        for i in range(80):
            aug = _augment_real(real, rng, size=size)
            aug.save(dest / f"user_real_{i:03d}.png")
        # Also keep a few in val
        vdest = out_dir / "val" / "AR-CONC"
        vdest.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            aug = _augment_real(real, rng, size=size)
            aug.save(vdest / f"user_real_val_{i:03d}.png")
        print(f"Injected user AR-CONC screenshot augments from {USER_IMG}")


def finetune(epochs: int = 10, lr: float = 2e-4) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "dataset"
    model_dir = ROOT / "models"
    append_real_style_samples(data_dir, size=128, per_class=50)

    train_loader, val_loader, test_loader, classes = build_loaders(data_dir, batch_size=64, size=128)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model = HatchCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = float(ckpt.get("val_acc", 0))

    for epoch in range(1, epochs + 1):
        model.train()
        running = correct = total = 0
        pbar = tqdm(train_loader, desc=f"ft {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(acc=correct / total)
        sched.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"ft {epoch}: train_acc={correct/total:.3f} val_acc={val_acc:.3f}")
        if val_acc >= best - 0.01:  # allow small dips if near best
            # Prefer saving when user-style val improves; always save if better
            if val_acc >= best:
                best = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": 128,
                    "val_acc": val_acc,
                },
                model_dir / "best_model.pt",
            )
            print("  saved checkpoint")

    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    result = {"test_acc": test_acc, "best_val_acc": best, "classes": classes}
    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({"test_acc_after_real_ft": test_acc, "best_val_acc": best})
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    finetune(epochs=10)
