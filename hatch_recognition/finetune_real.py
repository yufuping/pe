"""Fine-tune for real CAD scenes with interference (dimensions, grids, sparse walls)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

from pat_parser import COMMON_PATTERNS, parse_pat_file
from renderer import add_cad_crosshair, add_interference, render_pattern, suggest_scale
from train import HatchCNN, build_loaders, evaluate

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")

# Real user screenshots of AR-CONC with interference
REAL_AR_CONC = [
    ASSETS / "CE4D9686-1522-4EDE-A013-9618C6A66357_L0_001.jpg",
    ASSETS / "01a0d261-8064-7965-bbcb-9be9d02e1e5c.jpg",
    ASSETS / "01a0d262-d1c9-742e-9e6e-26968ab74972.jpg",
    ASSETS / "01a0d263-d93e-7705-873f-c3548132d977.jpg",
    ASSETS / "01a0d264-a8fa-7635-a753-8ac1bd36e145.jpg",
    ASSETS / "01a0d265-ce06-7d31-a510-70df7088e275.jpg",
]


def _augment_real(img: Image.Image, rng: np.random.Generator, size: int = 128) -> Image.Image:
    g = img.convert("L")
    w, h = g.size
    # Prefer denser hatch crops when possible
    ink = np.array(g)
    side = int(min(w, h) * float(rng.uniform(0.35, 0.9)))
    side = max(64, min(side, min(w, h)))
    best = (0.0, 0, 0)
    for _ in range(12):
        x0 = int(rng.integers(0, max(1, w - side)))
        y0 = int(rng.integers(0, max(1, h - side)))
        patch = ink[y0 : y0 + side, x0 : x0 + side]
        dens = float(((patch < 200) & (patch > 5)).mean())
        if dens > best[0]:
            best = (dens, x0, y0)
    _, x0, y0 = best
    crop = g.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.LANCZOS)

    if rng.random() < 0.45:
        crop = crop.rotate(float(rng.uniform(-25, 25)), fillcolor=255, expand=False)
    if rng.random() < 0.3:
        crop = ImageOps.mirror(crop)
    if rng.random() < 0.3:
        crop = ImageOps.flip(crop)
    if rng.random() < 0.5:
        crop = ImageEnhance.Contrast(crop).enhance(float(rng.uniform(0.85, 1.25)))
    if rng.random() < 0.4:
        crop = ImageEnhance.Brightness(crop).enhance(float(rng.uniform(0.9, 1.12)))
    if rng.random() < 0.35:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8))))
    if rng.random() < 0.55:
        crop = add_interference(
            crop, rng=rng, noise=0.02, lines=True, blur=False, crosshair_chance=0.45
        )
    return crop


def _paste_hatch_in_scene(
    hatch: Image.Image,
    rng: np.random.Generator,
    canvas: int = 256,
    size: int = 128,
) -> Image.Image:
    """Place hatch in a thin wall / U / L region on a large white CAD-like sheet."""
    sheet = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(sheet)
    mask = Image.new("L", (canvas, canvas), 0)
    mdraw = ImageDraw.Draw(mask)

    kind = int(rng.integers(0, 4))
    m = int(canvas * 0.08)
    t = int(canvas * float(rng.uniform(0.12, 0.28)))  # wall thickness
    if kind == 0:  # U
        mdraw.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        mdraw.rectangle([m, m, m + t, canvas - m], fill=255)
        mdraw.rectangle([canvas - m - t, m, canvas - m, canvas - m], fill=255)
    elif kind == 1:  # L
        mdraw.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        mdraw.rectangle([m, m, m + t, canvas - m], fill=255)
    elif kind == 2:  # thin horizontal strip
        y0 = int(rng.integers(m, canvas - m - t))
        mdraw.rectangle([m, y0, canvas - m, y0 + t], fill=255)
    else:  # irregular polygon blob
        cx, cy = canvas / 2, canvas / 2
        n = int(rng.integers(6, 10))
        pts = []
        for i in range(n):
            a = i * 2 * np.pi / n + float(rng.uniform(-0.2, 0.2))
            r = canvas * float(rng.uniform(0.22, 0.42))
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        mdraw.polygon(pts, fill=255)

    tiled = Image.new("L", (canvas, canvas), 255)
    hw, hh = hatch.size
    for yy in range(0, canvas, hh):
        for xx in range(0, canvas, hw):
            tiled.paste(hatch, (xx, yy))
    sheet.paste(tiled, mask=mask)

    # Interference: dimension-like ticks, centerlines, stairs, circles
    if rng.random() < 0.7:
        for _ in range(int(rng.integers(1, 4))):
            x = int(rng.integers(0, canvas))
            draw.line([(x, 0), (x, canvas)], fill=0, width=1)
        for _ in range(int(rng.integers(1, 3))):
            y = int(rng.integers(0, canvas))
            # dash-dot-ish centerline approximation
            for x in range(0, canvas, 12):
                draw.line([(x, y), (x + 6, y)], fill=40, width=1)
    if rng.random() < 0.4:
        # stairs-like parallel lines
        x0 = int(rng.integers(canvas // 3, 2 * canvas // 3))
        for i in range(8):
            y = int(canvas * 0.15) + i * 8
            draw.line([(x0, y), (x0 + 40, y)], fill=0, width=1)
    if rng.random() < 0.5:
        sheet = add_cad_crosshair(sheet, rng)
    if rng.random() < 0.35:
        r = int(rng.integers(8, 28))
        cx = int(rng.integers(r, canvas - r))
        cy = int(rng.integers(r, canvas - r))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=2)

    return sheet.resize((size, size), Image.Resampling.LANCZOS)


def append_scene_and_real_samples(out_dir: Path, size: int = 128, seed: int = 123) -> None:
    rng = np.random.default_rng(seed)
    patterns = parse_pat_file(ROOT / "acad_4270.pat")

    # 1) Synthetic scene samples for all classes (esp. AR-CONC)
    for name in COMMON_PATTERNS:
        pattern = patterns[name]
        dest = out_dir / "train" / name
        dest.mkdir(parents=True, exist_ok=True)
        n = 80 if name == "AR-CONC" else 35
        for i in range(n):
            if name == "AR-CONC":
                target = float(rng.uniform(10, 24))
            elif name == "GRAVEL":
                target = float(rng.uniform(10, 22))
            else:
                target = float(rng.uniform(7, 16))
            scale = float(suggest_scale(pattern, size=160, target_px=target) * rng.uniform(0.7, 1.4))
            hatch = render_pattern(
                pattern,
                size=160,
                scale=scale,
                rotation=float(rng.uniform(0, 360)),
                bg=255,
                fg=int(rng.choice([0, 0, 15])),
                stroke=1,
                shape="rect",
                supersample=2,
            )
            scene = _paste_hatch_in_scene(hatch, rng, canvas=280, size=size)
            if rng.random() < 0.5:
                scene = add_interference(
                    scene, rng=rng, noise=0.02, lines=False, blur=True, crosshair_chance=0.25
                )
            scene.save(dest / f"scene_{name}_{i:03d}.png")

    # 2) Inject all real user AR-CONC screenshots
    dest = out_dir / "train" / "AR-CONC"
    vdest = out_dir / "val" / "AR-CONC"
    dest.mkdir(parents=True, exist_ok=True)
    vdest.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for path in REAL_AR_CONC:
        if not path.exists():
            continue
        real = Image.open(path)
        for i in range(40):
            _augment_real(real, rng, size=size).save(dest / f"real_{path.stem}_{i:03d}.png")
        for i in range(6):
            _augment_real(real, rng, size=size).save(vdest / f"real_{path.stem}_val_{i:03d}.png")
        n_ok += 1
    print(f"Injected augments from {n_ok} real AR-CONC screenshots")


def finetune(epochs: int = 12, lr: float = 1.5e-4) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "dataset"
    model_dir = ROOT / "models"
    append_scene_and_real_samples(data_dir, size=128)

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
        correct = total = 0
        pbar = tqdm(train_loader, desc=f"scene-ft {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(acc=correct / total)
        sched.step()
        _, val_acc = evaluate(model, val_loader, device)
        print(f"scene-ft {epoch}: train={correct/total:.3f} val={val_acc:.3f}")
        if val_acc >= best - 0.015:
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
            print("  saved")

    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc = evaluate(model, test_loader, device)
    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["test_acc_scene_ft"] = test_acc
    metrics["best_val_acc"] = best
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print({"test_acc": test_acc, "best_val_acc": best})
    return {"test_acc": test_acc, "best_val_acc": best}


if __name__ == "__main__":
    finetune(epochs=12)
