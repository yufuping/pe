"""Fine-tune specifically to fix GRAVEL recognition on real CAD screenshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

from pat_parser import parse_pat_file
from renderer import (
    add_cad_crosshair,
    add_interference,
    render_gravel_pebbles,
    render_pattern,
    suggest_scale,
)
from train import HatchCNN, build_loaders, evaluate

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")

REAL_GRAVEL = [
    ASSETS / "01a0d27e-806d-7840-9018-9a389ae6733b.jpg",
    ASSETS / "01a0d281-37df-7874-a0f8-445326aa781f.jpg",
]


def _augment_real(img: Image.Image, rng: np.random.Generator, size: int = 128) -> Image.Image:
    g = img.convert("L")
    w, h = g.size
    ink = np.array(g)
    side = int(min(w, h) * float(rng.uniform(0.3, 0.85)))
    side = max(80, min(side, min(w, h)))
    best = (0.0, 0, 0)
    for _ in range(16):
        x0 = int(rng.integers(0, max(1, w - side)))
        y0 = int(rng.integers(0, max(1, h - side)))
        patch = ink[y0 : y0 + side, x0 : x0 + side]
        dens = float(((patch < 200) & (patch > 5)).mean())
        if dens > best[0]:
            best = (dens, x0, y0)
    _, x0, y0 = best
    crop = g.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.LANCZOS)
    if rng.random() < 0.5:
        crop = crop.rotate(float(rng.uniform(-30, 30)), fillcolor=255, expand=False)
    if rng.random() < 0.35:
        crop = ImageOps.mirror(crop)
    if rng.random() < 0.35:
        crop = ImageOps.flip(crop)
    if rng.random() < 0.5:
        crop = ImageEnhance.Contrast(crop).enhance(float(rng.uniform(0.85, 1.25)))
    if rng.random() < 0.4:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.7))))
    if rng.random() < 0.5:
        crop = add_interference(crop, rng=rng, noise=0.015, lines=True, blur=False, crosshair_chance=0.4)
    return crop


def _scene_mask(canvas: int, rng: np.random.Generator) -> Image.Image:
    mask = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(mask)
    m = int(canvas * 0.08)
    t = int(canvas * float(rng.uniform(0.18, 0.4)))
    kind = int(rng.integers(0, 5))
    if kind == 0:  # filled irregular polygon
        cx, cy = canvas / 2, canvas / 2
        n = int(rng.integers(5, 9))
        pts = []
        for i in range(n):
            a = i * 2 * np.pi / n + float(rng.uniform(-0.2, 0.2))
            r = canvas * float(rng.uniform(0.28, 0.45))
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        d.polygon(pts, fill=255)
    elif kind == 1:  # L
        d.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        d.rectangle([canvas - m - t, m, canvas - m, canvas - m], fill=255)
    elif kind == 2:  # U
        d.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        d.rectangle([m, m, m + t, canvas - m], fill=255)
        d.rectangle([canvas - m - t, m, canvas - m, canvas - m], fill=255)
    elif kind == 3:  # thick strip
        y0 = int(rng.integers(m, canvas - m - t))
        d.rectangle([m, y0, canvas - m, y0 + t], fill=255)
    else:
        d.rectangle([m, m, canvas - m, canvas - m], fill=255)
    return mask


def _paste_in_scene(hatch: Image.Image, rng: np.random.Generator, size: int = 128) -> Image.Image:
    canvas = 280
    sheet = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(sheet)
    mask = _scene_mask(canvas, rng)
    tiled = Image.new("L", (canvas, canvas), 255)
    hw, hh = hatch.size
    for yy in range(0, canvas, max(1, hh)):
        for xx in range(0, canvas, max(1, hw)):
            tiled.paste(hatch, (xx, yy))
    sheet.paste(tiled, mask=mask)
    # Interference
    if rng.random() < 0.65:
        for _ in range(int(rng.integers(1, 3))):
            y = int(rng.integers(0, canvas))
            for x in range(0, canvas, 14):
                draw.line([(x, y), (x + 7, y)], fill=40, width=1)
    if rng.random() < 0.5:
        sheet = add_cad_crosshair(sheet, rng)
    if rng.random() < 0.4:
        r = int(rng.integers(10, 30))
        cx = int(rng.integers(r + 5, canvas - r - 5))
        cy = int(rng.integers(r + 5, canvas - r - 5))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=2)
        draw.line([(cx - r - 8, cy), (cx + r + 8, cy)], fill=0, width=1)
        draw.line([(cx, cy - r - 8), (cx, cy + r + 8)], fill=0, width=1)
    if rng.random() < 0.35:
        # dimension-like number ticks
        y = int(rng.integers(10, 40))
        draw.line([(40, y), (canvas - 40, y)], fill=0, width=1)
    return sheet.resize((size, size), Image.Resampling.LANCZOS)


def append_gravel_samples(out_dir: Path, size: int = 128, seed: int = 77) -> None:
    rng = np.random.default_rng(seed)
    pattern = parse_pat_file(ROOT / "acad_4270.pat")["GRAVEL"]
    dest = out_dir / "train" / "GRAVEL"
    vdest = out_dir / "val" / "GRAVEL"
    dest.mkdir(parents=True, exist_ok=True)
    vdest.mkdir(parents=True, exist_ok=True)

    # A) Improved PAT-scale gravel
    for i in range(120):
        scale = float(suggest_scale(pattern, size=160, target_px=float(rng.uniform(8, 20))))
        hatch = render_pattern(
            pattern,
            size=160,
            scale=scale,
            rotation=float(rng.uniform(0, 360)),
            fg=int(rng.choice([0, 0, 15])),
            supersample=2,
        )
        scene = _paste_in_scene(hatch, rng, size=size)
        if rng.random() < 0.5:
            scene = add_interference(scene, rng=rng, noise=0.02, lines=False, blur=True, crosshair_chance=0.2)
        scene.save(dest / f"pat_gravel_{i:03d}.png")

    # B) Procedural pebble gravel (matches CAD screenshot look)
    for i in range(160):
        peb = render_gravel_pebbles(
            size=160,
            rng=rng,
            density=float(rng.uniform(0.7, 1.6)),
            fg=int(rng.choice([0, 0, 20])),
        )
        if rng.random() < 0.4:
            peb = peb.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.6))))
        scene = _paste_in_scene(peb, rng, size=size)
        if rng.random() < 0.45:
            scene = add_interference(scene, rng=rng, noise=0.015, lines=True, blur=False, crosshair_chance=0.35)
        scene.save(dest / f"peb_gravel_{i:03d}.png")

    # C) Real user gravel screenshots
    n_real = 0
    for path in REAL_GRAVEL:
        if not path.exists():
            continue
        real = Image.open(path)
        for i in range(50):
            _augment_real(real, rng, size=size).save(dest / f"real_{path.stem}_{i:03d}.png")
        for i in range(8):
            _augment_real(real, rng, size=size).save(vdest / f"real_{path.stem}_val_{i:03d}.png")
        n_real += 1
    print(f"GRAVEL samples ready; real screenshots used: {n_real}")


def finetune(epochs: int = 10, lr: float = 1.2e-4) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "dataset"
    model_dir = ROOT / "models"
    append_gravel_samples(data_dir, size=128)

    train_loader, val_loader, test_loader, classes = build_loaders(data_dir, batch_size=64, size=128)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model = HatchCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Mild class weighting toward GRAVEL if present
    counts = []
    for c in classes:
        n = len(list((data_dir / "train" / c).glob("*")))
        counts.append(max(n, 1))
    inv = [1.0 / n for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    # Boost GRAVEL a bit more
    if "GRAVEL" in classes:
        w[classes.index("GRAVEL")] *= 1.35
    criterion = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = float(ckpt.get("val_acc", 0))

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        pbar = tqdm(train_loader, desc=f"gravel-ft {epoch}/{epochs}")
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
        print(f"gravel-ft {epoch}: train={correct/total:.3f} val={val_acc:.3f}")
        if val_acc >= best - 0.02:
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
    metrics["test_acc_gravel_ft"] = test_acc
    metrics["best_val_acc"] = best
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print({"test_acc": test_acc, "best_val_acc": best})
    return {"test_acc": test_acc, "best_val_acc": best}


if __name__ == "__main__":
    # Quick visual check
    rng = np.random.default_rng(0)
    render_gravel_pebbles(256, rng, 1.1).save(ROOT / "samples/preview/GRAVEL_pebbles.png")
    finetune(epochs=10)
