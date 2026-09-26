"""Fine-tune for angular cobble-style GRAVEL (often confused with NET/BRICK)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

from renderer import add_cad_crosshair, add_interference, render_gravel_cobbles, render_gravel_pebbles
from train import HatchCNN, build_loaders, evaluate

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
REAL = [
    ASSETS / "01a0d2b4-45f1-7a20-a74b-4ea28e36ccc8.jpg",  # angular cobble polygon
    ASSETS / "01a0d2af-a2a6-7beb-9388-c0936f5ad85a.jpg",  # bathroom pebble pad
    ASSETS / "01a0d27e-806d-7840-9018-9a389ae6733b.jpg",
    ASSETS / "01a0d281-37df-7874-a0f8-445326aa781f.jpg",
]


def _augment_real(img: Image.Image, rng: np.random.Generator, size: int = 128) -> Image.Image:
    g = img.convert("L")
    w, h = g.size
    ink = np.array(g)
    side = int(min(w, h) * float(rng.uniform(0.28, 0.9)))
    side = max(64, min(side, min(w, h)))
    best = (0.0, 0, 0)
    for _ in range(24):
        x0 = int(rng.integers(0, max(1, w - side)))
        y0 = int(rng.integers(0, max(1, h - side)))
        patch = ink[y0 : y0 + side, x0 : x0 + side]
        dens = float(((patch < 200) & (patch > 5)).mean())
        if dens > best[0]:
            best = (dens, x0, y0)
    _, x0, y0 = best
    crop = g.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.LANCZOS)
    if rng.random() < 0.55:
        crop = crop.rotate(float(rng.uniform(-40, 40)), fillcolor=255)
    if rng.random() < 0.4:
        crop = ImageOps.mirror(crop)
    if rng.random() < 0.4:
        crop = ImageOps.flip(crop)
    if rng.random() < 0.5:
        crop = ImageEnhance.Contrast(crop).enhance(float(rng.uniform(0.8, 1.35)))
    if rng.random() < 0.35:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.6))))
    if rng.random() < 0.45:
        crop = add_interference(crop, rng=rng, noise=0.012, lines=True, blur=False, crosshair_chance=0.45)
    return crop


def _poly_scene(hatch: Image.Image, rng: np.random.Generator, size: int = 128) -> Image.Image:
    canvas = 300
    sheet = Image.new("L", (canvas, canvas), 255)
    mask = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = canvas / 2, canvas / 2 + float(rng.uniform(-20, 20))
    n = int(rng.integers(4, 7))
    pts = []
    for i in range(n):
        a = i * 2 * np.pi / n + float(rng.uniform(-0.25, 0.25))
        r = canvas * float(rng.uniform(0.28, 0.46))
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    d.polygon(pts, fill=255)
    tiled = Image.new("L", (canvas, canvas), 255)
    hw, hh = hatch.size
    for yy in range(0, canvas, hh):
        for xx in range(0, canvas, hw):
            tiled.paste(hatch, (xx, yy))
    sheet.paste(tiled, mask=mask)
    draw = ImageDraw.Draw(sheet)
    # label underlines outside fill
    if rng.random() < 0.7:
        for y in (18, canvas - 22, canvas - 12):
            x0 = int(rng.integers(20, 80))
            draw.line([(x0, y), (x0 + int(rng.integers(40, 120)), y)], fill=40, width=1)
    # dual crosshair circles (common on this style of drawing)
    if rng.random() < 0.75:
        for _ in range(int(rng.integers(1, 3))):
            r = int(rng.integers(8, 18))
            cx = int(rng.integers(r + 30, canvas - r - 30))
            cy = int(rng.integers(r + 30, canvas - r - 30))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=2)
            draw.line([(cx - r - 6, cy), (cx + r + 6, cy)], fill=0, width=1)
            draw.line([(cx, cy - r - 6), (cx, cy + r + 6)], fill=0, width=1)
    elif rng.random() < 0.4:
        sheet = add_cad_crosshair(sheet, rng)
    return sheet.resize((size, size), Image.Resampling.LANCZOS)


def make_samples() -> int:
    dest = ROOT / "data/dataset/train/GRAVEL"
    vdest = ROOT / "data/dataset/val/GRAVEL"
    dest.mkdir(parents=True, exist_ok=True)
    vdest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4242)
    n = 0
    if not list(dest.glob("cob_*.png")):
        for i in range(140):
            hatch = render_gravel_cobbles(
                168, rng, density=float(rng.uniform(0.75, 1.5)), fg=int(rng.choice([0, 0, 18]))
            )
            if rng.random() < 0.25:
                hatch = hatch.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.5))))
            scene = _poly_scene(hatch, rng)
            if rng.random() < 0.4:
                scene = add_interference(scene, rng=rng, noise=0.01, lines=False, blur=False, crosshair_chance=0.2)
            scene.save(dest / f"cob_{i:03d}.png")
            n += 1
        for i in range(40):
            # mix pebbles into same polygon+crosshair scenes so both styles stay GRAVEL
            hatch = render_gravel_pebbles(168, rng, density=float(rng.uniform(0.8, 1.4)))
            _poly_scene(hatch, rng).save(dest / f"cobpeb_{i:03d}.png")
            n += 1
    for path in REAL:
        if not path.exists():
            continue
        stem = path.stem[:12]
        if list(dest.glob(f"cobreal_{stem}_*.png")):
            continue
        real = Image.open(path)
        for i in range(55):
            _augment_real(real, rng).save(dest / f"cobreal_{stem}_{i:03d}.png")
            n += 1
        for i in range(8):
            _augment_real(real, rng).save(vdest / f"cobreal_{stem}_val_{i:03d}.png")
    print("new/updated cobble samples:", n, "train GRAVEL total:", len(list(dest.glob("*"))))
    return n


def main() -> None:
    make_samples()
    preview = ROOT / "samples/preview"
    preview.mkdir(parents=True, exist_ok=True)
    render_gravel_cobbles(256, np.random.default_rng(1), 1.1).save(preview / "GRAVEL_cobbles.png")

    device = torch.device("cpu")
    data_dir = ROOT / "data/dataset"
    ckpt = torch.load(ROOT / "models/best_model.pt", map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = HatchCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    train_loader, val_loader, _, _ = build_loaders(data_dir, 64, 128)
    counts = [max(len(list((data_dir / "train" / c).glob("*"))), 1) for c in classes]
    inv = [1.0 / n for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    w[classes.index("GRAVEL")] *= 1.55
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=9e-5, weight_decay=1e-4)
    best = float(ckpt.get("val_acc", 0))
    for epoch in range(1, 8):
        model.train()
        cor = tot = 0
        for x, y in tqdm(train_loader, desc=f"cob{epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
        _, va = evaluate(model, val_loader, device)
        print(f"cob{epoch} train={cor/tot:.3f} val={va:.3f}")
        if va >= best - 0.03:
            if va >= best:
                best = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": 128,
                    "val_acc": va,
                },
                ROOT / "models/best_model.pt",
            )
            print(" saved")
    print("done best", best)


if __name__ == "__main__":
    main()
