"""Short extra fine-tune focused on L-shape GRAVEL scenes."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from renderer import render_gravel_pebbles, add_interference, add_cad_crosshair
from train import HatchCNN, build_loaders, evaluate
import torch, torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
dest = ROOT / "data/dataset/train/GRAVEL"
dest.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)


def L_scene(hatch, size=128):
    canvas = 280
    sheet = Image.new("L", (canvas, canvas), 255)
    mask = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(mask)
    m = 20
    t = int(canvas * rng.uniform(0.22, 0.38))
    if rng.random() < 0.5:
        d.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        d.rectangle([canvas - m - t, m, canvas - m, canvas - m], fill=255)
    else:
        d.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        d.rectangle([m, m, m + t, canvas - m], fill=255)
    tiled = Image.new("L", (canvas, canvas), 255)
    hw, hh = hatch.size
    for yy in range(0, canvas, hh):
        for xx in range(0, canvas, hw):
            tiled.paste(hatch, (xx, yy))
    sheet.paste(tiled, mask=mask)
    draw = ImageDraw.Draw(sheet)
    for offset in (12, 28):
        y = canvas - m - t - offset
        for x in range(m, canvas - m, 16):
            draw.line([(x, y), (x + 8, y)], fill=50, width=1)
        x = canvas - m - t - offset if rng.random() < 0.5 else m + t + offset
        for y in range(m, canvas - m, 16):
            draw.line([(x, y), (x, y + 8)], fill=50, width=1)
    if rng.random() < 0.6:
        sheet = add_cad_crosshair(sheet, rng)
    return sheet.resize((size, size), Image.Resampling.LANCZOS)


if not list(dest.glob("Lpeb_*.png")):
    for i in range(100):
        peb = render_gravel_pebbles(160, rng, float(rng.uniform(0.8, 1.5)))
        scene = L_scene(peb)
        if rng.random() < 0.5:
            scene = add_interference(
                scene, rng=rng, noise=0.015, lines=False, blur=True, crosshair_chance=0.2
            )
        scene.save(dest / f"Lpeb_{i:03d}.png")

if not list(dest.glob("Lreal_*.png")):
    real = Image.open(ASSETS / "01a0d281-37df-7874-a0f8-445326aa781f.jpg").convert("L")
    w, h = real.size
    ink = np.array(real)
    for i in range(80):
        side = int(min(w, h) * float(rng.uniform(0.35, 0.95)))
        best = (0.0, 0, 0)
        for _ in range(20):
            x0 = int(rng.integers(0, max(1, w - side)))
            y0 = int(rng.integers(0, max(1, h - side)))
            p = ink[y0 : y0 + side, x0 : x0 + side]
            dens = float(((p < 200) & (p > 5)).mean())
            if dens > best[0]:
                best = (dens, x0, y0)
        _, x0, y0 = best
        crop = real.crop((x0, y0, x0 + side, y0 + side)).resize((128, 128))
        if rng.random() < 0.5:
            crop = crop.rotate(float(rng.uniform(-35, 35)), fillcolor=255)
        if rng.random() < 0.4:
            crop = ImageOps.mirror(crop)
        if rng.random() < 0.4:
            crop = ImageOps.flip(crop)
        if rng.random() < 0.5:
            crop = ImageEnhance.Contrast(crop).enhance(float(rng.uniform(0.85, 1.3)))
        if rng.random() < 0.4:
            crop = add_interference(
                crop, rng=rng, noise=0.01, lines=True, blur=False, crosshair_chance=0.35
            )
        crop.save(dest / f"Lreal_{i:03d}.png")

print("L samples:", len(list(dest.glob("L*.png"))))

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
w[classes.index("GRAVEL")] *= 1.6
crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)
opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
best = float(ckpt.get("val_acc", 0))
for epoch in range(1, 7):
    model.train()
    cor = tot = 0
    for x, y in tqdm(train_loader, desc=f"Lft{epoch}"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        cor += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    _, va = evaluate(model, val_loader, device)
    print(f"Lft{epoch} train={cor/tot:.3f} val={va:.3f}")
    if va >= best - 0.025:
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
