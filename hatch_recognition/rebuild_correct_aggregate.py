"""
Rebuild GRAVEL / AR-CONC training data with correct closed-aggregate look.

Problem: acad_4270.pat stroke rendering produces short dashes for GRAVEL/AR-CONC,
but real AutoCAD screenshots show packed closed pebble outlines. All prior
PAT-based samples for these two classes are wrong and are deleted here.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

from renderer import (
    add_cad_crosshair,
    add_interference,
    render_ar_conc_aggregate,
    render_gravel_cobbles,
    render_gravel_pebbles,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset"
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
SIZE = 128

REAL_GRAVEL = [
    ASSETS / "01a0d27e-806d-7840-9018-9a389ae6733b.jpg",
    ASSETS / "01a0d281-37df-7874-a0f8-445326aa781f.jpg",
    ASSETS / "01a0d2af-a2a6-7beb-9388-c0936f5ad85a.jpg",
    ASSETS / "01a0d2b4-45f1-7a20-a74b-4ea28e36ccc8.jpg",
]
REAL_ARCONC = [
    ASSETS / "01a0d261-8064-7965-bbcb-9be9d02e1e5c.jpg",
    ASSETS / "01a0d262-d1c9-742e-9e6e-26968ab74972.jpg",
    ASSETS / "01a0d263-d93e-7705-873f-c3548132d977.jpg",
    ASSETS / "01a0d264-a8fa-7635-a753-8ac1bd36e145.jpg",
    ASSETS / "01a0d265-ce06-7d31-a510-70df7088e275.jpg",
]


def _wipe(cls: str) -> None:
    for split in ("train", "val", "test"):
        d = DATA / split / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _mask_poly(canvas: int, rng: np.random.Generator) -> Image.Image:
    mask = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(mask)
    m = int(canvas * 0.06)
    kind = int(rng.integers(0, 5))
    if kind == 0:
        d.rectangle([m, m, canvas - m, canvas - m], fill=255)
    elif kind == 1:
        t = int(canvas * float(rng.uniform(0.22, 0.4)))
        d.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        d.rectangle([canvas - m - t, m, canvas - m, canvas - m], fill=255)
    elif kind == 2:
        # rounded rect approx
        d.rounded_rectangle([m, m + 20, canvas - m, canvas - m - 10], radius=30, fill=255)
    else:
        cx, cy = canvas / 2, canvas / 2
        n = int(rng.integers(4, 7))
        pts = []
        for i in range(n):
            a = i * 2 * np.pi / n + float(rng.uniform(-0.2, 0.2))
            r = canvas * float(rng.uniform(0.3, 0.46))
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        d.polygon(pts, fill=255)
    return mask


def _scene(hatch: Image.Image, rng: np.random.Generator, size: int = SIZE) -> Image.Image:
    canvas = 280
    sheet = Image.new("L", (canvas, canvas), 255)
    mask = _mask_poly(canvas, rng)
    tiled = Image.new("L", (canvas, canvas), 255)
    hw, hh = hatch.size
    for yy in range(0, canvas, max(1, hh)):
        for xx in range(0, canvas, max(1, hw)):
            tiled.paste(hatch, (xx, yy))
    sheet.paste(tiled, mask=mask)
    draw = ImageDraw.Draw(sheet)
    if rng.random() < 0.55:
        for _ in range(int(rng.integers(1, 3))):
            y = int(rng.integers(8, 40))
            x0 = int(rng.integers(20, 60))
            draw.line([(x0, y), (x0 + int(rng.integers(40, 140)), y)], fill=40, width=1)
            y = canvas - int(rng.integers(8, 28))
            draw.line([(x0, y), (x0 + int(rng.integers(40, 140)), y)], fill=40, width=1)
    if rng.random() < 0.55:
        for _ in range(int(rng.integers(1, 3))):
            r = int(rng.integers(8, 18))
            cx = int(rng.integers(r + 25, canvas - r - 25))
            cy = int(rng.integers(r + 25, canvas - r - 25))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=2)
            draw.line([(cx - r - 6, cy), (cx + r + 6, cy)], fill=0, width=1)
            draw.line([(cx, cy - r - 6), (cx, cy + r + 6)], fill=0, width=1)
    elif rng.random() < 0.35:
        sheet = add_cad_crosshair(sheet, rng)
    return sheet.resize((size, size), Image.Resampling.LANCZOS)


def _augment_real(img: Image.Image, rng: np.random.Generator, size: int = SIZE) -> Image.Image:
    g = img.convert("L")
    w, h = g.size
    ink = np.array(g)
    side = int(min(w, h) * float(rng.uniform(0.3, 0.9)))
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
    if rng.random() < 0.5:
        crop = crop.rotate(float(rng.uniform(-35, 35)), fillcolor=255)
    if rng.random() < 0.35:
        crop = ImageOps.mirror(crop)
    if rng.random() < 0.35:
        crop = ImageOps.flip(crop)
    if rng.random() < 0.45:
        crop = ImageEnhance.Contrast(crop).enhance(float(rng.uniform(0.85, 1.3)))
    if rng.random() < 0.3:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.55))))
    if rng.random() < 0.35:
        crop = add_interference(crop, rng=rng, noise=0.01, lines=True, blur=False, crosshair_chance=0.35)
    return crop


def _save_splits(paths: list[Image.Image], cls: str, prefix: str, rng: np.random.Generator) -> None:
    n = len(paths)
    idx = rng.permutation(n)
    n_test = max(1, int(n * 0.1))
    n_val = max(1, int(n * 0.15))
    for i, j in enumerate(idx):
        if i < n_test:
            split = "test"
        elif i < n_test + n_val:
            split = "val"
        else:
            split = "train"
        paths[j].save(DATA / split / cls / f"{prefix}_{i:04d}.png")


def rebuild_gravel(rng: np.random.Generator) -> None:
    _wipe("GRAVEL")
    imgs: list[Image.Image] = []
    # A) clean closed pebbles (majority)
    for i in tqdm(range(220), desc="GRAVEL clean round"):
        h = render_gravel_pebbles(SIZE, rng, float(rng.uniform(0.85, 1.35)), style="round")
        imgs.append(h)
    for i in tqdm(range(120), desc="GRAVEL clean cobble"):
        h = render_gravel_cobbles(SIZE, rng, float(rng.uniform(0.85, 1.4)))
        imgs.append(h)
    # B) scenes with interference
    for i in tqdm(range(160), desc="GRAVEL scenes"):
        style = "cobble" if rng.random() < 0.4 else "round"
        h = render_gravel_pebbles(168, rng, float(rng.uniform(0.9, 1.4)), style=style)
        scene = _scene(h, rng)
        if rng.random() < 0.4:
            scene = add_interference(scene, rng=rng, noise=0.012, lines=False, blur=True, crosshair_chance=0.2)
        imgs.append(scene)
    # C) real screenshot crops
    for path in REAL_GRAVEL:
        if not path.exists():
            continue
        real = Image.open(path)
        for _ in range(40):
            imgs.append(_augment_real(real, rng))
    _save_splits(imgs, "GRAVEL", "ok", rng)
    print("GRAVEL total", len(imgs))


def rebuild_arconc(rng: np.random.Generator) -> None:
    _wipe("AR-CONC")
    imgs: list[Image.Image] = []
    for _ in tqdm(range(220), desc="AR-CONC clean"):
        imgs.append(render_ar_conc_aggregate(SIZE, rng, float(rng.uniform(0.8, 1.4))))
    for _ in tqdm(range(140), desc="AR-CONC scenes"):
        h = render_ar_conc_aggregate(168, rng, float(rng.uniform(0.85, 1.35)))
        scene = _scene(h, rng)
        if rng.random() < 0.4:
            scene = add_interference(scene, rng=rng, noise=0.012, lines=True, blur=False, crosshair_chance=0.25)
        imgs.append(scene)
    for path in REAL_ARCONC:
        if not path.exists():
            continue
        real = Image.open(path)
        for _ in range(35):
            imgs.append(_augment_real(real, rng))
    _save_splits(imgs, "AR-CONC", "ok", rng)
    print("AR-CONC total", len(imgs))


def write_preview() -> None:
    preview = ROOT / "samples" / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    peb = render_gravel_pebbles(256, rng, 1.05, style="round")
    cob = render_gravel_cobbles(256, rng, 1.1)
    arc = render_ar_conc_aggregate(256, rng, 1.0)
    peb.save(preview / "GRAVEL_fixed_round.png")
    cob.save(preview / "GRAVEL_fixed_cobble.png")
    arc.save(preview / "ARCONC_fixed.png")
    user = Image.open(preview / "USER_GRAVEL_tile160.png").convert("L").resize((256, 256))
    cmp = Image.new("L", (256 * 4, 256), 255)
    for i, im in enumerate([user, peb, cob, arc]):
        cmp.paste(im, (i * 256, 0))
    cmp.save(preview / "COMPARE_user_peb_cob_arconc.png")


if __name__ == "__main__":
    rng = np.random.default_rng(20260924)
    write_preview()
    rebuild_gravel(rng)
    rebuild_arconc(rng)
    # counts
    for c in ("GRAVEL", "AR-CONC"):
        for s in ("train", "val", "test"):
            n = len(list((DATA / s / c).glob("*.png")))
            print(f"{c}/{s}: {n}")
