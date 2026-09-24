"""
Train expandable weak-periodic specialist.

Classes (temporary subset + reject):
  - AR-CONC, GRAVEL  — weak-periodic targets
  - OTHER            — strong-periodic / non-target (ANSI/LINE/NET/STEEL/BRICK/…)
                       so the CNN itself refuses ANSI31 instead of forced GRAVEL

Synthetic only — no real CAD screenshots.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pat_parser import parse_pat_file
from renderer import (
    add_cad_crosshair,
    add_interference,
    render_ar_conc_aggregate,
    render_gravel_cobbles,
    render_gravel_pebbles,
    render_pattern,
    suggest_scale,
)
from texture_features import extract_hatch_roi
from train import HatchCNN

ROOT = Path(__file__).resolve().parent
PAT = ROOT / "acad_4270.pat"
SPEC_DATA = ROOT / "data" / "aggregate_specialist"
MODEL_OUT = ROOT / "models" / "aggregate_specialist.pt"
META_OUT = ROOT / "models" / "aggregate_specialist_meta.json"
# Expandable; OTHER is the reject / non-weak-periodic bucket for now.
CLASSES = ["AR-CONC", "GRAVEL", "OTHER"]
OTHER_PATTERNS = ["ANSI31", "ANSI32", "LINE", "NET", "STEEL", "BRICK"]  # no SOLID (ambiguous)
SIZE = 128


def _scene_mask(canvas: int, rng: np.random.Generator) -> Image.Image:
    mask = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(mask)
    m = int(canvas * 0.06)
    kind = int(rng.integers(0, 6))
    if kind == 0:
        d.rectangle([m, m, canvas - m, canvas - m], fill=255)
    elif kind == 1:
        t = int(canvas * float(rng.uniform(0.22, 0.4)))
        d.rectangle([m, canvas - m - t, canvas - m, canvas - m], fill=255)
        d.rectangle([canvas - m - t, m, canvas - m, canvas - m], fill=255)
    elif kind == 2:
        d.rounded_rectangle([m, m + 30, canvas - m, canvas - m - 30], radius=40, fill=255)
    elif kind == 3:
        top = canvas * float(rng.uniform(0.2, 0.35))
        d.polygon(
            [(m + top, m), (canvas - m - top, m), (canvas - m, canvas - m), (m, canvas - m)],
            fill=255,
        )
    else:
        cx, cy = canvas / 2, canvas / 2
        n = int(rng.integers(4, 8))
        pts = []
        for i in range(n):
            a = i * 2 * np.pi / n + float(rng.uniform(-0.2, 0.2))
            r = canvas * float(rng.uniform(0.28, 0.46))
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        d.polygon(pts, fill=255)
    return mask


def _apply_scene(hatch: Image.Image, rng: np.random.Generator, size: int = SIZE) -> Image.Image:
    canvas = 280
    sheet = Image.new("L", (canvas, canvas), 255)
    mask = _scene_mask(canvas, rng)
    tiled = Image.new("L", (canvas, canvas), 255)
    hw, hh = hatch.size
    for yy in range(0, canvas, max(1, hh)):
        for xx in range(0, canvas, max(1, hw)):
            tiled.paste(hatch, (xx, yy))
    sheet.paste(tiled, mask=mask)
    draw = ImageDraw.Draw(sheet)
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 3))):
            y = int(rng.integers(8, 40))
            x0 = int(rng.integers(20, 60))
            draw.line([(x0, y), (x0 + int(rng.integers(40, 140)), y)], fill=40, width=1)
    if rng.random() < 0.45:
        for _ in range(int(rng.integers(1, 3))):
            r = int(rng.integers(8, 18))
            cx = int(rng.integers(r + 25, canvas - r - 25))
            cy = int(rng.integers(r + 25, canvas - r - 25))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=2)
            draw.line([(cx - r - 6, cy), (cx + r + 6, cy)], fill=0, width=1)
            draw.line([(cx, cy - r - 6), (cx, cy + r + 6)], fill=0, width=1)
    elif rng.random() < 0.3:
        sheet = add_cad_crosshair(sheet, rng)
    return sheet.resize((size, size), Image.Resampling.LANCZOS)


def _aug(img: Image.Image, rng: np.random.Generator, heavy: bool = True) -> Image.Image:
    if rng.random() < 0.5:
        img = img.rotate(float(rng.uniform(-25, 25)), fillcolor=255)
    if rng.random() < 0.35:
        img = ImageOps.mirror(img)
    if rng.random() < 0.35:
        img = ImageOps.flip(img)
    if rng.random() < 0.45:
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.85, 1.35)))
    if heavy and rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.55))))
    if heavy and rng.random() < 0.45:
        img = add_interference(img, rng=rng, noise=0.012, lines=True, blur=False, crosshair_chance=0.35)
    elif (not heavy) and rng.random() < 0.2:
        # light noise only — keep OTHER looking like clean periodic lines
        img = add_interference(img, rng=rng, noise=0.008, lines=False, blur=False, crosshair_chance=0.15)
    # JPEG / screen capture artifacts (real CAD screenshots)
    if heavy and rng.random() < 0.55:
        import io

        buf = io.BytesIO()
        q = int(rng.integers(55, 92))
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("L")
    elif (not heavy) and rng.random() < 0.25:
        import io

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=int(rng.integers(70, 95)))
        buf.seek(0)
        img = Image.open(buf).convert("L")
    return img


def _finalize(img: Image.Image) -> Image.Image:
    box = extract_hatch_roi(np.asarray(img.convert("L")))
    return img.crop(box).resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def _zoomed_out_scene(hatch: Image.Image, rng: np.random.Generator, size: int = SIZE) -> Image.Image:
    """Place a small hatch island in a large white sheet — mimics sparse real screenshots."""
    canvas = int(rng.integers(320, 480))
    sheet = Image.new("L", (canvas, canvas), 255)
    # Shrink hatch tile so after ROI+resize the fill looks sparse
    tile = int(rng.integers(90, 160))
    hsmall = hatch.resize((tile, tile), Image.Resampling.LANCZOS)
    mask = _scene_mask(canvas, rng)
    tiled = Image.new("L", (canvas, canvas), 255)
    for yy in range(0, canvas, max(1, tile)):
        for xx in range(0, canvas, max(1, tile)):
            tiled.paste(hsmall, (xx, yy))
    sheet.paste(tiled, mask=mask)
    return sheet.resize((size, size), Image.Resampling.LANCZOS)


def build_synthetic(n_target: int = 900, n_other: int = 480, seed: int = 20260926) -> None:
    """
    More AR-CONC/GRAVEL with scene interference + sparse zoomed-out views;
    cleaner OTHER so reject class does not swallow messy real aggregate.
    """
    if SPEC_DATA.exists():
        shutil.rmtree(SPEC_DATA)
    rng = np.random.default_rng(seed)
    patterns = parse_pat_file(PAT)

    splits = (
        ("train", n_target, n_other),
        ("val", max(100, n_target // 7), max(70, n_other // 7)),
    )
    for split, n_t, n_o in splits:
        for cls in CLASSES:
            (SPEC_DATA / split / cls).mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n_t), desc=f"synth {split} targets"):
            # GRAVEL — include hatched-interior pebbles (common in real CAD)
            style_roll = float(rng.random())
            dens_g = float(rng.uniform(0.7, 1.55))
            if style_roll < 0.4:
                h = render_gravel_cobbles(168, rng, dens_g)
            elif style_roll < 0.75:
                h = render_gravel_pebbles(168, rng, dens_g, style="round")
            else:
                h = render_gravel_pebbles(168, rng, dens_g, style="mixed")
            roll = float(rng.random())
            if roll < 0.55:
                gimg = _apply_scene(h, rng)
            elif roll < 0.8:
                gimg = _zoomed_out_scene(h, rng)
            else:
                gimg = h.resize((SIZE, SIZE))
            _finalize(_aug(gimg, rng, heavy=True)).save(
                SPEC_DATA / split / "GRAVEL" / f"g_{i:04d}.png"
            )

            # AR-CONC — include sparse/zoomed-out stipple (real dens can be ~0.02)
            dens_a = float(rng.choice([rng.uniform(0.3, 0.7), rng.uniform(0.75, 1.5)]))
            h = render_ar_conc_aggregate(168, rng, dens_a)
            roll = float(rng.random())
            if roll < 0.5:
                aimg = _apply_scene(h, rng)
            elif roll < 0.82:
                aimg = _zoomed_out_scene(h, rng)
            else:
                aimg = h.resize((SIZE, SIZE))
            _finalize(_aug(aimg, rng, heavy=True)).save(
                SPEC_DATA / split / "AR-CONC" / f"a_{i:04d}.png"
            )

        for i in tqdm(range(n_o), desc=f"synth {split} other"):
            name = OTHER_PATTERNS[int(rng.integers(0, len(OTHER_PATTERNS)))]
            pat = patterns[name]
            rot = float(rng.uniform(0, 360))
            ss = 2 if rng.random() < 0.5 else 1
            # Prefer clean full-frame periodic (teaches "lines ≠ aggregate")
            if rng.random() < 0.75:
                oimg = render_pattern(
                    pat,
                    size=SIZE,
                    scale=suggest_scale(pat, SIZE) * float(rng.uniform(0.8, 1.3)),
                    rotation=rot,
                    bg=255,
                    fg=0,
                    stroke=1,
                    shape="rect",
                    supersample=ss,
                )
                oimg = _finalize(_aug(oimg, rng, heavy=False))
            else:
                big = render_pattern(
                    pat,
                    size=168,
                    scale=suggest_scale(pat, 168) * float(rng.uniform(0.8, 1.25)),
                    rotation=rot,
                    bg=255,
                    fg=0,
                    stroke=1,
                    shape="rect",
                    supersample=1,
                )
                oimg = _finalize(_aug(_apply_scene(big, rng), rng, heavy=False))
            oimg.save(SPEC_DATA / split / "OTHER" / f"o_{name}_{i:04d}.png")


def train(epochs: int = 24, batch_size: int = 64, lr: float = 8e-4) -> dict:
    device = torch.device("cpu")
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.88, 1.12)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_ds = datasets.ImageFolder(SPEC_DATA / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(SPEC_DATA / "val", transform=eval_tf)
    assert train_ds.classes == CLASSES, train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    counts = [len(list((SPEC_DATA / "train" / c).glob("*"))) for c in CLASSES]
    inv = [1.0 / max(n, 1) for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    # Prefer not to dump uncertain aggregate into OTHER
    w[CLASSES.index("AR-CONC")] *= 1.5
    w[CLASSES.index("GRAVEL")] *= 1.35
    w[CLASSES.index("OTHER")] *= 0.7

    model = HatchCNN(num_classes=len(CLASSES)).to(device)
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.015)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        cor = tot = 0
        for x, y in tqdm(train_loader, desc=f"agg{epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            probs = torch.softmax(logits, dim=1)
            correct_p = probs.gather(1, y.view(-1, 1)).squeeze(1)
            # Stronger confidence shove on target classes only
            is_target = (y == CLASSES.index("AR-CONC")) | (y == CLASSES.index("GRAVEL"))
            if is_target.any():
                conf_loss = (0.92 - correct_p[is_target]).clamp(min=0).mean() * 0.7
            else:
                conf_loss = torch.tensor(0.0, device=device)
            (loss + conf_loss).backward()
            opt.step()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
        sched.step()

        model.eval()
        vcor = vtot = 0
        confs = []
        per = {c: {"ok": 0, "n": 0, "hi": 0} for c in CLASSES}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                p = torch.softmax(logits, dim=1)
                pred = logits.argmax(1)
                vcor += (pred == y).sum().item()
                vtot += y.size(0)
                for i in range(y.size(0)):
                    cname = CLASSES[int(y[i])]
                    per[cname]["n"] += 1
                    conf = float(p[i].max())
                    if int(pred[i]) == int(y[i]):
                        per[cname]["ok"] += 1
                        confs.append(conf)
                        if conf >= 0.8:
                            per[cname]["hi"] += 1
        va = vcor / max(vtot, 1)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        hi = float(np.mean([c >= 0.8 for c in confs])) if confs else 0.0
        # Prefer models that are both accurate and high-confidence on targets
        hi_ar = per["AR-CONC"]["hi"] / max(per["AR-CONC"]["n"], 1)
        hi_gr = per["GRAVEL"]["hi"] / max(per["GRAVEL"]["n"], 1)
        score = 0.35 * va + 0.35 * hi + 0.15 * hi_ar + 0.15 * hi_gr
        history.append(
            {
                "epoch": epoch,
                "train_acc": cor / tot,
                "val_acc": va,
                "val_mean_conf_correct": mean_conf,
                "val_frac_conf_ge_0.8": hi,
                "score": score,
                "per_class": {
                    c: {
                        "acc": per[c]["ok"] / max(per[c]["n"], 1),
                        "hi80_correct": per[c]["hi"] / max(per[c]["n"], 1),
                    }
                    for c in CLASSES
                },
            }
        )
        print(
            f"epoch {epoch}: train={cor/tot:.3f} val={va:.3f} conf_ok={mean_conf:.3f} hi80={hi:.3f} score={score:.3f} "
            + " ".join(f"{c}={per[c]['ok']/max(per[c]['n'],1):.2f}" for c in CLASSES)
        )
        if score >= best:
            best = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": SIZE,
                    "val_acc": va,
                    "score": score,
                    "temperature": 0.78,
                },
                MODEL_OUT,
            )
            print("  saved")

    meta = {"best_score": best, "history": history, "classes": CLASSES, "synthetic_only": True}
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    build_synthetic(n_target=900, n_other=480)
    train(epochs=24)


if __name__ == "__main__":
    main()