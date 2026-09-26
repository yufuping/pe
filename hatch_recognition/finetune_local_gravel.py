"""
Balanced FT so the CNN itself handles:
  - local pebble crops → GRAVEL (user screenshot TTA views)
  - dense packed gravel → GRAVEL (not OTHER)
  - irregular/L-shaped fills with white margin → GRAVEL (not AR-CONC)

Starts from hatched-gravel checkpoint. Synthetic only — no real drawings.
No class-score fusion overrides.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pat_parser import parse_pat_file
from renderer import (
    render_ar_conc_aggregate,
    render_gravel_cobbles,
    render_gravel_pebbles,
    render_pattern,
    suggest_scale,
)
from train import HatchCNN
from train_aggregate_specialist import (
    CLASSES,
    META_OUT,
    MODEL_OUT,
    OTHER_PATTERNS,
    PAT,
    SIZE,
    _aug,
    _finalize,
)

FT_DATA = Path(__file__).resolve().parent / "data" / "aggregate_ft_local_gravel"
HATCHED = Path(__file__).resolve().parent / "data" / "aggregate_ft_hatched"


def _rand_crop(img: Image.Image, rng: np.random.Generator, frac: float) -> Image.Image:
    w, h = img.size
    side = max(56, int(min(w, h) * frac))
    if side >= min(w, h):
        return img
    x = int(rng.integers(0, w - side + 1))
    y = int(rng.integers(0, h - side + 1))
    return img.crop((x, y, x + side, y + side))


def _mask_l_shape(size: int, rng: np.random.Generator) -> Image.Image:
    """White-outside L (or thick polygon) mask — 255 = keep fill."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    t = float(rng.uniform(0.32, 0.48))
    # L bar thickness as fraction of size
    if rng.random() < 0.55:
        # classic L
        d.rectangle([0, int(size * (1 - t)), size, size], fill=255)
        d.rectangle([0, 0, int(size * t), size], fill=255)
    else:
        # irregular hex-ish blob
        n = int(rng.integers(5, 8))
        cx, cy = size * 0.5, size * 0.5
        r = size * float(rng.uniform(0.38, 0.48))
        pts = []
        for i in range(n):
            a = i * 2 * math.pi / n + float(rng.uniform(-0.15, 0.15))
            rr = r * float(rng.uniform(0.85, 1.12))
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=255)
    return m


def _gravel_in_shape(size: int, rng: np.random.Generator, dens: float) -> Image.Image:
    big = max(size, 220)
    if rng.random() < 0.45:
        fill = render_gravel_cobbles(big, rng, dens)
    else:
        style = "round" if rng.random() < 0.6 else "mixed"
        fill = render_gravel_pebbles(big, rng, dens, style=style)
    mask = _mask_l_shape(big, rng)
    from PIL import ImageFilter

    out = Image.new("L", (big, big), 255)
    out.paste(fill, (0, 0), mask=mask)
    if rng.random() < 0.7:
        ring = mask.filter(ImageFilter.FIND_EDGES)
        edge = ring.point(lambda p: 255 if p > 20 else 0)
        out = Image.composite(Image.new("L", (big, big), 0), out, edge)
    return out.resize((size, size), Image.Resampling.LANCZOS)


def _make_gravel(rng: np.random.Generator) -> Image.Image:
    mode = float(rng.random())
    dens = float(rng.uniform(0.85, 1.65))
    if mode < 0.28:
        # local crop of packed pebbles (TTA failure mode)
        big = render_gravel_pebbles(256, rng, dens, style="round" if rng.random() < 0.5 else "mixed")
        if rng.random() < 0.5:
            big = render_gravel_cobbles(256, rng, dens)
        return _rand_crop(big, rng, float(rng.uniform(0.28, 0.58))).resize(
            (SIZE, SIZE), Image.Resampling.LANCZOS
        )
    if mode < 0.50:
        # dense packed small pebbles (high ink)
        d2 = float(rng.uniform(1.25, 1.85))
        return render_gravel_pebbles(SIZE, rng, d2, style="round").resize(
            (SIZE, SIZE), Image.Resampling.LANCZOS
        )
    if mode < 0.72:
        # irregular / L-shaped fill with white margin
        return _gravel_in_shape(SIZE, rng, dens)
    # full tile, mixed styles
    if rng.random() < 0.5:
        return render_gravel_cobbles(SIZE, rng, dens)
    return render_gravel_pebbles(SIZE, rng, dens, style="mixed")


def build(
    n_gravel: int = 720,
    n_ar: int = 240,
    n_other: int = 280,
    seed: int = 20260928,
) -> None:
    if FT_DATA.exists():
        shutil.rmtree(FT_DATA)
    rng = np.random.default_rng(seed)
    patterns = parse_pat_file(PAT)

    for split, ng, na, no in (
        ("train", n_gravel, n_ar, n_other),
        ("val", max(80, n_gravel // 8), max(40, n_ar // 6), max(40, n_other // 6)),
    ):
        for c in CLASSES:
            (FT_DATA / split / c).mkdir(parents=True, exist_ok=True)

        # Replay hatched FT tiles when available (anti-forgetting)
        replay = 0
        if HATCHED.exists() and (HATCHED / split / "GRAVEL").exists():
            srcs = list((HATCHED / split / "GRAVEL").glob("*.png"))
            rng.shuffle(srcs)
            for j, src in enumerate(srcs[: min(len(srcs), ng // 4)]):
                Image.open(src).convert("L").resize((SIZE, SIZE)).save(
                    FT_DATA / split / "GRAVEL" / f"rep_{j:04d}.png"
                )
                replay += 1

        for i in tqdm(range(ng - replay), desc=f"{split} gravel"):
            g = _make_gravel(rng)
            _finalize(_aug(g, rng, heavy=True)).save(
                FT_DATA / split / "GRAVEL" / f"g_{i:04d}.png"
            )

        for i in tqdm(range(na), desc=f"{split} ar"):
            dens = float(rng.uniform(0.35, 1.45))
            h = render_ar_conc_aggregate(220, rng, dens)
            if rng.random() < 0.4:
                h = _rand_crop(h, rng, float(rng.uniform(0.35, 0.7)))
            # some AR-CONC also in L shapes so margin ≠ GRAVEL cue alone
            if rng.random() < 0.25:
                big = render_ar_conc_aggregate(240, rng, dens)
                mask = _mask_l_shape(240, rng)
                canvas = Image.new("L", (240, 240), 255)
                canvas.paste(big, (0, 0), mask=mask)
                h = canvas
            h = h.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            _finalize(_aug(h, rng, heavy=True)).save(
                FT_DATA / split / "AR-CONC" / f"a_{i:04d}.png"
            )

        for i in tqdm(range(no), desc=f"{split} other"):
            name = OTHER_PATTERNS[int(rng.integers(0, len(OTHER_PATTERNS)))]
            pat = patterns[name]
            scale = suggest_scale(pat, 200) * float(rng.uniform(0.75, 1.35))
            big = render_pattern(
                pat,
                size=200,
                scale=scale,
                rotation=float(rng.uniform(0, 360)),
                bg=255,
                fg=0,
                stroke=1,
                shape="rect",
                supersample=1,
            )
            if rng.random() < 0.55:
                oimg = _rand_crop(big, rng, float(rng.uniform(0.32, 0.7))).resize(
                    (SIZE, SIZE), Image.Resampling.LANCZOS
                )
            else:
                oimg = big.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            _finalize(_aug(oimg, rng, heavy=False)).save(
                FT_DATA / split / "OTHER" / f"o_{name}_{i:04d}.png"
            )


def finetune(epochs: int = 10, lr: float = 1.5e-4) -> dict:
    device = torch.device("cpu")
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
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
    train_ds = datasets.ImageFolder(FT_DATA / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(FT_DATA / "val", transform=eval_tf)
    assert train_ds.classes == CLASSES
    train_loader = DataLoader(train_ds, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=48, shuffle=False, num_workers=0)

    ckpt = torch.load(MODEL_OUT, map_location=device, weights_only=False)
    model = HatchCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])

    counts = [len(list((FT_DATA / "train" / c).glob("*"))) for c in CLASSES]
    inv = [1.0 / max(n, 1) for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    w[CLASSES.index("GRAVEL")] *= 1.35
    w[CLASSES.index("OTHER")] *= 0.95

    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"bal{epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            probs = torch.softmax(logits, dim=1)
            correct_p = probs.gather(1, y.view(-1, 1)).squeeze(1)
            is_g = y == CLASSES.index("GRAVEL")
            conf_loss = (
                (0.85 - correct_p[is_g]).clamp(min=0).mean() * 0.45
                if is_g.any()
                else torch.tensor(0.0)
            )
            (loss + conf_loss).backward()
            opt.step()
        sched.step()

        model.eval()
        vcor = vtot = 0
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
                    if int(pred[i]) == int(y[i]):
                        per[cname]["ok"] += 1
                        if float(p[i].max()) >= 0.8:
                            per[cname]["hi"] += 1
        va = vcor / max(vtot, 1)
        hi_g = per["GRAVEL"]["hi"] / max(per["GRAVEL"]["n"], 1)
        hi_o = per["OTHER"]["hi"] / max(per["OTHER"]["n"], 1)
        score = 0.4 * va + 0.4 * hi_g + 0.2 * hi_o
        history.append(
            {
                "epoch": epoch,
                "val_acc": va,
                "score": score,
                "per_class": {
                    c: {
                        "acc": per[c]["ok"] / max(per[c]["n"], 1),
                        "hi80": per[c]["hi"] / max(per[c]["n"], 1),
                    }
                    for c in CLASSES
                },
            }
        )
        print(
            f"bal {epoch}: val={va:.3f} score={score:.3f} "
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
                    "temperature": float(ckpt.get("temperature", 0.78)),
                    "finetune": "balanced_local_gravel",
                },
                MODEL_OUT,
            )
            print("  saved")

    meta = json.loads(META_OUT.read_text(encoding="utf-8")) if META_OUT.exists() else {}
    meta["balanced_local_gravel_ft"] = {"best_score": best, "history": history}
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    build()
    finetune()


if __name__ == "__main__":
    main()
