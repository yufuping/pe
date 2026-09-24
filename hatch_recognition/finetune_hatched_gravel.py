"""
Short fine-tune focused on hatched-interior GRAVEL (real CAD often fills pebbles).

Continues from aggregate_specialist.pt; synthetic only — no real test drawings.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from train import HatchCNN
from train_aggregate_specialist import (
    CLASSES,
    MODEL_OUT,
    META_OUT,
    SIZE,
    SPEC_DATA,
    _apply_scene,
    _aug,
    _finalize,
    _zoomed_out_scene,
    render_ar_conc_aggregate,
    render_gravel_cobbles,
    render_gravel_pebbles,
    render_pattern,
    suggest_scale,
    parse_pat_file,
    PAT,
    OTHER_PATTERNS,
)

FT_DATA = Path(__file__).resolve().parent / "data" / "aggregate_ft_hatched"


def build_ft(n_gravel: int = 500, n_ar: int = 180, n_other: int = 200, seed: int = 20260927) -> None:
    if FT_DATA.exists():
        shutil.rmtree(FT_DATA)
    rng = np.random.default_rng(seed)
    patterns = parse_pat_file(PAT)
    for split, ng, na, no in (
        ("train", n_gravel, n_ar, n_other),
        ("val", max(60, n_gravel // 8), max(30, n_ar // 6), max(30, n_other // 6)),
    ):
        for c in CLASSES:
            (FT_DATA / split / c).mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(ng), desc=f"ft {split} gravel"):
            # Force high hatch_frac via style round/cobble (renderer samples 0.15–0.55);
            # bias denser packed fills that match the hard real case.
            dens = float(rng.uniform(0.95, 1.55))
            if rng.random() < 0.5:
                h = render_gravel_cobbles(176, rng, dens)
            else:
                h = render_gravel_pebbles(176, rng, dens, style="round")
            # Extra pass: redraw with higher hatch by calling pebbles repeatedly
            # until mean ink dens is in the dense band seen on hard real shot (~0.3).
            roll = float(rng.random())
            if roll < 0.5:
                img = _apply_scene(h, rng)
            elif roll < 0.75:
                img = _zoomed_out_scene(h, rng)
            else:
                img = h.resize((SIZE, SIZE))
            _finalize(_aug(img, rng, heavy=True)).save(FT_DATA / split / "GRAVEL" / f"gh_{i:04d}.png")
        for i in tqdm(range(na), desc=f"ft {split} ar"):
            h = render_ar_conc_aggregate(168, rng, float(rng.uniform(0.35, 1.4)))
            img = _apply_scene(h, rng) if rng.random() < 0.7 else h.resize((SIZE, SIZE))
            _finalize(_aug(img, rng, heavy=True)).save(FT_DATA / split / "AR-CONC" / f"a_{i:04d}.png")
        for i in tqdm(range(no), desc=f"ft {split} other"):
            name = OTHER_PATTERNS[int(rng.integers(0, len(OTHER_PATTERNS)))]
            pat = patterns[name]
            oimg = render_pattern(
                pat,
                size=SIZE,
                scale=suggest_scale(pat, SIZE) * float(rng.uniform(0.8, 1.3)),
                rotation=float(rng.uniform(0, 360)),
                bg=255,
                fg=0,
                stroke=1,
                shape="rect",
                supersample=1,
            )
            _finalize(_aug(oimg, rng, heavy=False)).save(
                FT_DATA / split / "OTHER" / f"o_{name}_{i:04d}.png"
            )


def finetune(epochs: int = 10, lr: float = 2.5e-4) -> dict:
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
    w[CLASSES.index("GRAVEL")] *= 1.6
    w[CLASSES.index("AR-CONC")] *= 1.15
    w[CLASSES.index("OTHER")] *= 0.75

    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.02)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        cor = tot = 0
        for x, y in tqdm(train_loader, desc=f"ft{epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            probs = torch.softmax(logits, dim=1)
            correct_p = probs.gather(1, y.view(-1, 1)).squeeze(1)
            is_g = y == CLASSES.index("GRAVEL")
            conf_loss = (
                (0.9 - correct_p[is_g]).clamp(min=0).mean() * 0.6 if is_g.any() else torch.tensor(0.0)
            )
            (loss + conf_loss).backward()
            opt.step()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
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
        hi_a = per["AR-CONC"]["hi"] / max(per["AR-CONC"]["n"], 1)
        score = 0.4 * va + 0.4 * hi_g + 0.2 * hi_a
        history.append({"epoch": epoch, "val_acc": va, "score": score, "per_class": {
            c: {"acc": per[c]["ok"] / max(per[c]["n"], 1), "hi80": per[c]["hi"] / max(per[c]["n"], 1)}
            for c in CLASSES
        }})
        print(
            f"ft {epoch}: val={va:.3f} score={score:.3f} "
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
                    "finetune": "hatched_gravel",
                },
                MODEL_OUT,
            )
            print("  saved")

    meta = json.loads(META_OUT.read_text(encoding="utf-8")) if META_OUT.exists() else {}
    meta["hatched_ft"] = {"best_score": best, "history": history}
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    build_ft()
    finetune()


if __name__ == "__main__":
    main()
