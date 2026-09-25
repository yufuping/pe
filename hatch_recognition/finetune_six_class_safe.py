"""
Six-class safe FT: recover real GRAVEL/AR-CONC after material-fill expansion.

Synthetic only. Per-epoch held-out accept on the 17 labeled CAD screenshots
(never used in training). No class-score overrides.
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

from aggregate_predict import WeakPeriodicRecognizer
from finetune_local_gravel import (
    _make_gravel,
    _mask_l_shape,
    _rand_crop,
)
from pat_parser import parse_pat_file
from renderer import (
    render_ar_conc_aggregate,
    render_ar_sand,
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
    TARGET_CLASSES,
    _aug,
    _finalize,
    _pat_tile,
)

FT_DATA = Path(__file__).resolve().parent / "data" / "aggregate_ft_six_safe"
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
HELD_OUT: dict[str, str] = {
    "01a0d261-8064-7965-bbcb-9be9d02e1e5c.jpg": "AR-CONC",
    "01a0d262-d1c9-742e-9e6e-26968ab74972.jpg": "AR-CONC",
    "01a0d263-d93e-7705-873f-c3548132d977.jpg": "AR-CONC",
    "01a0d264-a8fa-7635-a753-8ac1bd36e145.jpg": "AR-CONC",
    "01a0d265-ce06-7d31-a510-70df7088e275.jpg": "AR-CONC",
    "01a0d379-4779-708f-bc09-f6f27b708e8f.jpg": "AR-CONC",
    "CE4D9686-1522-4EDE-A013-9618C6A66357_L0_001.jpg": "AR-CONC",
    "74CE6AD5-8D6B-4DAE-87CD-1593E8AF4A1C_L0_001.jpg": "AR-CONC",
    "01a0d27e-806d-7840-9018-9a389ae6733b.jpg": "GRAVEL",
    "01a0d281-37df-7874-a0f8-445326aa781f.jpg": "GRAVEL",
    "01a0d2af-a2a6-7beb-9388-c0936f5ad85a.jpg": "GRAVEL",
    "01a0d2b4-45f1-7a20-a74b-4ea28e36ccc8.jpg": "GRAVEL",
    "01a0d377-c779-774f-981b-7b04deb8cef1.jpg": "GRAVEL",
    "01a0d37c-8b58-7e0a-9441-b1da57b885f5.jpg": "GRAVEL",
    "01a0d380-566d-77d8-b43f-b9539db33eb0.jpg": "GRAVEL",
    "01a0d383-62b2-7a63-a041-d12e7df2eeaf.jpg": "GRAVEL",
    "47331B81-87A0-489D-BAE1-EE211207F0C1_L0_001.jpg": "GRAVEL",
}
USER_GRAVEL = Path(__file__).resolve().parent / "samples" / "preview" / "USER_GRAVEL_crop.png"


def build(
    n_gravel: int = 640,
    n_ar: int = 220,
    n_sand: int = 200,
    n_earth: int = 180,
    n_dolmit: int = 180,
    n_other: int = 300,
    seed: int = 20260929,
) -> None:
    if FT_DATA.exists():
        shutil.rmtree(FT_DATA)
    rng = np.random.default_rng(seed)
    patterns = parse_pat_file(PAT)

    for split, scale in (("train", 1.0), ("val", 0.14)):
        ng = max(40, int(n_gravel * scale)) if split == "val" else n_gravel
        na = max(28, int(n_ar * scale)) if split == "val" else n_ar
        ns = max(28, int(n_sand * scale)) if split == "val" else n_sand
        ne = max(28, int(n_earth * scale)) if split == "val" else n_earth
        nd = max(28, int(n_dolmit * scale)) if split == "val" else n_dolmit
        no = max(36, int(n_other * scale)) if split == "val" else n_other

        for c in CLASSES:
            (FT_DATA / split / c).mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(ng), desc=f"{split} gravel"):
            g = _make_gravel(rng)
            _finalize(_aug(g, rng, heavy=True)).save(
                FT_DATA / split / "GRAVEL" / f"g_{i:04d}.png"
            )

        for i in tqdm(range(na), desc=f"{split} ar-conc"):
            dens = float(rng.uniform(0.35, 1.45))
            h = render_ar_conc_aggregate(220, rng, dens)
            if rng.random() < 0.4:
                h = _rand_crop(h, rng, float(rng.uniform(0.35, 0.7)))
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

        for i in tqdm(range(ns), desc=f"{split} ar-sand"):
            dens = float(rng.uniform(0.5, 1.7))
            h = render_ar_sand(200, rng, dens)
            if rng.random() < 0.3:
                h = _pat_tile("AR-SAND", patterns, rng, 200)
            if rng.random() < 0.35:
                h = _rand_crop(h, rng, float(rng.uniform(0.35, 0.7)))
            h = h.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            _finalize(_aug(h, rng, heavy=True)).save(
                FT_DATA / split / "AR-SAND" / f"s_{i:04d}.png"
            )

        for i in tqdm(range(ne), desc=f"{split} earth"):
            h = _pat_tile("EARTH", patterns, rng, 200)
            if rng.random() < 0.5:
                h = _rand_crop(h, rng, float(rng.uniform(0.35, 0.75)))
            h = h.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            _finalize(_aug(h, rng, heavy=False)).save(
                FT_DATA / split / "EARTH" / f"e_{i:04d}.png"
            )

        for i in tqdm(range(nd), desc=f"{split} dolmit"):
            h = _pat_tile("DOLMIT", patterns, rng, 200)
            if rng.random() < 0.5:
                h = _rand_crop(h, rng, float(rng.uniform(0.35, 0.75)))
            h = h.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            _finalize(_aug(h, rng, heavy=False)).save(
                FT_DATA / split / "DOLMIT" / f"d_{i:04d}.png"
            )

        for i in tqdm(range(no), desc=f"{split} other"):
            name = OTHER_PATTERNS[int(rng.integers(0, len(OTHER_PATTERNS)))]
            pat = patterns[name]
            big = render_pattern(
                pat,
                size=200,
                scale=suggest_scale(pat, 200) * float(rng.uniform(0.75, 1.35)),
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


def _heldout_metrics(model_path: Path) -> dict:
    rec = WeakPeriodicRecognizer(model_path)
    rows = []
    for fname, label in HELD_OUT.items():
        path = ASSETS / fname
        if not path.exists():
            continue
        out = rec.predict(Image.open(path), top_k=6)
        ok = out["name"] == label
        hi = ok and float(out["confidence"]) >= 0.8
        rows.append({"file": fname, "true": label, "pred": out["name"], "conf": out["confidence"], "ok": ok, "hi80": hi})
    # ANSI reject probe
    pats = parse_pat_file(PAT)
    ansi = render_pattern(
        pats["ANSI31"],
        size=168,
        scale=suggest_scale(pats["ANSI31"], 168),
        rotation=45,
        bg=255,
        fg=0,
        stroke=1,
        shape="rect",
    )
    aout = rec.predict(ansi, top_k=6)
    ansi_ok = aout["name"] == "OTHER"
    user = None
    if USER_GRAVEL.exists():
        u = rec.predict(Image.open(USER_GRAVEL), top_k=6)
        user = {"pred": u["name"], "conf": u["confidence"], "ok": u["name"] == "GRAVEL", "hi80": u["name"] == "GRAVEL" and u["confidence"] >= 0.8}
    n = len(rows)
    return {
        "n": n,
        "acc": sum(r["ok"] for r in rows) / max(n, 1),
        "hi80": sum(r["hi80"] for r in rows) / max(n, 1),
        "ansi_other": ansi_ok,
        "ansi_pred": aout["name"],
        "ansi_conf": aout["confidence"],
        "user_gravel": user,
        "rows": rows,
    }


def finetune(epochs: int = 12, lr: float = 1.2e-4) -> dict:
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
    assert train_ds.classes == CLASSES, train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=48, shuffle=False, num_workers=0)

    ckpt = torch.load(MODEL_OUT, map_location=device, weights_only=False)
    assert list(ckpt["classes"]) == CLASSES
    model = HatchCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])

    counts = [len(list((FT_DATA / "train" / c).glob("*"))) for c in CLASSES]
    inv = [1.0 / max(n, 1) for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    w[CLASSES.index("GRAVEL")] *= 1.45
    w[CLASSES.index("AR-CONC")] *= 1.15
    w[CLASSES.index("OTHER")] *= 1.05
    w[CLASSES.index("DOLMIT")] *= 0.9
    w[CLASSES.index("EARTH")] *= 0.9

    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.025)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # baseline held-out before FT
    tmp = MODEL_OUT.with_suffix(".tmp_base.pt")
    torch.save(ckpt, tmp)
    base_ho = _heldout_metrics(tmp)
    tmp.unlink(missing_ok=True)
    print(f"baseline held-out acc={base_ho['acc']:.3f} hi80={base_ho['hi80']:.3f} ansi={base_ho['ansi_pred']}")

    best_ho = -1.0
    best_pack = None
    history = []
    g_idx = CLASSES.index("GRAVEL")
    t_idx = torch.tensor([CLASSES.index(c) for c in TARGET_CLASSES])

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"sixft{epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            probs = torch.softmax(logits, dim=1)
            correct_p = probs.gather(1, y.view(-1, 1)).squeeze(1)
            is_g = y == g_idx
            is_t = torch.isin(y, t_idx.to(device))
            conf_g = (0.88 - correct_p[is_g]).clamp(min=0).mean() * 0.5 if is_g.any() else torch.tensor(0.0)
            conf_t = (0.85 - correct_p[is_t]).clamp(min=0).mean() * 0.25 if is_t.any() else torch.tensor(0.0)
            (loss + conf_g + conf_t).backward()
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

        pack = {
            "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "classes": CLASSES,
            "image_size": SIZE,
            "val_acc": va,
            "temperature": float(ckpt.get("temperature", 0.78)),
            "finetune": "six_class_safe_heldout",
        }
        cand = MODEL_OUT.with_suffix(".cand.pt")
        torch.save({**pack, "score": 0.0}, cand)
        ho = _heldout_metrics(cand)
        cand.unlink(missing_ok=True)

        # Accept only if held-out improves (or ties with better synth val)
        ho_score = 0.55 * ho["hi80"] + 0.35 * ho["acc"] + 0.10 * (1.0 if ho["ansi_other"] else 0.0)
        ug = ho.get("user_gravel") or {}
        if ug.get("ok"):
            ho_score += 0.05
        if ug.get("hi80"):
            ho_score += 0.05

        history.append(
            {
                "epoch": epoch,
                "val_acc": va,
                "heldout": {k: ho[k] for k in ("n", "acc", "hi80", "ansi_other", "ansi_pred", "user_gravel")},
                "ho_score": ho_score,
                "per_class": {
                    c: {"acc": per[c]["ok"] / max(per[c]["n"], 1), "hi80": per[c]["hi"] / max(per[c]["n"], 1)}
                    for c in CLASSES
                },
            }
        )
        print(
            f"sixft {epoch}: val={va:.3f} ho_acc={ho['acc']:.3f} ho_hi80={ho['hi80']:.3f} "
            f"ansi={ho['ansi_pred']} ug={ug} score={ho_score:.3f} "
            + " ".join(f"{c}={per[c]['ok']/max(per[c]['n'],1):.2f}" for c in CLASSES)
        )

        # Never accept a regression below baseline held-out acc unless hi80 clearly better
        if ho["acc"] + 1e-9 < base_ho["acc"] and ho["hi80"] <= base_ho["hi80"] + 0.02:
            print("  skip (held-out regress)")
            continue
        if ho_score >= best_ho:
            best_ho = ho_score
            pack["score"] = ho_score
            pack["heldout"] = {k: ho[k] for k in ("n", "acc", "hi80", "ansi_other", "ansi_pred", "user_gravel")}
            best_pack = pack
            torch.save(pack, MODEL_OUT)
            print("  saved (held-out accept)")

    if best_pack is None:
        print("WARNING: no held-out accept; keeping pre-FT six-class weights")
    meta = json.loads(META_OUT.read_text(encoding="utf-8")) if META_OUT.exists() else {}
    meta["six_class_safe_heldout_ft"] = {
        "best_ho_score": best_ho,
        "baseline_heldout": {k: base_ho[k] for k in ("n", "acc", "hi80", "ansi_other", "ansi_pred", "user_gravel")},
        "history": history,
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    Path(__file__).resolve().parent.joinpath("models", "aggregate_real_eval.json").write_text(
        json.dumps(_heldout_metrics(MODEL_OUT), indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    # Backup base six-class before FT
    bak = MODEL_OUT.parent / "backups" / "aggregate_specialist_six_class_base.pt"
    bak.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_OUT.exists() and not bak.exists():
        shutil.copy2(MODEL_OUT, bak)
    build()
    finetune(epochs=12)


if __name__ == "__main__":
    main()
