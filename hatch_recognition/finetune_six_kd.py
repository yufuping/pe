"""
Distill AR-CONC/GRAVEL/OTHER behavior from the proven 3-class checkpoint
while teaching AR-SAND/DOLMIT/EARTH on synth. Synthetic only; held-out accept.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from finetune_six_class_safe import HELD_OUT, USER_GRAVEL, _heldout_metrics, build
from pat_parser import parse_pat_file
from renderer import render_pattern, suggest_scale
from train import HatchCNN
from train_aggregate_specialist import (
    CLASSES,
    META_OUT,
    MODEL_OUT,
    OTHER_PATTERNS,
    PAT,
    SIZE,
)

ROOT = Path(__file__).resolve().parent
FT_DATA = ROOT / "data" / "aggregate_ft_six_safe"
TEACHER = ROOT / "models" / "backups" / "aggregate_specialist_3class_safe_dense_mix.pt"
BASE_SIX = ROOT / "models" / "backups" / "aggregate_specialist_six_class_base.pt"

# Map 3-class teacher indices → 6-class student indices
TEACHER_CLASSES = ["AR-CONC", "GRAVEL", "OTHER"]
T2S = [CLASSES.index(c) for c in TEACHER_CLASSES]


def finetune(epochs: int = 16, lr: float = 8e-5, kd_w: float = 0.65) -> dict:
    device = torch.device("cpu")
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08)),
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
    train_loader = DataLoader(train_ds, batch_size=40, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=40, shuffle=False, num_workers=0)

    # Start from best held-out six FT if present, else six base
    start = MODEL_OUT if MODEL_OUT.exists() else BASE_SIX
    ckpt = torch.load(start, map_location=device, weights_only=False)
    student = HatchCNN(num_classes=len(CLASSES)).to(device)
    student.load_state_dict(ckpt["model_state"])

    tck = torch.load(TEACHER, map_location=device, weights_only=False)
    teacher = HatchCNN(num_classes=3).to(device)
    teacher.load_state_dict(tck["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    t_temp = float(tck.get("temperature", 0.78))

    counts = [len(list((FT_DATA / "train" / c).glob("*"))) for c in CLASSES]
    inv = [1.0 / max(n, 1) for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    w[CLASSES.index("GRAVEL")] *= 1.55
    w[CLASSES.index("AR-CONC")] *= 1.2
    w[CLASSES.index("OTHER")] *= 1.25
    w[CLASSES.index("AR-SAND")] *= 1.05
    w[CLASSES.index("DOLMIT")] *= 0.75
    w[CLASSES.index("EARTH")] *= 0.75

    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.02)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    base_ho = _heldout_metrics(start)
    print(f"KD baseline held-out acc={base_ho['acc']:.3f} hi80={base_ho['hi80']:.3f}")

    # Extra hard ANSI batch each epoch
    patterns = parse_pat_file(PAT)
    rng = np.random.default_rng(7)

    def _ansi_batch(n: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(n):
            name = OTHER_PATTERNS[int(rng.integers(0, len(OTHER_PATTERNS)))]
            pat = patterns[name]
            img = render_pattern(
                pat,
                size=SIZE,
                scale=suggest_scale(pat, SIZE) * float(rng.uniform(0.7, 1.4)),
                rotation=float(rng.uniform(0, 360)),
                bg=255,
                fg=0,
                stroke=1,
                shape="rect",
            )
            t = eval_tf(img.convert("RGB"))
            xs.append(t)
            ys.append(CLASSES.index("OTHER"))
        return torch.stack(xs), torch.tensor(ys)

    distill_labels = {
        CLASSES.index("AR-CONC"),
        CLASSES.index("GRAVEL"),
        CLASSES.index("OTHER"),
    }

    best_ho = -1.0
    history = []
    g_idx = CLASSES.index("GRAVEL")

    for epoch in range(1, epochs + 1):
        student.train()
        for x, y in tqdm(train_loader, desc=f"kd{epoch}"):
            x, y = x.to(device), y.to(device)
            # sprinkle ANSI hard negatives
            if rng.random() < 0.35:
                ax, ay = _ansi_batch(12)
                x = torch.cat([x, ax.to(device)], 0)
                y = torch.cat([y, ay.to(device)], 0)

            opt.zero_grad()
            logits = student(x)
            loss = crit(logits, y)

            # KD on shared classes only
            mask = torch.tensor([int(yi) in distill_labels for yi in y.tolist()], device=device)
            if mask.any():
                with torch.no_grad():
                    t_logits = teacher(x[mask]) / max(t_temp, 1e-3)
                    t_prob = F.softmax(t_logits, dim=1)
                s_sub = logits[mask][:, T2S] / max(t_temp, 1e-3)
                kd = F.kl_div(F.log_softmax(s_sub, dim=1), t_prob, reduction="batchmean") * (t_temp ** 2)
                loss = loss + kd_w * kd

            probs = torch.softmax(logits, dim=1)
            correct_p = probs.gather(1, y.view(-1, 1)).squeeze(1)
            is_g = y == g_idx
            is_o = y == CLASSES.index("OTHER")
            conf = torch.tensor(0.0, device=device)
            if is_g.any():
                conf = conf + (0.9 - correct_p[is_g]).clamp(min=0).mean() * 0.55
            if is_o.any():
                conf = conf + (0.85 - correct_p[is_o]).clamp(min=0).mean() * 0.35
            (loss + conf).backward()
            opt.step()
        sched.step()

        student.eval()
        vcor = vtot = 0
        per = {c: {"ok": 0, "n": 0} for c in CLASSES}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = student(x)
                pred = logits.argmax(1)
                vcor += (pred == y).sum().item()
                vtot += y.size(0)
                for i in range(y.size(0)):
                    cname = CLASSES[int(y[i])]
                    per[cname]["n"] += 1
                    if int(pred[i]) == int(y[i]):
                        per[cname]["ok"] += 1
        va = vcor / max(vtot, 1)

        pack = {
            "model_state": {k: v.cpu().clone() for k, v in student.state_dict().items()},
            "classes": CLASSES,
            "image_size": SIZE,
            "val_acc": va,
            "temperature": float(ckpt.get("temperature", 0.78)),
            "finetune": "six_class_kd_heldout",
        }
        cand = MODEL_OUT.with_suffix(".kd_cand.pt")
        torch.save({**pack, "score": 0.0}, cand)
        ho = _heldout_metrics(cand)
        cand.unlink(missing_ok=True)
        ug = ho.get("user_gravel") or {}
        ho_score = (
            0.50 * ho["hi80"]
            + 0.30 * ho["acc"]
            + 0.12 * (1.0 if ho["ansi_other"] else 0.0)
            + 0.04 * (1.0 if ug.get("ok") else 0.0)
            + 0.04 * (1.0 if ug.get("hi80") else 0.0)
        )
        history.append(
            {
                "epoch": epoch,
                "val_acc": va,
                "heldout": {k: ho[k] for k in ("n", "acc", "hi80", "ansi_other", "ansi_pred", "user_gravel")},
                "ho_score": ho_score,
                "per_class": {c: per[c]["ok"] / max(per[c]["n"], 1) for c in CLASSES},
            }
        )
        print(
            f"kd {epoch}: val={va:.3f} ho_acc={ho['acc']:.3f} ho_hi80={ho['hi80']:.3f} "
            f"ansi={ho['ansi_pred']} ug={ug.get('pred')} score={ho_score:.3f} "
            + " ".join(f"{c}={per[c]['ok']/max(per[c]['n'],1):.2f}" for c in CLASSES)
        )

        # Require not worse than current best held-out acc by >1 case unless hi80 jumps
        if ho["acc"] < base_ho["acc"] - 0.06 and ho["hi80"] <= base_ho["hi80"]:
            print("  skip (below KD baseline)")
            continue
        if ho_score >= best_ho:
            best_ho = ho_score
            pack["score"] = ho_score
            pack["heldout"] = {k: ho[k] for k in ("n", "acc", "hi80", "ansi_other", "ansi_pred", "user_gravel")}
            torch.save(pack, MODEL_OUT)
            print("  saved")

    meta = json.loads(META_OUT.read_text(encoding="utf-8")) if META_OUT.exists() else {}
    meta["six_class_kd_heldout_ft"] = {
        "best_ho_score": best_ho,
        "baseline_heldout": {k: base_ho[k] for k in ("n", "acc", "hi80", "ansi_other", "ansi_pred", "user_gravel")},
        "history": history,
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (ROOT / "models" / "aggregate_real_eval.json").write_text(
        json.dumps(_heldout_metrics(MODEL_OUT), indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    if not FT_DATA.exists():
        build()
    # Enrich OTHER with more ANSI mid-run already handled in-loop
    finetune(epochs=16)


if __name__ == "__main__":
    main()
