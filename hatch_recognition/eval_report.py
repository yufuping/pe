"""
Systematic evaluation: held-out synthetic test set + labeled real CAD screenshots.

Does not train or fine-tune — report only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from predict import HatchRecognizer
from train import HatchCNN

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT = ROOT / "models" / "eval_report.json"

# Ground truth for real screenshots collected during this project.
# Labels are from visual CAD hatch identity (and on-image annotations when present).
REAL_LABELS: dict[str, str] = {
    # AR-CONC
    "01a0d261-8064-7965-bbcb-9be9d02e1e5c.jpg": "AR-CONC",
    "01a0d262-d1c9-742e-9e6e-26968ab74972.jpg": "AR-CONC",
    "01a0d263-d93e-7705-873f-c3548132d977.jpg": "AR-CONC",
    "01a0d264-a8fa-7635-a753-8ac1bd36e145.jpg": "AR-CONC",
    "01a0d265-ce06-7d31-a510-70df7088e275.jpg": "AR-CONC",
    "01a0d379-4779-708f-bc09-f6f27b708e8f.jpg": "AR-CONC",  # labeled CONC (AR-C…)
    "CE4D9686-1522-4EDE-A013-9618C6A66357_L0_001.jpg": "AR-CONC",
    # GRAVEL
    "01a0d27e-806d-7840-9018-9a389ae6733b.jpg": "GRAVEL",
    "01a0d281-37df-7874-a0f8-445326aa781f.jpg": "GRAVEL",
    "01a0d2af-a2a6-7beb-9388-c0936f5ad85a.jpg": "GRAVEL",
    "01a0d2b4-45f1-7a20-a74b-4ea28e36ccc8.jpg": "GRAVEL",
    "01a0d377-c779-774f-981b-7b04deb8cef1.jpg": "GRAVEL",
    "47331B81-87A0-489D-BAE1-EE211207F0C1_L0_001.jpg": "GRAVEL",
}


def _metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    per = {}
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "correct": tp,
        }

    overall = sum(cm[i, i] for i in range(n)) / max(cm.sum(), 1)
    confusions = []
    for i, t in enumerate(classes):
        for j, p in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confusions.append({"true": t, "pred": p, "count": int(cm[i, j])})
    confusions.sort(key=lambda x: -x["count"])

    return {
        "accuracy": round(float(overall), 4),
        "n": int(cm.sum()),
        "per_class": per,
        "confusion_offdiag": confusions,
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }


@torch.no_grad()
def eval_test_set(model_path: Path) -> dict:
    """Fast center-crop eval on ImageFolder test split (no TTA)."""
    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    size = int(ckpt.get("image_size", 128))
    model = HatchCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    ds = datasets.ImageFolder(ROOT / "data/dataset/test", transform=tf)
    # Align folder order to checkpoint class order
    folder_classes = ds.classes
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="test-set"):
        logits = model(x.to(device))
        pred = logits.argmax(1).cpu().tolist()
        for yi, pi in zip(y.tolist(), pred):
            y_true.append(folder_classes[yi])
            y_pred.append(folder_classes[pi])

    # Remap to checkpoint class order for matrix
    return _metrics(y_true, y_pred, classes)


def eval_real(model_path: Path, tta: bool = True) -> dict:
    rec = HatchRecognizer(model_path)
    rows = []
    y_true, y_pred = [], []
    missing = []
    for fname, label in sorted(REAL_LABELS.items()):
        path = ASSETS / fname
        if not path.exists():
            missing.append(fname)
            continue
        top = rec.predict(path, top_k=3, tta=tta)
        pred = top[0]["name"]
        conf = top[0]["confidence"]
        ok = pred == label
        rows.append(
            {
                "file": fname,
                "true": label,
                "pred": pred,
                "confidence": conf,
                "correct": ok,
                "top3": [(d["name"], d["confidence"]) for d in top],
            }
        )
        y_true.append(label)
        y_pred.append(pred)

    classes_present = sorted(set(y_true) | set(y_pred) | set(rec.classes))
    # Prefer checkpoint order for known classes
    classes = [c for c in rec.classes if c in classes_present] + [
        c for c in classes_present if c not in rec.classes
    ]
    summary = _metrics(y_true, y_pred, classes)
    by_true = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        by_true[r["true"]]["n"] += 1
        by_true[r["true"]]["correct"] += int(r["correct"])

    return {
        "tta": tta,
        "missing": missing,
        "cases": rows,
        "by_true_label": {k: dict(v) for k, v in by_true.items()},
        "summary": summary,
    }


def main() -> None:
    model_path = ROOT / "models/best_model.pt"
    print("Evaluating synthetic test set…")
    test_report = eval_test_set(model_path)
    print(f"  test accuracy: {test_report['accuracy']}  n={test_report['n']}")

    print("Evaluating labeled real CAD screenshots (TTA)…")
    real_report = eval_real(model_path, tta=True)
    s = real_report["summary"]
    print(f"  real accuracy: {s['accuracy']}  n={s['n']}")

    report = {
        "model": str(model_path.relative_to(ROOT)),
        "synthetic_test": test_report,
        "real_screenshots": real_report,
        "notes": [
            "Synthetic test uses center resize, no TTA (matches train.py evaluate).",
            "Real screenshots use HatchRecognizer TTA (production path).",
            "No training/fine-tuning in this script.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")

    print("\n=== Synthetic per-class (recall) ===")
    for c, m in test_report["per_class"].items():
        print(f"  {c:8s}  R={m['recall']:.3f}  P={m['precision']:.3f}  F1={m['f1']:.3f}  n={m['support']}")

    print("\n=== Top synthetic confusions ===")
    for row in test_report["confusion_offdiag"][:12]:
        print(f"  {row['true']:8s} -> {row['pred']:8s}  x{row['count']}")

    print("\n=== Real screenshot cases ===")
    for r in real_report["cases"]:
        mark = "OK" if r["correct"] else "MISS"
        print(f"  [{mark}] {r['true']:8s} -> {r['pred']:8s} ({r['confidence']:.3f})  {r['file'][:24]}")


if __name__ == "__main__":
    main()
