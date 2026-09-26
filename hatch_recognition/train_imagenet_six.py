"""
Experiment: ImageNet-pretrained ResNet18 fine-tuned on six-class hatch fills.
Saves a separate checkpoint; does not overwrite aggregate_specialist.pt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from aggregate_predict import WeakPeriodicRecognizer
from texture_features import extract_hatch_roi
from train_aggregate_specialist import CLASSES, SIZE

ROOT = Path(__file__).resolve().parent
SPEC_DATA = ROOT / "data" / "aggregate_specialist"
MODEL_OUT = ROOT / "models" / "aggregate_imagenet_resnet18.pt"
META_OUT = ROOT / "models" / "aggregate_imagenet_resnet18_meta.json"
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")

# User-provided AR-SAND-like screenshots
TEST_IMGS = [
    ASSETS / "01a0db7b-e2b1-74c5-b6fd-e3f91198908d.jpg",
    ASSETS / "01a0db7b-e2ae-733f-b027-05048ed98f0f.jpg",
]


class ImagenetHatchNet(nn.Module):
    """ResNet18 ImageNet backbone; grayscale → 3ch; new 6-way head."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1,H,W) grayscale → 3ch for ImageNet stem
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def _loaders(batch_size: int = 48):
    # ImageNet norm after mapping gray to 3ch in the model
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
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def _predict_image(model: nn.Module, path: Path, device: torch.device, temp: float = 0.78) -> dict:
    tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    img = Image.open(path).convert("RGB")
    g = np.asarray(img.convert("L"))
    box = extract_hatch_roi(g)
    crop = img.crop(box)
    x = tf(crop).unsqueeze(0).to(device)
    logits = model(x) / max(temp, 1e-3)
    probs = torch.softmax(logits, dim=1)[0]
    vals, idxs = torch.topk(probs, k=len(CLASSES))
    top = [
        {"name": CLASSES[i], "confidence": round(float(v), 4)}
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]
    return {"name": top[0]["name"], "confidence": top[0]["confidence"], "top": top, "roi_box": list(box)}


def train(epochs: int = 10, lr: float = 3e-4) -> dict:
    device = torch.device("cpu")
    train_loader, val_loader = _loaders()
    model = ImagenetHatchNet(num_classes=len(CLASSES)).to(device)

    # Freeze early layers first 2 epochs then unfreeze
    for name, p in model.backbone.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            p.requires_grad = False

    counts = [len(list((SPEC_DATA / "train" / c).glob("*"))) for c in CLASSES]
    inv = [1.0 / max(n, 1) for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    w[CLASSES.index("AR-SAND")] *= 1.2
    w[CLASSES.index("AR-CONC")] *= 1.1
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.02)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        if epoch == 3:
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - 2)
            print("  unfroze full backbone")

        model.train()
        cor = tot = 0
        for x, y in tqdm(train_loader, desc=f"imgnet{epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
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
        hi = float(np.mean([per[c]["hi"] / max(per[c]["n"], 1) for c in CLASSES]))
        score = 0.6 * va + 0.4 * hi
        history.append(
            {
                "epoch": epoch,
                "train_acc": cor / tot,
                "val_acc": va,
                "score": score,
                "per_class": {
                    c: {"acc": per[c]["ok"] / max(per[c]["n"], 1), "hi80": per[c]["hi"] / max(per[c]["n"], 1)}
                    for c in CLASSES
                },
            }
        )
        print(
            f"epoch {epoch}: train={cor/tot:.3f} val={va:.3f} score={score:.3f} "
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
                    "backbone": "resnet18_imagenet",
                    "finetune": "imagenet_resnet18_six",
                },
                MODEL_OUT,
            )
            print("  saved")

    meta = {"best_score": best, "history": history, "classes": CLASSES}
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def eval_compare() -> dict:
    device = torch.device("cpu")
    ck = torch.load(MODEL_OUT, map_location=device, weights_only=False)
    model = ImagenetHatchNet(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    specialist = WeakPeriodicRecognizer(ROOT / "models" / "aggregate_specialist.pt")
    rows = []
    for path in TEST_IMGS:
        if not path.exists():
            continue
        inn = _predict_image(model, path, device, temp=float(ck.get("temperature", 0.78)))
        sp = specialist.predict(Image.open(path), top_k=6)
        rows.append(
            {
                "file": path.name,
                "imagenet": inn,
                "specialist": {"name": sp["name"], "confidence": sp["confidence"], "top": sp["top"][:3]},
            }
        )
        print(f"\n=== {path.name} ===")
        print(f"  ImageNet-FT: {inn['name']} {inn['confidence']:.4f}  top2={inn['top'][:2]}")
        print(f"  Specialist:  {sp['name']} {sp['confidence']:.4f}  top2={sp['top'][:2]}")

    out = {"rows": rows, "imagenet_val_acc": ck.get("val_acc"), "imagenet_score": ck.get("score")}
    (ROOT / "models" / "imagenet_vs_specialist_eval.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out


def main() -> None:
    assert SPEC_DATA.exists(), f"missing {SPEC_DATA}"
    train(epochs=10)
    eval_compare()


if __name__ == "__main__":
    main()
