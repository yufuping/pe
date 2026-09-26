"""
Compare stronger backbones on six-class hatch fills (synth-only):
  1) EfficientNet-B0 @224 (ImageNet pretrained)
  2) DINOv2 ViT-S/14 (frozen then light FT) @224

Does not overwrite aggregate_specialist.pt.
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
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

from aggregate_predict import WeakPeriodicRecognizer
from texture_features import extract_hatch_roi
from train_aggregate_specialist import CLASSES
from train_imagenet_six import ImagenetHatchNet

ROOT = Path(__file__).resolve().parent
SPEC_DATA = ROOT / "data" / "aggregate_specialist"
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
TEST_IMGS = [
    ASSETS / "01a0db7b-e2b1-74c5-b6fd-e3f91198908d.jpg",
    ASSETS / "01a0db7b-e2ae-733f-b027-05048ed98f0f.jpg",
]
SIZE = 224
OUT_JSON = ROOT / "models" / "better_backbone_compare.json"


class EffNetB0Hatch(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class DinoV2Hatch(nn.Module):
    """DINOv2 ViT-S/14 + linear head. Input 224 (resized; patch 14 → 16 tokens/side)."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        dim = getattr(self.encoder, "embed_dim", 384)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # DINOv2 expects ImageNet-ish scale; our tensors are Normalize(0.5,0.5)
        feats = self.encoder(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return self.head(feats)


def _loaders(batch_size: int = 32):
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
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


def _class_weights(device) -> torch.Tensor:
    counts = [len(list((SPEC_DATA / "train" / c).glob("*"))) for c in CLASSES]
    inv = [1.0 / max(n, 1) for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    w[CLASSES.index("AR-SAND")] *= 1.25
    w[CLASSES.index("AR-CONC")] *= 1.1
    return w.to(device)


@torch.no_grad()
def _predict(model: nn.Module, path: Path, device: torch.device, temp: float = 0.78) -> dict:
    tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    img = Image.open(path).convert("RGB")
    box = extract_hatch_roi(np.asarray(img.convert("L")))
    x = tf(img.crop(box)).unsqueeze(0).to(device)
    probs = torch.softmax(model(x) / max(temp, 1e-3), dim=1)[0]
    vals, idxs = torch.topk(probs, k=len(CLASSES))
    top = [{"name": CLASSES[i], "confidence": round(float(v), 4)} for v, i in zip(vals.tolist(), idxs.tolist())]
    return {"name": top[0]["name"], "confidence": top[0]["confidence"], "top": top}


def train_one(
    name: str,
    model: nn.Module,
    out_path: Path,
    epochs: int,
    lr: float,
    freeze_epochs: int,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cpu")
    train_loader, val_loader = _loaders(batch_size=batch_size)
    model = model.to(device)
    w = _class_weights(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.02)

    def set_freeze(freeze_backbone: bool):
        if name.startswith("effnet"):
            for n, p in model.backbone.named_parameters():
                p.requires_grad = (not freeze_backbone) or n.startswith("classifier")
        elif name.startswith("dinov2"):
            for p in model.encoder.parameters():
                p.requires_grad = not freeze_backbone
            for p in model.head.parameters():
                p.requires_grad = True

    set_freeze(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            set_freeze(False)
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.25, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - freeze_epochs))
            print(f"[{name}] unfroze backbone")

        model.train()
        cor = tot = 0
        for x, y in tqdm(train_loader, desc=f"{name}{epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            crit(logits, y).backward()
            opt.step()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
        sched.step()

        model.eval()
        vcor = vtot = 0
        per = {c: {"ok": 0, "n": 0} for c in CLASSES}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                vcor += (pred == y).sum().item()
                vtot += y.size(0)
                for i in range(y.size(0)):
                    c = CLASSES[int(y[i])]
                    per[c]["n"] += 1
                    if int(pred[i]) == int(y[i]):
                        per[c]["ok"] += 1
        va = vcor / max(vtot, 1)
        history.append(
            {
                "epoch": epoch,
                "train_acc": cor / tot,
                "val_acc": va,
                "per_class": {c: per[c]["ok"] / max(per[c]["n"], 1) for c in CLASSES},
            }
        )
        print(
            f"[{name}] epoch {epoch}: train={cor/tot:.3f} val={va:.3f} "
            + " ".join(f"{c}={per[c]['ok']/max(per[c]['n'],1):.2f}" for c in CLASSES)
        )
        if va >= best:
            best = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": SIZE,
                    "val_acc": va,
                    "backbone": name,
                    "finetune": f"{name}_six",
                    "temperature": 0.78,
                },
                out_path,
            )
            print(f"[{name}] saved {out_path.name}")

    return {"best_val": best, "history": history, "checkpoint": str(out_path)}


def eval_all(models_spec: dict[str, tuple[nn.Module, Path]]) -> dict:
    device = torch.device("cpu")
    specialist = WeakPeriodicRecognizer(ROOT / "models" / "aggregate_specialist.pt")
    # ResNet18 imagenet (128) via existing helper
    r18_path = ROOT / "models" / "aggregate_imagenet_resnet18.pt"
    rows = []
    loaded = {}
    for name, (ctor, path) in models_spec.items():
        if not path.exists():
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = ctor().to(device)
        m.load_state_dict(ck["model_state"])
        m.eval()
        loaded[name] = (m, float(ck.get("temperature", 0.78)), ck.get("val_acc"))

    for path in TEST_IMGS:
        if not path.exists():
            continue
        entry = {"file": path.name, "preds": {}}
        sp = specialist.predict(Image.open(path), top_k=6)
        entry["preds"]["specialist_hatchcnn"] = {
            "name": sp["name"],
            "confidence": sp["confidence"],
            "top2": sp["top"][:2],
        }
        if r18_path.exists():
            # 128-size path from train_imagenet_six
            from train_imagenet_six import _predict_image as pred128

            m = ImagenetHatchNet(num_classes=len(CLASSES)).to(device)
            ck = torch.load(r18_path, map_location=device, weights_only=False)
            m.load_state_dict(ck["model_state"])
            m.eval()
            inn = pred128(m, path, device, temp=float(ck.get("temperature", 0.78)))
            entry["preds"]["resnet18_imagenet_128"] = {
                "name": inn["name"],
                "confidence": inn["confidence"],
                "top2": inn["top"][:2],
                "val_acc": ck.get("val_acc"),
            }
        for name, (m, temp, va) in loaded.items():
            out = _predict(m, path, device, temp=temp)
            entry["preds"][name] = {
                "name": out["name"],
                "confidence": out["confidence"],
                "top2": out["top"][:2],
                "val_acc": va,
            }
        rows.append(entry)
        print(f"\n=== {path.name} ===")
        for k, v in entry["preds"].items():
            print(f"  {k:28s} -> {v['name']:8s} {v['confidence']:.4f}  top2={v['top2']}")

    return {"size": SIZE, "classes": CLASSES, "rows": rows}


def main() -> None:
    assert SPEC_DATA.exists()
    results = {}
    # EfficientNet-B0: fewer epochs than R18 but 224 is heavier — 8 epochs
    results["effnet_b0_224"] = train_one(
        "effnet_b0",
        EffNetB0Hatch(len(CLASSES)),
        ROOT / "models" / "aggregate_effnet_b0_224.pt",
        epochs=8,
        lr=2e-4,
        freeze_epochs=2,
        batch_size=24,
    )
    # DINOv2: freeze longer, train head then light FT — 6 epochs (ViT slower on CPU)
    results["dinov2_vits14_224"] = train_one(
        "dinov2_vits14",
        DinoV2Hatch(len(CLASSES)),
        ROOT / "models" / "aggregate_dinov2_vits14_224.pt",
        epochs=6,
        lr=1e-3,
        freeze_epochs=3,
        batch_size=16,
    )
    compare = eval_all(
        {
            "effnet_b0_224": (lambda: EffNetB0Hatch(len(CLASSES)), ROOT / "models" / "aggregate_effnet_b0_224.pt"),
            "dinov2_vits14_224": (lambda: DinoV2Hatch(len(CLASSES)), ROOT / "models" / "aggregate_dinov2_vits14_224.pt"),
        }
    )
    compare["train_summaries"] = results
    OUT_JSON.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
