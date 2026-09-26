"""
CAD-hatch-oriented architectures (synth-only six-class):

  1) GaborCNN — fixed multi-orientation / multi-frequency Gabor stem + tiny CNN
  2) BilinearCNN — dual-stream CNN + compact bilinear (orderless) pooling
  3) GaborStemResNet18 — ImageNet ResNet18 with conv1 replaced by fixed Gabor bank

Does not overwrite aggregate_specialist.pt. Compares on user AR-SAND screenshots.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from aggregate_predict import WeakPeriodicRecognizer
from texture_features import extract_hatch_roi
from train_aggregate_specialist import CLASSES, SIZE
from train_imagenet_six import ImagenetHatchNet, _predict_image as pred128

ROOT = Path(__file__).resolve().parent
SPEC_DATA = ROOT / "data" / "aggregate_specialist"
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
TEST_IMGS = [
    ASSETS / "01a0db7b-e2b1-74c5-b6fd-e3f91198908d.jpg",
    ASSETS / "01a0db7b-e2ae-733f-b027-05048ed98f0f.jpg",
]
OUT_JSON = ROOT / "models" / "texture_arch_compare.json"
TEMP = 0.78


# ---------------------------------------------------------------------------
# Gabor bank
# ---------------------------------------------------------------------------
def _make_gabor_kernels(
    orientations: int = 8,
    wavelengths: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0),
    ksize: int = 15,
    sigma_scale: float = 0.45,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Return (C,1,k,k) real Gabor kernels covering angle × frequency."""
    kernels = []
    half = ksize // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
    for lam in wavelengths:
        sigma = max(lam * sigma_scale, 1.0)
        for i in range(orientations):
            theta = math.pi * i / orientations
            x_t = xx * math.cos(theta) + yy * math.sin(theta)
            y_t = -xx * math.sin(theta) + yy * math.cos(theta)
            gauss = np.exp(-(x_t**2 + (gamma**2) * y_t**2) / (2.0 * sigma**2))
            wave = np.cos(2.0 * math.pi * x_t / lam)
            k = gauss * wave
            k = k - k.mean()
            n = np.linalg.norm(k)
            if n > 1e-8:
                k = k / n
            kernels.append(k)
    arr = np.stack(kernels, axis=0)[:, None, :, :]
    return torch.from_numpy(arr)


class GaborStem(nn.Module):
    """Fixed Gabor filter bank + BN (learned). Input (N,1,H,W)."""

    def __init__(self, orientations: int = 8, wavelengths=(3.0, 5.0, 8.0, 12.0), ksize: int = 15):
        super().__init__()
        weight = _make_gabor_kernels(orientations, wavelengths, ksize)
        self.register_buffer("weight", weight)
        self.out_channels = weight.shape[0]
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.ksize = ksize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        pad = self.ksize // 2
        y = F.conv2d(x, self.weight, padding=pad)
        return F.relu(self.bn(y), inplace=True)


class GaborCNN(nn.Module):
    """Engineering-primary: Gabor stem + lightweight spatial CNN + GAP."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.stem = GaborStem()
        c = self.stem.out_channels  # 32
        self.body = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 160, 3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


class CompactBilinearPool(nn.Module):
    """Outer-product bilinear pool with signed-sqrt + L2 (orderless texture)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (N,C,H,W) — global sum of outer products via einsum
        n, c, h, w = a.shape
        a_flat = a.view(n, c, h * w)
        b_flat = b.view(n, c, h * w)
        # (N,C,C) ≈ Σ_spatial a ⊗ b
        bil = torch.bmm(a_flat, b_flat.transpose(1, 2)) / (h * w)
        bil = bil.view(n, c * c)
        bil = torch.sign(bil) * torch.sqrt(bil.abs().clamp_min(1e-8))
        bil = F.normalize(bil, p=2, dim=1)
        return bil


class BilinearCNN(nn.Module):
    """Dual-stream CNN + bilinear pooling (translation-orderless texture)."""

    def __init__(self, num_classes: int = 6, feat_dim: int = 48):
        super().__init__()
        self.feat_dim = feat_dim

        def stream() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, feat_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.stream_a = stream()
        self.stream_b = stream()
        self.pool = CompactBilinearPool()
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim * feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        a = self.stream_a(x)
        b = self.stream_b(x)
        return self.head(self.pool(a, b))


class GaborStemResNet18(nn.Module):
    """ResNet18 with first conv replaced by fixed Gabor bank (projected to 64)."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.gabor = GaborStem()
        # Project Gabor channels → ResNet stem width
        self.proj = nn.Sequential(
            nn.Conv2d(self.gabor.out_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Drop conv1; keep bn1/relu/maxpool/layers — but bn1 expects 64 after our proj
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.gabor(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def _loaders(batch_size: int = 48):
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


def _class_weights(device) -> torch.Tensor:
    counts = [len(list((SPEC_DATA / "train" / c).glob("*"))) for c in CLASSES]
    inv = [1.0 / max(n, 1) for n in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    w = w / w.mean()
    w[CLASSES.index("AR-SAND")] *= 1.25
    w[CLASSES.index("AR-CONC")] *= 1.1
    return w.to(device)


@torch.no_grad()
def _predict(model: nn.Module, path: Path, device: torch.device, temp: float = TEMP) -> dict:
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
    batch_size: int = 48,
) -> dict:
    device = torch.device("cpu")
    train_loader, val_loader = _loaders(batch_size=batch_size)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] trainable params={n_params:,}")
    w = _class_weights(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.02)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = -1.0
    history = []

    for epoch in range(1, epochs + 1):
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
                    "temperature": TEMP,
                    "trainable_params": n_params,
                },
                out_path,
            )
            print(f"[{name}] saved {out_path.name}")

    return {"best_val": best, "history": history, "checkpoint": str(out_path), "trainable_params": n_params}


def eval_all(models_spec: dict[str, tuple]) -> dict:
    device = torch.device("cpu")
    specialist = WeakPeriodicRecognizer(ROOT / "models" / "aggregate_specialist.pt")
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
        loaded[name] = (m, float(ck.get("temperature", TEMP)), ck.get("val_acc"), ck.get("trainable_params"))

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
            m = ImagenetHatchNet(num_classes=len(CLASSES)).to(device)
            ck = torch.load(r18_path, map_location=device, weights_only=False)
            m.load_state_dict(ck["model_state"])
            m.eval()
            inn = pred128(m, path, device, temp=float(ck.get("temperature", TEMP)))
            entry["preds"]["resnet18_imagenet_128"] = {
                "name": inn["name"],
                "confidence": inn["confidence"],
                "top2": inn["top"][:2],
                "val_acc": ck.get("val_acc"),
            }
        for name, (m, temp, va, npar) in loaded.items():
            out = _predict(m, path, device, temp=temp)
            entry["preds"][name] = {
                "name": out["name"],
                "confidence": out["confidence"],
                "top2": out["top"][:2],
                "val_acc": va,
                "trainable_params": npar,
            }
        rows.append(entry)
        print(f"\n=== {path.name} ===")
        for k, v in entry["preds"].items():
            print(f"  {k:28s} -> {v['name']:8s} {v['confidence']:.4f}  top2={v['top2']}")

    return {"size": SIZE, "classes": CLASSES, "rows": rows}


def main() -> None:
    assert SPEC_DATA.exists()
    results = {}

    results["gabor_cnn"] = train_one(
        "gabor_cnn",
        GaborCNN(len(CLASSES)),
        ROOT / "models" / "aggregate_gabor_cnn.pt",
        epochs=12,
        lr=3e-4,
        batch_size=64,
    )
    results["bilinear_cnn"] = train_one(
        "bilinear_cnn",
        BilinearCNN(len(CLASSES)),
        ROOT / "models" / "aggregate_bilinear_cnn.pt",
        epochs=12,
        lr=3e-4,
        batch_size=48,
    )
    results["gabor_stem_resnet18"] = train_one(
        "gabor_stem_resnet18",
        GaborStemResNet18(len(CLASSES)),
        ROOT / "models" / "aggregate_gabor_stem_resnet18.pt",
        epochs=8,
        lr=2e-4,
        batch_size=40,
    )

    compare = eval_all(
        {
            "gabor_cnn": (lambda: GaborCNN(len(CLASSES)), ROOT / "models" / "aggregate_gabor_cnn.pt"),
            "bilinear_cnn": (lambda: BilinearCNN(len(CLASSES)), ROOT / "models" / "aggregate_bilinear_cnn.pt"),
            "gabor_stem_resnet18": (
                lambda: GaborStemResNet18(len(CLASSES)),
                ROOT / "models" / "aggregate_gabor_stem_resnet18.pt",
            ),
        }
    )
    # Drop heavy per-epoch history from JSON; keep summary
    slim = {
        k: {
            "best_val": v["best_val"],
            "checkpoint": v["checkpoint"],
            "trainable_params": v["trainable_params"],
            "last_epoch": v["history"][-1] if v["history"] else None,
        }
        for k, v in results.items()
    }
    compare["train_summaries"] = slim
    OUT_JSON.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
