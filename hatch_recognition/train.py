"""CNN model and training for hatch pattern classification."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class HatchCNN(nn.Module):
    """Compact CNN for grayscale hatch pattern recognition."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_loaders(data_dir: str | Path, batch_size: int = 64, size: int = 128):
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    root = Path(data_dir)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        train_ds.classes,
    )


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(
    data_dir: str | Path,
    model_dir: str | Path,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    size: int = 128,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = build_loaders(data_dir, batch_size, size)
    model = HatchCNN(num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=running / total, acc=correct / total)
        sched.step()

        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(f"epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": size,
                    "val_acc": val_acc,
                },
                model_dir / "best_model.pt",
            )

    # Final test with best checkpoint
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    result = {
        "classes": classes,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "device": str(device),
    }
    (model_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test accuracy: {test_acc:.3f}")
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()
    train(args.data, args.model_dir, epochs=args.epochs, batch_size=args.batch_size, size=args.size)
