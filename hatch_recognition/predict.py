"""Inference for hatch pattern recognition from an image."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import HatchCNN


class HatchRecognizer:
    def __init__(self, model_path: str | Path, descriptions: dict[str, str] | None = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.size = int(ckpt.get("image_size", 128))
        self.model = HatchCNN(num_classes=len(self.classes)).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device
        self.descriptions = descriptions or {}
        self.tf = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path, top_k: int = 3) -> list[dict]:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        x = self.tf(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        k = min(top_k, len(self.classes))
        values, indices = torch.topk(probs, k)
        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            name = self.classes[idx]
            results.append(
                {
                    "name": name,
                    "confidence": round(float(score), 4),
                    "description": self.descriptions.get(name, ""),
                }
            )
        return results


def load_descriptions(meta_path: str | Path | None = None) -> dict[str, str]:
    if meta_path and Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return meta.get("descriptions", {})
    return {}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Recognize hatch pattern in an image")
    p.add_argument("image", help="Path to image file")
    p.add_argument("--model", default="models/best_model.pt")
    p.add_argument("--meta", default="data/dataset/meta.json")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    rec = HatchRecognizer(args.model, load_descriptions(args.meta))
    for i, r in enumerate(rec.predict(args.image, top_k=args.top_k), 1):
        print(f"{i}. {r['name']:12s}  {r['confidence']*100:5.1f}%  {r['description']}")
