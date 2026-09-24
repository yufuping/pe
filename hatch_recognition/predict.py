"""Inference for hatch pattern recognition from an image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import HatchCNN


def _ink_density_map(gray: np.ndarray, win: int = 64, stride: int = 32) -> list[tuple[float, int, int, int]]:
    """Return candidate crops ranked by local hatch-like ink density."""
    h, w = gray.shape
    # Ink = dark-ish pixels (CAD lines/dots), exclude near-black solid borders slightly
    ink = ((gray < 200) & (gray > 5)).astype(np.float32)
    # Prefer textured regions: high local variance among dark pixels helps vs solid lines
    cands: list[tuple[float, int, int, int]] = []
    side = min(win, h, w)
    if side < 16:
        return [(1.0, 0, 0, min(h, w))]

    for y in range(0, max(1, h - side + 1), stride):
        for x in range(0, max(1, w - side + 1), stride):
            patch = ink[y : y + side, x : x + side]
            dens = float(patch.mean())
            # Skip nearly empty (white) and nearly solid-black patches
            if dens < 0.02 or dens > 0.55:
                continue
            gpatch = gray[y : y + side, x : x + side]
            var = float(gpatch.std())
            score = dens * (1.0 + 0.02 * var)
            cands.append((score, x, y, side))

    cands.sort(reverse=True)
    return cands


def _collect_views(image: Image.Image, max_dense: int = 6) -> list[Image.Image]:
    """Full image + geometric crops + densest hatch patches for real CAD scenes."""
    views: list[Image.Image] = [image]
    w, h = image.size
    gray = np.array(image.convert("L"))

    # Geometric crops
    for frac in (0.12, 0.22):
        m = int(min(w, h) * frac)
        if w > 2 * m and h > 2 * m:
            views.append(image.crop((m, m, w - m, h - m)))
    if w > h * 1.25:
        views.append(image.crop((0, 0, w // 2, h)))
        views.append(image.crop((w // 2, 0, w, h)))
    if h > w * 1.25:
        views.append(image.crop((0, 0, w, h // 2)))
        views.append(image.crop((0, h // 2, w, h)))

    # Multi-scale dense windows (wall sections are often thin)
    for win, stride in ((96, 48), (128, 64), (160, 80), (64, 32)):
        if win > min(w, h):
            continue
        cands = _ink_density_map(gray, win=win, stride=max(16, stride))
        for score, x, y, side in cands[:max_dense]:
            views.append(image.crop((x, y, x + side, y + side)))
        if len(cands) >= 3:
            break

    # Deduplicate identical crops by box approx via size+mean
    uniq: list[Image.Image] = []
    seen: set[tuple] = set()
    for v in views:
        key = (v.size[0], v.size[1], int(np.array(v.convert("L")).mean()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    return uniq[:18]  # cap compute


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
    def predict(self, image: Image.Image | str | Path, top_k: int = 3, tta: bool = True) -> list[dict]:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        views = _collect_views(image) if tta else [image]

        # Softmax average + max fusion: strong hatch crops keep high confidence
        # even when full-frame white space / UI chrome votes elsewhere.
        weighted = None
        weight_sum = 0.0
        probs_list = []
        dens_list = []
        for view in views:
            x = self.tf(view).unsqueeze(0).to(self.device)
            logits = self.model(x)
            p = F.softmax(logits, dim=1)[0]
            g = np.array(view.convert("L"))
            dens = float(((g < 200) & (g > 5)).mean())
            dens_list.append(dens)
            wgt = 0.35 + dens
            if view.size == image.size:
                wgt *= 0.5
            weighted = p * wgt if weighted is None else weighted + p * wgt
            weight_sum += wgt
            probs_list.append(p)

        probs_avg = weighted / max(weight_sum, 1e-6)
        probs_max = torch.stack(probs_list, dim=0).max(dim=0).values
        probs = 0.55 * probs_avg + 0.45 * probs_max

        # Real drawings with textured hatch rarely are SOLID; soft-penalize SOLID
        # when selected crops look speckled rather than filled black.
        mean_dens = float(np.mean(dens_list)) if dens_list else 0.0
        if "SOLID" in self.classes and 0.02 < mean_dens < 0.45:
            solid_i = self.classes.index("SOLID")
            probs = probs.clone()
            probs[solid_i] *= 0.35

        probs = probs / probs.sum().clamp_min(1e-8)
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
