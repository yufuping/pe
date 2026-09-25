"""
Weak-periodic hatch recognition pipeline.

Flow:
  1) Programmatic periodic router → ANSI/LINE/NET-like → do NOT use CNN
  2) Else weak-periodic specialist CNN (AR-CONC / GRAVEL / OTHER, expandable)

No classical prior gating that overrides CNN scores (that would hide CNN defects).
OTHER is learned so ANSI31 is rejected by the model itself when routing is skipped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from periodic_router import route_periodic
from texture_features import extract_hatch_roi
from train import HatchCNN

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "aggregate_specialist.pt"


class WeakPeriodicRecognizer:
    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else DEFAULT_MODEL
        device = torch.device("cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.classes: list[str] = list(ckpt["classes"])
        self.size = int(ckpt.get("image_size", 128))
        self.model = HatchCNN(num_classes=len(self.classes)).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device
        self.temperature = float(ckpt.get("temperature", 0.85))
        self.tf = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _views(self, image: Image.Image, box: tuple[int, int, int, int]) -> list[Image.Image]:
        x0, y0, x1, y1 = box
        roi = image.crop((x0, y0, x1, y1))
        views = [roi]
        w, h = roi.size
        gray = np.asarray(roi.convert("L"))
        for frac in (0.08, 0.18, 0.28):
            m = int(min(w, h) * frac)
            if w > 2 * m and h > 2 * m:
                crop = roi.crop((m, m, w - m, h - m))
                cd = float(((np.asarray(crop.convert("L")) < 200) & (np.asarray(crop.convert("L")) > 5)).mean())
                # Skip margin crops that erased most of the fill
                if cd >= 0.02 or frac <= 0.1:
                    views.append(crop)
        ink = ((gray < 200) & (gray > 5)).astype(np.float32)
        global_dens = float(ink.mean())
        # Sparse CAD fills (AR-CONC zoomed out) need a lower dens floor.
        dens_lo = 0.008 if global_dens < 0.06 else 0.025
        dens_hi = 0.65
        for side_frac in (0.55, 0.72, 0.88):
            side = min(w, h, max(64, int(min(w, h) * side_frac)))
            stride = max(16, side // 4)
            cands: list[tuple[float, int, int]] = []
            for y in range(0, max(1, h - side + 1), stride):
                for x in range(0, max(1, w - side + 1), stride):
                    dens = float(ink[y : y + side, x : x + side].mean())
                    if dens_lo < dens < dens_hi:
                        cands.append((dens, x, y))
            cands.sort(reverse=True)
            for _, x, y in cands[:5]:
                views.append(roi.crop((x, y, x + side, y + side)))
            if len(views) >= 12:
                break
        # Always keep a few center crops even when dens map is flat.
        for frac in (0.65, 0.45):
            side = min(w, h, max(64, int(min(w, h) * frac)))
            cx = max(0, (w - side) // 2)
            cy = max(0, (h - side) // 2)
            views.append(roi.crop((cx, cy, cx + side, cy + side)))
        base = views[0]
        views.append(base.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        views.append(base.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
        # Dedup by size+corner while preserving order
        seen = set()
        uniq: list[Image.Image] = []
        for v in views:
            key = (v.size, v.getbbox())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(v)
        return uniq[:16]

    @torch.no_grad()
    def _cnn_probs(self, views: list[Image.Image]) -> torch.Tensor:
        probs, weights = [], []
        dens_list = []
        areas = []
        for v in views:
            g = np.asarray(v.convert("L"))
            dens = float(((g < 200) & (g > 5)).mean())
            dens_list.append(dens)
            areas.append(float(v.size[0] * v.size[1]))
        max_area = max(areas) if areas else 1.0
        for i, v in enumerate(views):
            dens = dens_list[i]
            # Near-empty margin crops dilute GRAVEL → AR-CONC; skip them.
            if dens < 0.015 and i > 0:
                continue
            x = self.tf(v).unsqueeze(0).to(self.device)
            logits = self.model(x) / self.temperature
            p = F.softmax(logits, dim=1)[0]
            dens_w = 0.35 + 1.2 * min(dens, 0.45)  # denser fill patches dominate
            area_w = 0.5 + 0.85 * (areas[i] / max_area)
            if i == 0:
                area_w *= 1.1  # mild full-ROI bonus (was 1.25; dense locals often cleaner)
            conf = float(p.max())
            other_idx = self.classes.index("OTHER") if "OTHER" in self.classes else -1
            if other_idx >= 0 and float(p[other_idx]) > 0.55:
                # Full-frame OTHER often = border symbols; trust denser gravel patches more
                area_w *= 0.45 if dens > 0.15 else 0.3
            ar_idx = self.classes.index("AR-CONC") if "AR-CONC" in self.classes else -1
            if ar_idx >= 0 and dens < 0.05 and float(p[ar_idx]) > 0.5:
                area_w *= 0.25
            # Boost high-conf GRAVEL on dense patches (hatched interiors)
            gr_idx = self.classes.index("GRAVEL") if "GRAVEL" in self.classes else -1
            if gr_idx >= 0 and dens >= 0.2 and float(p[gr_idx]) > 0.85:
                area_w *= 1.35
            probs.append(p)
            weights.append(dens_w * area_w * (0.5 + 0.5 * conf))
        if not probs:
            # Fallback: at least the full ROI
            v = views[0]
            x = self.tf(v).unsqueeze(0).to(self.device)
            return F.softmax(self.model(x) / self.temperature, dim=1)[0]
        stack = torch.stack(probs, dim=0)
        w = torch.tensor(weights, device=self.device).view(-1, 1)
        avg = (stack * w).sum(0) / w.sum()
        soft = (stack * w).amax(0)
        return 0.72 * avg + 0.28 * soft

    def predict_cnn_only(self, image: Image.Image | str | Path, top_k: int = 3) -> dict:
        """CNN path only (no router). Used for debugging / ablating routing."""
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        box = extract_hatch_roi(np.asarray(image.convert("L")))
        probs = self._cnn_probs(self._views(image, box))
        probs = probs / probs.sum().clamp_min(1e-8)
        k = min(top_k, len(self.classes))
        vals, idxs = torch.topk(probs, k)
        results = [
            {"name": self.classes[i], "confidence": round(float(v), 4)}
            for v, i in zip(vals.tolist(), idxs.tolist())
        ]
        return {
            "name": results[0]["name"],
            "confidence": results[0]["confidence"],
            "top": results,
            "path": "cnn",
            "roi_box": list(box),
            "classes": self.classes,
        }

    def predict(self, image: Image.Image | str | Path, top_k: int = 3, use_router: bool = False) -> dict:
        """
        Default: CNN only — no gate hiding model mistakes.

        use_router=True is optional later (ANSI/LINE → programmatic). Off for now.
        """
        if use_router:
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            routed = route_periodic(image)
            if routed.is_periodic:
                conf = round(min(0.99, 0.75 + 0.5 * routed.top_share), 4)
                return {
                    "name": routed.label_hint or "PERIODIC",
                    "confidence": conf,
                    "top": [{"name": routed.label_hint or "PERIODIC", "confidence": conf}],
                    "path": "programmatic",
                    "router": {
                        "score": round(routed.score, 4),
                        "top_share": round(routed.top_share, 4),
                        "label_hint": routed.label_hint,
                    },
                    "roi_box": list(routed.roi_box),
                    "classes": self.classes,
                }

        return self.predict_cnn_only(image, top_k=top_k)


def predict_image(path: str | Path) -> dict:
    return WeakPeriodicRecognizer().predict(path)
