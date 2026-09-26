"""
Programmatic router for strong-periodic hatch fills (ANSI31/LINE/NET/…).

Detects parallel-line families via FFT *angular* energy concentration.
Weak-periodic fills (AR-CONC, GRAVEL) have isotropic spectra and stay for CNN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from texture_features import extract_hatch_roi


@dataclass
class RouterResult:
    is_periodic: bool
    score: float
    angle_deg: float | None
    spacing_px: float | None
    label_hint: str | None
    roi_box: tuple[int, int, int, int]
    top_share: float


def _gray_roi(img: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    full = np.asarray(img.convert("L"), dtype=np.float32)
    box = extract_hatch_roi(full.astype(np.uint8))
    x0, y0, x1, y1 = box
    return full[y0:y1, x0:x1], box


def _periodicity(roi: np.ndarray) -> tuple[float, float, float | None, float | None]:
    """
    Returns (score, top_share, angle_deg, spacing_px).

    top_share = fraction of FFT power in the dominant orientation bin.
    Parallel-line hatches concentrate energy (typically >0.35);
    gravel/concrete stay diffuse (typically <0.20).
    """
    h, w = roi.shape
    if h < 24 or w < 24:
        return 0.0, 0.0, None, None

    g = roi - roi.mean()
    if float(g.std()) < 1e-3:
        return 0.0, 0.0, None, None

    f = np.fft.fftshift(np.fft.fft2(g))
    p = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    p[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0

    yy, xx = np.mgrid[:h, :w]
    dy, dx = yy - cy, xx - cx
    rr = np.sqrt(dx * dx + dy * dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    min_r = max(3.0, min(h, w) * 0.05)
    mask = rr >= min_r
    if not mask.any():
        return 0.0, 0.0, None, None

    n_bins = 36
    bins = np.linspace(0.0, 180.0, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = mask & (ang >= bins[i]) & (ang < bins[i + 1])
        hist[i] = float(p[m].sum())
    total = float(hist.sum() + 1e-12)
    hist /= total
    top_i = int(hist.argmax())
    top_share = float(hist[top_i])
    ang_conc = float(hist[top_i] / (hist.mean() + 1e-12))

    # Dominant wave-normal angle and spacing from peak in that wedge (+ opposite)
    wedge = mask & (
        ((ang >= bins[top_i]) & (ang < bins[top_i + 1]))
        | ((ang >= bins[(top_i + n_bins // 2) % n_bins]) & (ang < bins[(top_i + n_bins // 2) % n_bins + 1]))
    )
    wp = np.where(wedge, p, 0.0)
    idx = int(np.argmax(wp))
    py, px = divmod(idx, w)
    ddy, ddx = py - cy, px - cx
    import math

    angle = math.degrees(math.atan2(ddy, ddx)) % 180.0
    freq = math.hypot(ddy, ddx)
    spacing = (min(h, w) / freq) if freq > 1e-6 else None

    # Composite score used only for confidence reporting / thresholding
    score = 100.0 * top_share + 2.0 * ang_conc
    return score, top_share, angle, spacing


def route_periodic(img: Image.Image, top_share_thresh: float = 0.32) -> RouterResult:
    """
    If is_periodic, caller should NOT send the crop to the weak-periodic CNN.
    """
    roi, box = _gray_roi(img)
    score, top_share, ang, spacing = _periodicity(roi)

    ink = roi < 200
    dens = float(ink.mean()) if ink.size else 0.0
    # Nearly empty / nearly solid-black → not a useful line-hatch read
    if dens < 0.005 or dens > 0.85:
        top_share *= 0.5
        score *= 0.5

    is_per = top_share >= top_share_thresh
    hint = None
    if is_per:
        if spacing is not None and spacing < 8:
            hint = "STEEL-like"
        elif ang is not None and abs((ang % 90) - 45) < 15:
            hint = "ANSI31-like"
        else:
            hint = "LINE-like"

    return RouterResult(
        is_periodic=bool(is_per),
        score=float(score),
        angle_deg=ang,
        spacing_px=spacing,
        label_hint=hint,
        roi_box=box,
        top_share=float(top_share),
    )
