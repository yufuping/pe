"""
Classical texture cues for weak-periodic fills (GRAVEL vs AR-CONC).

GRAVEL: packed closed pebble/cobble outlines, little sand stipple.
AR-CONC: dense stipple dots + sparse hollow triangles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class AggregateTexture:
    ink_density: float
    stipple_score: float
    closed_loop_score: float
    triangle_score: float
    line_period_score: float
    gravel_score: float
    concrete_score: float
    is_aggregate: bool
    roi_box: tuple[int, int, int, int]  # x0,y0,x1,y1


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


def ink_mask(gray: np.ndarray, hi: int = 200) -> np.ndarray:
    """
    Dark ink pixels. Include pure black (0) — binary CAD exports are 0/255.
    Previously `g > 5` dropped all ink on such images and broke dens/TTA.
    """
    g = np.asarray(gray)
    return g < hi


def extract_hatch_roi(gray: np.ndarray, pad: float = 0.04) -> tuple[int, int, int, int]:
    """Tight bbox around dark ink (ignore sparse annotations)."""
    h, w = gray.shape
    # Include soft anti-aliased ink and pure-black binary strokes.
    ink = ink_mask(gray, hi=205)
    ys, xs = np.where(ink)
    if len(xs) < 50:
        return 0, 0, w, h

    dens = float(ink.mean())
    # Sparse fills: grow from densest window until most ink is covered.
    if dens < 0.07 and min(h, w) >= 96:
        win = max(64, min(h, w) // 4)
        step = max(8, win // 4)
        best = (-1.0, 0, 0, w, h)
        for y0 in range(0, max(1, h - win + 1), step):
            for x0 in range(0, max(1, w - win + 1), step):
                d = float(ink[y0 : y0 + win, x0 : x0 + win].mean())
                if d > best[0]:
                    best = (d, x0, y0, x0 + win, y0 + win)
        _, x0, y0, x1, y1 = best
        target = 0.85 * dens * h * w  # cover most ink mass
        for _ in range(24):
            covered = float(ink[y0:y1, x0:x1].sum())
            if covered >= target:
                break
            grown = False
            # Expand toward the side that adds the most ink
            cands = []
            if x0 > 0:
                cands.append(("l", float(ink[y0:y1, max(0, x0 - step) : x0].sum())))
            if x1 < w:
                cands.append(("r", float(ink[y0:y1, x1 : min(w, x1 + step)].sum())))
            if y0 > 0:
                cands.append(("u", float(ink[max(0, y0 - step) : y0, x0:x1].sum())))
            if y1 < h:
                cands.append(("d", float(ink[y1 : min(h, y1 + step), x0:x1].sum())))
            if not cands:
                break
            cands.sort(key=lambda t: t[1], reverse=True)
            side, add = cands[0]
            if add < 1:
                break
            if side == "l":
                x0 = max(0, x0 - step)
            elif side == "r":
                x1 = min(w, x1 + step)
            elif side == "u":
                y0 = max(0, y0 - step)
            else:
                y1 = min(h, y1 + step)
            grown = True
            if not grown:
                break
        px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
        return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)

    # Keep central mass: discard outlier ink far from median
    mx, my = float(np.median(xs)), float(np.median(ys))
    dist = np.hypot(xs - mx, ys - my)
    keep = dist < np.percentile(dist, 94)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 30:
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _stipple_score(binary: np.ndarray) -> float:
    """Fraction of ink that is isolated 1–3px blobs (sand dots)."""
    from scipy import ndimage  # optional; fallback below

    try:
        labeled, n = ndimage.label(binary)
        if n == 0:
            return 0.0
        sizes = ndimage.sum(binary, labeled, index=np.arange(1, n + 1))
        tiny = float((sizes <= 4).sum())
        return tiny / max(n, 1)
    except Exception:
        # Fallback without scipy: local peakiness
        ink = binary.astype(np.float32)
        if ink.sum() < 10:
            return 0.0
        # Count pixels with few neighbors
        pad = np.pad(ink, 1)
        nb = (
            pad[0:-2, 0:-2]
            + pad[0:-2, 1:-1]
            + pad[0:-2, 2:]
            + pad[1:-1, 0:-2]
            + pad[1:-1, 2:]
            + pad[2:, 0:-2]
            + pad[2:, 1:-1]
            + pad[2:, 2:]
        )
        lone = ((ink > 0) & (nb <= 2)).mean()
        return float(lone)


def _closed_loop_score(gray: np.ndarray) -> float:
    """
    Estimate packed closed outlines: edge pixels that form ring-like structures.
    High for GRAVEL, lower for stipple fields.
    """
    # Simple edge via gradient
    g = gray.astype(np.float32)
    gx = np.abs(g[:, 1:] - g[:, :-1])
    gy = np.abs(g[1:, :] - g[:-1, :])
    edge = np.zeros_like(g)
    edge[:, 1:] = np.maximum(edge[:, 1:], gx)
    edge[1:, :] = np.maximum(edge[1:, :], gy)
    ebin = edge > 28
    dens = float(ebin.mean())
    if dens < 0.01:
        return 0.0
    # Local circularity proxy: for each edge pixel, check if opposite side also edged
    # Sample grid of centers and look for ring signatures
    h, w = gray.shape
    hits = 0
    trials = 0
    step = max(4, min(h, w) // 28)
    radii = [max(3, min(h, w) // 40), max(5, min(h, w) // 28), max(7, min(h, w) // 20)]
    for cy in range(step, h - step, step):
        for cx in range(step, w - step, step):
            # Prefer centers in white-ish interior (inside pebble)
            if gray[cy, cx] < 180:
                continue
            trials += 1
            ring = 0
            for r in radii:
                cnt = 0
                for a in range(0, 360, 30):
                    rad = math.radians(a)
                    x = int(cx + r * math.cos(rad))
                    y = int(cy + r * math.sin(rad))
                    if 0 <= x < w and 0 <= y < h and ebin[y, x]:
                        cnt += 1
                if cnt >= 7:
                    ring += 1
            if ring >= 1:
                hits += 1
    if trials == 0:
        return dens * 2.0
    return float(hits / trials) * (0.5 + dens)


def _triangle_score(gray: np.ndarray) -> float:
    """
    Rough detector for small hollow triangles (AR-CONC aggregate shards).
    Uses corner-rich closed contours of moderate size.
    """
    try:
        from scipy import ndimage
    except Exception:
        return 0.0

    # Edges
    g = gray.astype(np.float32)
    blur = ndimage.gaussian_filter(g, 0.6)
    edge = np.abs(ndimage.sobel(blur))
    ebin = edge > np.percentile(edge, 88)
    # Invert: find bright holes enclosed by dark/edge — use ink outlines
    ink = gray < 160
    # Dilate slightly and look for small cavities? Simpler: connected components of edge
    labeled, n = ndimage.label(ebin)
    if n == 0:
        return 0.0
    h, w = gray.shape
    area = h * w
    good = 0
    checked = 0
    for i in range(1, min(n + 1, 800)):
        ys, xs = np.where(labeled == i)
        if len(xs) < 12 or len(xs) > area * 0.02:
            continue
        checked += 1
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw < 4 or bh < 4 or max(bw, bh) > min(h, w) * 0.15:
            continue
        # Compactness and triangularity via extent
        extent = len(xs) / max(bw * bh, 1)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if 0.25 < extent < 0.75 and aspect < 2.2:
            # Hollow: interior mean brighter than edge
            cy, cx = int(ys.mean()), int(xs.mean())
            if gray[cy, cx] > 180:
                good += 1
    if checked == 0:
        return 0.0
    return float(good) / max(checked, 1) * 5.0  # scale up sparse hits


def _line_period_score(gray: np.ndarray) -> float:
    """High if dominant parallel-line periodicity (ANSI/LINE) — reject from aggregate path."""
    g = gray.astype(np.float32)
    g = g - g.mean()
    # Autocorr on mid row/col projections
    row = g.mean(axis=0)
    col = g.mean(axis=1)

    def peak_ratio(sig: np.ndarray) -> float:
        if len(sig) < 32:
            return 0.0
        f = np.fft.rfft(sig)
        p = np.abs(f) ** 2
        p[0] = 0
        if p.sum() < 1e-6:
            return 0.0
        # Ignore very low freq
        lo = max(2, len(p) // 40)
        band = p[lo:]
        if band.size == 0:
            return 0.0
        return float(band.max() / (band.mean() + 1e-8))

    return max(peak_ratio(row), peak_ratio(col)) / 20.0  # normalize roughly to 0-1+


def analyze_aggregate(img: Image.Image) -> AggregateTexture:
    gray_full = _to_gray(img)
    box = extract_hatch_roi(gray_full)
    x0, y0, x1, y1 = box
    gray = gray_full[y0:y1, x0:x1]
    # Light denoise
    patch = Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=3))
    gray = np.asarray(patch, dtype=np.uint8)

    ink = ink_mask(gray, hi=200)
    dens = float(ink.mean())
    stipple = _stipple_score(ink)
    loops = _closed_loop_score(gray)
    tris = _triangle_score(gray)
    lines = _line_period_score(gray)

    # Heuristic combination (tuned to synthetic looks; fusion layer calibrates)
    gravel = 1.6 * loops + 0.2 * dens - 0.9 * stipple - 0.4 * tris - 0.8 * lines
    concrete = 1.2 * stipple + 0.9 * tris + 0.15 * dens - 0.9 * loops - 0.8 * lines

    is_agg = dens > 0.03 and lines < 0.85 and (loops > 0.02 or stipple > 0.15 or tris > 0.05)

    return AggregateTexture(
        ink_density=dens,
        stipple_score=float(stipple),
        closed_loop_score=float(loops),
        triangle_score=float(tris),
        line_period_score=float(lines),
        gravel_score=float(gravel),
        concrete_score=float(concrete),
        is_aggregate=bool(is_agg),
        roi_box=box,
    )
