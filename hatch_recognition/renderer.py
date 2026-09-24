"""Render AutoCAD hatch patterns to images."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pat_parser import HatchLine, HatchPattern


def _unit(angle_deg: float) -> tuple[float, float, float, float]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    # direction along line, perpendicular (CCW)
    return c, s, -s, c


def typical_spacing(pattern: HatchPattern) -> float:
    """Estimate characteristic spacing in pattern units (use dense-family min)."""
    vals = []
    for line in pattern.lines:
        span = math.hypot(line.delta_x, line.delta_y)
        if span > 1e-6:
            vals.append(span)
    if not vals:
        return 0.25
    vals.sort()
    # Prefer the densest families so overlapping patterns don't paint solid black.
    return vals[0]


def suggest_scale(pattern: HatchPattern, size: int = 128, target_px: float = 10.0) -> float:
    """Choose a scale so densest family spacing is about target_px pixels."""
    if pattern.name == "SOLID":
        return 1.0
    spacing = typical_spacing(pattern)
    scale = target_px / max(spacing, 1e-6)
    # Patterns with many overlapping families need extra zoom-out (larger scale).
    n = len(pattern.lines)
    if n >= 8:
        scale *= 1.0 + 0.08 * (n - 4)
    return max(scale, 0.5)


def _draw_dashed_ray(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    direction: tuple[float, float],
    dashes: Sequence[float],
    scale: float,
    half_len: float,
    color: int,
    width: int,
) -> None:
    ux, uy = direction
    ox, oy = origin

    if not dashes:
        draw.line(
            [(ox - ux * half_len, oy - uy * half_len), (ox + ux * half_len, oy + uy * half_len)],
            fill=color,
            width=width,
        )
        return

    scaled = []
    for d in dashes:
        if abs(d) < 1e-12:
            scaled.append((0.0, True))  # dot
        else:
            scaled.append((abs(d) * scale, d > 0))

    period = sum(seg for seg, _ in scaled)
    if period < 1e-9:
        draw.point([origin], fill=color)
        return

    # Parameter t along the infinite line; cover [-half_len, half_len].
    # Align dash phase to world projection so parallel lines share phase correctly.
    world_phase = (ox * ux + oy * uy) % period
    t_start = -half_len
    # Absolute parameter of segment start on infinite line:
    abs_start = (ox * ux + oy * uy) + t_start
    # Rewind to beginning of dash cycle before t_start
    dist_into = abs_start % period
    if dist_into < 0:
        dist_into += period

    # Find position in pattern
    acc = 0.0
    idx = 0
    skip = 0.0
    for i, (seg, _) in enumerate(scaled):
        if dist_into <= acc + seg + 1e-9:
            idx = i
            skip = dist_into - acc
            break
        acc += seg

    t = t_start
    first = True
    guard = 0
    while t < half_len and guard < 100000:
        guard += 1
        seg, pen = scaled[idx % len(scaled)]
        use = seg - skip if first else seg
        first = False
        skip = 0.0
        if use < 1e-9 and pen and scaled[idx % len(scaled)][0] == 0.0:
            px, py = ox + ux * t, oy + uy * t
            r = max(1, width)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
            t += 0.75
            idx += 1
            continue
        use = max(use, 0.0)
        draw_to = min(t + use, half_len)
        if pen and draw_to > t:
            x0, y0 = ox + ux * t, oy + uy * t
            x1, y1 = ox + ux * draw_to, oy + uy * draw_to
            if abs(draw_to - t) < 1.25:
                # short dash / dot
                r = max(1, width)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                draw.ellipse([mx - r, my - r, mx + r, my + r], fill=color)
            else:
                draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
        t = draw_to if use > 1e-9 else t + 0.5
        idx += 1


def _render_line_family(
    draw: ImageDraw.ImageDraw,
    line: HatchLine,
    size: int,
    scale: float,
    rotation: float,
    offset_x: float,
    offset_y: float,
    color: int,
    stroke: int,
) -> None:
    ux, uy, px, py = _unit(line.angle + rotation)
    dx = line.delta_x * scale
    dy = line.delta_y * scale

    # Successive line origins: n * (delta_x * dir + delta_y * perp)
    sx = dx * ux + dy * px
    sy = dx * uy + dy * py
    shift = math.hypot(sx, sy)
    if shift < 1e-6:
        # Fallback dense parallel family
        sx, sy = px * max(scale * 0.125, 2.0), py * max(scale * 0.125, 2.0)
        shift = math.hypot(sx, sy)

    ox0 = line.x_origin * scale + offset_x
    oy0 = line.y_origin * scale + offset_y

    half = math.hypot(size, size) * 1.5
    n = int(half * 2 / shift) + 3
    # Limit work for extremely dense definitions
    n = min(n, 400)

    for i in range(-n, n + 1):
        ox = ox0 + i * sx
        oy = oy0 + i * sy
        # Skip families whose nearest point is far outside the image
        # distance from image center to the line
        cx, cy = size / 2, size / 2
        # point on line closest to center
        dist = abs((cx - ox) * (-uy) + (cy - oy) * ux)  # using perp of dir... 
        # Actually distance from point to line: |(P-O) ¡Á dir|
        dist = abs((cx - ox) * uy - (cy - oy) * ux)
        if dist > half:
            continue
        _draw_dashed_ray(
            draw,
            (ox, oy),
            (ux, uy),
            line.dashes,
            scale,
            half,
            color,
            stroke,
        )


def render_pattern(
    pattern: HatchPattern,
    size: int = 224,
    scale: float | None = None,
    rotation: float = 0.0,
    offset: tuple[float, float] | None = None,
    bg: int = 255,
    fg: int = 0,
    stroke: int = 1,
    shape: str = "rect",
) -> Image.Image:
    if pattern.name == "SOLID":
        img = Image.new("L", (size, size), fg)
        return _apply_shape_mask(img, shape, bg)

    if scale is None:
        scale = suggest_scale(pattern, size=size)

    img = Image.new("L", (size, size), bg)
    draw = ImageDraw.Draw(img)
    if offset is None:
        offset = (size / 2.0, size / 2.0)

    for line in pattern.lines:
        _render_line_family(
            draw,
            line,
            size,
            scale=scale,
            rotation=rotation,
            offset_x=offset[0],
            offset_y=offset[1],
            color=fg,
            stroke=stroke,
        )

    return _apply_shape_mask(img, shape, bg)


def _apply_shape_mask(img: Image.Image, shape: str, bg: int) -> Image.Image:
    if shape == "rect":
        return img

    size = img.size[0]
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    margin = size * 0.08

    if shape == "circle":
        mdraw.ellipse([margin, margin, size - margin, size - margin], fill=255)
    elif shape == "polygon":
        cx, cy, r = size / 2, size / 2, size * 0.42
        pts = [
            (cx + r * math.cos(-math.pi / 2 + i * 2 * math.pi / 5),
             cy + r * math.sin(-math.pi / 2 + i * 2 * math.pi / 5))
            for i in range(5)
        ]
        mdraw.polygon(pts, fill=255)
    else:
        rng = np.random.default_rng()
        cx, cy = size / 2, size / 2
        n = int(rng.integers(6, 10))
        pts = []
        for i in range(n):
            a = i * 2 * math.pi / n + float(rng.uniform(-0.15, 0.15))
            r = size * float(rng.uniform(0.28, 0.46))
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        mdraw.polygon(pts, fill=255)

    out = Image.new("L", (size, size), bg)
    out.paste(img, mask=mask)
    return out


def add_interference(
    img: Image.Image,
    rng: np.random.Generator | None = None,
    noise: float = 0.05,
    lines: bool = True,
    blur: bool = True,
    invert_chance: float = 0.05,
) -> Image.Image:
    rng = rng or np.random.default_rng()
    arr = np.array(img, dtype=np.float32)

    if noise > 0:
        arr = arr + rng.normal(0, noise * 255, arr.shape)
        if rng.random() < 0.4:
            n_sp = int(arr.size * 0.004)
            ys = rng.integers(0, arr.shape[0], n_sp)
            xs = rng.integers(0, arr.shape[1], n_sp)
            arr[ys, xs] = rng.choice([0, 255], size=n_sp)

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(out)
    w, h = out.size

    if lines and rng.random() < 0.65:
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() < 0.5:
                x = int(rng.integers(0, w))
                draw.line([(x, 0), (x, h)], fill=0, width=int(rng.integers(1, 2)))
            else:
                y = int(rng.integers(0, h))
                draw.line([(0, y), (w, y)], fill=0, width=int(rng.integers(1, 2)))
        for _ in range(int(rng.integers(0, 4))):
            x0 = int(rng.integers(0, w))
            y0 = int(rng.integers(0, h))
            x1 = x0 + int(rng.integers(-35, 35))
            y1 = y0 + int(rng.integers(-35, 35))
            draw.line([(x0, y0), (x1, y1)], fill=0, width=1)

    if blur and rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.9))))

    if rng.random() < invert_chance:
        out = Image.fromarray(255 - np.array(out), mode="L")

    if rng.random() < 0.45:
        a = np.array(out, dtype=np.float32)
        contrast = float(rng.uniform(0.8, 1.2))
        brightness = float(rng.uniform(-15, 15))
        a = (a - 127.5) * contrast + 127.5 + brightness
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")

    return out
