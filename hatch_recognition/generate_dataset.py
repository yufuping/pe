"""Generate synthetic training images from .pat hatch definitions."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pat_parser import COMMON_PATTERNS, parse_pat_file
from renderer import add_interference, render_pattern, suggest_scale


def generate_dataset(
    pat_path: str | Path,
    out_dir: str | Path,
    patterns: list[str] | None = None,
    samples_per_class: int = 120,
    size: int = 128,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.1,
) -> dict:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    all_patterns = parse_pat_file(pat_path)
    names = patterns or COMMON_PATTERNS
    missing = [n for n in names if n not in all_patterns]
    if missing:
        raise KeyError(f"Patterns not found in PAT file: {missing}")

    out = Path(out_dir)
    for split in ("train", "val", "test"):
        for name in names:
            (out / split / name).mkdir(parents=True, exist_ok=True)

    shapes = ["rect", "rect", "rect", "circle", "polygon", "irregular"]
    class_to_idx = {n: i for i, n in enumerate(names)}
    meta = {
        "classes": names,
        "class_to_idx": class_to_idx,
        "descriptions": {n: all_patterns[n].description for n in names},
        "samples_per_class": samples_per_class,
        "image_size": size,
    }

    counts = {s: {n: 0 for n in names} for s in ("train", "val", "test")}

    for name in names:
        pattern = all_patterns[name]
        for i in tqdm(range(samples_per_class), desc=name):
            base = suggest_scale(pattern, size=size, target_px=float(rng.uniform(7.0, 16.0)))
            # Jitter scale around the suggested value for multi-scale training.
            scale = float(base * rng.uniform(0.55, 1.7))

            rotation = float(rng.uniform(0, 360))
            # Small random origin shift so tile phase varies
            offset = (
                size / 2 + float(rng.uniform(-scale * 2, scale * 2)),
                size / 2 + float(rng.uniform(-scale * 2, scale * 2)),
            )
            shape = shapes[int(rng.integers(0, len(shapes)))]
            stroke = int(rng.choice([1, 1, 1, 2]))
            # Drawing paper: usually white bg / black lines; sometimes dark scan
            if rng.random() < 0.08:
                bg, fg = 30, 220
            else:
                bg, fg = 255, 0

            img = render_pattern(
                pattern,
                size=size,
                scale=scale,
                rotation=rotation,
                offset=offset,
                bg=bg,
                fg=fg,
                stroke=stroke,
                shape=shape,
            )
            img = add_interference(
                img,
                rng=rng,
                noise=float(rng.uniform(0.01, 0.08)),
                lines=True,
                blur=True,
                invert_chance=0.03,
            )

            r = rng.random()
            if r < test_ratio:
                split = "test"
            elif r < test_ratio + val_ratio:
                split = "val"
            else:
                split = "train"

            fname = f"{name}_{i:04d}_s{scale:.1f}_r{rotation:.0f}_{shape}.png"
            img.save(out / split / name / fname)
            counts[split][name] += 1

    meta["counts"] = counts
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate hatch pattern training dataset")
    p.add_argument("--pat", default="acad_4270.pat")
    p.add_argument("--out", default="data/dataset")
    p.add_argument("--samples", type=int, default=120)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = generate_dataset(args.pat, args.out, samples_per_class=args.samples, size=args.size, seed=args.seed)
    print(json.dumps(meta["counts"], indent=2, ensure_ascii=False))
    print(f"Classes: {meta['classes']}")
