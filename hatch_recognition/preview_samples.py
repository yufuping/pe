"""Quick visual check: render one sample per common pattern."""

from pathlib import Path

from pat_parser import COMMON_PATTERNS, parse_pat_file
from renderer import add_interference, render_pattern, suggest_scale

ROOT = Path(__file__).resolve().parent
out = ROOT / "samples" / "preview"
out.mkdir(parents=True, exist_ok=True)

patterns = parse_pat_file(ROOT / "acad_4270.pat")

for name in COMMON_PATTERNS:
    p = patterns[name]
    scale = suggest_scale(p, size=256, target_px=12.0)
    img = render_pattern(p, size=256, scale=scale, rotation=0, shape="rect")
    img.save(out / f"{name}_clean.png")
    noisy = add_interference(img)
    noisy.save(out / f"{name}_noisy.png")
    print(f"wrote {name}: {p.description} (scale={scale:.2f})")

print(f"Preview images in {out}")
