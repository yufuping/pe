"""Parse AutoCAD .pat hatch pattern files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HatchLine:
    angle: float
    x_origin: float
    y_origin: float
    delta_x: float
    delta_y: float
    dashes: list[float] = field(default_factory=list)


@dataclass
class HatchPattern:
    name: str
    description: str
    lines: list[HatchLine] = field(default_factory=list)


def parse_pat_file(path: str | Path) -> dict[str, HatchPattern]:
    """Parse a .pat file into a name -> HatchPattern mapping."""
    patterns: dict[str, HatchPattern] = {}
    current: HatchPattern | None = None

    raw = Path(path).read_bytes()
    for enc in ("gbk", "gb18030", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin-1", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue

        if line.startswith("*"):
            body = line[1:]
            if "," in body:
                name, desc = body.split(",", 1)
            else:
                name, desc = body, ""
            name = name.strip().upper()
            current = HatchPattern(name=name, description=desc.strip())
            patterns[name] = current
            continue

        if current is None:
            continue

        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 5:
            continue
        values = [float(p) for p in parts]
        current.lines.append(
            HatchLine(
                angle=values[0],
                x_origin=values[1],
                y_origin=values[2],
                delta_x=values[3],
                delta_y=values[4],
                dashes=values[5:],
            )
        )

    return patterns


# 10 most common CAD hatches, including concrete (AR-CONC).
COMMON_PATTERNS = [
    "ANSI31",   # 铁、砖和石 — most common diagonal hatch
    "ANSI32",   # 钢
    "AR-CONC",  # 混凝土
    "BRICK",    # 砖
    "STEEL",    # 钢材质
    "EARTH",    # 地面
    "LINE",     # 平行水平线
    "NET",      # 栅格
    "GRAVEL",   # 沙砾
    "SOLID",    # 实体填充
]
