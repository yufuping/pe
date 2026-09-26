"""Quick smoke eval for six-class specialist: synth PAT tiles + held-out assets + ANSI reject."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aggregate_predict import WeakPeriodicRecognizer
from pat_parser import parse_pat_file
from renderer import (
    render_ar_conc_aggregate,
    render_ar_sand,
    render_gravel_pebbles,
    render_pattern,
    suggest_scale,
)

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
MODEL = ROOT / "models" / "aggregate_specialist.pt"
OUT = ROOT / "models" / "six_class_smoke_eval.json"

REAL = {
    "01a0d261-8064-7965-bbcb-9be9d02e1e5c.jpg": "AR-CONC",
    "01a0d262-d1c9-742e-9e6e-26968ab74972.jpg": "AR-CONC",
    "01a0d263-d93e-7705-873f-c3548132d977.jpg": "AR-CONC",
    "01a0d264-a8fa-7635-a753-8ac1bd36e145.jpg": "AR-CONC",
    "01a0d265-ce06-7d31-a510-70df7088e275.jpg": "AR-CONC",
    "01a0d379-4779-708f-bc09-f6f27b708e8f.jpg": "AR-CONC",
    "CE4D9686-1522-4EDE-A013-9618C6A66357_L0_001.jpg": "AR-CONC",
    "74CE6AD5-8D6B-4DAE-87CD-1593E8AF4A1C_L0_001.jpg": "AR-CONC",
    "01a0d27e-806d-7840-9018-9a389ae6733b.jpg": "GRAVEL",
    "01a0d281-37df-7874-a0f8-445326aa781f.jpg": "GRAVEL",
    "01a0d2af-a2a6-7beb-9388-c0936f5ad85a.jpg": "GRAVEL",
    "01a0d2b4-45f1-7a20-a74b-4ea28e36ccc8.jpg": "GRAVEL",
    "01a0d377-c779-774f-981b-7b04deb8cef1.jpg": "GRAVEL",
    "01a0d37c-8b58-7e0a-9441-b1da57b885f5.jpg": "GRAVEL",
    "01a0d380-566d-77d8-b43f-b9539db33eb0.jpg": "GRAVEL",
    "01a0d383-62b2-7a63-a041-d12e7df2eeaf.jpg": "GRAVEL",
    "47331B81-87A0-489D-BAE1-EE211207F0C1_L0_001.jpg": "GRAVEL",
}


def _pred(rec: WeakPeriodicRecognizer, img: Image.Image) -> tuple[str, float]:
    out = rec.predict(img.convert("RGB"), top_k=6)
    return out["name"], float(out["confidence"])


def main() -> None:
    rec = WeakPeriodicRecognizer(MODEL)
    assert rec.classes == ["AR-CONC", "AR-SAND", "DOLMIT", "EARTH", "GRAVEL", "OTHER"], rec.classes
    rng = np.random.default_rng(42)
    pats = parse_pat_file(ROOT / "acad_4270.pat")
    rows = []

    # Procedural / PAT synth probes
    probes = [
        ("synth", "AR-CONC", render_ar_conc_aggregate(168, rng, 1.0)),
        ("synth", "AR-SAND", render_ar_sand(168, rng, 1.1)),
        ("synth", "GRAVEL", render_gravel_pebbles(168, rng, 1.2, style="mixed")),
    ]
    for name in ("DOLMIT", "EARTH", "AR-SAND", "ANSI31"):
        pat = pats[name]
        img = render_pattern(
            pat,
            size=168,
            scale=suggest_scale(pat, 168),
            rotation=15,
            bg=255,
            fg=0,
            stroke=1,
            shape="rect",
        )
        label = "OTHER" if name == "ANSI31" else name
        probes.append(("pat", label, img))

    for kind, label, img in probes:
        pred, conf = _pred(rec, img)
        rows.append(
            {
                "source": kind,
                "file": label,
                "true": label,
                "pred": pred,
                "conf": round(conf, 4),
                "ok": pred == label,
                "hi80": pred == label and conf >= 0.8,
            }
        )

    # Held-out real AR-CONC / GRAVEL
    for fname, label in REAL.items():
        path = ASSETS / fname
        if not path.exists():
            continue
        pred, conf = _pred(rec, Image.open(path))
        rows.append(
            {
                "source": "real",
                "file": fname,
                "true": label,
                "pred": pred,
                "conf": round(conf, 4),
                "ok": pred == label,
                "hi80": pred == label and conf >= 0.8,
            }
        )

    # User gravel crop if present
    ug = ROOT / "samples" / "preview" / "USER_GRAVEL_crop.png"
    if ug.exists():
        pred, conf = _pred(rec, Image.open(ug))
        rows.append(
            {
                "source": "user",
                "file": ug.name,
                "true": "GRAVEL",
                "pred": pred,
                "conf": round(conf, 4),
                "ok": pred == "GRAVEL",
                "hi80": pred == "GRAVEL" and conf >= 0.8,
            }
        )

    summary = {
        "classes": rec.classes,
        "n": len(rows),
        "acc": round(sum(r["ok"] for r in rows) / max(len(rows), 1), 4),
        "hi80": round(sum(r["hi80"] for r in rows) / max(len(rows), 1), 4),
        "by_source": {},
        "rows": rows,
    }
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        summary["by_source"][src] = {
            "n": len(sub),
            "acc": round(sum(r["ok"] for r in sub) / len(sub), 4),
            "hi80": round(sum(r["hi80"] for r in sub) / len(sub), 4),
        }

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("classes", "n", "acc", "hi80", "by_source")}, indent=2))
    fails = [r for r in rows if not r["ok"] or not r["hi80"]]
    print(f"fails_or_lo_conf={len(fails)}")
    for r in fails[:20]:
        print(f"  {r['source']} {r['file'][:40]} true={r['true']} pred={r['pred']} conf={r['conf']}")


if __name__ == "__main__":
    main()
