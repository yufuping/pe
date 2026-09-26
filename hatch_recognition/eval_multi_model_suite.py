"""
Comprehensive multi-model eval on previous test suites:
  - held-out real CAD (17 labeled screenshots)
  - smoke probes (procedural + PAT + ANSI reject + user gravel)
  - user AR-SAND screenshots

No training; read-only checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from aggregate_predict import WeakPeriodicRecognizer
from eval_six_class_smoke import REAL
from finetune_six_class_safe import HELD_OUT, USER_GRAVEL
from pat_parser import parse_pat_file
from renderer import (
    render_ar_conc_aggregate,
    render_ar_sand,
    render_gravel_pebbles,
    render_pattern,
    suggest_scale,
)
from texture_features import extract_hatch_roi
from train_aggregate_specialist import CLASSES
from train_better_backbones import DinoV2Hatch, EffNetB0Hatch
from train_imagenet_six import ImagenetHatchNet
from train_texture_architectures import BilinearCNN, GaborCNN, GaborStemResNet18

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT = ROOT / "models" / "multi_model_suite_eval.json"
USER_SAND = {
    "01a0db7b-e2b1-74c5-b6fd-e3f91198908d.jpg": "AR-SAND",
    "01a0db7b-e2ae-733f-b027-05048ed98f0f.jpg": "AR-SAND",
}
HI = 0.8


def _tf(size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


@torch.no_grad()
def _predict_torch(model: nn.Module, img: Image.Image, size: int, temp: float) -> tuple[str, float]:
    device = next(model.parameters()).device
    rgb = img.convert("RGB")
    box = extract_hatch_roi(np.asarray(rgb.convert("L")))
    x = _tf(size)(rgb.crop(box)).unsqueeze(0).to(device)
    probs = torch.softmax(model(x) / max(temp, 1e-3), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return CLASSES[int(idx)], float(conf)


def _load_models() -> dict[str, dict]:
    device = torch.device("cpu")
    models: dict[str, dict] = {}

    # Specialist via existing recognizer
    sp_path = ROOT / "models" / "aggregate_specialist.pt"
    if sp_path.exists():
        models["specialist_hatchcnn"] = {
            "kind": "specialist",
            "rec": WeakPeriodicRecognizer(sp_path),
            "path": str(sp_path),
        }

    specs = [
        ("resnet18_imagenet_128", ImagenetHatchNet, ROOT / "models" / "aggregate_imagenet_resnet18.pt", 128),
        ("gabor_cnn_128", GaborCNN, ROOT / "models" / "aggregate_gabor_cnn.pt", 128),
        ("bilinear_cnn_128", BilinearCNN, ROOT / "models" / "aggregate_bilinear_cnn.pt", 128),
        ("gabor_stem_resnet18_128", GaborStemResNet18, ROOT / "models" / "aggregate_gabor_stem_resnet18.pt", 128),
        ("effnet_b0_224", EffNetB0Hatch, ROOT / "models" / "aggregate_effnet_b0_224.pt", 224),
        ("dinov2_vits14_224", DinoV2Hatch, ROOT / "models" / "aggregate_dinov2_vits14_224.pt", 224),
    ]
    for name, ctor, path, size in specs:
        if not path.exists():
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = ctor(num_classes=len(CLASSES)).to(device)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models[name] = {
            "kind": "torch",
            "model": m,
            "size": size,
            "temp": float(ck.get("temperature", 0.78)),
            "val_acc": ck.get("val_acc"),
            "path": str(path),
        }
    return models


def _predict_one(entry: dict, img: Image.Image) -> tuple[str, float]:
    if entry["kind"] == "specialist":
        out = entry["rec"].predict(img.convert("RGB"), top_k=6)
        return out["name"], float(out["confidence"])
    return _predict_torch(entry["model"], img, entry["size"], entry["temp"])


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    by_true: dict[str, dict] = {}
    for r in rows:
        t = r["true"]
        d = by_true.setdefault(t, {"n": 0, "ok": 0, "hi80": 0})
        d["n"] += 1
        d["ok"] += int(r["ok"])
        d["hi80"] += int(r["hi80"])
    return {
        "n": n,
        "acc": round(sum(r["ok"] for r in rows) / max(n, 1), 4),
        "hi80": round(sum(r["hi80"] for r in rows) / max(n, 1), 4),
        "by_class": {
            c: {
                "n": by_true[c]["n"],
                "acc": round(by_true[c]["ok"] / max(by_true[c]["n"], 1), 4),
                "hi80": round(by_true[c]["hi80"] / max(by_true[c]["n"], 1), 4),
            }
            for c in sorted(by_true)
        },
        "fails": [
            {"file": r["file"], "true": r["true"], "pred": r["pred"], "conf": r["conf"]}
            for r in rows
            if not r["ok"]
        ],
    }


def _build_cases() -> list[tuple[str, str, str, Image.Image]]:
    """(suite, file_id, true_label, image)"""
    cases: list[tuple[str, str, str, Image.Image]] = []
    rng = np.random.default_rng(42)
    pats = parse_pat_file(ROOT / "acad_4270.pat")

    # Smoke synth / PAT
    probes = [
        ("smoke_synth", "AR-CONC", "AR-CONC", render_ar_conc_aggregate(168, rng, 1.0)),
        ("smoke_synth", "AR-SAND", "AR-SAND", render_ar_sand(168, rng, 1.1)),
        ("smoke_synth", "GRAVEL", "GRAVEL", render_gravel_pebbles(168, rng, 1.2, style="mixed")),
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
        probes.append(("smoke_pat", name, label, img))
    cases.extend(probes)

    # Held-out real CAD (same as historical gate)
    for fname, label in HELD_OUT.items():
        path = ASSETS / fname
        if path.exists():
            cases.append(("heldout_real", fname, label, Image.open(path)))

    # Smoke REAL dict (same files; keep suite tag for smoke_eval parity)
    for fname, label in REAL.items():
        path = ASSETS / fname
        if path.exists():
            cases.append(("smoke_real", fname, label, Image.open(path)))

    if USER_GRAVEL.exists():
        cases.append(("user_gravel", USER_GRAVEL.name, "GRAVEL", Image.open(USER_GRAVEL)))

    for fname, label in USER_SAND.items():
        path = ASSETS / fname
        if path.exists():
            cases.append(("user_sand", fname, label, Image.open(path)))

    # Extra ANSI reject (45°) matching held-out gate
    ansi = render_pattern(
        pats["ANSI31"],
        size=168,
        scale=suggest_scale(pats["ANSI31"], 168),
        rotation=45,
        bg=255,
        fg=0,
        stroke=1,
        shape="rect",
    )
    cases.append(("ansi_reject", "ANSI31_rot45", "OTHER", ansi))
    return cases


def main() -> None:
    models = _load_models()
    cases = _build_cases()
    print(f"models={list(models)} cases={len(cases)}")

    report: dict = {"classes": CLASSES, "models": {}, "ranking": {}}

    for mname, entry in models.items():
        print(f"\n=== {mname} ===")
        by_suite: dict[str, list] = {}
        all_rows = []
        # Dedup heldout vs smoke_real for "heldout_unique" metrics: use heldout_real only for primary
        for suite, fid, label, img in cases:
            pred, conf = _predict_one(entry, img)
            row = {
                "suite": suite,
                "file": fid,
                "true": label,
                "pred": pred,
                "conf": round(conf, 4),
                "ok": pred == label,
                "hi80": pred == label and conf >= HI,
            }
            by_suite.setdefault(suite, []).append(row)
            all_rows.append(row)

        suites_sum = {s: _summarize(rows) for s, rows in by_suite.items()}
        # Primary historical metric: held-out 17 only
        held = _summarize(by_suite.get("heldout_real", []))
        # Combined labeled real (heldout + user sand + user gravel), unique files
        seen = set()
        real_rows = []
        for suite in ("heldout_real", "user_sand", "user_gravel"):
            for r in by_suite.get(suite, []):
                key = (r["file"], r["true"])
                if key in seen:
                    continue
                seen.add(key)
                real_rows.append(r)
        real_all = _summarize(real_rows)

        pack = {
            "checkpoint": entry.get("path"),
            "synth_val_acc": entry.get("val_acc"),
            "heldout_real": held,
            "real_labeled_all": real_all,
            "by_suite": {s: {k: v[k] for k in ("n", "acc", "hi80", "by_class", "fails")} for s, v in suites_sum.items()},
        }
        report["models"][mname] = pack
        print(
            f"  heldout n={held['n']} acc={held['acc']} hi80={held['hi80']} "
            f"| real_all acc={real_all['acc']} hi80={real_all['hi80']}"
        )
        for s in ("smoke_synth", "smoke_pat", "user_sand", "user_gravel", "ansi_reject"):
            if s in suites_sum:
                ss = suites_sum[s]
                print(f"  {s}: n={ss['n']} acc={ss['acc']} hi80={ss['hi80']} fails={ss['fails']}")

    # Rank by held-out hi80 then acc, then real_all
    ranking = sorted(
        report["models"].items(),
        key=lambda kv: (
            kv[1]["heldout_real"]["hi80"],
            kv[1]["heldout_real"]["acc"],
            kv[1]["real_labeled_all"]["hi80"],
            kv[1]["real_labeled_all"]["acc"],
        ),
        reverse=True,
    )
    report["ranking"] = [
        {
            "model": name,
            "heldout_acc": m["heldout_real"]["acc"],
            "heldout_hi80": m["heldout_real"]["hi80"],
            "real_all_acc": m["real_labeled_all"]["acc"],
            "real_all_hi80": m["real_labeled_all"]["hi80"],
            "synth_val_acc": m.get("synth_val_acc"),
        }
        for name, m in ranking
    ]
    print("\n=== RANKING (heldout hi80 → acc → real_all) ===")
    for r in report["ranking"]:
        print(
            f"  {r['model']:28s} held={r['heldout_acc']:.3f}/{r['heldout_hi80']:.3f} "
            f"real_all={r['real_all_acc']:.3f}/{r['real_all_hi80']:.3f} "
            f"synth_val={r['synth_val_acc']}"
        )

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
