"""FastAPI web app: upload a hatch image and get recognition results."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from aggregate_predict import WeakPeriodicRecognizer
from predict import HatchRecognizer, load_descriptions

ROOT = Path(__file__).resolve().parent
SPECIALIST_PATH = ROOT / "models" / "aggregate_specialist.pt"
LEGACY_PATH = ROOT / "models" / "best_model.pt"
META_PATH = ROOT / "data" / "dataset" / "meta.json"

SPECIALIST_DESC = {
    "AR-CONC": "混凝土：砂点 + 稀疏空心三角（弱周期）",
    "AR-SAND": "砂土：纯砂点/细粒点阵（弱周期，无三角）",
    "DOLMIT": "白云石：斜向短划线簇（PAT 材质填充）",
    "EARTH": "土壤/地面：交叉短线纹理（PAT 材质填充）",
    "GRAVEL": "砾石：密闭卵石/碎石轮廓（弱周期）",
    "OTHER": "非目标填充（ANSI/LINE/NET/砖钢等，由 CNN 拒识）",
}

app = FastAPI(title="CAD Hatch Pattern Recognizer", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
recognizer: WeakPeriodicRecognizer | HatchRecognizer | None = None
recognizer_mode: str = "none"

if (ROOT / "static").exists():
    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def get_recognizer():
    global recognizer, recognizer_mode
    if recognizer is None:
        if SPECIALIST_PATH.exists():
            recognizer = WeakPeriodicRecognizer(SPECIALIST_PATH)
            recognizer_mode = "specialist"
        elif LEGACY_PATH.exists():
            recognizer = HatchRecognizer(LEGACY_PATH, load_descriptions(META_PATH))
            recognizer_mode = "legacy"
        else:
            raise FileNotFoundError(
                f"No model found. Expected {SPECIALIST_PATH} or {LEGACY_PATH}."
            )
    return recognizer


def _descriptions(rec) -> dict[str, str]:
    if recognizer_mode == "specialist":
        return {c: SPECIALIST_DESC.get(c, "") for c in rec.classes}
    return {c: rec.descriptions.get(c, "") for c in rec.classes}


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>CAD 填充图案识别</title>
  <style>
    :root {
      --ink: #1a2332;
      --muted: #5a6a7e;
      --line: #c8d4e3;
      --paper: #eef2f7;
      --accent: #0b5f8a;
      --accent-2: #1a7a4c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Noto Sans SC", "PingFang SC", sans-serif;
      background:
        radial-gradient(ellipse 80% 50% at 10% -10%, #d5e6f2 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 100% 0%, #dce8de 0%, transparent 50%),
        linear-gradient(165deg, #f4f7fb 0%, #e8eef5 100%);
    }
    main { max-width: 720px; margin: 0 auto; padding: 40px 20px 60px; }
    h1 {
      font-family: "IBM Plex Serif", "Noto Serif SC", Georgia, serif;
      font-size: clamp(1.8rem, 4vw, 2.4rem);
      font-weight: 650;
      letter-spacing: -0.02em;
      margin: 0 0 8px;
      line-height: 1.2;
    }
    .sub { color: var(--muted); margin: 0 0 28px; font-size: 1.05rem; }
    .panel {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
      backdrop-filter: blur(8px);
      padding: 22px;
    }
    .drop {
      border: 1.5px dashed #8aa0b8;
      min-height: 220px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 10px;
      cursor: pointer;
      background: #f8fafc;
      transition: border-color .2s, background .2s;
      position: relative;
      overflow: hidden;
    }
    .drop:hover, .drop.drag { border-color: var(--accent); background: #eef6fb; }
    .drop img {
      max-width: 100%;
      max-height: 280px;
      object-fit: contain;
      display: none;
    }
    .drop.has-image .hint { display: none; }
    .drop.has-image img { display: block; }
    .hint { color: var(--muted); text-align: center; padding: 16px; }
    .hint strong { color: var(--ink); display: block; margin-bottom: 6px; }
    input[type=file] { display: none; }
    button {
      margin-top: 16px;
      width: 100%;
      border: 0;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 650;
      padding: 14px;
      cursor: pointer;
      letter-spacing: 0.02em;
    }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button:hover:not(:disabled) { background: #094d70; }
    .results { margin-top: 22px; display: none; }
    .results.show { display: block; }
    .results h2 {
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin: 0 0 12px;
      font-weight: 600;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: baseline;
      padding: 12px 0;
      border-bottom: 1px solid var(--line);
    }
    .row:first-of-type .name { color: var(--accent-2); font-size: 1.25rem; }
    .name { font-weight: 700; font-variant-numeric: tabular-nums; }
    .desc { color: var(--muted); font-size: 0.92rem; grid-column: 1 / -1; margin-top: -4px; }
    .pct { font-weight: 700; font-variant-numeric: tabular-nums; }
    .bar {
      grid-column: 1 / -1;
      height: 4px;
      background: #dce4ee;
      margin-top: 4px;
    }
    .bar > i {
      display: block;
      height: 100%;
      background: var(--accent);
      width: 0;
      transition: width .4s ease;
    }
    .err { color: #a12626; margin-top: 12px; min-height: 1.2em; }
    .classes {
      margin-top: 28px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.7;
    }
    .classes code {
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--line);
      padding: 1px 6px;
      font-size: 0.85em;
    }
  </style>
</head>
<body>
<main>
  <h1>CAD 填充图案识别</h1>
  <p class="sub">材质填充专家：AR-CONC / AR-SAND / DOLMIT / EARTH / GRAVEL + OTHER（合成训练，无程序路由改写）</p>

  <section class="panel">
    <label class="drop" id="drop" for="file">
      <div class="hint">
        <strong>点击或拖拽图片到此处</strong>
        支持 PNG / JPG / BMP / WEBP
      </div>
      <img id="preview" alt="预览">
      <input id="file" type="file" accept="image/*">
    </label>
    <button id="run" type="button" disabled>识别填充</button>
    <div class="err" id="err"></div>
    <div class="results" id="results">
      <h2>识别结果</h2>
      <div id="rows"></div>
    </div>
  </section>

  <p class="classes" id="classes">支持类别加载中…</p>
</main>
<script>
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  const preview = document.getElementById('preview');
  const run = document.getElementById('run');
  const err = document.getElementById('err');
  const results = document.getElementById('results');
  const rows = document.getElementById('rows');
  let blob = null;

  function setFile(f) {
    if (!f) return;
    blob = f;
    preview.src = URL.createObjectURL(f);
    drop.classList.add('has-image');
    run.disabled = false;
    err.textContent = '';
    results.classList.remove('show');
  }

  file.addEventListener('change', e => setFile(e.target.files[0]));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag');
    setFile(e.dataTransfer.files[0]);
  });

  run.addEventListener('click', async () => {
    if (!blob) return;
    run.disabled = true;
    err.textContent = '';
    const fd = new FormData();
    fd.append('file', blob);
    try {
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '识别失败');
      rows.innerHTML = data.predictions.map((p, i) => `
        <div class="row">
          <div class="name">${p.name}</div>
          <div class="pct">${(p.confidence * 100).toFixed(1)}%</div>
          <div class="desc">${p.description || ''}</div>
          <div class="bar"><i style="width:${(p.confidence * 100).toFixed(1)}%"></i></div>
        </div>`).join('');
      results.classList.add('show');
    } catch (e) {
      err.textContent = e.message || String(e);
    } finally {
      run.disabled = false;
    }
  });

  fetch('/api/classes').then(r => r.json()).then(d => {
    const mode = d.mode === 'specialist' ? '专家模型' : '旧 10 类模型';
    document.getElementById('classes').innerHTML =
      `${mode}（${d.model || ''}）：` + d.classes.map(c => `<code>${c}</code>`).join(' ');
  }).catch(() => {
    document.getElementById('classes').textContent = '模型尚未就绪，请先完成训练。';
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/classes")
def classes():
    rec = get_recognizer()
    return {
        "classes": list(rec.classes),
        "descriptions": _descriptions(rec),
        "mode": recognizer_mode,
        "model": str(
            SPECIALIST_PATH.name if recognizer_mode == "specialist" else LEGACY_PATH.name
        ),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        rec = get_recognizer()
    except FileNotFoundError as e:
        return JSONResponse({"detail": str(e)}, status_code=503)

    raw = await file.read()
    try:
        image = Image.open(BytesIO(raw))
    except Exception:
        return JSONResponse({"detail": "无法解析图片文件"}, status_code=400)

    if recognizer_mode == "specialist":
        out = rec.predict(image, top_k=3, use_router=False)
        desc = _descriptions(rec)
        preds = [
            {
                "name": t["name"],
                "confidence": t["confidence"],
                "description": desc.get(t["name"], ""),
            }
            for t in out["top"]
        ]
        return {
            "filename": file.filename,
            "predictions": preds,
            "mode": "specialist",
            "path": out.get("path", "cnn"),
            "roi_box": out.get("roi_box"),
        }

    preds = rec.predict(image, top_k=5)
    return {"filename": file.filename, "predictions": preds, "mode": "legacy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
