#!/usr/bin/env bash
# End-to-end: generate data, train, launch web UI
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

SAMPLES="${SAMPLES:-200}"
EPOCHS="${EPOCHS:-25}"
PORT="${PORT:-7860}"

python3 generate_dataset.py --samples "$SAMPLES" --size 128
python3 train.py --epochs "$EPOCHS" --batch-size 64 --size 128
echo "Model ready. Starting web UI on http://0.0.0.0:${PORT}"
python3 app.py
