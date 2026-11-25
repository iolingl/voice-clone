#!/usr/bin/env bash
# Launch the voice-cloning demo: CosyVoice worker (voiceclone env) + main web app (gptsovits env).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- configurable via environment -----------------------------------------
# Auto-detect conda; override with CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
if [[ -z "${CONDA_SH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  else
    for cand in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "$cand" ]]; then CONDA_SH="$cand"; break; fi
    done
  fi
fi
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VOICE_ENV="${VOICE_ENV:-gptsovits}"     # env with fastapi/uvicorn + GPT-SoVITS
COSY_ENV="${COSY_ENV:-voiceclone}"      # env with CosyVoice3
PORT="${PORT:-8000}"
export COSY_WORKER_PORT="${COSY_WORKER_PORT:-8102}"
export COSY_WORKER_HOST="${COSY_WORKER_HOST:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Optionally point at universal base models that are not bundled:
#   export COSYVOICE_MODEL_DIR=/path/to/Fun-CosyVoice3-0.5B
#   export GPT_SOVITS_REPO=/path/to/GPT-SoVITS

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[start] ERROR: conda.sh not found: $CONDA_SH"
  echo "[start] Set it first, e.g.  export CONDA_SH=\"\$HOME/miniconda3/etc/profile.d/conda.sh\""
  exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

COSY_PID=""
cleanup() {
  if [[ -n "$COSY_PID" ]] && kill -0 "$COSY_PID" 2>/dev/null; then
    echo "[start] stopping CosyVoice worker ($COSY_PID)"
    kill "$COSY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[start] GPU=$CUDA_VISIBLE_DEVICES  app_port=$PORT  cosy_port=$COSY_WORKER_PORT"

if [[ "${COSY_DISABLE:-0}" != "1" ]]; then
  echo "[start] launching CosyVoice worker (env=$COSY_ENV) ..."
  conda run -n "$COSY_ENV" --no-capture-output python cosy_worker.py --port "$COSY_WORKER_PORT" &
  COSY_PID=$!
  echo "[start] waiting for CosyVoice worker (model load ~30-60s) ..."
  ready=0
  for _ in $(seq 1 180); do
    if curl -sf "http://${COSY_WORKER_HOST}:${COSY_WORKER_PORT}/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 2
  done
  if [[ "$ready" == "1" ]]; then
    echo "[start] CosyVoice worker ready."
  else
    echo "[start] WARNING: CosyVoice worker did not become ready; 37 voice will be unavailable."
  fi
else
  echo "[start] COSY_DISABLE=1 -> skipping CosyVoice worker (37 voice unavailable)."
fi

echo "[start] launching web app (env=$VOICE_ENV) on http://0.0.0.0:${PORT}"
echo "[start] open http://localhost:${PORT} in your browser (wait for GPT-SoVITS to load)."
conda run -n "$VOICE_ENV" --no-capture-output uvicorn app:app --host 0.0.0.0 --port "$PORT"
