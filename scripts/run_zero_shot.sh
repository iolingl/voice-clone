#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate voiceclone
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "${COSYVOICE_REPO:-$PROJECT_ROOT/third_party/CosyVoice}"
python "$PROJECT_ROOT/scripts/cosyvoice_zero_shot.py" "$@"
