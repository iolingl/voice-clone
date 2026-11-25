#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${GPT_SOVITS_REPO:-$PROJECT_ROOT/third_party/GPT-SoVITS}"
DATA="${DATA:-$PROJECT_ROOT/data/finetune}"
EXP="${EXP:-dongxuelian}"
LIST="${LIST:-$DATA/gptsovits_clean.list}"
WAV_DIR="${WAV_DIR:-$DATA/wavs}"
OPT="$REPO/logs/$EXP"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate gptsovits
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export _CUDA_VISIBLE_DEVICES=${_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}
export PYTHONPATH="$REPO/GPT_SoVITS:$REPO${PYTHONPATH:+:$PYTHONPATH}"

export inp_text="$LIST"
export inp_wav_dir="$WAV_DIR"
export exp_name="$EXP"
export opt_dir="$OPT"
export i_part=0
export all_parts=1
export is_half=True
export bert_pretrained_dir="$REPO/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="$REPO/GPT_SoVITS/pretrained_models/chinese-hubert-base"
export pretrained_s2G="$REPO/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
export s2config_path="$REPO/GPT_SoVITS/configs/s2.json"

cd "$REPO"
mkdir -p "$OPT"

echo "=== 1-get-text ==="
python -s GPT_SoVITS/prepare_datasets/1-get-text.py
mv -f "$OPT/2-name2text-0.txt" "$OPT/2-name2text.txt"

echo "=== 2-get-hubert-wav32k ==="
python -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

echo "=== 3-get-semantic ==="
python -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
mv -f "$OPT/6-name2semantic-0.tsv" "$OPT/6-name2semantic.tsv"

echo "=== done ==="
wc -l "$OPT/2-name2text.txt" "$OPT/6-name2semantic.tsv"
ls "$OPT/5-wav32k" | wc -l
