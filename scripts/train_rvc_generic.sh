#!/usr/bin/env bash
# Generic RVC v2 40k training on a directory of 40 kHz wavs.
#
#   WAVS=/abs/path/to/wavs EXP=my_exp GPU_LIST=2-3 EPOCHS=200 \
#     bash scripts/train_rvc_generic.sh
#
# Defaults reproduce the dongxuelian setup.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC="${RVC_REPO:-$PROJECT_ROOT/third_party/RVC}"
WAVS="${WAVS:-$PROJECT_ROOT/data/finetune_all/wavs}"
EXP="${EXP:-dongxuelian_rvc40k}"
GPU_LIST="${GPU_LIST:-2-3}"
EPOCHS="${EPOCHS:-200}"
SAVE_EVERY="${SAVE_EVERY:-20}"
BS="${BS:-8}"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate gptsovits
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RVC${PYTHONPATH:+:$PYTHONPATH}"
cd "$RVC"

echo "=== exp=$EXP wavs=$WAVS gpus=$GPU_LIST epochs=$EPOCHS ==="
mkdir -p "logs/$EXP"
[ -f "logs/$EXP/config.json" ] || cp configs/v1/40k.json "logs/$EXP/config.json"

echo "=== preprocess ==="
python -m train.preprocess "$WAVS" 40000 8 "logs/$EXP" True 3.7

echo "=== f0 (rmvpe) ==="
python -m train.dataset.extract_f0 cuda 1 0 2 "logs/$EXP" True

echo "=== hubert ==="
python -m train.dataset.extract_hubert_feature cuda 1 0 2 "logs/$EXP" v2 True

echo "=== filelist ==="
python - "$RVC/logs/$EXP" <<'PY'
import os, sys
from random import shuffle
exp = sys.argv[1]
gt = f'{exp}/0_gt_wavs'
feat = f'{exp}/3_feature768'
f0 = f'{exp}/2a_f0'
f0nsf = f'{exp}/2b-f0nsf'
lines = []
for name in sorted(os.listdir(gt)):
    if not name.endswith('.wav'):
        continue
    key = name[:-4]
    if not all(os.path.exists(f'{d}/{key}.wav.npy') for d in (f0, f0nsf)):
        print('skip', key); continue
    if not os.path.exists(f'{feat}/{key}.npy'):
        print('skip feat', key); continue
    lines.append(f'{gt}/{name}|{feat}/{key}.npy|{f0}/{key}.wav.npy|{f0nsf}/{key}.wav.npy|0')
shuffle(lines)
if os.path.isdir('logs/mute/0_gt_wavs'):
    mute = 'logs/mute/0_gt_wavs/mute40k.wav'
    if os.path.exists(mute):
        for _ in range(2):
            lines.append(f'{os.path.abspath(mute)}|logs/mute/3_feature768/mute.npy|'
                         'logs/mute/2a_f0/mute.wav.npy|logs/mute/2b-f0nsf/mute.wav.npy|0')
with open(f'{exp}/filelist.txt', 'w') as f:
    f.write('\n'.join(lines))
print('filelist lines', len(lines))
PY

echo "=== train (gpus=$GPU_LIST, epochs=$EPOCHS) ==="
python -m train.train -e "$EXP" -sr 40k -f0 1 -bs "$BS" -g "$GPU_LIST" -te "$EPOCHS" \
  -se "$SAVE_EVERY" -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth \
  -l 1 -c 0 -sw 1 -v v2

echo "=== index ==="
mkdir -p assets/indices
python -m train.train_index "$EXP" v2 "assets/indices/$EXP.index" 8
echo "=== done: logs/$EXP ==="
