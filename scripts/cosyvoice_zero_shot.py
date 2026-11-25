import os, sys, argparse
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get('VOICE_CLONE_ROOT', Path(__file__).resolve().parents[1]))
REPO = Path(os.environ.get('COSYVOICE_REPO', PROJECT_ROOT / 'third_party' / 'CosyVoice'))
sys.path.insert(0, os.path.join(REPO, 'third_party/Matcha-TTS'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch, torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

SYS_PREFIX = 'You are a helpful assistant.<|endofprompt|>'

def wrap(text):
    return text if '<|endofprompt|>' in text else SYS_PREFIX + text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--text', required=True)
    args = ap.parse_args()
    model = AutoModel(model_dir=args.model_dir)
    print('sample_rate', model.sample_rate, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for i, out in enumerate(model.inference_cross_lingual(wrap(args.text), args.ref, stream=False)):
        torchaudio.save(args.out.format(i=i) if '{i}' in args.out else args.out, out['tts_speech'], model.sample_rate)
        print('saved', args.out, out['tts_speech'].shape, flush=True)

if __name__ == '__main__':
    main()
