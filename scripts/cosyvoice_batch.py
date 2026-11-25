import os, sys, argparse, time
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
    ap.add_argument('--refs', nargs='+', required=True)
    ap.add_argument('--texts_file', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='speech rate; <1 slower/longer vowels, better for singing')
    args = ap.parse_args()

    texts = [l.strip() for l in open(args.texts_file, encoding='utf-8') if l.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    print('loading model...', flush=True)
    model = AutoModel(model_dir=args.model_dir)
    print('sample_rate', model.sample_rate, flush=True)
    for ref in args.refs:
        ref_name = os.path.splitext(os.path.basename(ref))[0]
        for ti, text in enumerate(texts, 1):
            t0 = time.time()
            n = 0
            for out in model.inference_cross_lingual(wrap(text), ref, stream=False, speed=args.speed):
                outp = os.path.join(args.outdir, f'{args.tag}{ref_name}_t{ti}_{n}.wav')
                torchaudio.save(outp, out['tts_speech'], model.sample_rate)
                n += 1
            print(f'DONE {outp} ({time.time()-t0:.1f}s, {out["tts_speech"].shape[1]/model.sample_rate:.2f}s audio)', flush=True)

if __name__ == '__main__':
    main()
