"""CosyVoice3 soft/breathy Chinese TTS with a Japanese reference timbre.

Example:
  python scripts/cosyvoice_soft.py --text "你好" --out out.wav
  python scripts/cosyvoice_soft.py --texts_file texts.txt --outdir out/ --whisper 0.2
"""
import argparse, os, sys
from pathlib import Path
ORIG_CWD = os.getcwd()
PROJECT_ROOT = Path(os.environ.get('VOICE_CLONE_ROOT', Path(__file__).resolve().parents[1]))
REPO = Path(os.environ.get('COSYVOICE_REPO', PROJECT_ROOT / 'third_party' / 'CosyVoice'))
sys.path.insert(0, os.path.join(REPO, 'third_party/Matcha-TTS'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import soundfile as sf
import torch, torchaudio
import pyworld as pw
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.path.join(REPO, 'pretrained_models/Fun-CosyVoice3-0.5B')
DEFAULT_REF = str(PROJECT_ROOT / 'dataset_voice_clone' / 'reference_dongxuelian' / 'ref_02_3783.0s.wav')
SOFT_INSTRUCT = 'You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>'
SYS = 'You are a helpful assistant.<|endofprompt|>'

def breathify(wav_path, out_path, gamma, whisper, rms_db):
    y, sr = sf.read(wav_path, dtype='float64')
    if y.ndim > 1:
        y = y.mean(1)
    f0, t = pw.harvest(y, sr, f0_floor=60, f0_ceil=600, frame_period=5)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    ap2 = 1.0 - np.power(1.0 - np.clip(ap, 0, 1), gamma)
    freqs = np.linspace(0, sr / 2, sp.shape[1])
    hf = freqs >= 4000
    ap2[:, hf] = 1.0 - np.power(1.0 - ap2[:, hf], 1.3)
    out = pw.synthesize(f0, sp, np.clip(ap2, 0, 0.999), sr, frame_period=5)
    if whisper > 0:
        wh = pw.synthesize(np.zeros_like(f0), sp, np.ones_like(ap) * 0.999, sr, frame_period=5)
        out = (1 - whisper) * out + whisper * wh
    out = out[:len(y)]
    out *= (10 ** (rms_db / 20)) / (np.sqrt((out ** 2).mean()) + 1e-12)
    sf.write(out_path, out.astype(np.float32), sr)

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--text')
    ap_.add_argument('--texts_file')
    ap_.add_argument('--ref', default=DEFAULT_REF)
    ap_.add_argument('--out')
    ap_.add_argument('--outdir', default=str(PROJECT_ROOT / 'outputs' / 'soft'))
    ap_.add_argument('--model_dir', default=MODEL_DIR)
    ap_.add_argument('--rl', action='store_true', default=True)
    ap_.add_argument('--soft', action='store_true', default=True, help='use soft-voice instruction')
    ap_.add_argument('--whisper', type=float, default=0.0)
    ap_.add_argument('--gamma', type=float, default=1.5)
    ap_.add_argument('--rms_db', type=float, default=-24.0)
    a = ap_.parse_args()

    def absp(x):
        return x if os.path.isabs(x) else os.path.join(ORIG_CWD, x)
    a.outdir = absp(a.outdir)
    if a.out:
        a.out = absp(a.out)
    texts = [a.text] if a.text else [l.strip() for l in open(absp(a.texts_file), encoding='utf-8') if l.strip()]
    os.makedirs(a.outdir, exist_ok=True)
    m = AutoModel(model_dir=a.model_dir)
    if a.rl and os.path.exists(os.path.join(a.model_dir, 'llm.rl.pt')):
        m.model.load(os.path.join(a.model_dir, 'llm.rl.pt'),
                     os.path.join(a.model_dir, 'flow.pt'),
                     os.path.join(a.model_dir, 'hift.pt'))
    for i, text in enumerate(texts, 1):
        chunks = []
        if a.soft:
            gen = m.inference_instruct2(text, SOFT_INSTRUCT, a.ref, stream=False)
        else:
            gen = m.inference_cross_lingual(SYS + text, a.ref, stream=False)
        for out in gen:
            chunks.append(out['tts_speech'])
        sil = torch.zeros(1, int(0.12 * m.sample_rate))
        full = torch.cat([c for ch in chunks for c in (ch, sil)], dim=1)
        raw = a.out if (a.out and len(texts) == 1) else os.path.join(a.outdir, f'soft_{i:02d}.wav')
        raw = os.path.splitext(raw)[0] + '_raw.wav'
        torchaudio.save(raw, full, m.sample_rate)
        final = a.out if (a.out and len(texts) == 1) else os.path.join(a.outdir, f'soft_{i:02d}.wav')
        if a.whisper > 0:
            breathify(raw, final, a.gamma, a.whisper, a.rms_db)
            os.remove(raw)
        else:
            os.replace(raw, final)
        print('DONE', final, f'{full.shape[1]/m.sample_rate:.2f}s')

if __name__ == '__main__':
    main()
