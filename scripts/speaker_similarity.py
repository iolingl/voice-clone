"""Compute SV2TTS speaker-encoder cosine similarity between a reference wav and generated wavs."""
import sys, argparse, os
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(os.environ.get('VOICE_CLONE_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))
from encoder import inference as encoder
import librosa

def emb(path):
    wav, _ = librosa.load(str(path), sr=16000)
    wav = encoder.preprocess_wav(wav)
    return encoder.embed_utterance(wav)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('wavs', nargs='+')
    args = ap.parse_args()
    encoder.load_model(PROJECT_ROOT / 'saved_models/default/encoder.pt')
    r = emb(args.ref)
    for p in args.wavs:
        e = emb(p)
        print(f'{np.dot(r, e):.4f}\t{p}')

if __name__ == '__main__':
    main()
