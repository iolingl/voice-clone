#!/usr/bin/env python
"""Polish a Seed-VC singing conversion: EQ + air restoration + remix.

Seed-VC 44.1k output has almost no energy above 14 kHz (measured ~-76 dB
relative to total, vs ~-39 dB in the source), which makes it sound dull and
"electric".  This script

  1. applies a gentle EQ (highpass 80, -2 dB @300, +2.5 dB @4.5k, high shelf
     -3 dB @11k) using ffmpeg,
  2. takes the 13 kHz+ band of the *source* vocal and mixes it back in to
     restore air,
  3. normalises the vocal to the source vocal RMS and mixes it with the
     instrumental.

Usage:
  python scripts/polish_song.py --vocal <converted.wav> --source <src vocal.wav> \
      --inst <no_vocals.wav> --outdir <dir> --name cover_xxx
"""
import argparse
import os
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

EQ = ("highpass=f=80,"
      "equalizer=f=300:t=q:w=1:g=-2,"
      "equalizer=f=4500:t=q:w=1.0:g=2.5,"
      "equalizer=f=11000:t=h:g=-3,"
      "alimiter=limit=0.95")


def band_ratio(y, sr, lo=14000):
    s = np.abs(librosa.stft(y[: sr * 120], n_fft=4096, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=4096)
    return 10 * np.log10(s[f >= lo].sum() / s.sum() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocal', required=True, help='Seed-VC converted vocal')
    ap.add_argument('--source', required=True, help='source vocal (has real HF)')
    ap.add_argument('--inst', required=True, help='instrumental / no_vocals')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--air-gain', type=float, default=None,
                    help='gain for source HF; default = auto to match source level -1 dB')
    ap.add_argument('--sr', type=int, default=44100)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sr = args.sr

    eq_path = os.path.join(args.outdir, f'{args.name}_vocal_eq.wav')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', args.vocal,
                    '-af', EQ, '-ar', str(sr), '-ac', '1', eq_path], check=True)

    y, _ = librosa.load(eq_path, sr=sr, mono=True)
    src, _ = librosa.load(args.source, sr=sr, mono=True)
    n = min(len(y), len(src))
    y, src = y[:n], src[:n]

    hf = sosfilt(butter(6, 13000, 'highpass', fs=sr, output='sos'), src)
    if args.air_gain is None:
        target = band_ratio(src, sr) - 1.0
        lo, hi = 0.05, 4.0
        for _ in range(18):
            g = (lo + hi) / 2
            if band_ratio(y + g * hf, sr) < target:
                lo = g
            else:
                hi = g
        gain = (lo + hi) / 2
    else:
        gain = args.air_gain
    y = y + gain * hf
    print(f'air gain {gain:.2f}  band14+ {band_ratio(y, sr):.1f} dB '
          f'(source {band_ratio(src, sr):.1f} dB)')
    sf.write(os.path.join(args.outdir, f'{args.name}_vocal.wav'),
             (y / np.abs(y).max() * 0.95).astype(np.float32), sr, subtype='PCM_16')

    # normalise to source vocal RMS, then mix with instrumental
    inst, _ = librosa.load(args.inst, sr=sr, mono=False)
    if inst.ndim == 1:
        inst = np.stack([inst, inst], 1)
    ref_rms = float(np.sqrt(np.mean(src ** 2)))
    y = y * ref_rms / (np.sqrt(np.mean(y ** 2)) + 1e-9)
    m = min(len(y), inst.shape[1])
    mix = np.stack([y[:m], y[:m]], 1) + inst[:, :m].T
    mix *= 0.95 / np.abs(mix).max()
    out = os.path.join(args.outdir, f'{args.name}.wav')
    sf.write(out, mix.astype(np.float32), sr, subtype='PCM_16')
    print(f'wrote {out} ({m / sr:.1f}s)')


if __name__ == '__main__':
    main()
