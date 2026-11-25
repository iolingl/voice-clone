#!/usr/bin/env python
"""Generate diatonic backing harmonies from a lead vocal and mix a cover.

Uses WORLD (pyworld) to move only F0 while keeping the lead's spectral
envelope, so the harmonies keep the converted singer's timbre.  Harmony
intervals follow the song key, not fixed semitones.

Usage:
  python scripts/add_harmony.py --lead <rvc vocal.wav> --inst <no_vocals.wav> \
      --source <source vocal.wav> --key A --mode major --outdir <dir> --name cover_harm
"""
import argparse
import os
import subprocess

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

EQ = ("highpass=f=110,equalizer=f=300:t=q:w=1:g=-2,"
      "equalizer=f=4500:t=q:w=1.0:g=2.0,equalizer=f=11000:t=h:g=-3,alimiter=limit=0.95")

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def nearest_scale(m, pcs):
    base = int(np.floor(m))
    return min((base + k for k in range(-2, 3) if (base + k) % 12 in pcs),
               key=lambda x: abs(x - m))


def shift_steps(base_midi, steps, pcs):
    octv = base_midi // 12
    notes = sorted({12 * (octv + o) + pc for o in range(-2, 3) for pc in pcs})
    i = notes.index(min(notes, key=lambda x: abs(x - base_midi)))
    return notes[i + steps] - base_midi


def harmony_f0(f0, tonic_pc, mode, steps_hi, steps_lo, hi_thresh=74):
    pcs = [(tonic_pc + i) % 12 for i in (MAJOR if mode == 'major' else MINOR)]
    out = np.zeros_like(f0)
    for i, f in enumerate(f0):
        if f <= 0:
            continue
        m = 69 + 12 * np.log2(f / 440.0)
        base = nearest_scale(m, pcs)
        steps = steps_lo if base >= hi_thresh else steps_hi
        out[i] = f * 2.0 ** (shift_steps(base, steps, pcs) / 12.0)
    return out


def world_harmony(x, sr, tonic, mode, steps_hi, steps_lo, name, outdir):
    f0, t = pw.harvest(x.astype(np.float64), sr, f0_floor=65, f0_ceil=900,
                       frame_period=5.0)
    sp = pw.cheaptrick(x.astype(np.float64), f0, t, sr)
    ap = pw.d4c(x.astype(np.float64), f0, t, sr)
    f0h = harmony_f0(f0, tonic, mode, steps_hi, steps_lo)
    y = pw.synthesize(f0h, sp, ap, sr, frame_period=5.0)
    y = y / (np.abs(y).max() + 1e-9) * 0.95
    p = os.path.join(outdir, f'{name}.wav')
    sf.write(p, y.astype(np.float32), sr, subtype='PCM_16')
    print(f'wrote {p}')
    return y


def eq_file(path, out):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path, '-af', EQ,
                    '-ac', '1', out], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lead', required=True, help='RVC converted lead vocal')
    ap.add_argument('--inst', required=True, help='instrumental')
    ap.add_argument('--source', required=True, help='source vocal for loudness/HF')
    ap.add_argument('--key', default='A')
    ap.add_argument('--mode', default='major', choices=['major', 'minor'])
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--third-level', type=float, default=0.34)
    ap.add_argument('--fifth-level', type=float, default=0.22)
    ap.add_argument('--low-level', type=float, default=0.16,
                    help='octave-below double level')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sr = 44100
    tonic = NAMES.index(args.key)

    x, _ = librosa.load(args.lead, sr=sr, mono=True)
    print('lead %.1fs' % (len(x) / sr))

    # diatonic third above (or below on high notes) and fifth above
    third = world_harmony(x, sr, tonic, args.mode, 2, -2, 'harm_third', args.outdir)
    fifth = world_harmony(x, sr, tonic, args.mode, 4, -4, 'harm_fifth', args.outdir)
    # octave-down double, cheap with librosa (keeps envelope roughly)
    low = librosa.effects.pitch_shift(x, sr=sr, n_steps=-12)
    sf.write(os.path.join(args.outdir, 'harm_low.wav'),
             (low / (np.abs(low).max() + 1e-9) * 0.9).astype(np.float32), sr,
             subtype='PCM_16')

    # same EQ on every stem
    lead_eq = eq_file(args.lead, os.path.join(args.outdir, f'{args.name}_lead_eq.wav'))
    third_eq = eq_file(os.path.join(args.outdir, 'harm_third.wav'),
                       os.path.join(args.outdir, f'{args.name}_third_eq.wav'))
    fifth_eq = eq_file(os.path.join(args.outdir, 'harm_fifth.wav'),
                       os.path.join(args.outdir, f'{args.name}_fifth_eq.wav'))
    low_eq = eq_file(os.path.join(args.outdir, 'harm_low.wav'),
                     os.path.join(args.outdir, f'{args.name}_low_eq.wav'))

    L, _ = librosa.load(lead_eq, sr=sr, mono=True)
    T, _ = librosa.load(third_eq, sr=sr, mono=True)
    F, _ = librosa.load(fifth_eq, sr=sr, mono=True)
    Lo, _ = librosa.load(low_eq, sr=sr, mono=True)
    src, _ = librosa.load(args.source, sr=sr, mono=True)
    inst, _ = librosa.load(args.inst, sr=sr, mono=False)
    if inst.ndim == 1:
        inst = np.stack([inst, inst], 1)

    n = min(len(L), len(T), len(F), len(Lo), inst.shape[1])
    L, T, F, Lo = L[:n], T[:n], F[:n], Lo[:n]
    ref = float(np.sqrt(np.mean(src[:n] ** 2)))
    L = L * ref / (np.sqrt(np.mean(L ** 2)) + 1e-9)
    T = T * ref / (np.sqrt(np.mean(T ** 2)) + 1e-9) * args.third_level
    F = F * ref / (np.sqrt(np.mean(F ** 2)) + 1e-9) * args.fifth_level
    Lo = Lo * ref / (np.sqrt(np.mean(Lo ** 2)) + 1e-9) * args.low_level

    # width: pan third left, fifth right, small Haas delays
    d1, d2 = int(0.012 * sr), int(0.018 * sr)
    Td = np.concatenate([np.zeros(d1), T[:-d1]]) if d1 else T
    Fd = np.concatenate([np.zeros(d2), F[:-d2]]) if d2 else F
    left = L + 0.75 * Td + 0.35 * Fd + 0.5 * Lo
    right = L + 0.35 * Td + 0.75 * Fd + 0.5 * Lo
    mix = np.stack([left, right], 1) + inst[:, :n].T
    mix *= 0.95 / np.abs(mix).max()
    out = os.path.join(args.outdir, f'{args.name}.wav')
    sf.write(out, mix.astype(np.float32), sr, subtype='PCM_16')
    print('wrote', out)

    voc = np.stack([left, right], 1)
    voc *= 0.95 / np.abs(voc).max()
    sf.write(os.path.join(args.outdir, f'{args.name}_vocal.wav'),
             voc.astype(np.float32), sr, subtype='PCM_16')


if __name__ == '__main__':
    main()
