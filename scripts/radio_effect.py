#!/usr/bin/env python
"""Telephone/radio-style "synth" coloration for the RVC cover, matched to the
AI-cover reference by LTAS.

The reference cover (BV1x4ALeHEdw) measures ~10-19 dB darker than our RVC lead
below 200 Hz, with a presence bump around 1 kHz, a dip around 2 kHz and less
air above 12 kHz.  That is the "sounds like it comes out of a phone / synth"
colour the user hears.  This script

  1. measures the 1/3-octave LTAS of our vocal bus and of a reference vocal,
  2. builds a smooth EQ curve from their ratio (clamped to +-14 dB),
  3. applies it in the STFT domain (zero latency, no phase smearing),
  4. optionally adds tanh saturation (the "synth" grit),
  5. remixes with the instrumental at the original vocal/instrumental balance.

Usage:
  python scripts/radio_effect.py --vocal <vocal_bus.wav> --match <ai_lead.wav> \
      --inst <instrumental.wav> --outdir <dir> --name cover_radio [--drive 0.4]
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf

BANDS = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
         10000, 12500, 14000, 16000, 18000, 20000]


def ltas(y, sr, n_fft=4096, hop=512):
    """1/3-octave long-term average spectrum, loud frames only, dB."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    rms = np.sqrt(S.mean(axis=0))
    keep = rms > np.percentile(rms, 40)
    S = S[:, keep].mean(axis=1)
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    out = []
    for c in BANDS:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10(S[m].sum() + 1e-15) if m.any() else -120.0)
    return np.array(out)


def match_curve(y, ref, sr, cap=14.0):
    a = ltas(ref, sr)
    b = ltas(y, sr)
    band = np.array(BANDS)
    norm = (band >= 800) & (band <= 1600)
    g = a - a[norm].mean() - (b - b[norm].mean())
    g = np.clip(g, -cap, cap)
    # never boost the sub/rumble region, keep it a hard cut
    g[band < 200] = np.minimum(g[band < 200], 0.0)
    # above 12 kHz the reference is mostly noise floor; only allow cuts,
    # otherwise the ratio explodes into a huge ultrasonic boost
    g[band >= 12000] = np.minimum(g[band >= 12000], 0.0)
    return band, g


def apply_curve(y, sr, band, g, n_fft=2048, hop=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    gain_db = np.interp(f, band, g, left=g[0], right=g[-1])
    S = S * (10.0 ** (gain_db[:, None] / 20.0))
    out = librosa.istft(S, hop_length=hop, length=len(y))
    return out


def saturate(y, drive=0.0, mix=0.0):
    if drive <= 0 or mix <= 0:
        return y
    p = np.abs(y).max() + 1e-9
    z = np.tanh(drive * y / p) / np.tanh(drive) * p
    return (1 - mix) * y + mix * z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocal', default=None, help='vocal bus (lead+backing, stereo ok)')
    ap.add_argument('--mix', default=None,
                    help='alternative to --vocal: an existing full mix.  The vocal '
                         'bus is recovered exactly as mix - s*inst (s by least '
                         'squares), which preserves the vocal/instrumental balance')
    ap.add_argument('--vocal-ref', default=None,
                    help='original vocal bus, used as the loudness reference '
                         '(default: loudness of --vocal itself)')
    ap.add_argument('--match', required=True, help='reference vocal to match timbre')
    ap.add_argument('--inst', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--drive', type=float, default=0.0,
                    help='tanh saturation drive (0 = off, 2-4 = audible grit)')
    ap.add_argument('--drive-mix', type=float, default=0.3)
    ap.add_argument('--cap', type=float, default=14.0, help='max EQ boost/cut dB')
    ap.add_argument('--amount', type=float, default=1.0,
                    help='how much of the matched curve to apply (0-1)')
    ap.add_argument('--vocal-gain', type=float, default=1.0)
    ap.add_argument('--sr', type=int, default=44100)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sr = args.sr

    ref, _ = librosa.load(args.match, sr=sr, mono=True)
    inst, _ = librosa.load(args.inst, sr=sr, mono=False)
    if inst.ndim == 1:
        inst = np.stack([inst, inst])

    mix_in = None
    if args.mix:
        mix_in, _ = librosa.load(args.mix, sr=sr, mono=False)
        if mix_in.ndim == 1:
            mix_in = np.stack([mix_in, mix_in])
        n0 = min(mix_in.shape[1], inst.shape[1])
        m, i0 = mix_in[:, :n0], inst[:, :n0]
        s = float(np.sum(m * i0) / (np.sum(i0 * i0) + 1e-12))
        voc = m - s * i0
        inst = inst * s
        print(f'mix-derived vocal bus, instrumental scale s={s:.3f}')
    else:
        if not args.vocal:
            raise SystemExit('need --vocal or --mix')
        voc, _ = librosa.load(args.vocal, sr=sr, mono=False)

    voc_mono = voc.mean(0) if voc.ndim == 2 else voc
    band, g = match_curve(voc_mono, ref, sr, args.cap)
    g = g * args.amount
    print('EQ curve (dB, relative to 0.8-1.6 kHz):')
    print('  ' + ' '.join(f'{int(b)}:{v:+.1f}' for b, v in zip(band, g)))

    if voc.ndim == 2:
        voc_eq = np.stack([apply_curve(voc[0], sr, band, g),
                           apply_curve(voc[1], sr, band, g)])
    else:
        voc_eq = apply_curve(voc, sr, band, g)
    voc_eq = saturate(voc_eq, args.drive, args.drive_mix)

    n = min(voc_eq.shape[-1], inst.shape[1])
    voc_eq = voc_eq[..., :n]
    if mix_in is None:
        if args.vocal_ref:
            vr, _ = librosa.load(args.vocal_ref, sr=sr, mono=True)
            ref_rms = float(np.sqrt(np.mean(vr[:n] ** 2) + 1e-12))
        else:
            ref_rms = float(np.sqrt(np.mean(voc_mono[:n] ** 2) + 1e-12))
        cur = float(np.sqrt(np.mean(voc_eq ** 2) + 1e-12))
        voc_eq = voc_eq * (ref_rms * args.vocal_gain / cur)

    if voc_eq.ndim == 1:
        mix = np.stack([voc_eq, voc_eq], 1) + inst[:, :n].T
        voc_only = np.stack([voc_eq, voc_eq], 1)
    else:
        mix = voc_eq.T + inst[:, :n].T
        voc_only = voc_eq.T

    if mix_in is not None:
        # keep loudness identical to the input mix for a fair A/B
        target = float(np.sqrt(np.mean(mix_in[:, :n] ** 2) + 1e-12))
        cur = float(np.sqrt(np.mean(mix ** 2) + 1e-12))
        g = min(target / (cur + 1e-12), 0.97 / (np.abs(mix).max() + 1e-9))
        mix *= g
        voc_only *= g
    peak = np.abs(mix).max()
    scale = min(1.0, 0.97 / (peak + 1e-9))
    mix *= scale
    out = os.path.join(args.outdir, f'{args.name}.wav')
    sf.write(out, mix.astype(np.float32), sr, subtype='PCM_16')
    vp = os.path.join(args.outdir, f'{args.name}_vocal.wav')
    sf.write(vp, (voc_only * 0.95 / (np.abs(voc_only).max() + 1e-9)).astype(np.float32),
             sr, subtype='PCM_16')
    print(f'wrote {out}')
    print(f'wrote {vp}')


if __name__ == '__main__':
    main()
