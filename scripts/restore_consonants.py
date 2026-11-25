#!/usr/bin/env python
"""Restore consonant clarity lost by RVC.

RVC converts a singing voice with a speech-trained model; its output is
3-7 dB darker than the source in the 4-8 kHz region, and the loss is
concentrated in unvoiced consonant frames -- the objective reason Chinese
singing articulation sounds mushy.  This script takes the 3.5-9 kHz band of
the *source* vocal, gates it with a consonant mask estimated from the
source, and mixes it back into the converted vocal at a calibrated gain.

Usage:
  python scripts/restore_consonants.py --source orig_lead.wav \
      --converted rvc_lead.wav --output rvc_lead_cons.wav --target -4.8
"""
import argparse

import librosa
import numpy as np
import soundfile as sf

N_FFT = 1024
HOP = 256
LO, HI = 3500.0, 9000.0


def band_ratio(y, sr, lo=4000.0, hi=8000.0):
    """dB of 4-8 kHz energy over total energy (all frames)."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return 10 * np.log10(S[(f >= lo) & (f < hi)].sum() / (S.sum() + 1e-12) + 1e-12)


def hf_ratio(y, sr):
    """dB of 3.5-9 kHz energy over total energy on consonant frames."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    hf = S[(f >= LO) & (f < HI)].sum(0)
    tot = S.sum(0)
    ratio = hf / (tot + 1e-12)
    mask = ratio > 0.04
    return 10 * np.log10(hf[mask].mean() / (tot[mask].mean() + 1e-12) + 1e-12), int(mask.sum())


def align(src, con, sr):
    """Estimate the sample offset that best aligns src to con (envelope xcorr)."""
    es = librosa.feature.rms(y=src, frame_length=512, hop_length=128)[0]
    ec = librosa.feature.rms(y=con, frame_length=512, hop_length=128)[0]
    m = min(len(es), len(ec))
    a = es[:m] - es[:m].mean()
    b = ec[:m] - ec[:m].mean()
    c = np.correlate(a, b, 'full')
    lag = int(np.argmax(c) - (m - 1))
    # numpy/scipy convention: lag = -delay, i.e. positive => con is EARLIER
    return -lag * 128


def consonant_hf(src, sr, n):
    S = librosa.stft(src, n_fft=N_FFT, hop_length=HOP)
    f = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = ((f >= LO) & (f < HI)).astype(np.float32)
    tot = np.abs(S) ** 2
    ratio = tot[(f >= LO) & (f < HI)].sum(0) / (tot.sum(0) + 1e-12)
    e = tot.sum(0)
    ok = e > e.max() * 1e-3
    m = np.clip((ratio - 0.015) / 0.05, 0, 1) * ok
    # smooth ~30 ms so the gate does not chatter
    w = np.hanning(11)
    w /= w.sum()
    m = np.convolve(m, w, mode='same')
    Sh = S * band[:, None]
    hf = librosa.istft(Sh, hop_length=HOP, length=n)
    # mask is per-frame; apply it in the time domain after upsampling
    mt = np.interp(np.arange(n), np.arange(len(m)) * HOP + N_FFT // 2, m,
                   left=0, right=0)
    return hf * mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='original lead stem')
    ap.add_argument('--converted', required=True, help='RVC converted lead')
    ap.add_argument('--output', required=True)
    ap.add_argument('--metric', default='band', choices=['band', 'cons'],
                    help="'band' targets the overall 4-8 kHz energy ratio "
                         "(default, matches the source's brightness); "
                         "'cons' targets the consonant-frame 3.5-9 kHz ratio")
    ap.add_argument('--target', type=float, default=None,
                    help='target value in dB; default = the source\'s own value')
    ap.add_argument('--sr', type=int, default=44100)
    args = ap.parse_args()
    sr = args.sr

    src, _ = librosa.load(args.source, sr=sr, mono=True)
    con, _ = librosa.load(args.converted, sr=sr, mono=True)
    delay = align(src, con, sr)  # delay of con relative to src
    print(f'align delay {delay} samples ({delay / sr * 1000:.1f} ms)')
    # keep the source timeline: shift the converted stem, not the source
    if delay > 0:
        con = con[delay:]
    elif delay < 0:
        con = np.concatenate([np.zeros(-delay, dtype=np.float32), con])
    n = min(len(src), len(con))
    src, con = src[:n], con[:n]

    if args.metric == 'band':
        measure = lambda z: band_ratio(z, sr)
        tgt = band_ratio(src, sr) if args.target is None else args.target
    else:
        measure = lambda z: hf_ratio(z, sr)[0]
        tgt = hf_ratio(src, sr)[0] if args.target is None else args.target
    print(f'converted {args.metric} {measure(con):.2f} dB, source {measure(src):.2f} dB, '
          f'target {tgt:.2f} dB')

    hf = consonant_hf(src, sr, n)
    if measure(con) >= tgt:
        g = 0.0
        print('already at/above target, no HF added')
    else:
        lo, hi = 0.0, 4.0
        for _ in range(16):
            mid = (lo + hi) / 2
            if measure(con + mid * hf) < tgt:
                lo = mid
            else:
                hi = mid
        g = (lo + hi) / 2
    out = con + g * hf
    print(f'gain {g:.3f} -> {args.metric} {measure(out):.2f} dB '
          f'(target {tgt:.2f}), consonant HF ratio {hf_ratio(out, sr)[0]:.2f} dB')

    # do not exceed the converted peak too much
    peak = np.abs(out).max()
    if peak > 1.0:
        out *= 0.99 / peak
    sf.write(args.output, out.astype(np.float32), sr, subtype='PCM_16')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
