#!/usr/bin/env python
"""VAD-segment a long recording into utterance clips for speaker checks/training.

Runs with the `learn` env (needs webrtcvad).  Unlike segment_finetune.py it does
not skip the first two minutes and keeps every >= --min-len clip, which is what
we want for a 20-minute voice-line cut.

Usage:
  python scripts/vad_segments.py in.wav outdir [--min-len 2.5] [--max-len 12]
"""
import argparse
import csv
import os
import struct

import librosa
import numpy as np
import soundfile as sf
import webrtcvad

VAD_SR = 16000


def vad_regions(y16, aggressiveness=2, merge_gap=0.30):
    win = int(0.03 * VAD_SR)
    vad = webrtcvad.Vad(aggressiveness)
    flags = []
    for i in range(0, len(y16) - win, win):
        pcm = struct.pack('%dh' % win,
                          *(np.clip(y16[i:i + win], -1, 1) * 32767).astype(np.int16))
        flags.append(vad.is_speech(pcm, VAD_SR))
    flags = np.array(flags, dtype=np.float32)
    w = 8
    pad = np.concatenate((np.zeros((w - 1) // 2), flags, np.zeros(w // 2)))
    c = np.cumsum(pad)
    sm = ((c[w:] - c[:-w]) / w >= 0.5).astype(np.int8)
    regions, start, last = [], None, None
    for i, v in enumerate(sm):
        t0 = i * 0.03
        t1 = t0 + 0.03
        if v:
            if start is None:
                start = t0
            last = t1
        elif start is not None and t0 - last > merge_gap:
            regions.append((start, last))
            start = None
    if start is not None:
        regions.append((start, last))
    return regions


def split_long(y, sr, a, b, max_len=12.0, min_piece=3.0):
    """Split a long region at the quietest points into <= max_len pieces."""
    if b - a <= max_len:
        return [(a, b)]
    seg = y[int(a * sr):int(b * sr)]
    fl = int(0.05 * sr)
    n = len(seg) // fl
    if n < 20:
        return [(a, b)]
    e = np.sqrt(np.mean(seg[:n * fl].reshape(n, fl) ** 2, axis=1))
    e = np.convolve(e, np.ones(3) / 3, 'same')
    pieces, pos, end = [], 0.0, b - a
    while end - pos > max_len:
        lo = int((pos + min_piece) / 0.05)
        hi = min(n, int((pos + max_len * 0.9) / 0.05))
        if hi <= lo:
            break
        cut = (lo + int(np.argmin(e[lo:hi]))) * 0.05
        pieces.append((a + pos, a + cut))
        pos = cut
    if end - pos >= min_piece:
        pieces.append((a + pos, a + end))
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('outdir')
    ap.add_argument('--min-len', type=float, default=2.5)
    ap.add_argument('--max-len', type=float, default=12.0)
    ap.add_argument('--min-rms', type=float, default=0.006)
    ap.add_argument('--max-rms', type=float, default=0.25)
    ap.add_argument('--sr', type=int, default=40000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    y16, _ = librosa.load(args.src, sr=VAD_SR, mono=True)
    regions = vad_regions(y16)
    print(f'VAD regions: {len(regions)}  total speech {sum(b - a for a, b in regions) / 60:.1f} min')

    y, _ = librosa.load(args.src, sr=args.sr, mono=True)
    rows, kept = [], 0
    for a0, b0 in regions:
        for a, b in split_long(y, args.sr, a0, b0, args.max_len):
            d = b - a
            if d < args.min_len:
                continue
            seg = y[int(a * args.sr):int(b * args.sr)]
            rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
            peak = float(np.abs(seg).max())
            if not (args.min_rms <= rms <= args.max_rms):
                continue
            if peak > 0.98:
                seg = seg * (0.97 / peak)
                peak = 0.97
            kept += 1
            name = f'seg_{kept:05d}.wav'
            sf.write(os.path.join(args.outdir, name), seg.astype(np.float32),
                     args.sr, subtype='PCM_16')
            rows.append([name, f'{a:.2f}', f'{b:.2f}', f'{d:.2f}',
                         f'{20 * np.log10(rms + 1e-9):.1f}', f'{peak:.3f}'])

    with open(os.path.join(args.outdir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'start_s', 'end_s', 'duration_s', 'rms_dbfs', 'peak'])
        w.writerows(rows)
    total = sum(float(r[3]) for r in rows)
    print(f'kept {kept} clips, {total / 60:.1f} min, median dur '
          f'{np.median([float(r[3]) for r in rows]):.1f}s')


if __name__ == '__main__':
    main()
