#!/usr/bin/env python
"""Cluster VAD clips by speaker embedding to find stray voices in an autocut.

Usage:
  python scripts/cluster_speakers.py <clips_dir> [--threshold 0.30] [--out clusters.csv]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

PROJECT_ROOT = Path(os.environ.get('VOICE_CLONE_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))
from encoder import inference as encoder  # noqa: E402
import librosa  # noqa: E402


def emb(path):
    wav, _ = librosa.load(str(path), sr=16000)
    wav = encoder.preprocess_wav(wav)
    return encoder.embed_utterance(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips')
    ap.add_argument('--threshold', type=float, default=0.30,
                    help='cosine distance threshold (smaller = more clusters)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--ref', default=None, help='optional reference clip')
    args = ap.parse_args()

    man = os.path.join(args.clips, 'manifest.csv')
    names = [r['file'] for r in csv.DictReader(open(man))]
    encoder.load_model(PROJECT_ROOT / 'saved_models/default/encoder.pt')

    E = np.stack([emb(os.path.join(args.clips, n)) for n in names])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    cl = AgglomerativeClustering(n_clusters=None, metric='cosine',
                                 linkage='average',
                                 distance_threshold=args.threshold)
    lab = cl.fit_predict(E)
    order = np.argsort([-np.sum(lab == k) for k in range(lab.max() + 1)])
    remap = {int(old): new for new, old in enumerate(order)}
    lab = np.array([remap[int(k)] for k in lab])

    print(f'{len(names)} clips, {lab.max() + 1} clusters (threshold {args.threshold})')
    rows = []
    for k in range(lab.max() + 1):
        idx = np.where(lab == k)[0]
        cen = E[idx].mean(0)
        cen /= np.linalg.norm(cen) + 1e-9
        sims = E[idx] @ cen
        print(f'  cluster {k}: {len(idx):3d} clips, {np.median(sims):.3f} median sim, '
              f'rep {names[idx[0]]}')
        for i in idx:
            rows.append([names[i], k, f'{sims[list(idx).index(i)]:.4f}'])
    out = args.out or os.path.join(args.clips, 'clusters.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'cluster', 'sim_to_centroid'])
        w.writerows(rows)
    print('wrote', out)

    if args.ref:
        r = emb(args.ref)
        r /= np.linalg.norm(r) + 1e-9
        print('similarity of each centroid to ref:')
        for k in range(lab.max() + 1):
            idx = np.where(lab == k)[0]
            cen = E[idx].mean(0)
            cen /= np.linalg.norm(cen) + 1e-9
            print(f'  cluster {k}: {np.dot(cen, r):.3f}')


if __name__ == '__main__':
    main()
