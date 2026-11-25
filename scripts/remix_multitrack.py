#!/usr/bin/env python
"""Remix RVC-converted vocal stems with the original instrumental.

For each converted (mono) stem we

  1. apply a gentle vocal EQ (same chain as polish_song.py),
  2. match its RMS to the *original* stem's RMS so the arrangement's
     dynamics (and the number of harmony voices over time) are preserved,
  3. rebuild the stereo image from the original stem's L/R energy ratio,
  4. sum lead + backing + instrumental and peak-normalise.

Usage:
  python scripts/remix_multitrack.py \
      --lead rvc_lead.wav --back rvc_back.wav \
      --lead-ref orig_lead.wav --back-ref orig_back.wav \
      --inst instrumental.wav --outdir out --name cover_multitrack
"""
import argparse
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf

def eq_chain(presence=2.5):
    return ("highpass=f=80,"
            f"equalizer=f=4500:t=q:w=1.0:g={presence},"
            "equalizer=f=300:t=q:w=1:g=-2,"
            "equalizer=f=11000:t=h:g=-3,"
            "alimiter=limit=0.95")


def load_mono(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def eq_file(path, out, sr, presence=2.5, channels=1):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path,
                    '-af', eq_chain(presence), '-ar', str(sr),
                    '-ac', str(channels), out], check=True)
    return out


def stereo_from_mono(y, ref_stereo):
    """Rebuild a stereo image using the reference stem's L/R energy ratio."""
    n = min(len(y), ref_stereo.shape[1])
    ref = ref_stereo[:, :n]
    el = float(np.mean(ref[0] ** 2) + 1e-12)
    er = float(np.mean(ref[1] ** 2) + 1e-12)
    gl = np.sqrt(el / (el + er))
    gr = np.sqrt(er / (el + er))
    out = np.zeros((2, n), np.float32)
    out[0] = y[:n] * gl
    out[1] = y[:n] * gr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lead', required=True, help='RVC-converted lead vocal (mono)')
    ap.add_argument('--back', required=True, help='RVC-converted backing vocal (mono)')
    ap.add_argument('--lead-ref', required=True, help='original separated lead stem (stereo)')
    ap.add_argument('--back-ref', required=True, help='original separated backing stem (stereo)')
    ap.add_argument('--inst', required=True, help='instrumental (stereo)')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--lead-gain', type=float, default=1.0)
    ap.add_argument('--back-gain', type=float, default=1.0)
    ap.add_argument('--back-air', type=float, default=0.0,
                    help='optional air (13k+) restoration gain from the original back stem')
    ap.add_argument('--presence', type=float, default=2.5,
                    help='dB boost at 4.5 kHz (consonant clarity); RVC stems measure '
                         '3-4 dB darker than the source in 4-8 kHz')
    ap.add_argument('--vocal-only', action='store_true',
                    help='also write a vocals-only mix')
    ap.add_argument('--sr', type=int, default=44100)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sr = args.sr

    lead = load_mono(eq_file(args.lead, os.path.join(args.outdir,
                                                     f'{args.name}_lead_eq.wav'),
                             sr, args.presence), sr)
    back_path = eq_file(args.back, os.path.join(args.outdir,
                                                f'{args.name}_back_eq.wav'),
                        sr, args.presence, channels=2)
    back_raw, _ = librosa.load(back_path, sr=sr, mono=False)
    back_is_stereo = back_raw.ndim == 2
    back = back_raw.mean(0) if back_is_stereo else back_raw
    lead_ref, _ = librosa.load(args.lead_ref, sr=sr, mono=False)
    back_ref, _ = librosa.load(args.back_ref, sr=sr, mono=False)
    if lead_ref.ndim == 1:
        lead_ref = np.stack([lead_ref, lead_ref])
    if back_ref.ndim == 1:
        back_ref = np.stack([back_ref, back_ref])
    inst, _ = librosa.load(args.inst, sr=sr, mono=False)
    if inst.ndim == 1:
        inst = np.stack([inst, inst])

    n = min(lead_ref.shape[1], back_ref.shape[1], inst.shape[1], len(lead), len(back))

    # match converted stem loudness to the original stem
    lead = lead[:n] * rms(lead_ref[:, :n]) * args.lead_gain / (rms(lead[:n]) + 1e-9)
    if back_is_stereo:
        back_st = back_raw[:, :n] * (rms(back_ref[:, :n]) * args.back_gain
                                     / (rms(back_raw[:, :n]) + 1e-9))
        back = back_st.mean(0)
    else:
        back = back[:n] * rms(back_ref[:, :n]) * args.back_gain / (rms(back[:n]) + 1e-9)
        back_st = stereo_from_mono(back, back_ref[:, :n])

    if args.back_air > 0:
        from scipy.signal import butter, sosfilt
        hf = sosfilt(butter(6, 13000, 'highpass', fs=sr, output='sos'),
                     back_ref.mean(0)[:n])
        back_st = back_st + args.back_air * hf

    lead_st = stereo_from_mono(lead, lead_ref[:, :n])

    voc = lead_st + back_st
    mix = voc + inst[:, :n]
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix *= 0.95 / peak
        voc *= 0.95 / peak
    out = os.path.join(args.outdir, f'{args.name}.wav')
    sf.write(out, mix.T.astype(np.float32), sr, subtype='PCM_16')
    print(f'wrote {out} ({n / sr:.1f}s, peak gain {min(1, 0.95 / peak):.2f})')

    if args.vocal_only:
        voc *= 0.95 / (np.abs(voc).max() + 1e-9)
        p = os.path.join(args.outdir, f'{args.name}_vocal.wav')
        sf.write(p, voc.T.astype(np.float32), sr, subtype='PCM_16')
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
