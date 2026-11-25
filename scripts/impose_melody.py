#!/usr/bin/env python
"""Impose a note grid on read speech with WORLD, keeping the source spectral envelope.

Designed for "TTS speech -> singing" so that a subsequent RVC pass sees a real
melody instead of speech intonation. Word timings come from Whisper (JSON), the
syllable split is refined with an energy dip, and the voiced part of each
syllable is stretched/compressed to fill its note while consonant onsets/codas
keep (approximately) their original duration.

Plan JSON format::

    {
      "syllables": [2, 2, 2, 1, ...],          # per word, in word order
      "lines": [
        {"words": [0, 1, 2, 3],
         "notes": [[60, 1], [60, 1], ..., [67, 2]]},   # MIDI, beats
        ...
      ]
    }
"""

import argparse
import json

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import resample_poly


def frame_energy(x, sr, hop=0.005):
    n = int(round(hop * sr))
    pad = (-len(x)) % n
    xp = np.pad(x, (0, pad))
    return np.sqrt((xp.reshape(-1, n) ** 2).mean(axis=1) + 1e-12)


def refine_span(energy, hop, start, end, rel_db=-35.0, pad=0.015):
    i0 = max(0, int(np.floor((start - 0.05) / hop)))
    i1 = min(len(energy), int(np.ceil((end + 0.03) / hop)) + 1)
    seg = energy[i0:i1]
    if len(seg) == 0:
        return start, end
    thr = seg.max() * 10 ** (rel_db / 20.0)
    mask = seg > thr
    # drop isolated clicks: keep runs >= 30 ms, merge runs separated by < 60 ms
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return start, end
    runs = []
    s = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k - prev > 12:          # 60 ms at 5 ms hop
            runs.append((s, prev))
            s = k
        prev = k
    runs.append((s, prev))
    runs = [(a, b) for a, b in runs if (b - a + 1) * hop >= 0.03]
    if not runs:
        return start, end
    a = (i0 + runs[0][0]) * hop - pad
    b = (i0 + runs[-1][1] + 1) * hop + pad
    return max(start - 0.05, a), min(end + 0.03, b)


def split_syllables(energy, hop, start, end, n_syl):
    if n_syl <= 1:
        return [(start, end)]
    smooth = np.convolve(energy, np.ones(5) / 5.0, mode="same")
    cuts = [start]
    for k in range(1, n_syl):
        target = start + (end - start) * k / n_syl
        radius = max(0.04, 0.28 * (end - start) / n_syl)
        i0 = int(np.floor((target - radius) / hop))
        i1 = int(np.ceil((target + radius) / hop)) + 1
        i0 = max(0, i0)
        i1 = min(len(energy), i1)
        if i1 <= i0:
            cuts.append(target)
            continue
        cuts.append((i0 + int(np.argmin(smooth[i0:i1]))) * hop)
    cuts.append(end)
    cuts = sorted(cuts)
    return list(zip(cuts[:-1], cuts[1:]))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--words-json", required=True)
    ap.add_argument("--plan-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tempo", type=float, default=100.0, help="BPM, quarter note = beat")
    ap.add_argument("--line-gap", type=float, default=0.45)
    ap.add_argument("--vibrato-cents", type=float, default=20.0)
    ap.add_argument("--vibrato-rate", type=float, default=5.3)
    ap.add_argument("--frame-period", type=float, default=5.0)
    ap.add_argument("--keep-pitch", action="store_true",
                    help="debug: keep the source pitch contour, only remap time")
    ap.add_argument("--no-dips", action="store_true",
                    help="debug: skip the small dips at word boundaries")
    ap.add_argument("--dump-tmap", default=None,
                    help="debug: save the output->input time map as .npy")
    args = ap.parse_args()

    x, sr = sf.read(args.wav, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    words = json.load(open(args.words_json))
    plan = json.load(open(args.plan_json))
    syl_counts = plan["syllables"]
    assert len(syl_counts) == len(words), (len(syl_counts), len(words))

    # CTC aligners can squeeze short function words ("I", "a") into a couple
    # of frames; widen those into the neighbouring gaps.
    for i, w in enumerate(words):
        if w["end"] - w["start"] >= 0.15:
            continue
        a = w["start"]
        b = w["end"]
        if i > 0:
            a = max(a - 0.12, words[i - 1]["end"] + 0.02)
        if i < len(words) - 1:
            b = min(b + 0.12, words[i + 1]["start"] - 0.02)
        w["start"], w["end"] = a, b

    energy = frame_energy(x, sr)
    hop = 0.005

    spans = [refine_span(energy, hop, w["start"], w["end"]) for w in words]
    for i in range(len(spans) - 1):
        a1, b1 = spans[i]
        a2, b2 = spans[i + 1]
        if b1 > a2:
            mid = 0.5 * (b1 + a2)
            spans[i] = (a1, max(a1, mid))
            spans[i + 1] = (min(b2, mid), b2)

    f0, t = pw.harvest(x, sr, f0_floor=60.0, f0_ceil=1000.0,
                       frame_period=args.frame_period)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    voiced = f0 > 0

    beat = 60.0 / args.tempo
    anchors = []          # (input_time, output_time)
    word_bounds = []      # output times where a word starts (for small gaps)
    note_info = []        # (out_start, out_end, midi, prev_midi)
    cursor = 0.0
    prev_plan_midi = None

    for li, line in enumerate(plan["lines"]):
        line_start = cursor
        note_times = []
        tt = line_start
        for midi, beats in line["notes"]:
            note_times.append((tt, tt + beats * beat, midi))
            tt += beats * beat
        cursor = tt + args.line_gap

        units = []
        for wi in line["words"]:
            a, b = spans[wi]
            for sa, sb in split_syllables(energy, hop, a, b, syl_counts[wi]):
                units.append((sa, sb, wi))
        assert len(units) == len(note_times), (li, len(units), len(note_times))

        for ui, ((sa, sb, wi), (T0, T1, midi)) in enumerate(zip(units, note_times)):
            if ui == 0 and li > 0:
                anchors.append((sa, line_start))
            elif ui > 0 and units[ui - 1][2] != wi:
                word_bounds.append(T0)
            idx = np.flatnonzero(voiced & (t >= sa) & (t < sb))
            if len(idx) >= 3:
                tv0 = max(sa, t[idx[0]])
                tv1 = min(sb, t[idx[-1]] + args.frame_period / 1000.0)
                d_pre = min(max(0.0, tv0 - sa), 0.22)
                d_suf = min(max(0.0, sb - tv1), 0.22)
                if T1 - d_suf > T0 + d_pre + 0.02:
                    local = [(sa, T0), (tv0, T0 + d_pre),
                             (tv1, T1 - d_suf), (sb, T1)]
                else:
                    local = [(sa, T0), (sb, T1)]
            else:
                local = [(sa, T0), (sb, T0 + min(0.35, sb - sa))]
            for tin, tout in local:
                if anchors and tout < anchors[-1][1]:
                    tout = anchors[-1][1]
                anchors.append((tin, tout))
            note_info.append((T0, T1, midi, prev_plan_midi))
            prev_plan_midi = midi

    # keep strictly increasing output times (zero-length input gaps collapse)
    clean = [anchors[0]]
    for tin, tout in anchors[1:]:
        if tout <= clean[-1][1] + 1e-9:
            if tin > clean[-1][0]:
                clean.append((tin, clean[-1][1]))
        else:
            clean.append((tin, tout))
    xin = np.array([p[0] for p in clean])
    xout = np.array([p[1] for p in clean])

    total = cursor - args.line_gap + 0.3
    nframes = int(round(total / (args.frame_period / 1000.0)))
    frames = np.arange(nframes) * args.frame_period / 1000.0
    tmap = np.interp(frames, xout, xin)
    if args.dump_tmap:
        np.save(args.dump_tmap, tmap)

    sidx = np.clip(np.round(tmap / (args.frame_period / 1000.0)).astype(int),
                   0, len(t) - 1)
    sp_out = sp[sidx].copy()
    ap_out = ap[sidx].copy()
    f0_out = np.zeros(nframes)

    for T0, T1, midi, prev_midi in note_info:
        sel = (frames >= T0) & (frames < T1)
        if not sel.any():
            continue
        if args.keep_pitch:
            f0_out[sel] = f0[sidx[sel]]
            continue
        src_voiced = voiced[sidx[sel]]
        base = 440.0 * 2 ** ((midi - 69) / 12.0)
        tau = frames[sel] - T0
        # portamento from the previous note (80 ms)
        if prev_midi is not None:
            prev = 440.0 * 2 ** ((prev_midi - 69) / 12.0)
            g = smoothstep(tau / 0.08)
            cur = prev * (base / prev) ** g
        else:
            cur = np.full(len(tau), base)
        # delayed vibrato
        depth = args.vibrato_cents * smoothstep((tau - 0.28) / 0.30)
        vib = 2 ** (depth * np.sin(2 * np.pi * args.vibrato_rate * tau) / 1200.0)
        cur = cur * vib
        f0_out[sel] = np.where(src_voiced, cur, 0.0)

    y = pw.synthesize(f0_out, sp_out, ap_out, sr, args.frame_period)

    # small dips at word boundaries for intelligibility
    n = len(y)
    if not args.no_dips:
        for Tw in word_bounds:
            c = int(round(Tw * sr))
            half = int(0.008 * sr)
            hold = int(0.012 * sr)
            lo = max(0, c - half - hold)
            hi = min(n, c + half + hold)
            if hi > lo:
                idx = np.arange(lo, hi)
                d = np.abs(idx - c)
                gain = smoothstep((d - hold) / max(1, half))
                y[lo:hi] *= gain

    y = y / (np.abs(y).max() + 1e-9) * 0.95
    y44 = resample_poly(y, 44100, sr)
    sf.write(args.out, y44.astype(np.float32), 44100, subtype="PCM_24")
    print("wrote %s  %.2f s  (%.0f BPM, %d notes, %d syllables)"
          % (args.out, len(y44) / 44100.0, args.tempo,
             len(note_info), len(note_info)))


if __name__ == "__main__":
    main()
