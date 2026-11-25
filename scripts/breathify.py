"""Increase breathiness of a wav with WORLD vocoder (aperiodicity boost + optional whisper mix)."""
import argparse, numpy as np, soundfile as sf, pyworld as pw

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--inp', required=True)
    ap_.add_argument('--out', required=True)
    ap_.add_argument('--gamma', type=float, default=0.7, help='<1 increases aperiodicity')
    ap_.add_argument('--whisper', type=float, default=0.0, help='whisper mix ratio 0..1')
    ap_.add_argument('--rms_db', type=float, default=-24.0)
    a = ap_.parse_args()
    y, sr = sf.read(a.inp, dtype='float64')
    if y.ndim > 1: y = y.mean(1)
    f0, t = pw.harvest(y, sr, f0_floor=60, f0_ceil=600, frame_period=5)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    ap2 = 1.0 - np.power(1.0 - np.clip(ap, 0, 1), a.gamma)
    # extra aperiodicity above 4 kHz, where breath noise lives
    nfft_bins = sp.shape[1]
    freqs = np.linspace(0, sr / 2, nfft_bins)
    hf = freqs >= 4000
    ap2[:, hf] = 1.0 - np.power(1.0 - ap2[:, hf], 1.3)
    out = pw.synthesize(f0, sp, np.clip(ap2, 0, 0.999), sr, frame_period=5)
    if a.whisper > 0:
        wh = pw.synthesize(np.zeros_like(f0), sp, np.ones_like(ap) * 0.999, sr, frame_period=5)
        out = (1 - a.whisper) * out + a.whisper * wh
    out = out[:len(y)]
    rms = np.sqrt((out ** 2).mean()) + 1e-12
    out *= (10 ** (a.rms_db / 20)) / rms
    sf.write(a.out, out.astype(np.float32), sr)
    print('wrote', a.out, f'{len(out)/sr:.2f}s', 'gamma', a.gamma, 'whisper', a.whisper)

if __name__ == '__main__':
    main()
