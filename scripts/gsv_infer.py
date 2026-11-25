#!/usr/bin/env python
"""GPT-SoVITS v2/v3 批量推理：同一参考音色 + 多段中文文本。

用法示例：
  python scripts/gsv_infer.py --gpt GPT_weights_v2/xxx.ckpt --sovits SoVITS_weights_v2/xxx.pth \
      --ref data/finetune/wavs/seg_0136.wav --prompt_text "参考音频的日语转写" \
      --texts_file texts.txt --outdir data/outputs/gsv1
"""
import argparse
import os
import sys

from pathlib import Path
PROJECT_ROOT = Path(os.environ.get('VOICE_CLONE_ROOT', Path(__file__).resolve().parents[1]))
REPO = Path(os.environ.get('GPT_SOVITS_REPO', PROJECT_ROOT / 'third_party' / 'GPT-SoVITS'))
ORIG_CWD = os.getcwd()


def absp(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ORIG_CWD, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpt', required=True)
    ap.add_argument('--sovits', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--prompt_text', required=True)
    ap.add_argument('--prompt_lang', default='ja')
    ap.add_argument('--lang', default='zh')
    ap.add_argument('--text', default=None)
    ap.add_argument('--texts_file', default=None)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=15)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--is_half', type=int, default=1)
    ap.add_argument('--prefix', default='')
    args = ap.parse_args()

    if args.text:
        texts = [args.text]
    elif args.texts_file:
        with open(absp(args.texts_file), encoding='utf-8') as f:
            texts = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    else:
        raise SystemExit('need --text or --texts_file')

    gpt = absp(args.gpt)
    sovits = absp(args.sovits)
    ref = absp(args.ref)
    outdir = absp(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, 'GPT_SoVITS'))
    import numpy as np
    import soundfile as sf
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    configs = TTS_Config({
        'device': args.device,
        'is_half': bool(args.is_half),
        'version': 'v2',
        't2s_weights_path': gpt,
        'vits_weights_path': sovits,
        'cnhuhbert_base_path': f'{REPO}/GPT_SoVITS/pretrained_models/chinese-hubert-base',
        'bert_base_path': f'{REPO}/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
    })
    print(configs)
    tts = TTS(configs)
    tts.init_t2s_weights(gpt)
    tts.init_vits_weights(sovits)

    for i, text in enumerate(texts, 1):
        out = os.path.join(outdir, f'{args.prefix}t{i}.wav')
        inputs = {
            'text': text,
            'text_lang': args.lang,
            'ref_audio_path': ref,
            'prompt_text': args.prompt_text,
            'prompt_lang': args.prompt_lang,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'temperature': args.temperature,
            'speed_factor': args.speed,
            'seed': args.seed,
            'parallel_infer': True,
            'repetition_penalty': 1.35,
        }
        sr, audio = list(tts.run(inputs))[-1]
        sf.write(out, audio.astype(np.float32) / 32768.0, sr)
        print(f'[saved] {out}  sr={sr}  dur={len(audio)/sr:.2f}s')


if __name__ == '__main__':
    main()
