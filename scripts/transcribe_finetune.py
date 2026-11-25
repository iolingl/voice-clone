#!/usr/bin/env python
"""用 Whisper large-v3 转写微调切片，自动识别语种，输出 GPT-SoVITS list。

用法: python scripts/transcribe_finetune.py <wav_dir> <out.list> [--lang zh|auto]
"""
import argparse
import json
import os
import re
import sys

import librosa
import whisper

PUNCT = re.compile(r'[，。！？、,.!?;:;:\s」』"“”\'（）()\[\]…—~～]')


def clean_text(t):
    t = t.strip()
    t = re.sub(r'\s+', ' ', t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav_dir')
    ap.add_argument('out_list')
    ap.add_argument('--lang', default='auto')
    ap.add_argument('--model', default='large-v3')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.wav_dir) if f.endswith('.wav'))
    print(f'{len(files)} clips', flush=True)
    model = whisper.load_model(args.model, device=args.device)
    lang_opt = None if args.lang == 'auto' else args.lang

    rows = []
    for i, fn in enumerate(files, 1):
        p = os.path.join(args.wav_dir, fn)
        dur = librosa.get_duration(path=p)
        r = model.transcribe(p, language=lang_opt, temperature=0,
                             condition_on_previous_text=False)
        text = clean_text(r['text'])
        lang = r.get('language', 'zh')
        nsp = max((s.get('no_speech_prob', 0.0) for s in r['segments']), default=0.0)
        nchar = len(PUNCT.sub('', text))
        cps = nchar / max(dur, 1e-6)
        rows.append({'file': fn, 'dur': round(dur, 2), 'lang': lang,
                     'text': text, 'no_speech_prob': round(nsp, 3),
                     'cps': round(cps, 2)})
        print(f'[{i}/{len(files)}] {fn} {lang} {dur:.1f}s nsp={nsp:.2f} cps={cps:.1f} | {text}',
              flush=True)

    root = os.path.dirname(os.path.abspath(args.out_list))
    with open(os.path.join(root, 'zh_transcript_raw.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f'wrote {root}/zh_transcript_raw.json', flush=True)


if __name__ == '__main__':
    main()
