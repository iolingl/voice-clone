#!/usr/bin/env python
"""过滤中文切片转写结果，和日语数据合并成 GPT-SoVITS v3 训练 list。

输出目录结构（供 run_gsv_preprocess.sh 使用，需单一 wav 目录）：
  <wav_out>/seg_XXXX.wav      日语（软链）
  <wav_out>/zh_seg_XXXX.wav   中文（软链）
  <out_list>
"""
import argparse
import json
import os
import re
import shutil
import sys

import opencc

CJK = re.compile(r'[一-鿿぀-ヿ]')
WORD = re.compile(r'[A-Za-z0-9]')
BAD = ('字幕', '感谢观看', '订阅', '频道', '点击', '关注', '点赞', '投币')
REPEAT = re.compile(r'(.)\1{4,}')


def text_stats(t):
    stripped = re.sub(r'[\s，。！？、,.!?;:;:…—~～「」『』"“”\'（）()\[\]]', '', t)
    return len(stripped), len(CJK.findall(stripped)), len(WORD.findall(stripped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zh-json', default='data/finetune_zh/zh_transcript_raw.json')
    ap.add_argument('--zh-wav-dir', default='data/finetune_zh/wavs')
    ap.add_argument('--ja-list', default='data/finetune/gptsovits_final.list')
    ap.add_argument('--wav-out', default='data/finetune_all/wavs')
    ap.add_argument('--out-list', default='data/finetune_all/gptsovits_v3.list')
    ap.add_argument('--max-zh-min', type=float, default=None)
    args = ap.parse_args()

    cc = opencc.OpenCC('t2s')
    os.makedirs(args.wav_out, exist_ok=True)

    rows = json.load(open(args.zh_json, encoding='utf-8'))
    seen = {}
    kept, dropped = [], []
    for r in rows:
        txt = r['text'].strip()
        nchar, ncjk, nword = text_stats(txt)
        reasons = []
        if r['lang'] not in ('zh', 'ja'):
            reasons.append(f"lang={r['lang']}")
        if r['no_speech_prob'] > 0.6:
            reasons.append(f"nsp={r['no_speech_prob']}")
        if not (1.2 <= r['cps'] <= 9):
            reasons.append(f"cps={r['cps']}")
        if nchar < 5:
            reasons.append('too_short')
        if ncjk < 2:
            reasons.append('no_cjk')
        if REPEAT.search(txt):
            reasons.append('repeat')
        if any(b in txt for b in BAD):
            reasons.append('bad_word')
        if reasons:
            dropped.append((r['file'], ','.join(reasons), txt))
            continue
        key = re.sub(r'[\s，。！？、,.!?;:]', '', txt)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 2:
            dropped.append((r['file'], 'dup', txt))
            continue
        kept.append(r)

    # 时长上限（如果指定），按转录质量优先保留
    kept.sort(key=lambda r: (r['cps'] < 2.0, -min(r['dur'], 8), r['no_speech_prob']))
    if args.max_zh_min:
        acc, sel = 0.0, []
        for r in kept:
            if acc + r['dur'] / 60 > args.max_zh_min:
                continue
            sel.append(r); acc += r['dur'] / 60
        kept = sel

    lines_zh = []
    for r in kept:
        txt = cc.convert(r['text']).strip()
        dst = 'zh_' + r['file']
        src = os.path.join(args.zh_wav_dir, r['file'])
        link = os.path.join(args.wav_out, dst)
        if os.path.lexists(link):
            os.remove(link)
        shutil.copyfile(src, link)
        lines_zh.append(f'{os.path.abspath(link)}|dongxuelian|{r["lang"]}|{txt}')

    lines_ja = []
    for ln in open(args.ja_list, encoding='utf-8'):
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split('|')
        src = os.path.abspath(parts[0])
        base = os.path.basename(src)
        link = os.path.join(args.wav_out, base)
        if os.path.lexists(link) and os.path.realpath(link) != os.path.realpath(src):
            base = 'ja_' + base
            link = os.path.join(args.wav_out, base)
        if not os.path.lexists(link):
            shutil.copyfile(src, link)
        lines_ja.append('|'.join([os.path.abspath(link)] + parts[1:]))

    with open(args.out_list, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_ja + lines_zh) + '\n')

    dur_zh = sum(r['dur'] for r in kept) / 60
    print(f'ja={len(lines_ja)}  zh={len(lines_zh)}  zh_min={dur_zh:.1f}  dropped={len(dropped)}')
    print('dropped reasons:')
    from collections import Counter
    for reason, n in Counter(d[1] for d in dropped).most_common():
        print(f'  {reason}: {n}')
    print('sample zh:')
    for ln in lines_zh[:8]:
        print('  ', ln.split('|', 3)[-1])
    print('list ->', args.out_list)


if __name__ == '__main__':
    main()
