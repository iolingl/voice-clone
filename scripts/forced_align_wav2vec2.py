#!/usr/bin/env python
"""Word-level forced alignment with wav2vec2-base-960h (transformers).

Whisper word timestamps around pauses are loose enough to break syllable-level
time warping; this aligns the exact transcript instead.

Model: https://modelscope.cn/models/AI-ModelScope/wav2vec2-base-960h
Output JSON: [{"word": "twinkle", "start": 0.12, "end": 0.51}, ...]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from transformers import AutoProcessor, Wav2Vec2ForCTC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--model-dir",
                    default=str(Path(os.environ.get("VOICE_CLONE_ROOT", Path(__file__).resolve().parents[1])) / "assets" / "wav2vec2_base_960h"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    blank = model.config.pad_token_id
    if blank is None:
        blank = vocab.get("<pad>", 0)
    print("vocab size", len(vocab), "blank", blank, flush=True)

    wav, sr = torchaudio.load(args.wav)
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    n_frames = log_probs.shape[1]

    words = [w.strip().upper() for w in args.text.replace(".", "").replace(",", "").split()]
    tokens = []
    word_offsets = []
    for wi, w in enumerate(words):
        w = "".join(ch for ch in w if ch in vocab)
        if not w:
            continue
        start = len(tokens)
        tokens.extend(vocab[ch] for ch in w)
        word_offsets.append((wi, start, len(tokens)))
        tokens.append(vocab["|"])
    if tokens[-1] == vocab["|"]:
        tokens.pop()
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    input_lengths = torch.tensor([n_frames], dtype=torch.int32)
    target_lengths = torch.tensor([len(tokens)], dtype=torch.int32)
    alignment, scores = F.forced_align(
        log_probs.cpu(), targets.cpu(), input_lengths, target_lengths, blank=blank
    )

    # per-target-token spans (keep repeated letters separated by blanks)
    al = alignment[0].numpy()
    spans = []
    prev_blank = True
    prev_tok = -1
    for f, tok in enumerate(al):
        if tok == blank:
            prev_blank = True
            prev_tok = -1
            continue
        if prev_blank or tok != prev_tok:
            spans.append([f, f])
        else:
            spans[-1][1] = f
        prev_blank = False
        prev_tok = tok
    if len(spans) != len(tokens):
        print("WARN: %d spans for %d tokens" % (len(spans), len(tokens)))

    hop = 320.0 / 16000.0
    out = []
    for wi, a, b in word_offsets:
        if b - 1 >= len(spans):
            continue
        out.append({
            "word": words[wi].lower(),
            "start": round(spans[a][0] * hop, 3),
            "end": round((spans[b - 1][1] + 1) * hop, 3),
        })
    json.dump(out, open(args.out_json, "w"), indent=1)
    print("wrote %d words: %s ... %s" % (len(out), out[0], out[-1]))


if __name__ == "__main__":
    main()
