# 东雪莲 音色模型包 v1

东雪莲的两套模型：**GPT-SoVITS v3 说话** + **RVC v2 40k 唱歌**。模型权重不进入 Git，下载方法见仓库根目录 [`README.md`](../../README.md) 或 [`docs/models.md`](../../docs/models.md)。

## 内容

```text
dongxuelian_voice_v1/
├── gpt_sovits_speech/                # 说话模型（GPT-SoVITS v3）
│   ├── dongxuelian3-e15.ckpt         # GPT 分支，15 epoch，149 MB
│   ├── dongxuelian3_e12_s612.pth     # SoVITS 分支，12 epoch，82 MB
│   ├── tmp_s1_v3.yaml / tmp_s2_v3.json
│   └── gptsovits_v3.list             # 396 条训练清单
├── rvc_singing/                      # 唱歌模型（RVC v2 40k）
│   ├── dongxuelian_rvc40k.pth        # 推理小模型，53 MB
│   ├── added_IVF2679_Flat_nprobe_1_dongxuelian_rvc40k_v2.index  # 推理索引，315 MB
│   ├── trained_IVF2679_Flat_nprobe_1_dongxuelian_rvc40k_v2.index
│   ├── config.json
│   └── filelist.txt
└── reference_audio/                  # 12 条参考音频（15–19 s，24 kHz）+ references.csv
```

训练数据见 `dataset_voice_clone/train_dongxuelian/`（396 条，42.86 min，日语 134 + 中文 262）。

## 说话推理（GPT-SoVITS）

```bash
export GPT_SOVITS_REPO="$PWD/third_party/GPT-SoVITS"
python scripts/gsv_infer.py \
  --gpt models/dongxuelian_voice_v1/gpt_sovits_speech/dongxuelian3-e15.ckpt \
  --sovits models/dongxuelian_voice_v1/gpt_sovits_speech/dongxuelian3_e12_s612.pth \
  --ref models/dongxuelian_voice_v1/reference_audio/ref_02_3783.0s.wav \
  --prompt_text "<参考音频对应文本>" --prompt_lang ja \
  --lang zh --text "要说的中文" --outdir outputs
```

`prompt_text` 必须与参考音频内容一致；参考音频可用 `reference_audio/ref_02_3783.0s.wav`（综合相似度最高）。

## 唱歌推理（RVC）

```bash
cd third_party/RVC
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0
PKG="$PWD/../../models/dongxuelian_voice_v1/rvc_singing"
python -m infer.cli \
  --model "$PKG/dongxuelian_rvc40k.pth" \
  --index "$PKG/added_IVF2679_Flat_nprobe_1_dongxuelian_rvc40k_v2.index" \
  --input in.wav --output out.wav \
  --f0-method rmvpe --index-rate 0.75 --rms-mix-rate 0.25 --protect 0.45 --format wav
```

转换后用 `scripts/restore_consonants.py` 恢复 4–8 kHz 辅音，详见 [`docs/singing_conversion.md`](../../docs/singing_conversion.md)。
