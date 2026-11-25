# 37 音色模型包 v1

37 是《重返未来：1999》的角色。**说话**使用 CosyVoice3 零样本（参考音频即可，无训练说话 TTS）；**唱歌**使用 RVC v2 40k 模型。模型权重不进入 Git，下载方法见仓库根目录 [`README.md`](../../README.md) 或 [`docs/models.md`](../../docs/models.md)。

## 内容

```text
37_voice_v1/
├── rvc_singing/
│   ├── v2_p1p2/                 # 推荐：p1+p2 全量，416 条 / 37.1 min
│   │   ├── 37_rvc40k_v2.pth
│   │   ├── added_IVF2292_Flat_nprobe_1_37_rvc40k_v2_v2.index
│   │   ├── trained_IVF2292_Flat_nprobe_1_37_rvc40k_v2_v2.index
│   │   ├── config.json
│   │   └── filelist.txt
│   └── v1_speech_only/          # 仅 p1，147 条 / 11.4 min，对照版本
├── reference_audio/             # 12 条英文参考（5–8 s，40 kHz）+ references.csv
└── dataset_selection/           # 逐条筛选记录与说话人排查
```

数据来源：B 站 `BV1MqNmeQEsA` 两个分 P（p01 21:23、p02 40:25，英文游戏语音）。

## 唱歌推理（RVC v2）

```bash
cd third_party/RVC
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0
PKG="$PWD/../../models/37_voice_v1/rvc_singing/v2_p1p2"
python -m infer.cli \
  --model "$PKG/37_rvc40k_v2.pth" \
  --index "$PKG/added_IVF2292_Flat_nprobe_1_37_rvc40k_v2_v2.index" \
  --input in.wav --output out.wav \
  --f0-method rmvpe --index-rate 0.85 --rms-mix-rate 0.25 --protect 0.30 --format wav
```

推荐 `index-rate 0.85`、`protect 0.30`；转换后用 `scripts/restore_consonants.py` 补辅音，再用 `scripts/remix_multitrack.py` 混音。

## 说话：CosyVoice3 零样本

```bash
python scripts/cosyvoice_zero_shot.py \
  --model_dir third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --ref models/37_voice_v1/reference_audio/ref_11_355Hz.wav \
  --text "Hello, nice to meet you." --out outputs/37.wav
```

`reference_audio/ref_11_355Hz.wav` 对 37 的平均相似度最高（0.9433）；`ref_01_374Hz.wav`、`ref_03_393Hz.wav` 也很干净。

## 备注

37 的训练集全部是语音，没有歌声样本；RVC 在长音、颤音和高音区的音质退化问题与改进实验见 [`docs/experiments/37_model_diagnosis.md`](../../docs/experiments/37_model_diagnosis.md)。
