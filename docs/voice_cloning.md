# 说话声音克隆

本项目有两条说话克隆路线：

1. **GPT-SoVITS v3 微调**：东雪莲，396 条中日双语语料；音色稳定、可控，适合中文/日语朗读。
2. **CosyVoice3 零样本**：37（以及东雪莲的快速对照），几秒参考音频即可跨语种合成。

`demo/` 里的 Web Demo 同时提供这两条路线。

## 1. 数据准备

原始数据为 B 站录播/游戏语音，处理流程：

```text
源视频/音频
  → Demucs 人声分离
  → webrtcvad VAD 切分（3–10 s）
  → Whisper large-v3 转写（自动语种）
  → 清洗、按 HNR/平坦度/时长筛选
  → gptsovits_v3.list + metadata.csv
```

相关脚本：

| 脚本 | 作用 |
|---|---|
| `scripts/vad_segments.py` | 长录音 VAD 切分（保留所有 ≥ min-len 的片段） |
| `scripts/vad_stats.py` | 统计 VAD 片段时长分布 |
| `scripts/segment_finetune.py` | 按静音区间切分训练片段（限制条数，跳过开头 2 分钟） |
| `scripts/cluster_speakers.py` | 用 SV2TTS speaker embedding 聚类，剔除混入的其他说话人 |
| `scripts/transcribe_finetune.py` | Whisper large-v3 转写并输出 GPT-SoVITS list |
| `scripts/merge_finetune_lists.py` | 合并中/日文切片，生成 `gptsovits_v3.list` |
| `scripts/make_references.py` | 从长音频中挑选零样本参考片段 |

数据集统计与清单格式见 [`datasets.md`](datasets.md)。

## 2. GPT-SoVITS v3 微调（东雪莲）

### 2.1 安装

见 [`environment.md`](environment.md) 的 `gptsovits` 环境。GPT-SoVITS 仓库默认安装 `pretrained_models`：

```text
third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models/
├── gsv-v2final-pretrained/
├── chinese-hubert-base/
└── chinese-roberta-wwm-ext-large/
```

### 2.2 生成预处理配置

`scripts/make_gsv_configs.py` 会基于 GPT-SoVITS 的模板生成 `TEMP/tmp_s1.yaml` 与 `TEMP/tmp_s2.json`，其中预训练路径、实验名和保存目录均已指向 GPT-SoVITS 仓库：

```bash
export GPT_SOVITS_REPO="$PWD/third_party/GPT-SoVITS"
python scripts/make_gsv_configs.py
```

### 2.3 预处理

`scripts/run_gsv_preprocess.sh` 封装了 GPT-SoVITS 的 `1-gettext → 2-get-hubert-wav32k → 3-get-semantic`：

```bash
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export DATA="$PWD/dataset_voice_clone/train_dongxuelian"
export LIST="$DATA/gptsovits_v3.list"
export WAV_DIR="$DATA/wavs"
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_gsv_preprocess.sh
```

### 2.4 训练

在 GPT-SoVITS WebUI 或官方 CLI 中加载 `TEMP/tmp_s1.yaml`（GPT 分支）与 `TEMP/tmp_s2.json`（SoVITS 分支）训练。本项目使用的配置摘要：

| 分支 | 轮数 | 备注 |
|---|---:|---|
| GPT (s1) | 15 epoch | v3 数据集，batch 8，fp16 |
| SoVITS (s2) | 12 epoch | v3 数据集，batch 8，LoRA rank 32 |

产出的权重放在 `models/dongxuelian_voice_v1/gpt_sovits_speech/`：

- `dongxuelian3-e15.ckpt`
- `dongxuelian3_e12_s612.pth`

训练数据混合：日语 134 条 + 中文 262 条，共 42.86 分钟。语种标签在清单第三列（`ja`/`zh`）。

### 2.5 推理

`scripts/gsv_infer.py` 支持单条或多条文本：

```bash
export GPT_SOVITS_REPO="$PWD/third_party/GPT-SoVITS"
python scripts/gsv_infer.py \
  --gpt models/dongxuelian_voice_v1/gpt_sovits_speech/dongxuelian3-e15.ckpt \
  --sovits models/dongxuelian_voice_v1/gpt_sovits_speech/dongxuelian3_e12_s612.pth \
  --ref dataset_voice_clone/reference_dongxuelian/ref_02_3783.0s.wav \
  --prompt_text "<参考音频对应文本>" --prompt_lang ja \
  --lang zh --text "你好，这是声音克隆演示。" --outdir outputs/gsv
```

> **注意**：`--prompt_text` 必须与参考音频内容严格一致，否则会出现乱码/重复。Demo 使用的中文和日语参考音频分别为 `demo/assets/prompts/dongxuelian_prompt_zh.wav`、`dongxuelian_prompt_ja.wav`，其对应文本在 `demo/config.py` 的 `prompts` 字段中。

## 3. CosyVoice3 零样本

### 3.1 模型

```text
third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/
```

获取方式见 [`environment.md`](environment.md)。

### 3.2 推理脚本

```bash
# 单条文本
python scripts/cosyvoice_zero_shot.py \
  --model_dir third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --ref dataset_voice_clone/reference_37/ref_11_355Hz.wav \
  --text "The stars are beautiful tonight, aren't they?" --out outputs/37_test.wav

# 批量：多个参考 × 多行文本
python scripts/cosyvoice_batch.py \
  --model_dir third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --refs dataset_voice_clone/reference_37/ref_11_355Hz.wav \
         dataset_voice_clone/reference_37/ref_01_374Hz.wav \
  --texts_file texts.txt --outdir outputs/cosy --speed 1.0
```

两个脚本都会自动加上 CosyVoice3 的系统前缀：

```text
You are a helpful assistant.<|endofprompt|>
```

`cosyvoice_soft.py` 是柔声/气声版本，支持通过 WORLD 对输出做气声增强。

### 3.3 参考音频

37 推荐 `dataset_voice_clone/reference_37/ref_11_355Hz.wav`（5–8 s，英文游戏语音）；东雪莲可用 `reference_dongxuelian/ref_02_3783.0s.wav`。跨语种合成时参考音频语言与合成文本语言可以不同。

## 4. 评测

### 4.1 ASR 可懂度

用 Whisper large-v3 对合成音频转写，与输入文本比对，检查吞字、重复、乱码。示例：

```bash
python - <<'PY'
import whisper
m = whisper.load_model("large-v3", download_root="models/whisper")
print(m.transcribe("outputs/gsv/t1.wav", language="zh")["text"])
PY
```

### 4.2 说话人相似度

`scripts/speaker_similarity.py` 使用 SV2TTS speaker encoder 计算参考音频与合成音频的余弦相似度：

```bash
python scripts/speaker_similarity.py --ref ref.wav outputs/*.wav
```

需要先按上游 README 下载 `saved_models/default/encoder.pt`（`uv run` 会自动处理）。

### 4.3 本项目的参考结果

| 音色 | 评测方式 | 结果 |
|---|---|---:|
| 东雪莲（GPT-SoVITS） | 对 `ref_02_3783.0s.wav` 的相似度 | ≈ 0.89 |
| 37（CosyVoice3 零样本） | 对 `ref_11_355Hz.wav` 的相似度 | ≈ 0.78 |
| 东雪莲中文合成 | Whisper ASR | 「你好這是網頁演示生成的語音希望你喜歡」（正确） |
| 37 英文合成 | Whisper ASR | 「The stars are beautiful tonight, aren't they?」（正确） |

> 相似度会随着参考音频、文本与随机种子变化，以上仅为项目内一次固定测试的结果。
