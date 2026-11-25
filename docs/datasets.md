# 数据集

本项目的数据分为两部分：

1. **东雪莲声音克隆训练集**：396 条中日双语语音片段，用于 GPT-SoVITS 微调与 RVC 训练。
2. **零样本参考音频**：东雪莲、37 各 12 条，用于 GPT-SoVITS/CosyVoice3 的零样本音色参考。

> 训练 wav 体积较大，Git 仓库只保留清单、metadata 与参考音频；完整数据在 Release 资产
> `voice-clone-dataset-v1.0.zip` 中（内含 `dataset_voice_clone/` 与 `SHA256SUMS`）。

## 1. 东雪莲训练集

| 项目 | 数值 |
|---|---:|
| 片段数 | 396 |
| 总时长 | 42.86 min（2571.6 s） |
| 单条时长 | 1.5–10.0 s，平均 6.49 s |
| 语种 | 日语 134 条，中文 262 条 |
| 格式 | 32 kHz / 16-bit / 单声道 |

### 文件

```text
dataset_voice_clone/train_dongxuelian/
├── wavs/                 # Release 资产提供（396 个 wav）
├── gptsovits_v3.list     # 路径|说话人|语种|文本
└── metadata.csv          # 文件名、时长、采样率、语种、转写
```

### 来源与处理流程

- **源视频**
  - 东雪莲：B 站录播 `BV1ke4y1T7Zg`（约 79 min，日语为主）；另取中文杂谈回放 `BV1HeN4zkELe`（约 118 min）。
  - 37（《重返未来：1999》）：B 站语音合集 `BV1MqNmeQEsA` 两个分 P（英文游戏语音）。
- **人声分离**：Demucs（`htdemucs`）分离人声与伴奏/游戏音效，只保留人声轨。
- **VAD 切分**：`webrtcvad` 按 3–10 s 切分；连续说话超过 10 s 的段落按能量最低点二次切分，丢弃 < 3 s 尾片。
- **转写**：Whisper large-v3 自动识别语种并转写；中文经 OpenCC 转简体。
- **清洗**：剔除复读/幻觉、空白、噪声、口播套话；核对混合语种片段的语种标签。
- **选段**：按谱平坦度、时长、HNR 打分，控制中日比例，最终得到 396 条平衡训练集。

处理脚本见 [`scripts.md`](scripts.md) 中的 `vad_segments.py`、`segment_finetune.py`、`transcribe_finetune.py`、`merge_finetune_lists.py`。

### 训练清单格式

```text
wavs/seg_0001.wav|dongxuelian|ja|转写文本
```

字段依次为音频相对路径、说话人名、语种（`ja` / `zh`）、转写文本；路径相对于 `train_dongxuelian/`。

## 2. 零样本参考音频

| 目录 | 数量 | 采样率 | 说明 |
|---|---:|---|---|
| `dataset_voice_clone/reference_dongxuelian/` | 12 | 24 kHz | 15.5–19.1 s，来自日语杂谈 |
| `dataset_voice_clone/reference_37/` | 12 | 40 kHz | 5.1–8.2 s，英文游戏语音 |

每条参考的时长、F0 中位数、平坦度与相似度分数见各目录的 `references.csv`。

推荐参考：

- 东雪莲：`ref_02_3783.0s.wav`（综合相似度最高）
- 37：`ref_11_355Hz.wav`（对 37 平均相似度最高，0.9433）

## 3. 使用方式

### 下载完整训练数据

从 [Releases](https://github.com/iolingl/voice-clone/releases) 下载 `voice-clone-dataset-v1.0.zip`，解压合并到仓库根目录：

```bash
unzip voice-clone-dataset-v1.0.zip -d .
```

### GPT-SoVITS 微调

按 [`voice_cloning.md`](voice_cloning.md) 的步骤把 `gptsovits_v3.list` 与 `wavs/` 放进 GPT-SoVITS 工作目录做预处理。

### RVC 训练

`scripts/train_rvc_generic.sh` 接受任意 40 kHz（或会被 RVC 重采样）的 wav 目录，例如：

```bash
WAVS="$PWD/dataset_voice_clone/train_dongxuelian/wavs" \
EXP=dongxuelian_rvc40k GPU_LIST=0 EPOCHS=200 \
bash scripts/train_rvc_generic.sh
```

## 4. 版权与使用限制

音频来自公开视频平台，版权归原视频作者/权利方所有；本项目仅将其用于学习、研究和非商业演示。请勿将数据或训练出的音色用于商业用途、冒充他人或制作误导性内容。
