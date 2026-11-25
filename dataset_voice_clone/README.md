# 声音克隆数据集

本目录汇总项目用于**说话声音克隆**的数据：东雪莲 GPT-SoVITS 微调训练集，以及东雪莲/37 的零样本参考音频。歌声转换使用的数据与训练方法见 [`../docs/singing_conversion.md`](../docs/singing_conversion.md)。

> 训练 wav 共 396 条（174 MB），不进入 Git 仓库；请从 Release 资产 `voice-clone-dataset-v1.0.zip` 下载并解压到仓库根目录。

## 目录结构

```text
dataset_voice_clone/
├── README.md
├── train_dongxuelian/
│   ├── wavs/                   396 条语音片段（Release 提供，32 kHz / 16-bit / 单声道）
│   ├── gptsovits_v3.list       训练清单（相对路径）
│   └── metadata.csv            每条音频的时长、采样率、语种、转写文本
├── reference_dongxuelian/      零样本参考音频 12 条（24 kHz / 16-bit）+ references.csv
└── reference_37/               零样本参考音频 12 条（40 kHz / 16-bit）+ references.csv
```

## 统计

| 项目 | 数值 |
|---|---:|
| 训练片段 | 396 条 |
| 训练总时长 | 42.86 min（2571.6 s） |
| 单条时长 | 1.5 – 10.0 s，平均 6.49 s |
| 语种 | 日语 134 条，中文 262 条 |
| 格式 | 32 kHz / 16-bit / 单声道 |
| 零样本参考（东雪莲） | 12 条，15.5–19.1 s |
| 零样本参考（37） | 12 条，5.1–8.2 s |

## 数据来源与处理流程

1. **源视频**
   - 东雪莲：B 站录播 `BV1ke4y1T7Zg`（约 79 min，日语为主）；补充中文杂谈回放 `BV1HeN4zkELe`（约 118 min）。
   - 37（《重返未来：1999》）：B 站语音合集 `BV1MqNmeQEsA` 两个分 P（p01 21:23、p02 40:25，英文语音）。
2. **人声分离**：Demucs 分离人声与伴奏/游戏音效，只保留人声轨。
3. **VAD 切分**：`webrtcvad` 按 3–10 s 切分；连续说话超过 10 s 的段落按能量最低点二次切分，丢弃 < 3 s 尾片。
4. **转写**：Whisper large-v3 自动识别语种并转写；中文文本经 OpenCC 转简体。
5. **清洗**：剔除复读/幻觉、纯空白、噪声、口播套话；对混合语种片段逐条核对语种标签。
6. **选段与配比**：按谱平坦度、时长、HNR 打分，控制中/日比例，最终得到 396 条平衡训练集。

## 清单格式

`train_dongxuelian/gptsovits_v3.list` 每行：

```text
wavs/seg_0001.wav|dongxuelian|ja|转写文本
```

字段依次为：音频相对路径、说话人名、语种（`ja` / `zh`）、转写文本。路径相对于 `train_dongxuelian/`。

## 使用方式

**GPT-SoVITS 微调**（在 `gptsovits` 环境，需要 GPT-SoVITS 仓库）：

```bash
cd dataset_voice_clone/train_dongxuelian
# 把 gptsovits_v3.list 与 wavs/ 复制到 GPT-SoVITS 工作目录，按其 README 做
# 1-gettext → 2-get-hubert-wav32k → 3-get-semantic 预处理后训练
```

**CosyVoice3 / GPT-SoVITS 零样本推理**：直接使用任意参考音频。东雪莲推荐 `reference_dongxuelian/ref_02_3783.0s.wav`（17.4 s）；37 推荐 `reference_37/ref_11_355Hz.wav`（5.1 s）。

## 说明

- 数据仅用于学习、研究与非商业演示，版权归原视频作者/权利方所有；请勿再分发或商用。
- 训练音频与 `models/dongxuelian_voice_v1/` 的 GPT-SoVITS、RVC 模型配套。
- 37 的参考音频只用于零样本声音克隆；本课题没有用 37 的数据训练说话 TTS 模型。
