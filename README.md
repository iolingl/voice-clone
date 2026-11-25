# voice-clone

多语种个性化**声音克隆**与**歌声转换**实验项目。围绕两位目标音色——虚拟主播「东雪莲」与游戏角色「37」——搭建了从数据处理、模型训练、推理评测到 Web Demo 的完整流程。

> 本项目仅用于学习与研究。音频与音色版权归原视频作者/权利方所有，请勿用于商业用途。

## 功能概览

| 方向 | 实现 | 说明 |
|---|---|---|
| 零样本声音克隆 | CosyVoice3 (`Fun-CosyVoice3-0.5B`) | 几秒参考音频即可跨语种合成，37 音色的默认方案 |
| 微调声音克隆 | GPT-SoVITS v3 | 东雪莲 396 条中日双语语料（42.86 分钟）微调 |
| 歌声转换 | RVC v2 40k | 两位音色各一套唱歌模型；支持训练、推理与后处理 |
| 歌声转换（实验） | Seed-VC | 零样本歌声转换对照实验，见 `docs/experiments/37_model_diagnosis.md` |
| Web Demo | FastAPI + 原生 JS | 双音色试听与文本合成，见 [`demo/`](demo/) |
| 评测 | Whisper large-v3 + SV2TTS speaker encoder | ASR 可懂度与说话人相似度 |

## 技术路线

```text
说话克隆
├── 东雪莲：Demucs 分离 → VAD 切分 → Whisper 转写 → GPT-SoVITS v3 微调
└── 37    ：游戏语音 → 参考音频筛选 → CosyVoice3 零样本跨语种合成

歌声转换
├── 数据：B 站歌曲/游戏语音 → BS-RoFormer / Demucs 分离 → 切片
├── 训练：RVC v2 40k（rmvpe F0 + HuBERT 特征）
├── 推理：RVC / Seed-VC → 辅音恢复 → 和声/染色 → 混音
└── 可选：CosyVoice3 朗读 → WORLD 旋律贴合 → RVC（TTS→歌声）
```

## 仓库结构

```text
.
├── README.md
├── docs/                      # 技术文档、模型卡、实验记录
│   ├── voice_cloning.md       # 说话克隆：数据处理、GPT-SoVITS、CosyVoice3、评测
│   ├── singing_conversion.md  # 歌声转换：RVC 训练/推理、后处理、Seed-VC
│   ├── datasets.md            # 数据集说明
│   ├── models.md              # 模型清单与 Release 下载
│   ├── environment.md         # 环境与第三方仓库配置
│   ├── scripts.md             # 脚本索引
│   └── experiments/           # 37 模型诊断、歌曲目录等实验笔记
├── scripts/                   # 数据处理 / 训练 / 推理 / 后处理脚本
├── demo/                      # 双音色 Web Demo
├── dataset_voice_clone/       # 数据集说明、清单与零样本参考音频
├── models/                    # 模型卡、训练配置与文件清单（权重见 Release）
├── encoder/ synthesizer/ vocoder/ toolbox/ utils/   # SV2TTS（上游 RTVC）代码
└── samples/                   # 试听样例（含歌声转换片段）
```

## 快速开始

### 1. 克隆仓库并下载 Release 资产

```bash
git clone https://github.com/iolingl/voice-clone.git
cd voice-clone
```

从 [Releases](https://github.com/iolingl/voice-clone/releases) 下载：

| 资产 | 内容 | 解压位置 |
|---|---|---|
| `voice-clone-models-v1.0.zip` | 东雪莲 GPT-SoVITS / RVC 权重、37 RVC 权重、参考音频与 SHA256SUMS | 解压到仓库根目录（合并 `models/`） |
| `voice-clone-dataset-v1.0.zip` | 396 条东雪莲训练 wav、两套零样本参考音频 | 解压到仓库根目录（合并 `dataset_voice_clone/`） |

`models/` 中的 README、训练配置和 filelist 已随 Git 仓库提供，权重文件不进入 Git。

### 2. 准备第三方仓库

```bash
bash scripts/setup_third_party.sh
```

或按 [`docs/third_party.md`](docs/third_party.md) 手动克隆并切换到记录的 commit。主要依赖：

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 说话克隆微调/推理
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — 零样本声音克隆
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — 歌声转换
- [Seed-VC](https://github.com/Plachtaa/seed-vc) — 歌声转换对照实验

### 3. 配置环境

两个 TTS 推理环境分别负责不同模型，详见 [`docs/environment.md`](docs/environment.md)：

```bash
# 示例：启动 Demo
cd demo
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export COSYVOICE_MODEL_DIR="/path/to/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
export GPT_SOVITS_REPO="/path/to/GPT-SoVITS"
export CUDA_VISIBLE_DEVICES=0
bash start.sh
# 打开 http://localhost:8000
```

### 4. 常用入口

| 任务 | 命令/文档 |
|---|---|
| 说话克隆推理（GPT-SoVITS） | `python scripts/gsv_infer.py --help` |
| 零样本克隆（CosyVoice3） | `python scripts/cosyvoice_zero_shot.py --help` |
| RVC 训练 | `bash scripts/train_rvc_generic.sh`（见 [`docs/singing_conversion.md`](docs/singing_conversion.md)） |
| RVC 推理 | `third_party/RVC` 的 `python -m infer.cli`（见模型卡） |
| 歌曲后处理 | `restore_consonants.py` → `add_harmony.py` / `radio_effect.py` → `remix_multitrack.py` |
| 说话人相似度 | `python scripts/speaker_similarity.py --ref ref.wav out/*.wav` |

## 数据与模型

- 数据集：`dataset_voice_clone/`，包含东雪莲 396 条中日双语训练集（42.86 分钟）、两位音色的零样本参考音频。详见 [`docs/datasets.md`](docs/datasets.md)。
- 模型：东雪莲 GPT-SoVITS v3 说话模型 + RVC v2 40k 唱歌模型；37 RVC v2 40k 唱歌模型 + CosyVoice3 零样本参考。详见 [`docs/models.md`](docs/models.md) 和 `models/*/README.md`。
- 通用基座模型（CosyVoice3、GPT-SoVITS 预训练权重、Whisper、wav2vec2）不随仓库或 Release 提供，需按 [`docs/environment.md`](docs/environment.md) 自行下载。

## 文档

- [说话克隆技术报告](docs/voice_cloning.md)
- [歌声转换技术报告](docs/singing_conversion.md)
- [数据集说明](docs/datasets.md)
- [模型清单](docs/models.md)
- [环境配置](docs/environment.md)
- [脚本索引](docs/scripts.md)
- [37 模型唱歌音色诊断](docs/experiments/37_model_diagnosis.md)
- [歌曲目录与实验记录](docs/experiments/song_catalog.md)

## 说明

- 本仓库基于 [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)（MIT License）的代码结构，SV2TTS 部分用于说话人编码/相似度评测；上游许可见 [LICENSE](LICENSE)。
- GPT-SoVITS、CosyVoice、RVC、Seed-VC 均为各自作者的开源项目，请遵守其许可证。
- 所有音频、模型与音色仅用于学习、研究和非商业演示，版权归原视频作者/权利方所有。
