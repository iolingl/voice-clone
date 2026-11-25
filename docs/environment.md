# 环境配置

本项目包含多条相互独立的模型路线，建议按用途分开安装 conda 环境；不同 PyTorch 版本不要混用。

## 1. 硬件与系统

| 项目 | 建议 |
|---|---|
| 系统 | Linux x86_64（Ubuntu 20.04+） |
| GPU | NVIDIA，显存 ≥ 12 GB（Demo 同时加载两个模型，建议 16 GB 以上） |
| 驱动 | 支持 CUDA 12.1+ |
| 磁盘 | ≥ 50 GB（含第三方仓库、基座模型、数据集与输出） |

所有命令默认在仓库根目录执行：

```bash
git clone https://github.com/iolingl/voice-clone.git
cd voice-clone
export VOICE_CLONE_ROOT="$(pwd)"
```

## 2. 第三方仓库

运行：

```bash
bash scripts/setup_third_party.sh
```

该脚本会克隆并切换到本项目验证过的 commit（见 [`third_party.md`](third_party.md)）：

```text
third_party/GPT-SoVITS
third_party/CosyVoice
third_party/RVC
third_party/seed-vc
```

也可以手动克隆到 `third_party/` 下；脚本中的路径都可以用环境变量覆盖（见第 5 节）。

## 3. conda 环境

### 3.1 `gptsovits`：GPT-SoVITS + RVC + Demo 主应用

```bash
# GPT-SoVITS 官方安装
git clone https://github.com/RVC-Boss/GPT-SoVITS.git third_party/GPT-SoVITS
cd third_party/GPT-SoVITS
conda create -n gptsovits python=3.10 -y
conda activate gptsovits
bash install.sh --device CU128 --source HF      # 国内网络可用 --source ModelScope
pip install fastapi uvicorn soundfile

# RVC（使用同一环境即可；RVC 的依赖以其 requirements.txt 为准）
cd ../RVC
pip install -r requirements.txt
```

### 3.2 `voiceclone`：CosyVoice3 零样本

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
cd third_party/CosyVoice
conda create -n voiceclone python=3.10 -y
conda activate voiceclone
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
pip install -e .
```

### 3.3 `learn`：数据预处理（可选）

`webrtcvad`、`librosa`、`soundfile`、`pyworld`、`scipy` 等数据处理依赖建议单独放在 `learn` 环境：

```bash
conda create -n learn python=3.10 -y
conda activate learn
pip install webrtcvad librosa soundfile pyworld scipy numpy
```

## 4. 通用基座模型（不随仓库/Release 提供）

| 模型 | 用途 | 获取方式 |
|---|---|---|
| CosyVoice3 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`（约 9.1 GB） | 37 零样本 TTS | `python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"`（在 `third_party/CosyVoice` 下执行；海外可用 HuggingFace） |
| GPT-SoVITS 预训练权重（约 4.6 GB） | 东雪莲 GPT-SoVITS 微调 | `third_party/GPT-SoVITS/install.sh` 自动下载到 `GPT_SoVITS/pretrained_models`；至少需要 `gsv-v2final-pretrained/`、`chinese-hubert-base/`、`chinese-roberta-wwm-ext-large/` |
| Whisper large-v3（约 3.1 GB） | 数据转写与 ASR 验证 | 下载 `large-v3.pt` 放到 `models/whisper/large-v3.pt`（或修改脚本路径） |
| wav2vec2-base-960h（约 360 MB） | 歌词/文本强制对齐 | 下载到 `assets/wav2vec2_base_960h`，或设置 `WAV2VEC2_DIR` |
| SV2TTS 预训练（encoder/synthesizer/vocoder） | 说话人相似度评测 | `uv run --extra cuda` 首次会自动下载；也可按上游 README 手动下载 |

## 5. 可用环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VOICE_CLONE_ROOT` | 脚本所在仓库根目录 | 数据/输出根目录 |
| `GPT_SOVITS_REPO` | `$VOICE_CLONE_ROOT/third_party/GPT-SoVITS` | GPT-SoVITS 仓库 |
| `COSYVOICE_REPO` | `$VOICE_CLONE_ROOT/third_party/CosyVoice` | CosyVoice 仓库 |
| `RVC_REPO` | `$VOICE_CLONE_ROOT/third_party/RVC` | RVC 仓库 |
| `CONDA_SH` | 自动检测 `conda info --base` / `$HOME/miniconda3` | conda 初始化脚本 |
| `COSYVOICE_MODEL_DIR` | `$VOICE_CLONE_ROOT/third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B` | CosyVoice3 模型目录 |
| `CUDA_VISIBLE_DEVICES` | 各脚本默认 0 | 指定可用 GPU |
| `VOICE_DEMO_OUTPUT` | `demo/outputs` | Demo 输出目录 |

## 6. Demo 启动

```bash
cd demo
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export COSYVOICE_MODEL_DIR="$VOICE_CLONE_ROOT/third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
export GPT_SOVITS_REPO="$VOICE_CLONE_ROOT/third_party/GPT-SoVITS"
export CUDA_VISIBLE_DEVICES=0
bash start.sh
```

打开 <http://localhost:8000>。只使用东雪莲时可以 `COSY_DISABLE=1 bash start.sh` 跳过 CosyVoice worker。

## 7. 常见问题

- **找不到 conda**：显式设置 `CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh`。
- **CUDA out of memory**：两个模型约 8–12 GB；换一张空闲 GPU，或先只启动一个引擎。
- **GPT-SoVITS 乱码**：`prompt_text` 必须严格对应参考音频内容；参考音频与文本的对应关系见 `models/dongxuelian_voice_v1/README.md`。
- **CosyVoice 首次下载卡住**：首次运行会从 ModelScope 下载文本前端缓存，之后可离线。
- **RVC 报错找不到 `assets/pretrained_v2`**：确认在 `third_party/RVC` 目录下运行，或检查 RVC 安装是否完整。
