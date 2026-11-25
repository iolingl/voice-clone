# 声音克隆 Web Demo

双音色声音克隆演示：

- **东雪莲**：GPT-SoVITS v3 微调，支持中文/日语；
- **37**：CosyVoice3 零样本，支持英语/中文/日语。

可以试听已有音频，也可以输入文本实时合成语音并下载 WAV。

## 功能

- **选择音色**：两个音色卡片，显示引擎、简介与参考音色试听，并标注引擎是否就绪。
- **文本合成**：输入文本（≤400 字）、选择语种、调节语速，点击「生成语音」即可播放并下载 WAV。
- **示例音频**：项目已有的合成结果，按音色分组直接试听。

## 架构

两个音色的推理依赖不同的 PyTorch 环境，为了互不干扰，采用「主应用 + 独立 worker」结构：

```text
浏览器 ──HTTP──> FastAPI 主应用 (env: gptsovits, 端口 8000)
                   ├─ 东雪莲：进程内加载 GPT-SoVITS 微调模型      （text -> wav）
                   └─ 37    ：HTTP 转发给 CosyVoice worker       （text -> wav）
                                    │
                         cosy_worker.py (env: voiceclone, 端口 8102)
                         进程内加载 CosyVoice3，零样本合成
```

| 文件 | 作用 |
|---|---|
| `app.py` | FastAPI 主应用：静态页 + `/api/*` + 东雪莲推理 + 转发 37 |
| `gsv_engine.py` | GPT-SoVITS 引擎封装（在 `gptsovits` 环境加载） |
| `cosy_worker.py` | CosyVoice3 零样本小服务（stdlib HTTP，在 `voiceclone` 环境加载） |
| `config.py` | 路径与音色配置（自动适配 `model/`+`dataset/` 或 `models/`+`dataset_voice_clone/`） |
| `static/` | 前端页面（HTML/CSS/JS，无需构建） |
| `samples/` | 示例音频 + `manifest.json` |
| `assets/prompts/` | 两个音色的参考音频、东雪莲 GPT-SoVITS 的参考文本音频 |
| `start.sh` | 一键启动两个进程 |

## 准备

1. **模型权重**：从 [Releases](https://github.com/iolingl/voice-clone/releases) 下载 `voice-clone-models-v1.0.zip`，解压到仓库根目录，确保存在：
   - `models/dongxuelian_voice_v1/gpt_sovits_speech/dongxuelian3-e15.ckpt`、`dongxuelian3_e12_s612.pth`
   - `models/dongxuelian_voice_v1/reference_audio/ref_02_3783.0s.wav`
   - `models/37_voice_v1/reference_audio/ref_11_355Hz.wav`
2. **两个 conda 环境**：
   - `gptsovits`：PyTorch + fastapi、uvicorn + GPT-SoVITS 仓库
   - `voiceclone`：PyTorch + torchaudio、soundfile + CosyVoice 仓库
   安装步骤见 `../docs/environment.md`。
3. **通用基座模型（不随仓库提供，必须自行下载）**：
   - CosyVoice3：`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`（约 9.1 GB），默认放在
     `third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`
   - GPT-SoVITS：预训练权重约 4.6 GB，默认放在 `third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models`
   如放在别处，用环境变量覆盖（见下）。

## 启动

```bash
cd demo
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"   # 可选，脚本会自动检测
export COSYVOICE_MODEL_DIR="$PWD/../third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
export GPT_SOVITS_REPO="$PWD/../third_party/GPT-SoVITS"
export CUDA_VISIBLE_DEVICES=0
export PORT=8000
bash start.sh
# 浏览器打开 http://localhost:8000 ，等待页面顶部两个状态点变绿
```

启动脚本会先用 `voiceclone` 环境拉起 CosyVoice worker（模型加载约 30–60 s），
再用 `gptsovits` 环境启动主应用（加载 GPT-SoVITS 约 20–40 s）。Ctrl+C 会一并关闭 worker。

### 可配置环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `CONDA_SH` | 自动检测（`conda info --base`，回退到 `$HOME/miniconda3` 等） | conda 初始化脚本 |
| `VOICE_ENV` | `gptsovits` | 主应用环境 |
| `COSY_ENV` | `voiceclone` | CosyVoice worker 环境 |
| `PORT` | `8000` | 主应用端口 |
| `COSY_WORKER_PORT` | `8102` | 内部 worker 端口 |
| `CUDA_VISIBLE_DEVICES` | `0` | 指定一张空闲 GPU |
| `COSYVOICE_MODEL_DIR` | `third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B` | CosyVoice3 目录 |
| `GPT_SOVITS_REPO` | `third_party/GPT-SoVITS` | GPT-SoVITS 仓库 |
| `COSY_DISABLE` | `0` | 设为 `1` 可跳过 CosyVoice（此时只有东雪莲可用） |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 两个引擎的可用状态 |
| GET | `/api/voices` | 音色列表（含可用性、语种、示例句） |
| GET | `/api/samples` | 示例音频清单 |
| POST | `/api/tts` | `{voice, text, lang, speed}` → `audio/wav` 字节流 |

`voice` 取 `dongxuelian` 或 `voice37`。响应头 `X-Gen-Seconds` 给出合成耗时。

## 音色说明

- **东雪莲**：GPT-SoVITS 微调，训练集 396 条中日双语（42.9 分钟）。
  中文/日语各用一段训练集内的参考音频（`assets/prompts/dongxuelian_prompt_zh.wav` /
  `dongxuelian_prompt_ja.wav`），`config.py` 中的 `prompts.zh.text`、`prompts.ja.text`
  与其音频内容严格一致。支持中文（默认）与日语。
- **37**：《重返未来：1999》角色，**未做微调**，用 7 秒英文游戏语音（`ref_11_355Hz.wav`）做 CosyVoice3
  零样本克隆。以英文效果最好，也支持中文/日语等语种（跨语种合成）。

## 常见问题

- **页面顶部某引擎一直红点**：看终端日志；多半是基座模型路径不对或 GPU 显存不足。
- **37 语音不可用但东雪莲正常**：CosyVoice worker 没起来，检查 `COSY_ENV`、`COSYVOICE_MODEL_DIR` 与端口占用。
- **首次跑 37 卡在下载**：CosyVoice 文本前端（wetext）首次会联网下载到 `~/.cache/modelscope`，需要能访问 ModelScope；之后可离线运行。
- **显存不足**：`CUDA_VISIBLE_DEVICES` 换一张空卡；两个模型合计约占 8–12 GB。
- **东雪莲输出乱码**：`config.py` 里 `prompts.zh.text` / `prompts.ja.text` 必须与对应参考音频内容一致，不要随意替换参考音频。

> 数据与音色仅用于学习、研究与演示，版权归原视频作者/权利方所有，请勿再分发或商用。
