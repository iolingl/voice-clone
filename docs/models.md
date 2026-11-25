# 模型清单

模型权重不放进 Git 仓库，统一打包在 GitHub Release：

- **`voice-clone-models-v1.0.zip`**：说话模型 + 唱歌模型 + 参考音频 + SHA256SUMS
- **`voice-clone-dataset-v1.0.zip`**：训练集 wav 与参考音频

下载后解压到仓库根目录即可合并到 `models/` 与 `dataset_voice_clone/`。

Release v1.0.0 资产校验（SHA256）：

| 资产 | 大小 | SHA256 |
|---|---:|---|
| `voice-clone-models-v1.0.zip` | 815 MB | `1ae295eee452b4b4dc3a21dfb8cafc536f8ef65c37077490dd7e263dc9cc98ec` |
| `voice-clone-dataset-v1.0.zip` | 146 MB | `da373278b1e3398270d8d37bae6e81b9e9ec686573a919e6710e1e303b8f652c` |

## 1. 东雪莲 `models/dongxuelian_voice_v1`

### 说话：GPT-SoVITS v3

| 文件 | 大小 | 说明 |
|---|---:|---|
| `gpt_sovits_speech/dongxuelian3-e15.ckpt` | 149 MB | GPT 分支，v3 平衡数据集，15 epoch |
| `gpt_sovits_speech/dongxuelian3_e12_s612.pth` | 82 MB | SoVITS 分支，v3，12 epoch |
| `gpt_sovits_speech/tmp_s1_v3.yaml` / `tmp_s2_v3.json` | — | 训练配置 |
| `gpt_sovits_speech/gptsovits_v3.list` | — | 396 条带转写清单 |

训练数据：`dataset_voice_clone/train_dongxuelian/`（396 条，42.86 min）。

### 唱歌：RVC v2 40k

| 文件 | 大小 | 说明 |
|---|---:|---|
| `rvc_singing/dongxuelian_rvc40k.pth` | 53 MB | 推理用小模型（200 epoch 提取） |
| `rvc_singing/added_IVF2679_Flat_nprobe_1_dongxuelian_rvc40k_v2.index` | 315 MB | 推理检索索引 |
| `rvc_singing/trained_IVF2679_Flat_nprobe_1_dongxuelian_rvc40k_v2.index` | 7.9 MB | 训练产出原始索引 |
| `rvc_singing/config.json` / `filelist.txt` | — | 训练配置与 396 条清单 |

推荐推理参数：`--f0-method rmvpe --index-rate 0.60-0.80 --rms-mix-rate 0.25 --protect 0.45`。详见 [`singing_conversion.md`](singing_conversion.md)。

### 参考音频

`dataset_voice_clone/reference_dongxuelian/` 12 条。推荐 `ref_02_3783.0s.wav`。

## 2. 37 `models/37_voice_v1`

37 没有训练说话 TTS 模型，说话使用 CosyVoice3 零样本（参考 `ref_11_355Hz.wav`）；唱歌使用 RVC v2 40k。

### 唱歌：RVC v2 40k

| 文件 | 大小 | 说明 |
|---|---:|---|
| `rvc_singing/v2_p1p2/37_rvc40k_v2.pth` | 53 MB | p1+p2 全量模型（416 条 / 37.1 min，推荐） |
| `rvc_singing/v2_p1p2/added_IVF2292_Flat_nprobe_1_37_rvc40k_v2_v2.index` | 270 MB | 推理检索索引 |
| `rvc_singing/v1_speech_only/37_rvc40k.pth` | 53 MB | 仅 p1（147 条 / 11.4 min），对照版本 |
| `rvc_singing/v1_speech_only/added_IVF723_Flat_nprobe_1_37_rvc40k_v2.index` | 85 MB | v1 索引 |

推荐 v2 参数：`--f0-method rmvpe --index-rate 0.85 --rms-mix-rate 0.25 --protect 0.30`。

训练数据来自 B 站 `BV1MqNmeQEsA` 两个分 P；p1 147 条（11.4 min）+ p2 269 条（25.7 min），清洗后 416 条。逐条筛选记录见 `models/37_voice_v1/dataset_selection/`。

### 参考音频

`dataset_voice_clone/reference_37/` 12 条。推荐 `ref_11_355Hz.wav`。

## 3. 通用基座模型

以下模型不随 Release 提供，请按 [`environment.md`](environment.md) 下载：

| 模型 | 用途 |
|---|---|
| CosyVoice3 `Fun-CosyVoice3-0.5B-2512` | 37 零样本 TTS |
| GPT-SoVITS `gsv-v2final-pretrained`、`chinese-hubert-base`、`chinese-roberta-wwm-ext-large` | 东雪莲 GPT-SoVITS 微调/推理 |
| RVC `assets/pretrained_v2/f0G40k.pth`、`f0D40k.pth` | RVC 训练初始化 |
| Whisper large-v3 | 转写与 ASR 评测 |
| wav2vec2-base-960h | 强制对齐 |
| SV2TTS（encoder/synthesizer/vocoder） | 说话人相似度评测 |

## 4. 校验

`models/*/SHA256SUMS` 覆盖对应模型目录内的权重与配置；下载 Release 后可在各模型目录执行：

```bash
cd models/dongxuelian_voice_v1 && sha256sum -c SHA256SUMS
cd ../37_voice_v1 && sha256sum -c SHA256SUMS
```
