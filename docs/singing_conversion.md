# 歌声转换

项目为两位目标音色训练了 **RVC v2 40k** 唱歌模型，并在后期加入辅音恢复、和声、电话染色与多轨混音脚本。另有一条 **Seed-VC 零样本歌声转换**实验线，用于对比 RVC 在“语音训练集上的歌声音质退化”问题。

## 1. 模型

### 1.1 东雪莲

| 文件 | 说明 |
|---|---|
| `models/dongxuelian_voice_v1/rvc_singing/dongxuelian_rvc40k.pth` | RVC v2 40k，200 epoch 提取的推理小模型 |
| `models/dongxuelian_voice_v1/rvc_singing/added_IVF2679_Flat_nprobe_1_dongxuelian_rvc40k_v2.index` | 推理用混合索引 |
| `models/dongxuelian_voice_v1/rvc_singing/config.json` | v2 40k，hop 400 |
| `models/dongxuelian_voice_v1/rvc_singing/filelist.txt` | 396 条训练清单 |

训练数据：`dataset_voice_clone/train_dongxuelian/wavs`（396 条，42.86 min；日语 134 + 中文 262）。推荐参数：

```text
--f0-method rmvpe --index-rate 0.60–0.80 --rms-mix-rate 0.25 --protect 0.45
```

### 1.2 37

| 文件 | 说明 |
|---|---|
| `models/37_voice_v1/rvc_singing/v2_p1p2/37_rvc40k_v2.pth` | p1+p2 全量（416 条 / 37.1 min），推荐 |
| `models/37_voice_v1/rvc_singing/v2_p1p2/added_IVF2292_Flat_nprobe_1_37_rvc40k_v2_v2.index` | 推理索引 |
| `models/37_voice_v1/rvc_singing/v1_speech_only/37_rvc40k.pth` | 仅 p1（147 条 / 11.4 min），对照 |

推荐参数：

```text
--f0-method rmvpe --index-rate 0.85 --rms-mix-rate 0.25 --protect 0.30
```

37 的原始数据全部是游戏语音，没有歌声训练样本；RVC 在长音、颤音与高音区的表现会明显下降。详细的声学诊断与 Seed-VC 对照见 [`experiments/37_model_diagnosis.md`](experiments/37_model_diagnosis.md)。

## 2. 训练 RVC

### 2.1 准备

```bash
# 第三方 RVC 仓库
bash scripts/setup_third_party.sh

# gptsovits 环境（RVC 与本项目共用）
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gptsovits
pip install -r third_party/RVC/requirements.txt
```

### 2.2 训练

通用脚本接受任意 wav 目录：

```bash
WAVS="$PWD/dataset_voice_clone/train_dongxuelian/wavs" \
EXP=dongxuelian_rvc40k \
GPU_LIST=0 \
EPOCHS=200 \
bash scripts/train_rvc_generic.sh
```

脚本流程：

1. `train.preprocess`：重采样到 40 kHz、切片；
2. `extract_f0`：rmvpe 提取 F0；
3. `extract_hubert_feature`：提取 HuBERT 特征；
4. 生成 `filelist.txt`；
5. `train.train`：v2 40k，默认 batch 8，200 epoch，每 20 epoch 保存；
6. `train_index`：生成检索索引。

产物在 `third_party/RVC/logs/<EXP>/`，把最终 `G_*.pth` 转成推理用 `*.pth` 并复制 `added_*.index` 到 `models/<voice>/rvc_singing/`。

## 3. RVC 推理

```bash
cd third_party/RVC
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gptsovits
export PYTHONPATH="$PWD"

PKG="$PWD/../../models/dongxuelian_voice_v1/rvc_singing"
python -m infer.cli \
  --model  "$PKG/dongxuelian_rvc40k.pth" \
  --index  "$PKG/added_IVF2679_Flat_nprobe_1_dongxuelian_rvc40k_v2.index" \
  --input  input.wav --output output.wav \
  --f0-method rmvpe --index-rate 0.75 --rms-mix-rate 0.25 \
  --protect 0.45 --format wav
```

37 把 `PKG` 换成 `models/37_voice_v1/rvc_singing/v2_p1p2`，`--index-rate 0.85 --protect 0.30`。

> 如果目标歌曲音域明显高于训练数据（37 的语音 F0 p90 ≈ 471 Hz），可以整首降 2 个半音再转换，音色稳定性通常更好；`--pitch` 只改人声音高，改调要连伴奏一起移调。

## 4. 歌声后处理链

RVC 输出在 4–8 kHz 会比源人声暗 3–7 dB，中文咬字容易发闷。项目常用的处理顺序：

```text
RVC 输出
  → restore_consonants.py（从源人声恢复 3.5–9 kHz 辅音）
  → add_harmony.py（可选：生成三度/五度和声）
  → radio_effect.py（可选：电话/合成器染色）
  → remix_multitrack.py（按原曲比例混合主唱/和声/伴奏）
```

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `scripts/restore_consonants.py` | 恢复辅音清晰度 | `--target` 控制 4–8 kHz 增益 |
| `scripts/add_harmony.py` | 生成自然音阶和声 | `--key`、`--mode` |
| `scripts/radio_effect.py` | 匹配电话/合成器音色 | `--amount`、`--drive` |
| `scripts/remix_multitrack.py` | 多轨混音 | 输入 lead / backing / instrumental |
| `scripts/polish_song.py` | Seed-VC 输出 EQ + 空气感 + 混音 | `--air-gain` |
| `scripts/breathify.py` | WORLD 气声增强 | `--gamma`、`--whisper` |

## 5. TTS → 歌声（可选实验）

`impose_melody.py` + `forced_align_wav2vec2.py` 支持把 CosyVoice3 的朗读结果按音符网格拉伸，再交给 RVC：

```bash
# 1) 用 CosyVoice3 生成朗读
python scripts/cosyvoice_zero_shot.py --model_dir ... \
  --ref dataset_voice_clone/reference_37/ref_11_355Hz.wav \
  --text "..." --out outputs/tts.wav

# 2) 用 wav2vec2 得到词级时间戳（或使用 Whisper JSON）
python scripts/forced_align_wav2vec2.py --wav outputs/tts.wav --text "..." --out-json outputs/tts.json

# 3) 按旋律计划拉伸到音符
python scripts/impose_melody.py --input outputs/tts.wav --plan plan.json --output outputs/melody.wav
```

这条路线对音准和节奏的控制有限，主要用于实验“朗读 → 歌唱”的可听效果，不替代真人演唱。

## 6. Seed-VC 实验

`third_party/seed-vc` 用于零样本歌声转换。项目在 37 上做过系统对照：用 37 的 EP 高音区做参考时，Seed-VC 的 CPP（约 0.437）和频谱平衡明显优于 RVC（约 0.267），但本地权重版本存在 key 形状不匹配，输出可能有染色。详细记录与 A/B 文件说明见 [`experiments/37_model_diagnosis.md`](experiments/37_model_diagnosis.md)。

## 7. 试听样例

`samples/singing/` 提供几位音色的短片段（仅用于演示，版权归原曲/原唱所有）。完整歌曲与中间产物体积较大，不进入 Git 仓库。
