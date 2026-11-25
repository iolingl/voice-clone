# 脚本索引

所有脚本默认使用仓库根目录作为工作路径；可通过 `VOICE_CLONE_ROOT` 覆盖。Python 脚本依赖各自 conda 环境，见 [`environment.md`](environment.md)。

## 数据准备

| 脚本 | 说明 |
|---|---|
| `vad_segments.py` | 长录音按 VAD 切分为语句片段 |
| `vad_stats.py` | 统计 VAD 片段数量与时长分布 |
| `segment_finetune.py` | 按静音区间切分训练数据（跳过开头 2 分钟） |
| `cluster_speakers.py` | 用 SV2TTS embedding 聚类，排查混入的其他说话人 |
| `transcribe_finetune.py` | Whisper large-v3 转写，输出 GPT-SoVITS list |
| `merge_finetune_lists.py` | 合并日文/中文切片清单 |
| `make_references.py` | 从长音频中挑选零样本参考段 |

## 说话克隆

| 脚本 | 说明 |
|---|---|
| `make_gsv_configs.py` | 生成 GPT-SoVITS 训练配置 |
| `run_gsv_preprocess.sh` | 封装 GPT-SoVITS 的 gettext / hubert / semantic 预处理 |
| `gsv_infer.py` | GPT-SoVITS 批量推理 |
| `cosyvoice_zero_shot.py` | CosyVoice3 单条零样本合成 |
| `cosyvoice_batch.py` | CosyVoice3 多参考/多文本批量合成 |
| `cosyvoice_soft.py` | CosyVoice3 柔声/气声合成 |

## 歌声转换

| 脚本 | 说明 |
|---|---|
| `train_rvc.sh` | 在 396 条东雪莲数据上训练 RVC v2 40k（固定实验名） |
| `train_rvc_generic.sh` | 通用 RVC v2 40k 训练（`WAVS`/`EXP`/`GPU_LIST` 可配置） |
| `restore_consonants.py` | 从源人声恢复 3.5–9 kHz 辅音 |
| `add_harmony.py` | 生成自然音阶和声并混音 |
| `radio_effect.py` | 电话/合成器式音色染色 |
| `remix_multitrack.py` | 主唱/和声/伴奏多轨混音 |
| `polish_song.py` | Seed-VC 输出的 EQ、空气感恢复与混音 |
| `breathify.py` | WORLD 气声增强 |
| `impose_melody.py` | 把朗读语音按音符网格拉伸（TTS→歌声） |
| `forced_align_wav2vec2.py` | wav2vec2-base-960h 词级强制对齐 |

## 评测

| 脚本 | 说明 |
|---|---|
| `speaker_similarity.py` | SV2TTS speaker encoder 计算说话人相似度 |
