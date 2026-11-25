> 本文是歌曲翻唱实验的原始目录记录。音频、分轨与中间产物体积很大，不随 Git 仓库提供；
> `samples/singing/` 仅保留少量试听片段；完整成品与分轨体积很大，未随仓库或 Release 提供。

# 歌声翻唱素材与成品

所有文件按用途分四层：**sources（原音源）→ separated（分离）→ work（中间产物）→ covers（成品）**。
_archive 放失败/被取代的试验，可以随时删。

## 目录

```
sources/         B 站原始音源（m4a + 转好的 44.1k wav）
separated/       分离结果，每首歌一个目录
covers/          成品，推荐试听的就是这里
work/            中间产物：Seed-VC 时代版本、EQ 中间文件、实验混音
_archive/        失败的分轨试验、早期 RVC 参数试验
```

## 歌曲索引

| 目录名 | BV 号 | 曲目 | 时长 | 说明 |
|---|---|---|---|---|
| `520am` | BV1S1421d74g | 《5:20AM》日语填词版 | 157 s | 最早的 Seed-VC 翻唱；后期有 RVC 版与和声版 |
| `tiaolouji_orig` | BV13LFNzmEHG | 《跳楼机》原版（录音室） | 200 s | 用来逆向分轨：主唱/和声/伴奏三轨 |
| `tiaolouji_ai` | BV1x4ALeHEdw | 《跳楼机》AI 東雪蓮翻唱 | 200 s | 只作为音色/混音参考，不是成品 |
| `jiayibingding` | BV1Ekub6zE3D | 李佳薇《甲乙丙丁》 | 210 s | 最新一版：辅音恢复 + 电话音色 |
| `twinkle` | BV1yb4y1h7oQ | Twinkle, Twinkle, Little Star（英文儿歌） | 170 s | 37 音色 v2 模型翻唱 |
| `twinkle_jewel` | BV1qf8yz1EaH（36:09 起） | Jewel 版 Twinkle, Twinkle, Little Star | 186 s | 37 音色 v2 翻唱；美式口音源，无和声层 |
| `ep37` | BV16c411Z7rP | 官方 EP《37如是说 Thus Spoken 37》 | 159 s | 37 声优演唱；**多角色**，仅高音区是 37。用于音色诊断，不是翻唱成品 |

## 成品（covers/）

### jiayibingding — 李佳薇《甲乙丙丁》
| 文件 | 说明 |
|---|---|
| `cover37_v2_jiayibingding_cons_loud.wav` | **37 音色，v2 模型**（416 条 / 37 min）：辅音恢复 + 混音 |
| `cover37_v2_jiayibingding_cons_down2_loud.wav` | 同上，整首降 2 个半音（音色更稳） |
| `cover37_jiayibingding_cons_loud.wav` | 37 音色，v1 模型（147 条 / 11 min），留作对照 |
| `cover_jiayibingding_cons_loud.wav` | **東雪蓮音色**：辅音恢复，咬字最清楚 |
| `cover_jiayibingding_radio.wav` | 电话/合成器染色（EQ 量 0.85） |
| `cover_jiayibingding_radio_drive.wav` | 电话染色 + tanh 饱和，颗粒感最重 |
| `*_vocal_loud.wav` / `*_vocal.wav` | 对应的纯人声版 |

37 模型和参考音色见 [`models/37_voice_v1/`](../../models/37_voice_v1/)。

### twinkle — Twinkle, Twinkle, Little Star
| 文件 | 说明 |
|---|---|
| `cover37_twinkle_cons_loud.wav` | **推荐**：37 音色 v2 模型，辅音恢复 + 混音（−14.0 LUFS） |
| `cover37_twinkle_cons.wav` | 同一份音频，只小 3.00 dB（−17.0 LUFS） |
| `*_vocal_loud.wav` / `*_vocal.wav` | 对应的纯人声版；`_loud` 同样是 +3 dB |

`cons` 指"辅音恢复"（`scripts/restore_consonants.py`），不是保守版；`_loud` 只是做了 +3.00 dB
线性增益，和 `cons` 逐样本一致（残差 −101 dB），没有压缩或 EQ。

源音源是单声道儿歌（170 s，F0 中位约 349 Hz）；这次源人声 4–8 kHz 比 RVC 输出高 8.5 dB，
辅音恢复增益 1.37，所以混音时没有再额外加 presence EQ。

### ep37 — 官方 EP《37如是说》（诊断参考，不是成品）
| 文件 | 说明 |
|---|---|
| `separated/ep37/stage1/*_Vocals.wav` / `_Instrumental.wav` | BS-RoFormer 分离 |
| `separated/ep37/stems/ep_high_40s.wav` | 37 独唱高音区（19.5–60 s），约 0.7 min |
| `separated/ep37/stems/ep_low_30s.wav` | 低音角色（很可能是 210），**不是 37** |
| `separated/ep37/stems/rvc37/` | 各种 RVC 自转换/参数网格/索引试验 |
| `stems/seedvc/` 与 `stems/rvc37/ep_*seedvc*` | Seed-VC 参考试验 |

这个 EP 是 37、苏菲亚、210、星锑四角色的歌；能当 37 唱歌参考的只有约 53 s（20–60 s + 102–115 s）。
诊断结论和 A/B 试听包见 [37 模型诊断](37_model_diagnosis.md) 与 `data/37/diagnostics/`（本地实验文件）。

### twinkle_jewel — Jewel 版 Twinkle
| 文件 | 说明 |
|---|---|
| `cover37_twinkle_jewel_cons_loud.wav` | **推荐**：37 音色 v2，辅音恢复 + 混音，−14 LUFS |
| `cover37_twinkle_jewel_nat_loud.wav` | 不做辅音恢复的原始 RVC 混音 |
| `cover37_twinkle_jewel_cons.wav` | 同 cons_loud，小 3 dB |
| `cover37_twinkle_jewel_vocal_loud.wav` | 纯人声 |

详见 `data/songs/covers/twinkle_jewel/`（本地实验文件）。

### ep37 — 《37如是说》前 40 秒（37 音色完整混音）
| 文件 | 说明 |
|---|---|
| `ep37_19.5-59.5s_rvc37_full.wav` | **推荐**：37 高音区 40 秒，主唱换 37，原版和声 + 伴奏 |
| `ep37_19.5-59.5s_rvc37_detele.wav` | 去电话版：只校正主唱频谱，更亮更清楚 |
| `ep37_19.5-59.5s_rvc37_all37.wav` | 同上，和声也过 37 模型 |
| `ep37_0-40s_rvc37_full.wav` | 字面"前 40 秒"：19.5 s 前奏 + 20 s 主唱 |
| `ep37_0-40s_rvc37_detele.wav` | 前 40 秒的去电话版 |

详见 `data/songs/covers/ep37/`（本地实验文件）。

### tiaolouji — 《跳楼机》原版逆向分轨版
| 文件 | 说明 |
|---|---|
| `cover2_tiaolouji_cons_loud.wav` | 辅音恢复版，音色自然 |
| `cover2_tiaolouji_radio_cons.wav` | 电话染色 + 辅音恢复 |
| `cover2_tiaolouji_radio_cons_drive.wav` | 再加饱和 |
| `cover2_tiaolouji_cons_up3.wav` | 升 3 个半音（与 AI 参考同调性） |
| `cover2_tiaolouji_radio_cons_up3.wav` | 电话染色 + 升 3 半音 |
| `cover2_tiaolouji_multitrack_loud.wav` | 上一版正式成品（立体声和声分轨 L/R 各转一次） |
| `cover2_tiaolouji_multitrack_mono_loud.wav` | 同一版的单声道和声转法（声场更窄） |

### 520am — 《5:20AM》
| 文件 | 说明 |
|---|---|
| `cover1_520AM_rvc.wav` | RVC 翻唱成品 |
| `cover1_520AM_harm.wav` | 加三度/五度和声的版本 |

## separated/ 里有什么

每首歌统一结构：

```
<歌曲>/
├── stage1/      BS-RoFormer：Vocals + Instrumental
├── stage2_mb/   mega back-vocal：back-vocal（和声）+ other（主唱）
└── stems/       给 RVC 用的 mono / L / R 切片；stems/rvc 是转换结果
```

`tiaolouji_ai` 额外有 `htdemucs/`、`htdemucs_ft/`（旧 Demucs 分离）和 `tracks/`（AI 参考的主唱/和声）。

## 备注

- 源文件保持 BV 号命名，方便和分离中间文件对应。
- `tiaolouji_ai` 与 `jiayibingding` 的成品都做了“辅音恢复”，原理见 `scripts/restore_consonants.py`。
- 电话音色见 `scripts/radio_effect.py`，参考曲线来自 `tiaolouji_ai` 的主唱。
