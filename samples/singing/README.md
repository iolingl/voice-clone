# 歌声转换试听片段

这里只放 30 秒压缩试听片段（MP3 160 kbps），方便快速试听 RVC 歌声转换效果。完整成品与分轨在工作区 `data/songs/covers/`（不进入 Git）。

| 文件 | 音色 / 模型 | 曲目 | 来源完整文件 |
|---|---|---|---|
| `dongxuelian_jiayibingding.mp3` | 东雪莲 RVC v2 40k | 李佳薇《甲乙丙丁》 | `cover_jiayibingding_cons_loud.wav` |
| `voice37_jiayibingding_v2.mp3` | 37 RVC v2 40k（v2_p1p2） | 李佳薇《甲乙丙丁》 | `cover37_v2_jiayibingding_cons_loud.wav` |
| `voice37_twinkle.mp3` | 37 RVC v2 40k | Twinkle, Twinkle, Little Star | `cover37_twinkle_cons_loud.wav` |
| `voice37_twinkle_jewel.mp3` | 37 RVC v2 40k | Jewel 版 Twinkle | `cover37_twinkle_jewel_cons_loud.wav` |
| `dongxuelian_tiaolouji.mp3` | 东雪莲 RVC v2 40k | 《跳楼机》原版逆向分轨 | `cover2_tiaolouji_multitrack_loud.wav` |

截取起点均为 15 s，时长 30 s。训练、推理与后处理流程见 [`../../docs/singing_conversion.md`](../../docs/singing_conversion.md)。

> 音频与歌曲版权归原视频作者/权利方所有，仅用于学习与研究演示，请勿再分发或商用。
