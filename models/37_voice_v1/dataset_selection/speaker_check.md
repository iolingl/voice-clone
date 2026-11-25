# 37 语音 cut 说话人排查

源：`BV1MqNmeQEsA` 两个分 P，**都是英文语音**（B 站中文标题/字幕，不是中文配音）：

| 分 P | 标题 | 时长 | 分离结果 |
|---|---|---|---|
| p01 | 洞穴的囚徒 | 21:23 | `separated/source_Vocals.wav` |
| p02 | 孤独之歌 | 40:25 | `separated_p2/source_p2_Vocals.wav` |

> 注意：早期 `asr.csv` 是把 Whisper 强制成 `language=zh` 跑的，英文台词被转成了中文译文，
> 看起来像中文语音；自动语言检测重跑后全部为 `en`。转写文本不参与 RVC 训练。

## p01（165 条 / 12.9 分钟）

- 162/165 聚为同一说话人；F0 中位 293 Hz。
- 2 条明确可疑，已剔除：`seg_00131`（F0 144 Hz，“Are you afraid?”）、
  `seg_00135`（F0 545 Hz，“Hey, come on, keep up!”）；另有 `seg_00142` 只有一声“Oh”。
- 再剔除 15 条音乐/环境残留偏多的，保留 **147 条 / 11.43 分钟**。

## p02（291 条 / 27.6 分钟）

- 279/291 聚为同一说话人；每条对 p01 质心的相似度中位 0.817，整体质心相似度 **0.933**。
- 抽查所有聚类离群片段，内容都是 37 的台词（“But you look just like Mama.”、
  “Why are you still leaving?”、笑声等），没有发现明确的其他角色。
- 剔除 22 条：相似度 <0.66、F0 异常（199 / 843–941 Hz）、平坦度 >0.22（噪声/音乐残留）。
- 保留 **269 条 / 25.7 分钟**。

## 最终训练集

- `wavs_v2/`：416 条 / **37.1 分钟**（p01 147 + p02 269），40 kHz 单声道 PCM16
- `train_selection.csv`（p01）、`p2_selection.csv`（p02）：逐条记录 dur / F0 /
  flatness / 相似度 / 取舍原因
- 旧 v1 训练集 `wavs/` 147 条保留对照

筛选脚本：`scripts/vad_segments.py`（切分）、`scripts/cluster_speakers.py`（聚类）；
训练用 `scripts/train_rvc_generic.sh`（`EXP=37_rvc40k_v2`）。
