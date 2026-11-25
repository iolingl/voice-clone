# 模型目录

本目录只保留模型卡、训练配置与训练清单。权重文件（`.pth` / `.ckpt` / `.index` / `.pt`）不进入 Git，统一由 Release 资产提供：

- **`voice-clone-models-v1.0.zip`**：包含 `dongxuelian_voice_v1/` 与 `37_voice_v1/` 的完整权重、参考音频与 `SHA256SUMS`。

```text
models/
├── dongxuelian_voice_v1/
│   ├── gpt_sovits_speech/     # 东雪莲 GPT-SoVITS v3 权重（Release）
│   ├── rvc_singing/           # 东雪莲 RVC v2 40k 唱歌权重（Release）
│   └── README.md
└── 37_voice_v1/
    ├── rvc_singing/v1_speech_only/
    ├── rvc_singing/v2_p1p2/   # 推荐
    ├── dataset_selection/
    └── README.md
```

下载解压示例：

```bash
unzip voice-clone-models-v1.0.zip -d .
# 若模型目录下已有 README/配置，会合并覆盖；权重文件补齐
```

细节见各子目录 README 与 [`../docs/models.md`](../docs/models.md)。
