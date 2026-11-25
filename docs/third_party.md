# 第三方仓库与版本

本项目不把第三方项目源码直接提交到 Git，运行前请克隆到 `third_party/`。`scripts/setup_third_party.sh` 会按以下 commit 检出，保证可复现。

| 目录 | 上游仓库 | 验证 commit |
|---|---|---|
| `third_party/GPT-SoVITS` | <https://github.com/RVC-Boss/GPT-SoVITS> | `6d51ffceafbf240145619af4c92408a6e54b7924` |
| `third_party/CosyVoice` | <https://github.com/FunAudioLLM/CosyVoice> | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` |
| `third_party/RVC` | <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI> | `81eed5e8f68b6bed1789f682fe78cdd324495afc` |
| `third_party/seed-vc` | <https://github.com/Plachtaa/seed-vc> | `51383efd921027683c89e5348211d93ff12ac2a8` |

## 手动准备

```bash
mkdir -p third_party
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
git clone https://github.com/RVC-Boss/GPT-SoVITS.git third_party/GPT-SoVITS
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git third_party/RVC
git clone https://github.com/Plachtaa/seed-vc.git third_party/seed-vc

for d in third_party/*; do
  git -C "$d" submodule update --init --recursive || true
done
```

> CosyVoice 需要 `--recursive` 拉取 Matcha-TTS；首次使用 CosyVoice 时还要按 [`environment.md`](environment.md) 安装 `pynini` 和 Python 依赖。

## 许可证

各上游项目版权归原作者所有，使用前请阅读对应仓库的 LICENSE。本仓库的 Python 脚本仅做流程编排与后处理，不修改上游核心推理实现。
