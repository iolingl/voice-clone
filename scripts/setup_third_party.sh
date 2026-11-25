#!/usr/bin/env bash
# 按固定 commit 克隆第三方仓库到 third_party/，保证实验可复现。
# 用法：bash scripts/setup_third_party.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$PROJECT_ROOT/third_party"
mkdir -p "$THIRD_PARTY"

clone_pinned() {
  local name="$1" url="$2" commit="$3"
  local dir="$THIRD_PARTY/$name"

  if [ ! -d "$dir/.git" ]; then
    echo "==> 克隆 $name"
    if [ "$name" = "CosyVoice" ]; then
      git clone --recursive "$url" "$dir"
    else
      git clone "$url" "$dir"
    fi
  else
    echo "==> $name 已存在，跳过克隆"
  fi

  echo "==> $name 检出 $commit"
  git -C "$dir" fetch --tags origin || true
  git -C "$dir" checkout --detach "$commit"
  git -C "$dir" submodule update --init --recursive || true
}

clone_pinned GPT-SoVITS https://github.com/RVC-Boss/GPT-SoVITS.git 6d51ffceafbf240145619af4c92408a6e54b7924
clone_pinned CosyVoice https://github.com/FunAudioLLM/CosyVoice.git 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
clone_pinned RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git 81eed5e8f68b6bed1789f682fe78cdd324495afc
clone_pinned seed-vc https://github.com/Plachtaa/seed-vc.git 51383efd921027683c89e5348211d93ff12ac2a8

echo
echo "完成。接下来请按 docs/environment.md 创建 conda 环境并下载各项目的基座模型。"
