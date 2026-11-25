"""Central configuration for the voice-cloning web demo.

All paths are resolved relative to the demo directory so the same code works
both when running in-place (``<project>/demo``) and from the delivered archive
(``<delivery>/demo`` next to ``model/`` and ``dataset/``).
"""
from __future__ import annotations

import os
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent


def _first_existing(*cands: Path) -> Path | None:
    for c in cands:
        if c.exists():
            return c
    return None


# --- project-specific assets (bundled in the archive) ----------------------
MODEL_ROOT = _first_existing(ROOT / "model", ROOT / "models", DEMO_DIR / "model")
DATA_ROOT = _first_existing(ROOT / "dataset", ROOT / "dataset_voice_clone")

# --- universal base models (NOT bundled; large) ----------------------------
# Override with environment variables when they live somewhere else.
def _pick(env_name: str, *cands: Path) -> Path:
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    for c in cands:
        if c.exists():
            return c
    return cands[0]


COSY_MODEL_DIR = _pick(
    "COSYVOICE_MODEL_DIR",
    ROOT / "third_party" / "CosyVoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B",
)
GSV_REPO = _pick(
    "GPT_SOVITS_REPO",
    ROOT / "third_party" / "GPT-SoVITS",
)

# --- misc ------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("VOICE_DEMO_OUTPUT", DEMO_DIR / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COSY_WORKER_HOST = os.environ.get("COSY_WORKER_HOST", "127.0.0.1")
COSY_WORKER_PORT = int(os.environ.get("COSY_WORKER_PORT", "8102"))

# --- voice definitions -----------------------------------------------------
# Two voices.  东雪莲 uses the fine-tuned GPT-SoVITS model (native text->audio).
# 37 has no fine-tuned TTS, so we use CosyVoice3 zero-shot with her game-voice
# reference clip.
VOICES = {
    "dongxuelian": {
        "name": "东雪莲",
        "subtitle": "GPT-SoVITS 微调",
        "engine": "gpt_sovits",
        "description": (
            "以 396 条中日双语语料（42.9 分钟）微调 GPT-SoVITS。"
            "支持中文、日语朗读，音色稳定。"
        ),
        "langs": ["zh", "ja"],
        "default_lang": "zh",
        # 每种语言使用与参考音频严格对应的提示（prompt_text 必须与音频一致）
        "prompts": {
            "zh": {
                "audio": DEMO_DIR / "assets" / "prompts" / "dongxuelian_prompt_zh.wav",
                "text": "天冷是这样吗?我真的我醒三四个小时我已经开始困了",
                "lang": "zh",
            },
            "ja": {
                "audio": DEMO_DIR / "assets" / "prompts" / "dongxuelian_prompt_ja.wav",
                "text": "で、パプリカも無理です。ナスも無理です。でもナスよりトマトの方が無理。だからナスは味ないからさ。",
                "lang": "ja",
            },
        },
        "demo_ref": "/assets/prompts/dongxuelian_ref02.wav",
        "gpt": (MODEL_ROOT / "dongxuelian_voice_v1" / "gpt_sovits_speech" / "dongxuelian3-e15.ckpt") if MODEL_ROOT else None,
        "sovits": (MODEL_ROOT / "dongxuelian_voice_v1" / "gpt_sovits_speech" / "dongxuelian3_e12_s612.pth") if MODEL_ROOT else None,
        "examples": [
            "你好，今天天气真不错，很高兴认识你。",
            "大家好，我是东雪莲，这是声音克隆网页演示。",
        ],
    },
    "voice37": {
        "name": "37",
        "subtitle": "CosyVoice3 零样本",
        "engine": "cosyvoice",
        "description": (
            "《重返未来：1999》角色 37。未做微调，使用 7 秒游戏语音参考"
            "做零样本克隆，支持中/英/日等多语种。"
        ),
        "langs": ["en", "zh", "ja"],
        "default_lang": "en",
        "ref_audio": DEMO_DIR / "assets" / "prompts" / "voice37_ref.wav",
        "ref_lang": "en",
        "demo_ref": "/assets/prompts/voice37_ref.wav",
        "examples": [
            "Hello, this is a voice cloning web demo. Nice to meet you.",
            "The stars are beautiful tonight, aren't they?",
        ],
    },
}


def voice_public(vid: str) -> dict:
    """JSON-serialisable description of a voice for the frontend."""
    v = VOICES[vid]
    return {
        "id": vid,
        "name": v["name"],
        "subtitle": v["subtitle"],
        "engine": v["engine"],
        "description": v["description"],
        "langs": v["langs"],
        "default_lang": v["default_lang"],
        "ref_audio": v["demo_ref"],
        "examples": v["examples"],
    }
