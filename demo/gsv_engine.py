"""GPT-SoVITS engine wrapper (runs inside the ``gptsovits`` conda env).

Loads the fine-tuned 东雪莲 model once and synthesises text -> audio.
Import of GPT-SoVITS is deferred to :meth:`GsvEngine.load` so that the FastAPI
app can start even when this engine is unavailable.
"""
from __future__ import annotations

import os
import sys
import threading

from config import GSV_REPO


class GsvEngine:
    def __init__(self):
        self.tts = None
        self.sr = 32000
        self.error = None
        self._lock = threading.Lock()

    def load(self, gpt_path, sovits_path):
        try:
            repo = str(GSV_REPO)
            if repo not in sys.path:
                sys.path.insert(0, repo)
                sys.path.insert(0, os.path.join(repo, "GPT_SoVITS"))
            os.chdir(repo)  # GPT-SoVITS resolves some resources relative to repo

            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

            configs = TTS_Config({
                "device": "cuda",
                "is_half": True,
                "version": "v2",
                "t2s_weights_path": str(gpt_path),
                "vits_weights_path": str(sovits_path),
                "cnhuhbert_base_path": os.path.join(repo, "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
                "bert_base_path": os.path.join(repo, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            })
            tts = TTS(configs)
            tts.init_t2s_weights(str(gpt_path))
            tts.init_vits_weights(str(sovits_path))
            self.tts = tts
            print("[gsv] GPT-SoVITS engine ready", flush=True)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            print(f"[gsv] failed to load: {self.error}", flush=True)
        return self.tts is not None

    @property
    def available(self) -> bool:
        return self.tts is not None

    def synth(self, text, lang, ref_wav, prompt_text, prompt_lang,
              speed=1.0, seed=42, top_k=15, top_p=1.0, temperature=1.0):
        if self.tts is None:
            raise RuntimeError(f"GPT-SoVITS engine not available: {self.error}")
        import numpy as np
        inputs = {
            "text": text,
            "text_lang": lang,
            "ref_audio_path": str(ref_wav),
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed,
            "seed": seed,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
        }
        with self._lock:
            sr, audio = list(self.tts.run(inputs))[-1]
        audio = audio.astype(np.float32) / 32768.0
        return sr, audio
