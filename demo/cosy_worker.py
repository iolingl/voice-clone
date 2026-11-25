"""CosyVoice3 zero-shot worker (runs inside the ``voiceclone`` conda env).

A tiny stdlib HTTP service so the FastAPI app (which lives in the ``gptsovits``
env) can request 37-voice synthesis without mixing two PyTorch environments in
one process.  Endpoints:

    GET  /health           -> "ok"
    POST /tts   {text, speed} -> audio/wav bytes
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import COSY_MODEL_DIR, VOICES

SYS_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def _wrap(text: str) -> str:
    return text if "<|endofprompt|>" in text else SYS_PREFIX + text


class Model:
    def __init__(self, model_dir: str, ref_wav: str):
        repo = COSY_MODEL_DIR.parents[1]
        repo = os.environ.get("COSYVOICE_REPO", str(repo))
        for p in (os.path.join(repo, "third_party/Matcha-TTS"), repo):
            if p not in sys.path:
                sys.path.insert(0, p)
        os.chdir(repo)

        import torch  # noqa: F401
        import soundfile as sf
        from cosyvoice.cli.cosyvoice import AutoModel

        self.sf = sf
        self.ref = ref_wav
        print(f"[cosy] loading model from {model_dir} ...", flush=True)
        self.model = AutoModel(model_dir=model_dir)
        self.sr = self.model.sample_rate
        self.lock = threading.Lock()
        print(f"[cosy] ready, sr={self.sr}", flush=True)

    def synth_wav_bytes(self, text: str, speed: float = 1.0) -> bytes:
        import torch  # noqa: F401
        with self.lock:
            chunks = list(self.model.inference_cross_lingual(
                _wrap(text), self.ref, stream=False, speed=speed))
        speech = chunks[-1]["tts_speech"]
        arr = speech.squeeze(0).float().cpu().numpy()
        buf = io.BytesIO()
        self.sf.write(buf, arr, self.sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()


MODEL: Model | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/tts"):
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n) or b"{}")
            text = (data.get("text") or "").strip()
            if not text:
                self._send(400, b"empty text", "text/plain")
                return
            speed = float(data.get("speed", 1.0))
            wav = MODEL.synth_wav_bytes(text, speed=speed)
            self._send(200, wav, "audio/wav")
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}".encode()
            print("[cosy] error:", msg, flush=True)
            self._send(500, msg, "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--model-dir", default=str(COSY_MODEL_DIR))
    ap.add_argument("--ref", default=str(VOICES["voice37"]["ref_audio"]))
    args = ap.parse_args()

    global MODEL
    MODEL = Model(args.model_dir, args.ref)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[cosy] serving on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
