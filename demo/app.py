"""FastAPI web demo for the voice-cloning course project.

Runs inside the ``gptsovits`` conda env (has fastapi + uvicorn + torch).
- 东雪莲 voice: GPT-SoVITS fine-tuned model, synthesised in-process.
- 37 voice:     CosyVoice3 zero-shot, proxied to ``cosy_worker.py`` running in
                the ``voiceclone`` env (started by ``start.sh``).

Usage:  uvicorn app:app --host 0.0.0.0 --port 8000
or:     python app.py
"""
from __future__ import annotations

import io
import json
import time
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import (
    COSY_WORKER_HOST,
    COSY_WORKER_PORT,
    DEMO_DIR,
    VOICES,
    voice_public,
)
from gsv_engine import GsvEngine

MAX_CHARS = 400
DEMO = str(DEMO_DIR)

gsv = GsvEngine()

app = FastAPI(title="声音克隆网页演示", version="1.0")
app.mount("/static", StaticFiles(directory=f"{DEMO}/static"), name="static")
app.mount("/samples", StaticFiles(directory=f"{DEMO}/samples"), name="samples")
app.mount("/assets", StaticFiles(directory=f"{DEMO}/assets"), name="assets")


def cosy_health(timeout=1.0) -> bool:
    url = f"http://{COSY_WORKER_HOST}:{COSY_WORKER_PORT}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().strip() == b"ok"
    except Exception:
        return False


@app.on_event("startup")
def _startup():
    cfg = VOICES["dongxuelian"]
    if cfg.get("gpt") and cfg.get("sovits"):
        print("[app] loading GPT-SoVITS (东雪莲) ...", flush=True)
        gsv.load(cfg["gpt"], cfg["sovits"])
    else:
        print("[app] GPT-SoVITS weights not found (model dir missing?)", flush=True)


def _voice_available(vid: str) -> tuple[bool, str]:
    v = VOICES[vid]
    if v["engine"] == "gpt_sovits":
        return gsv.available, (gsv.error or "")
    if v["engine"] == "cosyvoice":
        ok = cosy_health()
        return ok, ("" if ok else "CosyVoice worker 未启动（请运行 start.sh）")
    return False, "unknown engine"


@app.get("/api/health")
def health():
    return {
        "gpt_sovits": gsv.available,
        "gpt_sovits_error": gsv.error,
        "cosy_worker": cosy_health(),
    }


@app.get("/api/voices")
def api_voices():
    out = []
    for vid in VOICES:
        pub = voice_public(vid)
        ok, err = _voice_available(vid)
        pub["available"] = ok
        pub["error"] = err
        out.append(pub)
    return out


@app.get("/api/samples")
def api_samples():
    path = DEMO_DIR / "samples" / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _wav_bytes(sr: int, audio) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.post("/api/tts")
async def api_tts(request: Request):
    data = await request.json()
    vid = data.get("voice")
    text = (data.get("text") or "").strip()
    if vid not in VOICES:
        raise HTTPException(400, "unknown voice")
    if not text:
        raise HTTPException(400, "文本为空")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"文本过长（上限 {MAX_CHARS} 字）")

    v = VOICES[vid]
    lang = data.get("lang") or v["default_lang"]
    if lang not in v["langs"]:
        lang = v["default_lang"]
    try:
        speed = float(data.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    speed = max(0.5, min(1.5, speed))

    t0 = time.time()
    if v["engine"] == "gpt_sovits":
        if not gsv.available:
            raise HTTPException(503, f"东雪莲引擎不可用：{gsv.error}")
        prompt = v["prompts"].get(lang) or v["prompts"][v["default_lang"]]
        sr, audio = gsv.synth(
            text, lang,
            ref_wav=prompt["audio"], prompt_text=prompt["text"],
            prompt_lang=prompt["lang"], speed=speed,
        )
        wav = _wav_bytes(sr, audio)
    else:  # cosyvoice via worker
        if not cosy_health():
            raise HTTPException(503, "CosyVoice worker 未启动（请运行 start.sh）")
        body = json.dumps({"text": text, "speed": speed}).encode()
        req = urllib.request.Request(
            f"http://{COSY_WORKER_HOST}:{COSY_WORKER_PORT}/tts",
            data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                wav = r.read()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"CosyVoice 合成失败：{e}")

    dt = time.time() - t0
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Gen-Seconds": f"{dt:.2f}",
            "X-Audio-Bytes": str(len(wav)),
            "Content-Disposition": 'inline; filename="tts.wav"',
        },
    )


@app.get("/")
def index():
    return FileResponse(f"{DEMO}/static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
