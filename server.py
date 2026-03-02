"""
server.py — FastAPI entrypoint.

Endpoints:
  POST /diarize          — batch: upload a WAV, get full diarization result
  WS   /ws/diarize       — streaming: send PCM chunks, receive segments in real time
"""
from __future__ import annotations

import io
import logging
import os
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from config import AppConfig
from pipeline import CausalDiarization, NonCausalDiarization
from schema import DiarizationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
MODELS_DIR  = os.environ.get("MODELS_DIR", "models")
BACKEND = os.environ.get("BACKEND", "onnx")

cfg = AppConfig.from_yaml(CONFIG_PATH) if Path(CONFIG_PATH).exists() else AppConfig.default()

app = FastAPI(title="Speaker Diarization API", version="1.0.0")

# Single shared pipeline instance (models loaded once at startup)
_batch_pipeline: Optional[NonCausalDiarization] = None


@app.on_event("startup")
def load_models():
    global _batch_pipeline
    _batch_pipeline = NonCausalDiarization(
        models_dir=MODELS_DIR,
        backend=BACKEND,
        config=cfg,
    )
    logger.info("Models loaded")



@app.post("/diarize", response_model=None)
async def diarize(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Query(None, description="Force number of speakers"),
):
    """
    Upload a WAV file, receive full diarization JSON.
    Accepts any sample rate / channel count — resampled to 16kHz mono internally.
    """
    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        _batch_pipeline.num_speakers = num_speakers
        result: DiarizationResult = _batch_pipeline(tmp_path)
        return JSONResponse(result.to_dict())
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND}


# WebSocket streaming endpoint
#
# Protocol (client → server):
#   1. First message: JSON config  {"sample_rate": 16000, "num_speakers": 2}
#   2. Subsequent messages: raw PCM bytes (int16, mono, 16kHz)
#      Send exactly chunk_samples * 2 bytes per message (160000 * 2 = 320KB for 10s)
#   3. Send text message "flush" to trigger final result
#
# Protocol (server → client):
#   Each committed segment: JSON {"speaker": "speaker_0", "start": 1.2, "end": 3.4}
#   On flush: JSON {"event": "done", "num_speakers": 2}

CHUNK_SAMPLES = cfg.audio.chunk_samples  # 160000

@app.websocket("/ws/diarize")
async def ws_diarize(websocket: WebSocket):
    await websocket.accept()

    pipeline = CausalDiarization(
        models_dir=MODELS_DIR,
        backend=BACKEND,
        config=cfg,
    )

    # PCM accumulation buffer
    pcm_buffer = bytearray()
    waveform_chunks: list[torch.Tensor] = []
    t_cursor = 0.0
    full_waveform: Optional[torch.Tensor] = None

    try:
        # First message: config
        init = await websocket.receive_json()
        num_speakers = init.get("num_speakers", None)
        pipeline.num_speakers = num_speakers

        while True:
            msg = await websocket.receive()

            # Text control message
            if "text" in msg:
                if msg["text"] == "flush" and full_waveform is not None:
                    result = pipeline.flush(full_waveform, t_cursor)
                    for seg in result.segments:
                        await websocket.send_json({
                            "speaker": seg.speaker,
                            "start": round(seg.start, 3),
                            "end": round(seg.end,   3),
                        })
                    await websocket.send_json({"event": "done", "num_speakers": result.num_speakers})
                    pipeline.reset()
                continue

            # Binary PCM chunk
            pcm_bytes = msg.get("bytes", b"")
            pcm_buffer.extend(pcm_bytes)

            # Process complete chunks
            bytes_per_chunk = CHUNK_SAMPLES * 2  # int16
            while len(pcm_buffer) >= bytes_per_chunk:
                raw = bytes(pcm_buffer[:bytes_per_chunk])
                del pcm_buffer[:bytes_per_chunk]

                # int16 → float32 tensor (1, 1, N)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                chunk   = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)  # (1,1,N)

                waveform_chunks.append(chunk)
                full_waveform = torch.cat(waveform_chunks, dim=-1)

                t_start = t_cursor
                t_end = t_cursor + cfg.audio.chunk_duration

                segments = pipeline.push_chunk(chunk, t_start, t_end, full_waveform)
                t_cursor = t_end

                for seg in segments:
                    await websocket.send_json({
                        "speaker": seg.speaker,
                        "start": round(seg.start, 3),
                        "end": round(seg.end,   3),
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        await websocket.close(code=1011)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)