"""
streamlit_demo.py — Live speaker diarization via WebSocket streaming.

Usage:
    pip install streamlit websockets
    # Start the API server first:
    PYTHONPATH=src MODELS_DIR=models python server.py
    # Then in another terminal:
    streamlit run streamlit_demo.py
"""

import asyncio
import io
import json

import numpy as np
import streamlit as st
import torchaudio

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16_000
CHUNK_SAMPLES = 160_000          # 10 s at 16 kHz
CHUNK_BYTES   = CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample

SPEAKER_COLORS = [
    "#4A90D9", "#E8724A", "#5BAD5B", "#B05BD9",
    "#D9B94A", "#4AD9C7", "#D94A90", "#7A7A7A",
]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio(file_bytes: bytes) -> tuple[np.ndarray, float]:
    """
    Load WAV from raw bytes, resample to 16 kHz mono, return (int16 array, duration_s).

    int16 PCM is the wire format sent over the WebSocket — it's the native output
    of the Web Audio API and avoids base64 overhead (~2× smaller than float32).
    The server normalises to float32 internally by dividing by 32768.
    """
    buf = io.BytesIO(file_bytes)
    waveform, sr = torchaudio.load(buf)

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    duration = waveform.shape[-1] / SAMPLE_RATE

    # float32 [-1, 1] → int16
    pcm = (waveform.squeeze().numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)
    return pcm, duration


def make_chunks(pcm: np.ndarray) -> list[bytes]:
    """Split PCM array into fixed-size 10 s byte chunks, zero-padding the last one."""
    raw = pcm.tobytes()
    chunks = []
    for i in range(0, max(len(raw), CHUNK_BYTES), CHUNK_BYTES):
        chunk = raw[i : i + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
        chunks.append(chunk)
    return chunks


# ── WebSocket streaming ───────────────────────────────────────────────────────

async def _stream(
    server_url: str,
    chunks: list[bytes],
    num_speakers: int | None,
    on_segment,
    on_progress,
):
    """
    Stream audio chunks to /ws/diarize and call back on each returned segment.

    Protocol:
      1. Send JSON config
      2. Send binary PCM chunks (160 000 int16 samples each)
      3. After each chunk, drain any segments already available (non-blocking)
      4. Send "flush" to finalize
      5. Collect remaining segments until "done" event
    """
    import websockets

    async with websockets.connect(server_url, max_size=2**23) as ws:
        # Step 1 — send config
        await ws.send(json.dumps({
            "sample_rate": SAMPLE_RATE,
            "num_speakers": num_speakers,
        }))

        # Step 2+3 — stream chunks
        for i, chunk in enumerate(chunks):
            await ws.send(chunk)
            on_progress((i + 1) / len(chunks))

            # Drain segments returned so far (non-blocking)
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    data = json.loads(msg)
                    if "speaker" in data:
                        on_segment(data)
            except asyncio.TimeoutError:
                pass

        # Step 4 — flush
        await ws.send("flush")

        # Step 5 — collect remaining segments until done
        async for msg in ws:
            data = json.loads(msg)
            if "event" in data and data["event"] == "done":
                break
            if "speaker" in data:
                on_segment(data)


def run_streaming(server_url, chunks, num_speakers, on_segment, on_progress):
    """Run the async streaming loop synchronously (Streamlit-safe)."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _stream(server_url, chunks, num_speakers, on_segment, on_progress)
        )
    finally:
        loop.close()


# ── Rendering ─────────────────────────────────────────────────────────────────

def _speaker_color(speaker: str, color_map: dict) -> str:
    if speaker not in color_map:
        color_map[speaker] = SPEAKER_COLORS[len(color_map) % len(SPEAKER_COLORS)]
    return color_map[speaker]


def render_segment_list(segments: list, color_map: dict) -> str:
    if not segments:
        return "<i style='color:#888'>Waiting for segments…</i>"
    lines = []
    for seg in segments:
        color = _speaker_color(seg["speaker"], color_map)
        lines.append(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{color};margin-right:6px"></span>'
            f'<strong style="color:{color}">{seg["speaker"]}</strong>'
            f'&nbsp;&nbsp;<span style="color:#aaa">'
            f'{seg["start"]:.2f}s → {seg["end"]:.2f}s</span>'
        )
    return "<br>".join(lines)


def render_timeline(segments: list, duration: float, color_map: dict) -> str:
    """Simple proportional HTML timeline bar."""
    if not segments or duration == 0:
        return ""
    bars = []
    for seg in segments:
        left  = seg["start"]  / duration * 100
        width = (seg["end"] - seg["start"]) / duration * 100
        color = _speaker_color(seg["speaker"], color_map)
        bars.append(
            f'<div title="{seg["speaker"]} {seg["start"]:.2f}s→{seg["end"]:.2f}s" '
            f'style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;'
            f'height:100%;background:{color};opacity:0.85"></div>'
        )
    return (
        '<div style="position:relative;height:28px;background:#222;'
        'border-radius:4px;overflow:hidden;margin-top:8px">'
        + "".join(bars)
        + "</div>"
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Diarization Demo", layout="wide")
st.title("Speaker Diarization — Live Streaming Demo")
st.caption(
    "Upload a WAV file. The app streams it in 10 s chunks to the WebSocket API "
    "and displays speaker turns as they're returned by the server."
)

# Sidebar — config
with st.sidebar:
    st.header("Configuration")
    server_url   = st.text_input("Server WebSocket URL", value="ws://localhost:8000/ws/diarize")
    num_speakers = st.number_input("Number of speakers (0 = auto-detect)", min_value=0, max_value=10, value=0)
    num_speakers = None if num_speakers == 0 else int(num_speakers)
    st.divider()
    st.markdown(
        "**Start the server first:**\n"
        "```bash\n"
        "PYTHONPATH=src MODELS_DIR=models python server.py\n"
        "```"
    )

# File upload
uploaded = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded:
    file_bytes = uploaded.read()

    with st.spinner("Loading audio…"):
        try:
            pcm, duration = load_audio(file_bytes)
        except Exception as e:
            st.error(f"Could not load audio: {e}")
            st.stop()

    chunks = make_chunks(pcm)
    st.info(
        f"**{uploaded.name}** — {duration:.1f} s &nbsp;|&nbsp; "
        f"{len(chunks)} chunk(s) × 10 s &nbsp;|&nbsp; "
        f"{len(pcm) * 2 / 1024:.0f} KB wire size"
    )

    # Audio player — uses the original upload bytes so quality is unaffected by resampling
    st.audio(file_bytes, format="audio/wav")

    if st.button("Run diarization", type="primary"):
        segments   = []
        color_map  = {}

        progress_bar   = st.progress(0.0, text="Streaming…")
        timeline_ph    = st.empty()
        segment_list_ph = st.empty()

        def on_segment(seg):
            segments.append(seg)
            segment_list_ph.markdown(
                render_segment_list(segments, color_map),
                unsafe_allow_html=True,
            )
            timeline_ph.markdown(
                render_timeline(segments, duration, color_map),
                unsafe_allow_html=True,
            )

        def on_progress(frac: float):
            progress_bar.progress(frac, text=f"Streaming… chunk {int(frac * len(chunks))}/{len(chunks)}")

        try:
            run_streaming(server_url, chunks, num_speakers, on_segment, on_progress)
        except Exception as e:
            st.error(
                f"**Connection error:** {e}\n\n"
                f"Make sure the server is running at `{server_url}`."
            )
            st.stop()

        progress_bar.progress(1.0, text="Done")

        st.success(
            f"Finished — **{len({s['speaker'] for s in segments})} speaker(s)**, "
            f"**{len(segments)} segment(s)**"
        )

        # Final summary table
        if segments:
            st.subheader("Segments")
            st.table([
                {
                    "Speaker": s["speaker"],
                    "Start (s)": f"{s['start']:.2f}",
                    "End (s)":   f"{s['end']:.2f}",
                    "Duration (s)": f"{s['end'] - s['start']:.2f}",
                }
                for s in segments
            ])
