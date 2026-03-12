"""
Pytest test suite — speaker diarization pipeline.

Covers (no model files required):
  - Pure logic: merge_intervals, merge_segments, chunk_to_intervals, cluster_embeddings
  - Schema: DiarizationResult JSON / RTTM serialization
  - Batch runner: process_file success, failure isolation, output file creation
  - StreamingDiarization.flush: tail-audio and no-tail cases
"""

import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import torch

import numpy as np
import pytest

# Make src/ and project root importable from anywhere
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from src.pipeline import chunk_to_intervals, merge_intervals, POWERSET, N_SLOTS
from src.clustering import cluster_embeddings, merge_segments
from src.schema import DiarizationResult, DiarizationSegment
from cli import process_file, to_rttm


# ── merge_intervals ───────────────────────────────────────────────────────────

def test_merge_intervals_empty():
    assert merge_intervals([]) == []


def test_merge_intervals_single():
    assert merge_intervals([(0.0, 1.0)]) == [(0.0, 1.0)]


def test_merge_intervals_contained():
    # Inner fully contained in outer — dropped
    assert merge_intervals([(0.0, 3.0), (1.0, 2.0)]) == [(0.0, 3.0)]


def test_merge_intervals_overlapping():
    assert merge_intervals([(0.0, 2.0), (1.5, 3.0)]) == [(0.0, 3.0)]


def test_merge_intervals_gap_bridged():
    # gap 0.05 < default min_silence 0.1 → bridged
    assert merge_intervals([(0.0, 1.0), (1.05, 2.0)]) == [(0.0, 2.0)]


def test_merge_intervals_gap_too_large():
    # gap 0.5 > min_silence 0.1 → kept separate
    result = merge_intervals([(0.0, 1.0), (1.5, 2.0)])
    assert result == [(0.0, 1.0), (1.5, 2.0)]


def test_merge_intervals_unsorted_input():
    # Should handle unsorted input
    result = merge_intervals([(1.0, 2.0), (0.0, 0.5)])
    assert result == [(0.0, 0.5), (1.0, 2.0)]


# ── merge_segments ────────────────────────────────────────────────────────────

def test_merge_segments_empty():
    assert merge_segments([]) == []


def test_merge_segments_same_speaker_within_gap():
    segs = [("speaker_0", 0.0, 1.0), ("speaker_0", 1.3, 2.0)]
    assert merge_segments(segs, gap_tolerance=0.5) == [("speaker_0", 0.0, 2.0)]


def test_merge_segments_same_speaker_gap_too_large():
    segs = [("speaker_0", 0.0, 1.0), ("speaker_0", 2.0, 3.0)]
    result = merge_segments(segs, gap_tolerance=0.5)
    assert len(result) == 2


def test_merge_segments_different_speakers_not_merged():
    segs = [("speaker_0", 0.0, 1.0), ("speaker_1", 1.1, 2.0)]
    result = merge_segments(segs, gap_tolerance=0.5)
    assert len(result) == 2
    assert result[0][0] == "speaker_0"
    assert result[1][0] == "speaker_1"


def test_merge_segments_chain():
    # Three adjacent same-speaker segments all merge into one
    segs = [("speaker_0", 0.0, 1.0), ("speaker_0", 1.1, 2.0), ("speaker_0", 2.1, 3.0)]
    result = merge_segments(segs, gap_tolerance=0.5)
    assert result == [("speaker_0", 0.0, 3.0)]


# ── chunk_to_intervals ────────────────────────────────────────────────────────

def _uniform_probs(n_frames: int, cls: int) -> np.ndarray:
    """All frames assigned to `cls` with confidence 1.0."""
    probs = np.zeros((n_frames, 7), dtype=np.float32)
    probs[:, cls] = 1.0
    return probs


def test_chunk_to_intervals_all_silence():
    # class 0 = silence → no intervals for any slot
    probs = _uniform_probs(20, cls=0)
    result = chunk_to_intervals(probs, t_start=0.0, speech_threshold=0.5, frame_dur=0.017)
    assert result == []


def test_chunk_to_intervals_single_speaker():
    # class 1 → POWERSET[1] = [0], so only slot 0 is active
    probs = _uniform_probs(20, cls=1)
    result = chunk_to_intervals(probs, t_start=0.0, speech_threshold=0.5, frame_dur=0.017)
    slots = [r[0] for r in result]
    assert 0 in slots
    assert 1 not in slots
    assert 2 not in slots


def test_chunk_to_intervals_two_speakers():
    # class 4 → POWERSET[4] = [0, 1], slots 0 and 1 both active
    probs = _uniform_probs(20, cls=4)
    result = chunk_to_intervals(probs, t_start=0.0, speech_threshold=0.5, frame_dur=0.017)
    slots = {r[0] for r in result}
    assert 0 in slots
    assert 1 in slots


def test_chunk_to_intervals_absolute_time_offset():
    # t_start=5.0 should shift all returned times by 5s
    probs = _uniform_probs(10, cls=1)
    result = chunk_to_intervals(probs, t_start=5.0, speech_threshold=0.5, frame_dur=0.1)
    for _, t0, t1 in result:
        assert t0 >= 5.0
        assert t1 > t0


def test_chunk_to_intervals_below_threshold_becomes_silence():
    # Low confidence frames should be silenced even if argmax isn't 0
    probs = np.zeros((20, 7), dtype=np.float32)
    probs[:, 1] = 0.3   # class 1, but confidence < 0.5 threshold
    probs[:, 0] = 0.7   # silence wins
    result = chunk_to_intervals(probs, t_start=0.0, speech_threshold=0.5, frame_dur=0.017)
    assert result == []


# ── cluster_embeddings ────────────────────────────────────────────────────────

def test_cluster_empty():
    labels = cluster_embeddings(np.empty((0, 256), dtype=np.float32))
    assert len(labels) == 0


def test_cluster_single_segment():
    emb = np.random.randn(1, 256).astype(np.float32)
    labels = cluster_embeddings(emb, num_speakers=1)
    assert labels.tolist() == [0]


def test_cluster_known_k_separates_speakers():
    rng = np.random.default_rng(0)
    a = rng.normal([5.0] * 256, 0.1, (8, 256)).astype(np.float32)
    b = rng.normal([-5.0] * 256, 0.1, (8, 256)).astype(np.float32)
    embs = np.vstack([a, b])

    labels = cluster_embeddings(embs, num_speakers=2)

    # All 8 from cluster A must share one label; all 8 from B another
    assert len(set(labels[:8])) == 1
    assert len(set(labels[8:])) == 1
    assert labels[0] != labels[8]


def test_cluster_labels_start_at_zero_by_first_appearance():
    rng = np.random.default_rng(1)
    a = rng.normal([3.0] * 4, 0.01, (5, 4)).astype(np.float32)
    b = rng.normal([-3.0] * 4, 0.01, (5, 4)).astype(np.float32)
    embs = np.vstack([a, b])  # first 5 = cluster A → should become speaker_0

    labels = cluster_embeddings(embs, num_speakers=2)
    assert labels[0] == 0   # first-appearing cluster is always 0


def test_cluster_cannot_exceed_num_segments():
    # 3 segments, requested k=5 — should clamp to 3
    embs = np.random.randn(3, 256).astype(np.float32)
    labels = cluster_embeddings(embs, num_speakers=5)
    assert len(labels) == 3


# ── DiarizationResult schema ──────────────────────────────────────────────────

def _make_result(file="test.wav") -> DiarizationResult:
    segs = [
        DiarizationSegment("speaker_0", 0.0, 2.5),
        DiarizationSegment("speaker_1", 3.0, 5.0),
    ]
    return DiarizationResult(
        file=file, segments=segs, duration=5.0, num_speakers=2, backend="onnx"
    )


def test_diarization_result_to_dict_shape():
    d = _make_result().to_dict()
    assert d["num_speakers"] == 2
    assert d["duration"] == 5.0
    assert d["backend"] == "onnx"
    assert len(d["segments"]) == 2


def test_diarization_result_segment_fields():
    seg = _make_result().to_dict()["segments"][0]
    assert seg["speaker"] == "speaker_0"
    assert seg["start"] == 0.0
    assert seg["end"] == 2.5
    assert seg["duration"] == 2.5


def test_diarization_result_json_roundtrip():
    result = _make_result()
    parsed = json.loads(result.to_json())
    assert parsed["segments"][1]["speaker"] == "speaker_1"
    assert parsed["segments"][1]["start"] == 3.0


def test_diarization_result_from_error():
    result = DiarizationResult.from_error("bad.wav", "model not found")
    assert result.error == "model not found"
    assert result.segments == []
    assert result.num_speakers == 0


# ── RTTM serialization ────────────────────────────────────────────────────────

def test_to_rttm_format():
    segs = [
        DiarizationSegment("speaker_0", 0.0, 4.2),
        DiarizationSegment("speaker_1", 4.5, 9.1),
    ]
    result = DiarizationResult(
        file="debate.wav", segments=segs, duration=9.1, num_speakers=2, backend="onnx"
    )
    lines = to_rttm(result).strip().split("\n")
    assert len(lines) == 2

    # Format: SPEAKER <file_id> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    fields_0 = lines[0].split()
    assert fields_0[0] == "SPEAKER"
    assert fields_0[1] == "debate"
    assert float(fields_0[3]) == pytest.approx(0.0, abs=1e-3)
    assert float(fields_0[4]) == pytest.approx(4.2, abs=1e-3)
    assert fields_0[7] == "speaker_0"

    fields_1 = lines[1].split()
    assert float(fields_1[3]) == pytest.approx(4.5, abs=1e-3)
    assert float(fields_1[4]) == pytest.approx(4.6, abs=1e-3)
    assert fields_1[7] == "speaker_1"


def test_to_rttm_duration_is_segment_duration_not_end_time():
    # RTTM stores duration, not end time
    segs = [DiarizationSegment("speaker_0", 2.0, 5.0)]
    result = DiarizationResult(
        file="f.wav", segments=segs, duration=5.0, num_speakers=1, backend="onnx"
    )
    fields = to_rttm(result).split()
    assert float(fields[3]) == pytest.approx(2.0, abs=1e-3)   # start
    assert float(fields[4]) == pytest.approx(3.0, abs=1e-3)   # duration = 5 - 2


# ── Batch runner: process_file ────────────────────────────────────────────────

def _mock_pipeline(result: DiarizationResult):
    m = MagicMock()
    m.return_value = result
    return m


def test_process_file_success(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.touch()
    result = _make_result(file=str(wav))

    path, ok, msg = process_file(wav, _mock_pipeline(result), output_dir=None, fmt="json")

    assert ok is True
    assert "1 spk" not in msg   # 2 speakers
    assert "2 spk" in msg
    assert "RTF=" in msg


def test_process_file_writes_json(tmp_path):
    wav = tmp_path / "sample.wav"
    wav.touch()
    out = tmp_path / "out"
    out.mkdir()

    result = _make_result(file=str(wav))
    process_file(wav, _mock_pipeline(result), output_dir=out, fmt="json")

    out_file = out / "sample.json"
    assert out_file.exists()
    parsed = json.loads(out_file.read_text())
    assert parsed["num_speakers"] == 2


def test_process_file_writes_rttm(tmp_path):
    wav = tmp_path / "sample.wav"
    wav.touch()
    out = tmp_path / "out"
    out.mkdir()

    result = _make_result(file=str(wav))
    process_file(wav, _mock_pipeline(result), output_dir=out, fmt="rttm")

    out_file = out / "sample.rttm"
    assert out_file.exists()
    assert "SPEAKER" in out_file.read_text()


def test_process_file_no_output_dir(tmp_path):
    # output_dir=None should not raise and not write any file
    wav = tmp_path / "audio.wav"
    wav.touch()
    result = _make_result(file=str(wav))

    _, ok, _ = process_file(wav, _mock_pipeline(result), output_dir=None, fmt="json")
    assert ok is True
    assert list(tmp_path.glob("*.json")) == []


# ── Batch runner: failure isolation ──────────────────────────────────────────

def test_process_file_failure_returns_false(tmp_path):
    wav = tmp_path / "bad.wav"
    wav.touch()
    broken = MagicMock(side_effect=RuntimeError("model exploded"))

    path, ok, msg = process_file(wav, broken, output_dir=None, fmt="json")

    assert ok is False
    assert "FAILED" in msg
    assert "model exploded" in msg


def test_batch_one_failure_does_not_stop_others(tmp_path):
    good = tmp_path / "good.wav"
    bad  = tmp_path / "bad.wav"
    good.touch()
    bad.touch()

    good_result = _make_result(file=str(good))

    def make_pipeline(path):
        if path == bad:
            return MagicMock(side_effect=RuntimeError("bad file"))
        return _mock_pipeline(good_result)

    files = [good, bad]
    outcomes = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(process_file, p, make_pipeline(p), None, "json"): p
            for p in files
        }
        for f in as_completed(futures):
            path, ok, _ = f.result()
            outcomes[path] = ok

    assert outcomes[good] is True
    assert outcomes[bad]  is False


def test_batch_all_failures_reported(tmp_path):
    files = [tmp_path / f"f{i}.wav" for i in range(3)]
    for f in files:
        f.touch()

    broken = MagicMock(side_effect=ValueError("exploded"))
    results = [process_file(f, broken, None, "json") for f in files]

    assert all(not ok for _, ok, _ in results)
    assert sum(1 for _, ok, _ in results if not ok) == 3


def test_batch_mixed_workers_same_result(tmp_path):
    """Sequential and parallel execution should produce the same ok/fail outcomes."""
    files = [tmp_path / f"f{i}.wav" for i in range(4)]
    for f in files:
        f.touch()
    result = _make_result()

    # Sequential
    seq_outcomes = {}
    for f in files:
        _, ok, _ = process_file(f, _mock_pipeline(result), None, "json")
        seq_outcomes[f] = ok

    # Parallel
    par_outcomes = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process_file, f, _mock_pipeline(result), None, "json"): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            _, ok, _ = fut.result()
            par_outcomes[f] = ok

    assert seq_outcomes == par_outcomes


# ── StreamingDiarization.flush ────────────────────────────────────────────────
#
# flush() is called by the WebSocket server when the client sends "flush".
# It must exist, return a DiarizationResult, and carry num_speakers from memory.
# We mock the backend so no model files are needed.

def _make_streaming_pipeline():
    from src.pipeline import StreamingDiarization
    from src.config import AppConfig

    p = StreamingDiarization.__new__(StreamingDiarization)
    p.cfg = AppConfig.default()
    p.backend_name = "onnx"

    # Fake audio processor that returns a trivial fbank
    audio_mock = MagicMock()
    audio_mock.config = p.cfg.audio
    audio_mock.compute_fbank.return_value = np.zeros((1, 98, 80), dtype=np.float32)
    p.audio = audio_mock

    # Fake backend that returns all-silence segmentation and zero embedding.
    # run_segmentation returns shape (1, 589, 7); push_chunk indexes [0] → (589, 7).
    # Use small positive values so np.log is finite (class 0 dominant = silence).
    backend_mock = MagicMock()
    seg_output = np.full((1, 589, 7), 1e-6, dtype=np.float32)
    seg_output[:, :, 0] = 1.0   # class 0 (silence) has the highest prob
    backend_mock.run_segmentation.return_value = np.log(seg_output)
    # Embedding: (1, 256) unit vector
    backend_mock.run_embedding.return_value = [np.ones((1, 256), dtype=np.float32) / 16.0]
    p.backend = backend_mock

    p.reset()
    return p


def test_flush_no_tail_returns_empty_segments():
    """When t_cursor == waveform end, no new segments and num_speakers from memory."""
    p = _make_streaming_pipeline()
    # Pre-populate memory with 2 known speakers
    p._memory = {0: np.ones(256) / 16.0, 1: -np.ones(256) / 16.0}
    p._next_id = 2

    sr = p.cfg.audio.sample_rate
    chunk_samples = p.cfg.audio.chunk_samples
    waveform = torch.zeros(1, 1, chunk_samples)
    t_cursor = chunk_samples / sr   # cursor is at the very end — no tail

    result = p.flush(waveform, t_cursor)

    assert result.segments == []
    assert result.num_speakers == 2
    assert result.backend == "onnx"


def test_flush_with_tail_processes_remaining_audio():
    """When t_cursor < waveform end, flush must call push_chunk for the tail."""
    p = _make_streaming_pipeline()
    # Mock push_chunk itself — we only care that flush delegates to it
    p.push_chunk = MagicMock(return_value=[])

    sr = p.cfg.audio.sample_rate
    chunk_samples = p.cfg.audio.chunk_samples
    # Waveform is 1.5 chunks — 0.5 chunk tail after t_cursor
    total_samples = int(chunk_samples * 1.5)
    waveform = torch.zeros(1, 1, total_samples)
    t_cursor = chunk_samples / sr   # one full chunk already processed

    result = p.flush(waveform, t_cursor)

    p.push_chunk.assert_called_once()
    assert isinstance(result, DiarizationResult)


def test_flush_result_is_valid_diarization_result():
    """flush always returns a properly structured DiarizationResult."""
    from src.schema import DiarizationResult as DR
    p = _make_streaming_pipeline()
    waveform = torch.zeros(1, 1, p.cfg.audio.chunk_samples)
    t_cursor  = waveform.shape[-1] / p.cfg.audio.sample_rate

    result = p.flush(waveform, t_cursor)

    assert isinstance(result, DR)
    assert isinstance(result.segments, list)
    assert isinstance(result.num_speakers, int)
    assert result.duration > 0


def test_flush_num_speakers_reflects_session():
    """num_speakers in flush result matches speakers seen across push_chunk calls."""
    p = _make_streaming_pipeline()
    # Manually inject 3 speakers into memory
    p._memory = {0: np.ones(256) / 16.0, 1: np.zeros(256), 2: -np.ones(256) / 16.0}
    p._next_id = 3

    waveform  = torch.zeros(1, 1, p.cfg.audio.chunk_samples)
    t_cursor  = waveform.shape[-1] / p.cfg.audio.sample_rate
    result    = p.flush(waveform, t_cursor)

    assert result.num_speakers == 3
