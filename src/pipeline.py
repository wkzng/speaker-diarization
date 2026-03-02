from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch

from src.audio import AudioProcessor
from src.backend import BackendType, InferenceBackend, create_backend
from src.clustering import cluster_embeddings, merge_segments
from src.config import AppConfig
from src.schema import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)



POWERSET = {0:[], 1:[0], 2:[1], 3:[2], 4:[0,1], 5:[0,2], 6:[1,2]}
N_SLOTS = 3

def chunk_to_intervals(
    probs: np.ndarray,       # (589, 7) softmax probabilities
    t_start: float,          # absolute start of this chunk in seconds
    speech_threshold: float, # silence frames below this confidence
    frame_dur: float,
) -> List[Tuple[int, float, float]]:
    """
    Convert one chunk's probabilities into (slot, t_start, t_end) intervals.

    Returns
    -------
    List of (slot, t_abs_start, t_abs_end) in absolute time.
    """
    # Step 1 — argmax + confidence gate
    class_index = np.argmax(probs, axis=-1)                      # (589,)
    confidence  = probs[np.arange(len(class_index)), class_index]
    class_index[confidence < speech_threshold] = 0               # silence uncertain frames

    intervals = []

    for slot in range(N_SLOTS):
        # Step 2 — build binary mask for this slot
        slot_mask = np.array([slot in POWERSET[cls] for cls in class_index])  # (589,) bool

        # Step 3 — find contiguous runs → (start_frame, end_frame) pairs
        edges = np.where(np.diff(slot_mask.astype(int), prepend=0, append=0))[0]
        for s_frame, e_frame in edges.reshape(-1, 2):
            intervals.append((
                slot,
                float(t_start + s_frame * frame_dur),
                float(t_start + e_frame * frame_dur),
            ))

    return intervals



def merge_intervals(
    intervals: List[Tuple[float, float]],
    min_silence: float = 0.1,  # seconds — bridge gaps shorter than this
) -> List[Tuple[float, float]]:
    """
    Merge a list of (t_start, t_end) intervals.

    Cases (top = last interval in merged stack):
      1. current fully contained in top         → skip
      2. current overlaps top                   → extend top's right boundary
      3. gap between top and current < min_silence → bridge (extend)
      4. otherwise                              → new interval
    """
    if not intervals:
        return []

    merged = [list(iv) for iv in sorted(intervals)][:1]  # seed with first

    for s, e in sorted(intervals)[1:]:
        top_s, top_e = merged[-1]

        if e <= top_e:                      # case 1: fully contained
            continue
        elif s <= top_e:                    # case 2: overlapping
            merged[-1][1] = e
        elif (s - top_e) <= min_silence:    # case 3: small gap
            merged[-1][1] = e
        else:                               # case 4: new interval
            merged.append([s, e])

    return [tuple(iv) for iv in merged]




def embed_intervals(
    merged_intervals: dict, # {slot: [(t_start, t_end), ...]}
    waveform: torch.Tensor, # (1, 1, N)
    audio: AudioProcessor,
    backend: InferenceBackend,
    min_samples: int,
) -> List[Tuple[int, float, float, np.ndarray]]:
    """
    Embed each merged interval.

    Returns
    -------
    List of (slot, t_start, t_end, embedding) where embedding is (256,) L2-normalised.
    """
    sr      = audio.config.sample_rate
    results = []

    for slot, intervals in merged_intervals.items():
        for t0, t1 in intervals:
            wave = waveform[:, :, int(t0 * sr) : int(t1 * sr)]

            if wave.shape[-1] < min_samples:
                wave = torch.nn.functional.pad(wave, (0, min_samples - wave.shape[-1]))

            fbank = audio.compute_fbank(wave)                        # (1, N_frames, 80)
            emb   = backend.run_embedding(fbank)[0]                  # (256,)

            if not np.all(np.isfinite(emb)) or np.allclose(emb, 0):
                continue

            emb = emb / (np.linalg.norm(emb) + 1e-8)
            results.append((slot, t0, t1, emb))

    return results




class BaseDiarization(ABC):

    def __init__(
        self,
        models_dir: Path | str,
        backend: BackendType = "onnx",
        num_speakers: Optional[int] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.backend_name = backend
        self.num_speakers = num_speakers
        self.cfg = config or AppConfig.default()
        self.audio = AudioProcessor(self.cfg.audio)
        self.backend: InferenceBackend = create_backend(backend, models_dir)
        logger.info(f"{self.__class__.__name__} ready — backend={backend}")



class Diarization(BaseDiarization):

    def __call__(self, audio_path) -> dict:
        waveform, duration = self.audio.load(audio_path)

        # Step 1: collect raw intervals from every chunk
        raw_intervals = {slot: [] for slot in range(N_SLOTS)}

        for chunk, t_start, t_end in self.audio.sliding_chunks(waveform):
            #print(t_start, t_end)
            probs = np.exp(self.backend.run_segmentation(np.asarray(chunk))[0])  # (589, 7)
            num_frames = probs.shape[0]
            frame_dur = (t_end - t_start) / num_frames

            for slot, t0, t1 in chunk_to_intervals(
                probs, t_start, 
                speech_threshold=self.cfg.pipeline.speech_on,
                frame_dur=frame_dur
            ):
                raw_intervals[slot].append((t0, t1))
 
        # Step 2: merge overlapping/nearby intervals per slot
        merged_intervals = {
            slot: merge_intervals(raw_intervals[slot], min_silence=self.cfg.pipeline.min_duration_off)
            for slot in range(N_SLOTS)
        }

        # Step 3: embed each merged interval
        embeddings = embed_intervals(
            merged_intervals,
            waveform,
            audio = self.audio,
            backend = self.backend,
            min_samples = self.cfg.pipeline.min_embedding_samples,
        )

        # # after embed_intervals, before clustering
        # for slot, t0, t1, emb in embeddings:
        #     print(f"slot={slot} {t0:.2f}→{t1:.2f}s  norm={np.linalg.norm(emb):.3f}")

        # Step 4: cluster embeddings → global speaker labels
        matrix = np.stack([emb for _, _, _, emb in embeddings])  # (N, 256)
        labels = cluster_embeddings(
            embeddings = matrix,
            num_speakers = self.num_speakers,
            max_speakers = self.cfg.clustering.max_speakers,
            method = self.cfg.clustering.method,
        )

        # Step 5: build result
        raw = sorted(
            [(f"speaker_{label}", t0, t1) for (_, t0, t1, _), label in zip(embeddings, labels)],
            key=lambda x: x[1]
        )
        merged = merge_segments(raw, gap_tolerance=self.cfg.pipeline.merge_gap)
        segments = [
            DiarizationSegment(speaker=spk, start=s, end=e)
            for spk, s, e in merged
            if (e - s) >= self.cfg.pipeline.min_segment_duration
        ]
        return DiarizationResult(
            file=str(audio_path),
            segments=segments,
            duration=duration,
            num_speakers=len({seg.speaker for seg in segments}),
            backend=self.backend_name,
        )
    





class StreamingDiarization(BaseDiarization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self._memory = {}   # {speaker_id: np.ndarray (256,)}
        self._next_id = 0

    def push_chunk(
        self,
        chunk: torch.Tensor,    # (1, 1, 160000)
        t_start: float,
        t_end: float,
        waveform: torch.Tensor, # full waveform (1, 1, N) — for slicing
    ) -> List[DiarizationSegment]:

        # Step 1 — segmentation
        probs = np.exp(self.backend.run_segmentation(np.asarray(chunk))[0])
        frame_dur = (t_end - t_start) / probs.shape[0]

        raw = {slot: [] for slot in range(N_SLOTS)}
        for slot, t0, t1 in chunk_to_intervals(probs, t_start, self.cfg.pipeline.speech_on, frame_dur):
            raw[slot].append((t0, t1))

        # Step 2 — merge within this chunk
        merged = {
            slot: merge_intervals(raw[slot], min_silence=self.cfg.pipeline.min_duration_off)
            for slot in range(N_SLOTS)
        }

        # Step 3 — embed
        embedded = embed_intervals(merged, waveform, self.audio, self.backend,
                                   self.cfg.pipeline.min_embedding_samples)
        if not embedded:
            return []

        # Step 4 — match to known speakers
        segments = []
        for slot, t0, t1, emb in embedded:
            speaker = self._match_or_create(emb)
            segments.append(DiarizationSegment(speaker=speaker, start=t0, end=t1))

        return segments

    def _match_or_create(self, emb: np.ndarray) -> str:
        if not self._memory:
            return self._new_speaker(emb)

        sims = {sid: float(emb @ ref) for sid, ref in self._memory.items()}
        best_sid, best_sim = max(sims.items(), key=lambda x: x[1])

        if best_sim >= self.cfg.pipeline.stitch_threshold:
            # EMA update
            updated = (self.cfg.pipeline.global_emb_decay * self._memory[best_sid]
                       + (1 - self.cfg.pipeline.global_emb_decay) * emb)
            self._memory[best_sid] = updated / (np.linalg.norm(updated) + 1e-8)
            return f"speaker_{best_sid}"

        return self._new_speaker(emb)

    def _new_speaker(self, emb: np.ndarray) -> str:
        sid = self._next_id
        self._next_id += 1
        self._memory[sid] = emb.copy()
        return f"speaker_{sid}"