from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from audio import AudioProcessor
from backend import BackendType, InferenceBackend, create_backend
from clustering import cluster_embeddings, merge_segments
from config import AppConfig
from schema import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)

import numpy as np
from typing import List, Tuple


POWERSET = {0:[], 1:[0], 2:[1], 3:[2], 4:[0,1], 5:[0,2], 6:[1,2]}
N_SLOTS = 3
FRAME_DUR = 10.0 / 589  # ~17ms segmentation model native frame duration


def chunk_to_intervals(
    probs: np.ndarray,       # (589, 7) softmax probabilities
    t_start: float,          # absolute start of this chunk in seconds
    speech_threshold: float, # silence frames below this confidence
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
                t_start + s_frame * FRAME_DUR,
                t_start + e_frame * FRAME_DUR,
            ))

    return intervals



class BaseDiarization(ABC):
    """
    Shared logic for batch and streaming diarization.
    Template method pattern: process() defines the pipeline skeleton,
    consolidate() is the abstract hook implemented by each subclass.
    """

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
        self.config = config or AppConfig.default()
        self.audio = AudioProcessor(self.config.audio)
        self.backend: InferenceBackend = create_backend(backend, models_dir)
        logger.info(f"{self.__class__.__name__} ready — backend={backend}")



class NonCausalDiarization(BaseDiarization):

    def __call__(self, audio_path) -> dict:
        waveform, duration = self.audio.load(audio_path)

        # Step 1 — collect raw intervals from every chunk
        raw_intervals = {slot: [] for slot in range(N_SLOTS)}

        for chunk, t_start, t_end in self.audio.sliding_chunks(waveform):
            probs = np.exp(self.backend.run_segmentation(np.asarray(chunk))[0])  # (589, 7)
            for slot, t0, t1 in chunk_to_intervals(probs, t_start, speech_threshold=0.5):
                raw_intervals[slot].append((t0, t1))

 
        # Step 2 — merge overlapping/nearby intervals per slot
        # TODO: call merge_intervals(raw_intervals[slot]) for each slot

        # Step 3 — embed each merged interval
        # TODO

        # Step 4 — cluster embeddings → global speaker labels
        # TODO

        # Step 5 — build result
        # TODO
        pass