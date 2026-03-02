"""
schema.py — Core data structures for diarization output.

All pipeline outputs are expressed as lists of DiarizationSegment.
JSON serialization is handled here to keep it out of the pipeline logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class DiarizationSegment:
    """A single speaker turn."""
    speaker: str       # e.g. "speaker_0", "speaker_1"
    start: float       # seconds
    end: float         # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"DiarizationSegment({self.speaker}, {self.start:.3f}s → {self.end:.3f}s)"


@dataclass
class DiarizationResult:
    """Full diarization output for one audio file."""
    file: str                          # source file path or identifier
    segments: List[DiarizationSegment]
    duration: float                    # total audio duration in seconds
    num_speakers: int
    backend: str                       # "onnx" or "openvino"
    error: Optional[str] = None        # set if processing failed

    @classmethod
    def from_error(cls, file: str, error: str) -> "DiarizationResult":
        """Create a failed result — used in batch processing."""
        return cls(
            file=file,
            segments=[],
            duration=0.0,
            num_speakers=0,
            backend="unknown",
            error=error,
        )

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "duration": round(self.duration, 3),
            "num_speakers": self.num_speakers,
            "backend": self.backend,
            "error": self.error,
            "segments": [
                {
                    "speaker": s.speaker,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "duration": round(s.duration, 3),
                }
                for s in self.segments
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json())