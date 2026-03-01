from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class AudioConfig:
    """Audio I/O and preprocessing parameters."""
    sample_rate: int = 16_000
    chunk_duration: float = 10.0   # PyanNet was traced on 10s
    step_ratio: float = 0.1        # hop as fraction of chunk
    fbank_num_mel_bins: int = 80
    fbank_frame_length: float = 25.0   # ms
    fbank_frame_shift: float = 10.0    # ms

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_duration * self.sample_rate)

    @property
    def chunk_step(self) -> int:
        return int(self.chunk_samples * self.step_ratio)



@dataclass
class AppConfig:
    """Top-level config — composes all sub-configs."""
    audio: AudioConfig = field(default_factory=AudioConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AppConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        # ── audio section (nested fbank key) ──────────────────────────────────
        raw_audio = dict(data.get("audio", {}))
        fbank = raw_audio.pop("fbank", {})
        audio = AudioConfig(
            **raw_audio,
            fbank_num_mel_bins=fbank.get("num_mel_bins", AudioConfig.fbank_num_mel_bins),
            fbank_frame_length=fbank.get("frame_length", AudioConfig.fbank_frame_length),
            fbank_frame_shift=fbank.get("frame_shift", AudioConfig.fbank_frame_shift),
        )
        return cls(audio=audio)

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()
