from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class AudioConfig:
    """Audio I/O and preprocessing parameters."""
    sample_rate: int = 16_000
    chunk_duration: float = 10.0   # PyanNet was traced on 10s
    hop_duration: float = 1.0      # step between successive chunks in seconds
    allow_padding: bool = False    # if True, pad the last chunk instead of dropping it
    fbank_num_mel_bins: int = 80
    fbank_frame_length: float = 25.0   # ms
    fbank_frame_shift: float = 10.0    # ms

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_duration * self.sample_rate)

    @property
    def chunk_step(self) -> int:
        return int(self.hop_duration * self.sample_rate)


@dataclass
class PipelineConfig:
    """Segmentation, embedding, and stitching thresholds."""
    speech_on: float = 0.5            # hysteresis activate threshold
    speech_off: float = 0.3           # hysteresis deactivate threshold
    min_segment_duration: float = 0.1 # filter applied post-clustering
    min_embedding_samples: int = 400  # pad short segs to this length
    stitch_threshold: float = 0.7     # cosine sim for cross-chunk matching
    global_emb_decay: float = 0.3     # EMA weight when updating global embeddings
    merge_gap: float = 0.1            # seconds to bridge same-speaker gaps
    
    min_duration_on: float = 0.1
    min_duration_off: float = 0.1
    min_active_frames: int = 1

@dataclass
class ClusteringConfig:
    """Speaker clustering parameters."""
    max_speakers: int = 10
    method: str = "kmeans"


@dataclass
class AppConfig:
    """Top-level config — composes all sub-configs."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

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

        pipeline = PipelineConfig(**data.get("pipeline", {}))
        clustering = ClusteringConfig(**data.get("clustering", {}))

        return cls(audio=audio, pipeline=pipeline, clustering=clustering)

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()
