import wave
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from config import AudioConfig


class AudioProcessor:
    """
    Encapsulates all audio loading and preprocessing operations.
    audio.py — Audio I/O and preprocessing.

    Responsibilities:
    - Load WAV files (mono, resample to target sample rate)
    - Chunk audio into overlapping windows for segmentation
    - Compute mel filterbank features (fbank) for the embedding model

    Fbank parameters must match what compute_fbank() produced during model export
    (verified: 98 frames per 1s at 16kHz, 80 mel bins). All parameters are now
    driven by AudioConfig rather than module-level constants.
    
    Parameters
    ----------
    config : AudioConfig
        Audio parameters (sample rate, chunk size, fbank settings).

    Example
    -------
    >>> from config import AppConfig
    >>> proc = AudioProcessor(AppConfig.default().audio)
    >>> waveform, duration = proc.load("audio/debate.wav")
    >>> for chunk, t0, t1 in proc.sliding_chunks(waveform):
    ...     fbank = proc.compute_fbank(chunk)
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, path: Path | str) -> Tuple[torch.Tensor, float]:
        """
        Load a WAV file, convert to mono at the configured sample rate.

        Returns
        -------
        waveform : torch.Tensor
            Shape (1, 1, N) — batch × channel × samples.
        duration : float
            Audio duration in seconds.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sr = torchaudio.load(str(path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        duration = waveform.shape[-1] / self.config.sample_rate
        return waveform.unsqueeze(0), duration

    def sliding_chunks(
        self,
        waveform: torch.Tensor,
    ) -> Iterator[Tuple[torch.Tensor, float, float]]:
        """
        Yield overlapping chunks from a waveform for segmentation inference.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape (1, 1, N)

        Yields
        ------
        chunk : torch.Tensor
            Shape (1, 1, chunk_samples) — zero-padded at the end if needed.
        t_start : float
            Start time of chunk in seconds.
        t_end : float
            End time of chunk in seconds (before padding).
        """
        cfg = self.config
        n_samples = waveform.shape[-1]
        start = 0

        while start < n_samples:
            end = start + cfg.chunk_samples
            chunk = waveform[:, :, start:end]

            t_start = start / cfg.sample_rate
            t_end = min(end, n_samples) / cfg.sample_rate

            if chunk.shape[-1] < cfg.chunk_samples:
                pad = cfg.chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            yield chunk, t_start, t_end
            start += cfg.chunk_step

    def compute_fbank(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Compute mel filterbank features from a raw waveform chunk.

        Replicates emb_model.compute_fbank() without pyannote.
        Produces shape (1, 98*duration_s, 80) at default settings.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape (1, 1, N) or (1, N) at the configured sample rate.

        Returns
        -------
        fbank : np.ndarray
            Shape (1, num_frames, fbank_num_mel_bins).
        """
        cfg = self.config
        wav = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        wav = wav * 32768.0  # scale to int16 range as kaldi expects

        features = kaldi.fbank(
            wav,
            num_mel_bins=cfg.fbank_num_mel_bins,
            frame_length=cfg.fbank_frame_length,
            frame_shift=cfg.fbank_frame_shift,
            sample_frequency=float(cfg.sample_rate),
            use_energy=False,
        )
        return features.unsqueeze(0).numpy()

    def profile(self, path: Path | str) -> dict:
        """
        Lightweight audio profiler — reads header only, no full waveform load.
        Used by the batch runner for scheduling.
        """
        path = Path(path)
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            num_frames = wf.getnframes()
        duration = num_frames / sample_rate
        return {
            "path": str(path),
            "duration": duration,
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "num_frames": num_frames,
            "size_bytes": path.stat().st_size,
            "estimated_chunks": max(1, int(duration / self.config.chunk_duration)),
        }


if __name__ == "__main__":
    from config import AppConfig

    proc = AudioProcessor(AppConfig.default().audio)

    audio_path = "audio/debate.wav"
    profile = proc.profile(audio_path)
    print(profile)
