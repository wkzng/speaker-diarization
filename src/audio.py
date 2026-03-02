from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from config import AudioConfig


class AudioProcessor:
    """
    Encapsulates all audio loading and preprocessing operations.

    Responsibilities:
    - Load WAV files (mono, resample to target sample rate)
    - Chunk audio into overlapping windows for segmentation
    - Compute mel filterbank features (fbank) for the embedding model

    Parameters
    ----------
    config : AudioConfig

    Example
    -------
    >>> proc = AudioProcessor(AppConfig.default().audio)
    >>> waveform, duration = proc.load("audio/debate.wav")
    >>> for chunk, t0, t1, t_real in proc.sliding_chunks(waveform):
    ...     fbank = proc.compute_fbank(chunk)
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def load(self, path: Path | str) -> Tuple[torch.Tensor, float]:
        """
        Load a WAV file, convert to mono at the configured sample rate.

        Returns
        -------
        waveform : torch.Tensor, shape (1, 1, N)
        duration : float
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

    def sliding_chunks(self, waveform: torch.Tensor) -> Generator:
        """
        Yield non-padded overlapping chunks for segmentation inference.

        Stops when the next chunk would extend beyond the audio —
        i.e. only full chunks are emitted (start + chunk_samples <= n_samples).
        This avoids feeding zero-padded tail chunks to the segmentation model,
        which would produce spurious silence votes for the padding region.

        Yields
        ------
        chunk : torch.Tensor, shape (1, 1, chunk_samples) — always full, never padded.
        t_start : float — chunk start in seconds.
        t_end : float — chunk end in seconds (t_start + chunk_duration).
        """
        cfg = self.config
        n_samples = waveform.shape[-1]
        start = 0
        min_chunk_size = 0 if cfg.allow_padding else cfg.chunk_samples

        while start + min_chunk_size <= n_samples:
            end = start + cfg.chunk_samples
            chunk = waveform[:, :, start:end]

            if cfg.allow_padding and chunk.shape[-1] < cfg.chunk_samples:
                pad   = cfg.chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            t_start = start / cfg.sample_rate
            t_end = min(end, n_samples) / cfg.sample_rate

            yield chunk, t_start, t_end
            start += cfg.chunk_step

    def compute_fbank(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Compute mel filterbank features.
        Produces shape (1, 98*duration_s, 80) at default settings.

        Parameters
        ----------
        waveform : torch.Tensor, shape (1, 1, N) or (1, N)

        Returns
        -------
        np.ndarray, shape (1, num_frames, num_mel_bins)
        """
        cfg = self.config
        wav = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        wav = wav * 32768.0

        features = kaldi.fbank(
            wav,
            num_mel_bins=cfg.fbank_num_mel_bins,
            frame_length=cfg.fbank_frame_length,
            frame_shift=cfg.fbank_frame_shift,
            sample_frequency=float(cfg.sample_rate),
            use_energy=False,
        )
        return features.unsqueeze(0).numpy()



if __name__ == "__main__":
    from config import AppConfig

    proc = AudioProcessor(AppConfig.default().audio)
    waveform, duration = proc.load("audio/debate.wav")
    print(f"Loaded: {waveform.shape}, {duration:.1f}s")

    for i, (chunk, t0, t1) in enumerate(proc.sliding_chunks(waveform)):
        print(f"  chunk {i:2d}: {t0:.1f}→{t1:.1f}s")