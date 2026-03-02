from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import normalize


def cluster_embeddings(
    embeddings: np.ndarray,
    num_speakers: Optional[int] = None,
    max_speakers: int = 10,
    method: str = "agglomerative",
) -> np.ndarray:
    """
    Cluster speaker embeddings into speaker labels.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, 256) — one embedding per speech segment.
    num_speakers : int, optional
        If known upfront, pass it. Otherwise estimated automatically.
    max_speakers : int
        Upper bound when auto-estimating num_speakers.
    method : str
        "agglomerative" (default) or "kmeans".

    Returns
    -------
    labels : np.ndarray
        Shape (N,) — integer speaker label per segment.
    """
    n_segments = len(embeddings)

    if n_segments == 0:
        return np.array([], dtype=int)

    if n_segments == 1:
        return np.array([0])

    # Filter out zero embeddings (padding artifacts — seen in export output)
    valid_mask = ~np.all(embeddings == 0, axis=1)
    if not valid_mask.any():
        return np.zeros(n_segments, dtype=int)

    # L2 normalize for cosine-equivalent distance in euclidean clustering
    normed = normalize(embeddings, norm="l2")

    k = num_speakers or _estimate_num_speakers(normed, max_speakers)
    k = min(k, n_segments)  # can't have more clusters than segments

    if method == "agglomerative":
        labels = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
        ).fit_predict(normed)
    elif method == "kmeans":
        labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        ).fit_predict(normed)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Relabel to speaker_0, speaker_1, ... in order of first appearance
    return _relabel_by_first_appearance(labels)


def _estimate_num_speakers(
    embeddings: np.ndarray,
    max_speakers: int,
) -> int:
    """
    Estimate number of speakers using the eigenvalue gap heuristic
    on the affinity matrix (spectral approach, no sklearn dependency).

    Falls back to 2 if estimation fails.
    """
    try:
        n = len(embeddings)
        max_k = min(max_speakers, n - 1)

        if max_k < 2:
            return 1

        # Cosine affinity matrix
        affinity = embeddings @ embeddings.T
        affinity = np.clip(affinity, 0, 1)

        # Normalized Laplacian eigenvalues
        degree = affinity.sum(axis=1)
        degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
        laplacian = np.eye(n) - (degree_inv_sqrt[:, None] * affinity * degree_inv_sqrt[None, :])

        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))[:max_k + 1]
        gaps = np.diff(eigenvalues)

        # Number of speakers = position of largest gap + 1
        k = int(np.argmax(gaps)) + 1
        return max(1, min(k, max_speakers))

    except Exception:
        return 2  # safe fallback for 2-speaker case (most common)


def _relabel_by_first_appearance(labels: np.ndarray) -> np.ndarray:
    """
    Remap cluster IDs so speaker_0 is always the first speaker to appear.

    Example: [2, 2, 0, 1] → [0, 0, 1, 2]
    """
    mapping = {}
    next_id = 0
    result = np.empty_like(labels)

    for i, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        result[i] = mapping[label]

    return result


def merge_segments(
    segments: List[Tuple[str, float, float]],
    gap_tolerance: float = 0.5,
) -> List[Tuple[str, float, float]]:
    """
    Merge consecutive segments from the same speaker
    if the gap between them is smaller than gap_tolerance seconds.

    Parameters
    ----------
    segments : list of (speaker, start, end)
    gap_tolerance : float
        Max gap in seconds to bridge between same-speaker segments.

    Returns
    -------
    Merged list of (speaker, start, end).
    """
    if not segments:
        return []

    merged = [list(segments[0])]

    for speaker, start, end in segments[1:]:
        prev = merged[-1]
        if speaker == prev[0] and (start - prev[2]) <= gap_tolerance:
            prev[2] = end  # extend previous segment
        else:
            merged.append([speaker, start, end])

    return [tuple(s) for s in merged]