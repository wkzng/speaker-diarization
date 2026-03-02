# Speaker Diarization Pipeline

A minimal, production-minded speaker diarization pipeline built on top of the [Community-1](https://github.com/pyannote/pyannote-audio?tab=readme-ov-file) pretrained model. Given an audio file, it outputs a chronogram of speaker turns. Models are exported to ONNX and OpenVINO format at build time. Inference runs on standard CPU with no cloud calls.

---

## Pipeline Overview

![Pipeline workflow](illustrations/workflow.svg)

Audio is processed in overlapping 10-second chunks with a 1-second hop. Each chunk goes through two stages:

**1. Segmentation** — PyanNet (SincNet + BiLSTM) runs on each chunk and outputs per-frame powerset probabilities over 7 classes (silence, speaker A, speaker B, speaker A+B, etc.). Frames below a confidence threshold are silenced. Contiguous active frames per speaker slot are collapsed into intervals and merged across chunk boundaries.

**2. Embedding** — Each merged speech interval is encoded into a 256-d L2-normalised vector by WeSpeakerResNet34. fbank features are extracted in Python (torchaudio) rather than relying on the model's internal ops, which don't export cleanly to ONNX.

**3. Clustering** — All embeddings are clustered globally using agglomerative clustering (Ward linkage on L2-normed vectors, cosine-equivalent). Speaker count is either provided or estimated via an eigenvalue gap heuristic on the affinity matrix. Consecutive same-speaker segments within a configurable gap are merged, and short segments below a minimum duration are dropped.

Two modes are supported:

- **`NonCausalDiarization`** — processes the full file, clusters globally. Produces the most accurate output.
- **`CausalDiarization`** — chunk-by-chunk streaming mode. Matches each new embedding against a rolling speaker memory via cosine similarity and EMA updates. No global clustering step; lower latency, less accurate.

---

## How to Run
### Downloade model artifacts

```bash
./download_models.sh
```

### Local (CLI)

```bash
PYTHONPATH=src python cli.py audio/ --models-dir models --num-speakers 2 --output results/
```

### Local (API server)

```bash
PYTHONPATH=src MODELS_DIR=models CONFIG_PATH=config.yaml python server.py
```

```bash
# Health check
curl http://localhost:8000/health

# Submit a file
curl -X POST http://localhost:8000/diarize \
  -F "file=@audio/debate.wav" \
  -F "num_speakers=2"
```

### Docker
```bash
# Build
docker build -t diarization .

# Run API server (models mounted at runtime — not baked into image)
docker run --rm -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  diarization
```

Or with Compose:

```bash
docker compose up
```

The default container entry point is the uvicorn API server. Override with `python cli.py` for batch CLI use.

---

## Benchmark — ONNX Runtime vs OpenVINO (CPU)

Measured on CPU with `time.perf_counter()` over 1000 runs (5 warmup), using random inputs.
Input sizes: segmentation `1×1×160000` (10s chunk), embedding `1×98×80` (1s chunk).

| Model | ONNX mean | ONNX std | OpenVINO mean | OpenVINO std | Winner |
|---|---|---|---|---|---|
| Segmentation (PyanNet) | 40.59ms | ±8.56ms | 48.40ms | ±5.80ms | ONNX ✓ |
| Embedding (WeSpeakerResNet34) | 19.39ms | ±1.75ms | 23.90ms | ±2.17ms | ONNX ✓ |

**ONNX Runtime wins on both models** on this hardware (~0.84x speedup on segmentation, ~0.81x on embedding).

Notably, OpenVINO shows **lower variance** on the segmentation model (±5.80ms vs ±8.56ms), which may matter in latency-sensitive deployments where worst-case tail latency is the constraint rather than average throughput.

> Results are hardware-dependent. OpenVINO is expected to close the gap or outperform on Intel hardware with AVX-512 or on devices with an integrated NPU. 

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|---|---|
| ONNX + OpenVINO, switchable via `--backend` | ONNX for portability and out-of-the-box CPU/CUDA support; OpenVINO for potential gains on Intel hardware with AVX-512 or NPU. On benchmarked hardware ONNX wins (~19ms vs ~24ms on embedding). |
| fbank extracted in Python (torchaudio) | WeSpeakerResNet34's internal fbank ops don't trace cleanly through ONNX export. torchaudio is equivalent and keeps the exported model minimal. |
| Models mounted at runtime, not baked into image | Keeps the Docker image lean and avoids HuggingFace token requirements at build time. Export is a separate offline step. |
| Agglomerative clustering (Ward linkage) | No need to pre-specify `k`; handles variable speaker count naturally. Ward linkage on L2-normed vectors is equivalent to cosine clustering without a custom metric. |
| Eigenvalue gap heuristic for speaker count | Simple, deterministic, no learned model needed. Falls back to 2 on failure (covers the most common case). |
| Causal mode uses EMA speaker memory | Avoids storing all embeddings for global clustering in streaming scenarios. Trade-off: speaker consistency degrades over long files as the memory drifts. |
| `ProcessPoolExecutor` for batch | Inference is CPU-bound — bypasses the GIL for true parallelism across files. |

---

## What I Would Improve Given More Time

- **Speaker count estimation**: replace the eigenvalue heuristic with a learned or calibrated approach (e.g. BIC on the affinity matrix, or a small classifier trained on the gap features).
- **Streaming/WebSocket API**: expose `CausalDiarization` over a real-time WebSocket endpoint for live microphone input.
- **INT8 quantization**: quantise the embedding model for faster CPU throughput, particularly relevant for batch workloads.
- **Proper evaluation harness**: compute DER (Diarization Error Rate) against ground-truth RTTM files so configuration changes have a measurable impact.
- **Chunk boundary artefacts**: the current sliding window can split a speaker turn at a chunk edge. A smarter overlap-and-stitch strategy (e.g. voting on the overlapping region) would reduce boundary errors.


## References Papers and Related Topics
- [1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](https://arxiv.org/abs/2109.08910)
- [2] MS-SincResNet: Joint Learning of 1D and 2D Kernels Using Multi-scale SincNet and ResNet for Music Genre Classification [Arxiv](https://arxiv.org/abs/2109.08910)
- [3] Curricular SincNet: Towards Robust Deep Speaker Recognition by Emphasizing Hard Samples in Latent Space
[Arxiv](https://arxiv.org/abs/2108.10714)
- [4] Interpretable SincNet-based Deep Learning for Emotion Recognition from EEG brain activity [Arxiv](https://arxiv.org/pdf/2107.10790)
- [5] Toward end-to-end interpretable convolutional neural networks for waveform signals [Arxiv](https://arxiv.org/pdf/2405.01815)
- [6] Filterband design for end-to-end speech separation [Arxiv](https://arxiv.org/pdf/1910.10400). This paper decomposes sinNet into a product sin * cos as implemented in this repo and bridgin the gap with Gabor filterbank


## Projects Worth Checking
- https://github.com/mravanelli/SincNet
- https://github.com/mravanelli/pytorch-kaldi
- https://github.com/PeiChunChang/MS-SincResNet
- https://github.com/ZaUt-bio/Exploring-Filters-in-SincNet-Access-and-Visualization/blob/main/SincNet_filters_visualization_initials.ipynb
- https://github.com/wkzng/iSincNet