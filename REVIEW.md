# Speaker Diarization — Design Review

> A chronological walkthrough of how this pipeline was built, the decisions made under time pressure, the trade-offs accepted, and what would be done differently with more time.
> **Context:** ~6 hours on a Sunday evening, interrupted by a consulting commitment.

---

## Step 1 — Understanding the Pipeline Before Touching Code

The first hour was spent not writing code at all.

I extracted a ~20s clip from a political debate on YouTube — a moderator and two politicians, with interruptions and overlapping speech. I ran it through the HuggingFace Community-1 model and observed the raw outputs before reading any implementation.

My approach was deliberately **practical-first, theory-second**: open the model brain, observe what comes out, then connect back to the papers. I intentionally avoided reading the pyannote source code — I wanted to test whether I could reconstruct the implementation from the papers alone.

The papers I read:
- Hervé Bredin — *Powerset multi-class cross entropy loss for diarization* (2023)
- Hervé Bredin — *pyannote.audio 2.1* (2023)
- WeSpeaker — *ResNet-based speaker embedding*

The output of this step is the pipeline diagram below — my own mental model before writing a line of code. The SVG was produced with Claude from the notes of this exploration.

![Pipeline workflow](https://raw.githubusercontent.com/wkzng/speaker-diarization/main/illustrations/workflow.svg)

**Key insight from this step:** the segmentation model does not output one decision per chunk. It outputs 589 micro-decisions per 10s chunk — one per ~17ms frame — over 7 mutually exclusive powerset classes. This is not obvious from the model card and took some time to reconstruct from the paper.

---

## Step 2 — Model Compression for Fast Inference

The instructions required ONNX. I was also curious about OpenVINO — I first encountered it in 2022 during an interview process and was genuinely impressed by the inference speedup at iso-accuracy. I exported both and ran a benchmark.

### Benchmark Results

Measured on CPU with `time.perf_counter()` over 1000 runs (5 warmup), random inputs.

| Model | Backend | Mean | Std | Winner |
|---|---|---|---|---|
| Segmentation (PyanNet) | ONNX | 40.59ms | ±8.56ms | ✓ |
| Segmentation (PyanNet) | OpenVINO | 48.40ms | ±5.80ms | |
| Embedding (WeSpeakerResNet34) | ONNX | 19.39ms | ±1.75ms | ✓ |
| Embedding (WeSpeakerResNet34) | OpenVINO | 23.90ms | ±2.17ms | |

**ONNX wins on mean latency on this hardware.** However, OpenVINO shows lower variance on the segmentation model (±5.80ms vs ±8.56ms) — which matters in latency-sensitive deployments where worst-case tail latency is the constraint rather than average throughput.

> Results are hardware-dependent. OpenVINO is expected to close the gap or outperform on Intel hardware with AVX-512 or an integrated NPU.

Both backends are supported at runtime via `--backend onnx|openvino`.

---

## Step 3 — Audio Preprocessing

The HuggingFace pipeline uses Kaldi-compatible filterbank features. Rather than reimplementing, I kept `torchaudio.compliance.kaldi.fbank` with identical settings to what WeSpeakerResNet34 was trained on (80 mel bins, 25ms window, 10ms shift). Using a different filterbank implementation would silently degrade embedding quality.

**One detail that puzzled me:** the waveform is scaled by ×32768 before being passed to `kaldi.fbank`. This converts float32 in [-1, 1] back to the int16 range that Kaldi expects internally. I took it as-is after confirming it matched the original training pipeline.

**Planned but not completed:** replace `torchaudio.compliance.kaldi.fbank` with a pure numpy/librosa implementation to remove the torch/torchaudio dependency from the production image entirely. The blocker is verification — the outputs must be bit-identical to what the model was trained on, which requires a side-by-side numerical comparison. I didn't have time to do this rigorously.

The chunking logic uses a generator to avoid loading the full waveform into memory multiple times — adapted from a personal project where I had to process long audio streams with bounded memory.

---

## Step 4 — Diarization Pipeline

### Architecture

Shared logic lives in `BaseDiarization`. Two modes derive from it:
- `NonCausalDiarization` — full file, global clustering
- `CausalDiarization` — chunk-by-chunk streaming, rolling speaker memory

### Segmentation → Intervals

For each chunk:
1. Run segmentation model → `(589, 7)` log-softmax
2. Argmax → `(589,)` winning class per frame
3. Gate frames below confidence threshold → silence
4. Apply powerset lookup → per-slot binary mask
5. Find contiguous True regions → `(slot, t_start, t_end)` intervals in absolute time
6. Merge nearby intervals across chunks with configurable `min_silence` gap

**Known weakness:** all segmentation outputs are computed first, then merged. Online interval merging would reduce peak memory. For very long audio, the right approach would be divide-and-conquer — split into blocks, process in parallel, stitch at boundaries.

**Improvement I validated post-submission:** accumulating raw softmax probabilities across overlapping chunks before decoding (overlap-and-vote) produces smoother segmentation. I had the intuition for this during development but didn't have time to implement it. Found afterwards that this is exactly what pyannote does internally — good validation.

### Embedding

Each merged interval is sliced from the waveform and passed to WeSpeakerResNet34. Short segments are zero-padded to a minimum length.

**The zero-padding problem:**

When a speech segment is shorter than the minimum window, the remainder is filled with silence:

```
Short segment (real speech):
[████████░░░░░░░░░░░░░░░░░░░░░░░░]
 ←─ 20% signal ──→←─ 80% zeros ─→

Embedding ≈ weighted average → dominated by silence, not speaker
```

This dilutes the embedding toward a generic "silence" vector — the resulting 256-d representation no longer reliably identifies the speaker, which hurts clustering.

**Planned fix — circular padding:**

```
Short segment repeated to fill window:
[████████████████████████████████]
 ←── copy ───→←── copy ───→←─ copy

Embedding ≈ average of same speaker → stable, noise-reduced
```

Repeating the segment is equivalent to computing an average embedding over multiple repetitions of the same voice — a built-in denoising effect. The more the segment is repeated, the more the embedding stabilises toward the true speaker centroid.

A second planned guard: discard segments where `len(signal) / window_size < threshold` (e.g. 0.3) entirely — if the segment is too short to produce a reliable embedding, including it in clustering only adds noise.

Neither was implemented due to time constraints.

### Clustering

Global agglomerative clustering (Ward linkage on L2-normalised vectors) with automatic speaker count estimation via eigenvalue gap heuristic on the affinity matrix.

**Regret:** I didn't benchmark against DBSCAN. DBSCAN doesn't require pre-specifying k and handles variable-density clusters — it would likely have performed better here, and it would have eliminated the eigenvalue heuristic entirely. Definitely the first thing I'd revisit.

---

## Step 5 — Streaming Mode

`CausalDiarization` replaces global clustering with a **rolling speaker memory** — a dictionary of `{speaker_id → 256-d embedding prototype}`.

For each new embedding:
1. Compute cosine similarity against all known prototypes
2. If best match > `stitch_threshold` (0.7): assign to that speaker, update prototype via EMA
3. Otherwise: new speaker, add to memory

### Connection to VQ-VAE

This is structurally identical to the VQ-VAE codebook lookup:

![VQ-VAE architecture](https://github.com/Vrushank264/VQVAE-PyTorch/raw/main/Results/model_arch.png)

In VQ-VAE: find the nearest codebook vector (argmin of L2 distance), use it as the quantised representation, update the codebook entry via EMA commitment loss.

Here: find the nearest speaker prototype (argmax of cosine similarity), use it as the speaker label, update the prototype via EMA. The denoising effect is the same — as the number of updates increases, the prototype stabilises and becomes a cleaner average representation of that speaker.

**Limitation:** without a global correction pass, speaker consistency degrades over long recordings as the EMA prototype drifts. Two acoustically similar speakers can merge if they appear in adjacent windows. For accurate long-form diarization, `NonCausalDiarization` is the right choice.

---

## Step 6 — Testing

Both pipelines were tested on the debate audio clip. Output was visually inspected against the HuggingFace reference.

**Regret:** I wanted to compute DER (Diarization Error Rate) against the HuggingFace pipeline output to have a quantitative signal. I didn't have time to set up the evaluation properly.

For a moment I considered using DTW to align the two output sequences — then thought better of it. DTW measures sequence similarity, not diarization accuracy. The right tool is `pyannote.metrics` DER, which handles speaker permutation and collar tolerance correctly.

---

## Step 7 — Deployment

FastAPI server with two endpoints:
- `POST /diarize` — REST, upload WAV, get full result
- `WS /ws/diarize` — WebSocket streaming, int16 PCM in, segment JSON out

CLI for batch processing with `ThreadPoolExecutor` (swap to `ProcessPoolExecutor` for true multi-core — one-line change).

Docker image with models mounted at runtime (not baked in) to keep the image lean and avoid HuggingFace token requirements at build time.

---

## What I Would Do With 2–3 More Hours

| Priority | Item |
|---|---|
| 1 | DER evaluation against HuggingFace reference — quantify the gap |
| 2 | Replace `kaldi.fbank` with numpy — remove torch/torchaudio from prod image |
| 3 | Overlap-and-vote on overlapping chunks — smoother segmentation |
| 4 | DBSCAN for clustering — no k estimation needed |
| 5 | INT8 quantization of embedding model — ~2× CPU throughput |
| 6 | Circular padding for short segments — reduce embedding dilution |
| 7 | Live browser UI connecting to `/ws/diarize` via Web Audio API |