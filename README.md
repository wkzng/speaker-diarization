# Speaker Diarization Pipeline

A minimal, production-minded speaker diarization pipeline built on top of the **Community-1** pretrained model. Given an audio file, it outputs a chronogram of speaker turns. Models are exported to ONNX and OpenVINO format at build time. Inference runs on standard CPU with no cloud calls.


## Pipeline Overview

![Pipeline workflow](illustrations/workflow.svg)

The pipeline runs two parallel branches on each audio chunk, then merges them:

- **Segmentation**: PyanNet (SincNet + BiLSTM) detects speech frames and local speaker boundaries
- **Embedding**: WeSpeakerResNet34 encodes each speech segment into a 256-d speaker vector
- **Clustering**: Agglomerative clustering assigns global speaker identities across the full file


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