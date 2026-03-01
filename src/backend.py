from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

BackendType = Literal["onnx", "openvino"]


class InferenceBackend(ABC):
    """Abstract base — both backends implement this interface."""

    @abstractmethod
    def run_segmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        audio : np.ndarray
            Shape (1, 1, 160000) float32

        Returns
        -------
        np.ndarray
            Shape (1, num_frames, 7) — per-frame speaker scores
        """

    @abstractmethod
    def run_embedding(self, fbank: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        fbank : np.ndarray
            Shape (1, num_frames, 80) float32

        Returns
        -------
        np.ndarray
            Shape (1, 256) — L2-normalized speaker embedding
        """




class ONNXBackend(InferenceBackend):
    """ ONNX Runtime inference.
        Supports CPU and CUDA execution providers automatically.
    """

    def __init__(self, models_dir: Path | str) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNXBackend. pip install onnxruntime")

        models_dir = Path(models_dir)
        seg_path = models_dir / "onnx" / "segmentation" / "model.onnx"
        emb_path = models_dir / "onnx" / "embedding" / "model.onnx"

        self._check_paths(seg_path, emb_path)

        # Prefer CUDA if available, fall back to CPU
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._seg_session = ort.InferenceSession(str(seg_path), opts, providers=providers)
        self._emb_session = ort.InferenceSession(str(emb_path), opts, providers=providers)

        self.device = providers[0].replace("ExecutionProvider", "").lower()

    def run_segmentation(self, audio: np.ndarray) -> np.ndarray:
        return self._seg_session.run(None, {"audio": audio.astype(np.float32)})[0]

    def run_embedding(self, fbank: np.ndarray) -> np.ndarray:
        return self._emb_session.run(None, {"fbank": fbank.astype(np.float32)})[0]

    @staticmethod
    def _check_paths(*paths: Path) -> None:
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")



class OpenVINOBackend(InferenceBackend):
    """ OpenVINO inference.
        Uses AUTO device selection (prefers GPU/NPU if available, falls back to CPU).
    """

    def __init__(self, models_dir: Path | str, device: str = "AUTO") -> None:
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("openvino is required for OpenVINOBackend. pip install openvino")

        models_dir = Path(models_dir)
        seg_xml = models_dir / "openvino" / "segmentation" / "model.xml"
        emb_xml = models_dir / "openvino" / "embedding" / "model.xml"

        self._check_paths(seg_xml, emb_xml)

        core = ov.Core()
        self._seg_model = core.compile_model(str(seg_xml), device)
        self._emb_model = core.compile_model(str(emb_xml), device)
        self.device = device

    def run_segmentation(self, audio: np.ndarray) -> np.ndarray:
        result = self._seg_model({"audio": audio.astype(np.float32)})
        return list(result.values())[0]

    def run_embedding(self, fbank: np.ndarray) -> np.ndarray:
        result = self._emb_model({"fbank": fbank.astype(np.float32)})
        return list(result.values())[0]

    @staticmethod
    def _check_paths(*paths: Path) -> None:
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")




def create_backend(backend: BackendType, models_dir: Path | str) -> InferenceBackend:
    """ Factory function — instantiate the right backend from a config string.

        Usage
        -----
        backend = create_backend("onnx", "models/")
        backend = create_backend("openvino", "models/")
    """
    if backend == "onnx":
        return ONNXBackend(models_dir)
    elif backend == "openvino":
        return OpenVINOBackend(models_dir)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'onnx' or 'openvino'.")
    


if __name__ == "__main__":

    backend = create_backend("onnx", models_dir="models/")
    print(backend._seg_session)
    print(backend._emb_session)

    backend = create_backend("openvino", models_dir="models/")
    print(backend._seg_model)
    print(backend._emb_model)
