from src.pipeline import Diarization
from src.config import AppConfig

config = AppConfig.from_yaml("config.yaml")


pipeline = Diarization(
    config=config,
    models_dir="models",
    backend="onnx",
    num_speakers=2
)
result = pipeline("audio/debate.wav")
print(result.to_json())

