from src.pipeline import CausalDiarization
from src.config import AppConfig

config = AppConfig.from_yaml("config.yaml")

pipeline = CausalDiarization(
    config=config,
    models_dir="models",
    backend="onnx",
)
pipeline.reset()

waveform, duration = pipeline.audio.load("audio/debate.wav")
all_segments = []

for chunk, t_start, t_end in pipeline.audio.sliding_chunks(waveform):
    segments = pipeline.push_chunk(chunk, t_start, t_end, waveform)
    print(segments)
    all_segments.extend(segments)