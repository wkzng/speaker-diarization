from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import AppConfig
from pipeline import Diarization
from schema import DiarizationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def to_rttm(result: DiarizationResult) -> str:
    file_id = Path(result.file).stem
    lines = []
    for seg in result.segments:
        dur = seg.end - seg.start
        lines.append(f"SPEAKER {file_id} 1 {seg.start:.3f} {dur:.3f} <NA> <NA> {seg.speaker} <NA> <NA>")
    return "\n".join(lines)


def process_file(path, pipeline, output_dir, fmt):
    try:
        t0 = time.perf_counter()
        result = pipeline(path)
        elapsed = time.perf_counter() - t0
        rtf = elapsed / result.duration if result.duration > 0 else 0

        output = to_rttm(result) if fmt == "rttm" else result.to_json()
        ext = ".rttm" if fmt == "rttm" else ".json"

        if output_dir:
            out_path = output_dir / (path.stem + ext)
            out_path.write_text(output)

        return path, True, f"{path.name} | {result.num_speakers} spk | RTF={rtf:.2f}"
    except Exception as e:
        return path, False, f"{path.name} FAILED: {e}"


def main():
    parser = argparse.ArgumentParser(description="Speaker diarization — batch CLI")
    parser.add_argument("input", type=Path)
    parser.add_argument("--config",       type=Path, default=Path("config.yaml"))
    parser.add_argument("--models-dir",   type=Path, default=Path("models"))
    parser.add_argument("--backend",      choices=["onnx", "openvino"], default="onnx")
    parser.add_argument("--num-speakers", type=int,  default=None)
    parser.add_argument("--workers",      type=int,  default=1)
    parser.add_argument("--output",       type=Path, default=None)
    parser.add_argument("--format",       choices=["json", "rttm"], default="json")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config) if args.config.exists() else AppConfig.default()
    files = sorted(args.input.glob("**/*.wav")) if args.input.is_dir() else [args.input]

    if not files:
        logger.error("No WAV files found"); sys.exit(1)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(files)} file(s) | workers={args.workers}")

    def make_pipeline():
        return Diarization(
            models_dir=args.models_dir, backend=args.backend,
            num_speakers=args.num_speakers, config=cfg,
        )

    failed = []
    t0 = time.perf_counter()

    if args.workers == 1:
        pipeline = make_pipeline()
        for path in files:
            _, ok, msg = process_file(path, pipeline, args.output, args.format)
            (logger.info if ok else logger.error)(msg)
            if not ok: failed.append(path)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, p, make_pipeline(), args.output, args.format): p for p in files}
            for f in as_completed(futures):
                _, ok, msg = f.result()
                (logger.info if ok else logger.error)(msg)
                if not ok: failed.append(futures[f])

    logger.info(f"Done: {len(files)-len(failed)}/{len(files)} in {time.perf_counter()-t0:.1f}s")
    if failed: sys.exit(1)


if __name__ == "__main__":
    main()