#!/bin/bash

set -e
BASE_URL="https://github.com/wkzng/speaker-diarization/releases/download/v1.0.0/"

mkdir -p models/onnx/segmentation models/onnx/embedding

echo "Downloading segmentation model..."
curl -L "${BASE_URL}/segmentation.onnx" -o models/onnx/segmentation/model.onnx

echo "Downloading embedding model..."
curl -L "${BASE_URL}/embedding.onnx" -o models/onnx/embedding/model.onnx

echo "Done. Models saved to models/"