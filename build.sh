# Build
docker build --no-cache -t diarization .
  
# Test REST
docker run --rm -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  diarization