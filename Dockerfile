FROM python:3.11-slim

# System deps for torchaudio (libsndfile) and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Source code
COPY src/ ./src/
COPY server.py cli.py config.yaml ./

# Models are mounted at runtime — not baked into image (keeps image small)
# docker run -v $(pwd)/models:/app/models ...
ENV MODELS_DIR=/app/models \
    CONFIG_PATH=/app/config.yaml \
    BACKEND=onnx \
    PYTHONPATH=/app/src

# Default: run the API server
# Override with: docker run ... python cli.py audio/ --output results/
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]