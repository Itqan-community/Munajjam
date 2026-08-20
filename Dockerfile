# ── Build & Run ────────────────────────────────────────────────────
#   docker build -t munajjam_api .
#
#   GPU: docker run --gpus all -p 8000:8000 \
#          -v model_data:/app/model_local \
#          munajjam_api
#
#   CPU: docker run -p 8000:8000 \
#          -v model_data:/app/model_local \
#          munajjam_api
# ───────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# gcc/g++ — compile ctc_segmentation (C extension)
# ffmpeg/libsndfile — audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git \
        ffmpeg libsndfile1 libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the core library first for dependency installation
COPY munajjam/pyproject.toml /app/munajjam/pyproject.toml
COPY munajjam/README.md /app/munajjam/README.md
COPY munajjam/munajjam /app/munajjam/munajjam/

WORKDIR /app/munajjam
# Install munajjam with the api option
RUN pip install --no-cache-dir ".[api]"
# Install faster-whisper and whisperx explicitly
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git faster-whisper

WORKDIR /app
# Copy the server and entrypoint
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Required directories
RUN mkdir -p /app/model_local/whisper \
             /app/model_local/whisperx \
             /app/temp_audio

ENV HF_TOKEN=""
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
