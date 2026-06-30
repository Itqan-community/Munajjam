#!/usr/bin/env bash
set -euo pipefail

# Create required directories
mkdir -p /app/model_local/whisper
mkdir -p /app/model_local/whisperx
mkdir -p /app/temp_audio

echo "Starting server on 0.0.0.0:8000..."

# CUDA diagnostics
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'[GPU] CUDA OK — {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}')
else:
    print('[GPU] WARNING: CUDA not available — models will run on CPU')
"

exec uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info
