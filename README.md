# Munajjam

**A Python library to synchronize Quran ayat with audio recitations.**

Munajjam uses AI-powered speech recognition to automatically generate precise timestamps for each ayah in a Quran audio recording.

## Installation

Clone the repository:

```bash
git clone https://github.com/Itqan-community/munajjam.git
cd munajjam/munajjam
```

Install the package:

```bash
pip install .
```

For faster transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper):

```bash
pip install ".[faster-whisper]"
```

For development (editable install):

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Download a sample recitation

Download a sample audio file (Surah Al-Fatiha):

```bash
curl -L -o 001.mp3 "https://pub-9ee413c8af4041c6bd5223d08f5d0f0f.r2.dev/media/uploads/assets/11/recitations/001.mp3"
```

> **Note:** Audio files should be named by surah number (e.g., `001.mp3`, `002.mp3`).
> Browse more recitations at [cms.itqan.dev](https://cms.itqan.dev)

### 2. Run the alignment

```python
from munajjam.transcription import WhisperTranscriber
from munajjam.core import align
from munajjam.data import load_surah_ayahs

# Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("001.mp3")

# Align to ayahs
ayahs = load_surah_ayahs(1)
results = align(segments, ayahs)

# Get timestamps
for result in results:
    print(f"Ayah {result.ayah.ayah_number}: {result.start_time:.2f}s - {result.end_time:.2f}s")
```

### 3. Output

```
Ayah 1: 5.62s - 9.57s
Ayah 2: 10.51s - 14.72s
Ayah 3: 15.45s - 18.53s
Ayah 4: 19.21s - 22.54s
Ayah 5: 23.27s - 28.19s
Ayah 6: 29.00s - 33.07s
Ayah 7: 33.98s - 46.44s
```

## Features

- **Whisper Transcription** - Uses Tarteel AI's Quran-tuned Whisper models
- **Multiple Alignment Strategies** - Greedy, Dynamic Programming, or Hybrid (recommended)
- **Arabic Text Normalization** - Handles diacritics, hamzas, and character variations
- **Automatic Drift Correction** - Fixes timing drift in long recordings
- **Quality Metrics** - Confidence scores for each aligned ayah
- **High-Performance Rust Core** - Optional Rust acceleration for ~6x faster processing

## Performance

Munajjam uses SIMD-accelerated string matching for fast alignment:

| Implementation | Speed | Installation |
|----------------|-------|--------------|
| **Rust (munajjam_rs)** | **~6x faster** | Optional (see below) |
| **rapidfuzz** | **~4x faster** | Included by default |

### Optional: Rust Core (Maximum Performance)

For maximum performance, install the Rust accelerator:

```bash
cd munajjam-rs
pip install maturin
maturin develop --release
```

The library automatically uses Rust when available, falling back to rapidfuzz.

### Benchmark Results (Apple M1)

```
Implementation    Ops/sec     Speedup
-----------------------------------------
Rust              159,430     5.89x
rapidfuzz         105,743     3.91x
```

## Alignment Strategies

```python
from munajjam.core import Aligner

# Hybrid (recommended) - best balance of speed and accuracy
aligner = Aligner(strategy="hybrid")

# Greedy - fastest, good for clean recordings
aligner = Aligner(strategy="greedy")

# DP - most accurate, slower
aligner = Aligner(strategy="dp")

results = aligner.align(segments, ayahs)
```

## Examples

See the [examples](./examples) directory for more usage patterns:

- `01_basic_usage.py` - Simple transcription and alignment
- `02_comparing_strategies.py` - Compare alignment strategies
- `03_advanced_configuration.py` - Custom settings and options
- `04_batch_processing.py` - Process multiple files

## Requirements

- Python 3.10+
- PyTorch 2.0+
- FFmpeg (for audio processing)

## Community

- [Website](https://munajjam.itqan.dev)
- [ITQAN Community](https://community.itqan.dev)

## Acknowledgments

- [Tarteel AI](https://tarteel.ai) for the Quran-specialized Whisper model

## License

MIT License - see [LICENSE](./LICENSE) for details.
