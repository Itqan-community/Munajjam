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

# Align to ayahs (uses auto strategy by default; override with "greedy", "dp", or "hybrid")
ayahs = load_surah_ayahs(1)
results = align("001.mp3", segments, ayahs)

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

- **Whisper Transcription** - Uses faster-whisper as default backend with Quran-tuned models
- **Four Alignment Strategies** - Auto, Hybrid, DP, and Greedy
- **Arabic Text Normalization** - Handles diacritics, hamzas, and character variations
- **Automatic Drift Correction** - Multi-pass zone realignment for long recordings
- **Quality Metrics** - Confidence scores for each aligned ayah
- **Phonetic Similarity** - Arabic ASR confusion-aware similarity scoring
- **Word-level Precision** - Uses per-word timestamps (when available) to improve drift recovery

## Alignment Strategies

The default `auto` strategy works best for most cases. You can override it:

```python
from munajjam.core import Aligner

# Auto (recommended) - picks the best strategy, full pipeline by default
aligner = Aligner("001.mp3")

# Hybrid - DP with greedy fallback (legacy)
aligner = Aligner("001.mp3", strategy="hybrid")

# Greedy - fastest, good for clean recordings
aligner = Aligner("001.mp3", strategy="greedy")

# DP - optimal alignment using dynamic programming
aligner = Aligner("001.mp3", strategy="dp")

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
- [pyquran-modern](https://github.com/mmzerda/pyquran-modern): A modern fork of pyquran supporting Python 3.13 and 114 Surahs.

## Acknowledgments

- [Tarteel AI](https://tarteel.ai) for the Quran-specialized Whisper model

## License

MIT License - see [LICENSE](./LICENSE) for details.
