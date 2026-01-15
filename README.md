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

```python
from munajjam.transcription import WhisperTranscriber
from munajjam.core import align
from munajjam.data import load_surah_ayahs

# 1. Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("surah_001.wav")

# 2. Align to ayahs
ayahs = load_surah_ayahs(1)
results = align(segments, ayahs)

# 3. Get timestamps
for result in results:
    print(f"Ayah {result.ayah.ayah_number}: {result.start_time:.2f}s - {result.end_time:.2f}s")
```

## Features

- **Whisper Transcription** - Uses Tarteel AI's Quran-tuned Whisper models
- **Multiple Alignment Strategies** - Greedy, Dynamic Programming, or Hybrid (recommended)
- **Arabic Text Normalization** - Handles diacritics, hamzas, and character variations
- **Automatic Drift Correction** - Fixes timing drift in long recordings
- **Quality Metrics** - Confidence scores for each aligned ayah

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
