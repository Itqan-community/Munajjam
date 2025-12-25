# مُنَجِّم (Munajjam)

> A Python library to synchronize Quran Ayat with audio recitations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

Munajjam is a Python library that:

- **Transcribes** Quran audio recitations using Whisper models fine-tuned for Arabic Quran
- **Aligns** transcribed text with canonical Quran ayahs
- **Provides precise timestamps** for each ayah in the audio
- **Outputs JSON** with timing information for further processing

## Installation

```bash
pip install munajjam
```

For faster transcription using CTranslate2:

```bash
pip install munajjam[faster-whisper]
```

## Quick Start

```python
from munajjam.transcription import WhisperTranscriber
from munajjam.core.aligner import align_segments
from munajjam.data import load_surah_ayahs
import json

# Step 1: Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("surah_001.wav")

# Step 2: Load reference ayahs
ayahs = load_surah_ayahs(1)

# Step 3: Align segments to ayahs
results = align_segments(segments, ayahs)

# Step 4: Output as JSON
output = []
for result in results:
    output.append({
        "ayah_number": result.ayah.ayah_number,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "text": result.ayah.text,
        "similarity_score": result.similarity_score,
    })

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
```

## Configuration

Configure via environment variables:

```bash
export MUNAJJAM_MODEL_ID="tarteel-ai/whisper-base-ar-quran"
export MUNAJJAM_DEVICE="cuda"
export MUNAJJAM_SIMILARITY_THRESHOLD="0.7"
```

Or programmatically:

```python
from munajjam import configure

settings = configure(
    model_id="tarteel-ai/whisper-base-ar-quran",
    device="cuda",
    similarity_threshold=0.7,
)
```

## Package Structure

```
munajjam/
├── core/           # Core algorithms (alignment, matching, Arabic normalization)
├── transcription/  # Audio transcription (Whisper implementation)
├── models/         # Pydantic data models
├── data/           # Bundled Quran reference data
├── config.py       # Configuration management
└── exceptions.py   # Custom exceptions
```

## Models

### Ayah
```python
from munajjam import Ayah

ayah = Ayah(
    id=1,
    surah_id=1,
    ayah_number=1,
    text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
)
```

### Segment
```python
from munajjam import Segment, SegmentType

segment = Segment(
    id=1,
    surah_id=1,
    start=0.0,
    end=5.32,
    text="بسم الله الرحمن الرحيم",
    type=SegmentType.AYAH,
)
```

### AlignmentResult
```python
from munajjam import AlignmentResult

# Contains aligned ayah with timing and quality metrics
result.ayah              # The matched Ayah
result.start_time        # Start timestamp (seconds)
result.end_time          # End timestamp (seconds)
result.similarity_score  # Match quality (0.0-1.0)
result.is_high_confidence  # True if score >= 0.8
```

## Core Functions

### Arabic Text Normalization

```python
from munajjam.core import normalize_arabic

normalized = normalize_arabic("بِسْمِ اللَّهِ")
# Returns: "بسم الله"
```

### Similarity Matching

```python
from munajjam.core import similarity

score = similarity("بسم الله", "بسم الله الرحمن")
# Returns: 0.75
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/abdullahmosaibah/munajjam
cd munajjam/munajjam
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy munajjam

# Linting
ruff check munajjam
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Tarteel AI](https://tarteel.ai/) for the Quran-specific Whisper models
- The Quran text data is sourced from publicly available Islamic texts
