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
from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs
import json

# Step 1: Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("surah_001.wav")

# Step 2: Load reference ayahs
ayahs = load_surah_ayahs(1)

# Step 3: Align segments to ayahs (uses hybrid strategy with drift fix)
aligner = Aligner(strategy="hybrid")
results = aligner.align(segments, ayahs)

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
├── core/           # Alignment algorithms (Aligner class)
├── transcription/  # Audio transcription (Whisper implementations)
├── models/         # Pydantic data models
├── data/           # Bundled Quran reference data
├── config.py       # Configuration management
└── exceptions.py   # Custom exceptions
```

## Core API

### Aligner Class

The `Aligner` class is the main entry point for alignment:

```python
from munajjam.core import Aligner, AlignmentStrategy

# Create an aligner with default settings (hybrid strategy)
aligner = Aligner()

# Or with custom configuration
aligner = Aligner(
    strategy="hybrid",      # "greedy", "dp", or "hybrid" (recommended)
    quality_threshold=0.85, # Similarity threshold for high-quality alignment
    fix_drift=True,         # Run zone realignment for long surahs
    fix_overlaps=True,      # Fix overlapping ayah timings
)

# Run alignment
results = aligner.align(
    segments=segments,
    ayahs=ayahs,
    silences_ms=silences,   # Optional: silence periods for better boundaries
    on_progress=callback,   # Optional: progress callback (current, total)
)

# Access stats from hybrid alignment
if aligner.last_stats:
    print(f"DP kept: {aligner.last_stats.dp_kept}")
    print(f"Fallback used: {aligner.last_stats.old_fallback}")
```

### Convenience Function

For simple usage with default settings:

```python
from munajjam.core import align

results = align(segments, ayahs)
```

### Alignment Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `greedy` | Fast, simple matching | Quick prototyping |
| `dp` | Dynamic programming for optimal alignment | High accuracy needed |
| `hybrid` | DP with fallback to greedy | **Recommended** - best balance |

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

## Text Utilities

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
