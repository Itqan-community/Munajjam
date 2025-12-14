# 🎯 Munajjam v2.0 Roadmap

> **مُنَجِّم** — A Python library to synchronize Quran Ayat with audio recitations

## Vision

Transform Munajjam from a standalone script into a **professional-grade Python library** that can be:

- **Imported** as a Python package in other projects
- **Used as a backend** for web/mobile applications
- **Async-ready** for high-throughput processing
- **Observable** via hooks for monitoring, logging, and telemetry

---

## 🏗️ Architecture Overview

### Current State (v1.x)

```
main.py → transcribe.py → align_segments.py → save_to_db.py
    ↓         ↓                 ↓                  ↓
  Files     Files             Files             SQLite
```

### Target State (v2.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                        munajjam (library)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Core      │  │   Models    │  │   Storage   │              │
│  │  - Aligner  │  │  - Ayah     │  │  - Base     │              │
│  │  - Matcher  │  │  - Segment  │  │  - SQLite   │              │
│  │  - Arabic   │  │  - Surah    │  │  - JSON     │              │
│  └─────────────┘  │  - Result   │  │  - Custom   │              │
│                   └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Transcriber │  │   Hooks     │  │   Config    │              │
│  │  - Whisper  │  │  - Events   │  │  - Settings │              │
│  │  - Custom   │  │  - Quality  │  │  - Paths    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐                                                │
│  │   Async     │                                                │
│  │  - Tasks    │                                                │
│  │  - Queue    │                                                │
│  └─────────────┘                                                │
├─────────────────────────────────────────────────────────────────┤
│                      Public API Layer                            │
│   Munajjam.sync() | Munajjam.transcribe() | Munajjam.align()    │
├─────────────────────────────────────────────────────────────────┤
│                      Hooks / Callbacks                           │
│   on_transcription_* | on_alignment_* | on_quality_report       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Proposed Package Structure

```
munajjam/
├── pyproject.toml           # Modern Python packaging
├── README.md
│
├── munajjam/                # Core library
│   ├── __init__.py          # from munajjam import Munajjam
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── aligner.py       # Ayah-segment alignment logic
│   │   ├── matcher.py       # Similarity matching algorithms
│   │   ├── arabic.py        # Arabic text normalization
│   │   └── overlap.py       # Overlap detection & removal
│   │
│   ├── transcription/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract transcriber interface
│   │   ├── whisper.py       # Tarteel Whisper implementation
│   │   └── silence.py       # Silence detection
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ayah.py          # Ayah data model
│   │   ├── segment.py       # Audio segment model
│   │   ├── surah.py         # Surah metadata
│   │   ├── result.py        # Alignment result model
│   │   ├── recitation.py    # Recitation session model
│   │   └── quality.py       # QualityReport model
│   │
│   ├── hooks/
│   │   ├── __init__.py      # Hook exports
│   │   ├── base.py          # MunajjamHooks base class
│   │   └── events.py        # Event definitions
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract storage interface
│   │   ├── sqlite.py        # SQLite implementation
│   │   ├── json.py          # JSON file storage
│   │   └── memory.py        # In-memory (for testing)
│   │
│   ├── async_/
│   │   ├── __init__.py
│   │   ├── tasks.py         # Async task wrappers
│   │   └── queue.py         # Task queue management
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── quran.py         # Quran text loader
│   │   ├── surah_names.py   # Surah metadata
│   │   └── quran_ayat.csv   # Canonical ayah text (bundled)
│   │
│   ├── config.py            # Configuration management
│   ├── exceptions.py        # Custom exceptions
│   └── logging.py           # Structured logging
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures
│   ├── test_aligner.py
│   ├── test_matcher.py
│   ├── test_arabic.py
│   ├── test_transcriber.py
│   ├── test_storage.py
│   ├── test_hooks.py        # Hook system tests
│   ├── test_async.py
│   └── fixtures/            # Test audio samples, expected outputs
│
└── examples/
    ├── basic_usage.py
    ├── with_hooks.py        # Using hooks for monitoring
    ├── batch_processing.py
    ├── async_processing.py
    └── custom_storage.py
```

---

## 🚀 Milestones

### Phase 1: Foundation (v2.0-alpha)

**Goal:** Restructure codebase into clean library architecture

- [ ] **1.1** Create package structure with `pyproject.toml`
- [ ] **1.2** Define data models using Pydantic
  - `Ayah`, `Segment`, `Surah`, `AlignmentResult`, `Recitation`, `QualityReport`
- [ ] **1.3** Extract core business logic
  - Arabic text normalization (`core/arabic.py`)
  - Similarity matching (`core/matcher.py`)
  - Overlap detection (`core/overlap.py`)
  - Alignment algorithm (`core/aligner.py`)
- [ ] **1.4** Abstract transcription interface
  - Base interface (`transcription/base.py`)
  - Whisper implementation (`transcription/whisper.py`)
- [ ] **1.5** Abstract storage interface
  - Base interface (`storage/base.py`)
  - SQLite implementation (`storage/sqlite.py`)
- [ ] **1.6** Configuration management
  - Pydantic settings with environment variable support
- [ ] **1.7** Custom exceptions
  - `MunajjamError`, `TranscriptionError`, `AlignmentError`, etc.
- [ ] **1.8** Hooks system
  - Base hooks class (`hooks/base.py`)
  - Event definitions (`hooks/events.py`)
  - Quality report model (`models/quality.py`)

### Phase 2: Public API (v2.0-beta)

**Goal:** Clean, intuitive public interface

- [ ] **2.1** Design main `Munajjam` class API

  ```python
  from munajjam import Munajjam

  m = Munajjam()
  result = m.sync(audio_path="surah_1.wav", surah_id=1)
  # or step-by-step:
  segments = m.transcribe(audio_path)
  aligned = m.align(segments, surah_id=1)
  m.save(aligned)
  ```

- [ ] **2.2** Implement builder pattern for configuration

  ```python
  m = Munajjam.builder()
      .with_model("tarteel-ai/whisper-base-ar-quran")
      .with_storage(SQLiteStorage("quran.db"))
      .with_reciter("عمر النبراوي")
      .with_hooks(MyHooks())
      .build()
  ```

- [ ] **2.3** Integrate hooks into all pipeline stages
  - Transcription hooks
  - Alignment hooks
  - Quality report hooks

### Phase 3: Async Support (v2.0-rc)

**Goal:** High-throughput async processing

- [ ] **3.1** Async transcription

  ```python
  segments = await m.transcribe_async(audio_path)
  ```

- [ ] **3.2** Async alignment

  ```python
  result = await m.align_async(segments, surah_id)
  ```

- [ ] **3.3** Batch processing with concurrency control
  ```python
  results = await m.sync_batch(
      audio_files=["s1.wav", "s2.wav", "s3.wav"],
      max_concurrent=3
  )
  ```

### Phase 4: Testing (v2.0)

**Goal:** Comprehensive test coverage

- [ ] **4.1** Unit tests for core modules
  - Arabic normalization (parametrized tests)
  - Similarity matching algorithms
  - Overlap detection
  - Hooks system
- [ ] **4.2** Integration tests
  - Full sync pipeline with test audio
  - Storage backends
  - Hooks integration
- [ ] **4.3** Performance benchmarks
  - Alignment speed
  - Memory usage
- [ ] **4.4** Test fixtures
  - Short audio samples (~10 seconds)
  - Expected alignment outputs
  - Edge cases (overlaps, short ayahs)

### Phase 5: Documentation & Release (v2.0)

**Goal:** Production-ready release

- [ ] **5.1** API documentation (auto-generated)
- [ ] **5.2** Getting started guide
- [ ] **5.3** Hooks documentation & examples
- [ ] **5.4** Architecture documentation
- [ ] **5.5** Contributing guide
- [ ] **5.6** Changelog
- [ ] **5.7** PyPI publication

---

## 📋 Detailed Design Decisions

### Data Models (Pydantic)

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class Ayah(BaseModel):
    """Canonical Quran ayah"""
    id: int
    surah_id: int
    ayah_number: int
    text: str

class Segment(BaseModel):
    """Transcribed audio segment"""
    id: int
    start: float  # seconds
    end: float
    text: str
    confidence: Optional[float] = None

class AlignmentResult(BaseModel):
    """Result of aligning a segment to an ayah"""
    ayah: Ayah
    start_time: float
    end_time: float
    transcribed_text: str
    similarity_score: float
    overlap_detected: bool

class Recitation(BaseModel):
    """A complete recitation processing session"""
    id: UUID
    reciter_name: str
    surah_id: int
    created_at: datetime
    alignments: list[AlignmentResult]
    status: str  # pending, processing, completed, failed

class QualityReport(BaseModel):
    """Quality metrics for a sync operation (used by hooks)"""
    # Identification
    recitation_id: UUID
    surah_id: int

    # Alignment quality
    total_ayahs: int
    aligned_ayahs: int
    avg_similarity_score: float
    min_similarity_score: float
    low_confidence_count: int      # ayahs below threshold

    # Processing stats
    total_segments: int
    segments_merged: int
    overlaps_detected: int

    # Performance
    transcription_time_seconds: float
    alignment_time_seconds: float
    total_time_seconds: float

    # Issues
    failed_ayahs: list[int]        # ayah numbers that failed
    warnings: list[str]
```

---

### 🪝 Hooks System

The hooks system allows users to observe and react to events during processing.

#### Base Hooks Class

```python
# hooks/base.py
from typing import Optional
from munajjam.models import Segment, Ayah, AlignmentResult, Recitation, QualityReport

class MunajjamHooks:
    """
    Base class for Munajjam hooks.

    Override any method to handle that event.
    All methods have default no-op implementations.
    """

    # ============ Transcription Hooks ============

    def on_transcription_start(self, audio_path: str, surah_id: int) -> None:
        """Called when transcription begins."""
        pass

    def on_segment_transcribed(self, segment: Segment) -> None:
        """Called after each segment is transcribed."""
        pass

    def on_transcription_complete(self, segments: list[Segment]) -> None:
        """Called when all segments are transcribed."""
        pass

    # ============ Alignment Hooks ============

    def on_alignment_start(self, surah_id: int, total_ayahs: int) -> None:
        """Called when alignment begins."""
        pass

    def on_ayah_aligned(self, result: AlignmentResult) -> None:
        """Called after each ayah is aligned."""
        pass

    def on_segments_merged(self, ayah: Ayah, segment_count: int) -> None:
        """Called when segments are merged for an ayah."""
        pass

    def on_overlap_detected(self, ayah: Ayah, overlap_text: str) -> None:
        """Called when overlap is detected and removed."""
        pass

    def on_alignment_complete(self, results: list[AlignmentResult]) -> None:
        """Called when alignment finishes."""
        pass

    # ============ Quality Hooks ============

    def on_quality_report(self, report: QualityReport) -> None:
        """
        Called with quality metrics after sync completes.

        This is the primary hook for telemetry/analytics.
        """
        pass

    # ============ Progress Hooks ============

    def on_progress(self, current: int, total: int, stage: str) -> None:
        """
        Called to report progress.

        Args:
            current: Current item number
            total: Total items to process
            stage: "transcription" | "alignment" | "saving"
        """
        pass

    # ============ Error Hooks ============

    def on_warning(self, message: str, context: Optional[dict] = None) -> None:
        """Called for non-fatal warnings."""
        pass

    def on_error(self, error: Exception, context: Optional[dict] = None) -> None:
        """Called when an error occurs."""
        pass

    def on_retry(self, attempt: int, max_attempts: int, reason: str) -> None:
        """Called when an operation is retried."""
        pass

    # ============ Lifecycle Hooks ============

    def on_sync_start(self, audio_path: str, surah_id: int, reciter: str) -> None:
        """Called when sync() begins."""
        pass

    def on_sync_complete(self, recitation: Recitation) -> None:
        """Called when sync() completes successfully."""
        pass
```

#### Usage Examples

```python
from munajjam import Munajjam
from munajjam.hooks import MunajjamHooks
from munajjam.models import QualityReport, AlignmentResult

# Example 1: Simple progress logging
class ProgressHooks(MunajjamHooks):
    def on_progress(self, current: int, total: int, stage: str):
        print(f"[{stage}] {current}/{total}")

    def on_ayah_aligned(self, result: AlignmentResult):
        print(f"✅ Ayah {result.ayah.ayah_number}: {result.similarity_score:.2f}")

m = Munajjam(hooks=ProgressHooks())
m.sync("surah_1.wav", surah_id=1)


# Example 2: Quality monitoring
class QualityMonitorHooks(MunajjamHooks):
    def on_quality_report(self, report: QualityReport):
        if report.avg_similarity_score < 0.7:
            print(f"⚠️ Low quality alignment: {report.avg_similarity_score:.2f}")
        if report.low_confidence_count > 0:
            print(f"⚠️ {report.low_confidence_count} ayahs with low confidence")
        print(f"📊 Processed in {report.total_time_seconds:.1f}s")


# Example 3: Telemetry (user opts in)
class TelemetryHooks(MunajjamHooks):
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint

    def on_quality_report(self, report: QualityReport):
        # Send anonymized metrics to your cloud
        import requests
        requests.post(self.api_endpoint, json={
            "surah_id": report.surah_id,
            "avg_score": report.avg_similarity_score,
            "ayah_count": report.total_ayahs,
            "time_seconds": report.total_time_seconds,
            "version": "2.0.0"
        })

# User explicitly opts in
m = Munajjam(hooks=TelemetryHooks("https://api.munajjam.io/telemetry"))


# Example 4: Multiple hooks (combine behaviors)
class CompositeHooks(MunajjamHooks):
    def __init__(self, *hooks: MunajjamHooks):
        self.hooks = hooks

    def on_quality_report(self, report):
        for hook in self.hooks:
            hook.on_quality_report(report)

    # ... delegate other methods similarly

m = Munajjam(hooks=CompositeHooks(
    ProgressHooks(),
    QualityMonitorHooks(),
))
```

---

### Core Interfaces

```python
# transcription/base.py
from abc import ABC, abstractmethod

class BaseTranscriber(ABC):
    """Abstract interface for audio transcription"""

    @abstractmethod
    def transcribe(self, audio_path: str) -> list[Segment]:
        """Transcribe audio file to segments"""
        pass

    @abstractmethod
    async def transcribe_async(self, audio_path: str) -> list[Segment]:
        """Async transcription"""
        pass

# storage/base.py
class BaseStorage(ABC):
    """Abstract interface for result storage"""

    @abstractmethod
    def save_recitation(self, recitation: Recitation) -> None:
        pass

    @abstractmethod
    def get_recitation(self, recitation_id: UUID) -> Optional[Recitation]:
        pass

    @abstractmethod
    def list_recitations(self, reciter: str = None) -> list[Recitation]:
        pass
```

### Configuration

```python
from pydantic_settings import BaseSettings

class MunajjamSettings(BaseSettings):
    """Configuration with environment variable support"""

    # Model settings
    model_id: str = "tarteel-ai/whisper-base-ar-quran"
    device: str = "auto"  # auto, cpu, cuda

    # Audio processing
    silence_threshold_db: int = -30
    min_silence_ms: int = 300

    # Alignment
    similarity_threshold: float = 0.6
    n_check_words: int = 3

    # Storage
    database_url: str = "sqlite:///munajjam.db"

    class Config:
        env_prefix = "MUNAJJAM_"
```

---

## 🎯 API Usage Examples

### Basic Usage

```python
from munajjam import Munajjam

# Simple one-liner
m = Munajjam()
result = m.sync("path/to/surah_1.wav", surah_id=1, reciter="عمر النبراوي")

print(f"Aligned {len(result.alignments)} ayahs")
for alignment in result.alignments:
    print(f"Ayah {alignment.ayah.ayah_number}: {alignment.start_time}s - {alignment.end_time}s")
```

### With Hooks (Progress & Quality Monitoring)

```python
from munajjam import Munajjam
from munajjam.hooks import MunajjamHooks

class MyHooks(MunajjamHooks):
    def on_progress(self, current, total, stage):
        percent = (current / total) * 100
        print(f"[{stage}] {percent:.0f}% ({current}/{total})")

    def on_ayah_aligned(self, result):
        emoji = "✅" if result.similarity_score > 0.8 else "⚠️"
        print(f"{emoji} Ayah {result.ayah.ayah_number}: {result.similarity_score:.2f}")

    def on_quality_report(self, report):
        print(f"\n📊 Quality Report:")
        print(f"   Average similarity: {report.avg_similarity_score:.2f}")
        print(f"   Low confidence ayahs: {report.low_confidence_count}")
        print(f"   Total time: {report.total_time_seconds:.1f}s")

m = Munajjam(hooks=MyHooks())
result = m.sync("surah_1.wav", surah_id=1, reciter="عمر النبراوي")
```

### Step-by-Step Processing

```python
from munajjam import Munajjam
from munajjam.storage import SQLiteStorage

m = Munajjam(
    storage=SQLiteStorage("my_quran.db"),
    reciter="محمود خليل الحصري"
)

# Step 1: Transcribe
segments = m.transcribe("surah_2.wav")
print(f"Got {len(segments)} segments")

# Step 2: Align
alignments = m.align(segments, surah_id=2)

# Step 3: Validate
if len(alignments) != m.get_ayah_count(surah_id=2):
    print("Warning: Some ayahs may be missing!")

# Step 4: Save
m.save(alignments)
```

### Async Batch Processing

```python
import asyncio
from munajjam import Munajjam

async def process_quran():
    m = Munajjam()

    # Process multiple surahs concurrently
    tasks = [
        m.sync_async(f"surah_{i}.wav", surah_id=i)
        for i in range(1, 115)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if not isinstance(r, Exception))
    print(f"Successfully processed {successful}/114 surahs")

asyncio.run(process_quran())
```

### Using as Backend Service

```python
from munajjam import Munajjam
from munajjam.storage import SQLiteStorage

# Initialize once at app startup
munajjam = Munajjam(
    storage=SQLiteStorage("production.db"),
    model_id="tarteel-ai/whisper-base-ar-quran"
)

# Use in your web framework of choice
def handle_sync_request(audio_file, surah_id: int, reciter: str):
    """Called by your web framework (FastAPI, Flask, Django, etc.)"""
    result = munajjam.sync(audio_file, surah_id=surah_id, reciter=reciter)
    return {
        "recitation_id": str(result.id),
        "ayah_count": len(result.alignments),
        "status": result.status
    }
```

---

## 🤝 How to Contribute

We welcome contributions! Here's how you can help:

### Good First Issues

- [ ] Add type hints to existing modules
- [ ] Write unit tests for `normalize_arabic()`
- [ ] Add docstrings following Google style
- [ ] Create test fixtures with short audio samples

### Intermediate

- [ ] Implement JSON storage backend
- [ ] Add support for different Whisper model sizes
- [ ] Create example hook implementations

### Advanced

- [ ] Implement async transcription pipeline
- [ ] Add word-level timestamp extraction
- [ ] Performance optimization for large files

### Code Style

- Use **Black** for formatting
- Use **Ruff** for linting
- Use **mypy** for type checking
- Follow **Google docstring** style
- Write tests with **pytest**

---

## 📊 Success Metrics

| Metric        | Current | Target v2.0     |
| ------------- | ------- | --------------- |
| Test Coverage | 0%      | >80%            |
| Type Hints    | Partial | 100%            |
| Documentation | Basic   | Full API docs   |
| PyPI Ready    | No      | Yes             |
| Async Support | No      | Yes             |
| Hooks System  | No      | Yes             |
| Plugin System | No      | Storage plugins |

---

## 🗓️ Timeline

| Phase               | Duration  | Target Date |
| ------------------- | --------- | ----------- |
| Phase 1: Foundation | 2-3 weeks | TBD         |
| Phase 2: Public API | 2 weeks   | TBD         |
| Phase 3: Async      | 2 weeks   | TBD         |
| Phase 4: Testing    | 2 weeks   | TBD         |
| Phase 5: Release    | 1 week    | TBD         |

---

## 💡 Future Ideas (v2.x+)

- **Word-level timestamps** — Extract precise word boundaries
- **Multiple ASR backends** — Support for other Arabic ASR models
- **Quality scoring** — Automatic alignment quality assessment
- **Export formats** — SRT subtitles, VTT, Audacity labels
- **Tajweed detection** — Identify tajweed rules in recitation
- **Comparison mode** — Compare alignments across reciters
- **Built-in telemetry module** — Optional cloud reporting (opt-in)

---

## 🙏 Acknowledgments

- [Tarteel AI](https://tarteel.ai/) for the Quran-specialized Whisper model
- The open source community for inspiring this project

---

**Let's build something beautiful for the Ummah together! 🌙**
