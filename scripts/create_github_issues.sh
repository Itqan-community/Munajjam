#!/bin/bash

# Munajjam v2.0 GitHub Issues Creation Script
# Repository: Itqan-community/Munajjam
#
# Prerequisites:
#   1. Install GitHub CLI: brew install gh
#   2. Authenticate: gh auth login
#
# Usage:
#   chmod +x scripts/create_github_issues.sh
#   ./scripts/create_github_issues.sh

REPO="Itqan-community/Munajjam"

echo "🚀 Creating GitHub issues for Munajjam v2.0 Phase 1..."
echo "Repository: $REPO"
echo ""

# ============================================================
# Create Labels First
# ============================================================
echo "📌 Creating labels..."

gh label create "phase-1-foundation" --repo $REPO --color "1d76db" --description "Phase 1: Foundation (v2.0-alpha)" 2>/dev/null || echo "  Label 'phase-1-foundation' already exists"
gh label create "good first issue" --repo $REPO --color "7057ff" --description "Good for newcomers" 2>/dev/null || echo "  Label 'good first issue' already exists"
gh label create "help wanted" --repo $REPO --color "008672" --description "Extra attention is needed" 2>/dev/null || echo "  Label 'help wanted' already exists"

echo ""
echo "✅ Labels created!"
echo ""

# ============================================================
# PHASE 1: Foundation (v2.0-alpha)
# ============================================================
echo "📦 Creating Phase 1 issues (Foundation)..."

gh issue create --repo $REPO \
  --title "[1.1] Create package structure with pyproject.toml" \
  --label "phase-1-foundation,good first issue" \
  --body "## Description
Set up the modern Python package structure for Munajjam library.

## Tasks
- [ ] Create \`pyproject.toml\` with project metadata
- [ ] Configure build system (setuptools or hatch)
- [ ] Set up package structure under \`munajjam/\`
- [ ] Add \`__init__.py\` files with proper exports
- [ ] Configure dependencies (pydantic, torch, transformers, etc.)

## Proposed Structure
\`\`\`
munajjam/
├── pyproject.toml
├── README.md
├── munajjam/
│   ├── __init__.py
│   ├── core/
│   ├── models/
│   ├── transcription/
│   ├── storage/
│   ├── hooks/
│   └── ...
└── tests/
\`\`\`

## Acceptance Criteria
- [ ] Package can be installed with \`pip install -e .\`
- [ ] \`from munajjam import Munajjam\` works
- [ ] All dependencies are properly declared

## References
- [Python Packaging Guide](https://packaging.python.org/)
- [pyproject.toml specification](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)"

gh issue create --repo $REPO \
  --title "[1.2] Define data models using Pydantic" \
  --label "phase-1-foundation,good first issue" \
  --body "## Description
Create Pydantic data models for type-safe data handling throughout the library.

## Models to Create
- [ ] \`Ayah\` - Canonical Quran ayah (id, surah_id, ayah_number, text)
- [ ] \`Segment\` - Transcribed audio segment (id, start, end, text, confidence)
- [ ] \`Surah\` - Surah metadata (id, name_arabic, name_english, ayah_count)
- [ ] \`AlignmentResult\` - Result of aligning segment to ayah
- [ ] \`Recitation\` - Complete recitation session
- [ ] \`QualityReport\` - Quality metrics for hooks/telemetry

## File Structure
\`\`\`
munajjam/models/
├── __init__.py
├── ayah.py
├── segment.py
├── surah.py
├── result.py
├── recitation.py
└── quality.py
\`\`\`

## Example Model
\`\`\`python
from pydantic import BaseModel
from typing import Optional

class Ayah(BaseModel):
    id: int
    surah_id: int
    ayah_number: int
    text: str
    
    class Config:
        frozen = True  # Immutable
\`\`\`

## Acceptance Criteria
- [ ] All models have proper type hints
- [ ] Models are immutable where appropriate
- [ ] Validation rules are in place (e.g., surah_id 1-114)
- [ ] Models can be serialized to/from JSON"

gh issue create --repo $REPO \
  --title "[1.3] Extract core business logic" \
  --label "phase-1-foundation" \
  --body "## Description
Extract and refactor core alignment logic from current scripts into clean, testable modules.

## Modules to Create

### \`core/arabic.py\` - Arabic text normalization
- [ ] \`normalize_arabic(text)\` - Full normalization
- [ ] \`remove_diacritics(text)\` - Remove tashkeel
- [ ] \`normalize_alef(text)\` - Normalize أ إ آ → ا
- [ ] \`get_words(text)\` - Split into words

### \`core/matcher.py\` - Similarity matching
- [ ] \`calculate_similarity(text1, text2)\` - Using SequenceMatcher
- [ ] \`get_first_words(text, n)\` - First n words
- [ ] \`get_last_words(text, n)\` - Last n words
- [ ] \`match_last_words(segment, ayah, threshold)\`
- [ ] \`match_first_words(segment, ayah, threshold)\`

### \`core/overlap.py\` - Overlap detection
- [ ] \`detect_overlap(text1, text2)\` - Find overlapping words
- [ ] \`remove_overlap(text1, text2)\` - Merge without duplicates

### \`core/aligner.py\` - Main alignment algorithm
- [ ] \`Aligner\` class with configurable thresholds
- [ ] \`align(segments, ayahs)\` method
- [ ] Hook integration points

## Source Reference
Current logic is in:
- \`src/transcribe.py\` → \`normalize_arabic()\`
- \`src/align_segments.py\` → alignment algorithm

## Acceptance Criteria
- [ ] All functions are pure (no side effects)
- [ ] Each module has clear single responsibility
- [ ] No file I/O in core modules
- [ ] Ready for unit testing"

gh issue create --repo $REPO \
  --title "[1.4] Abstract transcription interface" \
  --label "phase-1-foundation" \
  --body "## Description
Create an abstract interface for audio transcription, allowing different ASR backends.

## Files to Create

### \`transcription/base.py\` - Abstract interface
\`\`\`python
from abc import ABC, abstractmethod
from munajjam.models import Segment

class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> list[Segment]:
        pass
    
    @abstractmethod
    async def transcribe_async(self, audio_path: str) -> list[Segment]:
        pass
\`\`\`

### \`transcription/whisper.py\` - Tarteel Whisper implementation
- [ ] Load Tarteel model (\`tarteel-ai/whisper-base-ar-quran\`)
- [ ] Implement \`transcribe()\` method
- [ ] Handle silence detection
- [ ] Skip Isti'aza and Basmala segments
- [ ] Return list of \`Segment\` models

### \`transcription/silence.py\` - Silence detection utilities
- [ ] \`detect_silence(audio, threshold, min_length)\`
- [ ] \`detect_speech_chunks(audio)\`

## Acceptance Criteria
- [ ] Can swap transcription backends without changing other code
- [ ] Whisper implementation matches current \`transcribe.py\` behavior
- [ ] Proper error handling for missing files, invalid audio"

gh issue create --repo $REPO \
  --title "[1.5] Abstract storage interface" \
  --label "phase-1-foundation" \
  --body "## Description
Create an abstract storage interface for persisting alignment results.

## Files to Create

### \`storage/base.py\` - Abstract interface
\`\`\`python
from abc import ABC, abstractmethod
from uuid import UUID
from munajjam.models import Recitation

class BaseStorage(ABC):
    @abstractmethod
    def save_recitation(self, recitation: Recitation) -> None:
        pass
    
    @abstractmethod
    def get_recitation(self, id: UUID) -> Recitation | None:
        pass
    
    @abstractmethod
    def list_recitations(self, reciter: str = None) -> list[Recitation]:
        pass
    
    @abstractmethod
    def delete_recitation(self, id: UUID) -> bool:
        pass
\`\`\`

### \`storage/sqlite.py\` - SQLite implementation
- [ ] Implement all BaseStorage methods
- [ ] Create tables if not exist
- [ ] Match current \`save_to_db.py\` schema

### \`storage/memory.py\` - In-memory storage (for testing)
- [ ] Simple dict-based storage
- [ ] Useful for unit tests

### \`storage/json.py\` - JSON file storage (optional)
- [ ] Save recitations as JSON files
- [ ] Good for debugging/inspection

## Acceptance Criteria
- [ ] Can swap storage backends via configuration
- [ ] SQLite implementation matches current behavior
- [ ] Memory storage works for testing"

gh issue create --repo $REPO \
  --title "[1.6] Configuration management with Pydantic Settings" \
  --label "phase-1-foundation,good first issue" \
  --body "## Description
Implement centralized configuration using Pydantic Settings with environment variable support.

## File: \`munajjam/config.py\`

\`\`\`python
from pydantic_settings import BaseSettings

class MunajjamSettings(BaseSettings):
    # Model settings
    model_id: str = \"tarteel-ai/whisper-base-ar-quran\"
    device: str = \"auto\"  # auto, cpu, cuda, mps
    
    # Audio processing
    silence_threshold_db: int = -30
    min_silence_ms: int = 300
    sample_rate: int = 16000
    
    # Alignment
    similarity_threshold: float = 0.6
    first_words_threshold: float = 0.6
    last_words_threshold: float = 0.6
    n_check_words: int = 3
    
    # Storage
    database_url: str = \"sqlite:///munajjam.db\"
    
    class Config:
        env_prefix = \"MUNAJJAM_\"
        env_file = \".env\"
\`\`\`

## Tasks
- [ ] Create \`MunajjamSettings\` class
- [ ] Add \`get_settings()\` singleton function
- [ ] Add \`configure(**kwargs)\` function for programmatic config
- [ ] Document all settings in docstrings

## Usage Examples
\`\`\`python
# Via environment variables
# export MUNAJJAM_DEVICE=cuda
# export MUNAJJAM_SIMILARITY_THRESHOLD=0.7

# Via code
from munajjam.config import configure
configure(device=\"cuda\", similarity_threshold=0.7)
\`\`\`

## Acceptance Criteria
- [ ] Settings load from environment variables
- [ ] Settings can be overridden programmatically
- [ ] Default values match current behavior"

gh issue create --repo $REPO \
  --title "[1.7] Custom exceptions" \
  --label "phase-1-foundation,good first issue" \
  --body "## Description
Create a hierarchy of custom exceptions for better error handling.

## File: \`munajjam/exceptions.py\`

\`\`\`python
class MunajjamError(Exception):
    \"\"\"Base exception for all Munajjam errors\"\"\"
    pass

class ConfigurationError(MunajjamError):
    \"\"\"Invalid configuration\"\"\"
    pass

class TranscriptionError(MunajjamError):
    \"\"\"Audio transcription failed\"\"\"
    pass

class AlignmentError(MunajjamError):
    \"\"\"Alignment process failed\"\"\"
    pass

class StorageError(MunajjamError):
    \"\"\"Storage operation failed\"\"\"
    pass

class ValidationError(MunajjamError):
    \"\"\"Data validation failed\"\"\"
    pass

class AudioError(MunajjamError):
    \"\"\"Audio file issues\"\"\"
    pass

class QuranDataError(MunajjamError):
    \"\"\"Quran data issues (invalid surah, missing ayah)\"\"\"
    pass
\`\`\`

## Tasks
- [ ] Create exception hierarchy
- [ ] Add docstrings with examples
- [ ] Export from \`munajjam/__init__.py\`

## Acceptance Criteria
- [ ] Users can catch \`MunajjamError\` for all library errors
- [ ] Specific exceptions available for fine-grained handling
- [ ] All exceptions have helpful error messages"

gh issue create --repo $REPO \
  --title "[1.8] Hooks system" \
  --label "phase-1-foundation" \
  --body "## Description
Implement a hooks/callback system for observing library events (progress, quality metrics, errors).

## Files to Create

### \`hooks/base.py\` - Base hooks class
\`\`\`python
class MunajjamHooks:
    # Transcription hooks
    def on_transcription_start(self, audio_path: str, surah_id: int) -> None: pass
    def on_segment_transcribed(self, segment: Segment) -> None: pass
    def on_transcription_complete(self, segments: list[Segment]) -> None: pass
    
    # Alignment hooks
    def on_alignment_start(self, surah_id: int, total_ayahs: int) -> None: pass
    def on_ayah_aligned(self, result: AlignmentResult) -> None: pass
    def on_overlap_detected(self, ayah: Ayah, overlap_text: str) -> None: pass
    def on_alignment_complete(self, results: list[AlignmentResult]) -> None: pass
    
    # Quality hooks
    def on_quality_report(self, report: QualityReport) -> None: pass
    
    # Progress hooks
    def on_progress(self, current: int, total: int, stage: str) -> None: pass
    
    # Error hooks
    def on_error(self, error: Exception, context: dict = None) -> None: pass
    def on_warning(self, message: str, context: dict = None) -> None: pass
    def on_retry(self, attempt: int, max_attempts: int, reason: str) -> None: pass
    
    # Lifecycle hooks
    def on_sync_start(self, audio_path: str, surah_id: int, reciter: str) -> None: pass
    def on_sync_complete(self, recitation: Recitation) -> None: pass
\`\`\`

### \`models/quality.py\` - QualityReport model
\`\`\`python
class QualityReport(BaseModel):
    recitation_id: UUID
    surah_id: int
    total_ayahs: int
    aligned_ayahs: int
    avg_similarity_score: float
    min_similarity_score: float
    low_confidence_count: int
    total_segments: int
    segments_merged: int
    overlaps_detected: int
    transcription_time_seconds: float
    alignment_time_seconds: float
    total_time_seconds: float
    failed_ayahs: list[int]
    warnings: list[str]
\`\`\`

## Usage Example
\`\`\`python
class MyHooks(MunajjamHooks):
    def on_ayah_aligned(self, result):
        print(f\"✅ Ayah {result.ayah.ayah_number}\")
    
    def on_quality_report(self, report):
        print(f\"Score: {report.avg_similarity_score:.2f}\")

m = Munajjam(hooks=MyHooks())
\`\`\`

## Acceptance Criteria
- [ ] All hooks have no-op default implementations
- [ ] Hooks are called at appropriate points in pipeline
- [ ] QualityReport captures all relevant metrics
- [ ] Multiple hooks can be composed"

echo ""
echo "✅ Phase 1 issues created!"
echo ""

# ============================================================
# Summary
# ============================================================
echo "🎉 All issues created successfully!"
echo ""
echo "Summary:"
echo "  - Phase 1 (Foundation): 8 issues"
echo ""
echo "View issues at: https://github.com/$REPO/issues"
