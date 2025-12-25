"""
مُنَجِّم (Munajjam) — A Python library to synchronize Quran Ayat with audio recitations.

Usage:
    from munajjam.transcription import WhisperTranscriber
    from munajjam.core import align_segments
    from munajjam.data import load_surah_ayahs

    # Transcribe
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe("surah_1.wav")

    # Align
    ayahs = load_surah_ayahs(1)
    results = align_segments(segments, ayahs)

    # Results contain timing information for each ayah
    for result in results:
        print(f"Ayah {result.ayah.ayah_number}: {result.start_time:.2f}s - {result.end_time:.2f}s")
"""

from munajjam.models import (
    AlignmentResult,
    Ayah,
    Recitation,
    RecitationStatus,
    Segment,
    SegmentType,
    Surah,
)
from munajjam.config import MunajjamSettings, get_settings, configure
from munajjam.exceptions import (
    MunajjamError,
    TranscriptionError,
    AlignmentError,
    ConfigurationError,
    AudioFileError,
    ModelNotLoadedError,
    QuranDataError,
)

__version__ = "2.0.0a1"
__all__ = [
    # Version
    "__version__",
    # Models
    "Ayah",
    "Segment",
    "SegmentType",
    "Surah",
    "AlignmentResult",
    "Recitation",
    "RecitationStatus",
    # Config
    "MunajjamSettings",
    "get_settings",
    "configure",
    # Exceptions
    "MunajjamError",
    "TranscriptionError",
    "AlignmentError",
    "ConfigurationError",
    "AudioFileError",
    "ModelNotLoadedError",
    "QuranDataError",
]
