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

from munajjam._version import __version__
from munajjam.config import MunajjamSettings, configure, get_settings
from munajjam.exceptions import (
    AlignmentError,
    AudioFileError,
    ConfigurationError,
    ModelNotLoadedError,
    MunajjamError,
    QuranDataError,
    TranscriptionError,
)
from munajjam.formatters import (
    AlignmentMetadata,
    AlignmentOutput,
    FormattedAyahResult,
    format_alignment_results,
)
from munajjam.models import (
    AlignmentResult,
    Ayah,
    Segment,
    SegmentType,
    Surah,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "Ayah",
    "Segment",
    "SegmentType",
    "Surah",
    "AlignmentResult",
    # Config
    "MunajjamSettings",
    "get_settings",
    "configure",
    # Formatters
    "AlignmentOutput",
    "FormattedAyahResult",
    "AlignmentMetadata",
    "format_alignment_results",
    # Exceptions
    "MunajjamError",
    "TranscriptionError",
    "AlignmentError",
    "ConfigurationError",
    "AudioFileError",
    "ModelNotLoadedError",
    "QuranDataError",
]
