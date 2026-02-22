"""
Smoke tests for example scripts.


These tests exercise the core logic of each example with a mocked transcriber,
ensuring examples stay in sync with the library's public API.
No real model downloads or audio files are needed.


Closes #48
"""

from __future__ import annotations


from munajjam.models import Segment, Ayah, AlignmentResult
from munajjam.transcription.base import BaseTranscriber
from munajjam.core import Aligner, normalize_arabic, similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FAKE_SEGMENTS = [
    Segment(
        id=1,
        surah_id=1,
        text="بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
        start=0.0,
        end=3.5,
    ),
    Segment(
        id=2,
        surah_id=1,
        text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        start=3.5,
        end=7.0,
    ),
    Segment(
        id=3,
        surah_id=1,
        text="الرَّحْمَنِ الرَّحِيمِ",
        start=7.0,
        end=9.5,
    ),
    Segment(
        id=4,
        surah_id=1,
        text="مَالِكِ يَوْمِ الدِّينِ",
        start=9.5,
        end=12.0,
    ),
    Segment(
        id=5,
        surah_id=1,
        text="إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
        start=12.0,
        end=15.5,
    ),
    Segment(
        id=6,
        surah_id=1,
        text="اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
        start=15.5,
        end=19.0,
    ),
]


# ---------------------------------------------------------------------------
# Smoke test: advanced_alignment example pattern (02)
# ---------------------------------------------------------------------------


class TestAdvancedAlignmentSmoke:
    """Smoke tests mirroring examples/02_advanced_alignment.py logic."""

    def test_aligner_class_usage(self, surah_1_ayahs: list[Ayah]) -> None:
        """Verify Aligner class and its methods used in example 02."""
        aligner = Aligner()
        assert hasattr(aligner, "align")

        # Mock segments for a subset of ayahs
        segments = [FAKE_SEGMENTS[0], FAKE_SEGMENTS[1]]
        results = aligner.align(segments, surah_1_ayahs[:2])

        assert len(results) == 2
        assert all(isinstance(r, AlignmentResult) for r in results)


# ---------------------------------------------------------------------------
# Smoke test: custom_transcriber example pattern (03)
# ---------------------------------------------------------------------------


class TestCustomTranscriberSmoke:
    """Smoke tests mirroring examples/03_custom_transcriber.py logic."""

    def test_custom_transcriber_inheritance(self) -> None:
        """Verify that a custom class can inherit from BaseTranscriber."""

        class MyTranscriber(BaseTranscriber):
            def transcribe(self, audio_path):
                return []

            async def transcribe_async(self, audio_path):
                return []

            def load(self):
                pass

            def unload(self):
                pass

            @property
            def is_loaded(self):
                return True

        transcriber = MyTranscriber()
        assert isinstance(transcriber, BaseTranscriber)


# ---------------------------------------------------------------------------
# Smoke test: text_processing example pattern (04)
# ---------------------------------------------------------------------------


class TestTextProcessingSmoke:
    """Smoke tests mirroring examples/04_text_processing.py logic."""

    def test_normalization_and_similarity(self) -> None:
        """Verify text processing utilities used in example 04."""
        t1 = "بِسْمِ اللَّهِ"
        t2 = "بسم الله"

        n1 = normalize_arabic(t1)
        n2 = normalize_arabic(t2)

        assert n1 == n2
        assert similarity(t1, t2) > 0.9
