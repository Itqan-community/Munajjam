"""
Integration tests for Munajjam library.
These tests use real data and can be slower.
"""

import pytest
from munajjam.data import load_surah_ayahs, get_ayah_count
from munajjam.core import Aligner
from munajjam.models import Segment, SegmentType


@pytest.mark.integration
@pytest.mark.slow
class TestRealDataAlignment:
    """Integration tests with real Quran data."""

    @pytest.mark.parametrize(
        "surah_id,expected_count",
        [
            (1, 7),
            (114, 6),
        ],
    )
    def test_load_real_surah(self, surah_id, expected_count):
        """Test loading real surah ayahs."""
        ayahs = load_surah_ayahs(surah_id)

        assert len(ayahs) == expected_count
        assert ayahs[0].surah_id == surah_id
        assert ayahs[0].ayah_number == 1

    @pytest.mark.parametrize("surah_id", [1, 2, 114])
    def test_ayah_count_matches_loaded(self, surah_id):
        """Test that get_ayah_count matches loaded ayahs."""
        count = get_ayah_count(surah_id)
        ayahs = load_surah_ayahs(surah_id)

        assert count == len(ayahs)

    def test_alignment_with_real_ayahs(self):
        """Test alignment with real ayah data."""
        ayahs = load_surah_ayahs(1)

        segments = [
            Segment(
                id=i,
                surah_id=1,
                start=i * 5.0,
                end=(i + 1) * 5.0,
                text=ayah.text[:30],
                type=SegmentType.AYAH,
            )
            for i, ayah in enumerate(ayahs[:3])
        ]

        aligner = Aligner(audio_path="test.wav", strategy="hybrid", energy_snap=False)
        results = aligner.align(segments, ayahs[:3])

        assert len(results) > 0
        assert all(0 <= r.similarity_score <= 1.0 for r in results)

    @pytest.mark.parametrize("strategy", ["greedy", "dp", "hybrid"])
    def test_strategies_with_real_data(self, strategy):
        """Test all strategies produce results with real data."""
        ayahs = load_surah_ayahs(114)

        segments = [
            Segment(
                id=i,
                surah_id=114,
                start=i * 5.0,
                end=(i + 1) * 5.0,
                text=ayah.text[:20],
                type=SegmentType.AYAH,
            )
            for i, ayah in enumerate(ayahs)
        ]

        aligner = Aligner(audio_path="test.wav", strategy=strategy, energy_snap=False)
        results = aligner.align(segments, ayahs)

        assert len(results) > 0
