"""
Unit tests for silence detection.
"""

from unittest.mock import patch
from munajjam.models import Segment
from munajjam.transcription.silence import (
    BreathBoundary,
    annotate_segments_with_breaths,
    detect_reciter_breaths,
)


class TestBreathDetection:
    """Test reciter breath boundary detection and annotation."""

    @patch("munajjam.transcription.silence.detect_silences")
    def test_detect_reciter_breaths(self, mock_detect_silences):
        mock_detect_silences.return_value = [(1000, 1600), (4000, 4800)]
        breaths = detect_reciter_breaths("dummy.wav", min_pause_duration_ms=300)
        assert len(breaths) == 2
        assert breaths[0].start_sec == 1.0
        assert breaths[0].end_sec == 1.6
        assert breaths[0].duration_sec == 0.6
        assert breaths[0].is_breath_boundary is True

    @patch("munajjam.transcription.silence.detect_reciter_breaths")
    def test_annotate_segments_with_breaths(self, mock_detect_breaths):
        mock_detect_breaths.return_value = [
            BreathBoundary(
                start_sec=1.5, end_sec=2.1, duration_sec=0.6, is_breath_boundary=True
            )
        ]
        seg1 = Segment(id=1, surah_id=1, start=0.0, end=1.5, text="بِسْمِ ٱللَّهِ")
        seg2 = Segment(id=2, surah_id=1, start=2.1, end=5.0, text="ٱلْحَمْدُ لِلَّهِ")

        annotated = annotate_segments_with_breaths([seg1, seg2], "dummy.wav")
        assert annotated[0].is_breath_boundary is True
        assert annotated[0].pause_duration == 0.6
        assert annotated[1].is_breath_boundary is False
        assert annotated[1].pause_duration == 0.0

    def test_annotate_segments_with_breaths_single_assignment_per_boundary(self):
        """Verify that a breath boundary is assigned to only ONE segment, not reused by consecutive segments."""
        breath = BreathBoundary(
            start_sec=1.5, end_sec=2.1, duration_sec=0.6, is_breath_boundary=True
        )
        seg1 = Segment(id=1, surah_id=1, start=0.0, end=1.4, text="seg1")
        seg2 = Segment(id=2, surah_id=1, start=1.4, end=1.5, text="seg2")

        annotated = annotate_segments_with_breaths([seg1, seg2], breaths=[breath])
        assigned_count = sum(1 for s in annotated if s.is_breath_boundary)
        assert assigned_count == 1
        assert annotated[0].is_breath_boundary is True
        assert annotated[1].is_breath_boundary is False
