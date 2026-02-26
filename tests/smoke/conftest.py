"""
Pytest fixtures for smoke tests.

Provides mocked transcriber and other utilities for testing example scripts
without requiring actual audio files or ML models.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from munajjam.models import Segment, SegmentType


@pytest.fixture
def mock_transcriber():
    """
    Mock transcriber that returns predetermined segments.
    
    This fixture patches WhisperTranscriber to return mock segments
    without loading any ML models or processing actual audio.
    """
    mock_segments = [
        Segment(
            id=0, surah_id=114, start=0.0, end=4.5,
            text="أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ",
            type=SegmentType.ISTIADHA
        ),
        Segment(
            id=1, surah_id=114, start=5.0, end=8.5,
            text="بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
            type=SegmentType.BASMALA
        ),
        Segment(
            id=2, surah_id=114, start=9.0, end=13.5,
            text="قُلْ أَعُوذُ بِرَبِّ النَّاسِ",
            type=SegmentType.AYAH
        ),
        Segment(
            id=3, surah_id=114, start=14.0, end=18.0,
            text="مَلِكِ النَّاسِ",
            type=SegmentType.AYAH
        ),
        Segment(
            id=4, surah_id=114, start=18.5, end=23.0,
            text="إِلَٰهِ النَّاسِ",
            type=SegmentType.AYAH
        ),
        Segment(
            id=5, surah_id=114, start=23.5, end=28.0,
            text="مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ",
            type=SegmentType.AYAH
        ),
    ]
    
    with patch('munajjam.transcription.WhisperTranscriber') as MockTranscriber:
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = mock_segments
        mock_instance.is_loaded = True
        MockTranscriber.return_value = mock_instance
        MockTranscriber.__enter__ = MagicMock(return_value=mock_instance)
        MockTranscriber.__exit__ = MagicMock(return_value=None)
        yield mock_instance, mock_segments


@pytest.fixture
def mock_transcriber_with_words():
    """
    Mock transcriber that returns segments with word-level timestamps.
    
    Useful for testing word-level alignment features.
    """
    from munajjam.models import WordTimestamp
    
    mock_segments = [
        Segment(
            id=0, surah_id=114, start=0.0, end=4.5,
            text="أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ",
            type=SegmentType.ISTIADHA,
            words=[
                WordTimestamp(word="أَعُوذُ", start=0.0, end=1.0, probability=0.95),
                WordTimestamp(word="بِاللَّهِ", start=1.1, end=2.0, probability=0.95),
                WordTimestamp(word="مِنَ", start=2.1, end=2.5, probability=0.95),
                WordTimestamp(word="الشَّيْطَانِ", start=2.6, end=3.5, probability=0.95),
                WordTimestamp(word="الرَّجِيمِ", start=3.6, end=4.5, probability=0.95),
            ]
        ),
        Segment(
            id=1, surah_id=114, start=5.0, end=8.5,
            text="بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
            type=SegmentType.BASMALA,
            words=[
                WordTimestamp(word="بِسْمِ", start=5.0, end=5.8, probability=0.95),
                WordTimestamp(word="اللَّهِ", start=5.9, end=6.5, probability=0.95),
                WordTimestamp(word="الرَّحْمَنِ", start=6.6, end=7.5, probability=0.95),
                WordTimestamp(word="الرَّحِيمِ", start=7.6, end=8.5, probability=0.95),
            ]
        ),
        Segment(
            id=2, surah_id=114, start=9.0, end=13.5,
            text="قُلْ أَعُوذُ بِرَبِّ النَّاسِ",
            type=SegmentType.AYAH,
            words=[
                WordTimestamp(word="قُلْ", start=9.0, end=9.8, probability=0.95),
                WordTimestamp(word="أَعُوذُ", start=9.9, end=10.8, probability=0.95),
                WordTimestamp(word="بِرَبِّ", start=10.9, end=11.8, probability=0.95),
                WordTimestamp(word="النَّاسِ", start=11.9, end=13.5, probability=0.95),
            ]
        ),
        Segment(
            id=3, surah_id=114, start=14.0, end=18.0,
            text="مَلِكِ النَّاسِ",
            type=SegmentType.AYAH,
            words=[
                WordTimestamp(word="مَلِكِ", start=14.0, end=15.5, probability=0.95),
                WordTimestamp(word="النَّاسِ", start=15.6, end=18.0, probability=0.95),
            ]
        ),
    ]
    
    with patch('munajjam.transcription.WhisperTranscriber') as MockTranscriber:
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = mock_segments
        mock_instance.is_loaded = True
        MockTranscriber.return_value = mock_instance
        MockTranscriber.__enter__ = MagicMock(return_value=mock_instance)
        MockTranscriber.__exit__ = MagicMock(return_value=None)
        yield mock_instance, mock_segments


@pytest.fixture
def mock_silence_detection():
    """
    Mock silence detection to return predefined silence periods.
    """
    mock_silences = [(4500, 5000), (8500, 9000), (13500, 14000)]
    
    with patch('munajjam.transcription.detect_silences') as mock_detect:
        mock_detect.return_value = mock_silences
        yield mock_silences


@pytest.fixture
def mock_audio_file(tmp_path):
    """
    Create a mock audio file path for testing.
    
    Returns a Path to a dummy file that won't actually be read
    since the transcriber is mocked.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_file = audio_dir / "114.wav"
    audio_file.write_text("dummy audio content")
    return audio_file


@pytest.fixture
def mock_audio_directory(tmp_path):
    """
    Create a mock audio directory with multiple files.
    
    Returns a Path to a directory containing dummy audio files.
    """
    audio_dir = tmp_path / "Quran" / "badr_alturki_audio"
    audio_dir.mkdir(parents=True)
    
    # Create dummy audio files
    for surah_num in [1, 2, 114]:
        audio_file = audio_dir / f"{surah_num:03d}.wav"
        audio_file.write_text("dummy audio content")
    
    return audio_dir


@pytest.fixture
def patched_examples_dir():
    """
    Patch sys.path to include the examples directory for importing.
    """
    import sys
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    
    # Store original sys.path
    original_path = sys.path.copy()
    
    # Add examples directory to path
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    
    yield examples_dir
    
    # Restore original path
    sys.path = original_path
