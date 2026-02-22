"""
Shared fixtures and test configuration for Munajjam tests.
"""

import pytest
from munajjam.models import Segment, SegmentType, Ayah
from munajjam.data import load_surah_ayahs


@pytest.fixture
def sample_segments():
    """Sample transcribed segments for testing."""
    return [
        Segment(
            id=0,
            surah_id=1,
            start=0.0,
            end=4.5,
            text="أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ",
            type=SegmentType.ISTIADHA,
        ),
        Segment(
            id=1,
            surah_id=1,
            start=5.0,
            end=8.5,
            text="بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
            type=SegmentType.BASMALA,
        ),
        Segment(
            id=2,
            surah_id=1,
            start=9.0,
            end=13.5,
            text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
            type=SegmentType.AYAH,
        ),
        Segment(
            id=3,
            surah_id=1,
            start=14.0,
            end=18.0,
            text="الرَّحْمَٰنِ الرَّحِيمِ",
            type=SegmentType.AYAH,
        ),
    ]


@pytest.fixture
def sample_ayahs():
    """Sample ayah objects for testing (Surah Al-Fatiha first 4 ayahs)."""
    return [
        Ayah(id=1, surah_id=1, ayah_number=1, text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"),
        Ayah(id=2, surah_id=1, ayah_number=2, text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"),
        Ayah(id=3, surah_id=1, ayah_number=3, text="الرَّحْمَٰنِ الرَّحِيمِ"),
        Ayah(id=4, surah_id=1, ayah_number=4, text="مَالِكِ يَوْمِ الدِّينِ"),
    ]


@pytest.fixture
def sample_silences():
    """Sample silence periods in milliseconds."""
    return [(4500, 5000), (8500, 9000), (13000, 14000)]


@pytest.fixture
def real_surah_1_ayahs():
    """Load real ayahs for Surah Al-Fatiha from CSV."""
    try:
        return load_surah_ayahs(1)
    except Exception:
        pytest.skip("Could not load real ayah data")


@pytest.fixture
def short_arabic_text_pairs():
    """Various Arabic text pairs for similarity testing."""
    return [
        ("بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم", 1.0),
        ("بسم الله", "بسم الله الرحمن الرحيم", 0.5),
        ("الحمد لله رب العالمين", "الحمد لله رب العلمين", 0.98),
    ]


@pytest.fixture
def normalization_test_cases():
    """Test cases for Arabic normalization."""
    return [
        ("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", "بسم الله الرحمن الرحيم"),
        ("أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ", "اعوذ بالله من الشيطان الرجيم"),
        ("ٱلۡحَمۡدُ لِلَّهِ رَبِّ ٱلۡعَٰلَمِينَ", "الحمد لله رب العالمين"),
    ]
