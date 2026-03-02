"""
Unit tests for data loading functions.
"""

import pytest
from munajjam.data import load_surah_ayahs, get_surah_name, get_ayah_count


class TestLoadSurahAyahs:
    """Test loading of surah ayahs from CSV."""

    @pytest.mark.parametrize(
        "surah_id,expected_count",
        [
            (1, 7),  # Al-Fatiha
            (2, 286),  # Al-Baqarah
            (114, 6),  # An-Nas
        ],
    )
    def test_load_surah(self, surah_id, expected_count):
        """Test loading surah ayahs."""
        ayahs = load_surah_ayahs(surah_id)

        assert len(ayahs) == expected_count
        assert ayahs[0].surah_id == surah_id
        assert ayahs[0].ayah_number == 1
        assert ayahs[-1].ayah_number == expected_count

    def test_invalid_surah_id(self):
        """Test loading with invalid surah ID."""
        with pytest.raises(Exception):
            load_surah_ayahs(999)

    def test_ayahs_have_valid_data(self, real_surah_1_ayahs):
        """Test loaded ayahs have valid data."""
        ayah_ids = []
        for i, ayah in enumerate(real_surah_1_ayahs):
            assert len(ayah.text) > 0
            assert ayah.ayah_number == i + 1
            ayah_ids.append(ayah.id)

        assert len(ayah_ids) == len(set(ayah_ids))  # IDs are unique


class TestGetSurahName:
    """Test getting surah names."""

    @pytest.mark.parametrize(
        "surah_id,possible_names",
        [
            (1, ["الفاتحة", "Fatiha", "Fātihah"]),
            (2, ["البقرة", "Baqarah"]),
            (114, ["الناس", "Nas"]),
        ],
    )
    def test_get_surah_name(self, surah_id, possible_names):
        """Test getting surah names."""
        name = get_surah_name(surah_id)
        assert any(n in name for n in possible_names)

    def test_invalid_surah_id(self):
        """Test getting name of invalid surah."""
        with pytest.raises(Exception):
            get_surah_name(999)


class TestGetAyahCount:
    """Test getting ayah counts for surahs."""

    @pytest.mark.parametrize(
        "surah_id,expected_count",
        [
            (1, 7),
            (2, 286),
            (114, 6),
        ],
    )
    def test_get_ayah_count(self, surah_id, expected_count):
        """Test ayah count for various surahs."""
        assert get_ayah_count(surah_id) == expected_count

    def test_invalid_surah_id(self):
        """Test getting ayah count for invalid surah."""
        with pytest.raises(Exception):
            get_ayah_count(999)

    def test_matches_loaded_ayahs(self, real_surah_1_ayahs):
        """Test that get_ayah_count matches loaded ayahs."""
        assert get_ayah_count(1) == len(real_surah_1_ayahs)
