"""
Unit tests for Arabic text normalization.
"""

import pytest
from munajjam.core.arabic import normalize_arabic


class TestArabicNormalization:
    """Test Arabic text normalization functions."""

    @pytest.mark.parametrize(
        "input_text,should_contain,should_not_contain",
        [
            ("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", ["بسم", "الله", "الرحمن"], ["بِسْمِ", "اللَّهِ"]),
            ("ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَالَمِينَ", ["الحمد", "العالمين"], ["ٱلْحَمْدُ", "ٱلْعَالَمِينَ"]),
        ],
    )
    def test_normalize_diacritics_and_alif(
        self, input_text, should_contain, should_not_contain
    ):
        """Test removal of diacritics and alif normalization."""
        normalized = normalize_arabic(input_text)

        for text in should_contain:
            assert text in normalized
        for text in should_not_contain:
            assert text not in normalized

    @pytest.mark.parametrize("variant", ["أ", "إ", "آ"])
    def test_normalize_hamza_to_alif(self, variant):
        """Test normalization of hamza variants to alif."""
        assert normalize_arabic(variant) == "ا"

    def test_normalize_ta_marbuta(self):
        """Test normalization of ta marbuta to ha."""
        normalized = normalize_arabic("رَحِيمَة")
        assert "ه" in normalized
        assert "ة" not in normalized

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        assert normalize_arabic("") == ""

    def test_normalize_preserves_word_count(self, normalization_test_cases):
        """Test that normalization preserves word boundaries."""
        for original, _ in normalization_test_cases:
            normalized = normalize_arabic(original)
            assert len(original.split()) == len(normalized.split())
