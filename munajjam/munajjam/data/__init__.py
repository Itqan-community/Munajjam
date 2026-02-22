"""
Quran data module for Munajjam library.

Provides access to bundled Quran reference data (ayahs, surah metadata).
"""

from munajjam.data.quran import (
    get_all_surahs,
    get_ayah,
    get_ayah_count,
    get_surah,
    get_surah_name,
    load_ayahs,
    load_surah_ayahs,
)

__all__ = [
    "load_ayahs",
    "load_surah_ayahs",
    "get_ayah",
    "get_ayah_count",
    "get_all_surahs",
    "get_surah",
    "get_surah_name",
]

