"""
Quran reference data loader.

Provides functions to load and access canonical Quran text from the
bundled CSV file.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional

from munajjam.exceptions import QuranDataError
from munajjam.models import Ayah, Surah
from munajjam.models.surah import SURAH_NAMES, SURAH_AYAH_COUNTS


def _get_data_path() -> Path:
    """Get path to the bundled data directory."""
    return Path(__file__).parent


def _get_quran_csv_path() -> Path:
    """Get path to the Quran ayahs CSV file."""
    # First try bundled data
    bundled = _get_data_path() / "quran_ayat.csv"
    if bundled.exists():
        return bundled

    # Fall back to original data location (for development)
    project_data = Path("data") / "Quran Ayas List.csv"
    if project_data.exists():
        return project_data

    raise QuranDataError(
        "Quran ayahs CSV not found. "
        "Expected at: munajjam/data/quran_ayat.csv or data/Quran Ayas List.csv"
    )


@lru_cache(maxsize=1)
def load_ayahs() -> list[Ayah]:
    """
    Load all ayahs from the CSV file.

    Returns:
        List of all Ayah objects (6236 ayahs)

    Raises:
        QuranDataError: If CSV file cannot be loaded
    """
    try:
        csv_path = _get_quran_csv_path()
        ayahs = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ayah = Ayah(
                    id=int(row["id"]),
                    surah_id=int(row["sura_id"]),
                    ayah_number=int(row["index"]),
                    text=row["text"],
                )
                ayahs.append(ayah)

        return ayahs

    except FileNotFoundError:
        raise QuranDataError("Quran ayahs CSV file not found")
    except Exception as e:
        raise QuranDataError(f"Failed to load Quran ayahs: {e}")


def load_surah_ayahs(surah_id: int) -> list[Ayah]:
    """
    Load ayahs for a specific surah.

    Args:
        surah_id: Surah number (1-114)

    Returns:
        List of Ayah objects for the surah
    """
    if surah_id < 1 or surah_id > 114:
        raise ValueError(f"Invalid surah_id: {surah_id}. Must be 1-114.")

    all_ayahs = load_ayahs()
    return [a for a in all_ayahs if a.surah_id == surah_id]


def get_ayah(surah_id: int, ayah_number: int) -> Optional[Ayah]:
    """
    Get a specific ayah by surah and ayah number.

    Args:
        surah_id: Surah number (1-114)
        ayah_number: Ayah number within the surah

    Returns:
        Ayah if found, None otherwise
    """
    ayahs = load_surah_ayahs(surah_id)

    for ayah in ayahs:
        if ayah.ayah_number == ayah_number:
            return ayah

    return None


def get_ayah_count(surah_id: int) -> int:
    """
    Get the total number of ayahs in a surah.

    Args:
        surah_id: Surah number (1-114)

    Returns:
        Number of ayahs in the surah
    """
    if surah_id < 1 or surah_id > 114:
        raise ValueError(f"Invalid surah_id: {surah_id}. Must be 1-114.")

    return SURAH_AYAH_COUNTS[surah_id]


def get_all_surahs() -> list[Surah]:
    """
    Get metadata for all 114 surahs.

    Returns:
        List of Surah objects with metadata
    """
    return [Surah.from_id(i) for i in range(1, 115)]


def get_surah(surah_id: int) -> Surah:
    """
    Get metadata for a specific surah.

    Args:
        surah_id: Surah number (1-114)

    Returns:
        Surah object with metadata
    """
    return Surah.from_id(surah_id)


def get_surah_name(surah_id: int) -> str:
    """
    Get the Arabic name of a surah.

    Args:
        surah_id: Surah number (1-114)

    Returns:
        Arabic name of the surah
    """
    if surah_id < 1 or surah_id > 114:
        raise ValueError(f"Invalid surah_id: {surah_id}. Must be 1-114.")

    return SURAH_NAMES[surah_id]


# Convenience function for quick access
def ayahs_for_surah(surah_id: int | str) -> list[Ayah]:
    """
    Load ayahs for a surah (accepts int or zero-padded string).

    Args:
        surah_id: Surah number as int (1-114) or string ("001", "114")

    Returns:
        List of Ayah objects
    """
    if isinstance(surah_id, str):
        surah_id = int(surah_id)

    return load_surah_ayahs(surah_id)

