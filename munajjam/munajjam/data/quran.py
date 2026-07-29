"""
Quran reference data loader.

Provides functions to load and access canonical Quran text from the
bundled JSON file.
"""

import json
from functools import lru_cache
from pathlib import Path

from munajjam.config import get_settings
from munajjam.exceptions import QuranDataError
from munajjam.models import Ayah, Surah
from munajjam.models.surah import SURAH_AYAH_COUNTS, SURAH_NAMES


def _get_data_path() -> Path:
    """Get path to the bundled data directory."""
    return Path(__file__).parent


def _get_quran_json_path(riwaya: str) -> Path:
    """Get path to the Quran ayahs JSON file for the given riwaya."""
    bundled = _get_data_path() / f"quran_{riwaya}.json"
    if bundled.exists():
        return bundled

    raise QuranDataError(
        f"Quran ayahs JSON for riwaya '{riwaya}' not found. "
        f"Expected at: {bundled}"
    )


@lru_cache(maxsize=2)
def load_ayahs(riwaya: str | None = None) -> list[Ayah]:
    """
    Load all ayahs from the JSON file based on riwaya.

    Args:
        riwaya: The riwaya to load (e.g., 'hafs', 'warsh'). If None, uses config.

    Returns:
        List of all Ayah objects

    Raises:
        QuranDataError: If JSON file cannot be loaded
    """
    if riwaya is None:
        riwaya = get_settings().riwaya

    try:
        json_path = _get_quran_json_path(riwaya)
        ayahs = []

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        global_id = 1
        for surah_idx in range(1, 115):
            surah_str = str(surah_idx)
            if surah_str in data:
                for idx, text in enumerate(data[surah_str]):
                    ayah = Ayah(
                        id=global_id,
                        surah_id=surah_idx,
                        ayah_number=idx + 1,
                        text=text,
                    )
                    ayahs.append(ayah)
                    global_id += 1

        return ayahs

    except FileNotFoundError as e:
        raise QuranDataError(f"Quran ayahs JSON file not found for riwaya {riwaya}") from e
    except Exception as e:
        raise QuranDataError(f"Failed to load Quran ayahs: {e}") from e


def load_surah_ayahs(surah_id: int, riwaya: str | None = None) -> list[Ayah]:
    """
    Load ayahs for a specific surah.

    Args:
        surah_id: Surah number (1-114)
        riwaya: The riwaya to load.

    Returns:
        List of Ayah objects for the surah
    """
    if surah_id < 1 or surah_id > 114:
        raise ValueError(f"Invalid surah_id: {surah_id}. Must be 1-114.")

    all_ayahs = load_ayahs(riwaya)
    return [a for a in all_ayahs if a.surah_id == surah_id]


def get_ayah(surah_id: int, ayah_number: int, riwaya: str | None = None) -> Ayah | None:
    """
    Get a specific ayah by surah and ayah number.

    Args:
        surah_id: Surah number (1-114)
        ayah_number: Ayah number within the surah
        riwaya: The riwaya to load.

    Returns:
        Ayah if found, None otherwise
    """
    ayahs = load_surah_ayahs(surah_id, riwaya)

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
def ayahs_for_surah(surah_id: int | str, riwaya: str | None = None) -> list[Ayah]:
    """
    Load ayahs for a surah (accepts int or zero-padded string).

    Args:
        surah_id: Surah number as int (1-114) or string ("001", "114")
        riwaya: The riwaya to load.

    Returns:
        List of Ayah objects
    """
    if isinstance(surah_id, str):
        surah_id = int(surah_id)

    return load_surah_ayahs(surah_id, riwaya)
