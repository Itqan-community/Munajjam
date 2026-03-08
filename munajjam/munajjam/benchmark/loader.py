"""
Ground-truth file loader for the benchmark harness.
"""

import json
from pathlib import Path

from munajjam.benchmark.models import GroundTruthSurah


def load_ground_truth(path: Path | str) -> GroundTruthSurah:
    """Load and validate a single ground-truth JSON file.

    Args:
        path: Path to the ground-truth JSON file.

    Returns:
        Validated GroundTruthSurah instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
        json.JSONDecodeError: If the file is not valid JSON.
        pydantic.ValidationError: If the data fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return GroundTruthSurah.model_validate(data)


def list_ground_truth_files(ground_truth_dir: Path) -> list[Path]:
    """Return all .json files in the ground-truth directory, sorted.

    Args:
        ground_truth_dir: Directory containing ground-truth JSON files.

    Returns:
        Sorted list of Path objects for each JSON file.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not ground_truth_dir.is_dir():
        msg = f"Ground truth directory not found: {ground_truth_dir}"
        raise FileNotFoundError(msg)
    return sorted(ground_truth_dir.glob("*.json"))
