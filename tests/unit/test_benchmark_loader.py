"""
Unit tests for benchmark ground-truth loader (Issue #45).

Tests cover:
- load_ground_truth with valid JSON file
- load_ground_truth with nonexistent path
- load_ground_truth with malformed JSON
- list_ground_truth_files with multiple files
"""

import json

import pytest

from munajjam.benchmark.loader import list_ground_truth_files, load_ground_truth
from munajjam.benchmark.models import GroundTruthSurah


class TestLoadGroundTruth:
    """Test load_ground_truth function."""

    def test_loads_valid_json(self, tmp_path) -> None:
        data = {
            "surah_id": 1,
            "reciter": "sample",
            "audio_filename": "001.wav",
            "ayahs": [
                {
                    "ayah_number": 1,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "بسم الله",
                },
            ],
        }
        gt_file = tmp_path / "surah_001.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")

        result = load_ground_truth(gt_file)
        assert isinstance(result, GroundTruthSurah)
        assert result.surah_id == 1
        assert len(result.ayahs) == 1

    def test_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_ground_truth(tmp_path / "nonexistent.json")

    def test_raises_on_malformed_json(self, tmp_path) -> None:
        gt_file = tmp_path / "bad.json"
        gt_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(Exception):  # json.JSONDecodeError
            load_ground_truth(gt_file)

    def test_raises_on_invalid_data(self, tmp_path) -> None:
        """Valid JSON but fails Pydantic validation."""
        data = {"surah_id": 999, "reciter": "x", "audio_filename": "x.wav", "ayahs": []}
        gt_file = tmp_path / "invalid.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(Exception):  # ValidationError
            load_ground_truth(gt_file)

    def test_accepts_string_path(self, tmp_path) -> None:
        data = {
            "surah_id": 112,
            "reciter": "test",
            "audio_filename": "112.wav",
            "ayahs": [
                {
                    "ayah_number": 1,
                    "start_time": 0.0,
                    "end_time": 4.0,
                    "text": "قل هو الله احد",
                },
            ],
        }
        gt_file = tmp_path / "surah_112.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")

        result = load_ground_truth(str(gt_file))
        assert result.surah_id == 112


class TestListGroundTruthFiles:
    """Test list_ground_truth_files function."""

    def test_returns_sorted_json_files(self, tmp_path) -> None:
        (tmp_path / "surah_112.json").write_text("{}", encoding="utf-8")
        (tmp_path / "surah_001.json").write_text("{}", encoding="utf-8")
        (tmp_path / "readme.txt").write_text("ignore", encoding="utf-8")

        files = list_ground_truth_files(tmp_path)
        assert len(files) == 2
        assert files[0].name == "surah_001.json"
        assert files[1].name == "surah_112.json"

    def test_empty_directory(self, tmp_path) -> None:
        files = list_ground_truth_files(tmp_path)
        assert files == []
