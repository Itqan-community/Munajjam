"""
Smoke tests for example scripts.

Exercises the core logic of each example with a mocked transcriber
so that no real model download or audio file is needed. If an example
references a removed API, the test will fail.
"""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from munajjam.core.aligner import Aligner, AlignmentStrategy
from munajjam.models import Segment, SegmentType
from munajjam.data import load_surah_ayahs

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _fake_segments(surah_id: int = 114):
    """Build fake transcription segments from real ayah text."""
    ayahs = load_surah_ayahs(surah_id)
    return [
        Segment(
            id=i,
            surah_id=surah_id,
            start=i * 5.0,
            end=(i + 1) * 5.0,
            text=ayah.text[:40],
            type=SegmentType.AYAH,
        )
        for i, ayah in enumerate(ayahs)
    ]


def _mock_transcriber(surah_id: int = 114):
    """Create a mock WhisperTranscriber that returns fake segments."""
    segments = _fake_segments(surah_id)
    mock = MagicMock()
    mock.transcribe.return_value = segments
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.load.return_value = None
    mock.unload.return_value = None
    return mock


def _load_example(name: str):
    """Dynamically import an example module by filename."""
    path = EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# AST-based validation: catches removed APIs without running the script
# ---------------------------------------------------------------------------

class TestExampleAPIValidity:
    """Validate that examples only use current public APIs."""

    def test_02_only_valid_strategies(self):
        """Verify 02_comparing_strategies only uses strategies in the current enum."""
        valid = {s.value for s in AlignmentStrategy}
        source = (EXAMPLES_DIR / "02_comparing_strategies.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "strategies":
                        if isinstance(node.value, ast.List):
                            strategies = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
                            for s in strategies:
                                assert s in valid, (
                                    f"Example uses removed strategy '{s}'. "
                                    f"Valid: {valid}"
                                )

    def test_03_no_removed_aligner_params(self):
        """Verify 03_advanced_configuration doesn't pass removed Aligner params."""
        valid_params = set(inspect.signature(Aligner.__init__).parameters.keys())
        valid_params.discard("self")

        source = (EXAMPLES_DIR / "03_advanced_configuration.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "Aligner":
                    for kw in node.keywords:
                        if kw.arg is not None:
                            assert kw.arg in valid_params, (
                                f"Example passes removed param '{kw.arg}'. "
                                f"Valid: {valid_params}"
                            )

    def test_02_no_ctc_seg_reference(self):
        """Verify 02_comparing_strategies doesn't attempt ctc_seg."""
        source = (EXAMPLES_DIR / "02_comparing_strategies.py").read_text()
        assert "ctc_seg" not in source, "Example still references removed ctc_seg strategy"

    def test_03_no_ctc_refine_reference(self):
        """Verify 03_advanced_configuration doesn't use ctc_refine."""
        source = (EXAMPLES_DIR / "03_advanced_configuration.py").read_text()
        assert "ctc_refine" not in source, "Example still references removed ctc_refine parameter"


# ---------------------------------------------------------------------------
# Runtime smoke tests: actually run each example's main() with mocked I/O
# ---------------------------------------------------------------------------

class TestExample01Smoke:
    """Smoke test for 01_basic_usage.py."""

    def test_main_runs(self, capsys):
        mock_t = _mock_transcriber(114)

        with patch(
            "munajjam.transcription.WhisperTranscriber", return_value=mock_t,
        ):
            mod = _load_example("01_basic_usage.py")

            with patch.object(mod, "WhisperTranscriber", return_value=mock_t):
                with patch("builtins.open", mock_open()):
                    mod.main()

        captured = capsys.readouterr()
        assert "Processing Surah 114" in captured.out
        assert "Aligned" in captured.out


class TestExample02Smoke:
    """Smoke test for 02_comparing_strategies.py."""

    def test_main_runs(self, capsys):
        mock_t = _mock_transcriber(114)

        with patch(
            "munajjam.transcription.WhisperTranscriber", return_value=mock_t,
        ):
            mod = _load_example("02_comparing_strategies.py")

            with patch.object(mod, "WhisperTranscriber", return_value=mock_t):
                mod.main()

        captured = capsys.readouterr()
        assert "COMPARISON SUMMARY" in captured.out
        assert "RECOMMENDATIONS" in captured.out


class TestExample03Smoke:
    """Smoke test for 03_advanced_configuration.py."""

    def test_main_runs(self, capsys):
        mock_t = _mock_transcriber(114)
        fake_silences = [(4500, 5000), (8500, 9000)]

        with patch(
            "munajjam.transcription.WhisperTranscriber", return_value=mock_t,
        ), patch(
            "munajjam.transcription.detect_silences", return_value=fake_silences,
        ):
            mod = _load_example("03_advanced_configuration.py")

            with patch.object(mod, "WhisperTranscriber", return_value=mock_t), \
                 patch.object(mod, "detect_silences", return_value=fake_silences), \
                 patch("builtins.open", mock_open()):
                mod.main()

        captured = capsys.readouterr()
        assert "DETAILED RESULTS" in captured.out


class TestExample04Smoke:
    """Smoke test for 04_batch_processing.py."""

    def test_main_runs(self, capsys, tmp_path):
        mock_t = _mock_transcriber(114)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "114.wav").write_bytes(b"fake")

        output_dir = tmp_path / "output"

        with patch(
            "munajjam.transcription.WhisperTranscriber", return_value=mock_t,
        ):
            mod = _load_example("04_batch_processing.py")

            with patch.object(mod, "WhisperTranscriber", return_value=mock_t), \
                 patch.object(mod, "Path") as MockPath:

                mock_audio_path = MagicMock()
                mock_output_path = MagicMock(spec=Path)
                mock_output_path.mkdir = MagicMock()

                mock_audio_file = MagicMock()
                mock_audio_file.exists.return_value = True
                mock_audio_file.__str__ = lambda self: str(audio_dir / "114.wav")
                mock_audio_path.__truediv__ = MagicMock(return_value=mock_audio_file)

                mock_output_file = MagicMock()
                mock_output_path.__truediv__ = MagicMock(return_value=mock_output_file)

                calls = iter([mock_audio_path, mock_output_path])
                MockPath.side_effect = lambda x: next(calls)

                with patch("builtins.open", mock_open()):
                    mod.main()

        captured = capsys.readouterr()
        assert "BATCH PROCESSING SUMMARY" in captured.out
