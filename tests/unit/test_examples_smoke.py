"""
Smoke tests for example scripts.

Verify that example scripts stay in sync with the library API.
All heavy dependencies (WhisperTranscriber, audio files, models) are mocked.
No model downloads, no audio files, no PyTorch required.
"""

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from munajjam.core.aligner import AlignmentStrategy
from munajjam.models import AlignmentResult, Ayah, Segment, SegmentType

# --------------- Path constants ---------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_ROOT = REPO_ROOT / "examples"
MUNAJJAM_EXAMPLES = REPO_ROOT / "munajjam" / "examples"


# --------------- Shared fixtures ---------------


@pytest.fixture()
def mock_segments() -> list[Segment]:
    """Fake transcribed segments returned by the mocked transcriber."""
    return [
        Segment(
            id=0,
            surah_id=1,
            start=0.0,
            end=4.5,
            text="أعوذ بالله من الشيطان الرجيم",
            type=SegmentType.ISTIADHA,
        ),
        Segment(
            id=1,
            surah_id=1,
            start=5.0,
            end=8.5,
            text="بسم الله الرحمن الرحيم",
            type=SegmentType.BASMALA,
        ),
        Segment(
            id=2,
            surah_id=1,
            start=9.0,
            end=13.5,
            text="الحمد لله رب العالمين",
            type=SegmentType.AYAH,
        ),
        Segment(
            id=3,
            surah_id=1,
            start=14.0,
            end=18.0,
            text="الرحمن الرحيم",
            type=SegmentType.AYAH,
        ),
    ]


@pytest.fixture()
def mock_ayahs() -> list[Ayah]:
    """Reference ayahs for Surah Al-Fatiha (first 4)."""
    return [
        Ayah(
            id=1,
            surah_id=1,
            ayah_number=1,
            text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        ),
        Ayah(
            id=2,
            surah_id=1,
            ayah_number=2,
            text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        ),
        Ayah(
            id=3,
            surah_id=1,
            ayah_number=3,
            text="الرَّحْمَٰنِ الرَّحِيمِ",
        ),
        Ayah(
            id=4,
            surah_id=1,
            ayah_number=4,
            text="مَالِكِ يَوْمِ الدِّينِ",
        ),
    ]


@pytest.fixture()
def mock_alignment_results(mock_ayahs: list[Ayah]) -> list[AlignmentResult]:
    """Fake alignment results."""
    return [
        AlignmentResult(
            ayah=mock_ayahs[0],
            start_time=5.0,
            end_time=8.5,
            transcribed_text="بسم الله الرحمن الرحيم",
            similarity_score=0.95,
        ),
        AlignmentResult(
            ayah=mock_ayahs[1],
            start_time=9.0,
            end_time=13.5,
            transcribed_text="الحمد لله رب العالمين",
            similarity_score=0.92,
        ),
    ]


def _load_module_from_file(file_path: Path, module_name: str) -> types.ModuleType:
    """Load a Python module from a file path using importlib."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------- Static Import Verification ---------------


class TestExampleImportsStatic:
    """Statically verify that all munajjam imports in example files resolve."""

    @pytest.mark.parametrize(
        "example_path",
        [
            EXAMPLES_ROOT / "01_basic_usage.py",
            EXAMPLES_ROOT / "02_comparing_strategies.py",
            EXAMPLES_ROOT / "03_advanced_configuration.py",
            EXAMPLES_ROOT / "04_batch_processing.py",
            MUNAJJAM_EXAMPLES / "basic_usage.py",
            MUNAJJAM_EXAMPLES / "test_alignment.py",
        ],
        ids=lambda p: p.name,
    )
    def test_all_munajjam_imports_resolve(self, example_path: Path) -> None:
        """Parse example source and verify each munajjam import resolves."""
        source = example_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("munajjam"):
                    mod = importlib.import_module(node.module)
                    for alias in node.names:
                        assert hasattr(mod, alias.name), (
                            f"{example_path.name}: "
                            f"{node.module}.{alias.name} does not exist"
                        )


# --------------- Strategy Validation ---------------


class TestExampleStrategies:
    """Verify examples only reference valid alignment strategies."""

    VALID_STRATEGIES = {s.value for s in AlignmentStrategy}

    def test_02_only_uses_valid_strategies(self) -> None:
        """02_comparing_strategies.py must only list valid strategies."""
        source = (EXAMPLES_ROOT / "02_comparing_strategies.py").read_text()
        tree = ast.parse(source)

        found_strategies_list = False
        # Find the strategies list assignment
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "strategies":
                        if isinstance(node.value, ast.List):
                            found_strategies_list = True
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    assert elt.value in self.VALID_STRATEGIES, (
                                        f"02_comparing_strategies.py references "
                                        f"removed strategy '{elt.value}'. "
                                        f"Valid: {self.VALID_STRATEGIES}"
                                    )
        assert found_strategies_list, (
            "02_comparing_strategies.py: could not find 'strategies = [...]' "
            "list literal. Was the variable renamed?"
        )

    def test_02_no_ctc_seg_block(self) -> None:
        """02_comparing_strategies.py must not reference ctc_seg strategy."""
        source = (EXAMPLES_ROOT / "02_comparing_strategies.py").read_text()
        assert "ctc_seg" not in source, (
            "02_comparing_strategies.py still references removed 'ctc_seg' strategy"
        )

    def test_03_no_ctc_refine_kwarg(self) -> None:
        """03_advanced_configuration.py must not pass ctc_refine to Aligner."""
        source = (EXAMPLES_ROOT / "03_advanced_configuration.py").read_text()
        assert "ctc_refine" not in source, (
            "03_advanced_configuration.py passes removed 'ctc_refine' kwarg to Aligner"
        )


# --------------- API Contract Verification ---------------


class TestBasicUsageContract:
    """Verify munajjam/examples/basic_usage.py uses correct API signatures."""

    def test_align_called_with_audio_path(self) -> None:
        """basic_usage.py must pass audio_path as first arg to align()."""
        source = (MUNAJJAM_EXAMPLES / "basic_usage.py").read_text()
        # align(segments, ayahs) is wrong — must be align(audio_path, segments, ayahs)
        assert "align(segments, ayahs)" not in source, (
            "basic_usage.py calls align(segments, ayahs) without required "
            "audio_path. Correct: align(audio_path, segments, ayahs)"
        )


# --------------- Smoke Tests: Repo-Root Examples ---------------


class TestExample01BasicUsage:
    """Smoke tests for examples/01_basic_usage.py."""

    def test_main_runs(
        self,
        mock_segments: list[Segment],
        mock_ayahs: list[Ayah],
        mock_alignment_results: list[AlignmentResult],
    ) -> None:
        """01_basic_usage.py main() runs end-to-end with mocked deps."""
        mock_transcriber = MagicMock()
        mock_transcriber.__enter__ = MagicMock(return_value=mock_transcriber)
        mock_transcriber.__exit__ = MagicMock(return_value=False)
        mock_transcriber.transcribe.return_value = mock_segments

        # Module must be loaded INSIDE the patch block so imports bind
        # to the mocked objects at exec_module() time.
        with (
            patch(
                "munajjam.transcription.WhisperTranscriber",
                return_value=mock_transcriber,
            ),
            patch(
                "munajjam.data.load_surah_ayahs",
                return_value=mock_ayahs,
            ),
            patch(
                "munajjam.core.align",
                return_value=mock_alignment_results,
            ),
        ):
            mod = _load_module_from_file(
                EXAMPLES_ROOT / "01_basic_usage.py",
                "example_01_basic_usage",
            )
            mod.main()

            mock_transcriber.transcribe.assert_called_once()


class TestExample02ComparingStrategies:
    """Smoke tests for examples/02_comparing_strategies.py."""

    def test_align_with_strategy_runs(
        self,
        mock_segments: list[Segment],
        mock_ayahs: list[Ayah],
        mock_alignment_results: list[AlignmentResult],
    ) -> None:
        """align_with_strategy() runs with mocked Aligner."""
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = mock_alignment_results

        with patch(
            "munajjam.core.Aligner",
            return_value=mock_aligner,
        ):
            mod = _load_module_from_file(
                EXAMPLES_ROOT / "02_comparing_strategies.py",
                "example_02_comparing",
            )
            results, elapsed, avg_sim = mod.align_with_strategy(
                mock_segments, mock_ayahs, "greedy", "/fake/audio.wav",
            )
            assert isinstance(results, list)
            assert len(results) > 0
            assert isinstance(elapsed, float)


class TestExample03AdvancedConfiguration:
    """Smoke tests for examples/03_advanced_configuration.py."""

    def test_aligner_kwargs_valid(self) -> None:
        """03 must only pass valid kwargs to Aligner.__init__."""
        import inspect

        from munajjam.core import Aligner

        valid_params = set(inspect.signature(Aligner.__init__).parameters.keys())
        valid_params.discard("self")

        source = (EXAMPLES_ROOT / "03_advanced_configuration.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match Aligner(...) calls
                if isinstance(func, ast.Name) and func.id == "Aligner":
                    for kw in node.keywords:
                        if kw.arg is not None:
                            assert kw.arg in valid_params, (
                                f"03_advanced_configuration.py passes "
                                f"invalid kwarg '{kw.arg}' to Aligner(). "
                                f"Valid: {valid_params}"
                            )

    def test_aligner_align_kwargs_valid(self) -> None:
        """03 must only pass valid kwargs to Aligner.align()."""
        import inspect

        from munajjam.core import Aligner

        valid_params = set(inspect.signature(Aligner.align).parameters.keys())
        valid_params.discard("self")

        source = (EXAMPLES_ROOT / "03_advanced_configuration.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match aligner.align(...) calls
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "align"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "aligner"
                ):
                    for kw in node.keywords:
                        if kw.arg is not None:
                            assert kw.arg in valid_params, (
                                f"03_advanced_configuration.py passes "
                                f"invalid kwarg '{kw.arg}' to aligner.align(). "
                                f"Valid: {valid_params}"
                            )

    def test_main_runs(
        self,
        mock_segments: list[Segment],
        mock_ayahs: list[Ayah],
        mock_alignment_results: list[AlignmentResult],
    ) -> None:
        """03_advanced_configuration.py main() runs end-to-end with mocked deps."""
        mock_transcriber = MagicMock()
        mock_transcriber.__enter__ = MagicMock(return_value=mock_transcriber)
        mock_transcriber.__exit__ = MagicMock(return_value=False)
        mock_transcriber.transcribe.return_value = mock_segments

        mock_aligner = MagicMock()
        mock_aligner.align.return_value = mock_alignment_results
        mock_aligner.last_stats = None

        mock_silences = [(1000, 1500), (3000, 3800)]

        with (
            patch(
                "munajjam.transcription.WhisperTranscriber",
                return_value=mock_transcriber,
            ),
            patch(
                "munajjam.transcription.detect_silences",
                return_value=mock_silences,
            ),
            patch(
                "munajjam.data.load_surah_ayahs",
                return_value=mock_ayahs,
            ),
            patch(
                "munajjam.core.Aligner",
                return_value=mock_aligner,
            ),
            patch("munajjam.config.configure"),
            patch("builtins.open", MagicMock()),
        ):
            mod = _load_module_from_file(
                EXAMPLES_ROOT / "03_advanced_configuration.py",
                "example_03_advanced",
            )
            mod.main()

            mock_transcriber.transcribe.assert_called_once()
            mock_aligner.align.assert_called_once()


class TestExample04BatchProcessing:
    """Smoke tests for examples/04_batch_processing.py."""

    def test_process_surah_runs(
        self,
        mock_segments: list[Segment],
        mock_ayahs: list[Ayah],
        mock_alignment_results: list[AlignmentResult],
    ) -> None:
        """process_surah() runs with mocked transcriber and aligner."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_segments

        mock_aligner = MagicMock()
        mock_aligner.align.return_value = mock_alignment_results

        with (
            patch(
                "munajjam.core.Aligner",
                return_value=mock_aligner,
            ),
            patch(
                "munajjam.data.load_surah_ayahs",
                return_value=mock_ayahs,
            ),
        ):
            mod = _load_module_from_file(
                EXAMPLES_ROOT / "04_batch_processing.py",
                "example_04_batch",
            )
            results, stats = mod.process_surah(
                mock_transcriber,
                Path("/fake/audio.wav"),
                1,
            )
            assert isinstance(results, list)
            assert isinstance(stats, dict)
            assert "surah_number" in stats
            assert "avg_similarity" in stats
            assert "high_confidence_count" in stats


# --------------- Smoke Tests: munajjam/examples/ ---------------


class TestMunajjamExampleBasicUsage:
    """Smoke tests for munajjam/examples/basic_usage.py."""

    def test_process_surah_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_segments: list[Segment],
        mock_ayahs: list[Ayah],
        mock_alignment_results: list[AlignmentResult],
    ) -> None:
        """process_surah() runs end-to-end with mocked deps."""
        mock_transcriber = MagicMock()
        mock_transcriber.__enter__ = MagicMock(return_value=mock_transcriber)
        mock_transcriber.__exit__ = MagicMock(return_value=False)
        mock_transcriber.transcribe.return_value = mock_segments

        monkeypatch.syspath_prepend(str(MUNAJJAM_EXAMPLES))

        with (
            patch(
                "munajjam.transcription.WhisperTranscriber",
                return_value=mock_transcriber,
            ),
            patch(
                "munajjam.data.load_surah_ayahs",
                return_value=mock_ayahs,
            ),
            patch(
                "munajjam.core.align",
                return_value=mock_alignment_results,
            ),
        ):
            if "basic_usage" in sys.modules:
                del sys.modules["basic_usage"]
            try:
                import basic_usage  # type: ignore[import-not-found]

                output = basic_usage.process_surah(
                    "/fake/audio.wav", 1, "Test Reciter",
                )

                assert isinstance(output, list)
                assert len(output) > 0
                for item in output:
                    assert "id" in item
                    assert "sura_id" in item
                    assert "ayah_index" in item
                    assert "start" in item
                    assert "end" in item
                    assert "transcribed_text" in item
                    assert "similarity_score" in item
            finally:
                if "basic_usage" in sys.modules:
                    del sys.modules["basic_usage"]


class TestMunajjamExampleAlignment:
    """Smoke tests for munajjam/examples/test_alignment.py."""

    def test_core_functions_runs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """test_core_functions() runs without error (pure computation)."""
        monkeypatch.syspath_prepend(str(MUNAJJAM_EXAMPLES))

        if "test_alignment" in sys.modules:
            del sys.modules["test_alignment"]
        try:
            import test_alignment  # type: ignore[import-not-found]

            # Uses only normalize_arabic() and similarity() — no mocking needed
            test_alignment.test_core_functions()
        finally:
            if "test_alignment" in sys.modules:
                del sys.modules["test_alignment"]

    def test_segment_types_valid(self) -> None:
        """test_alignment.py uses valid SegmentType enum values."""
        assert hasattr(SegmentType, "AYAH")
        assert hasattr(SegmentType, "ISTIADHA")
        assert hasattr(SegmentType, "BASMALA")
