"""
Smoke test for 03_advanced_configuration.py example.

Verifies that the advanced configuration example runs without errors when
the transcriber and silence detection are mocked.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add examples directory to path
examples_dir = Path(__file__).parent.parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


class TestAdvancedConfiguration:
    """Smoke tests for advanced configuration example."""
    
    def test_example_imports(self):
        """Test that the example script can be imported."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_03_advanced_configuration", 
                examples_dir / "03_advanced_configuration.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert hasattr(module, 'main')
            assert hasattr(module, 'progress_callback')
        except Exception as e:
            pytest.fail(f"Failed to import 03_advanced_configuration.py: {e}")
    
    def test_silence_detection_mock(self, mock_silence_detection, mock_audio_file):
        """
        Test that silence detection can be mocked.
        
        The example uses detect_silences, so we verify our mock works.
        """
        from munajjam.transcription import detect_silences
        
        silences = detect_silences(
            audio_path=str(mock_audio_file),
            min_silence_len=300,
            silence_thresh=-30
        )
        
        # Should return our mocked silences
        assert len(silences) == 3
        assert silences[0] == (4500, 5000)
    
    def test_configuration_setup(self):
        """
        Test that global configuration can be set.
        
        The example calls configure() to set up global settings.
        """
        from munajjam.config import configure, get_settings
        
        # Configure with test settings
        configure(
            model_id="test-model",
            device="cpu",
            model_type="transformers",
            silence_threshold_db=-30,
            min_silence_ms=300,
            buffer_seconds=0.3,
        )
        
        settings = get_settings()
        assert settings.model_id == "test-model"
        assert settings.device == "cpu"
    
    def test_aligner_with_silences(
        self, 
        mock_transcriber, 
        mock_silence_detection, 
        mock_audio_file
    ):
        """
        Test alignment with provided silence periods.
        
        The example passes detected silences to the align method.
        """
        mock_instance, mock_segments = mock_transcriber
        mock_silences = mock_silence_detection
        
        from munajjam.core import Aligner
        from munajjam.data import load_surah_ayahs
        
        ayahs = load_surah_ayahs(114)
        
        aligner = Aligner(
            audio_path=str(mock_audio_file),
            strategy="auto",
            quality_threshold=0.85,
            fix_drift=True,
            fix_overlaps=True,
            ctc_refine=True,
            energy_snap=True,
        )
        
        # Progress callback for testing
        def progress_callback(current, total):
            assert 0 <= current <= total
        
        results = aligner.align(
            segments=mock_segments,
            ayahs=ayahs[:len(mock_segments)],
            silences_ms=mock_silences,
            on_progress=progress_callback
        )
        
        assert len(results) > 0
    
    def test_quality_distribution_calculation(self, mock_transcriber, mock_audio_file):
        """
        Test quality distribution calculation like the example does.
        
        The example groups results by quality levels.
        """
        mock_instance, mock_segments = mock_transcriber
        
        from munajjam.core import Aligner
        from munajjam.data import load_surah_ayahs
        
        ayahs = load_surah_ayahs(114)
        
        aligner = Aligner(audio_path=str(mock_audio_file))
        results = aligner.align(mock_segments, ayahs[:len(mock_segments)])
        
        # Group by quality like the example
        excellent = [r for r in results if r.similarity_score >= 0.95]
        good = [r for r in results if 0.85 <= r.similarity_score < 0.95]
        fair = [r for r in results if 0.70 <= r.similarity_score < 0.85]
        poor = [r for r in results if r.similarity_score < 0.70]
        
        # Verify groupings are valid
        total = len(excellent) + len(good) + len(fair) + len(poor)
        assert total == len(results)
        
        # Check overlap detection
        overlaps = [r for r in results if r.overlap_detected]
        assert isinstance(overlaps, list)
    
    def test_segment_type_inspection(self, mock_transcriber):
        """
        Test segment type inspection like the example does.
        
        The example counts different segment types (ayah, istiadha, basmala).
        """
        from munajjam.models import SegmentType
        
        _, mock_segments = mock_transcriber
        
        ayah_segments = [s for s in mock_segments if s.type == SegmentType.AYAH]
        istiadha_segments = [s for s in mock_segments if s.type == SegmentType.ISTIADHA]
        basmala_segments = [s for s in mock_segments if s.type == SegmentType.BASMALA]
        
        # Verify we have the expected mix
        assert len(istiadha_segments) >= 1
        assert len(basmala_segments) >= 1
        assert len(ayah_segments) >= 1
        
        total = len(ayah_segments) + len(istiadha_segments) + len(basmala_segments)
        assert total == len(mock_segments)
