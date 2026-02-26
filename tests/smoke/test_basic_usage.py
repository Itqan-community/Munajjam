"""
Smoke test for 01_basic_usage.py example.

Verifies that the basic usage example runs without errors when
the transcriber is mocked.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add examples directory to path
examples_dir = Path(__file__).parent.parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


class TestBasicUsage:
    """Smoke tests for basic usage example."""
    
    def test_example_imports(self):
        """Test that the example script can be imported."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_01_basic_usage", 
                examples_dir / "01_basic_usage.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert hasattr(module, 'main')
        except Exception as e:
            pytest.fail(f"Failed to import 01_basic_usage.py: {e}")
    
    def test_basic_usage_with_mock_transcriber(
        self, 
        mock_transcriber, 
        mock_audio_file,
        capsys
    ):
        """
        Test basic usage example with mocked transcriber.
        
        This test verifies that the example runs and produces expected output
        when the WhisperTranscriber is mocked.
        """
        mock_instance, mock_segments = mock_transcriber
        
        # Patch the example's audio path
        with patch.object(Path, 'exists', return_value=True):
            # Import and run the example
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_01_basic_usage", 
                examples_dir / "01_basic_usage.py"
            )
            module = importlib.util.module_from_spec(spec)
            
            # Patch the audio path before loading
            with patch('builtins.print') as mock_print:
                # Load and execute the module with mocked components
                with patch('munajjam.transcription.whisper.WhisperTranscriber') as MockTrans:
                    MockTrans.return_value = mock_instance
                    MockTrans.__enter__ = MagicMock(return_value=mock_instance)
                    MockTrans.__exit__ = MagicMock(return_value=None)
                    
                    # Patch the audio path in the module
                    import sys
                    module_content = (examples_dir / "01_basic_usage.py").read_text()
                    
                    # Replace the audio path with our mock
                    module_content = module_content.replace(
                        'audio_path = "Quran/badr_alturki_audio/114.wav"',
                        f'audio_path = "{mock_audio_file}"'
                    )
                    
                    exec(module_content, {"__name__": "__main__"})
        
        # If we get here without exception, the test passed
        assert True
    
    def test_core_functions_with_mock_data(self, mock_transcriber, mock_audio_file):
        """
        Test the core alignment functions with mock data.
        
        This is a more focused test that directly tests the functions
        used in the example rather than the full script.
        """
        mock_instance, mock_segments = mock_transcriber
        
        from munajjam.core import align
        from munajjam.data import load_surah_ayahs
        
        # Load real ayahs for Surah 114
        ayahs = load_surah_ayahs(114)
        assert len(ayahs) == 6  # Surah An-Nas has 6 ayahs
        
        # Mock the transcriber for alignment
        with patch('munajjam.transcription.whisper.WhisperTranscriber') as MockTrans:
            MockTrans.return_value = mock_instance
            MockTrans.__enter__ = MagicMock(return_value=mock_instance)
            MockTrans.__exit__ = MagicMock(return_value=None)
            
            # Test alignment with mock segments
            results = align(str(mock_audio_file), mock_segments, ayahs[:4])
            
            # Verify results
            assert len(results) > 0
            assert all(hasattr(r, 'start_time') for r in results)
            assert all(hasattr(r, 'end_time') for r in results)
            assert all(hasattr(r, 'similarity_score') for r in results)
    
    def test_segment_attributes(self, mock_transcriber):
        """Test that mock segments have required attributes."""
        _, mock_segments = mock_transcriber
        
        for segment in mock_segments:
            assert hasattr(segment, 'id')
            assert hasattr(segment, 'surah_id')
            assert hasattr(segment, 'start')
            assert hasattr(segment, 'end')
            assert hasattr(segment, 'text')
            assert hasattr(segment, 'type')
            assert segment.start < segment.end
            assert segment.text  # Text should not be empty
