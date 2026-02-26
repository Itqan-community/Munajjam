"""
Smoke test for 04_batch_processing.py example.

Verifies that the batch processing example runs without errors when
the transcriber is mocked.
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add examples directory to path
examples_dir = Path(__file__).parent.parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


class TestBatchProcessing:
    """Smoke tests for batch processing example."""
    
    def test_example_imports(self):
        """Test that the example script can be imported."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_04_batch_processing", 
                examples_dir / "04_batch_processing.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert hasattr(module, 'main')
            assert hasattr(module, 'process_surah')
        except Exception as e:
            pytest.fail(f"Failed to import 04_batch_processing.py: {e}")
    
    def test_process_surah_function(
        self, 
        mock_transcriber, 
        mock_audio_directory
    ):
        """
        Test the process_surah function from the example.
        
        This simulates what the example does for a single surah.
        """
        mock_instance, mock_segments = mock_transcriber
        
        from munajjam.transcription import WhisperTranscriber
        from munajjam.core import Aligner
        from munajjam.data import load_surah_ayahs
        
        surah_number = 114
        audio_file = mock_audio_directory / "114.wav"
        
        # Create mock transcriber
        transcriber = WhisperTranscriber()
        transcriber.load()
        
        # Transcribe
        segments = transcriber.transcribe(str(audio_file))
        assert len(segments) == len(mock_segments)
        
        # Load ayahs
        ayahs = load_surah_ayahs(surah_number)
        assert len(ayahs) == 6
        
        # Align
        aligner = Aligner(audio_path=str(audio_file))
        results = aligner.align(segments, ayahs)
        
        assert len(results) > 0
        
        # Calculate stats like the example
        avg_similarity = sum(r.similarity_score for r in results) / len(results)
        high_confidence = len([r for r in results if r.is_high_confidence])
        overlaps = sum(r.overlap_detected for r in results)
        
        stats = {
            "surah_number": surah_number,
            "total_ayahs": len(results),
            "avg_similarity": round(avg_similarity, 4),
            "high_confidence_count": high_confidence,
            "high_confidence_pct": round(high_confidence / len(results), 4),
            "overlaps": overlaps,
        }
        
        # Verify stats
        assert stats["surah_number"] == 114
        assert stats["total_ayahs"] == len(results)
        assert 0 <= stats["avg_similarity"] <= 1
        assert 0 <= stats["high_confidence_pct"] <= 1
    
    def test_json_export(self, mock_transcriber, mock_audio_directory, tmp_path):
        """
        Test JSON export functionality from the example.
        
        The example exports results to JSON files.
        """
        mock_instance, mock_segments = mock_transcriber
        
        from munajjam.core import Aligner
        from munajjam.data import load_surah_ayahs
        
        surah_number = 114
        audio_file = mock_audio_directory / "114.wav"
        
        ayahs = load_surah_ayahs(surah_number)
        aligner = Aligner(audio_path=str(audio_file))
        results = aligner.align(mock_segments, ayahs[:len(mock_segments)])
        
        # Export to JSON like the example
        output_data = {
            "surah_number": surah_number,
            "results": [
                {
                    "ayah_number": r.ayah.ayah_number,
                    "start_time": round(r.start_time, 3),
                    "end_time": round(r.end_time, 3),
                    "similarity_score": round(r.similarity_score, 3),
                    "overlap_detected": r.overlap_detected,
                }
                for r in results
            ]
        }
        
        output_file = tmp_path / f"surah_{surah_number:03d}_alignment.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Verify file was created and is valid JSON
        assert output_file.exists()
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["surah_number"] == surah_number
        assert len(loaded_data["results"]) == len(results)
    
    def test_batch_summary_generation(
        self, 
        mock_transcriber, 
        mock_audio_directory, 
        tmp_path
    ):
        """
        Test batch summary generation from the example.
        
        The example generates a summary report after batch processing.
        """
        mock_instance, mock_segments = mock_transcriber
        
        from munajjam.core import Aligner
        from munajjam.data import load_surah_ayahs
        
        # Simulate processing multiple surahs
        all_stats = []
        
        for surah_number in [1, 114]:
            audio_file = mock_audio_directory / f"{surah_number:03d}.wav"
            
            if audio_file.exists():
                try:
                    ayahs = load_surah_ayahs(surah_number)
                    aligner = Aligner(audio_path=str(audio_file))
                    results = aligner.align(mock_segments[:len(ayahs)], ayahs)
                    
                    avg_similarity = sum(r.similarity_score for r in results) / len(results)
                    high_confidence = len([r for r in results if r.is_high_confidence])
                    
                    stats = {
                        "surah_number": surah_number,
                        "total_ayahs": len(results),
                        "avg_similarity": round(avg_similarity, 4),
                        "high_confidence_count": high_confidence,
                        "high_confidence_pct": round(high_confidence / len(results), 4),
                    }
                    all_stats.append(stats)
                except Exception as e:
                    pytest.fail(f"Failed to process surah {surah_number}: {e}")
        
        # Generate summary like the example
        if all_stats:
            total_ayahs = sum(s["total_ayahs"] for s in all_stats)
            avg_similarity_overall = sum(s["avg_similarity"] for s in all_stats) / len(all_stats)
            avg_processing_time = 1.0  # Mock value
            
            summary_data = {
                "processed_surahs": len(all_stats),
                "failed_surahs": [],
                "total_processing_time": 10.0,
                "overall_stats": {
                    "total_ayahs": total_ayahs,
                    "avg_similarity": round(avg_similarity_overall, 4),
                    "avg_processing_time": round(avg_processing_time, 2),
                },
                "per_surah_stats": all_stats,
            }
            
            summary_file = tmp_path / "batch_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            
            # Verify summary
            assert summary_file.exists()
            with open(summary_file, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["processed_surahs"] == len(all_stats)
            assert "overall_stats" in loaded
    
    def test_transcriber_reuse(self, mock_transcriber, mock_audio_directory):
        """
        Test that transcriber can be reused across multiple files.
        
        The example creates one transcriber and reuses it for efficiency.
        """
        mock_instance, mock_segments = mock_transcriber
        
        from munajjam.transcription import WhisperTranscriber
        
        # Create and load transcriber once
        transcriber = WhisperTranscriber()
        transcriber.load()
        
        assert transcriber.is_loaded
        
        # Process multiple files
        for surah_number in [1, 114]:
            audio_file = mock_audio_directory / f"{surah_number:03d}.wav"
            if audio_file.exists():
                segments = transcriber.transcribe(str(audio_file))
                assert len(segments) == len(mock_segments)
        
        # Unload when done
        transcriber.unload()
        assert not transcriber.is_loaded
    
    def test_transcriber_context_manager(self, mock_transcriber, mock_audio_directory):
        """
        Test transcriber context manager pattern.
        
        The example uses the context manager (with statement).
        """
        mock_instance, mock_segments = mock_transcriber
        
        from munajjam.transcription import WhisperTranscriber
        
        # Use context manager
        with WhisperTranscriber() as transcriber:
            assert transcriber.is_loaded
            audio_file = mock_audio_directory / "114.wav"
            segments = transcriber.transcribe(str(audio_file))
            assert len(segments) == len(mock_segments)
        
        # After exiting context, should be unloaded
        assert not mock_instance.is_loaded
