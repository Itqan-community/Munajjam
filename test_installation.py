"""
Test script to verify Munajjam installation and GPU support.

This script tests:
1. Python and package imports
2. PyTorch CUDA availability
3. Whisper model loading
4. Basic transcription functionality
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Test 1: Package Imports")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"[OK] Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"[FAIL] Transformers import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"[OK] Librosa: {librosa.__version__}")
    except ImportError as e:
        print(f"[FAIL] Librosa import failed: {e}")
        return False
    
    try:
        import faster_whisper
        print(f"[OK] Faster Whisper: {faster_whisper.__version__}")
    except ImportError as e:
        print(f"[WARN] Faster Whisper not available: {e}")
        print("  (This is optional, but recommended for faster transcription)")
    
    try:
        from munajjam import config, transcription, core, data, models
        print("[OK] Munajjam library imported successfully")
    except ImportError as e:
        print(f"✗ Munajjam import failed: {e}")
        return False
    
    print()
    return True


def test_gpu():
    """Test GPU availability and CUDA."""
    print("=" * 60)
    print("Test 2: GPU and CUDA Support")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
            # Test tensor creation on GPU
            try:
                test_tensor = torch.randn(10, 10).cuda()
                print("[OK] GPU tensor operations working")
            except Exception as e:
                print(f"[WARN] GPU tensor test failed: {e}")
        else:
            print("[WARN] CUDA not available - will use CPU")
            print("  If you have an NVIDIA GPU, make sure:")
            print("  1. NVIDIA drivers are installed")
            print("  2. CUDA toolkit is installed")
            print("  3. PyTorch was installed with CUDA support")
        
        print()
        return True
        
    except Exception as e:
        print(f"[FAIL] GPU test failed: {e}")
        print()
        return False


def test_whisper_model():
    """Test loading the Whisper model."""
    print("=" * 60)
    print("Test 3: Whisper Model Loading")
    print("=" * 60)
    
    try:
        from munajjam.transcription import WhisperTranscriber
        from munajjam.config import get_settings
        
        settings = get_settings()
        print(f"Model ID: {settings.model_id}")
        print(f"Device: {settings.device}")
        print(f"Model Type: {settings.model_type}")
        print()
        
        print("Loading Whisper model (this may take a minute)...")
        transcriber = WhisperTranscriber()
        
        try:
            transcriber.load()
            print("[OK] Model loaded successfully")
            print(f"  Device: {transcriber.device}")
            print(f"  Model ID: {transcriber.model_id}")
            
            transcriber.unload()
            print("[OK] Model unloaded successfully")
            print()
            return True
            
        except Exception as e:
            print(f"[FAIL] Model loading failed: {e}")
            print()
            return False
            
    except Exception as e:
        print(f"[FAIL] Whisper test setup failed: {e}")
        print()
        return False


def test_basic_functionality():
    """Test basic library functionality."""
    print("=" * 60)
    print("Test 4: Basic Functionality")
    print("=" * 60)
    
    try:
        from munajjam.data import load_surah_ayahs
        from munajjam.core import normalize_arabic, similarity
        
        # Test data loading
        print("Testing data loading...")
        ayahs = load_surah_ayahs(1)
        print(f"[OK] Loaded {len(ayahs)} ayahs for Surah 1")
        
        # Test Arabic normalization
        print("Testing Arabic normalization...")
        test_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        normalized = normalize_arabic(test_text)
        try:
            print(f"[OK] Normalized: {normalized[:30]}...")
        except UnicodeEncodeError:
            print("[OK] Normalized text (Arabic characters)")
        
        # Test similarity
        print("Testing similarity matching...")
        score = similarity("بسم الله", "بسم الله الرحمن")
        print(f"[OK] Similarity score: {score:.2f}")
        
        print()
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_transcription_with_sample():
    """Test transcription if a sample audio file exists."""
    print("=" * 60)
    print("Test 5: Transcription Test (Optional)")
    print("=" * 60)
    
    # Look for sample audio files
    sample_paths = [
        "data/001.wav",
        "data/surah_001.wav",
        "001.wav",
        "surah_001.wav",
    ]
    
    audio_file = None
    for path in sample_paths:
        if Path(path).exists():
            audio_file = path
            break
    
    if not audio_file:
        print("[WARN] No sample audio file found - skipping transcription test")
        print("  To test transcription, place a WAV file named '001.wav' or 'surah_001.wav'")
        print("  in the data/ directory or project root")
        print()
        return True
    
    try:
        from munajjam.transcription import WhisperTranscriber
        
        print(f"Found audio file: {audio_file}")
        print("Transcribing (this may take a few minutes)...")
        
        with WhisperTranscriber() as transcriber:
            segments = transcriber.transcribe(audio_file)
        
        print(f"[OK] Transcription successful!")
        print(f"  Found {len(segments)} segments")
        
        if segments:
            print("\n  First few segments:")
            for seg in segments[:3]:
                print(f"    {seg.start:.2f}s - {seg.end:.2f}s: {seg.text[:50]}...")
        
        print()
        return True
        
    except Exception as e:
        print(f"[WARN] Transcription test failed: {e}")
        print("  This might be due to missing audio file or model issues")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Munajjam Installation Test Suite")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("GPU Support", test_gpu()))
    results.append(("Whisper Model", test_whisper_model()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Transcription", test_transcription_with_sample()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
        if result:
            passed += 1
    
    print()
    print(f"Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n[SUCCESS] All tests passed! Your installation is working correctly.")
        return 0
    elif passed >= len(results) - 1:
        print("\n[WARN] Most tests passed. Check any failed tests above.")
        return 0
    else:
        print("\n[ERROR] Some critical tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

