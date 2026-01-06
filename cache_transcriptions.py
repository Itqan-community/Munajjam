#!/usr/bin/env python3
"""
Cache transcription segments and silences for faster algorithm iteration.

This script:
1. Transcribes audio using Whisper
2. Detects silences using pydub
3. Saves both to cache files so alignment can be re-run without re-transcription

Usage:
    # Cache a single surah
    python cache_transcriptions.py --surah 9
    
    # Cache multiple surahs
    python cache_transcriptions.py --surah 9 2 18
    
    # Cache all available surahs
    python cache_transcriptions.py --all

Output:
    cache/surah_XXX_segments.json  - Transcribed segments
    cache/surah_XXX_silences.json  - Detected silence periods
"""

import json
import sys
import argparse
import time
from pathlib import Path
from dataclasses import asdict

# Add munajjam to path
sys.path.insert(0, str(Path(__file__).parent / "munajjam"))

from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.data import get_surah_name


# Configuration - same as batch_process.py
AUDIO_FOLDER = Path("Quran/badr_alturki_audio")  # Main audio folder
CACHE_FOLDER = Path("cache")
MODEL_ID = "OdyAsh/faster-whisper-base-ar-quran"
MODEL_TYPE = "faster-whisper"
DEVICE = "cpu"


def get_audio_path(surah_id: int) -> Path | None:
    """Find audio file for a surah."""
    # Check main folder
    audio_path = AUDIO_FOLDER / f"{surah_id:03d}.wav"
    if audio_path.exists():
        return audio_path
    
    # Check alternative folders
    alt_folders = [
        Path("Quran/naser_alosfour"),
    ]
    for folder in alt_folders:
        alt_path = folder / f"{surah_id:03d}.wav"
        if alt_path.exists():
            return alt_path
    
    return None


def cache_exists(surah_id: int) -> bool:
    """Check if cache already exists for a surah."""
    segments_file = CACHE_FOLDER / f"surah_{surah_id:03d}_segments.json"
    silences_file = CACHE_FOLDER / f"surah_{surah_id:03d}_silences.json"
    return segments_file.exists() and silences_file.exists()


def save_segments(surah_id: int, segments: list):
    """Save segments to cache."""
    CACHE_FOLDER.mkdir(exist_ok=True)
    output_file = CACHE_FOLDER / f"surah_{surah_id:03d}_segments.json"
    
    data = [
        {
            "id": seg.id,
            "surah_id": seg.surah_id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "type": seg.type.value,
            "confidence": getattr(seg, 'confidence', None),
        }
        for seg in segments
    ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"   💾 Saved segments to {output_file}")


def save_silences(surah_id: int, silences: list):
    """Save silences to cache."""
    CACHE_FOLDER.mkdir(exist_ok=True)
    output_file = CACHE_FOLDER / f"surah_{surah_id:03d}_silences.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(silences, f)
    
    print(f"   💾 Saved silences to {output_file}")


def cache_surah(surah_id: int, transcriber: "WhisperTranscriber", force: bool = False):
    """Cache segments and silences for a single surah."""
    surah_name = get_surah_name(surah_id)
    
    print(f"\n{'='*60}")
    print(f"  Caching Surah {surah_id}: {surah_name}")
    print(f"{'='*60}")
    
    # Check if already cached
    if cache_exists(surah_id) and not force:
        print(f"  ⏭️  Already cached, skipping (use --force to overwrite)")
        return True
    
    # Find audio file
    audio_path = get_audio_path(surah_id)
    if not audio_path:
        print(f"  ❌ Audio file not found for surah {surah_id}")
        return False
    
    print(f"  📁 Audio: {audio_path}")
    
    start_time = time.time()
    
    # 1. Detect silences
    print(f"  🔇 Detecting silences...")
    try:
        silences = detect_silences(str(audio_path))
        print(f"     Found {len(silences)} silence gaps")
        save_silences(surah_id, silences)
    except Exception as e:
        print(f"  ❌ Silence detection failed: {e}")
        return False
    
    # 2. Transcribe segments
    print(f"  🎤 Transcribing audio...")
    
    def progress_callback(current: int, total: int, text: str):
        percent = (current / total) * 100
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        display_text = text[:30] + "..." if len(text) > 30 else text
        print(f"\r     [{bar}] {current}/{total} ({percent:.0f}%) {display_text}", end="", flush=True)
    
    try:
        segments = transcriber.transcribe(str(audio_path), progress_callback=progress_callback)
        print()  # New line after progress bar
        print(f"     Transcribed {len(segments)} segments")
        save_segments(surah_id, segments)
    except Exception as e:
        print(f"  ❌ Transcription failed: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"  ✅ Cached in {elapsed:.1f}s")
    
    return True


def get_all_available_surahs() -> list[int]:
    """Get list of all surahs with available audio files."""
    surahs = []
    for surah_id in range(1, 115):
        if get_audio_path(surah_id):
            surahs.append(surah_id)
    return surahs


def main():
    parser = argparse.ArgumentParser(description="Cache transcriptions for fast iteration")
    parser.add_argument("--surah", "-s", type=int, nargs="+", help="Surah ID(s) to cache")
    parser.add_argument("--all", "-a", action="store_true", help="Cache all available surahs")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing cache")
    args = parser.parse_args()
    
    if not args.surah and not args.all:
        parser.error("Specify --surah ID(s) or --all")
    
    # Determine which surahs to cache
    if args.all:
        surah_ids = get_all_available_surahs()
        print(f"\n📚 Found {len(surah_ids)} surahs with audio files")
    else:
        surah_ids = args.surah
    
    print(f"\n🚀 Initializing Whisper transcriber...")
    print(f"   Model: {MODEL_ID}")
    print(f"   Device: {DEVICE}")
    
    transcriber = WhisperTranscriber(
        model_id=MODEL_ID,
        model_type=MODEL_TYPE,
        device=DEVICE,
    )
    transcriber.load()
    print(f"   ✅ Model loaded")
    
    # Process each surah
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for surah_id in surah_ids:
        if cache_exists(surah_id) and not args.force:
            skip_count += 1
            print(f"\n  ⏭️  Surah {surah_id}: Already cached")
            continue
        
        if cache_surah(surah_id, transcriber, args.force):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  CACHING SUMMARY")
    print(f"{'='*60}")
    print(f"  ✅ Cached: {success_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"{'='*60}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
