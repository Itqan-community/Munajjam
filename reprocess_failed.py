"""
Re-process failed surahs using Transformers Whisper backend.

Uses tarteel-ai/whisper-base-ar-quran model (Transformers backend)
instead of faster-whisper to re-process surahs that had alignment issues.

Problematic surahs:
    - 042 (Ash-Shura): 96% missing
    - 074 (Al-Muddathir): 82% missing
    - 027 (An-Naml): 68% missing
    - 035 (Fatir): 53% missing
    - 002 (Al-Baqarah): 23% missing
    - 107 (Al-Ma'un): 29% missing

Usage:
    python reprocess_failed.py
"""

import json
import os
import time
from pathlib import Path

from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.core import align_segments
from munajjam.data import load_surah_ayahs, get_surah_name


# Configuration
AUDIO_FOLDER = Path("Quran/badr_alturki_audio")
OUTPUT_FOLDER = Path("output")
RECITER_NAME = "Badr Al-Turki"

# Surahs to reprocess (had alignment issues with faster-whisper)
FAILED_SURAHS = [2, 27, 35, 42, 74, 107]


def delete_existing_output(surah_id: int) -> None:
    """Delete existing output file for a surah."""
    output_path = OUTPUT_FOLDER / f"surah_{surah_id:03d}.json"
    if output_path.exists():
        output_path.unlink()
        print(f"   🗑️  Deleted existing: {output_path}")


def save_output(surah_id: int, surah_name: str, results: list, total_ayahs: int) -> None:
    """Save alignment results to JSON file."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Calculate average similarity
    similarities = [r.similarity_score for r in results]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    # Build output data
    output_data = {
        "surah_id": surah_id,
        "surah_name": surah_name,
        "reciter": RECITER_NAME,
        "total_ayahs": total_ayahs,
        "aligned_ayahs": len(results),
        "avg_similarity": round(avg_similarity, 3),
        "ayahs": [
            {
                "ayah_number": r.ayah.ayah_number,
                "start": r.start_time,
                "end": r.end_time,
                "text": r.ayah.text,
                "similarity": round(r.similarity_score, 3),
            }
            for r in results
        ],
    }

    output_path = OUTPUT_FOLDER / f"surah_{surah_id:03d}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"   💾 Saved to {output_path}")


def process_surah(audio_path: Path, transcriber: WhisperTranscriber) -> dict:
    """Process a single surah and return result stats."""
    surah_id = int(audio_path.stem)
    surah_name = get_surah_name(surah_id)

    print(f"\n{'─' * 60}")
    print(f"📖 Processing Surah {surah_id:03d}: {surah_name}")
    print(f"   Audio: {audio_path}")
    print(f"{'─' * 60}")

    start_time = time.time()

    # Delete existing output first
    delete_existing_output(surah_id)

    try:
        # 1. Detect silences
        print("   🔇 Detecting silences...")
        silences = detect_silences(str(audio_path))
        print(f"   ✓ Found {len(silences)} silence gaps")

        # 2. Transcribe with progress
        print("   🎤 Transcribing segments...")

        def progress_callback(current: int, total: int, text: str):
            """Show transcription progress."""
            percent = (current / total) * 100
            bar_width = 30
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            display_text = text[:35] + "..." if len(text) > 35 else text
            print(f"\r      [{bar}] {current}/{total} ({percent:.0f}%) {display_text}", end="", flush=True)

        segments = transcriber.transcribe(str(audio_path), progress_callback=progress_callback)
        print()  # New line after progress bar
        print(f"   ✓ Transcribed {len(segments)} segments")

        # 3. Load ayahs and align
        print("   📝 Aligning segments...")
        ayahs = load_surah_ayahs(surah_id)
        aligned_results = align_segments(segments, ayahs, silences_ms=silences)

        # 4. Save output
        save_output(surah_id, surah_name, aligned_results, len(ayahs))

        processing_time = time.time() - start_time
        avg_similarity = (
            sum(r.similarity_score for r in aligned_results) / len(aligned_results)
            if aligned_results else 0.0
        )

        print(f"   ✅ Aligned {len(aligned_results)}/{len(ayahs)} ayahs "
              f"({avg_similarity:.1%} avg similarity) in {processing_time:.1f}s")

        return {
            "surah_id": surah_id,
            "surah_name": surah_name,
            "success": True,
            "total_ayahs": len(ayahs),
            "aligned_ayahs": len(aligned_results),
            "avg_similarity": avg_similarity,
            "processing_time": processing_time,
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            "surah_id": surah_id,
            "surah_name": surah_name,
            "success": False,
            "error": str(e),
        }


def main():
    """Main function to reprocess failed surahs."""
    print("\n" + "#" * 60)
    print("  REPROCESS FAILED SURAHS")
    print("  Using Transformers Whisper (tarteel-ai/whisper-base-ar-quran)")
    print("#" * 60)

    # Verify audio files exist
    audio_files = []
    for surah_id in FAILED_SURAHS:
        audio_path = AUDIO_FOLDER / f"{surah_id:03d}.wav"
        if audio_path.exists():
            audio_files.append(audio_path)
        else:
            print(f"⚠️  Audio file not found: {audio_path}")

    if not audio_files:
        print("❌ No audio files found!")
        return

    print(f"\n📁 Found {len(audio_files)} audio files to reprocess")
    print(f"   Surahs: {[int(f.stem) for f in audio_files]}")

    # Load model (Transformers backend)
    print("\n" + "=" * 60)
    print("Loading Transformers Whisper model...")
    print("=" * 60)

    transcriber = WhisperTranscriber(
        model_id="tarteel-ai/whisper-base-ar-quran",
        model_type="transformers"
    )
    transcriber.load()

    # Process each surah
    print("\n" + "=" * 60)
    print("Processing surahs...")
    print("=" * 60)

    results = []
    batch_start = time.time()

    try:
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]", end="")
            result = process_surah(audio_file, transcriber)
            results.append(result)

    finally:
        print("\n🧹 Unloading model...")
        transcriber.unload()

    batch_time = time.time() - batch_start

    # Print summary
    print("\n" + "=" * 60)
    print("REPROCESSING SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    for result in results:
        if result.get("success"):
            print(
                f"✅ Surah {result['surah_id']:03d} ({result['surah_name']}): "
                f"{result['aligned_ayahs']}/{result['total_ayahs']} ayahs "
                f"({result['avg_similarity']:.1%} avg) in {result['processing_time']:.1f}s"
            )
        else:
            print(f"❌ Surah {result['surah_id']:03d} ({result['surah_name']}): {result.get('error', 'Unknown error')}")

    print("=" * 60)
    print(f"Successfully processed: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {batch_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
