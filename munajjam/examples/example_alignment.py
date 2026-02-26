#!/usr/bin/env python3
"""
Example script for Munajjam library.

Usage:
    python example_alignment.py <audio_path> <surah_id>

Examples:
    python example_alignment.py ../Quran/badr_alturki_audio/114.wav 114
    python example_alignment.py ../Quran/badr_alturki_audio/001.wav 1
"""

import json
import sys
import time
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from munajjam.core import Aligner, normalize_arabic, similarity
from munajjam.data import get_ayah_count, load_surah_ayahs
from munajjam.models import Segment, SegmentType
from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences


def run_with_transcription(audio_path: str, surah_id: int):
    """Full example: transcribe audio and align."""

    print("=" * 60)
    print(f"🧪 TESTING MUNAJJAM - SURAH {surah_id}")
    print("=" * 60)

    # Verify file exists
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    print(f"\n📂 Audio: {audio_path}")
    print(f"📖 Expected ayahs: {get_ayah_count(surah_id)}")

    # Step 1: Detect silences
    print("\n🔇 Step 1: Detecting silences...")
    start = time.time()
    silences = detect_silences(audio_path)
    print(f"   Found {len(silences)} silence periods in {time.time() - start:.2f}s")

    # Step 2: Transcribe
    print("\n📝 Step 2: Transcribing audio...")
    print("   (This may take a while depending on audio length)")
    start = time.time()

    transcriber = WhisperTranscriber()
    transcriber.load()
    print("   Model loaded.")

    try:
        segments = transcriber.transcribe(audio_path)
    finally:
        transcriber.unload()

    transcribe_time = time.time() - start
    print(f"   Transcribed {len(segments)} segments in {transcribe_time:.2f}s")

    # Show segments
    print("\n   Segments:")
    for seg in segments:
        icon = "🔹" if seg.type == SegmentType.AYAH else "⭐"
        text_preview = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        print(f"   {icon} [{seg.start:.2f}s-{seg.end:.2f}s] {text_preview}")

    # Step 3: Align
    print("\n🔗 Step 3: Aligning segments to ayahs...")
    ayahs = load_surah_ayahs(surah_id)
    print(f"   Loaded {len(ayahs)} reference ayahs")

    start = time.time()
    aligner = Aligner(audio_path=audio_path)
    results = aligner.align(segments, ayahs, silences_ms=silences)
    align_time = time.time() - start
    print(f"   Aligned {len(results)}/{len(ayahs)} ayahs in {align_time:.2f}s")

    # Show results
    print("\n📊 ALIGNMENT RESULTS:")
    print("-" * 60)
    for result in results:
        conf = "✅" if result.is_high_confidence else "⚠️"
        print(
            f"{conf} Ayah {result.ayah.ayah_number}: {result.start_time:.2f}s - {result.end_time:.2f}s"
        )
        print(f"   Score: {result.similarity_score:.2f} | Duration: {result.duration:.2f}s")
        text_preview = (
            result.ayah.text[:60] + "..." if len(result.ayah.text) > 60 else result.ayah.text
        )
        print(f"   Reference: {text_preview}")
        trans_preview = (
            result.transcribed_text[:60] + "..."
            if len(result.transcribed_text) > 60
            else result.transcribed_text
        )
        print(f"   Transcribed: {trans_preview}")
        print()

    # Generate JSON output
    output_file = f"output_surah_{surah_id:03d}.json"
    output = []
    for result in results:
        output.append(
            {
                "id": result.ayah.ayah_number,
                "sura_id": result.ayah.surah_id,
                "ayah_index": result.ayah.ayah_number - 1,
                "start": round(result.start_time, 2),
                "end": round(result.end_time, 2),
                "transcribed_text": result.transcribed_text,
                "corrected_text": result.ayah.text,
                "similarity_score": round(result.similarity_score, 3),
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved to: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"Total ayahs: {len(ayahs)}")
    print(f"Aligned ayahs: {len(results)}")
    print(f"Coverage: {len(results) / len(ayahs) * 100:.0f}%")
    if results:
        avg_score = sum(r.similarity_score for r in results) / len(results)
        print(f"Average similarity: {avg_score:.2f}")
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Alignment time: {align_time:.2f}s")
    print("=" * 60)


def run_with_existing_segments(surah_id: int, audio_path: str | None = None):
    """Example: alignment using existing segment data (skip transcription)."""

    print("=" * 60)
    print(f"🧪 TESTING ALIGNMENT ONLY - SURAH {surah_id}")
    print("=" * 60)

    # Load existing segments from cache directory
    segments_file = Path(f"../../cache/surah_{surah_id:03d}_segments.json")
    silences_file = Path(f"../../cache/surah_{surah_id:03d}_silences.json")

    if not segments_file.exists():
        print(f"❌ Segments file not found: {segments_file}")
        print("   Run transcription first or use run_with_transcription()")
        return

    print(f"\n📂 Loading existing segments from: {segments_file}")

    with open(segments_file, encoding="utf-8") as f:
        raw_segments = json.load(f)

    # Convert to Segment objects
    segments = []
    for seg in raw_segments:
        seg_type = SegmentType.AYAH
        if seg.get("type") == "istiadha":
            seg_type = SegmentType.ISTIADHA
        elif seg.get("type") == "basmala":
            seg_type = SegmentType.BASMALA

        segments.append(
            Segment(
                id=seg["id"],
                surah_id=seg["sura_id"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                type=seg_type,
            )
        )

    print(f"   Loaded {len(segments)} segments")

    # Load silences if available
    silences = []
    if silences_file.exists():
        with open(silences_file, encoding="utf-8") as f:
            silences = json.load(f)
        print(f"   Loaded {len(silences)} silence periods")

    # Show segments
    print("\n   Segments:")
    for seg in segments[:10]:  # Show first 10
        icon = "🔹" if seg.type == SegmentType.AYAH else "⭐"
        text_preview = seg.text[:40] + "..." if len(seg.text) > 40 else seg.text
        print(f"   {icon} [{seg.start:.2f}s-{seg.end:.2f}s] {text_preview}")
    if len(segments) > 10:
        print(f"   ... and {len(segments) - 10} more")

    # Load reference ayahs
    print(f"\n📖 Loading reference ayahs for Surah {surah_id}...")
    ayahs = load_surah_ayahs(surah_id)
    print(f"   Loaded {len(ayahs)} ayahs")

    # Align
    if audio_path is None:
        audio_path = f"../../Quran/badr_alturki_audio/{surah_id:03d}.wav"

    print("\n🔗 Aligning...")
    start = time.time()
    aligner = Aligner(audio_path=audio_path)
    results = aligner.align(segments, ayahs, silences_ms=silences)
    align_time = time.time() - start
    print(f"   Aligned {len(results)}/{len(ayahs)} ayahs in {align_time:.2f}s")

    # Show results
    print("\n📊 RESULTS (first 10):")
    print("-" * 60)
    for result in results[:10]:
        conf = "✅" if result.is_high_confidence else "⚠️"
        print(
            f"{conf} Ayah {result.ayah.ayah_number}: "
            f"{result.start_time:.2f}s - {result.end_time:.2f}s "
            f"(score: {result.similarity_score:.2f})"
        )

    if len(results) > 10:
        print(f"... and {len(results) - 10} more")

    # Compare with existing corrected segments
    corrected_file = Path(f"../../data/corrected_segments/corrected_segments_{surah_id:03d}.json")
    if corrected_file.exists():
        print(f"\n📁 Comparing with existing: {corrected_file}")
        with open(corrected_file, encoding="utf-8") as f:
            existing = json.load(f)
        existing_ayahs = [e for e in existing if e.get("id", 0) != 0]
        print(f"   Existing has {len(existing_ayahs)} ayah entries")

        # Quick comparison
        matches = 0
        for result in results:
            existing_match = next(
                (e for e in existing_ayahs if e.get("ayah_index") == result.ayah.ayah_number - 1),
                None,
            )
            if existing_match:
                time_diff = abs(result.start_time - existing_match["start"])
                if time_diff < 0.5:  # Within 0.5s
                    matches += 1

        print(f"   Timing matches (within 0.5s): {matches}/{len(results)}")

    print("\n" + "=" * 60)


def run_core_functions():
    """Example: core functions without audio."""

    print("=" * 60)
    print("🧪 TESTING CORE FUNCTIONS")
    print("=" * 60)

    # Test Arabic normalization
    print("\n📝 Arabic Normalization:")
    test_cases = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ",
        "ٱلۡحَمۡدُ لِلَّهِ رَبِّ ٱلۡعَٰلَمِينَ",
    ]
    for text in test_cases:
        normalized = normalize_arabic(text)
        print(f"   Original:   {text}")
        print(f"   Normalized: {normalized}")
        print()

    # Test similarity
    print("📊 Similarity Tests:")
    pairs = [
        ("بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم"),
        ("بسم الله", "بسم الله الرحمن الرحيم"),
        ("الحمد لله رب العالمين", "الحمد لله رب العلمين"),  # Typo
    ]
    for text1, text2 in pairs:
        score = similarity(text1, text2)
        print(f"   '{text1[:30]}...' vs '{text2[:30]}...'")
        print(f"   Score: {score:.2f}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Munajjam Example Script")
        print()
        print("Usage:")
        print(
            "  python example_alignment.py <audio_path> <surah_id>  # Full test with transcription"
        )
        print(
            "  python example_alignment.py --existing <surah_id>    # Test with existing segments"
        )
        print("  python example_alignment.py --core                   # Test core functions only")
        print()
        print("Examples:")
        print(
            "  python example_alignment.py ../../Quran/badr_alturki_audio/001.wav 1"
            "   # Al-Fatiha (7 ayahs)"
        )
        print(
            "  python example_alignment.py ../../Quran/badr_alturki_audio/062.wav 62"
            "  # Al-Jumu'ah (11 ayahs)"
        )
        print(
            "  python example_alignment.py --existing 67"
            "                               # Use existing data"
        )
        print(
            "  python example_alignment.py --core"
            "                                      # Test core only"
        )
        sys.exit(0)

    if sys.argv[1] == "--core":
        run_core_functions()
    elif sys.argv[1] == "--existing":
        if len(sys.argv) < 3:
            print("Usage: python example_alignment.py --existing <surah_id>")
            sys.exit(1)
        surah_id = int(sys.argv[2])
        run_with_existing_segments(surah_id)
    else:
        if len(sys.argv) < 3:
            print("Usage: python example_alignment.py <audio_path> <surah_id>")
            sys.exit(1)
        audio_path = sys.argv[1]
        surah_id = int(sys.argv[2])
        run_with_transcription(audio_path, surah_id)
