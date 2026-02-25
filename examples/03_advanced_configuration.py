"""
Advanced Configuration Example

This example demonstrates advanced usage:
- Custom configuration settings
- Silence detection and usage
- Progress tracking
- CTC refinement and energy snap
- Detailed result inspection
"""

from munajjam.transcription import WhisperTranscriber, detect_silences
from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs
from munajjam.config import configure


def progress_callback(current, total):
    """Progress callback for alignment."""
    percentage = (current / total) * 100
    print(f"  Progress: {current}/{total} ayahs ({percentage:.1f}%)", end="\r")


def main():
    # Configuration
    audio_path = "Quran/badr_alturki_audio/114.wav"
    surah_number = 114

    print("Advanced Munajjam Configuration Example")
    print("=" * 80)

    # Step 1: Configure global settings
    print("\nStep 1: Configuring Munajjam...")
    configure(
        model_id="OdyAsh/faster-whisper-base-ar-quran",
        device="auto",  # Auto-detect GPU/CPU
        model_type="faster-whisper",
        silence_threshold_db=-30,
        min_silence_ms=300,
        buffer_seconds=0.3,
    )
    print("  Configuration complete")

    # Step 2: Detect silences in audio (optional but recommended)
    print("\nStep 2: Detecting silences in audio...")

    silences_ms = detect_silences(
        audio_path=audio_path, min_silence_len=300, silence_thresh=-30
    )

    print(f"  Found {len(silences_ms)} silence periods")
    total_silence = sum(end - start for start, end in silences_ms) / 1000
    print(f"  Total silence duration: {total_silence:.2f} seconds")

    # Step 3: Transcribe with custom settings
    print("\nStep 3: Transcribing audio...")
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(audio_path)

    print(f"  Found {len(segments)} segments")

    # Inspect segment types
    from munajjam.models import SegmentType

    ayah_segments = [s for s in segments if s.type == SegmentType.AYAH]
    istiadha_segments = [s for s in segments if s.type == SegmentType.ISTIADHA]
    basmala_segments = [s for s in segments if s.type == SegmentType.BASMALA]

    print(f"    - Ayah segments: {len(ayah_segments)}")
    print(f"    - Istiadha segments: {len(istiadha_segments)}")
    print(f"    - Basmala segments: {len(basmala_segments)}")

    # Step 4: Load reference ayahs
    print("\nStep 4: Loading reference ayahs...")
    ayahs = load_surah_ayahs(surah_number)
    print(f"  Loaded {len(ayahs)} ayahs")

    # Step 5: Align with advanced configuration
    print("\nStep 5: Aligning with advanced settings...")

    aligner = Aligner(
        audio_path=audio_path,  # Audio file (required)
        strategy="auto",
        quality_threshold=0.85,  # Threshold for high-quality alignment
        fix_drift=True,  # Enable zone realignment
        fix_overlaps=True,  # Fix overlapping ayahs
        ctc_refine=True,  # Refine boundaries with CTC forced alignment (default)
        energy_snap=True,  # Snap boundaries to energy minima (default)
    )

    results = aligner.align(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences_ms,  # Use detected silences
        on_progress=progress_callback,  # Track progress
    )

    print(f"\n  Alignment complete: {len(results)} ayahs")

    # Step 6: Inspect hybrid stats (if available)
    if aligner.last_stats:
        print("\nHybrid Strategy Statistics:")
        stats = aligner.last_stats
        print(f"  Total ayahs: {stats.total_ayahs}")
        print(f"  DP kept (high quality): {stats.dp_kept}")
        print(f"  Greedy fallback used: {stats.old_fallback}")
        print(f"  Split-and-restitch improved: {stats.split_improved}")
        print(f"  Still low quality: {stats.still_low}")

    # Step 7: Detailed result inspection
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    # Group results by quality
    excellent = [r for r in results if r.similarity_score >= 0.95]
    good = [r for r in results if 0.85 <= r.similarity_score < 0.95]
    fair = [r for r in results if 0.70 <= r.similarity_score < 0.85]
    poor = [r for r in results if r.similarity_score < 0.70]

    print("\nQuality Distribution:")
    print(f"  Excellent (≥95%): {len(excellent)} ayahs")
    print(f"  Good (85-95%): {len(good)} ayahs")
    print(f"  Fair (70-85%): {len(fair)} ayahs")
    print(f"  Poor (<70%): {len(poor)} ayahs")

    # Show overlaps
    overlaps = [r for r in results if r.overlap_detected]
    if overlaps:
        print(f"\nOverlaps detected: {len(overlaps)} ayahs")
        for r in overlaps[:5]:  # Show first 5
            print(
                f"  Ayah {r.ayah.ayah_number}: {r.start_time:.2f}s - {r.end_time:.2f}s"
            )

    # Show poor quality ayahs for investigation
    if poor:
        print("\nPoor quality ayahs (need review):")
        for r in poor:
            print(f"  Ayah {r.ayah.ayah_number}: {r.similarity_score:.2%}")
            print(f"    Expected: {r.ayah.text[:50]}...")
            print(f"    Got: {r.transcribed_text[:50]}...")

    # Step 8: Export to JSON
    print("\nStep 8: Exporting results...")
    import json

    output = []
    for r in results:
        output.append(
            {
                "ayah_number": r.ayah.ayah_number,
                "start_time": round(r.start_time, 3),
                "end_time": round(r.end_time, 3),
                "duration": round(r.duration, 3),
                "similarity_score": round(r.similarity_score, 3),
                "overlap_detected": r.overlap_detected,
                "transcribed_text": r.transcribed_text,
            }
        )

    output_path = f"surah_{surah_number:03d}_alignment.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
