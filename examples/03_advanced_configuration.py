"""
Advanced Configuration Example

This example demonstrates advanced usage:
- Custom configuration settings
- Silence detection and usage
- Progress tracking
- Energy snap for precise boundaries
- Detailed result inspection
"""

from munajjam.config import configure
from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs
from munajjam.formatters import format_alignment_results
from munajjam.transcription import WhisperTranscriber, detect_silences


def progress_callback(current, total):
    """Progress callback for alignment."""
    percentage = (current / total) * 100
    print(f"  Progress: {current}/{total} ayahs ({percentage:.1f}%)", end="\r")


def main():
    audio_path = "Quran/badr_alturki_audio/114.wav"
    surah_number = 114

    print("Advanced Munajjam Configuration Example")
    print("=" * 80)

    print("\nStep 1: Configuring Munajjam...")
    configure(
        model_id="OdyAsh/faster-whisper-base-ar-quran",
        device="auto",
        model_type="faster-whisper",
        silence_threshold_db=-30,
        min_silence_ms=300,
        buffer_seconds=0.3,
    )
    print("  Configuration complete")

    print("\nStep 2: Detecting silences in audio...")

    silences_ms = detect_silences(
        audio_path=audio_path, min_silence_len=300, silence_thresh=-30
    )

    print(f"  Found {len(silences_ms)} silence periods")
    total_silence = sum(end - start for start, end in silences_ms) / 1000
    print(f"  Total silence duration: {total_silence:.2f} seconds")

    print("\nStep 3: Transcribing audio...")
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(audio_path)

    print(f"  Found {len(segments)} segments")

    from munajjam.models import SegmentType

    ayah_segments = [s for s in segments if s.type == SegmentType.AYAH]
    istiadha_segments = [s for s in segments if s.type == SegmentType.ISTIADHA]
    basmala_segments = [s for s in segments if s.type == SegmentType.BASMALA]

    print(f"    - Ayah segments: {len(ayah_segments)}")
    print(f"    - Istiadha segments: {len(istiadha_segments)}")
    print(f"    - Basmala segments: {len(basmala_segments)}")

    print("\nStep 4: Loading reference ayahs...")
    ayahs = load_surah_ayahs(surah_number)
    print(f"  Loaded {len(ayahs)} ayahs")

    print("\nStep 5: Aligning with advanced settings...")

    aligner = Aligner(
        audio_path=audio_path,
        strategy="auto",
        quality_threshold=0.85,
        fix_drift=True,
        fix_overlaps=True,
        energy_snap=True,
    )

    results = aligner.align(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences_ms,
        on_progress=progress_callback,
    )

    print(f"\n  Alignment complete: {len(results)} ayahs")

    if aligner.last_stats:
        print("\nHybrid Strategy Statistics:")
        stats = aligner.last_stats
        print(f"  Total ayahs: {stats.total_ayahs}")
        print(f"  DP kept (high quality): {stats.dp_kept}")
        print(f"  Greedy fallback used: {stats.old_fallback}")
        print(f"  Split-and-restitch improved: {stats.split_improved}")
        print(f"  Still low quality: {stats.still_low}")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    excellent = [r for r in results if r.similarity_score >= 0.95]
    good = [r for r in results if 0.85 <= r.similarity_score < 0.95]
    fair = [r for r in results if 0.70 <= r.similarity_score < 0.85]
    poor = [r for r in results if r.similarity_score < 0.70]

    print("\nQuality Distribution:")
    print(f"  Excellent (>=95%): {len(excellent)} ayahs")
    print(f"  Good (85-95%): {len(good)} ayahs")
    print(f"  Fair (70-85%): {len(fair)} ayahs")
    print(f"  Poor (<70%): {len(poor)} ayahs")

    overlaps = [r for r in results if r.overlap_detected]
    if overlaps:
        print(f"\nOverlaps detected: {len(overlaps)} ayahs")
        for r in overlaps[:5]:
            print(
                f"  Ayah {r.ayah.ayah_number}: {r.start_time:.2f}s - {r.end_time:.2f}s"
            )

    if poor:
        print("\nPoor quality ayahs (need review):")
        for r in poor:
            print(f"  Ayah {r.ayah.ayah_number}: {r.similarity_score:.2%}")
            print(f"    Expected: {r.ayah.text[:50]}...")
            print(f"    Got: {r.transcribed_text[:50]}...")

    print("\nStep 8: Exporting results...")

    output = format_alignment_results(
        results=results,
        surah_id=surah_number,
        audio_file=audio_path,
    )

    output_path = f"surah_{surah_number:03d}_alignment.json"
    output.to_file(output_path)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
