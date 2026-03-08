"""
Batch Processing Example

This example demonstrates how to process multiple surahs efficiently:
- Reuse the loaded transcription model
- Process surahs in a loop
- Aggregate statistics across surahs
- Generate summary reports
"""

import json
import time
from pathlib import Path

from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs
from munajjam.formatters import format_alignment_results
from munajjam.transcription import WhisperTranscriber


def process_surah(transcriber, audio_path, surah_number):
    """Process a single surah and return results with stats."""
    print(f"\nProcessing Surah {surah_number}...")

    start_time = time.time()

    # Transcribe
    segments = transcriber.transcribe(str(audio_path))

    # Load ayahs
    ayahs = load_surah_ayahs(surah_number)

    # Align (audio_path is required, full pipeline runs by default)
    aligner = Aligner(audio_path=str(audio_path))
    results = aligner.align(segments, ayahs)

    elapsed = time.time() - start_time

    # Calculate stats
    avg_similarity = sum(r.similarity_score for r in results) / len(results)
    high_confidence = len([r for r in results if r.is_high_confidence])
    overlaps = sum(r.overlap_detected for r in results)

    stats = {
        "surah_number": surah_number,
        "total_ayahs": len(results),
        "processing_time": round(elapsed, 2),
        "avg_similarity": round(avg_similarity, 4),
        "high_confidence_count": high_confidence,
        "high_confidence_pct": round(high_confidence / len(results), 4),
        "overlaps": overlaps,
    }

    print(f"  ✓ Completed in {elapsed:.2f}s")
    print(f"    Avg similarity: {avg_similarity:.2%}")
    print(f"    High confidence: {high_confidence}/{len(results)}")

    return results, stats


def main():
    # Configuration
    audio_directory = Path("Quran/badr_alturki_audio")
    output_directory = Path("output_examples")
    output_directory.mkdir(exist_ok=True)

    # Define which surahs to process
    # Option 1: Process specific surahs
    surahs_to_process = [114]  # Just test with Surah An-Nas (shortest)

    # Option 2: Process all surahs
    # surahs_to_process = range(1, 115)

    print("Batch Processing Quran Surahs")
    print("=" * 80)
    print(f"Processing {len(surahs_to_process)} surahs")
    print(f"Output directory: {output_directory}")

    # Initialize transcriber and aligner (reused across all surahs)
    print("\nInitializing models...")
    transcriber = WhisperTranscriber()
    transcriber.load()

    print("  ✓ Models loaded")

    # Process each surah
    all_stats = []
    failed_surahs = []

    total_start = time.time()

    for i, surah_number in enumerate(surahs_to_process, 1):
        print(f"\n[{i}/{len(surahs_to_process)}]", end=" ")

        # Find audio file (adjust pattern as needed)
        audio_file = audio_directory / f"{surah_number:03d}.wav"

        if not audio_file.exists():
            print(f"Surah {surah_number}: ⚠ Audio file not found: {audio_file}")
            failed_surahs.append(surah_number)
            continue

        try:
            # Process surah
            results, stats = process_surah(transcriber, audio_file, surah_number)

            # Save results using the standardized formatter
            output_file = output_directory / f"surah_{surah_number:03d}_alignment.json"

            output = format_alignment_results(
                results=results,
                surah_id=surah_number,
                audio_file=str(audio_file),
            )
            output.to_file(str(output_file))

            all_stats.append(stats)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_surahs.append(surah_number)

    total_elapsed = time.time() - total_start

    # Cleanup
    transcriber.unload()

    # Generate summary report
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)

    print(f"\nTotal processing time: {total_elapsed:.2f} seconds")
    print(f"Successfully processed: {len(all_stats)}/{len(surahs_to_process)} surahs")

    if failed_surahs:
        print(f"Failed surahs: {failed_surahs}")

    if all_stats:
        # Aggregate statistics
        total_ayahs = sum(s["total_ayahs"] for s in all_stats)
        avg_similarity_overall = sum(s["avg_similarity"] for s in all_stats) / len(
            all_stats
        )
        total_overlaps = sum(s["overlaps"] for s in all_stats)
        avg_processing_time = sum(s["processing_time"] for s in all_stats) / len(
            all_stats
        )

        print("\nOverall Statistics:")
        print(f"  Total ayahs processed: {total_ayahs}")
        print(f"  Overall avg similarity: {avg_similarity_overall:.2%}")
        print(f"  Total overlaps detected: {total_overlaps}")
        print(f"  Avg processing time per surah: {avg_processing_time:.2f}s")

        # Find best and worst surahs
        best_surah = max(all_stats, key=lambda s: s["avg_similarity"])
        worst_surah = min(all_stats, key=lambda s: s["avg_similarity"])

        print(
            f"\nBest alignment: Surah {best_surah['surah_number']} ({best_surah['avg_similarity']:.2%})"
        )
        print(
            f"Worst alignment: Surah {worst_surah['surah_number']} ({worst_surah['avg_similarity']:.2%})"
        )

        # Save summary report
        summary_file = output_directory / "batch_summary.json"
        summary_data = {
            "processed_surahs": len(all_stats),
            "failed_surahs": failed_surahs,
            "total_processing_time": round(total_elapsed, 2),
            "overall_stats": {
                "total_ayahs": total_ayahs,
                "avg_similarity": round(avg_similarity_overall, 4),
                "total_overlaps": total_overlaps,
                "avg_processing_time": round(avg_processing_time, 2),
            },
            "per_surah_stats": all_stats,
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
