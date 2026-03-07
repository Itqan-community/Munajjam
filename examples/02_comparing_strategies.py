"""
Comparing Alignment Strategies

This example demonstrates the differences between the six alignment strategies:
- Greedy: Fast, simple linear matching
- DP: Optimal alignment using dynamic programming
- Hybrid: DP with greedy fallback
- Word-level DP: Sub-segment precision using per-word timestamps
- CTC Segmentation: Acoustic-based alignment (requires audio_path)
- Auto: Automatically picks the best strategy (recommended)
"""

import time

from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs
from munajjam.formatters import format_alignment_results
from munajjam.transcription import WhisperTranscriber


def align_with_strategy(segments, ayahs, strategy_name, audio_path):
    """Align segments using the specified strategy and measure time."""
    print(f"\n{'=' * 80}")
    print(f"Testing {strategy_name.upper()} Strategy")
    print("=" * 80)

    start_time = time.time()

    kwargs = dict(strategy=strategy_name, fix_drift=True, fix_overlaps=True)
    aligner = Aligner(audio_path=audio_path, **kwargs)
    results = aligner.align(segments, ayahs)

    elapsed = time.time() - start_time

    # Calculate metrics
    avg_similarity = sum(r.similarity_score for r in results) / len(results)
    high_confidence = len([r for r in results if r.is_high_confidence])
    overlaps = sum(r.overlap_detected for r in results)

    print("\nResults:")
    print(f"  Time taken: {elapsed:.3f} seconds")
    print(f"  Average similarity: {avg_similarity:.2%}")
    print(
        f"  High confidence: {high_confidence}/{len(results)} ({high_confidence / len(results):.1%})"
    )
    print(f"  Overlaps detected: {overlaps}")

    # Show first 5 results as sample
    print("\n  First 5 ayahs:")
    for result in results[:5]:
        print(
            f"    Ayah {result.ayah.ayah_number:3d}: "
            f"{result.start_time:6.2f}s - {result.end_time:6.2f}s "
            f"(sim: {result.similarity_score:.2%})"
        )

    return results, elapsed, avg_similarity


def main():
    # Configuration
    audio_path = "Quran/badr_alturki_audio/114.wav"
    surah_number = 114

    print("Munajjam Alignment Strategy Comparison")
    print("=" * 80)

    # Step 1: Transcribe once (shared across all strategies)
    print("\nTranscribing audio...")
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(audio_path)

    print(f"Found {len(segments)} segments")

    # Step 2: Load reference ayahs
    ayahs = load_surah_ayahs(surah_number)
    print(f"Loaded {len(ayahs)} ayahs")

    # Step 3: Test each strategy
    strategies = ["greedy", "dp", "hybrid", "word_dp"]
    results_map = {}

    for strategy in strategies:
        results, elapsed, avg_sim = align_with_strategy(
            segments, ayahs, strategy, audio_path
        )
        results_map[strategy] = {
            "results": results,
            "time": elapsed,
            "avg_similarity": avg_sim,
        }

    # Step 3b: Test CTC segmentation (requires torchaudio)
    try:
        results, elapsed, avg_sim = align_with_strategy(
            segments, ayahs, "ctc_seg", audio_path=audio_path
        )
        results_map["ctc_seg"] = {
            "results": results,
            "time": elapsed,
            "avg_similarity": avg_sim,
        }
        strategies.append("ctc_seg")
    except Exception as e:
        print(f"\nSkipping CTC segmentation: {e}")

    # Step 4: Compare results
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Strategy':<12} {'Time (s)':<12} {'Avg Similarity':<16} {'Winner'}")
    print("-" * 80)

    # Find winners
    fastest = min(strategies, key=lambda s: results_map[s]["time"])
    most_accurate = max(strategies, key=lambda s: results_map[s]["avg_similarity"])

    for strategy in strategies:
        data = results_map[strategy]
        winner = []
        if strategy == fastest:
            winner.append("Fastest")
        if strategy == most_accurate:
            winner.append("Most Accurate")

        print(
            f"{strategy:<12} {data['time']:<12.3f} {data['avg_similarity']:<16.2%} {', '.join(winner)}"
        )

    # Step 5: Recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
For most use cases:
  • Use AUTO strategy (recommended) - Automatically picks the best approach
  • Includes automatic drift correction, overlap fixing, and zone realignment

For word-level precision:
  • Use WORD_DP strategy - Sub-segment alignment using per-word timestamps
  • Best when faster-whisper provides word-level timing

For acoustic alignment:
  • Use CTC_SEG strategy - Frame-accurate boundaries from audio signal
  • Requires audio_path and torchaudio

For simple recordings:
  • Use GREEDY strategy - Fast and sufficient for 1:1 segment-to-ayah mapping

For legacy workflows:
  • HYBRID or DP strategies remain available for backward compatibility
    """)

    # Step 6: Export best strategy results using the standardized formatter
    print(f"\n{'=' * 80}")
    print("EXPORT")
    print("=" * 80)

    best_results = results_map[most_accurate]["results"]
    output = format_alignment_results(
        results=best_results,
        surah_id=surah_number,
        audio_file=audio_path,
    )
    output_path = f"surah_{surah_number:03d}_best_alignment.json"
    output.to_file(output_path)
    print(f"\nBest strategy ({most_accurate}) results saved to: {output_path}")


if __name__ == "__main__":
    main()
