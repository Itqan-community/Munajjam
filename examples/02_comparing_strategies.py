"""
Comparing Alignment Strategies

This example demonstrates the differences between the four alignment strategies:
- Greedy: Fast, simple linear matching
- DP: Optimal alignment using dynamic programming
- Hybrid: DP with greedy fallback (recommended)
- Auto: Automatically picks the best strategy
"""

from munajjam.transcription import WhisperTranscriber
from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs
import time


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
    strategies = ["greedy", "dp", "hybrid", "auto"]
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

For balanced performance:
  • Use HYBRID strategy - Combines DP quality with greedy fallback

For optimal alignment:
  • Use DP strategy - Best alignment quality using dynamic programming

For simple recordings:
  • Use GREEDY strategy - Fast and sufficient for 1:1 segment-to-ayah mapping
    """)


if __name__ == "__main__":
    main()
