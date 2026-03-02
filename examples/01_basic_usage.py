"""
Basic Usage Example for Munajjam

This example demonstrates the simplest way to use Munajjam:
1. Transcribe an audio file
2. Load reference ayahs
3. Align segments to ayahs
4. Access results
"""

from munajjam.transcription import WhisperTranscriber
from munajjam.core import align
from munajjam.data import load_surah_ayahs


def main():
    # Path to your audio file
    audio_path = "Quran/badr_alturki_audio/114.wav"
    surah_number = 114

    print(f"Processing Surah {surah_number}...\n")

    # Step 1: Transcribe the audio
    print("Step 1: Transcribing audio...")
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(audio_path)

    print(f"  Found {len(segments)} segments")
    print(f"  Total duration: {segments[-1].end:.2f} seconds\n")

    # Step 2: Load reference ayahs for the surah
    print("Step 2: Loading reference ayahs...")
    ayahs = load_surah_ayahs(surah_number)
    print(f"  Loaded {len(ayahs)} ayahs\n")

    # Step 3: Align segments to ayahs (auto strategy by default, or pass strategy="word_dp" etc.)
    print("Step 3: Aligning segments to ayahs...")
    results = align(audio_path, segments, ayahs)
    print(f"  Aligned {len(results)} ayahs\n")

    # Step 4: Display results
    print("Results:")
    print("-" * 80)
    for result in results:
        print(
            f"Ayah {result.ayah.ayah_number:3d}: "
            f"{result.start_time:6.2f}s - {result.end_time:6.2f}s "
            f"(similarity: {result.similarity_score:.2%})"
        )

    # Step 5: Check alignment quality
    print("\n" + "=" * 80)
    print("Quality Metrics:")
    print("=" * 80)

    high_quality = [r for r in results if r.is_high_confidence]
    low_quality = [r for r in results if not r.is_high_confidence]
    avg_similarity = sum(r.similarity_score for r in results) / len(results)

    print(f"Average similarity: {avg_similarity:.2%}")
    print(
        f"High confidence ayahs: {len(high_quality)}/{len(results)} ({len(high_quality) / len(results):.1%})"
    )
    print(f"Low confidence ayahs: {len(low_quality)}/{len(results)}")

    if low_quality:
        print("\nLow confidence ayahs:")
        for r in low_quality:
            print(f"  Ayah {r.ayah.ayah_number}: {r.similarity_score:.2%}")


if __name__ == "__main__":
    main()
