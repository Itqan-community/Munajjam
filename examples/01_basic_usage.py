"""
Basic Usage Example for Munajjam

This example demonstrates the simplest way to use Munajjam:
1. Transcribe an audio file
2. Load reference ayahs
3. Align segments to ayahs
4. Format and export results using the standardized JSON formatter
"""

from munajjam.transcription import WhisperTranscriber
from munajjam.core import align
from munajjam.data import load_surah_ayahs
from munajjam.formatters import format_alignment_results


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

    # Step 3: Align segments to ayahs (auto strategy by default, or pass strategy="dp" etc.)
    print("Step 3: Aligning segments to ayahs...")
    results = align(audio_path, segments, ayahs)
    print(f"  Aligned {len(results)} ayahs\n")

    # Step 4: Format results using the standardized JSON formatter
    print("Step 4: Formatting results...")
    output = format_alignment_results(
        results=results,
        surah_id=surah_number,
        reciter="Badr Al-Turki",
        audio_file=audio_path,
    )

    # Display formatted JSON output
    print("\nFormatted JSON Output:")
    print("=" * 80)
    print(output.to_json())

    # Save to file
    output.to_file("output/surah_114.json")
    print("\nResults saved to output/surah_114.json")

    # Step 5: Check alignment quality from metadata
    print("\n" + "=" * 80)
    print("Quality Metrics:")
    print("=" * 80)
    meta = output.metadata
    print(f"Average similarity: {meta.average_confidence:.2%}")
    print(f"High confidence ayahs: {meta.high_confidence_count}/{meta.total_ayahs}")
    print(f"Total duration: {meta.total_duration:.2f}s")


if __name__ == "__main__":
    main()
