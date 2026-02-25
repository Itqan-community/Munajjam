"""
Basic usage example for Munajjam library.

This example demonstrates the core workflow:
1. Transcribe audio to segments
2. Align segments to ayahs
3. Format and export results using the standardized JSON formatter
"""

from pathlib import Path

from munajjam.core import align
from munajjam.data import load_surah_ayahs
from munajjam.formatters import format_alignment_results

# Import core components
from munajjam.transcription import WhisperTranscriber


def process_surah(audio_path: str, surah_id: int, reciter: str = "Unknown"):
    """
    Process a single surah audio file.

    Args:
        audio_path: Path to the audio file (WAV)
        surah_id: Surah number (1-114)
        reciter: Name of the reciter

    Returns:
        An AlignmentOutput instance.
    """
    print(f"Processing Surah {surah_id} from {audio_path}")
    print("=" * 50)

    # Step 1: Transcribe audio
    print("\n📝 Step 1: Transcribing audio...")

    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(audio_path)

    print(f"   Transcribed {len(segments)} segments")
    for seg in segments[:3]:  # Show first 3
        print(f"   - {seg.start:.2f}s-{seg.end:.2f}s: {seg.text[:40]}...")

    # Step 2: Load reference ayahs
    print("\n📖 Step 2: Loading reference ayahs...")

    ayahs = load_surah_ayahs(surah_id)
    print(f"   Loaded {len(ayahs)} ayahs for Surah {surah_id}")

    # Step 3: Align segments to ayahs
    print("\n🔗 Step 3: Aligning segments to ayahs...")

    results = align(audio_path, segments, ayahs)
    print(f"   Aligned {len(results)} ayahs")

    # Step 4: Format results using the standardized JSON formatter
    print("\n📄 Step 4: Formatting results...")

    output = format_alignment_results(
        results=results,
        surah_id=surah_id,
        reciter=reciter,
        audio_file=audio_path,
    )

    # Show alignment results
    print("\n📊 Alignment Results Summary:")
    for result in output.results[:5]:  # Show first 5
        confidence = "✅" if result.is_high_confidence else "⚠️"
        print(
            f"   {confidence} Ayah {result.ayah_number}: "
            f"{result.start_time:.2f}s - {result.end_time:.2f}s "
            f"(score: {result.similarity_score:.2f})"
        )

    if len(output.results) > 5:
        print(f"   ... and {len(output.results) - 5} more")

    return output


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python basic_usage.py <audio_path> <surah_id> [reciter_name]")
        print("Example: python basic_usage.py surah_001.wav 1 'Badr Al-Turki'")
        sys.exit(1)

    audio_path = sys.argv[1]
    surah_id = int(sys.argv[2])
    reciter = sys.argv[3] if len(sys.argv) > 3 else "Unknown Reciter"

    # Verify audio file exists
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Process the surah
    output = process_surah(audio_path, surah_id, reciter)

    # Save to JSON using the standardized formatter's method
    output_path = f"corrected_segments_{surah_id:03d}.json"
    print(f"\n💾 Saving to: {output_path}")
    output.to_file(output_path, ensure_ascii=False)
    print("   ✅ Saved successfully!")

    print("\n🎉 Done!")
