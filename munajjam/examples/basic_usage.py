"""
Basic usage example for Munajjam library.

This example demonstrates the core workflow:
1. Transcribe audio to segments
2. Align segments to ayahs
3. Output results as JSON using the standardized formatter
"""

import json
from pathlib import Path

# Import core components
from munajjam.transcription import WhisperTranscriber
from munajjam.core import align
from munajjam.data import load_surah_ayahs
from munajjam.output import format_alignment_results


def process_surah(audio_path: str, surah_id: int, reciter: str = "Unknown"):
    """
    Process a single surah audio file.
    
    Args:
        audio_path: Path to the audio file (WAV)
        surah_id: Surah number (1-114)
        reciter: Name of the reciter
    
    Returns:
        AlignmentOutput object with formatted results
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
    
    results = align(segments, ayahs)
    print(f"   Aligned {len(results)} ayahs")
    
    # Show alignment results
    print("\n📊 Alignment Results:")
    for result in results[:5]:  # Show first 5
        confidence = "✅" if result.is_high_confidence else "⚠️"
        print(
            f"   {confidence} Ayah {result.ayah.ayah_number}: "
            f"{result.start_time:.2f}s - {result.end_time:.2f}s "
            f"(score: {result.similarity_score:.2f})"
        )
    
    if len(results) > 5:
        print(f"   ... and {len(results) - 5} more")
    
    # Step 4: Create standardized output using the formatter
    print("\n📄 Step 4: Creating standardized JSON output...")
    
    output = format_alignment_results(
        results=results,
        surah_id=surah_id,
        reciter=reciter,
    )
    
    print(f"   ✅ Formatted {output.total_ayahs} ayahs")
    print(f"   📈 Average similarity: {output.avg_similarity:.3f}")
    print(f"   🎯 High confidence: {output.high_confidence_count}/{output.total_ayahs}")
    
    return output


def save_to_json(output, output_path: str):
    """Save output to JSON file."""
    print(f"\n💾 Saving to: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.to_json(indent=2))
    
    print("   ✅ Saved successfully!")


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
    
    # Save to JSON
    output_path = f"corrected_segments_{surah_id:03d}.json"
    save_to_json(output, output_path)
    
    print("\n🎉 Done!")
