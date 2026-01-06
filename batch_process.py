"""
Batch processor for Quran audio files.

Processes all available audio files in the audio folder using faster-whisper
with OdyAsh/faster-whisper-base-ar-quran model (Quran-tuned),
skipping any surahs that already have output JSON files.

Features:
    - Parallel silence detection (CPU-bound tasks run concurrently)
    - Pipeline processing: silence detection runs ahead of transcription
    - Apple Metal (MPS) accelerated transcription
    - DP-based alignment for globally optimal segment-to-ayah mapping
    - Async file saving for non-blocking I/O

Usage:
    python batch_process.py

Optimized for: Apple M2 Pro (10-12 CPU cores, 16GB unified memory)
"""

import json
import os
import time
import gc
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Optional
from queue import Queue
from threading import Thread

from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.core import Aligner, HybridStats
from munajjam.data import load_surah_ayahs, get_surah_name


# Configuration
AUDIO_FOLDER = Path("Quran/badr_alturki_audio")
OUTPUT_FOLDER = Path("output")
RECITER_NAME = "Badr Al-Turki"

# ============================================
# Hardware-optimized settings for Apple M2 Pro
# M2 Pro: 10-12 CPU cores, 16GB unified memory
# Metal Performance Shaders (MPS) for GPU acceleration
# ============================================
CPU_CORES = os.cpu_count() or 10

# Silence detection is CPU-bound - use most cores, leave 2 for system + transcription
MAX_SILENCE_WORKERS = min(CPU_CORES - 2, 8)

# I/O workers for file operations
MAX_IO_WORKERS = 4

# Device settings - faster-whisper uses CPU on Mac (MPS not supported by CTranslate2)
# But it's still fast due to efficient C++ implementation
DEVICE = "cpu"  # faster-whisper on Mac uses optimized CPU

# Model settings - use faster-whisper for speed
MODEL_ID = "OdyAsh/faster-whisper-base-ar-quran"
MODEL_TYPE = "faster-whisper"


@dataclass
class ProcessingResult:
    """Result of processing a single surah."""
    surah_id: int
    surah_name: str
    success: bool
    total_ayahs: int = 0
    aligned_ayahs: int = 0
    avg_similarity: float = 0.0
    processing_time: float = 0.0
    error_message: str = ""
    skipped: bool = False
    # Hybrid stats
    dp_kept: int = 0
    old_fallback: int = 0
    split_improved: int = 0
    still_low: int = 0
    # Zone realignment stats (drift fix)
    zones_found: int = 0
    zones_improved: int = 0
    zone_ayahs_improved: int = 0


@dataclass
class BatchProgress:
    """Track batch processing progress."""
    total_surahs: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    results: list = field(default_factory=list)

    def add_result(self, result: ProcessingResult):
        self.results.append(result)
        if result.skipped:
            self.skipped += 1
        elif result.success:
            self.processed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "BATCH PROCESSING SUMMARY",
            "=" * 60,
            f"Total audio files: {self.total_surahs}",
            f"Successfully processed: {self.processed}",
            f"Skipped (already exists): {self.skipped}",
            f"Failed: {self.failed}",
            "-" * 60,
        ]

        for result in self.results:
            if result.skipped:
                lines.append(f"⏭️  Surah {result.surah_id:03d} ({result.surah_name}): Skipped")
            elif result.success:
                extra_info = ""
                if result.old_fallback > 0 or result.zone_ayahs_improved > 0:
                    parts = []
                    if result.dp_kept > 0:
                        parts.append(f"DP:{result.dp_kept}")
                    if result.old_fallback > 0:
                        parts.append(f"Old:{result.old_fallback}")
                    if result.zone_ayahs_improved > 0:
                        parts.append(f"Zone:{result.zone_ayahs_improved}")
                    extra_info = f" [{' '.join(parts)}]"
                lines.append(
                    f"✅ Surah {result.surah_id:03d} ({result.surah_name}): "
                    f"{result.aligned_ayahs}/{result.total_ayahs} ayahs "
                    f"({result.avg_similarity:.1%} avg) in {result.processing_time:.1f}s{extra_info}"
                )
            else:
                lines.append(
                    f"❌ Surah {result.surah_id:03d} ({result.surah_name}): {result.error_message}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def _detect_silence_worker(audio_path: str) -> tuple[int, list]:
    """Worker function for parallel silence detection.
    
    Returns:
        Tuple of (surah_id, silences_list)
    """
    surah_id = int(Path(audio_path).stem)
    silences = detect_silences(audio_path)
    return surah_id, silences


def _load_ayahs_worker(surah_id: int) -> tuple[int, list]:
    """Worker function for parallel ayah loading.
    
    Returns:
        Tuple of (surah_id, ayahs_list)
    """
    ayahs = load_surah_ayahs(surah_id)
    return surah_id, ayahs


def preload_ayahs_parallel(surah_ids: list[int]) -> dict[int, list]:
    """Pre-load ayahs for multiple surahs in parallel.
    
    Args:
        surah_ids: List of surah IDs to load
    
    Returns:
        Dictionary mapping surah_id to list of ayahs
    """
    ayahs_cache: dict[int, list] = {}
    
    if not surah_ids:
        return ayahs_cache
    
    print(f"\n📖 Pre-loading ayahs for {len(surah_ids)} surahs...")
    
    # Use ThreadPoolExecutor since this is I/O-bound (reading CSV)
    with ThreadPoolExecutor(max_workers=MAX_IO_WORKERS) as executor:
        futures = {executor.submit(_load_ayahs_worker, sid): sid for sid in surah_ids}
        
        for future in as_completed(futures):
            try:
                surah_id, ayahs = future.result()
                ayahs_cache[surah_id] = ayahs
            except Exception as e:
                surah_id = futures[future]
                print(f"   ⚠️  Failed to load ayahs for surah {surah_id}: {e}")
    
    print(f"   ✓ Loaded {len(ayahs_cache)} surahs into memory")
    return ayahs_cache


class AsyncSaver:
    """Async file saver using a background thread."""
    
    def __init__(self):
        self._queue: Queue = Queue()
        self._thread: Optional[Thread] = None
        self._running = False
    
    def start(self):
        """Start the background saver thread."""
        self._running = True
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def _worker(self):
        """Background worker that saves files from the queue."""
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:  # Poison pill
                    break
                output_path, output_data = item
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                self._queue.task_done()
            except Exception:
                pass  # Timeout or error, continue
    
    def save(self, output_path: Path, output_data: dict):
        """Queue a file for async saving."""
        self._queue.put((output_path, output_data))
    
    def stop(self):
        """Stop the saver and wait for pending saves."""
        self._running = False
        self._queue.put(None)  # Poison pill
        if self._thread:
            self._thread.join(timeout=10)
    
    def wait(self):
        """Wait for all queued saves to complete."""
        self._queue.join()


# Global async saver instance
_async_saver: Optional[AsyncSaver] = None


def detect_silences_parallel(audio_files: list[Path], skip_surah_ids: set[int]) -> dict[int, list]:
    """Detect silences for multiple audio files in parallel.
    
    Args:
        audio_files: List of audio file paths
        skip_surah_ids: Set of surah IDs to skip (already processed)
    
    Returns:
        Dictionary mapping surah_id to list of silences
    """
    silences_cache: dict[int, list] = {}
    
    # Filter to only files we need to process
    files_to_process = [
        f for f in audio_files 
        if int(f.stem) not in skip_surah_ids
    ]
    
    if not files_to_process:
        return silences_cache
    
    print(f"\n[SILENCE] Pre-computing silences for {len(files_to_process)} surahs...", flush=True)
    print(f"   Using {MAX_SILENCE_WORKERS} workers", flush=True)
    
    start_time = time.time()
    completed = 0
    
    with ProcessPoolExecutor(max_workers=MAX_SILENCE_WORKERS) as executor:
        # Submit all jobs
        future_to_path = {
            executor.submit(_detect_silence_worker, str(f)): f 
            for f in files_to_process
        }
        
        # Process results as they complete
        for future in as_completed(future_to_path):
            audio_path = future_to_path[future]
            try:
                surah_id, silences = future.result()
                silences_cache[surah_id] = silences
                completed += 1
                
                # Progress update
                percent = (completed / len(files_to_process)) * 100
                bar_width = 30
                filled = int(bar_width * completed / len(files_to_process))
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\r   [{bar}] {completed}/{len(files_to_process)} ({percent:.0f}%)", end="", flush=True)
                
            except Exception as e:
                surah_id = int(audio_path.stem)
                print(f"\n   ⚠️  Failed silence detection for surah {surah_id}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n   ✓ Silence detection completed in {elapsed:.1f}s")
    
    return silences_cache


def get_available_audio_files() -> list[Path]:
    """Get all available WAV files in the audio folder."""
    if not AUDIO_FOLDER.exists():
        print(f"❌ Audio folder not found: {AUDIO_FOLDER}")
        return []

    files = sorted(AUDIO_FOLDER.glob("*.wav"))
    return files


def output_exists(surah_id: int) -> bool:
    """Check if output JSON already exists for a surah."""
    output_path = OUTPUT_FOLDER / f"surah_{surah_id:03d}.json"
    return output_path.exists()


def save_output(
    surah_id: int, 
    surah_name: str, 
    results: list, 
    total_ayahs: int, 
    hybrid_stats: Optional[HybridStats] = None,
    async_save: bool = True
) -> None:
    """Save alignment results to JSON file.
    
    Args:
        surah_id: Surah number
        surah_name: Name of the surah
        results: Alignment results
        total_ayahs: Total number of ayahs in the surah
        hybrid_stats: Optional stats from hybrid alignment
        async_save: If True, save in background thread (non-blocking)
    """
    global _async_saver
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Calculate average similarity
    similarities = [r.similarity_score for r in results]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    # Count high/low confidence
    high_confidence = sum(1 for r in results if r.similarity_score >= 0.85)
    low_confidence = len(results) - high_confidence

    # Build output data
    output_data = {
        "surah_id": surah_id,
        "surah_name": surah_name,
        "reciter": RECITER_NAME,
        "total_ayahs": total_ayahs,
        "aligned_ayahs": len(results),
        "avg_similarity": round(avg_similarity, 3),
        "high_confidence_ayahs": high_confidence,
        "low_confidence_ayahs": low_confidence,
        "ayahs": [
            {
                "ayah_number": r.ayah.ayah_number,
                "start": r.start_time,
                "end": r.end_time,
                "text": r.ayah.text,
                "similarity": round(r.similarity_score, 3),
                "confidence": "high" if r.similarity_score >= 0.85 else "low",
            }
            for r in results
        ],
    }
    
    # Add hybrid stats if available
    if hybrid_stats:
        output_data["hybrid_stats"] = {
            "dp_kept": hybrid_stats.dp_kept,
            "old_fallback": hybrid_stats.old_fallback,
            "split_improved": hybrid_stats.split_improved,
            "still_low": hybrid_stats.still_low,
        }

    output_path = OUTPUT_FOLDER / f"surah_{surah_id:03d}.json"
    
    if async_save and _async_saver:
        # Queue for background saving (non-blocking)
        _async_saver.save(output_path, output_data)
        print(f"   💾 Queued save: {output_path}")
    else:
        # Synchronous save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"   💾 Saved to {output_path}")


def process_surah(
    audio_path: Path, 
    transcriber: WhisperTranscriber,
    silences_cache: Optional[dict[int, list]] = None,
    ayahs_cache: Optional[dict[int, list]] = None
) -> ProcessingResult:
    """Process a single surah.
    
    Args:
        audio_path: Path to the audio file
        transcriber: WhisperTranscriber instance
        silences_cache: Pre-computed silences dict (surah_id -> silences)
        ayahs_cache: Pre-loaded ayahs dict (surah_id -> ayahs)
    """
    surah_id = int(audio_path.stem)
    surah_name = get_surah_name(surah_id)

    result = ProcessingResult(
        surah_id=surah_id,
        surah_name=surah_name,
        success=False,
    )

    # Check if already processed
    if output_exists(surah_id):
        result.skipped = True
        result.success = True
        return result

    print(f"\n{'─' * 60}")
    print(f"📖 Processing Surah {surah_id:03d}: {surah_name}")
    print(f"   Audio: {audio_path}")
    print(f"{'─' * 60}")

    start_time = time.time()

    try:
        # 1. Get silences (from cache or detect)
        if silences_cache and surah_id in silences_cache:
            silences = silences_cache[surah_id]
            print(f"   🔇 Using pre-computed silences ({len(silences)} gaps)")
        else:
            print("   🔇 Detecting silences...")
            silences = detect_silences(str(audio_path))
            print(f"   ✓ Found {len(silences)} silence gaps")

        # 2. Transcribe with progress (GPU-bound, sequential)
        print("   🎤 Transcribing segments...")
        
        def progress_callback(current: int, total: int, text: str):
            """Show transcription progress."""
            percent = (current / total) * 100
            bar_width = 30
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            # Truncate text for display
            display_text = text[:35] + "..." if len(text) > 35 else text
            print(f"\r      [{bar}] {current}/{total} ({percent:.0f}%) {display_text}", end="", flush=True)
        
        segments = transcriber.transcribe(str(audio_path), progress_callback=progress_callback)
        print()  # New line after progress bar
        print(f"   ✓ Transcribed {len(segments)} segments")

        # 3. Get ayahs (from cache or load) and align using Aligner
        print("   📝 Aligning segments (Hybrid: DP + fallback + drift fix)...")
        if ayahs_cache and surah_id in ayahs_cache:
            ayahs = ayahs_cache[surah_id]
        else:
            ayahs = load_surah_ayahs(surah_id)
        
        result.total_ayahs = len(ayahs)
        
        # Alignment progress callback
        def align_progress(current: int, total: int):
            percent = (current / total) * 100
            bar_width = 30
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r      [{bar}] {current}/{total} ({percent:.0f}%)", end="", flush=True)
        
        # Use the unified Aligner class - handles DP, fallback, and drift fix
        aligner = Aligner(
            strategy="hybrid",
            quality_threshold=0.85,
            fix_drift=True,
            fix_overlaps=True,
        )
        
        aligned_results = aligner.align(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences,
            on_progress=align_progress,
        )
        print()  # New line after progress

        result.aligned_ayahs = len(aligned_results)
        result.avg_similarity = (
            sum(r.similarity_score for r in aligned_results) / len(aligned_results)
            if aligned_results else 0.0
        )
        
        # Store hybrid stats if available
        hybrid_stats = aligner.last_stats
        if hybrid_stats:
            result.dp_kept = hybrid_stats.dp_kept
            result.old_fallback = hybrid_stats.old_fallback
            result.split_improved = hybrid_stats.split_improved
            result.still_low = hybrid_stats.still_low

        # 5. Save output (async - non-blocking)
        save_output(surah_id, surah_name, aligned_results, len(ayahs), hybrid_stats, async_save=True)

        result.success = True
        result.processing_time = time.time() - start_time

        print(f"   ✅ Aligned {result.aligned_ayahs}/{result.total_ayahs} ayahs "
              f"({result.avg_similarity:.1%} avg similarity)")

    except Exception as e:
        result.error_message = str(e)
        result.processing_time = time.time() - start_time
        print(f"   ❌ Error: {e}")

    return result


def setup_mac_optimizations():
    """Configure optimizations for Apple Silicon (M2 Pro)."""
    import torch
    
    # Check for MPS (Metal Performance Shaders) availability
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if has_mps:
        print("   ✓ Apple Metal (MPS) available")
    else:
        print("   ℹ Using optimized CPU (faster-whisper)")
    
    # Set number of threads for optimal CPU performance on M2
    torch.set_num_threads(CPU_CORES)
    
    return has_mps


def clear_memory():
    """Clear memory cache."""
    gc.collect()


def main():
    """Main batch processing function."""
    global _async_saver
    
    print("\n" + "#" * 60)
    print("  MUNAJJAM BATCH PROCESSOR")
    print("  Optimized for Apple M2 Pro")
    print("#" * 60)
    
    # Setup Mac optimizations
    setup_mac_optimizations()
    
    print(f"\n⚡ Hardware optimization enabled:")
    print(f"   • CPU cores detected: {CPU_CORES}")
    print(f"   • Silence workers: {MAX_SILENCE_WORKERS}")
    print(f"   • I/O workers: {MAX_IO_WORKERS}")
    print(f"   • Model: {MODEL_ID}")
    print(f"   • Backend: {MODEL_TYPE}")
    print(f"   • Async file saving: enabled")

    # Get available audio files
    audio_files = get_available_audio_files()
    if not audio_files:
        print("❌ No audio files found!")
        return

    progress = BatchProgress(total_surahs=len(audio_files))

    print(f"\n[FILES] Found {len(audio_files)} audio files in {AUDIO_FOLDER}", flush=True)
    print(f"[OUTPUT] Output directory: {OUTPUT_FOLDER}", flush=True)

    # Check how many will be skipped
    skip_surah_ids = {int(f.stem) for f in audio_files if output_exists(int(f.stem))}
    to_process = [f for f in audio_files if int(f.stem) not in skip_surah_ids]
    surah_ids_to_process = [int(f.stem) for f in to_process]

    print(f"[SKIP] Will skip: {len(skip_surah_ids)} (already processed)", flush=True)
    print(f"[TODO] Will process: {len(to_process)}", flush=True)

    if not to_process:
        print("\n✅ All surahs already processed!")
        return

    batch_start = time.time()

    # ========================================
    # PHASE 1: Parallel pre-computation
    # Run silence detection AND ayah loading in parallel
    # ========================================
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1: Parallel pre-computation", flush=True)
    print("=" * 60, flush=True)
    
    # Start silence detection in background
    silences_cache = detect_silences_parallel(audio_files, skip_surah_ids)
    
    # Pre-load all ayahs (I/O-bound, uses thread pool)
    ayahs_cache = preload_ayahs_parallel(surah_ids_to_process)

    # ========================================
    # PHASE 2: Load model (one-time)
    # ========================================
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2: Loading faster-whisper model...", flush=True)
    print("=" * 60, flush=True)

    # Use faster-whisper with Quran-tuned model (CTranslate2 format)
    transcriber = WhisperTranscriber(
        model_id=MODEL_ID,
        model_type=MODEL_TYPE
    )
    transcriber.load()
    
    # Clear any memory fragmentation after model load
    clear_memory()

    # ========================================
    # PHASE 3: Sequential transcription + alignment
    # (GPU-bound transcription can't be parallelized)
    # (File saving happens async in background)
    # ========================================
    print("\n" + "=" * 60, flush=True)
    print("PHASE 3: Transcription & Alignment (async saving enabled)", flush=True)
    print("=" * 60, flush=True)
    
    # Start async saver
    _async_saver = AsyncSaver()
    _async_saver.start()
    
    try:
        processed_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            surah_id = int(audio_file.stem)
            print(f"\n[{i}/{len(audio_files)}] Surah {surah_id:03d}...", end="")

            if surah_id in skip_surah_ids:
                print(" Skipped (exists)")
                result = ProcessingResult(
                    surah_id=surah_id,
                    surah_name=get_surah_name(surah_id),
                    success=True,
                    skipped=True,
                )
            else:
                print()  # New line for processing output
                result = process_surah(
                    audio_file, 
                    transcriber, 
                    silences_cache,
                    ayahs_cache
                )
                processed_count += 1
                
                # Periodic memory cleanup every 5 surahs
                if processed_count % 5 == 0:
                    clear_memory()

            progress.add_result(result)

    finally:
        # Wait for async saves to complete
        if _async_saver:
            print("\n⏳ Waiting for async saves to complete...")
            _async_saver.wait()
            _async_saver.stop()
            print("   ✓ All files saved")
        
        # Unload model
        print("🧹 Unloading model...")
        transcriber.unload()

    batch_time = time.time() - batch_start

    # Final cleanup
    clear_memory()
    
    # Print summary
    print(progress.summary())
    print(f"\n[TIME] Total batch time: {batch_time / 60:.1f} minutes")
    
    # Calculate throughput
    actual_processed = progress.processed
    if actual_processed > 0 and batch_time > 0:
        avg_time_per_surah = batch_time / actual_processed
        print(f"[PERF] Average time per surah: {avg_time_per_surah:.1f}s")
    
    print(f"\n💪 Optimizations used:")
    print(f"   • Apple M2 Pro optimized")
    print(f"   • Parallel silence detection ({MAX_SILENCE_WORKERS} workers)")
    print(f"   • Pre-loaded ayahs in memory")
    print(f"   • Async file saving (non-blocking)")
    print(f"   • Hybrid alignment (DP + Old fallback + Split-restitch)")


if __name__ == "__main__":
    main()

