#!/usr/bin/env python3
"""
Munajjam Benchmark Harness

A benchmarking tool to evaluate alignment accuracy against ground truth timestamps.

Usage:
    python benchmark.py --config benchmark_config.json
    python benchmark.py --audio-dir ./test_audio --ground-truth ./ground_truth.json

Output:
    - benchmark_results.json: Detailed results for each test file
    - leaderboard.md: Markdown leaderboard comparing different configurations
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# Add parent directory to path to import munajjam
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from munajjam.transcription import WhisperTranscriber
from munajjam.core import Aligner, AlignmentStrategy
from munajjam.data import load_surah_ayahs


@dataclass
class GroundTruthAyah:
    """Ground truth timing for a single ayah."""
    ayah_number: int
    start_time: float
    end_time: float
    text: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    name: str
    audio_dir: Path
    ground_truth_file: Path
    output_dir: Path
    strategy: str = "auto"
    fix_drift: bool = True
    fix_overlaps: bool = True
    quality_threshold: float = 0.85


@dataclass
class AyahMetrics:
    """Metrics for a single ayah alignment."""
    ayah_number: int
    ground_truth_start: float
    ground_truth_end: float
    predicted_start: float
    predicted_end: float
    start_error: float
    end_error: float
    duration_error: float
    similarity_score: float


@dataclass
class FileResult:
    """Results for a single audio file."""
    filename: str
    surah_id: int
    reciter: Optional[str]
    duration_seconds: float
    ayah_count: int
    ayah_metrics: list[AyahMetrics] = field(default_factory=list)
    
    # Aggregate metrics
    mean_start_error: float = 0.0
    mean_end_error: float = 0.0
    mean_duration_error: float = 0.0
    rmse_start: float = 0.0
    rmse_end: float = 0.0
    max_start_error: float = 0.0
    max_end_error: float = 0.0
    mean_similarity: float = 0.0
    alignment_time_seconds: float = 0.0


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    config_name: str
    timestamp: str
    strategy: str
    total_files: int
    file_results: list[FileResult] = field(default_factory=list)
    
    # Overall metrics
    overall_mean_start_error: float = 0.0
    overall_mean_end_error: float = 0.0
    overall_rmse_start: float = 0.0
    overall_rmse_end: float = 0.0
    overall_max_start_error: float = 0.0
    overall_max_end_error: float = 0.0
    overall_mean_similarity: float = 0.0
    total_ayahs_aligned: int = 0
    total_alignment_time: float = 0.0


def load_ground_truth(ground_truth_file: Path) -> dict:
    """
    Load ground truth timestamps from JSON file.
    
    Expected format:
    {
        "files": [
            {
                "filename": "001.mp3",
                "surah_id": 1,
                "reciter": "Badr Al-Turki",
                "ayahs": [
                    {"ayah_number": 1, "start": 5.72, "end": 9.74, "text": "..."},
                    ...
                ]
            }
        ]
    }
    """
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ground_truth = {}
    for file_entry in data.get('files', []):
        filename = file_entry['filename']
        ground_truth[filename] = {
            'surah_id': file_entry['surah_id'],
            'reciter': file_entry.get('reciter'),
            'ayahs': [
                GroundTruthAyah(
                    ayah_number=ayah['ayah_number'],
                    start_time=ayah['start'],
                    end_time=ayah['end'],
                    text=ayah.get('text')
                )
                for ayah in file_entry['ayahs']
            ]
        }
    
    return ground_truth


def calculate_ayah_metrics(
    predicted: list,
    ground_truth: list[GroundTruthAyah]
) -> list[AyahMetrics]:
    """Calculate metrics comparing predicted to ground truth timings."""
    metrics = []
    
    # Create lookup by ayah number
    gt_lookup = {gt.ayah_number: gt for gt in ground_truth}
    pred_lookup = {p.ayah.ayah_number: p for p in predicted}
    
    for ayah_num in sorted(gt_lookup.keys()):
        gt = gt_lookup[ayah_num]
        pred = pred_lookup.get(ayah_num)
        
        if pred is None:
            continue
        
        start_error = abs(pred.start_time - gt.start_time)
        end_error = abs(pred.end_time - gt.end_time)
        duration_error = abs((pred.end_time - pred.start_time) - (gt.end_time - gt.start_time))
        
        metrics.append(AyahMetrics(
            ayah_number=ayah_num,
            ground_truth_start=gt.start_time,
            ground_truth_end=gt.end_time,
            predicted_start=pred.start_time,
            predicted_end=pred.end_time,
            start_error=start_error,
            end_error=end_error,
            duration_error=duration_error,
            similarity_score=pred.similarity_score
        ))
    
    return metrics


def calculate_file_metrics(
    ayah_metrics: list[AyahMetrics],
    duration: float,
    alignment_time: float
) -> dict:
    """Calculate aggregate metrics for a file."""
    if not ayah_metrics:
        return {}
    
    start_errors = [m.start_error for m in ayah_metrics]
    end_errors = [m.end_error for m in ayah_metrics]
    duration_errors = [m.duration_error for m in ayah_metrics]
    similarities = [m.similarity_score for m in ayah_metrics]
    
    return {
        'mean_start_error': statistics.mean(start_errors),
        'mean_end_error': statistics.mean(end_errors),
        'mean_duration_error': statistics.mean(duration_errors),
        'rmse_start': statistics.mean([e ** 2 for e in start_errors]) ** 0.5,
        'rmse_end': statistics.mean([e ** 2 for e in end_errors]) ** 0.5,
        'max_start_error': max(start_errors),
        'max_end_error': max(end_errors),
        'mean_similarity': statistics.mean(similarities),
        'duration_seconds': duration,
        'alignment_time_seconds': alignment_time
    }


def process_single_file(
    audio_path: Path,
    ground_truth: dict,
    config: BenchmarkConfig
) -> Optional[FileResult]:
    """Process a single audio file and return results."""
    filename = audio_path.name
    
    if filename not in ground_truth:
        print(f"  ⚠️  No ground truth for {filename}, skipping")
        return None
    
    gt_data = ground_truth[filename]
    surah_id = gt_data['surah_id']
    reciter = gt_data['reciter']
    gt_ayahs = gt_data['ayahs']
    
    print(f"  🎵 Processing {filename} (Surah {surah_id})...")
    
    try:
        # Get audio duration
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
    except Exception:
        duration = 0.0
    
    # Transcribe audio
    print(f"     Transcribing...")
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(str(audio_path))
    
    # Load reference ayahs
    ayahs = load_surah_ayahs(surah_id)
    
    # Align
    print(f"     Aligning with strategy '{config.strategy}'...")
    aligner = Aligner(
        audio_path=str(audio_path),
        strategy=config.strategy,
        quality_threshold=config.quality_threshold,
        fix_drift=config.fix_drift,
        fix_overlaps=config.fix_overlaps
    )
    
    start_time = time.time()
    results = aligner.align(segments, ayahs)
    alignment_time = time.time() - start_time
    
    # Calculate metrics
    ayah_metrics = calculate_ayah_metrics(results, gt_ayahs)
    aggregate = calculate_file_metrics(ayah_metrics, duration, alignment_time)
    
    file_result = FileResult(
        filename=filename,
        surah_id=surah_id,
        reciter=reciter,
        duration_seconds=duration,
        ayah_count=len(ayahs),
        ayah_metrics=ayah_metrics,
        **aggregate
    )
    
    print(f"     ✓ Mean start error: {file_result.mean_start_error:.3f}s")
    print(f"     ✓ Mean end error: {file_result.mean_end_error:.3f}s")
    print(f"     ✓ Mean similarity: {file_result.mean_similarity:.3f}")
    
    return file_result


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Run the full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"Munajjam Benchmark: {config.name}")
    print(f"{'='*60}")
    print(f"Audio directory: {config.audio_dir}")
    print(f"Ground truth file: {config.ground_truth_file}")
    print(f"Strategy: {config.strategy}")
    print(f"Fix drift: {config.fix_drift}")
    print(f"Fix overlaps: {config.fix_overlaps}")
    print(f"{'='*60}\n")
    
    # Load ground truth
    print("📂 Loading ground truth data...")
    ground_truth = load_ground_truth(config.ground_truth_file)
    print(f"   Loaded {len(ground_truth)} files from ground truth\n")
    
    # Find audio files
    audio_files = list(config.audio_dir.glob("*.mp3")) + list(config.audio_dir.glob("*.wav"))
    audio_files = [f for f in audio_files if f.name in ground_truth]
    
    if not audio_files:
        print("❌ No matching audio files found!")
        return BenchmarkResults(
            config_name=config.name,
            timestamp=datetime.now().isoformat(),
            strategy=config.strategy,
            total_files=0
        )
    
    print(f"🎯 Found {len(audio_files)} audio files to benchmark\n")
    
    # Process each file
    file_results = []
    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}]")
        result = process_single_file(audio_path, ground_truth, config)
        if result:
            file_results.append(result)
        print()
    
    # Calculate overall metrics
    if file_results:
        all_start_errors = []
        all_end_errors = []
        all_similarities = []
        total_alignment_time = 0.0
        total_ayahs = 0
        
        for fr in file_results:
            all_start_errors.extend([m.start_error for m in fr.ayah_metrics])
            all_end_errors.extend([m.end_error for m in fr.ayah_metrics])
            all_similarities.extend([m.similarity_score for m in fr.ayah_metrics])
            total_alignment_time += fr.alignment_time_seconds
            total_ayahs += len(fr.ayah_metrics)
        
        overall = BenchmarkResults(
            config_name=config.name,
            timestamp=datetime.now().isoformat(),
            strategy=config.strategy,
            total_files=len(file_results),
            file_results=file_results,
            overall_mean_start_error=statistics.mean(all_start_errors),
            overall_mean_end_error=statistics.mean(all_end_errors),
            overall_rmse_start=statistics.mean([e ** 2 for e in all_start_errors]) ** 0.5,
            overall_rmse_end=statistics.mean([e ** 2 for e in all_end_errors]) ** 0.5,
            overall_max_start_error=max(all_start_errors),
            overall_max_end_error=max(all_end_errors),
            overall_mean_similarity=statistics.mean(all_similarities),
            total_ayahs_aligned=total_ayahs,
            total_alignment_time=total_alignment_time
        )
    else:
        overall = BenchmarkResults(
            config_name=config.name,
            timestamp=datetime.now().isoformat(),
            strategy=config.strategy,
            total_files=0
        )
    
    print(f"{'='*60}")
    print("📊 BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {overall.total_files}")
    print(f"Total ayahs aligned: {overall.total_ayahs_aligned}")
    print(f"\nOverall Metrics:")
    print(f"  Mean start error: {overall.overall_mean_start_error:.3f}s")
    print(f"  Mean end error: {overall.overall_mean_end_error:.3f}s")
    print(f"  RMSE start: {overall.overall_rmse_start:.3f}s")
    print(f"  RMSE end: {overall.overall_rmse_end:.3f}s")
    print(f"  Max start error: {overall.overall_max_start_error:.3f}s")
    print(f"  Max end error: {overall.overall_max_end_error:.3f}s")
    print(f"  Mean similarity: {overall.overall_mean_similarity:.3f}")
    print(f"  Total alignment time: {overall.total_alignment_time:.2f}s")
    print(f"{'='*60}\n")
    
    return overall


def save_results(results: BenchmarkResults, output_dir: Path):
    """Save benchmark results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict for JSON serialization
    results_dict = {
        'config_name': results.config_name,
        'timestamp': results.timestamp,
        'strategy': results.strategy,
        'total_files': results.total_files,
        'overall_mean_start_error': results.overall_mean_start_error,
        'overall_mean_end_error': results.overall_mean_end_error,
        'overall_rmse_start': results.overall_rmse_start,
        'overall_rmse_end': results.overall_rmse_end,
        'overall_max_start_error': results.overall_max_start_error,
        'overall_max_end_error': results.overall_max_end_error,
        'overall_mean_similarity': results.overall_mean_similarity,
        'total_ayahs_aligned': results.total_ayahs_aligned,
        'total_alignment_time': results.total_alignment_time,
        'file_results': [
            {
                'filename': fr.filename,
                'surah_id': fr.surah_id,
                'reciter': fr.reciter,
                'duration_seconds': fr.duration_seconds,
                'ayah_count': fr.ayah_count,
                'mean_start_error': fr.mean_start_error,
                'mean_end_error': fr.mean_end_error,
                'mean_duration_error': fr.mean_duration_error,
                'rmse_start': fr.rmse_start,
                'rmse_end': fr.rmse_end,
                'max_start_error': fr.max_start_error,
                'max_end_error': fr.max_end_error,
                'mean_similarity': fr.mean_similarity,
                'alignment_time_seconds': fr.alignment_time_seconds,
                'ayah_metrics': [
                    {
                        'ayah_number': m.ayah_number,
                        'ground_truth_start': m.ground_truth_start,
                        'ground_truth_end': m.ground_truth_end,
                        'predicted_start': m.predicted_start,
                        'predicted_end': m.predicted_end,
                        'start_error': m.start_error,
                        'end_error': m.end_error,
                        'duration_error': m.duration_error,
                        'similarity_score': m.similarity_score
                    }
                    for m in fr.ayah_metrics
                ]
            }
            for fr in results.file_results
        ]
    }
    
    results_file = output_dir / f"benchmark_results_{results.config_name}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {results_file}")


def generate_leaderboard(
    results_list: list[BenchmarkResults],
    output_dir: Path
):
    """Generate Markdown leaderboard from benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_file = output_dir / "leaderboard.md"
    
    lines = [
        "# Munajjam Benchmark Leaderboard",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall Results",
        "",
        "| Rank | Configuration | Strategy | Files | Ayahs | Mean Start Error | Mean End Error | RMSE Start | RMSE End | Mean Similarity |",
        "|------|---------------|----------|-------|-------|------------------|----------------|------------|----------|-----------------|"
    ]
    
    # Sort by mean start error (lower is better)
    sorted_results = sorted(
        results_list,
        key=lambda r: r.overall_mean_start_error + r.overall_mean_end_error
    )
    
    for rank, result in enumerate(sorted_results, 1):
        lines.append(
            f"| {rank} | {result.config_name} | {result.strategy} | "
            f"{result.total_files} | {result.total_ayahs_aligned} | "
            f"{result.overall_mean_start_error:.3f}s | {result.overall_mean_end_error:.3f}s | "
            f"{result.overall_rmse_start:.3f}s | {result.overall_rmse_end:.3f}s | "
            f"{result.overall_mean_similarity:.3f} |"
        )
    
    lines.extend([
        "",
        "## Detailed Results",
        ""
    ])
    
    for result in sorted_results:
        lines.extend([
            f"### {result.config_name}",
            "",
            f"- **Strategy**: {result.strategy}",
            f"- **Timestamp**: {result.timestamp}",
            f"- **Files Processed**: {result.total_files}",
            f"- **Total Ayahs Aligned**: {result.total_ayahs_aligned}",
            "",
            "#### Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Start Error | {result.overall_mean_start_error:.3f}s |",
            f"| Mean End Error | {result.overall_mean_end_error:.3f}s |",
            f"| RMSE Start | {result.overall_rmse_start:.3f}s |",
            f"| RMSE End | {result.overall_rmse_end:.3f}s |",
            f"| Max Start Error | {result.overall_max_start_error:.3f}s |",
            f"| Max End Error | {result.overall_max_end_error:.3f}s |",
            f"| Mean Similarity | {result.overall_mean_similarity:.3f} |",
            f"| Total Alignment Time | {result.total_alignment_time:.2f}s |",
            ""
        ])
        
        if result.file_results:
            lines.extend([
                "#### Per-File Results",
                "",
                "| File | Surah | Mean Start Error | Mean End Error | Similarity |",
                "|------|-------|------------------|----------------|------------|"
            ])
            
            for fr in result.file_results:
                lines.append(
                    f"| {fr.filename} | {fr.surah_id} | "
                    f"{fr.mean_start_error:.3f}s | {fr.mean_end_error:.3f}s | "
                    f"{fr.mean_similarity:.3f} |"
                )
            
            lines.append("")
    
    with open(leaderboard_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"📋 Leaderboard saved to: {leaderboard_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Munajjam Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python benchmark.py --config benchmark_config.json
  
  # Run with explicit paths
  python benchmark.py --audio-dir ./test_audio --ground-truth ./ground_truth.json
  
  # Run specific strategy comparison
  python benchmark.py --audio-dir ./test_audio --ground-truth ./ground_truth.json --strategy hybrid
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to benchmark configuration JSON file'
    )
    parser.add_argument(
        '--audio-dir',
        type=Path,
        help='Directory containing test audio files'
    )
    parser.add_argument(
        '--ground-truth',
        type=Path,
        help='Path to ground truth JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./benchmark_output'),
        help='Directory for output files (default: ./benchmark_output)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='auto',
        choices=['auto', 'greedy', 'dp', 'hybrid'],
        help='Alignment strategy to use (default: auto)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='default',
        help='Name for this benchmark configuration'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Run all strategies and generate comparison leaderboard'
    )
    
    args = parser.parse_args()
    
    # Load from config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = BenchmarkConfig(
            name=config_data.get('name', args.name),
            audio_dir=Path(config_data['audio_dir']),
            ground_truth_file=Path(config_data['ground_truth_file']),
            output_dir=Path(config_data.get('output_dir', './benchmark_output')),
            strategy=config_data.get('strategy', args.strategy),
            fix_drift=config_data.get('fix_drift', True),
            fix_overlaps=config_data.get('fix_overlaps', True),
            quality_threshold=config_data.get('quality_threshold', 0.85)
        )
    else:
        if not args.audio_dir or not args.ground_truth:
            parser.error("--audio-dir and --ground-truth are required if --config is not provided")
        
        config = BenchmarkConfig(
            name=args.name,
            audio_dir=args.audio_dir,
            ground_truth_file=args.ground_truth,
            output_dir=args.output_dir,
            strategy=args.strategy
        )
    
    # Run benchmark(s)
    if args.compare_all:
        strategies = ['greedy', 'dp', 'hybrid']
        results_list = []
        
        for strategy in strategies:
            print(f"\n{'#'*60}")
            print(f"# Running benchmark with strategy: {strategy}")
            print(f"{'#'*60}")
            
            config.strategy = strategy
            config.name = f"{args.name}_{strategy}"
            
            results = run_benchmark(config)
            save_results(results, config.output_dir)
            results_list.append(results)
        
        # Generate leaderboard
        generate_leaderboard(results_list, config.output_dir)
    else:
        results = run_benchmark(config)
        save_results(results, config.output_dir)
        generate_leaderboard([results], config.output_dir)


if __name__ == '__main__':
    main()
