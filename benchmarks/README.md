# Munajjam Benchmarks

This directory contains the benchmark harness for measuring Munajjam performance.

## Files

- `benchmark.py` - Main benchmark harness
- `benchmark_results.json` - JSON output of benchmark results
- `LEADERBOARD.md` - Markdown leaderboard with performance rankings

## Usage

### Run All Benchmarks

```bash
python benchmarks/benchmark.py
```

### Benchmark Specific Strategy

```bash
python benchmarks/benchmark.py --strategy hybrid
```

### Include Ground Truth Comparison

```bash
python benchmarks/benchmark.py --compare-ground-truth
```

### Generate Only JSON Output

```bash
python benchmarks/benchmark.py --output json
```

### Increase Iterations for Statistical Significance

```bash
python benchmarks/benchmark.py --iterations 50
```

## Benchmarks

The harness runs the following benchmarks:

1. **Surah Al-Fatiha (7 ayahs)** - Short surah with basmala
2. **Surah Al-Ikhlas (4 ayahs)** - Very short surah
3. **Surah Al-Falaq (5 ayahs)** - Medium short surah
4. **Surah An-Nas (6 ayahs)** - Medium short surah

Each benchmark tests all three alignment strategies:
- `greedy` - Fast greedy matching
- `dp` - Dynamic programming optimal alignment
- `hybrid` - DP with fallback to greedy (recommended)

## Metrics

- **Avg/ayah** - Average processing time per ayah (milliseconds)
- **Total** - Total alignment time (milliseconds)
- **P95/P99** - 95th and 99th percentile latencies
- **Success Rate** - Percentage of successful alignments
- **Similarity** - Text similarity between predicted and reference text

## Ground Truth Comparison

When `--compare-ground-truth` is enabled, the harness compares predicted timings
against synthetic ground truth to measure timing accuracy:

- **Start Error** - Difference between predicted and ground truth start time
- **End Error** - Difference between predicted and ground truth end time
- **Duration Error** - Difference in segment duration
