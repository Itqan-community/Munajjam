# Munajjam Benchmark Suite

This directory contains the benchmark harness for evaluating Munajjam alignment accuracy.

## Structure

```
benchmarks/
├── benchmark.py          # Main benchmark harness
├── sample_ground_truth.json  # Example ground truth format
├── sample_config.json    # Example benchmark configuration
└── README.md            # This file
```

## Ground Truth Format

The ground truth file should be a JSON file with the following structure:

```json
{
  "files": [
    {
      "filename": "001.mp3",
      "surah_id": 1,
      "reciter": "Badr Al-Turki",
      "ayahs": [
        {
          "ayah_number": 1,
          "start": 5.72,
          "end": 9.74,
          "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        },
        ...
      ]
    }
  ]
}
```

## Usage

### Basic Usage

```bash
# Run with explicit paths
python benchmarks/benchmark.py \
    --audio-dir ./test_audio \
    --ground-truth ./ground_truth.json \
    --strategy hybrid

# Run with config file
python benchmarks/benchmark.py --config benchmarks/sample_config.json
```

### Compare All Strategies

```bash
# Run all alignment strategies and generate comparison leaderboard
python benchmarks/benchmark.py \
    --audio-dir ./test_audio \
    --ground-truth ./ground_truth.json \
    --compare-all \
    --name strategy_comparison
```

### Available Options

- `--config`: Path to JSON configuration file
- `--audio-dir`: Directory containing test audio files (MP3/WAV)
- `--ground-truth`: Path to ground truth JSON file
- `--output-dir`: Directory for output files (default: ./benchmark_output)
- `--strategy`: Alignment strategy: auto, greedy, dp, hybrid (default: auto)
- `--name`: Name for this benchmark configuration
- `--compare-all`: Run all strategies and generate comparison leaderboard

## Output

The benchmark generates:

1. **JSON Results** (`benchmark_output/benchmark_results_<name>.json`):
   - Detailed per-ayah metrics
   - Per-file aggregate metrics
   - Overall benchmark statistics

2. **Markdown Leaderboard** (`benchmark_output/leaderboard.md`):
   - Comparison table of different configurations
   - Detailed per-file breakdown
   - Rankings by accuracy

## Metrics

The benchmark calculates the following accuracy metrics:

| Metric | Description |
|--------|-------------|
| Mean Start Error | Average absolute difference between predicted and ground truth start times |
| Mean End Error | Average absolute difference between predicted and ground truth end times |
| RMSE Start | Root mean square error for start times |
| RMSE End | Root mean square error for end times |
| Max Start Error | Maximum start time error across all ayahs |
| Max End Error | Maximum end time error across all ayahs |
| Mean Similarity | Average text similarity score from alignment |
| Duration Error | Difference in predicted vs actual ayah duration |

## Creating Ground Truth Data

To create ground truth timestamps for your test audio files:

1. Use a reliable annotation tool (e.g., Praat, Audacity)
2. Mark the exact start and end times for each ayah
3. Export the timestamps in the JSON format shown above
4. Verify the annotations with multiple reviewers for accuracy

## Configuration File Format

```json
{
  "name": "my_benchmark",
  "audio_dir": "./test_audio",
  "ground_truth_file": "./ground_truth.json",
  "output_dir": "./benchmark_output",
  "strategy": "hybrid",
  "fix_drift": true,
  "fix_overlaps": true,
  "quality_threshold": 0.85
}
```

## Example Workflow

```bash
# 1. Prepare your test audio files
mkdir test_audio
cp /path/to/surah_001.mp3 test_audio/
cp /path/to/surah_002.mp3 test_audio/

# 2. Create ground truth file (see sample_ground_truth.json)

# 3. Run benchmark
python benchmarks/benchmark.py \
    --audio-dir ./test_audio \
    --ground-truth ./ground_truth.json \
    --strategy hybrid \
    --name my_test

# 4. View results
cat benchmark_output/leaderboard.md
```
