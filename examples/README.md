# Munajjam Examples

This directory contains example scripts demonstrating various use cases of the Munajjam library.

## Available Examples

### 1. Basic Usage (`01_basic_usage.py`)

**Difficulty:** Beginner

The simplest way to use Munajjam. Demonstrates:
- Transcribing an audio file
- Loading reference ayahs
- Aligning segments to ayahs
- Accessing and displaying results

**Usage:**
```bash
python 01_basic_usage.py
```

**Prerequisites:**
- An audio file of a Quran recitation (WAV format recommended)

---

### 2. Comparing Strategies (`02_comparing_strategies.py`)

**Difficulty:** Intermediate

Compare the four alignment strategies side-by-side. Demonstrates:
- Greedy strategy (fast, simple)
- DP strategy (optimal, slower)
- Hybrid strategy (balanced)
- Auto strategy (recommended — picks the best approach)
- Performance metrics and trade-offs

**Usage:**
```bash
python 02_comparing_strategies.py
```

**Learn:**
- When to use each strategy
- Trade-offs between speed and accuracy
- How to interpret alignment metrics

---

### 3. Advanced Configuration (`03_advanced_configuration.py`)

**Difficulty:** Advanced

Deep dive into configuration options. Demonstrates:
- Custom configuration settings
- Silence detection and usage
- Progress tracking callbacks
- Energy snap for precise boundary placement
- Detailed result inspection
- Strategy statistics
- Exporting results to JSON

**Usage:**
```bash
python 03_advanced_configuration.py
```

**Learn:**
- How to configure Munajjam for specific needs
- How to use silence detection to improve alignment
- How to inspect and debug alignment results
- How to export results for further processing

---

### 4. Batch Processing (`04_batch_processing.py`)

**Difficulty:** Advanced

Process multiple surahs efficiently. Demonstrates:
- Reusing loaded models for performance
- Processing surahs in a loop
- Aggregating statistics across surahs
- Generating summary reports
- Error handling for missing files

**Usage:**
```bash
python 04_batch_processing.py
```

**Learn:**
- How to build production pipelines
- How to optimize for batch processing
- How to generate reports and analytics
- How to handle errors gracefully

---

## Before Running

### 1. Install Munajjam

```bash
pip install munajjam
```

Or for development:
```bash
pip install -e .
```

### 2. Prepare Audio Files

Make sure you have Quran recitation audio files in WAV format. Recommended settings:
- Sample rate: 16kHz (will be resampled automatically if different)
- Channels: Mono
- Format: WAV or MP3

### 3. Update File Paths

Each example script has placeholder paths that you need to update:

```python
# Change this:
audio_path = "path/to/surah_001.wav"

# To your actual path:
audio_path = "/Users/yourname/audio/surah_001.wav"
```

---

## Common Patterns

### Loading a Model Once

For batch processing, reuse the transcriber:

```python
# Good (efficient) ✅
transcriber = WhisperTranscriber()
transcriber.load()
for audio_file in audio_files:
    segments = transcriber.transcribe(audio_file)
transcriber.unload()

# Bad (inefficient) ❌
for audio_file in audio_files:
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(audio_file)
    # Model is reloaded every iteration!
```

### Choosing a Strategy

```python
# Auto (recommended) - full pipeline by default
aligner = Aligner("001.mp3")

# For speed (simple recordings)
aligner = Aligner("001.mp3", strategy="greedy")

# For maximum control
aligner = Aligner("001.mp3", strategy="hybrid")  # DP with greedy fallback
aligner = Aligner("001.mp3", strategy="dp")      # Full dynamic programming
```

### Inspecting Results

```python
for result in results:
    # Access ayah information
    print(f"Ayah {result.ayah.ayah_number}")
    print(f"Text: {result.ayah.text}")

    # Access timing
    print(f"Start: {result.start_time}s")
    print(f"End: {result.end_time}s")
    print(f"Duration: {result.duration}s")

    # Access quality metrics
    print(f"Similarity: {result.similarity_score:.2%}")
    print(f"High confidence: {result.is_high_confidence}")
    print(f"Overlap detected: {result.overlap_detected}")
```

---

## Troubleshooting

### Model Download Issues

On first run, Munajjam downloads the Whisper model (~150MB). If you get download errors:

```bash
# Set Hugging Face cache directory (optional)
export HF_HOME=/path/to/cache

# Or use offline mode if model is cached
export TRANSFORMERS_OFFLINE=1
```

### GPU/CUDA Issues

By default, Munajjam auto-detects your device. To force CPU:

```python
from munajjam.config import configure
configure(device="cpu")
```

### Memory Issues

For large audio files or batch processing:

```python
# Use faster-whisper backend (more memory efficient)
configure(model_type="faster-whisper")

# Or process in chunks
# (See advanced examples for chunking strategies)
```

---

## Contributing Examples

Have a useful example? We'd love to include it!

1. Follow the naming convention: `XX_descriptive_name.py`
2. Include a docstring at the top explaining what it demonstrates
3. Add comments throughout the code
4. Update this README with a description
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.

---

## Need Help?

- **Documentation**: See the main [README.md](../README.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/munajjam/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/munajjam/discussions)
