# Munajjam مُنَجِّم

> A Python library to synchronize Quran Ayat with audio recitations

This project automatically synchronizes Quranic verses (ayahs) with recitation audio by generating accurate timestamps for the start and end of each ayah.

## Table of Contents

- [Roadmap](#-roadmap)
- [How It Works](#-how-it-works)
- [Algorithm Overview](#-algorithm-overview)
- [Folder Structure](#folder-structure)
- [Documentation](#documentation)
- [Demo](#demo)
- [Contributing](#contributing)
- [Debugging Video](#debugging-video) ⭐️

## 🔬 How It Works

Munajjam uses a sophisticated two-stage pipeline to synchronize Quranic audio with verse timestamps:

### Stage 1: Transcription

- Uses **Tarteel AI's Whisper model** (specialized for Quranic Arabic)
- Detects silence periods to segment the audio intelligently
- Identifies special segments (Isti'aza and Basmala) with pattern matching
- Supports both standard transformers and faster-whisper backends
- Optimized for Apple Silicon (MPS) GPU acceleration

### Stage 2: Intelligent Alignment

- Matches transcribed segments with canonical Quranic text
- Implements smart merging when multiple segments form one ayah
- Uses **buffer extension** to prevent word cutoffs at boundaries
- Applies **silence gap detection** to identify ayah boundaries
- Handles overlapping text removal for clean alignment

---

## 🧮 Algorithm Overview

### 1. **Smart Buffer System** 🎯

The buffer system extends ayah timestamps into adjacent silence periods to capture complete recitations without cutting off words.

**How it works:**

- **Before ayah start**: Extends backward up to 0.3s into preceding silence
- **After ayah end**: Extends forward up to 0.3s into following silence
- **Overlap prevention**: Ensures no overlap with adjacent ayahs
- **Adaptive**: Uses actual silence data, not fixed offsets

**Benefits:**

- Eliminates word cutoffs at ayah boundaries
- Preserves natural pause patterns in recitation
- Maintains clean separation between ayahs

```python
# Example: An ayah detected at 10.0s - 15.0s with silences at:
# - [8.5s - 9.8s] (before)
# - [15.2s - 16.0s] (after)
#
# Applied buffer extends to: 9.7s - 15.5s
# (0.3s backward into first silence, 0.3s forward into second)
```

### 2. **Silence Gap Detection** 🔍

Identifies ayah boundaries by detecting significant silence gaps between segments, combined with textual analysis.

**Algorithm:**

1. **Acoustic check**: Look for silence gaps ≥ 0.18s between segments
2. **Textual check**: Verify next segment starts the next ayah (similarity > 0.6)
3. **Boundary confirmation**: Only treat as ayah boundary if both conditions met

**Why it matters:**

- Handles cases where reciter pauses mid-ayah (doesn't split incorrectly)
- Detects merged ayahs that were transcribed as one segment
- Improves alignment accuracy for complex recitation patterns

### 3. **Special Segment Handling** 🕌

Properly tracks Isti'aza (أعوذ بالله من الشيطان الرجيم) and Basmala (بسم الله الرحمن الرحيم) segments separately from ayahs.

**Features:**

- Assigns special `id = 0` and `ayah_index = -1` to these segments
- Pattern-based detection even when metadata is missing
- Excluded from ayah counting and alignment logic
- Preserved in output with proper `type` field

### 4. **Text Similarity Matching** 📝

Uses multiple similarity checks for robust alignment:

**Last words check** (primary): Compares last N words of segment with expected ayah

- Adaptive N: Uses 3 words for long ayahs, 2 for medium, 1 for short
- Threshold: 0.6 similarity score

**Full text similarity** (secondary): Compares entire segment with canonical text

- Guards against premature termination
- Coverage ratio check ensures ≥70% of ayah is captured

**Required tokens guard**: Prevents early cutoff for specific ayahs

- Example: Ayah 2 requires both "ارجع" and "فطور" before finalizing

### 5. **Overlap Removal** 🧹

Intelligently merges segments while removing duplicate words:

**Algorithm:**

1. Count word frequencies in first segment
2. For each word in second segment:
   - If word exists in first segment, decrement counter and skip
   - Otherwise, append to merged text
3. Return cleaned merged text

**Prevents:** "...الرحيم بسم الله..." → "...الرحيم..."

### 6. **Performance Optimizations** ⚡

**Model Caching:**

- Loads model once and caches it for entire session
- Avoids expensive model reloading between surahs
- Supports both transformers and faster-whisper

**Device Optimization:**

- Auto-detects best device: CUDA > MPS > CPU
- Apple Silicon: Uses MPS (Metal Performance Shaders) with float32
- CUDA: Uses float16 for faster inference
- Model compilation with `torch.compile()` (when supported)

**Inference Optimization:**

- Greedy decoding (`num_beams=1`) instead of beam search
- Limited token generation (`max_new_tokens=128`)
- Explicit attention mask passing

---

## Folder Structure

Here is the high-level structure of the project directory:

```
E:\Quran\
├───.gitignore
├───current_config.json
├───docs.md
├───main.py
├───requirements.md
├───README.md
├───__pycache__\
├───.git\
├───.vscode\
├───data\
├───ffmpeg-8.0\
└───src\
```

- **src/**: Contains the main Python source code for processing.
- **data/**: Holds all data files, including raw audio, CSVs with timestamps, and JSON outputs.
- **docs.md**: Contains the pseudocode and detailed documentation for the project.
- **requirements.md**: Lists the prerequisites and dependencies for the project.
- **main.py**: The main script to run the application.

## Documentation

The project's pseudocode can be found in the `docs.md` file.
[View Pseudocode](/PSEUDO%20CODE.md)

Munajjam Workflow of 1st Edittion: [View Munajjam V0.1 Workflow](https://www.figma.com/board/3OYO15uIX1B2PowhkTybdk/Munjjam?node-id=0-1&t=dEGKe5WE9LjQQ44z-1)
Short PRD
[View Full Requirements](/docs/Requirements.md)

For a more detailed explanation of the project, you can view the full documentation here: [View Full Documentation](https://duuniv-my.sharepoint.com/:w:/g/personal/reemrefaat_students_du_edu_eg/EQCiSn240TpCirIrPUrxWEgB-hBbnqoRrZD2g4cU9BLRlg)

## Demo

A video demonstration of the project is available at the link below.

[Watch the Demo Video](https://drive.google.com/file/d/169TmJ8W_LIyuZ3hNSI25mfedSofHxbyG/view?usp=sharing)

## Contributing

We welcome contributions from the community! 🤝

1. **Check the [Roadmap](plan.md)** to see planned features
2. **Read [CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines
3. **Browse [open issues](https://github.com/Itqan-community/Munajjam/issues)** for tasks

Look for issues labeled `good first issue` if you're new!

---

## Debugging Video

[Watch the Debugging Video](https://drive.google.com/file/d/1mOZ8sYCLRmXXD0WMnA89kzRUemZmQmkE/view?usp=sharing)

---

## 📊 Technical Specifications

### Model

- **Primary**: Tarteel AI Whisper Base (Arabic Quran-specialized)
- **Backend**: Supports both Hugging Face Transformers and faster-whisper
- **Device**: CUDA, MPS (Apple Silicon), or CPU

### Audio Processing

- **Silence Detection**: -30dB threshold, 300ms minimum duration
- **Sample Rate**: 16kHz
- **Format**: WAV (mono recommended)

### Alignment Parameters

- **Similarity Threshold**: 0.6 (60%)
- **Buffer Duration**: 0.3 seconds
- **Minimum Silence Gap**: 0.18 seconds
- **Coverage Requirement**: 0.7 (70%)

---

## Acknowledgments

- [Tarteel AI](https://tarteel.ai/) for the Quran-specialized Whisper model
- The open source community

---

**Let's build something beautiful for the Ummah together! 🌙**
