# Faster Transcription Backend Research

> **Issue**: [#46](https://github.com/Itqan-community/Munajjam/issues/46)
> **Date**: February 2026
> **Scope**: Evaluate faster alternatives to Munajjam's current Whisper backends for 114-surah batch runs.

## Table of Contents

- [Current State](#current-state)
- [Backends Evaluated](#backends-evaluated)
  - [1. faster-whisper (current default)](#1-faster-whisper-current-default)
  - [2. whisper.cpp](#2-whispercpp)
  - [3. WhisperX](#3-whisperx)
  - [4. Whisper Large v3 Turbo](#4-whisper-large-v3-turbo)
  - [5. mlx-whisper](#5-mlx-whisper)
  - [6. insanely-fast-whisper](#6-insanely-fast-whisper)
  - [7. distil-whisper](#7-distil-whisper)
- [Comparison Matrix](#comparison-matrix)
- [Available Quran ASR Models](#available-quran-asr-models)
- [Recommendations](#recommendations)
- [Sources](#sources)

---

## Current State

Munajjam currently supports two transcription backends (configured via `model_type` in `munajjam/config.py`):

| Setting | Value |
|---------|-------|
| Default model | `OdyAsh/faster-whisper-base-ar-quran` |
| Default backend | `faster-whisper` |
| Alternative backend | `transformers` (HuggingFace pipeline) |
| Device | Auto-detect (CUDA > MPS > CPU) |
| Word timestamps | faster-whisper only (two-pass strategy) |

The **faster-whisper** backend uses CTranslate2 under the hood and is already 4-5x faster than vanilla OpenAI Whisper. The **transformers** backend provides compatibility but does not produce word-level timestamps.

Key bottleneck: the faster-whisper backend uses a **two-pass transcription strategy** (first pass for text, second pass for word timestamps), effectively doubling transcription time when word timestamps are needed.

---

## Backends Evaluated

### 1. faster-whisper (current default)

**What it is**: CTranslate2-based Whisper implementation by Systran. Converts OpenAI Whisper models to an optimized inference format.

| Criterion | Details |
|-----------|---------|
| Speed vs OpenAI Whisper | **4-5x faster** on GPU, with 3x less memory |
| Arabic support | Yes (inherits from Whisper) |
| Tarteel model compat | Yes -- `OdyAsh/faster-whisper-base-ar-quran` is CTranslate2 format |
| Word timestamps | Yes (built-in) |
| Platform | CPU (int8/float32), CUDA (float16/int8), **no native MPS** (falls back to CPU) |
| Integration effort | Already integrated |

**Strengths**: Best GPU throughput of all backends tested. Mature, battle-tested, widely used.

**Weaknesses**: MPS fallback to CPU is painful for Apple Silicon users. Two-pass word timestamp strategy doubles transcription time.

**Sources**: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)

---

### 2. whisper.cpp

**What it is**: C/C++ port of Whisper by Georgi Gerganov (author of llama.cpp). Zero runtime allocations, GGML tensor library.

| Criterion | Details |
|-----------|---------|
| Speed vs faster-whisper (CPU) | **1.3x slower** on CPU (Intel i7-12700H, FP32) |
| Speed vs faster-whisper (GPU) | **~10x slower** (3.4 min vs 20s for 12-min audio, RTX 4080) |
| Arabic support | Yes, but **67.7% hallucination/repetition rate** in independent testing (vs 45.1% for faster-whisper) |
| Tarteel model compat | Requires conversion via `convert-h5-to-ggml.py`. **Conversion of fine-tuned models is flaky** -- documented cases of garbage output after conversion |
| Word timestamps | Yes, via token timestamps or DTW (Dynamic Time Warping) |
| Platform | **Broadest**: CPU (AVX/NEON), CUDA, Metal, Vulkan, OpenVINO, CoreML, WebAssembly, iOS, Android |
| Integration effort | **Medium** -- requires model conversion + Python bindings (`pywhispercpp`) |
| Quantization | q4_0, q4_1, q5_0, q5_1, q8_0 (e.g., base model: 142 MiB full, 57 MiB q5_0) |

**Strengths**: Unmatched platform breadth. No Python/PyTorch dependency. Excellent for embedded/mobile (iOS, Android, Raspberry Pi, WebAssembly). Smallest memory footprint with quantization.

**Weaknesses**: Slower than faster-whisper on both CPU and GPU. High hallucination rate on Arabic. Fine-tuned model conversion is unreliable.

**When to consider**: Mobile deployment, WebAssembly targets, environments where Python/PyTorch cannot be installed.

**Sources**: [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), [whisper.cpp#1127](https://github.com/ggml-org/whisper.cpp/issues/1127), [Quids Showdown](https://quids.tech/blog/showdown-of-whisper-variants/)

---

### 3. WhisperX

**What it is**: Complete transcription pipeline by Max Bain (m-bain). Combines faster-whisper + VAD pre-segmentation + batched inference + phoneme-based forced alignment.

| Criterion | Details |
|-----------|---------|
| Speed | **12x faster** than vanilla Whisper via batched inference. Claims **70x realtime** with large-v2 |
| Arabic support | **Yes** -- both transcription and forced alignment. Uses `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` for Arabic phoneme alignment |
| Tarteel model compat | Yes -- uses faster-whisper as backend, so any CTranslate2 model works |
| Word timestamps | **Best-in-class** -- phoneme-level forced alignment via wav2vec2, not just attention-based |
| Platform | Primarily NVIDIA GPU (CUDA). CPU possible but slow |
| Integration effort | **Medium-High** -- separate pipeline (load model, transcribe, load alignment model, align). Additional dependency on wav2vec2 Arabic model (~1.2 GB) |
| Extra features | Speaker diarization (optional, via pyannote.audio), VAD-based hallucination reduction |

**Strengths**: Best word-level timestamp accuracy of all backends (phoneme-based, not attention-based). VAD pre-segmentation reduces hallucination. Batched inference for throughput. Arabic alignment model exists.

**Weaknesses**: Heavier dependency footprint (wav2vec2 model + pyannote). More complex API (multi-step pipeline). Some reports of timestamp quality issues vs Montreal Forced Aligner.

**When to consider**: When word-level timestamp precision is critical. Batch processing of many surahs where throughput matters.

**Sources**: [m-bain/whisperX](https://github.com/m-bain/whisperX), [WhisperX paper (arXiv:2303.00747)](https://arxiv.org/abs/2303.00747)

---

### 4. Whisper Large v3 Turbo

**What it is**: OpenAI's pruned + fine-tuned Whisper. Decoder reduced from 32 to 4 layers, encoder unchanged. 809M params (vs 1,550M for large-v3).

| Criterion | Details |
|-----------|---------|
| Speed vs large-v3 | **2.7x-7.5x faster** (hardware-dependent). ~7.5x on RTX 2080 Ti with faster-whisper |
| VRAM | **2.5 GB** vs 4.5 GB for large-v3 (fp16, faster-whisper) |
| Arabic WER | **No published numbers**. OpenAI documents regression on Thai and Cantonese. Community reports Arabic phoneme errors (e.g., "نعم" transcribed as "Naah") |
| Tarteel model compat | Not directly -- it's a different model. But can be used as a replacement base model |
| Word timestamps | Yes, on both faster-whisper and whisper.cpp backends |
| Platform | All Whisper-compatible backends: faster-whisper (CTranslate2), whisper.cpp (GGML), transformers |
| Integration effort | **Low** -- drop-in replacement model. `WhisperModel("turbo")` in faster-whisper |
| Quantization | GGML: q5_0 (547 MiB), q8_0 (874 MiB). CTranslate2: fp16, int8 |

**Strengths**: Massive speedup with minimal accuracy loss for most languages. Drop-in model swap (no code changes). Available in all formats (CTranslate2, GGML, transformers).

**Weaknesses**: Not fine-tuned on Quran Arabic -- general Arabic only. Translation quality degraded (training excluded translation data). No per-language WER published.

**When to consider**: As a higher-accuracy alternative to the base model for users with GPU. Could replace the current base model for initial transcription, with Tarteel model as a fine-tuning target.

**Sources**: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [faster-whisper#1030](https://github.com/SYSTRAN/faster-whisper/issues/1030), [openai/whisper#2363](https://github.com/openai/whisper/discussions/2363)

---

### 5. mlx-whisper

**What it is**: Whisper implementation on Apple's MLX framework. Part of Apple's official `ml-explore/mlx-examples` repository.

| Criterion | Details |
|-----------|---------|
| Speed (Apple Silicon) | **2x faster** than whisper.cpp on M-series chips (verified Jan 2026) |
| Speed (Linux) | CPU-only or CUDA (via `mlx[cuda12]`), but **no benchmarks available for Linux** |
| Arabic support | Yes (inherits from Whisper) |
| Tarteel model compat | Requires conversion via `convert.py` script. Not verified with tarteel-ai model specifically |
| Word timestamps | Yes (`word_timestamps=True`) |
| Platform | macOS (Metal), Linux (CPU, CUDA since MLX 0.30.x) |
| Integration effort | **Medium** -- one-time model conversion. API mirrors OpenAI Whisper's `transcribe()` |
| Quantization | 4-bit, 8-bit via conversion script |

**Strengths**: Fastest option on Apple Silicon. API mirrors standard Whisper (easy migration). Official Apple maintenance.

**Weaknesses**: Linux support is new and untested for this use case. Lives in a monorepo (less focused maintenance). Streaming described as unstable. Smaller ecosystem than faster-whisper.

**When to consider**: Apple Silicon-first deployments. macOS developer workflows where CUDA is unavailable.

**Sources**: [ml-explore/mlx-examples/whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [Bill Mill benchmark (Jan 2026)](https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html)

---

### 6. insanely-fast-whisper

**What it is**: Opinionated CLI/wrapper by Vaibhav Srivastav (HuggingFace) using Flash Attention 2 + batched inference on HuggingFace Transformers pipeline.

| Criterion | Details |
|-----------|---------|
| Speed | 150 min audio in **98 seconds** on A100 (large-v3, fp16 + Flash Attention 2) |
| Arabic support | Yes (via Whisper) |
| Tarteel model compat | Yes -- loads any HuggingFace Whisper model |
| Word timestamps | **Cannot combine with Flash Attention 2**. Must disable FA2 to get word timestamps (losing the speed benefit) |
| Platform | **NVIDIA GPU only** (Flash Attention 2 requires CUDA). Mac with MPS partially |
| Integration effort | **Low** -- thin wrapper over HF Transformers pipeline. Can use pipeline directly without the package |

**Strengths**: Fastest raw throughput on high-end NVIDIA GPUs. Simple integration (just a Transformers pipeline with optimizations).

**Weaknesses**: **Critical limitation**: word timestamps and Flash Attention 2 are mutually exclusive. GPU-only. The package itself is essentially a CLI convenience -- the same optimizations can be applied directly in Transformers code.

**When to consider**: High-throughput batch transcription on A100/H100 GPUs where word timestamps are not needed (e.g., text-only output).

**Sources**: [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper), [insanely-fast-whisper#40](https://github.com/Vaibhavs10/insanely-fast-whisper/issues/40)

---

### 7. distil-whisper

**What it is**: Knowledge-distilled Whisper by HuggingFace. Decoder reduced from 32 to 2 layers. Claims "6x faster, 50% smaller, within 1% WER."

| Criterion | Details |
|-----------|---------|
| Speed | **6.3x faster** than whisper-large-v3 |
| Arabic support | **NO. English-only.** All official checkpoints (distil-large-v2, v3, v3.5) are English-only |
| Tarteel model compat | N/A -- no Arabic variant exists |
| Word timestamps | Limited (fewer decoder layers = fewer cross-attention heads for alignment) |
| Integration effort | N/A |

**Verdict**: **Eliminated.** distil-whisper is English-only. No community Arabic variant exists. Arabic was explicitly excluded from the multilingual distillation research (Ferraz et al., ICASSP 2024) due to dialectal mismatch in evaluation datasets.

**Sources**: [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper), [distil-whisper#6](https://github.com/huggingface/distil-whisper/issues/6)

---

## Comparison Matrix

| Criterion | faster-whisper | whisper.cpp | WhisperX | large-v3-turbo | mlx-whisper | insanely-fast-whisper | distil-whisper |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Arabic Quran support** | Yes | Yes (with caveats) | Yes | Yes (general) | Yes | Yes | **No** |
| **Tarteel model** | Yes (default) | Conversion needed (flaky) | Yes | Different model | Conversion needed | Yes | N/A |
| **Word timestamps** | Yes | Yes (DTW) | **Best** (phoneme) | Yes | Yes | **Not with FA2** | N/A |
| **CPU speed** | Good (int8) | Good | Slow | Good | Unknown | No CPU | N/A |
| **CUDA speed** | **Best** | Slow | **Best** (batched) | Very good | Unknown | **Best** (FA2) | N/A |
| **Apple Silicon** | CPU fallback | Metal | CPU fallback | CPU fallback | **Best** | No | N/A |
| **Memory footprint** | Medium | **Smallest** (quantized) | Large (wav2vec2 extra) | Medium-Large | Medium | Large | N/A |
| **Integration effort** | **Already done** | Medium | Medium-High | **Low** (model swap) | Medium | Low | N/A |
| **Hallucination risk** | Low | **High (67.7%)** | **Lowest** (VAD) | Low | Low | Low | N/A |
| **Maturity** | **High** | High | Medium | High | Medium | Medium | N/A |

---

## Available Quran ASR Models

Models discovered on Hugging Face, ranked by adoption:

| Model | Architecture | Params | Downloads | Format | License |
|-------|-------------|--------|-----------|--------|---------|
| `tarteel-ai/whisper-base-ar-quran` | Whisper base | ~74M | **160.7K** | PyTorch | Apache-2.0 |
| `tarteel-ai/whisper-tiny-ar-quran` | Whisper tiny | ~39M | 24.8K | PyTorch | Apache-2.0 |
| `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final` | Wav2Vec2 CTC | 315.5M | 8.2K | safetensors | Apache-2.0 |
| `IJyad/whisper-large-v3-Tarteel` | Whisper large-v3 | 1,543M | 2.1K | safetensors | MIT |
| `OdyAsh/faster-whisper-base-ar-quran` | Whisper base | ~74M | 1.2K | PyTorch | Apache-2.0 |

**Evaluation data**: `tarteel-ai/whisper-base-ar-quran` achieves **5.75% WER** on its evaluation set (Quran recitation).

### Ecosystem gaps

1. **No distil-whisper for Arabic** -- English-only. No community variant exists.
2. **No production-quality GGML conversions** -- two repos exist with zero downloads.
3. **No whisper-small or whisper-medium Quran fine-tunes** from tarteel-ai (only tiny and base).
4. **No non-Whisper/non-Wav2Vec2 Quran ASR** (no Conformer, NeMo, etc.).

---

## Recommendations

### Priority 1: WhisperX backend (High impact, Medium effort)

**Rationale**: WhisperX combines multiple optimizations that directly address Munajjam's needs:
- Uses faster-whisper internally (already compatible with current Tarteel model)
- VAD pre-segmentation reduces hallucination -- critical for continuous reciters (see issue [#47](https://github.com/Itqan-community/Munajjam/issues/47))
- Batched inference provides 12x throughput for 114-surah batch runs
- Arabic phoneme alignment via `wav2vec2-large-xlsr-53-arabic` produces word timestamps far more precise than Whisper's attention-based method
- **Eliminates the two-pass word timestamp bottleneck** in the current faster-whisper backend

**Integration approach**:
1. Add `whisperx` as an optional dependency (`pip install munajjam[whisperx]`)
2. Implement a `WhisperXTranscriber` class extending `BaseTranscriber`
3. The VAD segmentation could replace or augment the current `detect_silences()` function
4. Forced alignment replaces the current two-pass transcription strategy

**Dependencies**: `whisperx`, `torch`, `torchaudio`, plus ~1.2 GB wav2vec2 Arabic model download.

### Priority 2: Whisper Large v3 Turbo as model option (Low effort, Medium impact)

**Rationale**: Drop-in model swap with no code changes. 2.7-7.5x faster than large-v3 at ~large-v2 accuracy. Already available in CTranslate2 format.

**Integration approach**:
1. Add `"turbo"` as a recognized model preset in config
2. Map it to `deepdml/faster-whisper-large-v3-turbo-ct2`
3. Document the accuracy vs speed tradeoff (not Quran-fine-tuned, but good for initial rough alignment)

**Use case**: Users with GPU who want fast first-pass transcription. Could be paired with a Tarteel model refinement step.

### Priority 3: mlx-whisper backend (Medium effort, Niche impact)

**Rationale**: 2x faster than whisper.cpp on Apple Silicon. Addresses the MPS gap where faster-whisper falls back to CPU. Growing user base on macOS.

**Integration approach**:
1. Add `mlx-whisper` as an optional dependency (`pip install munajjam[mlx]`)
2. Implement `MLXTranscriber` extending `BaseTranscriber`
3. Convert tarteel-ai model to MLX format and host on HuggingFace
4. Auto-select when `device="mps"` is detected

**Caveat**: Only benefits Apple Silicon users. Linux CUDA users are already well-served by faster-whisper.

### Not recommended

| Backend | Reason |
|---------|--------|
| **whisper.cpp** | Slower than faster-whisper on both CPU and GPU. 67.7% hallucination rate. Flaky fine-tuned model conversion. Only consider for mobile/embedded targets. |
| **insanely-fast-whisper** | Word timestamps incompatible with Flash Attention 2. Without FA2, it's just a standard Transformers pipeline (already supported). |
| **distil-whisper** | English-only. No Arabic variant exists or is planned. |

---

## Sources

### Backend repositories
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2-based Whisper
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) -- C/C++ Whisper port
- [m-bain/whisperX](https://github.com/m-bain/whisperX) -- Batched + forced alignment pipeline
- [ml-explore/mlx-examples/whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) -- Apple MLX Whisper
- [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) -- Flash Attention 2 wrapper
- [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper) -- Distilled Whisper (English-only)

### Models
- [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) -- Primary Quran ASR model (5.75% WER)
- [OdyAsh/faster-whisper-base-ar-quran](https://huggingface.co/OdyAsh/faster-whisper-base-ar-quran) -- CTranslate2 variant
- [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) -- Pruned large-v3 (809M params)
- [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) -- Turbo in CTranslate2
- [jonatasgrosman/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic) -- Arabic alignment model used by WhisperX

### Benchmarks and comparisons
- [faster-whisper vs whisper.cpp (GitHub issue #1127)](https://github.com/ggml-org/whisper.cpp/issues/1127)
- [faster-whisper turbo benchmark (GitHub issue #1030)](https://github.com/SYSTRAN/faster-whisper/issues/1030)
- [Quids: Showdown of Whisper Variants](https://quids.tech/blog/showdown-of-whisper-variants/)
- [Bill Mill: mlx-whisper vs whisper.cpp (Jan 2026)](https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html)
- [WhisperX paper (arXiv:2303.00747)](https://arxiv.org/abs/2303.00747)
- [Modal: Choosing Whisper Variants](https://modal.com/blog/choosing-whisper-variants)

### Arabic-specific
- [Multilingual DistilWhisper (arXiv:2311.01070)](https://arxiv.org/abs/2311.01070) -- Arabic excluded from distillation experiments
- [whisper.cpp hallucination rate](https://quids.tech/blog/showdown-of-whisper-variants/) -- 67.7% line repetition
- [WhisperX Arabic alignment model](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py) -- `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`
