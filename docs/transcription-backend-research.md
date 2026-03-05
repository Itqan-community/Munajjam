# Faster Transcription Backend Research

> **Issue**: [#46](https://github.com/Itqan-community/Munajjam/issues/46)
> **Current backend**: `faster-whisper` with `OdyAsh/faster-whisper-base-ar-quran` (CTranslate2)
> **Problem**: Transcription is the bottleneck for full 114-surah batch runs.

## Executive Summary

| Backend | Speed vs Current | Arabic Quality | Tarteel Compat | Word Timestamps | Platforms | Integration Effort |
|---------|-----------------|----------------|----------------|-----------------|-----------|-------------------|
| **WhisperX** | ~3-5x | Good (uses faster-whisper) | Yes | Yes (forced alignment) | CPU/CUDA | Low |
| **large-v3-turbo** | ~2-3x | Good | Drop-in swap | Yes | CPU/CUDA/MPS | Very Low |
| **insanely-fast-whisper** | ~8-12x (GPU) | Good | Yes (HF format) | Yes | CUDA only | Medium |
| **lightning-whisper-mlx** | ~4-10x (Apple) | Good | Needs conversion | Yes | MPS only | Medium |
| **whisper.cpp** | ~2-4x | Poor (hallucinations) | Needs GGML convert | Limited | CPU/CUDA/MPS | High |
| **distil-whisper** | ~6x | English only | No | Yes | CPU/CUDA | N/A |

**Top recommendation**: WhisperX (short-term) + large-v3-turbo model swap (immediate)

---

## 1. WhisperX (m-bain/whisperX)

### How It Works
WhisperX wraps faster-whisper and adds three layers on top:
1. **VAD (Voice Activity Detection)** via pyannote — segments audio before transcription, reducing hallucinations on silent sections
2. **Batched inference** — processes multiple segments in parallel
3. **Forced alignment** via wav2vec2 — produces word-level timestamps without the slower Whisper word-timestamp mode

### Speed
- **60-70x real-time** on GPU with large-v2 (vs ~15x for standard faster-whisper)
- Batched processing is the key speedup — groups audio segments and processes them as a batch
- For a typical 30-minute surah: ~25-30 seconds vs ~2+ minutes with current setup

### Arabic Support
- Inherits faster-whisper's Arabic capabilities since it uses it as the transcription engine
- VAD reduces hallucination on silent sections (common in Quran recordings between ayahs)
- **Known issue**: batched inference with Arabic fine-tuned models can produce hallucinations in some cases ([faster-whisper #954](https://github.com/SYSTRAN/faster-whisper/issues/954)). Single-segment inference remains accurate.

### Tarteel Model Compatibility
- **Yes** — WhisperX calls faster-whisper internally, so `OdyAsh/faster-whisper-base-ar-quran` works directly
- Set via `whisperx.load_model("OdyAsh/faster-whisper-base-ar-quran")`

### Word Timestamps
- Uses phoneme-based forced alignment (wav2vec2) instead of Whisper's native word timestamps
- Arabic alignment model exists: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`
- Produces more precise word boundaries than Whisper's cross-attention timestamps

### Platform Support
- **CUDA**: Full support, recommended
- **CPU**: Supported but slow (no batching benefit)
- **MPS**: Not officially supported by pyannote VAD

### Integration Effort: Low
WhisperX's API is similar to faster-whisper. Key changes in `WhisperTranscriber`:
```python
import whisperx

model = whisperx.load_model(
    "OdyAsh/faster-whisper-base-ar-quran",
    device="cuda",
    compute_type="float16",
)
audio = whisperx.load_audio(audio_path)
result = model.transcribe(audio, batch_size=16)

# Optional: forced alignment for word timestamps
align_model, metadata = whisperx.load_align_model(
    language_code="ar", device="cuda"
)
result = whisperx.align(
    result["segments"], align_model, metadata, audio, device="cuda"
)
```

### Risks
- Batched inference hallucination with Arabic fine-tuned models needs testing
- pyannote VAD requires a HuggingFace token (gated model)
- No MPS support limits Apple Silicon users

---

## 2. large-v3-turbo Model (Drop-in Swap)

### How It Works
OpenAI's `whisper-large-v3-turbo` is a pruned version of large-v3 with decoder layers reduced from 32 to 4. Same architecture, much faster inference.

### Speed
- **~2-3x faster** than large-v3 at similar quality
- 216x real-time with faster-whisper on GPU
- Available as CTranslate2 format: `deepdml/faster-whisper-large-v3-turbo-ct2`

### Arabic Support
- Trained on the same multilingual data as large-v3
- ~1% lower WER than distil-whisper across languages
- Not fine-tuned on Quran recitation — would need quality testing against Tarteel models

### Tarteel Model Compatibility
- **Not directly compatible** — different model weights
- Could be used as a **fallback/fast mode** when Tarteel precision isn't critical
- Tarteel could potentially fine-tune on top of turbo architecture for future gains

### Word Timestamps
- Full word-level timestamp support via faster-whisper

### Platform Support
- CPU, CUDA, MPS — all supported via faster-whisper

### Integration Effort: Very Low
```python
# Just change the model_id in config
configure(model_id="deepdml/faster-whisper-large-v3-turbo-ct2")
```

### Risks
- Not Quran-specific — may produce lower quality transcriptions for Quranic Arabic
- Useful as a speed-mode option, not a primary replacement for Tarteel models

---

## 3. insanely-fast-whisper (Vaibhavs10)

### How It Works
Uses HuggingFace Transformers pipeline with Flash Attention 2 and BetterTransformer optimizations. Maximizes GPU throughput through large-batch processing.

### Speed
- **150 minutes of audio in <98 seconds** on A100 GPU
- ~12x faster than standard faster-whisper on high-end GPUs
- Speed comes from Flash Attention 2 — requires CUDA GPU with sufficient VRAM

### Arabic Support
- Uses standard Whisper models — Arabic quality matches the base model
- Can load any HuggingFace Whisper model including Tarteel's

### Tarteel Model Compatibility
- **Yes** — loads HuggingFace format models directly
- Tarteel's base model (`tarteel-ai/whisper-base-ar-quran`) is in HF format
- The faster-whisper CTranslate2 variant (`OdyAsh/faster-whisper-base-ar-quran`) would need the original HF weights

### Word Timestamps
- Supported via HuggingFace pipeline `return_timestamps="word"`

### Platform Support
- **CUDA only** — Flash Attention 2 requires NVIDIA GPU
- No CPU or MPS support for the optimized path
- Requires significant VRAM (8GB+ recommended)

### Integration Effort: Medium
Would need a new transcriber backend class since it uses HF Transformers instead of CTranslate2:
```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="tarteel-ai/whisper-base-ar-quran",
    torch_dtype=torch.float16,
    device="cuda:0",
)
result = pipe(
    audio_path,
    chunk_length_s=30,
    batch_size=24,
    return_timestamps="word",
)
```

### Risks
- CUDA-only limits accessibility
- High VRAM requirement
- Flash Attention 2 compilation can be tricky

---

## 4. MLX Whisper / Lightning Whisper MLX

### How It Works
Native Apple Silicon implementation using Apple's MLX framework. Processes Whisper inference directly on the Neural Engine / GPU unified memory.

### Speed
- **lightning-whisper-mlx**: 10x faster than whisper.cpp, 4x faster than standard mlx-whisper
- Peak memory under 2GB (vs 3-4GB for WhisperX)
- Best performance on M2/M3/M4 chips

### Arabic Support
- Uses standard Whisper model weights — Arabic quality matches base model
- No known Arabic-specific issues

### Tarteel Model Compatibility
- **Needs conversion** — MLX uses its own weight format
- Conversion from HuggingFace format is possible via `mlx_whisper.convert`
- The faster-whisper CTranslate2 format cannot be used directly

### Word Timestamps
- Supported in mlx-whisper

### Platform Support
- **MPS/Apple Silicon only** — this is the entire point
- No CUDA or CPU fallback

### Integration Effort: Medium
```python
import mlx_whisper

result = mlx_whisper.transcribe(
    audio_path,
    path_or_hf_repo="mlx-community/whisper-base",  # need Tarteel MLX conversion
    word_timestamps=True,
)
```

### Risks
- Apple Silicon only — limits CI/CD and server deployment
- Tarteel model needs conversion and quality validation
- Smaller community, fewer battle-tested edge cases

---

## 5. whisper.cpp (ggml-org)

### How It Works
C/C++ port of Whisper using GGML tensor library. Runs without Python or PyTorch dependencies.

### Speed
- **2-4x faster than OpenAI Whisper** on CPU
- Good for edge/embedded deployments
- Metal acceleration on Apple Silicon

### Arabic Support
- **Poor** — reported 67.7% hallucination rate on Arabic audio
- 20% higher sentence repetition rate vs other variants
- Hallucination issues on silent sections are well-documented
- Model conversion to GGML format can introduce quality degradation

### Tarteel Model Compatibility
- **Requires GGML conversion** — complex and lossy process
- No pre-converted Tarteel GGML models available
- Conversion: HuggingFace → GGML via `whisper.cpp/models/convert-h5-to-ggml.py`

### Word Timestamps
- Basic support, less accurate than faster-whisper

### Platform Support
- CPU, CUDA, Metal (Apple Silicon) — very broad
- Minimal dependencies (no Python needed for inference)

### Integration Effort: High
- Would need C++ bindings or subprocess calls
- Different output format than current pipeline
- Quality issues with Arabic make this unsuitable

### Verdict: **Not recommended** for Arabic Quran transcription

---

## 6. distil-whisper (HuggingFace)

### Status
- **English only** — no Arabic support
- Research exists for multilingual distillation but Arabic was excluded due to dialect complexity
- Community could fine-tune a distilled Arabic model but this is a significant effort

### Verdict: **Not applicable** — no Arabic support

---

## Recommendations

### Priority 1: WhisperX Integration (High Impact, Low Effort)
**Why**: 3-5x speedup with minimal code changes. Uses faster-whisper internally so Tarteel model works directly. VAD preprocessing reduces hallucination on inter-ayah silences.

**Action items**:
1. Add `whisperx` as optional dependency
2. Create `WhisperXTranscriber` class extending `BaseTranscriber`
3. Test batched inference quality with `OdyAsh/faster-whisper-base-ar-quran`
4. If batched hallucination occurs, fall back to single-segment inference with VAD-based chunking (still faster than current approach due to VAD pre-segmentation)

**Risk mitigation**: WhisperX's VAD-first approach naturally produces better silence-split audio segments, which aligns with Munajjam's existing `detect_non_silent_chunks()` → transcribe pipeline. Even without batching, the VAD improvement alone is valuable.

### Priority 2: large-v3-turbo as Fast Mode (Immediate, Trivial)
**Why**: Single config change, 2-3x faster. Not Quran-specific but useful for previewing/draft alignment runs.

**Action items**:
1. Add `deepdml/faster-whisper-large-v3-turbo-ct2` as alternative model
2. Expose as `quality="draft"` vs `quality="precise"` (Tarteel) config option
3. Benchmark Arabic WER against Tarteel model on known surahs

### Priority 3: MLX Backend for Apple Silicon Users (Medium Effort)
**Why**: Significantly faster on Mac hardware where many contributors develop. Good for local development and testing workflows.

**Action items**:
1. Convert Tarteel model to MLX format
2. Create `MLXTranscriber` class
3. Auto-detect Apple Silicon and offer MLX as default on macOS

### Not Recommended
- **whisper.cpp**: Arabic hallucination rate is unacceptable for Quran text
- **distil-whisper**: English only, no Arabic support
- **insanely-fast-whisper**: CUDA-only requirement too restrictive; WhisperX provides similar batching benefits with broader platform support

---

## References

- [WhisperX](https://github.com/m-bain/whisperX) — Automatic Speech Recognition with word-level timestamps
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper
- [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) — Optimized MLX implementation
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — C/C++ Whisper port
- [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) — Flash Attention 2 Whisper
- [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) — Pruned Whisper model
- [faster-whisper batched inference issue #954](https://github.com/SYSTRAN/faster-whisper/issues/954) — Arabic hallucination report
- [Choosing between Whisper variants](https://modal.com/blog/choosing-whisper-variants) — Comparative analysis
- [CTranslate2 model conversion](https://medium.com/@balaragavesh/converting-your-fine-tuned-whisper-model-to-faster-whisper-using-ctranslate2-b272063d3204)
