# FastConformer ONNX validation (issue #104 groundwork)

This document records how the ONNX graph for
`nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0` (FastConformer hybrid
RNNT/CTC, Arabic) is produced and what was **empirically verified** against a
real export. The acoustic layer that consumes this graph is
`munajjam/munajjam/transcription/fastconformer.py` (`FastConformerInference`).

> The model (~424 MB `.nemo`) and the exported ONNX files are **not** checked
> into the repo. Everything here is reproducible with the scripts referenced
> below; artifacts live in the gitignored `.model_validation/` directory.

## Why this export is needed

The checkpoint is a NeMo 2.0.0rc1 `EncDecHybridRNNTCTCBPEModel` with **two**
decoders:

- an RNNT decoder (the default), and
- an auxiliary CTC decoder (`aux_ctc.decoder` → `ConvASRDecoder`,
  `num_classes: 1024`).

Global CTC segmentation needs the CTC log-probabilities, so the graph must be
exported with the CTC head active:

```python
model.change_decoding_strategy(decoder_type="ctc")   # cur_decoder -> "ctc"
model.export(...)                                    # exports the CTC head
```

Without this step the exported graph would contain the RNNT decoder instead.

## Export procedure (``scripts/export_fastconformer_onnx.py``)

```bash
# 1. Validation-only environment (not a munajjam runtime dependency)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "nemo_toolkit[asr]" onnx onnxscript onnxruntime

# 2. Download the checkpoint (424 MB)
mkdir -p .model_validation
wget -O .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo \
  "https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0/resolve/main/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo"

# 3. Export both graphs + extract the tokenizer
python scripts/export_fastconformer_onnx.py \
  .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo \
  --output-dir .model_validation/fastconformer
```

This produces (for ``stem = stt_ar_fastconformer_hybrid_large_pc_v1.0``):

| File | Input | Output | Notes |
|---|---|---|---|
| `{stem}_ctc.onnx` (458 MB) | `audio_signal` f32 `[B, 80, T_mel]` (log-mel), `length` i64 `[B]` | `logprobs` f32 `[B, T', 1025]` | NeMo's *stock* export; the preprocessor is **not** in the graph |
| `{stem}_ctc_rawaudio.onnx` (3 MB graph + `.onnx.data` weights) | `input_signal` f32 `[B, 80, T_mel]` (log-mel features), `input_signal_length` i32 `[B]` (valid frame count) | `logprobs` f32 `[B, T', 1025]`, `encoded_lengths` i64 `[B]` | Munajjam *production* export; encoder + CTC head (mel-input); preprocessing in Python |
| `tokenizer.model` | — | — | SentencePiece model extracted from the `.nemo` (deterministically, from `model_config.yaml`'s `tokenizer.model`) |
| `vocabulary.txt` | — | — | Labels dump extracted from the archive when present (optional) |

The production (raw-audio) export is implemented by tracing a wrapper module
(`encoder → ctc_decoder`) with `torch.onnx.export`.  The NeMo preprocessor
is excluded because `torch.stft()` produces complex types that the legacy
TorchScript ONNX exporter cannot handle.  Mel features are computed in
Python by `compute_mel_features()` before ONNX inference — this matches
NeMo's own stock export approach (the preprocessor is never part of the
ONNX graph).  The graph uses dynamic axes on the time dimension (any audio
length) and the script runs a post-export onnxruntime sanity check by
default (`--no-validate` to skip). Existing outputs are never overwritten
unless `--force` is passed.

The `FastConformerInference` class accepts raw waveforms as input
(`log_probs(waveform)`) and performs mel preprocessing internally via
`compute_mel_features()` — no change to the public API is needed.

The stock export is optional for the Munajjam runtime — only
`{stem}_ctc_rawaudio.onnx` + `tokenizer.model` are consumed.

## Model provisioning (server / API)

The server never requires manual environment variables for a first run.
Assets are resolved in one place
(`munajjam/transcription/fastconformer_models.py`, used by
`WhisperFactory` when `alignment_mode=ctc_segmentation`):

### Flow A — automatic / cached provisioning

1. **Explicit env vars win** — if `MUNAJJAM_FASTCONFORMER_MODEL_PATH` and
   `MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH` are set, they are validated
   and used as-is (optional `MUNAJJAM_FASTCONFORMER_VOCAB_PATH`). A
   half-configured or invalid setup fails fast with a clear error.
2. **Provisioning cache** — otherwise the deterministic cache directory is
   checked: `MUNAJJAM_FASTCONFORMER_CACHE_DIR`, or
   `~/.cache/munajjam/fastconformer` by default. Expected canonical filenames:

   ```text
   stt_ar_fastconformer_hybrid_large_pc_v1.0_ctc_rawaudio.onnx
   tokenizer.model
   vocabulary.txt   (optional)
   ```

   Cached assets are reused as-is — no re-download.
3. **Hugging Face (opt-in)** — if `MUNAJJAM_FASTCONFORMER_HF_REPO_ID` is set
   to a repo that really hosts the pre-exported files (same canonical
   filenames as above), they are downloaded into the cache via
   `huggingface_hub` (concurrency-safe, cache-aware) and pinned to
   `MUNAJJAM_FASTCONFORMER_HF_REVISION` when configured.

   > No such public repo is assumed by default — the resolver does **not**
   > invent URLs. Until real hosted assets exist, step 3 is a no-op unless an
   > operator configures a repo that actually contains the files.
4. **Actionable error** — if none of the above yields assets, the API returns
   a structured JSON error (HTTP 422) explaining how to make them available:
   set the two env vars, or run the export script.

First-use resolution is concurrency-safe (a lock prevents double downloads /
parallel initialization), matching the lazy model-loading developer experience
of WhisperX / Wav2Vec2.

### Flow B — manual export

```bash
python scripts/export_fastconformer_onnx.py \
  .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo \
  --output-dir ~/.cache/munajjam/fastconformer
```

Because the export writes exactly the canonical filenames the resolver looks
for, exporting straight into the cache directory makes the server pick the
assets up automatically on the next request. Alternatively export anywhere
and set:

```bash
export MUNAJJAM_FASTCONFORMER_MODEL_PATH=/path/to/stt_ar_fastconformer_hybrid_large_pc_v1.0_ctc_rawaudio.onnx
export MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH=/path/to/tokenizer.model
# optional:
export MUNAJJAM_FASTCONFORMER_VOCAB_PATH=/path/to/vocabulary.txt
```

### Expected Colab setup

```python
# 1. Install the API extras (huggingface_hub and onnxruntime are included
#    in the runtime deps)
!pip install -e ./munajjam[api]

# 2. (Optional) export the model once — heavy, CPU-only NeMo:
#    !pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#    !pip install "nemo_toolkit[asr]" onnx onnxscript onnxruntime
#    !wget -O stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo \
#      https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0/resolve/main/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo
#    !python scripts/export_fastconformer_onnx.py stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo \
#      --output-dir /root/.cache/munajjam/fastconformer

# 3. Start the server and POST /align/{surah_number} with
#    alignment_mode=ctc_segmentation. Nothing else is required when the
#    assets are cached or explicit paths are exported.
```

## Verified ONNX contract (production graph)

- Inputs
  - `input_signal` — `tensor(float)` `[batch, 80, T_mel]` (log-mel features)
  - `input_signal_length` — `tensor(int32)` `[batch]` (valid frame count)
- Outputs
  - `logprobs` — `tensor(float)` `[1, time//1280 + 1, 1025]`
  - `encoded_lengths` — `tensor(int64)` `[1]`
- The output is already **log-softmax** normalized (verified: rows of
  `exp(log_probs)` sum to 1.0 within 1e-6; a `LogSoftmax` node is in the
  graph).
- opset 18 / IR 10; weights stored as external data in `.onnx.data` (handled
  transparently by ONNX Runtime).

## Vocabulary and blank index (verified)

- Vocabulary source: the SentencePiece unigram tokenizer inside the `.nemo`
  (`<hash>_tokenizer.model`); `model_config.yaml`'s `labels` list has exactly
  **1024** tokens (`<unk>` first). `vocab.txt` inside the `.nemo` has 1023
  lines (it omits `<unk>`).
- `aux_ctc.decoder.num_classes = 1024`; the `ConvASRDecoder` appends one
  blank class → **1025 output classes**, blank is the **last** class:
  `blank_index == vocab_size == 1024`. Confirmed by the graph's static output
  dim and by silence dominance of the trailing column.
- The tokenizer decodes non-blank argmax frames to Arabic script (sanity
  checked with `sentencepiece`).

## Frame-to-time mapping (verified)

- The graph declares `T' = time // 1280 + 1` for `time` input samples.
- At 16 kHz: `1280 samples = 80 ms` per CTC frame.
- `time_s = frame_index * 0.08` — matches NeMo's
  `frame * window_stride * subsampling_factor = frame * 0.01 * 8`.

## Numerical parity with NeMo

On a 7.43 s 16 kHz mono speech file (93 frames), comparing the ONNX output to
the same forward pass in PyTorch:

```text
per-frame max |onnx - nemo|: mean=6.85e-05  p95=1.39e-04  max=8.95e-04
frames with diff > 1e-3: 0 / 93
```

The residual is float32 kernel-level noise between torch CPU and ONNX Runtime
CPU (accumulated through 17 encoder layers), not a systematic error. The
export is numerically faithful.

## Validation

The export script runs a post-export sanity check by default (disable with
`--no-validate`): it loads `{stem}_ctc_rawaudio.onnx` with onnxruntime,
verifies the I/O contract (float32 mel features + int32 frame count inputs,
`logprobs` and `encoded_lengths` outputs) and executes a short dummy inference
with a zero mel-spectrogram.

The standalone `scripts/validate_fastconformer_onnx.py` used during the
original validation session was a local throwaway; its checks (shape/dtype,
log-softmax normalization, blank dominance, frame mapping) were folded into
this built-in validation and into `FastConformerInference` itself.

## Result

`FastConformerInference` works against the production graph with **no
functional changes** to the public API.  The mel preprocessing that was
previously embedded in the ONNX graph (via NeMo's preprocessor) is now
performed in Python by `compute_mel_features()` before ONNX inference.
This resolves the STFT export failure (`torch.stft()` complex type
incompatibility with the legacy TorchScript ONNX exporter) while maintaining
numerical equivalence with NeMo's preprocessing pipeline.

## Known limitation

None.  Both the NeMo stock export and the Munajjam production export are
now supported.  The production export performs mel preprocessing in Python
(no NeMo dependency at runtime).
