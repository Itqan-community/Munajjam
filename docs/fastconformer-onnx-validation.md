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

## Export procedure

```bash
# 1. Validation-only environment (not a munajjam runtime dependency)
pip install torch torchaudio            # CPU build: --index-url https://download.pytorch.org/whl/cpu
pip install "nemo_toolkit[asr]" onnx onnxscript onnxruntime

# 2. Download the checkpoint (424 MB)
wget -O .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo \
  "https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0/resolve/main/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo"

# 3. Export both graphs
python scripts/export_fastconformer_onnx.py \
  .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo \
  .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0
```

This produces:

| File | Input | Output | Notes |
|---|---|---|---|
| `..._ctc.onnx` (458 MB) | `audio_signal` f32 `[B, 80, T_mel]` (log-mel), `length` i64 `[B]` | `logprobs` f32 `[B, T', 1025]` | NeMo's *stock* export; the preprocessor is **not** in the graph |
| `..._ctc_rawaudio.onnx` (3 MB graph + `.onnx.data` weights) | `input_signal` f32 `[B, T]` (raw 16 kHz waveform), `input_signal_length` i32 `[B]` | `logprobs` f32 `[B, T', 1025]`, `encoded_lengths` i64 `[B]` | Munajjam *production* export; preprocessor + encoder + CTC head in one graph |

The production (raw-audio) export is implemented by tracing a wrapper module
(`preprocessor → encoder → ctc_decoder`) with `torch.onnx.export`, so the
audio front-end is bit-identical to NeMo's training-time preprocessing (no
numpy reimplementation of STFT/mel/log/per-feature normalization needed).

## Verified ONNX contract (production graph)

- Inputs
  - `input_signal` — `tensor(float)` `[batch, time]`
  - `input_signal_length` — `tensor(int32)` `[batch]`
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

```
per-frame max |onnx - nemo|: mean=6.85e-05  p95=1.39e-04  max=8.95e-04
frames with diff > 1e-3: 0 / 93
```

The residual is float32 kernel-level noise between torch CPU and ONNX Runtime
CPU (accumulated through 17 encoder layers), not a systematic error. The
export is numerically faithful.

## Validation script

```bash
python scripts/validate_fastconformer_onnx.py \
  .model_validation/stt_ar_fastconformer_hybrid_large_pc_v1_ctc_rawaudio.onnx \
  <16kHz_mono_wav> \
  [.model_validation/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo]   # optional NeMo parity
```

It prints the graph contract, runs a real inference (shape/dtype/
log-softmax/blank behavior/frame mapping), optionally compares against NeMo,
and runs `FastConformerInference.log_probs()` end-to-end.

## Result

`FastConformerInference` works against the production graph with **no
functional changes**; only documentation was updated. It is safe to use as
the acoustic layer for `ctc_segmentation.py` (next phase: VAD chunking,
quranic phonemization, `ctc-segmentation`, blank reward, dynamic trimming).

## Known limitation

NeMo's *stock* mel-input export is **not** supported by
`FastConformerInference` (its input is log-mel features `[B, 80, T]`, which
would require reimplementing the mel front-end outside the graph). Use the
raw-audio production export.
