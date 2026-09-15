# Experimental: Cloud-based WhisperX Alignment via Replicate

> **Status**: Experimental / opt-in. The default transcription path (local
> WhisperX) is unaffected — this adds an additional, optional mode.

## Overview

Munajjam's default transcription pipeline runs WhisperX (VAD + Wav2Vec2 CTC
alignment) locally, which requires a GPU for reasonable performance. This
feature adds an alternative: sending audio to a WhisperX model hosted on
[Replicate](https://replicate.com)'s GPUs instead, so the backend can run
without any local GPU at all.

This is experimental and intended for evaluating whether serverless inference
can match the accuracy of the local pipeline — it is not a replacement for the
default local backend.

## How it works

1. A request to `POST /align/{surah_number}` includes `alignment_mode=replicate`
   in the form data (default is `local`, i.e. the existing behavior — no change
   for existing clients).
2. The backend uploads the audio to a hosted WhisperX model on Replicate and
   polls synchronously until the prediction completes.
3. The returned word-level timestamps are mapped onto the surah's reference
   ayah text using the same fuzzy-matching / boundary-refinement logic the
   local WhisperX backend uses, producing the standard `Segment` objects.
4. The response shape from `/align/status/{job_id}` is identical regardless of
   which mode was used.

## Setup

### 1. Get a Replicate API token

Create an account and generate a token at
[replicate.com/account/api-tokens](https://replicate.com/account/api-tokens).

> **Note**: the default model (`victor-upmeet/whisperx`) is not covered by
> Replicate's free tier. A funded Replicate account is required to actually run
> predictions through this mode — see [Known limitations](#known-limitations).

### 2. Set the environment variable

Add to your `.env` file (same folder as `docker-compose.yml`):

```
REPLICATE_API_TOKEN=r8_your_token_here
```

This is read directly from the environment (not through `MunajjamSettings`), to
match the Replicate SDK's own standard variable name.

If you're running via Docker, `docker-compose.yml` forwards it into the
container the same way `HF_TOKEN` is already forwarded:

```yaml
environment:
  - HF_TOKEN=${HF_TOKEN:-}
  - REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN:-}
```

### 3. Rebuild

```bash
docker compose up --build
```

The `replicate` Python package is installed as part of the image build. If
`REPLICATE_API_TOKEN` is not set, the server still starts normally — the token
is only required at the moment a request actually uses `alignment_mode=replicate`.

## Usage

```bash
curl -X POST "http://localhost:8000/align/1" \
  -F "file=@surah1.mp3" \
  -F "alignment_mode=replicate"
```

Poll for the result the same way as the default mode:

```bash
curl "http://localhost:8000/align/status/<job_id>"
```

## Architecture notes

- **`ReplicateWhisperXTranscriber`** (`transcription/replicate_transcriber.py`)
  implements the same `BaseTranscriber` interface as the local `Whisperx` class,
  so it's a drop-in alternative via `WhisperFactory`.
- **Synchronous by design.** `.transcribe()` blocks until the Replicate
  prediction finishes (submit → poll → parse), rather than using webhooks. This
  fits directly into the existing background-job architecture in `server.py`
  with no changes to the job queue or status-polling endpoints, and avoids
  needing a publicly reachable webhook URL for local development.
- **Model version resolved dynamically.** Community-hosted models on Replicate
  (unlike Replicate's own "official" models) require a specific version hash,
  not just an `owner/name` reference. Rather than hardcoding a version hash
  (which would go stale whenever the model owner pushes an update), the latest
  version is resolved at runtime via the Replicate API.
- **Shared alignment logic.** The fuzzy word-matching and ayah-boundary
  refinement logic used to build `Segment` objects is shared with the local
  WhisperX backend via `transcription/alignment_utils.py`, so both backends
  produce segments the same way rather than duplicating ~150 lines of alignment
  code.

## Known limitations

- **Requires a funded Replicate account.** The default model is not available
  on Replicate's free tier. Attempting to run without credit returns a `402
  Insufficient credit` error, surfaced through the job's `"error"` status.
- **Slower per-request than local, for short audio.** Network upload + queueing
  time on Replicate's side adds latency that a local GPU with a warm model
  doesn't have. This mode is being evaluated for infrastructure simplicity, not
  raw latency.
- **Accuracy vs. the local pipeline has not yet been validated end-to-end**
  with real audio, due to the billing limitation above. Unit tests cover
  request construction, auth handling, output parsing, and error propagation,
  but not a live comparison of transcription accuracy against the local
  WhisperX backend.

## Testing

Unit tests: `tests/unit/test_replicate_transcriber.py`. Covers token handling,
version resolution, output parsing (including edge cases like missing
confidence scores, unaligned words, and empty output), and error propagation on
failed predictions.