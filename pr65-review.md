## Review: feat: implement adaptive silence detection with retry logic

The algorithm is well-designed and the implementation is clean. The main question is whether this should ship without being wired into the actual pipeline.

All tests pass (14 new tests for the adaptive logic).

### Feature isn't connected to anything

The adaptive retry is implemented in `detect_non_silent_chunks()` with new parameters (`adaptive`, `expected_chunks`, `min_chunks_ratio`), but `WhisperTranscriber` in `whisper.py` (lines 221-225) still calls `detect_non_silent_chunks` without any of them. The adaptive mode can never activate through normal usage. Is there a follow-up PR planned for wiring this in, or should it be part of this one?

### Algorithm looks correct

The retry logic progressively relaxes thresholds:

| Level | silence_thresh | min_silence_len |
|-------|---------------|-----------------|
| 1 | +5 dB | 0.75x |
| 2 | +10 dB | 0.5x |
| 3 | +15 dB | 0.35x |
| 4 | +20 dB | 0.25x |

Good safety bounds: threshold capped at -10 dBFS, min silence length floored at 50ms. Best-result tracking across retries handles the case where more aggressive settings actually produce fewer chunks.

The refactoring into `_detect_non_silent_chunks_raw()` as a private helper is clean, and the public API is fully backward compatible (all new params default to off).

### Small issues in tests

- `test_adaptive_thresh_capped_at_minus_10` (line 321) has `assert thresh >= -10 or thresh <= -10` which is always true for any number. The real assertion is on the next line, so this is just dead code.
- Unused `MagicMock` import on line 6 of `test_silence.py`.

### Performance note

Each retry re-decodes the audio from disk via `librosa.load()`. With 4 retry levels that's up to 5 full audio decodes. For long recitations (30+ minutes) this could be noticeable. Not a blocker for v1 but worth keeping in mind.
