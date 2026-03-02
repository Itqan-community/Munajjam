## Review: feat: add CLI entry point (munajjam command)

Well-structured CLI with clean subcommands and lazy imports. A few bugs need fixing.

All 111 tests pass. 22 new CLI tests cover parser creation, argument parsing, surah inference, result formatting, and error paths.

### `--model` flag is accepted but ignored

Both `align` and `batch` subcommands define a `--model` flag (lines 74-79, 119-124), but neither `cmd_align` nor `cmd_batch` passes `args.model` to `WhisperTranscriber()`. The constructor accepts `model_id` as its first parameter. Users would set `--model` expecting it to work and it would silently do nothing.

### `_infer_surah_number` produces wrong results

The function joins every digit character in the filename:

```python
digits = "".join(c for c in stem if c.isdigit())
```

So `surah_1_v2.mp3` produces `"12"` (surah 12, Al-Yusuf) instead of surah 1 (Al-Fatiha). `reciter_3_v5.mp3` would silently give surah 35 instead of 3. Silent wrong results are worse than errors.

### No `--surah` range validation

`--surah 0` or `--surah 200` is accepted without complaint and passed to `load_surah_ayahs()`, which raises an unhandled `ValueError`. The user gets a raw traceback instead of a clean error message.

### Status message goes to stdout

In `_write_output` (line 178), the "Results written to..." message prints to stdout. Every other status message in the file uses `file=sys.stderr`. This one breaks piped workflows like `munajjam align ... -o file | jq`.

### Smaller things

- No `__main__.py` for `python -m munajjam` support
- No happy-path tests for `cmd_align`/`cmd_batch` (only error paths are tested)
- Unused `patch` import in `test_cli.py`
- `hasattr(r.ayah, "text")` guard on line 156 is unnecessary since `Ayah` is a Pydantic model where `text` is required
- `batch` mode has no way to explicitly set the surah number - relies entirely on filename inference
- CSV format omits the text column unlike JSON and text formats
