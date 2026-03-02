## Review: feat: add standardized JSON output formatter

Nice schema design overall - the `metadata` + `results` separation makes sense and the Pydantic models are well thought out. But there are some problems blocking this from merging.

### Circular import breaks the package

This is the big one. `__init__.py` line 30 imports from `formatters`, and `formatters.py` line 33 does `from munajjam import __version__`. But `__version__` isn't defined until `__init__.py` line 46, after the formatter import. This creates a circular import that crashes on any `import munajjam`:

```
ImportError: cannot import name '__version__' from partially initialized module 'munajjam'
(most likely due to a circular import)
```

I confirmed this locally - the entire test suite fails to even start because `conftest.py` can't import the package. None of the 22 new tests can run.

### `ensure_ascii` parameter does nothing

`to_json()` on line 149 accepts `ensure_ascii` but never passes it to `model_dump_json()` on line 159. Passing `ensure_ascii=True` has no effect - Arabic text still appears unescaped. The parameter is dead code.

### Duration rounding can be inconsistent

In `_format_single_result()`, `start_time`, `end_time`, and `duration` are each rounded independently from the raw values (lines 197-199). Since rounding doesn't distribute over subtraction, you can end up with output where `duration != end_time - start_time`. For example: `start=1.125, end=6.175` produces `start_time=1.12, end_time=6.18, duration=5.05`, but `6.18 - 1.12 = 5.06`.

### `basic_usage.py` not updated

Issue #52 lists as an acceptance criterion that `basic_usage.py` should use the new formatter instead of hand-rolling its own JSON. The example still builds its own dicts with different field names (`sura_id` vs `surah_id`, `corrected_text` vs `original_text`, `start`/`end` vs `start_time`/`end_time`). That's the exact inconsistency #52 is about.

### Smaller things

- `import json` on line 27 is unused - nothing in the file uses it
- `Path` aliased as `P` inside `to_file()` for no reason - just use `Path`
- `to_file()` only accepts `str`, should also take `Path` objects
