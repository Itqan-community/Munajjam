## Review: fix: resolve Bug #49 and Bug #55

The MockTranscriber is well done and the test structure is clean. But the coverage only hits 1 of the 4 example scripts, and the missing ones are the ones that are actually broken right now.

All 103 tests pass (14 new smoke + 89 existing).

### Only covers example 01, and the broken examples are not tested

The examples/ directory has four scripts but only `01_basic_usage.py` is covered. The other three are missing, and I confirmed locally that examples 02 and 03 are already broken against the current API:

**`02_comparing_strategies.py`** - line 74 iterates over `["greedy", "dp", "hybrid", "word_dp"]` and line 87 tries `"ctc_seg"`. Both `word_dp` and `ctc_seg` are no longer valid strategies:

```
>>> Aligner(audio_path='fake.wav', strategy='word_dp')
ValueError: 'word_dp' is not a valid AlignmentStrategy
>>> Aligner(audio_path='fake.wav', strategy='ctc_seg')
ValueError: 'ctc_seg' is not a valid AlignmentStrategy
```

**`03_advanced_configuration.py`** - line 88 passes `ctc_refine=True` to `Aligner()`. That parameter doesn't exist anymore:

```
>>> Aligner(audio_path='fake.wav', ctc_refine=True, energy_snap=True)
TypeError: Aligner.__init__() got an unexpected keyword argument 'ctc_refine'
```

**`04_batch_processing.py`** - imports `get_all_surahs`, uses manual `load()`/`unload()` lifecycle instead of context manager. These still work, but there are no smoke tests to keep them that way.

This is exactly the kind of API drift that smoke tests are supposed to catch. The one example that is tested (01) happens to be the one that still works fine.

### Unused imports

Five imports are never used anywhere in the file: `importlib`, `sys`, `Iterator`, `MagicMock`, `patch`. Looks like the file was scaffolded to do more but that work wasn't finished.

### No `pytest.mark.smoke` marker

`pytest.ini` registers `integration` and `slow` markers but there's no `smoke` marker. The tests live in `tests/smoke/` but you can't selectively run them with `pytest -m smoke`.

### Other things

- `test_aligner_class_available` is redundant. `Aligner` is already imported at line 24 of the file. If that import fails, every test in the file fails. The test adds nothing.
- `TestExampleAlignmentSmoke` says it mirrors `examples/example_alignment.py` but that file doesn't exist in `examples/`. It seems to reference the older `munajjam/examples/test_alignment.py`.
- Branch needs a rebase onto latest main.
