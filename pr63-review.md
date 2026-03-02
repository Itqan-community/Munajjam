## Review: fix: rename test_alignment.py to prevent pytest auto-collection

The rename itself is correct but incomplete - the file's contents still reference the old name everywhere.

All 89 tests pass.

### Internal references not updated

The file was renamed from `test_alignment.py` to `example_alignment.py`, but there are 12 references to `test_alignment.py` still inside it:

- Docstring (lines 6, 9, 10)
- Usage/help print statements (lines 276-278, 281-284, 291, 297)

Someone reading the help output would try to run `python test_alignment.py` and the file wouldn't exist. The module docstring on line 3 also still says "Test script for Munajjam library."

### Was the rename strictly necessary?

The existing `pytest.ini` already constrains test collection to `testpaths = tests`, so this file wouldn't be collected in a normal `pytest` run. That said, the rename is still good practice for IDE test runners and direct `pytest .` invocations.

### Conflicts with PR #59

PR #59 by MohamedAboAlaa does the same rename but to `alignment_demo.py` instead, and also fixes the `basic_usage.py` API call (Bug #55). Only one of these can be merged. PR #59 is more comprehensive since it addresses two bugs, though it also has its own issues (see that review).
