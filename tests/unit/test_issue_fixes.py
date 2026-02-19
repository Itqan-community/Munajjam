"""
Regression tests for:
  - Bug #49: Fix example script being auto-collected by pytest
  - Bug #55: Fix broken examples referencing removed APIs

Place this file in:
    tests/unit/test_issue_fixes.py

Then run:
    pytest tests/unit/test_issue_fixes.py -v
"""

import ast
import inspect
import os
import re
import pytest

try:
    from munajjam.core import align as _align_fn
    _align_importable = True
except ImportError:
    _align_importable = False
    _align_fn = None

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

# Root of the repo — tests run from inside the repo directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo_path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def read_source(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────
# Bug #49 — test_alignment.py must NOT exist
# ─────────────────────────────────────────────────────────────

class TestBug49:
    """
    Bug #49: Fix example script being auto-collected by pytest.

    Problem: munajjam/examples/test_alignment.py had a name starting with test_
    and contained functions starting with test_, so pytest was collecting it
    as a test module and failing to run it.

    Fix: Rename the file to alignment_demo.py.
    """

    BAD_FILE  = repo_path("munajjam", "examples", "test_alignment.py")
    GOOD_FILE = repo_path("munajjam", "examples", "alignment_demo.py")

    def test_old_filename_does_not_exist(self):
        """test_alignment.py must have been deleted or renamed."""
        assert not os.path.exists(self.BAD_FILE), (
            f"File '{self.BAD_FILE}' must be deleted or renamed.\n"
            "pytest will auto-collect it due to its name and fail."
        )

    def test_renamed_file_exists(self):
        """alignment_demo.py must exist after the rename."""
        assert os.path.exists(self.GOOD_FILE), (
            f"File '{self.GOOD_FILE}' does not exist.\n"
            "Expected a rename from test_alignment.py to alignment_demo.py."
        )

    def test_renamed_file_is_valid_python(self):
        """alignment_demo.py must have valid Python syntax."""
        source = read_source(self.GOOD_FILE)
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in alignment_demo.py: {e}")

    def test_pytest_would_not_collect_renamed_file(self):
        """
        The new filename must not start with test_ so that pytest does not
        collect it based on the python_files = test_*.py rule in pytest.ini.
        """
        filename = os.path.basename(self.GOOD_FILE)
        assert not filename.startswith("test_"), (
            f"File '{filename}' must not start with test_ to prevent pytest from collecting it."
        )


# ─────────────────────────────────────────────────────────────
# Bug #55 — align() API signature + examples use it correctly
# ─────────────────────────────────────────────────────────────

class TestBug55:
    """
    Bug #55: Fix broken examples referencing removed APIs.

    Problem: Examples were calling:
        align(segments, ayahs)

    But the current API requires:
        align(audio_path, segments, ayahs)

    Fix: Update the call in every affected example file.
    """

    BASIC_USAGE = repo_path("munajjam", "examples", "basic_usage.py")
    ALIGN_DEMO  = repo_path("munajjam", "examples", "alignment_demo.py")

    # ── The API itself ────────────────────────────────────────

    def test_align_function_exists(self):
        assert _align_importable and callable(_align_fn)

    def test_align_first_param_is_audio_path(self):
        assert _align_importable
        params = list(inspect.signature(_align_fn).parameters.keys())
        assert params[0] == "audio_path", (
            f"First parameter of align() is '{params[0]}' instead of 'audio_path'.\n"
            f"Full signature: {params}"
        )

    def test_align_second_param_is_segments(self):
        """Second parameter of align() must be segments."""
        from munajjam.core import align
        params = list(inspect.signature(align).parameters.keys())
        assert params[1] == "segments", (
            f"Second parameter of align() is '{params[1]}' instead of 'segments'."
        )

    def test_align_third_param_is_ayahs(self):
        """Third parameter of align() must be ayahs."""
        from munajjam.core import align
        params = list(inspect.signature(align).parameters.keys())
        assert params[2] == "ayahs", (
            f"Third parameter of align() is '{params[2]}' instead of 'ayahs'."
        )

    # ── basic_usage.py ────────────────────────────────────────

    def test_basic_usage_exists(self):
        """munajjam/examples/basic_usage.py must exist."""
        assert os.path.exists(self.BASIC_USAGE), (
            f"File '{self.BASIC_USAGE}' does not exist."
        )

    def test_basic_usage_no_broken_align_call(self):
        """basic_usage.py must not call align(segments, ...) without audio_path."""
        source = read_source(self.BASIC_USAGE)
        assert "align(segments," not in source, (
            "basic_usage.py must use align(audio_path, segments, ayahs) "
            "not align(segments, ayahs)."
        )

    def test_basic_usage_has_correct_align_call(self):
        """basic_usage.py must use align(audio_path, ...) with the correct signature."""
        source = read_source(self.BASIC_USAGE)
        assert "align(audio_path," in source, (
            "basic_usage.py does not contain align(audio_path, ...) — "
            "make sure the call was updated."
        )

    def test_basic_usage_is_valid_python(self):
        """basic_usage.py must have valid Python syntax."""
        source = read_source(self.BASIC_USAGE)
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in basic_usage.py: {e}")

    # ── alignment_demo.py ─────────────────────────────────────

    def test_alignment_demo_no_broken_align_call(self):
        """
        alignment_demo.py must not call the convenience align() without audio_path.

        Note: aligner.align(segments, ...) is a valid method call and is not a problem.
        We only check that there is no standalone align(segments, ...) without an object.
        """
        if not os.path.exists(self.ALIGN_DEMO):
            pytest.skip("alignment_demo.py not found — run the Bug #49 fix first.")

        source = read_source(self.ALIGN_DEMO)
        # A standalone align( is one not preceded by . (method call)
        broken_calls = re.findall(r'(?<!\.)(?<!\w)align\(segments,', source)
        assert not broken_calls, (
            "alignment_demo.py contains a standalone align(segments, ...) without audio_path.\n"
            "It must be align(audio_path, segments, ayahs)."
        )

    def test_alignment_demo_is_valid_python(self):
        """alignment_demo.py must have valid Python syntax."""
        if not os.path.exists(self.ALIGN_DEMO):
            pytest.skip("alignment_demo.py not found — run the Bug #49 fix first.")

        source = read_source(self.ALIGN_DEMO)
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in alignment_demo.py: {e}")

    # ── Imports in examples are correct ──────────────────────

    def test_basic_usage_imports_align_from_core(self):
        """basic_usage.py must import align from munajjam.core."""
        source = read_source(self.BASIC_USAGE)
        assert re.search(r'from munajjam\.core import[^\n]*\balign\b', source), (
            "basic_usage.py must contain: from munajjam.core import align"
        )

    def test_munajjam_core_align_importable(self):
        """from munajjam.core import align must work without errors."""
        try:
            from munajjam.core import align  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import align from munajjam.core: {e}")