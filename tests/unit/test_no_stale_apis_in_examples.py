"""Tests that examples do not reference removed public APIs.

After the alignment-v2 refactor, the following public APIs were removed:
- AlignmentStrategy.CTC_REFINE / strategy="ctc_refine"
- AlignmentStrategy.WORD_DP / strategy="word_dp"
- AlignmentStrategy.CTC_SEG / strategy="ctc_seg"
- ctc_refine= keyword argument to Aligner()

Examples and their README must only reference the four valid strategies:
greedy, dp, hybrid, auto.
"""

import re
from pathlib import Path

EXAMPLES_DIR: Path = Path(__file__).resolve().parent.parent.parent / "examples"

# Removed PUBLIC APIs that should NOT appear in examples
REMOVED_API_PATTERNS: list[str] = [
    r"\bword_dp\b",
    r"\bctc_seg\b",
    r"\bctc_refine\b",
    r"\bWORD_DP\b",
    r"\bCTC_SEG\b",
    r"\bCTC_REFINE\b",
    r"\brefine_low_confidence_zones_with_ctc\b",
]


def _discover_example_files() -> list[str]:
    """Dynamically discover all .py and .md files in the examples directory."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        str(p.relative_to(EXAMPLES_DIR))
        for p in EXAMPLES_DIR.rglob("*")
        if p.is_file() and p.suffix in (".py", ".md")
    )


class TestNoStaleApisInExamples:
    """Ensure examples don't reference removed public APIs."""

    def test_examples_dir_exists(self) -> None:
        assert EXAMPLES_DIR.is_dir(), f"Examples directory not found: {EXAMPLES_DIR}"

    def test_no_removed_strategies_in_examples(self) -> None:
        """No example file should reference removed strategy names."""
        example_files = _discover_example_files()
        assert example_files, "No example files found"

        violations: list[str] = []

        for filename in example_files:
            filepath = EXAMPLES_DIR / filename
            content = filepath.read_text()
            for pattern in REMOVED_API_PATTERNS:
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    violations.append(f"{filename}:{line_num}: found '{match.group()}'")

        assert not violations, (
            f"Found {len(violations)} references to removed APIs in examples:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_strategies_list_only_valid(self) -> None:
        """02_comparing_strategies.py should only list valid strategies."""
        filepath = EXAMPLES_DIR / "02_comparing_strategies.py"
        content = filepath.read_text()

        valid_strategies = {"greedy", "dp", "hybrid", "auto"}

        # Find the strategies list assignment
        match = re.search(r"strategies\s*=\s*\[([^\]]+)\]", content)
        assert match, "Could not find strategies list in 02_comparing_strategies.py"

        # Extract strategy names from the list
        strategies_str = match.group(1)
        found_strategies = set(re.findall(r'["\'](\w+)["\']', strategies_str))

        invalid = found_strategies - valid_strategies
        missing = valid_strategies - found_strategies
        assert not invalid, (
            f"02_comparing_strategies.py contains invalid strategies: {invalid}. "
            f"Valid strategies are: {valid_strategies}"
        )
        assert not missing, (
            f"02_comparing_strategies.py is missing strategies: {missing}. "
            f"Expected exactly: {valid_strategies}"
        )
