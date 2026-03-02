## Review: Add pre-commit hooks configuration

This PR addresses issue #50 but has several critical problems that make it non-functional. There is also a competing PR #62 by ahmed-alramah that addresses the same issue.

### Placeholder versions - config won't work

Both hooks use `rev: v0.0.0` with comments saying "Replace with the latest version." That tag doesn't exist in either repository. Running `pre-commit install && pre-commit run --all-files` would fail immediately. Issue #50 explicitly requires "All hooks pass cleanly on the current codebase" - this doesn't meet that.

### Wrong ruff repository

The PR uses `https://github.com/pre-commit/mirrors-ruff`, which is the old deprecated mirror. The official repo is `https://github.com/astral-sh/ruff-pre-commit`.

### Missing ruff-format hook

The CI pipeline runs both `ruff check .` and `ruff format --check .`. This PR only configures linting (`ruff`), not formatting (`ruff-format`). Pre-commit wouldn't catch formatting issues that CI rejects.

### Bare mypy hook

The mypy hook has no `additional_dependencies` (the project uses pydantic), no `args` (CI uses `--ignore-missing-imports`), and no `pass_filenames` config. It would likely fail or give false positives.

### CONTRIBUTING.md destroyed

This is the worst part. The original `CONTRIBUTING.md` was 62 lines covering prerequisites, setup, testing, submission workflow, code style, and community links. It was entirely deleted and replaced with 5 incomplete lines that end mid-sentence:

```markdown
## Setting Up Pre-commit Hooks

To ensure code quality, set up pre-commit hooks locally:

1. Install pre-commit:
```

Issue #50 says the contributing guide should "mention" pre-commit setup - meaning add a section, not replace the whole file.

### Missing trailing newlines

Both `.pre-commit-config.yaml` and `CONTRIBUTING.md` lack a final newline.

### Note

PR #62 addresses the same issue with correct version pins, the official ruff repo, both ruff + ruff-format hooks, proper mypy config with pydantic dependencies, and preserves the existing CONTRIBUTING.md content. I'd recommend going with that one instead.
