## Review: feat: add pre-commit hooks configuration

This looks good. Clean config, correct repos, proper version pins, and the CONTRIBUTING.md update is additive without destroying existing content.

All 89 tests pass (expected for a config-only change).

### What's here

- Ruff linting with `--fix` and ruff-format as separate hooks (matches what CI runs)
- Mypy with correct `additional_dependencies` for pydantic and `--config-file=munajjam/pyproject.toml`
- Five general-purpose hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, check-merge-conflict
- CONTRIBUTING.md gets a new "Setting Up Pre-commit Hooks" section appended under "Getting Started"

### Minor suggestions (non-blocking)

- Hook versions are a bit behind (ruff v0.9.6, mypy v1.15.0, pre-commit-hooks v5.0.0). Running `pre-commit autoupdate` before merging would bring them current.
- The general-purpose hooks (`pre-commit-hooks`) could be listed first since they're the cheapest to run. Ruff next, mypy last (most expensive). Faster feedback on trivial issues.
- Could add `pre-commit` to dev dependencies in `pyproject.toml` so it installs with `pip install -e ".[dev]"` instead of needing a separate install step.

### Note on PR #56

PR #56 by loganionian addresses the same issue (#50) but has placeholder versions (`v0.0.0`), uses the deprecated ruff mirror, and destroyed the CONTRIBUTING.md. This PR is the one to go with. PR #56 should be closed.
