# Contributing to Munajjam

Thank you for your interest in contributing to Munajjam!

## Getting Started

### Prerequisites

- Python 3.10+
- FFmpeg (for audio processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/Itqan-community/munajjam.git
cd munajjam/munajjam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Setting Up Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically run linting and type checks before each commit. This helps catch issues early and keeps the codebase consistent.

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run hooks against all files
pre-commit run --all-files
```

Once installed, the following checks will run automatically on every `git commit`:

- **Ruff** — linting and auto-fixing (import sorting, style, common bugs)
- **Ruff Format** — code formatting
- **Mypy** — static type checking
- **General hooks** — trailing whitespace, end-of-file fixer, YAML validation, large file checks, merge conflict detection

If any hook fails, the commit will be blocked until the issues are resolved. Many issues (like formatting and import sorting) are fixed automatically — just re-stage the changes and commit again.

## Running Tests

```bash
pytest
```

To run specific tests:

```bash
pytest tests/unit/test_arabic.py
pytest tests/unit -v
```

## Submitting Changes

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Run tests to ensure everything passes
5. Submit a pull request

### Code Style

- Use type hints for all functions
- Follow PEP 8 conventions
- Run `ruff check` before submitting
- Pre-commit hooks will enforce style automatically if installed

## Community

- [Discord](https://discord.gg/24CskUbuuB)
- [Email](mailto:connect@itqan.dev)
- [GitHub Issues](https://github.com/Itqan-community/munajjam/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
