# Munajjam Test Suite

This directory contains comprehensive unit and integration tests for the Munajjam library.

## Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Fast unit tests
│   ├── test_arabic.py        # Arabic text normalization tests
│   ├── test_matcher.py       # Similarity calculation tests
│   ├── test_data.py          # Data loading tests
│   ├── test_silence.py       # Silence detection tests
│   ├── test_aligner.py       # Alignment strategy tests
│   └── test_models.py       # Data model tests
├── integration/              # Slower integration tests
│   └── test_full_pipeline.py # End-to-end pipeline tests
└── fixtures/               # Test data fixtures
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run only unit tests (fast):
```bash
pytest -m "not integration"
```

### Run only specific test file:
```bash
pytest tests/unit/test_arabic.py
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage report:
```bash
pytest --cov=munajjam --cov-report=html
```

### Run specific test:
```bash
pytest tests/unit/test_matcher.py::TestSimilarity::test_identical_texts
```

### Skip slow tests:
```bash
pytest -m "not slow"
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Fast, isolated tests
- Test individual functions and classes
- Use mocks for external dependencies
- No model loading or file I/O

### Integration Tests (`tests/integration/`)
- Slower tests that use real data
- Test full pipeline: transcription → alignment
- May load ML models
- Require audio files

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `sample_ayah_text()` - Sample Arabic ayah text
- `normalized_text()` - Normalized version of sample text
- `sample_segments()` - Mock transcribed segments
- `sample_ayahs()` - Mock ayah objects
- `sample_silences()` - Mock silence periods
- `sample_alignment_results()` - Mock alignment results
- `real_surah_1_ayahs()` - Real data from CSV
- `mock_whisper_transcriber()` - Mocked transcriber

## Test Markers

- `@pytest.mark.integration` - Integration tests (slow)
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.unit` - Unit tests (default)

## Best Practices Used

1. **Descriptive test names** - Test names describe what they test
2. **AAA pattern** - Arrange, Act, Assert structure
3. **Independent tests** - Tests don't depend on each other
4. **Fast unit tests** - Unit tests avoid I/O and model loading
5. **Fixtures for setup** - Reusable test data and mocks
6. **Clear assertions** - Specific, meaningful error messages
7. **Edge cases** - Tests handle empty strings, invalid inputs, etc.

## Adding New Tests

1. Create test file in `tests/unit/` or `tests/integration/`
2. Import required modules and fixtures
3. Create test class with `Test` prefix
4. Add test methods with `test_` prefix
5. Use fixtures from `conftest.py`
6. Mark integration tests with `@pytest.mark.integration`
7. Run tests to verify

Example:
```python
import pytest
from munajjam.core import Aligner

class TestMyFeature:
    def test_basic_usage(self, sample_segments, sample_ayahs):
        aligner = Aligner(strategy="hybrid")
        results = aligner.align(sample_segments, sample_ayahs)

        assert len(results) > 0
```

## Coverage

Run tests with coverage to see which code is tested:
```bash
pytest --cov=munajjam --cov-report=term-missing
```

View HTML report:
```bash
open htmlcov/index.html
```

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -r test-requirements.txt
    pytest -m "not integration"  # Only fast tests in CI

- name: Run integration tests
  if: github.event_name == 'push'
  run: |
    pytest  # All tests
```
