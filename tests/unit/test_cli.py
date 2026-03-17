"""Tests for the munajjam CLI entry point."""

import pytest
from unittest.mock import MagicMock
from munajjam.cli import (
    create_parser,
    main,
    _infer_surah_number,
    _format_results,
    _validate_surah_number,
)


class TestCreateParser:
    """Tests for the argument parser creation."""

    def test_parser_exists(self):
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "munajjam"

    def test_version_flag(self, capsys):
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self, capsys):
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_align_subcommand(self):
        parser = create_parser()
        args = parser.parse_args(["align", "001.mp3"])
        assert args.command == "align"
        assert args.audio_file == "001.mp3"
        assert args.strategy == "auto"
        assert args.format == "json"
        assert args.output is None

    def test_align_with_options(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "align",
                "001.mp3",
                "--surah",
                "1",
                "--strategy",
                "greedy",
                "--output",
                "output.json",
                "--format",
                "csv",
            ]
        )
        assert args.surah == 1
        assert args.strategy == "greedy"
        assert args.output == "output.json"
        assert args.format == "csv"

    def test_batch_subcommand(self):
        parser = create_parser()
        args = parser.parse_args(["batch", "/path/to/audio"])
        assert args.command == "batch"
        assert args.directory == "/path/to/audio"
        assert args.pattern == "*.mp3"
        assert args.format == "json"

    def test_batch_with_options(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "batch",
                "/path/to/audio",
                "--pattern",
                "*.wav",
                "--output-dir",
                "/output",
                "--format",
                "text",
                "--strategy",
                "dp",
            ]
        )
        assert args.pattern == "*.wav"
        assert args.output_dir == "/output"
        assert args.format == "text"
        assert args.strategy == "dp"

    def test_invalid_strategy(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["align", "001.mp3", "--strategy", "invalid"])

    def test_invalid_format(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["align", "001.mp3", "--format", "xml"])

    def test_model_flag_removed(self):
        """Ensure --model flag is no longer accepted."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["align", "001.mp3", "--model", "some-model"])


class TestValidateSurahNumber:
    """Tests for surah number validation."""

    def test_valid_surah_min(self):
        _validate_surah_number(1)  # Should not raise

    def test_valid_surah_max(self):
        _validate_surah_number(114)  # Should not raise

    def test_valid_surah_middle(self):
        _validate_surah_number(36)  # Should not raise

    def test_invalid_surah_zero(self):
        with pytest.raises(ValueError, match="Invalid surah number: 0"):
            _validate_surah_number(0)

    def test_invalid_surah_negative(self):
        with pytest.raises(ValueError, match="Invalid surah number: -1"):
            _validate_surah_number(-1)

    def test_invalid_surah_too_high(self):
        with pytest.raises(ValueError, match="Invalid surah number: 200"):
            _validate_surah_number(200)

    def test_invalid_surah_115(self):
        with pytest.raises(ValueError, match="Invalid surah number: 115"):
            _validate_surah_number(115)


class TestInferSurahNumber:
    """Tests for surah number inference from filenames."""

    def test_simple_number(self):
        assert _infer_surah_number("001.mp3") == 1

    def test_three_digit_number(self):
        assert _infer_surah_number("114.mp3") == 114

    def test_with_prefix(self):
        assert _infer_surah_number("surah_001.mp3") == 1

    def test_two_digit(self):
        assert _infer_surah_number("36.mp3") == 36

    def test_invalid_number(self):
        with pytest.raises(ValueError, match="Cannot infer surah number"):
            _infer_surah_number("no_number.mp3")

    def test_number_out_of_range(self):
        with pytest.raises(ValueError, match="Cannot infer surah number"):
            _infer_surah_number("200.mp3")

    def test_zero(self):
        with pytest.raises(ValueError, match="Cannot infer surah number"):
            _infer_surah_number("000.mp3")

    def test_filename_with_multiple_numbers(self):
        """Regression: 'surah_1_v2.mp3' should infer surah 1, not 12."""
        assert _infer_surah_number("surah_1_v2.mp3") == 1

    def test_filename_reciter_prefix(self):
        """Regression: 'reciter_3_v5.mp3' should infer surah 3, not 35."""
        assert _infer_surah_number("reciter_3_v5.mp3") == 3

    def test_filename_with_path(self):
        assert _infer_surah_number("/path/to/audio/001.mp3") == 1


class TestFormatResults:
    """Tests for result formatting."""

    def _make_mock_result(self, ayah_num, start, end, text="بسم الله"):
        result = MagicMock()
        result.ayah.ayah_number = ayah_num
        result.ayah.text = text
        result.start_time = start
        result.end_time = end
        return result

    def test_json_format(self):
        results = [self._make_mock_result(1, 5.62, 9.57, "بسم الله")]
        output = _format_results(results, "json")
        import json

        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["ayah_number"] == 1
        assert data[0]["start_time"] == 5.62
        assert data[0]["end_time"] == 9.57
        assert data[0]["text"] == "بسم الله"

    def test_json_format_preserves_arabic(self):
        """Ensure Arabic text is not escaped in JSON output."""
        results = [self._make_mock_result(1, 0.0, 1.0, "بسم الله")]
        output = _format_results(results, "json")
        assert "بسم الله" in output

    def test_csv_format_includes_text(self):
        """CSV format should include the text column."""
        results = [
            self._make_mock_result(1, 5.62, 9.57, "بسم الله"),
            self._make_mock_result(2, 10.51, 14.72, "الحمد لله"),
        ]
        output = _format_results(results, "csv")
        lines = output.strip().split("\n")
        assert lines[0] == "ayah_number,start_time,end_time,text"
        assert "5.62" in lines[1]
        assert "بسم الله" in lines[1]

    def test_text_format(self):
        results = [self._make_mock_result(1, 5.62, 9.57)]
        output = _format_results(results, "text")
        assert "Ayah 1: 5.62s - 9.57s" in output


class TestMainFunction:
    """Tests for the main CLI entry point."""

    def test_no_command_shows_help(self):
        result = main([])
        assert result == 0

    def test_align_missing_file(self):
        result = main(["align", "nonexistent.mp3"])
        assert result == 1

    def test_batch_missing_directory(self):
        result = main(["batch", "/nonexistent/dir"])
        assert result == 1

    def test_align_invalid_surah_zero(self, capsys):
        """Surah 0 should produce a clean error, not a traceback."""
        result = main(["align", "nonexistent.mp3", "--surah", "0"])
        assert result == 1

    def test_align_invalid_surah_200(self, capsys):
        """Surah 200 should produce a clean error, not a traceback."""
        result = main(["align", "nonexistent.mp3", "--surah", "200"])
        assert result == 1


class TestWriteOutput:
    """Tests for output writing."""

    def test_write_to_stderr(self, capsys, tmp_path):
        """Status message 'Results written to...' should go to stderr, not stdout."""
        from munajjam.cli import _write_output

        output_file = tmp_path / "output.json"
        _write_output("test content", str(output_file))

        captured = capsys.readouterr()
        # stdout should be empty
        assert captured.out == ""
        # stderr should contain the status message
        assert "Results written to" in captured.err

    def test_write_to_stdout(self, capsys):
        """When no output path, content should go to stdout."""
        from munajjam.cli import _write_output

        _write_output("test content", None)

        captured = capsys.readouterr()
        assert "test content" in captured.out
