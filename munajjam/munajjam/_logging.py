"""
Structured logging utilities for Munajjam library.

Provides a configured logger and helper functions for consistent logging.
"""

import logging
import sys
from typing import Optional


# Default format for Munajjam logs
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "munajjam") -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (default: "munajjam")

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    stream: Optional[object] = None,
) -> logging.Logger:
    """
    Configure logging for the Munajjam library.

    Args:
        level: Logging level (default: INFO)
        format_string: Log format string (default: DEFAULT_FORMAT)
        date_format: Date format string (default: DEFAULT_DATE_FORMAT)
        stream: Output stream (default: sys.stderr)

    Returns:
        Configured root logger for munajjam
    """
    logger = logging.getLogger("munajjam")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        format_string or DEFAULT_FORMAT,
        datefmt=date_format or DEFAULT_DATE_FORMAT,
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def enable_debug_logging() -> None:
    """Enable debug-level logging for the Munajjam library."""
    configure_logging(level=logging.DEBUG)


def disable_logging() -> None:
    """Disable all Munajjam logging."""
    logger = logging.getLogger("munajjam")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())


# Create default logger
_logger = get_logger()


def log_transcription_start(audio_path: str, surah_id: int) -> None:
    """Log transcription start event."""
    _logger.info(f"Starting transcription: {audio_path} (Surah {surah_id})")


def log_transcription_complete(segment_count: int, duration: float) -> None:
    """Log transcription complete event."""
    _logger.info(f"Transcription complete: {segment_count} segments in {duration:.1f}s")


def log_alignment_start(surah_id: int, total_ayahs: int) -> None:
    """Log alignment start event."""
    _logger.info(f"Starting alignment: Surah {surah_id} ({total_ayahs} ayahs)")


def log_alignment_complete(aligned: int, total: int, duration: float) -> None:
    """Log alignment complete event."""
    _logger.info(f"Alignment complete: {aligned}/{total} ayahs in {duration:.1f}s")


def log_ayah_aligned(
    surah_id: int,
    ayah_number: int,
    similarity: float,
    duration: float,
) -> None:
    """Log individual ayah alignment."""
    _logger.debug(
        f"Aligned Surah {surah_id} Ayah {ayah_number}: "
        f"similarity={similarity:.2f}, duration={duration:.2f}s"
    )


def log_warning(message: str, **context) -> None:
    """Log a warning with optional context."""
    if context:
        ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
        _logger.warning(f"{message} ({ctx_str})")
    else:
        _logger.warning(message)


def log_error(message: str, exc_info: bool = False, **context) -> None:
    """Log an error with optional context and exception info."""
    if context:
        ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
        _logger.error(f"{message} ({ctx_str})", exc_info=exc_info)
    else:
        _logger.error(message, exc_info=exc_info)

