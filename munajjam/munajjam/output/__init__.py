"""
JSON output formatter for Munajjam alignment results.

This module provides standardized JSON formatting for alignment results,
ensuring consistent output structure across all consumers.
"""

from munajjam.output.formatter import (
    AlignmentOutput,
    FormattedAyahResult,
    format_alignment_results,
)

__all__ = [
    "AlignmentOutput",
    "FormattedAyahResult",
    "format_alignment_results",
]
