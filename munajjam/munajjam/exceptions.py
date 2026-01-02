"""
Custom exceptions for Munajjam library.

All exceptions inherit from MunajjamError for easy catching of library-specific errors.
"""

from typing import Any


class MunajjamError(Exception):
    """Base exception for all Munajjam errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class TranscriptionError(MunajjamError):
    """Raised when audio transcription fails."""

    def __init__(
        self,
        message: str,
        audio_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if audio_path:
            ctx["audio_path"] = audio_path
        super().__init__(message, ctx)
        self.audio_path = audio_path


class AlignmentError(MunajjamError):
    """Raised when ayah alignment fails."""

    def __init__(
        self,
        message: str,
        surah_id: int | None = None,
        ayah_number: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if surah_id is not None:
            ctx["surah_id"] = surah_id
        if ayah_number is not None:
            ctx["ayah_number"] = ayah_number
        super().__init__(message, ctx)
        self.surah_id = surah_id
        self.ayah_number = ayah_number


class ConfigurationError(MunajjamError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        setting_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if setting_name:
            ctx["setting"] = setting_name
        super().__init__(message, ctx)
        self.setting_name = setting_name


class ModelNotLoadedError(TranscriptionError):
    """Raised when attempting to transcribe without a loaded model."""

    def __init__(self, message: str = "Model not loaded. Call load() first.") -> None:
        super().__init__(message)


class AudioFileError(TranscriptionError):
    """Raised when audio file cannot be read or is invalid."""

    def __init__(self, audio_path: str, reason: str | None = None) -> None:
        message = f"Cannot read audio file: {audio_path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, audio_path=audio_path)


class QuranDataError(MunajjamError):
    """Raised when Quran reference data cannot be loaded."""

    def __init__(self, message: str = "Failed to load Quran reference data.") -> None:
        super().__init__(message)

