"""
Configuration management for Munajjam library.

Uses Pydantic Settings for type-safe configuration with environment variable support.
All settings can be overridden via environment variables with the MUNAJJAM_ prefix.
"""

from typing import Literal
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MunajjamSettings(BaseSettings):
    """
    Configuration settings for Munajjam library.

    All settings can be overridden via environment variables with MUNAJJAM_ prefix.

    Example:
        export MUNAJJAM_MODEL_ID="tarteel-ai/whisper-base-ar-quran"
        export MUNAJJAM_DEVICE="cuda"
        export MUNAJJAM_SIMILARITY_THRESHOLD="0.7"
    """

    model_config = SettingsConfigDict(
        env_prefix="MUNAJJAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ============ Model Settings ============

    model_id: str = Field(
        default="tarteel-ai/whisper-base-ar-quran",
        description="HuggingFace model ID for Whisper transcription",
    )

    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device for model inference (auto, cpu, cuda, mps)",
    )

    model_type: Literal["transformers", "faster-whisper"] = Field(
        default="transformers",
        description="Model backend type",
    )

    hf_inference_endpoint: str | None = Field(
        default=None,
        description="HuggingFace Inference Endpoint URL",
    )

    hf_api_token: str | None = Field(
        default=None,
        description="HuggingFace API token for inference endpoint",
    )

    # ============ Audio Processing ============

    silence_threshold_db: int = Field(
        default=-30,
        description="Silence detection threshold in dB",
        ge=-60,
        le=0,
    )

    min_silence_ms: int = Field(
        default=300,
        description="Minimum silence duration in milliseconds",
        ge=100,
        le=2000,
    )

    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate for processing",
    )

    # ============ Alignment Settings ============

    similarity_threshold: float = Field(
        default=0.6,
        description="Minimum similarity score for ayah matching",
        ge=0.0,
        le=1.0,
    )

    n_check_words: int = Field(
        default=3,
        description="Number of words to check for boundary detection",
        ge=1,
        le=10,
    )

    buffer_seconds: float = Field(
        default=0.3,
        description="Buffer duration in seconds to add around ayah boundaries",
        ge=0.0,
        le=1.0,
    )

    min_silence_gap: float = Field(
        default=0.18,
        description="Minimum silence gap for ayah boundary detection (seconds)",
        ge=0.0,
        le=1.0,
    )

    coverage_threshold: float = Field(
        default=0.7,
        description="Minimum word coverage ratio for accepting alignment",
        ge=0.0,
        le=1.0,
    )

    # ============ Output Settings ============

    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files",
    )

    # ============ Processing Settings ============

    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed operations",
        ge=1,
        le=10,
    )

    max_concurrent_workers: int = Field(
        default=3,
        description="Maximum concurrent workers for batch processing",
        ge=1,
        le=16,
    )

    # ============ Validators ============

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Keep 'auto' as-is; resolution happens at runtime."""
        return v

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    def get_resolved_device(self) -> str:
        """
        Resolve 'auto' device to the best available option.

        Returns:
            str: The resolved device (cuda, mps, or cpu)
        """
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"


# Default settings instance
_default_settings: MunajjamSettings | None = None


def get_settings() -> MunajjamSettings:
    """
    Get the default settings instance (lazily created).

    Returns:
        MunajjamSettings: The default settings
    """
    global _default_settings
    if _default_settings is None:
        _default_settings = MunajjamSettings()
    return _default_settings


def configure(**kwargs) -> MunajjamSettings:
    """
    Create and set new default settings.

    Args:
        **kwargs: Settings to override

    Returns:
        MunajjamSettings: The new settings instance
    """
    global _default_settings
    _default_settings = MunajjamSettings(**kwargs)
    return _default_settings

