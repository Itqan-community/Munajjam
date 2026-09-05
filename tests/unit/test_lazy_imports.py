"""
Tests to verify that the lazy backend loading architecture works correctly.

The critical acceptance criterion for Issue #105 is that selecting Deepgram
mode must NOT import torch or whisperx. These tests verify this by running
isolated subprocess checks.
"""

import os
import subprocess
import sys


class TestLazyBackendImports:
    """Verify that importing whisperFactory does not eagerly load heavy backends."""

    def test_whisper_factory_import_does_not_load_torch(self) -> None:
        """Importing whisperFactory should not trigger torch import."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from munajjam.transcription.whisperFactory import "
                    "WhisperFactory, WhisperBackend; "
                    "assert 'torch' not in sys.modules, "
                    "'torch was imported just by importing whisperFactory!'; "
                    "assert 'whisperx' not in sys.modules, "
                    "'whisperx was imported just by importing whisperFactory!'; "
                    "print('PASS: No torch or whisperx loaded on import')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Lazy import test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_deepgram_backend_does_not_load_torch(self) -> None:
        """Creating a DeepgramTranscriber must not import torch or whisperx."""
        env = {**os.environ, "MUNAJJAM_DEEPGRAM_API_KEY": "test-key-for-import-check"}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from munajjam.transcription.whisperFactory import "
                    "WhisperFactory, WhisperBackend; "
                    "factory = WhisperFactory(); "
                    "transcriber = factory.create_whisper("
                    "backend=WhisperBackend.DEEPGRAM); "
                    "assert 'torch' not in sys.modules, "
                    "'torch was imported in Deepgram mode!'; "
                    "assert 'whisperx' not in sys.modules, "
                    "'whisperx was imported in Deepgram mode!'; "
                    "print('PASS: Deepgram mode loaded without torch/whisperx')"
                ),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Deepgram torch-bypass test failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_whisper_backend_enum_has_deepgram(self) -> None:
        """WhisperBackend enum must include DEEPGRAM value."""
        from munajjam.transcription.whisperFactory import WhisperBackend

        assert hasattr(WhisperBackend, "DEEPGRAM")
        assert WhisperBackend.DEEPGRAM.value == "deepgram"

    def test_server_import_does_not_load_torch(self) -> None:
        """Importing server.py itself must not load torch or whisperx into sys.modules."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import server; "
                    "assert 'torch' not in sys.modules, "
                    "'torch was imported on server.py import!'; "
                    "assert 'whisperx' not in sys.modules, "
                    "'whisperx was imported on server.py import!'; "
                    "print('PASS: server.py imported without torch/whisperx')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Server import isolation test failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_server_deepgram_transcriber_resolution_does_not_load_torch(self) -> None:
        """Resolving Deepgram transcriber via server helper must not load torch."""
        env = {**os.environ, "MUNAJJAM_DEEPGRAM_API_KEY": "test-key-for-import-check"}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from server import _get_transcriber; "
                    "from munajjam.transcription.whisperFactory import WhisperBackend; "
                    "t = _get_transcriber(WhisperBackend.DEEPGRAM); "
                    "assert 'torch' not in sys.modules, "
                    "'torch was imported during server Deepgram transcriber resolution!'; "
                    "assert 'whisperx' not in sys.modules, "
                    "'whisperx was imported during server Deepgram transcriber resolution!'; "
                    "print('PASS: server Deepgram transcriber resolved without torch/whisperx')"
                ),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Server Deepgram resolution test failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

