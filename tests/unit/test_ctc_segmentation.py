from pathlib import Path

import pytest
from munajjam.transcription.ctc_segmentation import FastConformerCTCTranscriber


def test_load_vocab_returns_tokens_and_blank_id(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("a\nb\nc\n<blk>\n", encoding="utf-8")

    transcriber = FastConformerCTCTranscriber(model_id="test-model")

    vocab, blank_id = transcriber._load_vocab(vocab_path)

    assert vocab == ["a", "b", "c", "<blk>"]
    assert blank_id == 3


def test_load_vocab_rejects_missing_file(tmp_path: Path):
    transcriber = FastConformerCTCTranscriber(model_id="test-model")

    with pytest.raises(FileNotFoundError, match="CTC vocabulary not found"):
        transcriber._load_vocab(tmp_path / "missing.txt")


def test_load_vocab_rejects_empty_vocab(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("", encoding="utf-8")

    transcriber = FastConformerCTCTranscriber(model_id="test-model")

    with pytest.raises(ValueError, match="CTC vocabulary is empty"):
        transcriber._load_vocab(vocab_path)


def test_load_vocab_requires_blank_token(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("a\nb\nc\n", encoding="utf-8")

    transcriber = FastConformerCTCTranscriber(model_id="test-model")

    with pytest.raises(ValueError, match=r"does not contain <blk>"):
        transcriber._load_vocab(vocab_path)


def test_normalize_for_ctc_removes_quranic_marks():
    transcriber = FastConformerCTCTranscriber(model_id="test-model")

    result = transcriber._normalize_for_ctc("ٱلْحَمْدُ لِلَّهِۥ")

    assert result == "الحمد لله"


def test_normalize_for_ctc_removes_tatweel_and_small_letters():
    transcriber = FastConformerCTCTranscriber(model_id="test-model")

    result = transcriber._normalize_for_ctc("الرَّحْمَـٰنُ هُۥ هِۦ")

    assert "ـ" not in result
    assert "ۥ" not in result
    assert "ۦ" not in result


def test_normalize_for_ctc_collapses_spaces():
    transcriber = FastConformerCTCTranscriber(model_id="test-model")

    result = transcriber._normalize_for_ctc("  الحمد   لله  ")

    assert result == "الحمد لله"
