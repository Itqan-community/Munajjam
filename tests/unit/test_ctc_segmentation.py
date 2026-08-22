"""
Unit tests for the FastConformer CTC segmentation pipeline (issue #104).

All tests run offline with mocked inference/tokenizer/audio: no NVIDIA model,
no onnxruntime, no NeMo, no network.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import soundfile as sf

from munajjam.data.quran import load_ayahs, load_surah_ayahs
from munajjam.exceptions import TranscriptionError
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.ctc_segmentation import (
    AudioChunk,
    FastConformerCTCTranscriber,
    QuranicPhonemizerAdapter,
    SileroVADChunker,
    SinglePassChunker,
    _fit_reference_prefix,
    align_words_to_log_probs,
    chunk_local_to_global,
    frames_to_time,
    normalize_quran_text,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _log_softmax_rows(lpz: np.ndarray) -> np.ndarray:
    return lpz - np.log(np.exp(lpz).sum(axis=1, keepdims=True))


class FakeTokenizer:
    """Deterministic CTC reference tokenizer for tests (vocab 1024, unk 0)."""

    vocab_size = 1024
    unk_id = 0

    def __init__(self, word_to_ids: dict[str, list[int]] | None = None) -> None:
        self._word_to_ids = word_to_ids or {}
        self._next_id = 1
        self._fallback: dict[str, list[int]] = {}

    def encode(self, text: str) -> list[int]:
        if text in self._word_to_ids:
            return list(self._word_to_ids[text])
        if text not in self._fallback:
            self._fallback[text] = [self._next_id]
            self._next_id += 1
        return list(self._fallback[text])


class FakeInference:
    """Fake acoustic layer: returns synthetic CTC log-probs with a peak per
    token id in ``1..30`` (blank high everywhere else)."""

    sample_rate = 16000
    vocabulary_size = 1024
    blank_index = 1024
    frame_duration_seconds = 0.08

    def __init__(self, n_frames: int = 70) -> None:
        self._n_frames = n_frames

    def log_probs(self, waveform: np.ndarray) -> np.ndarray:
        lpz = np.full((self._n_frames, 1025), -6.0, dtype=np.float32)
        lpz[:, self.blank_index] = 0.0
        for k in range(1, 31):
            frame = 2 * k
            if frame < self._n_frames:
                lpz[frame, k] = 0.0  # token k peaks at frame 2k (wide spacing)
        return _log_softmax_rows(lpz)

    def unload(self) -> None:
        pass


class SequenceFakeInference(FakeInference):
    """Fake acoustic layer returning a different frame count per call."""

    def __init__(self, frame_counts: list[int]) -> None:
        super().__init__(n_frames=frame_counts[0])
        self._frame_counts = list(frame_counts)
        self._call = 0

    def log_probs(self, waveform: np.ndarray) -> np.ndarray:
        n = self._frame_counts[min(self._call, len(self._frame_counts) - 1)]
        self._call += 1
        lpz = np.full((n, 1025), -6.0, dtype=np.float32)
        lpz[:, self.blank_index] = 0.0
        for k in range(1, 31):
            frame = 2 * k
            if frame < n:
                lpz[frame, k] = 0.0
        return _log_softmax_rows(lpz)


class FailingInference(FakeInference):
    """Fake acoustic layer that fails on the second chunk."""

    def __init__(self) -> None:
        super().__init__(n_frames=6)  # small so chunk 1 can't consume all words
        self._calls = 0

    def log_probs(self, waveform: np.ndarray) -> np.ndarray:
        self._calls += 1
        if self._calls > 1:
            raise TranscriptionError("boom: ONNX inference failed")
        return super().log_probs(waveform)


def _surah_words(surah_id: int) -> list[str]:
    words: list[str] = []
    for ayah in load_surah_ayahs(surah_id):
        words.extend(ayah.text.split())
    return words


def _fake_tokenizer_for_surah(surah_id: int) -> FakeTokenizer:
    mapping = {normalize_quran_text(w): [i + 1] for i, w in enumerate(_surah_words(surah_id))}
    return FakeTokenizer(mapping)


def _make_wav(tmp_path: pytest.TempPathFactory, seconds: float = 1.0) -> str:
    path = tmp_path / "audio.wav"
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype=np.float32), 16000)
    return str(path)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def test_normalize_known_ayah_matches_verified_output() -> None:
    # Verified against the real SentencePiece tokenizer: 0 <unk>, pieces
    # ['▁بس', 'م', '▁الله', '▁الرحمن', '▁الرحيم'].
    assert normalize_quran_text("بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ") == "بسم الله الرحمن الرحيم"


def test_normalize_strips_diacritics_marks_tatweel_and_wasla() -> None:
    text = "بِسْمِۦ ٱللَّهِۖ الرَّحْمَـٰنِ"
    normalized = normalize_quran_text(text)
    assert normalized == "بسم الله الرحمن"
    for ch in normalized:
        assert not (
            0x0610 <= ord(ch) <= 0x061A
            or 0x064B <= ord(ch) <= 0x065F
            or ord(ch) == 0x0670
            or 0x06D6 <= ord(ch) <= 0x06ED
            or ord(ch) == 0x0640
            or ord(ch) == 0x0671
        )


def test_normalize_preserves_uthmani_letters() -> None:
    # ة ى ؤ ئ exist in the model vocabulary as-is and must not be rewritten.
    text = "الْمَلَائِكَةُ الْهُدَى مُؤْمِنٌ شَيْءٌ"
    assert normalize_quran_text(text) == "الملائكة الهدى مؤمن شيء"


def test_normalize_collapses_whitespace_and_handles_empty() -> None:
    assert normalize_quran_text("   بِسْمِ    اللَّهِ  ") == "بسم الله"
    assert normalize_quran_text("") == ""


def test_normalize_full_quran_removes_unrepresentable_chars() -> None:
    # Property test over the whole bundled Quran (6236 ayahs, offline).
    for ayah in load_ayahs():
        normalized = normalize_quran_text(ayah.text)
        assert normalized == " ".join(normalized.split())
        for ch in normalized:
            assert not (
                0x0610 <= ord(ch) <= 0x061A
                or 0x064B <= ord(ch) <= 0x065F
                or ord(ch) == 0x0670
                or 0x06D6 <= ord(ch) <= 0x06ED
                or ord(ch) == 0x0640
                or ord(ch) == 0x0671
            ), f"unrepresentable char U+{ord(ch):04X} in surah {ayah.surah_id}:{ayah.ayah_number}"


# --------------------------------------------------------------------------- #
# Frame -> time and chunk offsets
# --------------------------------------------------------------------------- #


def test_frames_to_time_uses_80ms_frames() -> None:
    assert frames_to_time(0) == 0.0
    assert frames_to_time(5) == 0.4
    assert frames_to_time(25) == 2.0
    arr = frames_to_time(np.array([0, 1, 2], dtype=np.int64))
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, [0.0, 0.08, 0.16])
    assert frames_to_time(5, frame_duration=0.1) == 0.5


def test_chunk_local_to_global_offset() -> None:
    assert chunk_local_to_global(0.0, 10) == 0.8
    assert chunk_local_to_global(3.2, 10) == 4.0
    assert chunk_local_to_global(3.2, 0) == 3.2


# --------------------------------------------------------------------------- #
# Forced alignment
# --------------------------------------------------------------------------- #


def _boosted_lpz(
    peaks: list[tuple[int, int]],
    n_frames: int = 40,
    n_classes: int = 1025,
    blank: int = 1024,
) -> np.ndarray:
    lpz = np.full((n_frames, n_classes), -8.0, dtype=np.float32)
    lpz[:, blank] = 0.0
    for frame, token in peaks:
        lpz[frame, token] = 0.0
    return _log_softmax_rows(lpz)


def test_align_success_returns_monotonic_word_times() -> None:
    lpz = _boosted_lpz([(5, 10), (10, 20), (15, 30)])
    result = align_words_to_log_probs(lpz, ["w1", "w2", "w3"], [[10], [20], [30]], blank_index=1024)
    assert [w.text for w in result] == ["w1", "w2", "w3"]
    assert result[0].start >= 0.0
    for w in result:
        assert w.end >= w.start
        assert 0.0 <= w.probability <= 1.0
    # Monotonic and close to the boosted frames (frame * 0.08 s).
    expected = [0.4, 0.8, 1.2]
    for w, exp in zip(result, expected, strict=True):
        assert abs((w.start + w.end) / 2 - exp) < 0.4


def test_align_repeated_tokens_handles_blank() -> None:
    # Two identical tokens at frames 5 and 7 must be separated by a blank.
    lpz = _boosted_lpz([(5, 10), (7, 10)])
    result = align_words_to_log_probs(lpz, ["w1", "w2"], [[10], [10]], blank_index=1024)
    assert len(result) == 2
    assert result[1].start >= result[0].end
    assert abs(result[0].end - result[1].start) < 0.16


def test_align_uses_blank_index_from_model() -> None:
    # blank_index == vocabulary_size (1024); alignment succeeds with it.
    lpz = _boosted_lpz([(5, 10)])
    result = align_words_to_log_probs(lpz, ["w"], [[10]], blank_index=1024, frame_duration=0.08)
    assert len(result) == 1
    with pytest.raises(TranscriptionError):
        align_words_to_log_probs(lpz, ["w"], [[10]], blank_index=2000)


def test_align_impossible_when_audio_shorter_than_text() -> None:
    lpz = _boosted_lpz([], n_frames=5)
    with pytest.raises(TranscriptionError, match="longer than the audio"):
        align_words_to_log_probs(lpz, ["w1", "w2", "w3"], [[1], [2], [3]], blank_index=1024)


def test_align_rejects_token_outside_class_range() -> None:
    lpz = _boosted_lpz([(5, 10)])
    with pytest.raises(TranscriptionError, match="outside CTC class range"):
        align_words_to_log_probs(lpz, ["w"], [[1025]], blank_index=1024)
    with pytest.raises(TranscriptionError, match="outside CTC class range"):
        align_words_to_log_probs(lpz, ["w"], [[-1]], blank_index=1024)


def test_align_rejects_blank_collision() -> None:
    lpz = _boosted_lpz([(5, 10)])
    with pytest.raises(TranscriptionError, match="blank class"):
        align_words_to_log_probs(lpz, ["w"], [[1024]], blank_index=1024)


def test_align_rejects_empty_token_sequence() -> None:
    lpz = _boosted_lpz([(5, 10)])
    with pytest.raises(TranscriptionError, match="empty sequence"):
        align_words_to_log_probs(lpz, ["w"], [[]], blank_index=1024)


def test_align_rejects_bad_log_probs_shape() -> None:
    with pytest.raises(TranscriptionError, match="log_probs"):
        align_words_to_log_probs(np.zeros(5, dtype=np.float32), ["w"], [[1]], blank_index=1024)
    with pytest.raises(TranscriptionError, match="log_probs"):
        align_words_to_log_probs(
            np.zeros((2, 3, 4), dtype=np.float32), ["w"], [[1]], blank_index=1024
        )


def test_align_min_confidence_fails_low_quality_alignment() -> None:
    # Uniform log-probs: no token is favored, so the alignment is low quality.
    n_classes = 1025
    lpz = np.full((40, n_classes), -np.log(n_classes), dtype=np.float32)
    with pytest.raises(TranscriptionError, match="confidence below threshold"):
        align_words_to_log_probs(lpz, ["w"], [[10]], blank_index=1024, min_confidence=0.5)


# --------------------------------------------------------------------------- #
# Transcriber
# --------------------------------------------------------------------------- #


def test_transcribe_builds_per_ayah_segments(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    tokenizer = _fake_tokenizer_for_surah(1)
    transcriber = FastConformerCTCTranscriber(inference=FakeInference(), tokenizer=tokenizer)
    segments = transcriber.transcribe(audio, surah_id=1)

    assert len(segments) == 7  # surah 1 has 7 ayahs
    assert sum(len(s.words or []) for s in segments) == len(_surah_words(1))

    previous_end = 0.0
    for segment in segments:
        assert isinstance(segment, Segment)
        assert segment.surah_id == 1
        assert segment.type == SegmentType.AYAH
        assert segment.end >= segment.start >= 0.0
        assert segment.start >= previous_end - 1e-6
        previous_end = segment.end
        assert segment.words is not None and len(segment.words) > 0
        assert segment.confidence is not None and 0.0 <= segment.confidence <= 1.0
        for wt in segment.words:
            assert isinstance(wt, WordTimestamp)
            assert wt.end >= wt.start >= 0.0
            assert 0.0 <= wt.probability <= 1.0
    # Last segment ends where the last word ends.
    assert segments[-1].end >= segments[-1].start


def test_transcribe_preserves_canonical_text(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    transcriber = FastConformerCTCTranscriber(
        inference=FakeInference(), tokenizer=_fake_tokenizer_for_surah(1)
    )
    segments = transcriber.transcribe(audio, surah_id=1)
    ayahs = load_surah_ayahs(1)
    for segment, ayah in zip(segments, ayahs, strict=True):
        assert segment.text == ayah.text  # canonical Uthmani text preserved
        assert segment.id == ayah.ayah_number
        expected_words = ayah.text.split()
        assert [wt.word for wt in segment.words or []] == expected_words


def test_transcribe_requires_tokenizer(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    transcriber = FastConformerCTCTranscriber(inference=FakeInference(), tokenizer=None)
    with pytest.raises(TranscriptionError, match="No reference tokenizer"):
        transcriber.transcribe(audio, surah_id=1)


def test_transcribe_fails_on_out_of_vocabulary_word(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    words = _surah_words(1)
    mapping = {normalize_quran_text(w): [i + 1] for i, w in enumerate(words)}
    mapping[normalize_quran_text(words[0])] = [0]  # force <unk>
    transcriber = FastConformerCTCTranscriber(
        inference=FakeInference(), tokenizer=FakeTokenizer(mapping)
    )
    with pytest.raises(TranscriptionError, match="out-of-vocabulary"):
        transcriber.transcribe(audio, surah_id=1)


def test_transcribe_fails_when_audio_too_short(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    # 10 frames fit only ~4 single-token words (cost: 2+2*4<=10); the rest
    # of the surah's ~29 words can never be aligned, so transcribe raises.
    transcriber = FastConformerCTCTranscriber(
        inference=FakeInference(n_frames=10), tokenizer=_fake_tokenizer_for_surah(1)
    )
    with pytest.raises(TranscriptionError, match="not all reference words"):
        transcriber.transcribe(audio, surah_id=1)


def _word_sequence(segments: list[Segment]) -> list[str]:
    return [wt.word for s in segments for wt in s.words or []]


def _two_chunks_chunker(offset_2: float = 5.0, split_at: int = 8000):
    class TwoChunks:
        def chunk(self, waveform, sample_rate):
            yield AudioChunk(waveform=waveform[:split_at], start_seconds=0.0)
            yield AudioChunk(waveform=waveform[split_at:], start_seconds=offset_2)

    return TwoChunks()


def test_transcribe_multi_chunk_no_duplicates_and_global_times(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    reference = _surah_words(1)
    # Chunk 1: 10 frames fits 4 words (cost 2+2*4=10). The remaining ~25
    # words must land in chunk 2 and carry its global offset of 5.0 s.
    transcriber = FastConformerCTCTranscriber(
        inference=SequenceFakeInference([10, 60]),
        tokenizer=_fake_tokenizer_for_surah(1),
        chunker=_two_chunks_chunker(),
    )
    segments = transcriber.transcribe(audio, surah_id=1)

    assert _word_sequence(segments) == reference
    # Words aligned in the second chunk carry its global offset.
    assert any(wt.start >= 5.0 for s in segments for wt in s.words or [])
    # Timestamps monotonic across the chunk boundary.
    all_times = [(wt.start, wt.end) for s in segments for wt in s.words or []]
    for (_, prev_end), (start, _) in zip(all_times, all_times[1:], strict=False):
        assert start >= prev_end - 1e-6


def test_transcribe_multi_chunk_silence_gap_preserved(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    transcriber = FastConformerCTCTranscriber(
        inference=FakeInference(n_frames=60),
        tokenizer=_fake_tokenizer_for_surah(1),
        chunker=_two_chunks_chunker(offset_2=6.0),
    )
    segments = transcriber.transcribe(audio, surah_id=1)
    for s in segments:
        for wt in s.words or []:
            # No word may fall inside the silence gap (4.8s..6.0s); chunk 1
            # words are capped at 60 frames * 0.08s = 4.8s.
            assert wt.end <= 4.8 + 1e-6 or wt.start >= 6.0 - 1e-6


def test_transcribe_words_cross_chunk_boundary_once(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    reference = _surah_words(1)
    # Chunk 1 has 6 frames: fits 2 words (cost 2 + 2*2 = 6); words 2..29 must
    # be aligned in chunk 2 at the global offset 5.0 s, with no duplicates.
    transcriber = FastConformerCTCTranscriber(
        inference=SequenceFakeInference([6, 70]),
        tokenizer=_fake_tokenizer_for_surah(1),
        chunker=_two_chunks_chunker(),
    )
    segments = transcriber.transcribe(audio, surah_id=1)
    words = _word_sequence(segments)
    assert words == reference
    # Words 0..1 land in chunk 1 (< 6 frames * 0.08 s = 0.48 s); the rest
    # carry the second chunk's global offset.
    flattened = [(wt.start, wt.end) for s in segments for wt in s.words or []]
    assert flattened[0][1] <= 0.5 and flattened[1][1] <= 0.5
    assert all(start >= 5.0 for start, _ in flattened[2:])


def test_transcribe_chunk_too_short_raises(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    transcriber = FastConformerCTCTranscriber(
        inference=SequenceFakeInference([3, 70]),  # 3 frames fit no word
        tokenizer=_fake_tokenizer_for_surah(1),
        chunker=_two_chunks_chunker(),
    )
    with pytest.raises(TranscriptionError, match="chunk too short"):
        transcriber.transcribe(audio, surah_id=1)


def test_transcribe_skips_empty_chunk(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    reference = _surah_words(1)

    class EmptyThenSpeech:
        def chunk(self, waveform, sample_rate):
            yield AudioChunk(waveform=np.zeros(0, dtype=np.float32), start_seconds=0.0)
            yield AudioChunk(waveform=waveform, start_seconds=5.0)

    transcriber = FastConformerCTCTranscriber(
        inference=FakeInference(n_frames=70),
        tokenizer=_fake_tokenizer_for_surah(1),
        chunker=EmptyThenSpeech(),
    )
    segments = transcriber.transcribe(audio, surah_id=1)
    assert _word_sequence(segments) == reference
    assert all(wt.start >= 5.0 for s in segments for wt in s.words or [])


def test_transcribe_no_speech_raises(tmp_path) -> None:
    audio = _make_wav(tmp_path)

    class NoChunks:
        def chunk(self, waveform, sample_rate):
            return iter(())

    transcriber = FastConformerCTCTranscriber(
        inference=FakeInference(),
        tokenizer=_fake_tokenizer_for_surah(1),
        chunker=NoChunks(),
    )
    with pytest.raises(TranscriptionError, match="No speech detected"):
        transcriber.transcribe(audio, surah_id=1)


def test_transcribe_inference_failure_in_chunk_propagates(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    # Only 6 frames per chunk: chunk 1 fits 2 words; chunk 2 fails.
    transcriber = FastConformerCTCTranscriber(
        inference=FailingInference(),
        tokenizer=_fake_tokenizer_for_surah(1),
        chunker=_two_chunks_chunker(),
    )
    with pytest.raises(TranscriptionError, match="boom"):
        transcriber.transcribe(audio, surah_id=1)


def test_fit_reference_prefix() -> None:
    words = ["a", "b", "c"]
    tokens = [[1], [2, 3, 4], [5]]  # costs: 2, 4, 2 (blank + tokens)
    # ground_truth length for k words = 2 + k + sum(len(tokens))
    assert _fit_reference_prefix(words, tokens, 1) == 0  # even word 0 needs 4
    assert _fit_reference_prefix(words, tokens, 4) == 1  # 2 + 1 + 1 = 4
    assert _fit_reference_prefix(words, tokens, 9) == 2  # 2 + 2 + 4 = 8; +word2 = 12 > 9
    assert _fit_reference_prefix(words, tokens, 13) == 3  # 2 + 3 + 6 = 11 <= 13


def test_silero_vad_chunker_requires_dependency(monkeypatch) -> None:
    # silero-vad is optional and lazy: construction is cheap, chunk() fails
    # with a clear error when the package is missing.
    chunker = SileroVADChunker()
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    with pytest.raises(TranscriptionError, match="silero-vad"):
        list(chunker.chunk(np.zeros(16000, dtype=np.float32), 16000))


def test_transcribe_invalid_surah_raises(tmp_path) -> None:
    audio = _make_wav(tmp_path)
    transcriber = FastConformerCTCTranscriber(inference=FakeInference(), tokenizer=FakeTokenizer())
    with pytest.raises(ValueError):
        transcriber.transcribe(audio, surah_id=999)  # invalid surah id


def test_single_pass_chunker_yields_one_offset_zero_chunk() -> None:
    waveform = np.zeros(16000, dtype=np.float32)
    chunks = list(SinglePassChunker().chunk(waveform, 16000))
    assert len(chunks) == 1
    assert chunks[0].start_seconds == 0.0
    np.testing.assert_array_equal(chunks[0].waveform, waveform)


# --------------------------------------------------------------------------- #
# Phonemizer adapter (validation only — never used for alignment)
# --------------------------------------------------------------------------- #


class FakePhonemizeResult:
    def __init__(self, text: str, phonemes: tuple[str, ...]) -> None:
        self._text = text
        self._phonemes = phonemes

    def text(self) -> str:
        return self._text

    def phonemes(self) -> tuple[str, ...]:
        return self._phonemes


class FakePhonemizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def phonemize(self, ref: str, **kwargs) -> FakePhonemizeResult:
        self.calls.append(ref)
        if ref not in {"1:1", "1:1-1:7"}:
            raise ValueError(f"unknown reference {ref}")
        return FakePhonemizeResult(
            text="بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
            phonemes=("b", "i", "s", "m", "i"),
        )


def test_phonemizer_adapter_validates_and_returns_ipa() -> None:
    fake = FakePhonemizer()
    adapter = QuranicPhonemizerAdapter(phonemizer=fake)
    assert adapter.validate_reference("1:1") is True
    assert adapter.validate_reference("99:99") is False
    assert adapter.canonical_text("1:1") == "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
    assert adapter.phonemes("1:1") == ("b", "i", "s", "m", "i")
    assert fake.calls == ["1:1", "99:99", "1:1", "1:1"]


def test_phonemizer_adapter_never_called_by_transcribe(tmp_path) -> None:
    class SpyAdapter:
        def validate_reference(self, ref: str) -> bool:  # pragma: no cover
            raise AssertionError("must not be called during transcribe")

        def canonical_text(self, ref: str) -> str:  # pragma: no cover
            raise AssertionError("must not be called during transcribe")

        def phonemes(self, ref: str) -> tuple[str, ...]:  # pragma: no cover
            raise AssertionError("must not be called during transcribe")

    audio = _make_wav(tmp_path)
    transcriber = FastConformerCTCTranscriber(
        inference=FakeInference(),
        tokenizer=_fake_tokenizer_for_surah(1),
        phonemizer_adapter=SpyAdapter(),
    )
    # The alignment path must never consult the phonemizer; the spy raises
    # if any of its methods are invoked.
    segments = transcriber.transcribe(audio, surah_id=1)
    assert len(segments) == 7
