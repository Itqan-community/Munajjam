"""
Word-level Dynamic Programming alignment.

Instead of mapping whole Whisper segments to ayahs, this module flattens all
segments into a word stream (with estimated per-word timestamps) and runs DP
at word granularity.  This allows splitting precisely at ayah boundaries even
when they fall mid-segment.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..models import AlignmentResult, Ayah, Segment
from .arabic import normalize_arabic
from .dp_core import _filter_special_segments
from .matcher import similarity
from .phonetic import phonetic_similarity

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TranscribedWord:
    """A single word extracted from a Whisper segment with estimated timing."""

    text: str
    normalized: str
    estimated_start: float
    estimated_end: float
    segment_idx: int
    word_idx_in_segment: int


# ---------------------------------------------------------------------------
# 1. Build word stream
# ---------------------------------------------------------------------------


def build_word_stream(segments: list[Segment]) -> list[TranscribedWord]:
    """
    Flatten segments into a word stream with per-word timestamps.

    If a segment carries real word-level timestamps (``segment.words``),
    those are used directly.  Otherwise, timestamps are distributed within
    each segment proportional to the character length of each word.
    """
    words: list[TranscribedWord] = []

    for seg_idx, seg in enumerate(segments):
        # Prefer real word timestamps from faster-whisper
        if seg.words:
            for word_idx, wt in enumerate(seg.words):
                words.append(
                    TranscribedWord(
                        text=wt.word,
                        normalized=normalize_arabic(wt.word),
                        estimated_start=wt.start,
                        estimated_end=wt.end,
                        segment_idx=seg_idx,
                        word_idx_in_segment=word_idx,
                    )
                )
            continue

        # Fallback: character-weighted time distribution
        raw_words = seg.text.split()
        if not raw_words:
            continue

        char_lengths = [len(w) for w in raw_words]
        total_chars = sum(char_lengths)
        if total_chars == 0:
            total_chars = 1

        seg_duration = seg.end - seg.start
        current_time = seg.start

        for word_idx, (word, char_len) in enumerate(zip(raw_words, char_lengths, strict=False)):
            word_duration = (char_len / total_chars) * seg_duration
            word_end = current_time + word_duration

            words.append(
                TranscribedWord(
                    text=word,
                    normalized=normalize_arabic(word),
                    estimated_start=current_time,
                    estimated_end=word_end,
                    segment_idx=seg_idx,
                    word_idx_in_segment=word_idx,
                )
            )

            current_time = word_end

    return words


# ---------------------------------------------------------------------------
# 2. Build reference word lists
# ---------------------------------------------------------------------------


def build_reference_words(ayahs: list[Ayah]) -> list[list[str]]:
    """Return normalized word lists per ayah."""
    return [normalize_arabic(a.text).split() for a in ayahs]


# ---------------------------------------------------------------------------
# 3. Word-level DP
# ---------------------------------------------------------------------------


def _jaccard_word_overlap(words_a: set[str], words_b: set[str]) -> float:
    """Fast Jaccard similarity between two word sets."""
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def _bigram_overlap(text_a: str, text_b: str) -> float:
    """Compute character bigram overlap between two texts (normalised)."""
    na = normalize_arabic(text_a)
    nb = normalize_arabic(text_b)
    if len(na) < 2 or len(nb) < 2:
        return 0.0
    bg_a = {na[i : i + 2] for i in range(len(na) - 1)}
    bg_b = {nb[i : i + 2] for i in range(len(nb) - 1)}
    inter = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return inter / union if union > 0 else 0.0


def _word_alignment_cost(
    merged_text: str,
    ayah_text: str,
    n_assigned: int,
    ref_count: int,
    actual_duration: float | None = None,
    median_sec_per_word: float | None = None,
) -> float:
    """
    Compute cost for assigning n_assigned words to an ayah.

    Uses Indel similarity as the primary signal. Phonetic similarity is
    blended in only when Indel is in the ambiguous range (0.4-0.85) where
    phonetic features can help disambiguate similar-sounding ayahs.

    A log-normal duration penalty discourages assigning absurdly short or
    long audio spans relative to the expected duration (ref_count * median
    seconds per word).
    """
    indel_sim = similarity(merged_text, ayah_text)

    # Only blend phonetic when Indel is ambiguous
    if 0.4 <= indel_sim <= 0.85:
        phon_sim = phonetic_similarity(merged_text, ayah_text)
        blended_sim = 0.85 * indel_sim + 0.15 * phon_sim
    else:
        blended_sim = indel_sim

    # Coverage ratio penalty (from compute_alignment_cost logic)
    from .matcher import compute_coverage_ratio

    coverage = compute_coverage_ratio(merged_text, ayah_text)
    coverage_penalty = 0.0
    if coverage < 0.7:
        coverage_penalty = (0.7 - coverage) * 0.5
    elif coverage > 1.3:
        coverage_penalty = (coverage - 1.3) * 0.3

    base_cost = (1 - blended_sim) + coverage_penalty

    # Critical threshold penalty for very low similarity
    if blended_sim < 0.4:
        base_cost += 0.5 * (0.4 - blended_sim) / 0.4

    # Word-count ratio penalty: penalise being far from ref_count
    if ref_count > 0:
        ratio = n_assigned / ref_count
        if ratio < 0.5:
            base_cost += (0.5 - ratio) * 0.4
        elif ratio > 2.0:
            base_cost += (ratio - 2.0) * 0.2

    # Duration prior: log-normal penalty on actual vs expected duration.
    # Prevents assigning 4 seconds to a 27-word ayah or 9 seconds to a
    # 5-word ayah.  Uses a gentle sigma so only extreme outliers are hit.
    if (
        actual_duration is not None
        and median_sec_per_word is not None
        and median_sec_per_word > 0
        and actual_duration > 0
        and ref_count > 0
    ):
        expected_duration = ref_count * median_sec_per_word
        log_ratio = math.log(actual_duration / expected_duration)
        sigma = 0.8  # fairly permissive — only extreme deviations penalised
        duration_penalty = 0.15 * (log_ratio / sigma) ** 2
        base_cost += duration_penalty

    return max(0.0, base_cost)


def _build_silence_bonus(
    words: list[TranscribedWord],
    silences_ms: list[tuple[int, int]] | None,
    bonus: float = 0.15,
    penalty: float = 0.05,
) -> list[float]:
    """
    Build per-word-boundary silence bonus/penalty array.

    For each word boundary (before word *i*), check if a silence period
    falls in the gap.  If so, assign a negative cost adjustment (bonus);
    otherwise a small positive penalty.  This guides the DP to align ayah
    boundaries with actual silence pauses in the audio, critical for
    disambiguating repetitive short ayahs.

    Returns:
        List of floats, length = len(words) + 1.  Index 0 and len(words) are
        always 0 (start/end boundaries are free).  silence_bonus[i] < 0 means
        placing a boundary before word *i* is encouraged.
    """
    n = len(words)
    result = [0.0] * (n + 1)

    if not silences_ms or n == 0:
        return result

    # Convert silences to seconds and sort
    silences_sec = sorted((s / 1000.0, e / 1000.0) for s, e in silences_ms)

    # For each word boundary, check if a silence overlaps the gap
    for i in range(1, n):
        gap_start = words[i - 1].estimated_end
        gap_end = words[i].estimated_start
        gap_dur = gap_end - gap_start

        if gap_dur < 0.05:
            # No meaningful gap — penalise boundary here
            result[i] = penalty
            continue

        # Check if any silence overlaps this gap
        found = False
        for s_start, s_end in silences_sec:
            if s_start > gap_end + 1.0:
                break  # Past this gap
            # Check overlap
            overlap_start = max(gap_start, s_start)
            overlap_end = min(gap_end, s_end)
            if overlap_end > overlap_start:
                found = True
                break

        if found:
            result[i] = -bonus  # Encourage boundary here
        else:
            result[i] = penalty  # Discourage boundary here

    return result


def align_words_dp(
    words: list[TranscribedWord],
    ayahs: list[Ayah],
    ref_words: list[list[str]],
    max_word_ratio: float = 3.0,
    beam_width: int = 50,
    silences_ms: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    """
    Run DP at word granularity to find optimal word-to-ayah mapping.

    DP state: dp[w][a] = minimum cost to have consumed the first *w* words
    and assigned the first *a* ayahs.

    Transition: for ayah *a*, try assigning k consecutive words
    (1 <= k <= max_k) where max_k is bounded by ``max_word_ratio`` times the
    reference word count for that ayah.

    Args:
        beam_width: Keep only the top-K lowest-cost states per ayah layer.
        silences_ms: Silence periods in milliseconds for boundary guidance.

    Returns:
        List of (word_start_idx, word_end_idx, ayah_idx) tuples.
        word_end_idx is exclusive.
    """
    n_words = len(words)
    n_ayahs = len(ayahs)

    if n_words == 0 or n_ayahs == 0:
        return []

    INF = float("inf")

    # Pre-compute cumulative reference word counts for expected-position window
    cum_ref = [0] * (n_ayahs + 1)
    for i in range(n_ayahs):
        cum_ref[i + 1] = cum_ref[i] + max(len(ref_words[i]), 1)
    total_ref_words = cum_ref[n_ayahs]
    scale = n_words / total_ref_words if total_ref_words > 0 else 1.0

    # Pre-build word text array for fast range joins via prefix approach
    word_texts = [w.text for w in words]

    # Pre-build normalized word sets for Jaccard pre-filter
    word_norm = [w.normalized for w in words]

    # Pre-compute ayah word sets for Jaccard
    ayah_word_sets = [set(rw) for rw in ref_words]

    # Compute median seconds-per-word for duration prior
    total_audio_duration = words[-1].estimated_end - words[0].estimated_start
    median_sec_per_word = total_audio_duration / n_words if n_words > 0 else 0.5

    # Build silence bonus array for boundary guidance
    silence_bonus = _build_silence_bonus(words, silences_ms)

    # dp_prev[w] = cost to assign first (a-1) ayahs using first w words
    dp_prev: dict[int, float] = {0: 0.0}

    # backtrack[a] = dict mapping w -> prev_w
    backtrack: list[dict[int, int]] = []

    # Similarity cache keyed by (merged_text_hash, ayah_index)
    _cost_cache: dict[tuple[int, int], float] = {}

    # Pre-compute normalised ayah texts for context matching
    ayah_norm_texts = [normalize_arabic(a.text) for a in ayahs]

    for a in range(n_ayahs):
        ref_count = max(len(ref_words[a]), 1)
        max_k = max(int(ref_count * max_word_ratio), 3)
        min_k = max(1, int(ref_count * 0.3))

        remaining_ayahs = n_ayahs - a - 1
        ayah_wset = ayah_word_sets[a]

        # Short-ayah flag: enable context scoring for repetitive short ayahs
        is_short_ayah = ref_count <= 8

        # Expected position window
        expected_end = int(cum_ref[a + 1] * scale)
        slack = max(int(n_words * 0.05), max_k * 2)
        window_lo = max(a + 1, expected_end - slack)
        window_hi = min(n_words - remaining_ayahs + 1, expected_end + slack)

        dp_cur: dict[int, float] = {}
        bt: dict[int, int] = {}

        ayah_text = ayahs[a].text

        for prev_w, prev_cost in dp_prev.items():
            lo = max(prev_w + min_k, window_lo)
            hi = min(prev_w + max_k, window_hi)

            for w_end in range(lo, hi + 1):
                if w_end > n_words:
                    break

                # Fast skip: if prev_cost already exceeds current best for w_end
                if w_end in dp_cur and prev_cost >= dp_cur[w_end]:
                    continue

                n_assigned = w_end - prev_w

                # Jaccard pre-filter: skip clearly wrong spans
                span_wset = set(word_norm[prev_w:w_end])
                jaccard = _jaccard_word_overlap(span_wset, ayah_wset)
                if jaccard < 0.15:
                    # Assign high cost without expensive similarity call
                    total = prev_cost + 1.5
                    if w_end not in dp_cur or total < dp_cur[w_end]:
                        dp_cur[w_end] = total
                        bt[w_end] = prev_w
                    continue

                merged_text = " ".join(word_texts[prev_w:w_end])

                # Compute actual duration for this span
                actual_dur = words[w_end - 1].estimated_end - words[prev_w].estimated_start

                # Cache by (text_hash, ayah_index, duration_bucket) to
                # avoid recomputing identical text spans.  Duration is
                # bucketed to 0.5s to keep cache effective.
                dur_bucket = round(actual_dur * 2) / 2
                cache_key = (hash(merged_text), a, dur_bucket)
                if cache_key in _cost_cache:
                    cost = _cost_cache[cache_key]
                else:
                    cost = _word_alignment_cost(
                        merged_text,
                        ayah_text,
                        n_assigned,
                        ref_count,
                        actual_duration=actual_dur,
                        median_sec_per_word=median_sec_per_word,
                    )
                    _cost_cache[cache_key] = cost

                # Add silence bonus/penalty at boundary
                cost += silence_bonus[w_end]

                # N-gram context bonus for short repetitive ayahs:
                # Check if words before/after this span match prev/next ayahs
                if is_short_ayah and cost < 1.0:
                    context_bonus = 0.0
                    ctx_count = 0

                    # Check preceding context against previous ayah
                    if a > 0 and prev_w >= 2:
                        ctx_start = max(0, prev_w - ref_count)
                        ctx_text = " ".join(word_norm[ctx_start:prev_w])
                        if ctx_text:
                            ctx_bonus = _bigram_overlap(ctx_text, ayah_norm_texts[a - 1])
                            context_bonus += ctx_bonus
                            ctx_count += 1

                    # Check following context against next ayah
                    if a < n_ayahs - 1 and w_end + 2 <= n_words:
                        next_ref_count = max(len(ref_words[a + 1]), 1)
                        ctx_end = min(n_words, w_end + next_ref_count)
                        ctx_text = " ".join(word_norm[w_end:ctx_end])
                        if ctx_text:
                            ctx_bonus = _bigram_overlap(ctx_text, ayah_norm_texts[a + 1])
                            context_bonus += ctx_bonus
                            ctx_count += 1

                    # Apply context discount (lower cost when context matches)
                    if ctx_count > 0:
                        avg_ctx = context_bonus / ctx_count
                        cost -= avg_ctx * 0.35  # Up to 0.35 discount
                        cost = max(0.0, cost)

                total = prev_cost + cost

                if w_end not in dp_cur or total < dp_cur[w_end]:
                    dp_cur[w_end] = total
                    bt[w_end] = prev_w

        # Adaptive beam pruning: widen beam when costs are uniformly high
        # (indicates a difficult region where pruning too early loses the
        # correct path).
        effective_beam = beam_width
        if beam_width > 0 and len(dp_cur) > beam_width:
            costs = list(dp_cur.values())
            if costs:
                min_cost = min(costs)
                # Count states within 20% of best — if most are close, it's
                # an ambiguous region and we should keep more candidates.
                close_count = sum(1 for c in costs if c < min_cost * 1.2 + 0.1)
                if close_count > beam_width * 0.6:
                    effective_beam = min(beam_width * 3, len(dp_cur))
            sorted_states = sorted(dp_cur.items(), key=lambda x: x[1])
            dp_cur = dict(sorted_states[:effective_beam])
            bt = {w: bt[w] for w in dp_cur}

        backtrack.append(bt)
        dp_prev = dp_cur

        # Clear old cost cache entries periodically to limit memory
        if a % 50 == 0 and a > 0:
            _cost_cache.clear()

    # Find best ending
    best_w = None
    best_cost = INF

    for w, cost in dp_prev.items():
        if cost < best_cost:
            best_cost = cost
            best_w = w

    if best_w is None:
        return []

    # Backtrack
    assignments: list[tuple[int, int, int]] = []
    w = best_w
    for a in range(n_ayahs - 1, -1, -1):
        prev_w = backtrack[a].get(w)
        if prev_w is None:
            break
        assignments.append((prev_w, w, a))
        w = prev_w

    assignments.reverse()
    return assignments


# ---------------------------------------------------------------------------
# 3b. Chunked Word-DP for large surahs
# ---------------------------------------------------------------------------


def _chunked_align_words_dp(
    words: list[TranscribedWord],
    ayahs: list[Ayah],
    ref_words: list[list[str]],
    max_word_ratio: float = 3.0,
    chunk_size: int = 60,
    overlap: int = 10,
    silences_ms: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    """
    Split large alignment problems into overlapping chunks and stitch results.

    Each chunk covers ``chunk_size`` ayahs with ``overlap`` ayahs shared
    between consecutive chunks.  Word ranges per chunk are estimated from
    cumulative reference word counts.  In overlap regions, assignments from
    the chunk with lower per-ayah cost are kept.

    Returns:
        List of (word_start_idx, word_end_idx, ayah_idx) tuples.
    """
    n_words = len(words)
    n_ayahs = len(ayahs)

    if n_ayahs <= chunk_size:
        return align_words_dp(words, ayahs, ref_words, max_word_ratio, silences_ms=silences_ms)

    # Pre-compute cumulative reference word counts
    cum_ref = [0] * (n_ayahs + 1)
    for i in range(n_ayahs):
        cum_ref[i + 1] = cum_ref[i] + max(len(ref_words[i]), 1)
    total_ref_words = cum_ref[n_ayahs]
    scale = n_words / total_ref_words if total_ref_words > 0 else 1.0

    # Build chunk definitions: (ayah_start, ayah_end) — end exclusive
    chunks: list[tuple[int, int]] = []
    a_start = 0
    while a_start < n_ayahs:
        a_end = min(a_start + chunk_size, n_ayahs)
        chunks.append((a_start, a_end))
        if a_end >= n_ayahs:
            break
        a_start = a_end - overlap

    # Run DP on each chunk
    chunk_results: list[list[tuple[int, int, int]]] = []
    for a_start, a_end in chunks:
        # Estimate word range for this chunk with generous margin
        margin = max(int(n_words * 0.03), 50)
        w_lo = max(0, int(cum_ref[a_start] * scale) - margin)
        w_hi = min(n_words, int(cum_ref[a_end] * scale) + margin)

        chunk_words = words[w_lo:w_hi]
        chunk_ayahs = ayahs[a_start:a_end]
        chunk_ref = ref_words[a_start:a_end]

        assignments = align_words_dp(
            chunk_words,
            chunk_ayahs,
            chunk_ref,
            max_word_ratio,
            silences_ms=silences_ms,
        )

        # Remap indices back to global
        remapped = [(ws + w_lo, we + w_lo, ai + a_start) for ws, we, ai in assignments]
        chunk_results.append(remapped)

    # Stitch: merge chunks, resolving overlaps by cost
    # Build per-ayah assignment map; for overlapping ayahs, pick lowest cost
    best_per_ayah: dict[int, tuple[int, int, int, float]] = {}

    for assignments in chunk_results:
        for ws, we, ai in assignments:
            merged = " ".join(w.text for w in words[ws:we])
            cost = _word_alignment_cost(
                merged,
                ayahs[ai].text,
                we - ws,
                max(len(ref_words[ai]), 1),
            )
            existing = best_per_ayah.get(ai)
            if existing is None or cost < existing[3]:
                best_per_ayah[ai] = (ws, we, ai, cost)

    # Extract sorted assignments
    final = sorted(
        [(ws, we, ai) for ws, we, ai, _ in best_per_ayah.values()],
        key=lambda x: x[2],
    )

    # Fix any word-range overlaps between consecutive ayahs caused by stitching
    for i in range(1, len(final)):
        prev_ws, prev_we, prev_ai = final[i - 1]
        cur_ws, cur_we, cur_ai = final[i]
        if cur_ws < prev_we:
            # Split at midpoint
            mid = (prev_we + cur_ws) // 2
            mid = max(mid, prev_ws + 1)
            mid = min(mid, cur_we - 1)
            final[i - 1] = (prev_ws, mid, prev_ai)
            final[i] = (mid, cur_we, cur_ai)

    return final


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------


def align_segments_word_dp(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[tuple[int, int]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    max_word_ratio: float = 3.0,
) -> list[AlignmentResult]:
    """
    Align segments to ayahs using word-level DP.

    1. Filter special segments (istiadha, basmala).
    2. Build a word stream from all segments.
    3. Run word-level DP to find the optimal word-to-ayah mapping.
    4. Convert to AlignmentResult objects.

    For large surahs (>2000 words), automatically uses chunked DP for speed.

    Falls back to segment-level DP if word-level produces no assignments.

    Args:
        segments: Whisper transcription segments
        ayahs: Reference Quran ayahs (in order)
        silences_ms: Silence periods used to guide boundary placement
        on_progress: Optional progress callback
        max_word_ratio: Max words per ayah relative to reference word count

    Returns:
        List of AlignmentResult with timing and similarity info
    """
    if not segments or not ayahs:
        return []

    filtered = _filter_special_segments(segments, ayahs)

    # Build word stream
    words = build_word_stream(filtered)
    if not words:
        return []

    # Build reference words
    ref_words = build_reference_words(ayahs)

    if on_progress:
        on_progress(0, len(ayahs))

    # Choose chunked vs direct DP based on problem size
    if len(words) > 2000:
        assignments = _chunked_align_words_dp(
            words,
            ayahs,
            ref_words,
            max_word_ratio,
            silences_ms=silences_ms,
        )
    else:
        assignments = align_words_dp(
            words,
            ayahs,
            ref_words,
            max_word_ratio,
            silences_ms=silences_ms,
        )

    if on_progress:
        on_progress(len(ayahs), len(ayahs))

    if not assignments:
        # Fallback to segment-level DP
        from .dp_core import align_segments_dp_with_constraints

        return align_segments_dp_with_constraints(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
            on_progress=on_progress,
        )

    # Convert assignments to AlignmentResult
    results: list[AlignmentResult] = []

    for word_start, word_end, ayah_idx in assignments:
        ayah = ayahs[ayah_idx]

        # Timing from word estimates
        start_time = words[word_start].estimated_start
        end_time = words[word_end - 1].estimated_end

        # Transcribed text
        transcribed = " ".join(w.text for w in words[word_start:word_end])

        sim = similarity(transcribed, ayah.text)

        results.append(
            AlignmentResult(
                ayah=ayah,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                transcribed_text=transcribed,
                similarity_score=round(sim, 4),
                overlap_detected=False,
            )
        )

    return results
