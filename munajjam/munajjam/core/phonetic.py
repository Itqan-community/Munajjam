"""
Arabic phonetic similarity scoring.

Computes similarity between Arabic words based on phonetic features
(place and manner of articulation). This is a secondary signal to
supplement Indel string similarity, especially useful when transcription
introduces phonetically-close substitutions.
"""

from __future__ import annotations

from functools import lru_cache

from .arabic import normalize_arabic

# ---------------------------------------------------------------------------
# Phonetic feature tables
# ---------------------------------------------------------------------------

# Place of articulation groups (letters that share the same makhraj)
_PLACE_GROUPS: list[set[str]] = [
    {"ب", "م"},  # bilabial
    {"ف", "و"},  # labiodental / labial
    {"ث", "ذ", "ظ"},  # interdental
    {"ت", "د", "ط", "ض", "ن", "ل", "ر"},  # alveolar / dental
    {"س", "ز", "ص"},  # sibilant
    {"ش", "ج", "ي"},  # palatal
    {"ك", "غ", "خ"},  # velar
    {"ق"},  # uvular
    {"ع", "ح"},  # pharyngeal
    {"ه", "ا"},  # glottal / laryngeal
]

# Build letter → group-index lookup
_LETTER_PLACE: dict[str, int] = {}
for _i, _group in enumerate(_PLACE_GROUPS):
    for _ch in _group:
        _LETTER_PLACE[_ch] = _i

# Manner of articulation: emphatic letters
_EMPHATIC: set[str] = {"ص", "ض", "ط", "ظ"}

# Common transcription confusions (pairs that often swap in ASR output)
_CONFUSION_PAIRS: set[frozenset[str]] = {
    frozenset({"ت", "ط"}),
    frozenset({"د", "ض"}),
    frozenset({"ذ", "ز"}),
    frozenset({"ث", "س"}),
    frozenset({"ص", "س"}),
    frozenset({"ق", "ك"}),
    frozenset({"ع", "ا"}),
    frozenset({"ه", "ح"}),
    frozenset({"ا", "ه"}),
}


# ---------------------------------------------------------------------------
# Character-level phonetic distance
# ---------------------------------------------------------------------------


def _char_distance(a: str, b: str) -> float:
    """Return phonetic distance between two Arabic characters (0.0 = same, 1.0 = max)."""
    if a == b:
        return 0.0

    pair = frozenset({a, b})

    # Known ASR confusion pair → very close
    if pair in _CONFUSION_PAIRS:
        return 0.15

    # Same place of articulation → close
    pa = _LETTER_PLACE.get(a, -1)
    pb = _LETTER_PLACE.get(b, -1)
    if pa >= 0 and pa == pb:
        # Emphatic vs non-emphatic variant
        if (a in _EMPHATIC) != (b in _EMPHATIC):
            return 0.2
        return 0.3

    # Adjacent place groups → moderate
    if pa >= 0 and pb >= 0 and abs(pa - pb) == 1:
        return 0.5

    # Completely different → max
    return 1.0


# ---------------------------------------------------------------------------
# Word-level phonetic similarity
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def phonetic_word_similarity(word_a: str, word_b: str) -> float:
    """
    Compute phonetic similarity between two Arabic words.

    Uses a simple Needleman-Wunsch style alignment of characters
    with phonetic-distance substitution costs.

    Returns:
        Float between 0.0 (completely different) and 1.0 (identical).
    """
    if word_a == word_b:
        return 1.0
    if not word_a or not word_b:
        return 0.0

    n, m = len(word_a), len(word_b)

    # Short-circuit for very different lengths
    if max(n, m) > 2 * min(n, m):
        return 0.0

    # DP alignment
    gap_cost = 0.8
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_cost
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + _char_distance(word_a[i - 1], word_b[j - 1])
            ins = dp[i][j - 1] + gap_cost
            dele = dp[i - 1][j] + gap_cost
            dp[i][j] = min(sub, ins, dele)

    max_possible = max(n, m) * 1.0  # worst case: all max-distance subs
    distance = dp[n][m]
    return max(0.0, 1.0 - distance / max_possible)


def phonetic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute phonetic similarity between two Arabic text strings.

    Normalises both texts, then computes average word-level phonetic
    similarity using a simple word alignment by position.

    Returns:
        Float between 0.0 and 1.0.
    """
    norm_a = normalize_arabic(text_a)
    norm_b = normalize_arabic(text_b)
    if not norm_a or not norm_b:
        return 0.0

    words_a = norm_a.split()
    words_b = norm_b.split()

    # Align by position (simple approach matching word-DP's usage)
    n = max(len(words_a), len(words_b))
    if n == 0:
        return 1.0

    total = 0.0
    for i in range(n):
        wa = words_a[i] if i < len(words_a) else ""
        wb = words_b[i] if i < len(words_b) else ""
        total += phonetic_word_similarity(wa, wb)

    return total / n
