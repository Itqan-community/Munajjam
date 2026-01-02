# Munajjam Alignment Algorithm: Deep Dive

A comprehensive study guide covering the alignment algorithm, problems encountered, solutions implemented, and computer science concepts learned.

---

## Table of Contents

1. [Overview: What is Alignment?](#1-overview-what-is-alignment)
2. [The Original Greedy Algorithm](#2-the-original-greedy-algorithm)
3. [Similarity Score Calculation](#3-similarity-score-calculation)
4. [The False Positive Problem](#4-the-false-positive-problem)
5. [Fix Attempts and Learnings](#5-fix-attempts-and-learnings)
6. [Computer Science Concepts](#6-computer-science-concepts)
7. [The DP Solution](#7-the-dp-solution)
8. [Implementation Details](#8-implementation-details)
9. [Results Comparison](#9-results-comparison)
10. [Key Takeaways](#10-key-takeaways)
11. [Further Reading](#11-further-reading)

---

## 1. Overview: What is Alignment?

### The Problem

We have:
- **Audio file**: A recitation of a Quran surah (e.g., Surah Al-Baqarah, 2+ hours)
- **Transcribed segments**: Whisper AI transcription broken into ~831 segments
- **Reference ayahs**: 286 ayahs with known text

**Goal**: Find the start/end timestamps for each ayah in the audio.

### Why is This Hard?

1. **Multiple segments per ayah**: One ayah might span 2-5 transcribed segments
2. **Imperfect transcription**: Whisper makes mistakes, especially with Arabic diacritics
3. **Similar text patterns**: Some ayahs start with identical phrases
4. **No clear boundaries**: Audio doesn't have markers between ayahs

### Visual Representation

```
Audio:     [=========================================]
Segments:  [s1][s2][s3][s4][s5][s6][s7][s8][s9][s10]...
Ayahs:     [  Ayah 1  ][  Ayah 2  ][   Ayah 3   ]...

Goal: Map segments to ayahs
  s1 + s2 → Ayah 1 (timestamps: 0.0s - 15.2s)
  s3 + s4 + s5 → Ayah 2 (timestamps: 15.2s - 32.1s)
  ...
```

---

## 2. The Original Greedy Algorithm

### How It Works

The greedy algorithm processes ayahs one at a time:

```python
for each ayah in ayahs:
    merged_text = ""
    
    while not done:
        # Merge next segment
        merged_text += next_segment.text
        
        # Check 1: Do the last words match the ayah's ending?
        if last_words_match(merged_text, ayah) and coverage >= 70%:
            finalize_ayah()
            break
        
        # Check 2: Does the next segment start the next ayah?
        if next_segment_starts_next_ayah():
            if coverage >= threshold:
                finalize_ayah()
                break
        
        # Check 3: Is there a silence gap?
        if silence_gap_found and next_segment_starts_next_ayah():
            finalize_ayah()
            break
        
        # Keep merging...
```

### Key Checks

| Check | Purpose | How It Works |
|-------|---------|--------------|
| **Last words match** | Detect ayah completion | Compare last 3 words of merged text to ayah's last 3 words |
| **Next ayah starts** | Detect boundary | Compare next segment's first 3 words to next ayah's first 3 words |
| **Silence gap** | Natural boundary | Use audio silence detection to find pauses |
| **Coverage ratio** | Validation | Ensure merged text covers enough of the reference ayah |

### The Problem: Greedy is Short-Sighted

Greedy makes **local decisions** without seeing the big picture:

```
Step 1: [seg1 seg2] → Ayah 1? Maybe... COMMIT ✓
Step 2: [seg3] → Ayah 2? No, merge more...
Step 3: [seg3 seg4 seg5...] → Still no match... keep merging...
Step 4: Eventually fails or merges too much

❌ If Step 1 was wrong, ALL subsequent steps are affected!
```

---

## 3. Similarity Score Calculation

### The `similarity()` Function

Located in `munajjam/munajjam/core/matcher.py`:

```python
from difflib import SequenceMatcher

def similarity(text1: str, text2: str, normalize: bool = True) -> float:
    """
    Calculate similarity between two Arabic texts.
    
    Returns a value between 0.0 (no match) and 1.0 (perfect match).
    """
    if normalize:
        text1 = normalize_arabic(text1)
        text2 = normalize_arabic(text2)
    
    return SequenceMatcher(None, text1, text2).ratio()
```

### Arabic Normalization

Before comparison, text is normalized:

```python
def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for comparison.
    
    Operations:
    1. Replace alef variants (أ إ آ ا) → ا
    2. Replace alef maqsura (ى) → ي
    3. Replace ta marbuta (ة) → ه
    4. Remove diacritics (tashkeel): ً ٌ ٍ َ ُ ِ ّ ْ
    5. Remove punctuation
    6. Collapse multiple spaces
    """
```

### Example

```python
text1 = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
text2 = "بسم الله الرحمن الرحيم"

# After normalization, both become similar
# similarity ≈ 0.95+
```

### Similarity Threshold Usage

| Context | Threshold | Purpose |
|---------|-----------|---------|
| `_check_end_of_ayah` | 0.6 | Detect if last words match |
| `_check_next_ayah_starts` | 0.6 | Detect if next segment starts new ayah |
| `coverage_threshold` | 0.7 | Ensure 70% of ayah words are covered |

---

## 4. The False Positive Problem

### The Scenario

Surah Al-Baqarah, Ayahs 106-107:

```
Ayah 106: "مَا نَنسَخۡ مِنۡ ءَايَةٍ... أَلَمۡ تَعۡلَمۡ أَنَّ ٱللَّهَ عَلَىٰ كُلِّ شَيۡءٖ قَدِيرٌ"
                                    ^^^^^^^^^^^^^^^^^^^^^^^^
                                    (Don't you know that Allah...)

Ayah 107: "أَلَمۡ تَعۡلَمۡ أَنَّ ٱللَّهَ لَهُۥ مُلۡكُ ٱلسَّمَٰوَٰتِ..."
           ^^^^^^^^^^^^^^^^^^^^^^^^
           (Don't you know that Allah...)
```

Both ayahs contain "أَلَمۡ تَعۡلَمۡ أَنَّ ٱللَّهَ" (Don't you know that Allah)!

### What Went Wrong

```
Segments:
  seg1: "ما ننسخ من آية أو ننسها نأت بخير منها أو مثلها"  (part of ayah 106)
  seg2: "ألم تعلم أن الله على كل شيء قدير"                (rest of ayah 106)
  seg3: "ألم تعلم أن الله له ملك السموات..."              (ayah 107)

Problem:
  After seg1, the algorithm checks if seg2 starts ayah 107
  First 3 words of seg2: "ألم تعلم أن"
  First 3 words of ayah 107: "ألم تعلم أن"
  
  Similarity = 1.0 → FALSE POSITIVE!
  
  The algorithm incorrectly thinks seg2 is the start of ayah 107,
  so it finalizes ayah 106 with only seg1 (incomplete)!
```

### The Cascade Effect

Once the wrong boundary is set:
- Ayah 106 is incomplete (low similarity)
- Ayah 107 gets seg2 (wrong content)
- Ayah 108 gets seg3 (wrong content)
- ... errors cascade through remaining ayahs
- Eventually, the algorithm merges everything into one ayah

---

## 5. Fix Attempts and Learnings

### Attempt 1: Increase Word Count

**Idea**: Check 5-6 words instead of 3 to distinguish similar starts.

```python
# Before: Check 3 words
seg2 first 3 words: "ألم تعلم أن"
ayah 107 first 3 words: "ألم تعلم أن"
Similarity = 1.0 → FALSE POSITIVE

# After: Check 5 words  
seg2 first 5 words: "ألم تعلم أن الله علي"
ayah 107 first 5 words: "ألم تعلم أن الله له"
Similarity ≈ 0.85 → Still might pass 0.6 threshold!
```

**Result**: Helped slightly, but not enough.

### Attempt 2: Higher Threshold

**Idea**: Use 0.9 threshold instead of 0.6 for "next ayah starts" check.

```python
# With 0.9 threshold:
# 0.85 < 0.9 → Check fails → Continue merging
```

**Result**: Too strict! Algorithm couldn't detect ANY boundaries, aligned only 116/286 ayahs.

### Attempt 3: Coverage Check

**Idea**: Only finalize if the current ayah has sufficient coverage.

```python
if _check_next_ayah_starts(next_segment, next_ayah):
    coverage = compute_coverage_ratio(merged_text, ayah.text)
    if coverage >= 0.5:  # At least 50% covered
        finalize_ayah()
    # else: continue merging
```

**Result**: Better! But still had issues with edge cases.

### Attempt 4: Last Words Check

**Idea**: Only finalize via "next ayah starts" if current merged text ends correctly.

```python
if _check_next_ayah_starts(next_segment, next_ayah):
    # Check if last words match
    last_words_match = similarity(seg_last, ayah_last) >= threshold
    
    # Finalize if: last words match OR coverage is very high
    if last_words_match or coverage >= 0.9:
        finalize_ayah()
```

**Result**: Improved 106-107 case, but still aligned only 221/286 ayahs.

### Key Learning

**Greedy algorithms can't recover from mistakes.** Once a wrong boundary is set, all subsequent decisions are compromised. We needed a **global optimization** approach.

---

## 6. Computer Science Concepts

### 6.1 Dynamic Programming (DP)

**Definition**: A method for solving complex problems by breaking them into overlapping subproblems and storing results to avoid recomputation.

**Key Properties**:
1. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
2. **Overlapping Subproblems**: Same subproblems are solved multiple times

**Example: Fibonacci**

```python
# Without DP (exponential time)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)  # Recalculates same values!

# With DP (linear time)
def fib_dp(n):
    dp = [0, 1]
    for i in range(2, n+1):
        dp.append(dp[i-1] + dp[i-2])  # Store and reuse!
    return dp[n]
```

### 6.2 Sequence Alignment (Needleman-Wunsch)

**Problem**: Align two sequences to find the best matching.

**Classic Use**: DNA sequence alignment (bioinformatics)

```
Sequence 1: A C G T A
Sequence 2: A G T A

Alignment:
  A C G T A
  A - G T A
  ↑ gap
```

**DP Formulation**:

```
dp[i][j] = best score to align seq1[0:i] with seq2[0:j]

Recurrence:
dp[i][j] = max(
    dp[i-1][j-1] + match_score(seq1[i], seq2[j]),  # Match/mismatch
    dp[i-1][j] + gap_penalty,                       # Gap in seq2
    dp[i][j-1] + gap_penalty                        # Gap in seq1
)
```

### 6.3 Dynamic Time Warping (DTW)

**Problem**: Align sequences of different lengths that may vary in speed.

**Classic Use**: Speech recognition (matching spoken words to reference)

**Key Feature**: Allows "warping" time to find best alignment.

```
Reference: [Hello]    [World]
Spoken:    [Hel] [lo] [Wor] [ld]

DTW aligns them despite different segment counts:
  Hel + lo → Hello
  Wor + ld → World
```

### 6.4 How These Apply to Our Problem

| Concept | Classic Application | Our Application |
|---------|---------------------|-----------------|
| Needleman-Wunsch | Align DNA sequences | Align segments to ayahs |
| DTW | Align audio frames | Handle variable segment counts |
| DP | Optimize any recursive problem | Find globally optimal alignment |

**Our Adaptation**:
- Instead of characters, we align **segments** to **ayahs**
- Instead of match/mismatch, we use **similarity score**
- We allow **many-to-one** mapping (multiple segments → one ayah)

---

## 7. The DP Solution

### Conceptual Model

We build a 2D grid:
- **Y-axis**: Segments (0 to 831)
- **X-axis**: Ayahs (1 to 286)

Each cell `dp[i][j]` represents the **best cost** to align segments `[0:i]` to ayahs `[0:j]`.

```
         Ayahs →
         1    2    3    4    ...  286
Seg 0   [0]   
Seg 1   [?]  [?]
Seg 2   [?]  [?]  [?]
Seg 3   [?]  [?]  [?]  [?]
...                              
Seg 831                          [GOAL]
```

### The Recurrence Relation

For each cell `dp[i][j]`, we try merging `k = 1, 2, 3, ...` segments into ayah `j`:

```python
dp[i][j] = min over k of (
    dp[i-k][j-1] + cost(segments[i-k:i], ayah[j])
)

Where:
- dp[i-k][j-1] = best cost to align first i-k segments to first j-1 ayahs
- cost() = 1 - similarity (lower is better)
```

### Visual Example

```
Trying to fill dp[5][2] (5 segments aligned to 2 ayahs):

Option 1: k=1 (1 segment for ayah 2)
  dp[4][1] + cost(seg[4:5], ayah2)
  
Option 2: k=2 (2 segments for ayah 2)
  dp[3][1] + cost(seg[3:5], ayah2)
  
Option 3: k=3 (3 segments for ayah 2)
  dp[2][1] + cost(seg[2:5], ayah2)

Pick the minimum cost!
```

### Backtracking

After filling the DP table, we find the optimal path:

```python
# Start at dp[n_seg][n_ayah]
# Follow parent pointers back to dp[0][0]
# This gives us the optimal segment-to-ayah mapping

path = []
current = (n_seg, n_ayah)
while current:
    cell = dp[current]
    path.append(cell)
    current = cell.parent
path.reverse()
```

---

## 8. Implementation Details

### The DPCell Data Structure

```python
@dataclass
class DPCell:
    cost: float           # Accumulated cost to reach this cell
    merged_text: str      # Text merged so far for current ayah
    seg_start_idx: int    # Index of first segment in current ayah
    parent: tuple | None  # Previous cell for backtracking
```

### The Cost Function

```python
def compute_alignment_cost(merged_text: str, ayah_text: str) -> float:
    """
    Compute cost of aligning merged segments to an ayah.
    Lower cost = better alignment.
    Returns value between 0 (perfect) and 1 (no match).
    """
    # Primary: text similarity
    sim = similarity(merged_text, ayah_text)
    
    # Secondary: coverage penalty
    coverage = compute_coverage_ratio(merged_text, ayah_text)
    
    if coverage < 0.7:
        coverage_penalty = (0.7 - coverage) * 0.5
    elif coverage > 1.3:
        coverage_penalty = (coverage - 1.3) * 0.3
    else:
        coverage_penalty = 0
    
    return (1 - sim) + coverage_penalty
```

### Optimizations Applied

1. **Merged Text Cache**: Pre-compute and cache merged texts
   ```python
   merged_cache: dict[tuple[int, int], str] = {}
   
   def get_merged_text(start: int, end: int) -> str:
       if (start, end) not in merged_cache:
           merged_cache[(start, end)] = " ".join(...)
       return merged_cache[(start, end)]
   ```

2. **Similarity Cache**: Avoid recomputing same similarities
   ```python
   sim_cache: dict[tuple[str, int], float] = {}
   ```

3. **Constrained Search Space**: Limit valid ranges
   ```python
   min_i = j  # At least j segments for j ayahs
   max_i = n_seg - (n_ayah - j) + 1  # Leave room for remaining
   ```

4. **Limited Merge Count**: Max 8 segments per ayah (was 15)

5. **Early Pruning**: Skip paths that are already too costly

### Complexity Analysis

| Version | Complexity | For Surah 002 |
|---------|------------|---------------|
| Naive DP | O(n × m × k × sim) | 831 × 286 × 15 × O(sim) = Very slow |
| Optimized | O(n × m × k) with caching | 831 × 286 × 8 = ~1.9M ops, 57 seconds |

---

## 9. Results Comparison

### Final Results: Surah Al-Baqarah (286 ayahs)

| Metric | Greedy | DP | Improvement |
|--------|--------|-----|-------------|
| **Ayahs aligned** | 114 (40%) | **286 (100%)** | +172 ayahs |
| **Avg similarity** | 0.869 | 0.865 | Similar |
| **Min similarity** | 0.032 | 0.109 | +0.077 |
| **Low sim count** | 14 | 50 | Trade-off |
| **Time** | 0.34s | 57s | Slower |

### Problem Ayahs Fixed

| Ayah | Greedy | DP | Improvement |
|------|--------|-----|-------------|
| 106 | 0.044 | **0.994** | **+0.950** |
| 107 | 0.032 | **0.980** | **+0.948** |
| 110 | 0.069 | **0.973** | **+0.904** |
| 105 | 0.093 | **0.992** | **+0.899** |
| 103 | 0.096 | **0.992** | **+0.896** |

### Why DP Works Better

1. **Global Optimization**: Considers ALL possible alignments, not just local decisions
2. **No Cascade Errors**: Wrong decision in one place doesn't affect others
3. **Backtracking**: Can find optimal path through the entire sequence
4. **Handles Ambiguity**: When multiple segments could start an ayah, DP evaluates all options

---

## 10. Key Takeaways

### Algorithm Design Lessons

1. **Greedy vs DP Trade-off**
   - Greedy: Fast, simple, but can get stuck in local optima
   - DP: Slower, complex, but finds global optimum

2. **When to Use DP**
   - Problem has optimal substructure
   - Same subproblems appear multiple times
   - Need globally optimal solution
   - Can afford the time/space overhead

3. **Optimization Matters**
   - Naive DP can be impractically slow
   - Caching, pruning, and constraints make it feasible

### Problem-Solving Lessons

1. **Understand the Root Cause**
   - The false positive wasn't just a threshold issue
   - It was a fundamental problem with greedy decision-making

2. **Iterative Improvement**
   - First fix: Increase word count → Partially worked
   - Second fix: Higher threshold → Too strict
   - Third fix: Coverage check → Better but incomplete
   - Final fix: DP algorithm → Solved

3. **Trade-offs are Inevitable**
   - DP is slower but more accurate
   - Choose based on requirements (batch processing = accuracy > speed)

### Arabic NLP Lessons

1. **Normalization is Critical**
   - Diacritics, alef variants, etc. affect similarity
   - Consistent normalization enables fair comparison

2. **Similar Text Patterns**
   - Religious texts often have repeated phrases
   - Must account for this in algorithm design

---

## 11. Further Reading

### Dynamic Programming

| Resource | Type | Description |
|----------|------|-------------|
| MIT 6.006 Lectures 19-22 | Video | DP foundations from MIT OCW |
| CLRS "Introduction to Algorithms" | Book | Definitive algorithms textbook |
| LeetCode #70 (Climbing Stairs) | Practice | Basic DP problem |
| LeetCode #72 (Edit Distance) | Practice | String DP, very relevant |

### Sequence Alignment

| Resource | Type | Description |
|----------|------|-------------|
| "Biological Sequence Analysis" (Durbin) | Book | Chapters 2-3 on alignment |
| Needleman-Wunsch Wikipedia | Article | Algorithm overview |
| Smith-Waterman Algorithm | Article | Local alignment variant |

### Speech/Audio Alignment

| Resource | Type | Description |
|----------|------|-------------|
| Dynamic Time Warping tutorials | Video | Core concept for audio alignment |
| Montreal Forced Aligner | Tool | State-of-the-art forced alignment |
| "Speech and Language Processing" (Jurafsky) | Book | Chapter 9 on alignment |

### Practice Problems

| Problem | Platform | Relevance |
|---------|----------|-----------|
| Edit Distance | LeetCode #72 | String DP |
| Longest Common Subsequence | LeetCode #1143 | Sequence matching |
| Minimum Path Sum | LeetCode #64 | 2D DP |
| Regular Expression Matching | LeetCode #10 | Advanced string DP |

---

## Appendix: Code Files

| File | Purpose |
|------|---------|
| `munajjam/core/aligner.py` | Original greedy alignment algorithm |
| `munajjam/core/aligner_dp.py` | New DP-based alignment algorithm |
| `munajjam/core/matcher.py` | Similarity and coverage functions |
| `munajjam/core/arabic.py` | Arabic text normalization |
| `test_dp_aligner.py` | Test script comparing algorithms |
| `test_dp_surah002.py` | Surah 002 specific test with caching |

---

*Document created: January 2, 2026*
*Last updated: January 2, 2026*
