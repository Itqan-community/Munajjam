# Effects of Increasing Number of Words Checked

## Current Implementation

Currently, the algorithm uses an **adaptive approach**:
- **3+ words in ayah** → Check **3 words**
- **2 words in ayah** → Check **2 words**  
- **1 word in ayah** → Check **1 word**

## What Happens If We Check More Words (5-6)?

### Example Scenario

**Ayah:** "بسم الله الرحمن الرحيم" (4 words)
- **Current:** Checks last **3 words**: "الرحمن الرحيم"
- **If 5-6 words:** Would check last **4 words**: "بسم الله الرحمن الرحيم" (all words)

---

## Benefits of Checking More Words (5-6)

### ✅ 1. **More Context = Higher Confidence**

**Example:**
```
Ayah: "وما أدراك ما القارعة"
Transcribed: "وما أدراك ما القارعة"

Current (3 words): Checks "ما القارعة" → similarity = 1.0 ✅
With 5 words: Checks "وما أدراك ما القارعة" → similarity = 1.0 ✅

But if transcription has error:
Transcribed: "وما أدراك ما القارع" (missing last letter)

Current (3 words): Checks "ما القارع" vs "ما القارعة" → similarity = 0.8 ✅ (might pass)
With 5 words: Checks full text → similarity = 0.95 ✅ (still passes, more robust)
```

**Benefit:** More words = more context = harder to get false positives from partial matches

### ✅ 2. **Better for Long Ayahs**

**Example:**
```
Long Ayah: "إن الذين كفروا سواء عليهم أأنذرتهم أم لم تنذرهم لا يؤمنون" (10 words)

Current: Checks last 3 words → "لا يؤمنون"
With 5-6: Checks last 5-6 words → "أأنذرتهم أم لم تنذرهم لا يؤمنون"

More words = more unique context to identify the boundary
```

**Benefit:** Long ayahs have more unique endings, so checking more words helps distinguish them

### ✅ 3. **More Robust Against Mid-Ayah Errors**

**Example:**
```
Ayah: "بسم الله الرحمن الرحيم"
Transcribed: "بسم الله الرحم الرحيم" (error in middle)

Current (3 words): Checks "الرحم الرحيم" vs "الرحمن الرحيم" 
  → similarity = 0.7 ✅ (might pass despite error)

With 5 words: Checks "بسم الله الرحم الرحيم" vs "بسم الله الرحمن الرحيم"
  → similarity = 0.85 ✅ (still passes, error doesn't affect boundary detection)
```

**Benefit:** Errors in the middle don't affect boundary detection as much

---

## Drawbacks of Checking More Words (5-6)

### ❌ 1. **Stricter Matching = Harder to Pass**

**Example:**
```
Ayah: "بسم الله الرحمن الرحيم"
Transcribed: "بسم الله الرحمن الرحي" (missing last letter)

Current (3 words): Checks "الرحمن الرحي" vs "الرحمن الرحيم"
  → similarity = 0.85 ✅ (passes threshold 0.6)

With 5 words: Checks "بسم الله الرحمن الرحي" vs "بسم الله الرحمن الرحيم"
  → similarity = 0.92 ✅ (still passes, but more sensitive to errors)
```

**Problem:** If transcription has errors near the end, checking more words makes it harder to pass

### ❌ 2. **Might Miss Boundaries**

**Example:**
```
Ayah: "بسم الله الرحمن الرحيم"
Transcribed segments:
  Segment 1: "بسم الله الرحمن"
  Segment 2: "الرحيم" (but transcription error: "الرحي")

Current (3 words): 
  After merge: "بسم الله الرحمن الرحي"
  Checks "الرحمن الرحي" vs "الرحمن الرحيم" → similarity = 0.85 ✅
  → Might still pass and finalize correctly

With 5 words:
  Checks "بسم الله الرحمن الرحي" vs "بسم الله الرحمن الرحيم"
  → similarity = 0.92 ✅
  → But if error is worse, might fail and keep merging incorrectly
```

**Problem:** More words = more chances for errors to accumulate

### ❌ 3. **Short Ayahs Can't Benefit**

**Example:**
```
Short Ayah: "قل" (1 word)
Current: Checks 1 word ✅
With 5 words: Still checks 1 word (code uses min(5, 1) = 1) ✅

But for 2-word ayahs:
Ayah: "قل هو"
Current: Checks 2 words ✅
With 5 words: Still checks 2 words (min(5, 2) = 2) ✅
```

**Note:** Code handles this with `min(n_words, len(ayah_words))`, so short ayahs aren't affected

### ❌ 4. **Might Cause Over-Merging**

**Example:**
```
Ayah 1: "بسم الله الرحمن الرحيم"
Ayah 2: "الحمد لله رب العالمين"

Transcribed:
  Segment 1: "بسم الله الرحمن"
  Segment 2: "الرحيم الحمد" (overlap/error)

Current (3 words):
  Checks if Segment 2 starts Ayah 2: "الرحيم الحمد" vs "الحمد لله رب"
  → similarity = 0.2 ❌ (correctly doesn't match)
  → Keeps Segment 2 with Ayah 1 ✅

With 5 words:
  Checks "الرحيم الحمد" vs "الحمد لله رب العالمين"
  → similarity = 0.15 ❌ (still doesn't match)
  → But if threshold is lower or error is different, might incorrectly merge
```

**Problem:** More words = more complex comparison = harder to distinguish boundaries

---

## Real-World Impact

### Scenario 1: Perfect Transcription
```
✅ More words (5-6): Better - more confidence, fewer false positives
✅ Current (3): Good - sufficient for most cases
```

### Scenario 2: Transcription with Errors
```
⚠️ More words (5-6): Mixed - more robust to mid-ayah errors, but stricter for end errors
✅ Current (3): Better - more forgiving, focuses on boundary words
```

### Scenario 3: Long Ayahs
```
✅ More words (5-6): Better - more unique context
✅ Current (3): Good - sufficient for most long ayahs
```

### Scenario 4: Short Ayahs
```
✅ Both: Same - code adapts automatically
```

---

## Recommendation

### **Keep Current Adaptive Approach (1-3 words)**

**Reasons:**
1. **Boundary detection focuses on endings** - Last 3 words are usually sufficient
2. **More forgiving** - Handles transcription errors better
3. **Faster** - Less computation (though negligible)
4. **Proven** - Works well in practice
5. **Adaptive** - Already handles different ayah lengths

### **When to Consider More Words (5-6)**

Consider increasing if:
- ✅ You have very long ayahs (>10 words) that are hard to distinguish
- ✅ Transcription quality is very high (few errors)
- ✅ You're getting too many false positives with 3 words
- ✅ You want stricter boundary detection

### **How to Test**

If you want to experiment, you could modify `_get_n_check_words()`:

```python
def _get_n_check_words(ayah_text: str) -> int:
    words = ayah_text.strip().split()
    if len(words) >= 6:
        return 6      # Very long: check 6 words
    elif len(words) >= 3:
        return 3      # Long: check 3 words
    elif len(words) == 2:
        return 2      # Medium: check 2 words
    return 1          # Short: check 1 word
```

Then test on your data and compare:
- Number of ayahs aligned
- Average similarity scores
- False positive/negative rates

---

## Summary Table

| Aspect | Current (1-3 words) | More Words (5-6) |
|--------|---------------------|------------------|
| **Confidence** | Good | Better |
| **False Positives** | Low | Lower |
| **Error Tolerance** | High | Medium |
| **Long Ayahs** | Good | Better |
| **Short Ayahs** | Perfect | Same |
| **Computation** | Fast | Slightly slower |
| **Strictness** | Moderate | Higher |
| **Recommendation** | ✅ **Use this** | ⚠️ Test first |

**Bottom Line:** Current approach (1-3 words) is a good balance. Only increase if you have specific issues with long ayahs or want stricter matching.
