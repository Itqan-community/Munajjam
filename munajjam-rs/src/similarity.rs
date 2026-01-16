//! String similarity computation
//!
//! Implements a SequenceMatcher-like algorithm for computing similarity
//! between two strings using longest common subsequence matching.

use crate::arabic::normalize_arabic;
use std::collections::HashMap;

/// Compute similarity ratio between two strings
///
/// Uses a SequenceMatcher-like algorithm to compute a ratio between 0.0 (no similarity)
/// and 1.0 (identical strings).
///
/// # Arguments
/// * `text1` - First string to compare
/// * `text2` - Second string to compare
/// * `normalize` - Whether to normalize Arabic text before comparison
///
/// # Returns
/// Similarity ratio between 0.0 and 1.0
pub fn similarity(text1: &str, text2: &str, normalize: bool) -> f64 {
    let (s1, s2) = if normalize {
        (normalize_arabic(text1), normalize_arabic(text2))
    } else {
        (text1.to_string(), text2.to_string())
    };

    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    // Use character-based similarity for Arabic text
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();

    sequence_matcher_ratio(&chars1, &chars2)
}

/// SequenceMatcher-like ratio computation
///
/// This implements the same algorithm as Python's difflib.SequenceMatcher.ratio()
/// which computes 2.0 * M / T where:
/// - M is the number of matches (characters in matching blocks)
/// - T is the total number of elements in both sequences
fn sequence_matcher_ratio(a: &[char], b: &[char]) -> f64 {
    let matches = count_matching_chars(a, b);
    let total = a.len() + b.len();

    if total == 0 {
        return 1.0;
    }

    2.0 * matches as f64 / total as f64
}

/// Count matching characters using longest common subsequence matching blocks
fn count_matching_chars(a: &[char], b: &[char]) -> usize {
    // Build a map of character positions in b for quick lookup
    let mut b_positions: HashMap<char, Vec<usize>> = HashMap::new();
    for (i, &c) in b.iter().enumerate() {
        b_positions.entry(c).or_default().push(i);
    }

    // Find matching blocks and sum their sizes
    find_matching_blocks(a, b, &b_positions)
        .iter()
        .map(|(_, _, size)| size)
        .sum()
}

/// Find matching blocks between two sequences
///
/// Returns a list of (a_start, b_start, size) tuples representing matching blocks
fn find_matching_blocks(
    a: &[char],
    b: &[char],
    b_positions: &HashMap<char, Vec<usize>>,
) -> Vec<(usize, usize, usize)> {
    let mut blocks = Vec::new();
    find_matching_blocks_recursive(a, b, 0, a.len(), 0, b.len(), b_positions, &mut blocks);
    blocks
}

/// Recursive helper for finding matching blocks
fn find_matching_blocks_recursive(
    a: &[char],
    b: &[char],
    a_lo: usize,
    a_hi: usize,
    b_lo: usize,
    b_hi: usize,
    b_positions: &HashMap<char, Vec<usize>>,
    blocks: &mut Vec<(usize, usize, usize)>,
) {
    // Find longest matching block in the specified range
    if let Some((a_start, b_start, size)) =
        find_longest_match(a, b, a_lo, a_hi, b_lo, b_hi, b_positions)
    {
        if size > 0 {
            // Recursively find matches before this block
            if a_lo < a_start && b_lo < b_start {
                find_matching_blocks_recursive(
                    a,
                    b,
                    a_lo,
                    a_start,
                    b_lo,
                    b_start,
                    b_positions,
                    blocks,
                );
            }

            blocks.push((a_start, b_start, size));

            // Recursively find matches after this block
            if a_start + size < a_hi && b_start + size < b_hi {
                find_matching_blocks_recursive(
                    a,
                    b,
                    a_start + size,
                    a_hi,
                    b_start + size,
                    b_hi,
                    b_positions,
                    blocks,
                );
            }
        }
    }
}

/// Find the longest matching block in the specified ranges
fn find_longest_match(
    a: &[char],
    b: &[char],
    a_lo: usize,
    a_hi: usize,
    b_lo: usize,
    b_hi: usize,
    b_positions: &HashMap<char, Vec<usize>>,
) -> Option<(usize, usize, usize)> {
    let mut best_i = a_lo;
    let mut best_j = b_lo;
    let mut best_size = 0;

    // j2len[j] = length of longest match ending with a[i-1] and b[j-1]
    let mut j2len: HashMap<usize, usize> = HashMap::new();

    for i in a_lo..a_hi {
        let mut new_j2len: HashMap<usize, usize> = HashMap::new();

        if let Some(positions) = b_positions.get(&a[i]) {
            for &j in positions {
                if j < b_lo {
                    continue;
                }
                if j >= b_hi {
                    break;
                }

                let k = j2len.get(&(j.wrapping_sub(1))).copied().unwrap_or(0) + 1;
                new_j2len.insert(j, k);

                if k > best_size {
                    best_i = i + 1 - k;
                    best_j = j + 1 - k;
                    best_size = k;
                }
            }
        }

        j2len = new_j2len;
    }

    // Extend the match as far as possible (handle edge cases)
    while best_i > a_lo
        && best_j > b_lo
        && a.get(best_i - 1) == b.get(best_j - 1)
    {
        best_i -= 1;
        best_j -= 1;
        best_size += 1;
    }

    while best_i + best_size < a_hi
        && best_j + best_size < b_hi
        && a.get(best_i + best_size) == b.get(best_j + best_size)
    {
        best_size += 1;
    }

    if best_size > 0 {
        Some((best_i, best_j, best_size))
    } else {
        None
    }
}

/// Compute the word coverage ratio of transcribed text vs ayah text
///
/// # Arguments
/// * `transcribed_text` - Text from transcription
/// * `ayah_text` - Reference ayah text
///
/// # Returns
/// Ratio of transcribed words to ayah words (can be > 1.0)
pub fn compute_coverage_ratio(transcribed_text: &str, ayah_text: &str) -> f64 {
    let trans_normalized = normalize_arabic(transcribed_text);
    let ayah_normalized = normalize_arabic(ayah_text);

    let trans_words = if trans_normalized.is_empty() {
        0
    } else {
        trans_normalized.split_whitespace().count()
    };

    let ayah_words = if ayah_normalized.is_empty() {
        0
    } else {
        ayah_normalized.split_whitespace().count()
    };

    if ayah_words == 0 {
        return 0.0;
    }

    trans_words as f64 / ayah_words as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_identical() {
        let text = "بسم الله الرحمن الرحيم";
        assert!((similarity(text, text, true) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_empty() {
        assert!((similarity("", "", true) - 1.0).abs() < 0.001);
        assert!((similarity("test", "", true) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_with_diacritics() {
        let text1 = "بِسْمِ اللَّهِ";
        let text2 = "بسم الله";
        // With normalization, these should be very similar
        let sim = similarity(text1, text2, true);
        assert!(sim > 0.9);
    }

    #[test]
    fn test_coverage_ratio() {
        let trans = "بسم الله";
        let ayah = "بسم الله الرحمن الرحيم";
        let ratio = compute_coverage_ratio(trans, ayah);
        assert!((ratio - 0.5).abs() < 0.001);
    }
}
