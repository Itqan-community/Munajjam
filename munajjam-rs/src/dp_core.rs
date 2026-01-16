//! Core Dynamic Programming alignment algorithm
//!
//! Contains the fundamental DP components for aligning transcribed segments
//! to reference Quran ayahs.

use crate::similarity::{similarity, compute_coverage_ratio};
use crate::{Segment, Ayah, AlignmentResult};
use std::collections::HashMap;

/// A cell in the DP matrix
#[derive(Clone, Debug)]
pub struct DPCell {
    /// Accumulated cost to reach this cell
    pub cost: f64,
    /// Text merged so far for current ayah
    pub merged_text: String,
    /// Index of first segment in current ayah
    pub seg_start_idx: usize,
    /// Previous cell for backtracking (i, j)
    pub parent: Option<(usize, usize)>,
}

/// Compute the cost of aligning merged segments to an ayah
///
/// Lower cost = better alignment.
///
/// # Arguments
/// * `merged_text` - The merged segment text
/// * `ayah_text` - The reference ayah text
/// * `critical_threshold` - Similarity below this triggers heavy penalty
/// * `critical_penalty` - Extra penalty added when below critical threshold
///
/// # Returns
/// Value between 0 (perfect match) and higher values for worse matches
pub fn compute_alignment_cost(
    merged_text: &str,
    ayah_text: &str,
    critical_threshold: f64,
    critical_penalty: f64,
) -> f64 {
    let merged_trimmed = merged_text.trim();
    if merged_trimmed.is_empty() {
        return 1.5; // High cost for empty text
    }

    // Primary: text similarity
    let sim = similarity(merged_text, ayah_text, true);

    // Secondary: coverage ratio penalty
    let coverage = compute_coverage_ratio(merged_text, ayah_text);

    // Penalize under-coverage (missing words) and over-coverage (extra words)
    let coverage_penalty = if coverage < 0.7 {
        (0.7 - coverage) * 0.5 // Up to 0.35 penalty
    } else if coverage > 1.3 {
        (coverage - 1.3) * 0.3 // Penalty for too much text
    } else {
        0.0
    };

    // Cost = 1 - similarity + coverage penalty
    let mut cost = (1.0 - sim) + coverage_penalty;

    // CRITICAL THRESHOLD: Heavy penalty for very low similarity
    if sim < critical_threshold {
        cost += critical_penalty * (critical_threshold - sim) / critical_threshold;
    }

    cost.max(0.0)
}

/// Filter out non-ayah segments (istiadha, basmala for non-Fatiha)
fn filter_special_segments(segments: &[Segment], ayahs: &[Ayah]) -> Vec<Segment> {
    if segments.is_empty() || ayahs.is_empty() {
        return segments.to_vec();
    }

    let is_fatiha = ayahs[0].surah_id == 1;

    segments
        .iter()
        .filter(|seg| {
            if seg.seg_type == "istiadha" {
                return false;
            }
            if seg.seg_type == "basmala" && !is_fatiha {
                return false;
            }
            true
        })
        .cloned()
        .collect()
}

/// Align segments to ayahs using dynamic programming
///
/// Finds the globally optimal alignment that minimizes total cost,
/// allowing multiple segments to be merged into each ayah.
///
/// # Arguments
/// * `segments` - List of transcribed segments
/// * `ayahs` - List of reference ayahs
/// * `max_segments_per_ayah` - Maximum segments that can be merged into one ayah
///
/// # Returns
/// List of AlignmentResult objects
pub fn align_segments_dp(
    segments: &[Segment],
    ayahs: &[Ayah],
    max_segments_per_ayah: usize,
) -> Vec<AlignmentResult> {
    if segments.is_empty() || ayahs.is_empty() {
        return Vec::new();
    }

    let segments = filter_special_segments(segments, ayahs);
    let n_seg = segments.len();
    let n_ayah = ayahs.len();

    // Pre-compute merged texts for efficiency
    let mut merged_cache: HashMap<(usize, usize), String> = HashMap::new();

    let get_merged_text = |cache: &mut HashMap<(usize, usize), String>, start: usize, end: usize, segs: &[Segment]| -> String {
        if let Some(text) = cache.get(&(start, end)) {
            return text.clone();
        }
        let text: String = segs[start..end]
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        cache.insert((start, end), text.clone());
        text
    };

    // DP table
    let mut dp: HashMap<(usize, usize), DPCell> = HashMap::new();
    dp.insert(
        (0, 0),
        DPCell {
            cost: 0.0,
            merged_text: String::new(),
            seg_start_idx: 0,
            parent: None,
        },
    );

    // Fill DP table
    for j in 1..=n_ayah {
        let ayah = &ayahs[j - 1];

        // Relaxed windowing for better coverage
        let min_i = j.max(1);
        let remaining_ayahs = n_ayah - j;
        let min_segs_needed = remaining_ayahs.max(1);
        let max_i = (n_seg + 1).min(n_seg - min_segs_needed + 1 + max_segments_per_ayah);
        let max_i = max_i.max(min_i + 1);

        for i in min_i..max_i {
            let mut best_cell: Option<DPCell> = None;
            let mut best_cost = f64::INFINITY;

            let max_k = max_segments_per_ayah.min(i);

            for k in 1..=max_k {
                let prev_i = i - k;
                let prev_j = j - 1;

                if let Some(prev_cell) = dp.get(&(prev_i, prev_j)) {
                    // Early pruning
                    if prev_cell.cost > best_cost {
                        continue;
                    }

                    let merged_text = get_merged_text(&mut merged_cache, prev_i, i, &segments);
                    let cost = compute_alignment_cost(&merged_text, &ayah.text, 0.4, 0.5);
                    let total_cost = prev_cell.cost + cost;

                    if total_cost < best_cost {
                        best_cost = total_cost;
                        best_cell = Some(DPCell {
                            cost: total_cost,
                            merged_text,
                            seg_start_idx: prev_i,
                            parent: Some((prev_i, prev_j)),
                        });
                    }
                }
            }

            if let Some(cell) = best_cell {
                dp.insert((i, j), cell);
            }
        }
    }

    // Find best ending point
    let mut best_end: Option<(usize, usize)> = None;
    let mut best_end_cost = f64::INFINITY;

    for i in n_ayah..=n_seg {
        if let Some(cell) = dp.get(&(i, n_ayah)) {
            if cell.cost < best_end_cost {
                best_end_cost = cell.cost;
                best_end = Some((i, n_ayah));
            }
        }
    }

    // Fallback if no complete path found
    if best_end.is_none() {
        for j in (1..=n_ayah).rev() {
            for i in (1..=n_seg).rev() {
                if dp.contains_key(&(i, j)) {
                    best_end = Some((i, j));
                    break;
                }
            }
            if best_end.is_some() {
                break;
            }
        }
    }

    let best_end = match best_end {
        Some(e) => e,
        None => return Vec::new(),
    };

    // Backtrack to reconstruct alignment
    let mut path: Vec<(usize, usize, usize, String)> = Vec::new();
    let mut current = Some(best_end);

    while let Some((i, j)) = current {
        if let Some(cell) = dp.get(&(i, j)) {
            if let Some((prev_i, _)) = cell.parent {
                path.push((prev_i, i, j, cell.merged_text.clone()));
            }
            current = cell.parent;
        } else {
            break;
        }
    }

    path.reverse();

    // Convert path to AlignmentResult objects
    let mut results: Vec<AlignmentResult> = Vec::new();

    for (seg_start, seg_end, ayah_idx, merged_text) in path {
        if seg_start >= segments.len() || seg_end > segments.len() {
            continue;
        }

        let ayah = &ayahs[ayah_idx - 1];
        let start_time = segments[seg_start].start;
        let end_time = segments[seg_end - 1].end;

        let sim_score = similarity(&merged_text, &ayah.text, true);

        results.push(AlignmentResult {
            ayah_id: ayah.id,
            ayah_number: ayah.ayah_number,
            start_time,
            end_time,
            transcribed_text: merged_text,
            similarity_score: sim_score,
            overlap_detected: false,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_alignment_cost() {
        // Perfect match should have low cost
        let cost = compute_alignment_cost("بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم", 0.4, 0.5);
        assert!(cost < 0.1);

        // Empty text should have high cost
        let cost = compute_alignment_cost("", "بسم الله", 0.4, 0.5);
        assert!(cost > 1.0);
    }

    #[test]
    fn test_filter_special_segments() {
        let segments = vec![
            Segment {
                id: 1,
                surah_id: 1,
                start: 0.0,
                end: 1.0,
                text: "أعوذ بالله".to_string(),
                seg_type: "istiadha".to_string(),
            },
            Segment {
                id: 2,
                surah_id: 1,
                start: 1.0,
                end: 2.0,
                text: "بسم الله".to_string(),
                seg_type: "ayah".to_string(),
            },
        ];

        let ayahs = vec![Ayah {
            id: 1,
            surah_id: 1,
            ayah_number: 1,
            text: "بسم الله".to_string(),
        }];

        let filtered = filter_special_segments(&segments, &ayahs);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, 2);
    }
}
