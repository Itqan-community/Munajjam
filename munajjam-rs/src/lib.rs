//! Munajjam Rust Core Library
//!
//! High-performance implementations of:
//! - Arabic text normalization
//! - String similarity computation (SequenceMatcher-like)
//! - Dynamic programming alignment core

use pyo3::prelude::*;
use rayon::prelude::*;

mod arabic;
mod similarity;
mod dp_core;

pub use arabic::normalize_arabic;
pub use similarity::{similarity, compute_coverage_ratio};
pub use dp_core::{DPCell, align_segments_dp, compute_alignment_cost};

/// Python module definition
#[pymodule]
fn munajjam_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_normalize_arabic, m)?)?;
    m.add_function(wrap_pyfunction!(py_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_coverage_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_alignment_cost, m)?)?;
    m.add_function(wrap_pyfunction!(py_align_segments_dp, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_similarity, m)?)?;
    Ok(())
}

/// Normalize Arabic text for comparison (Python binding)
#[pyfunction]
#[pyo3(name = "normalize_arabic")]
fn py_normalize_arabic(text: &str) -> String {
    normalize_arabic(text)
}

/// Compute similarity ratio between two strings (Python binding)
#[pyfunction]
#[pyo3(name = "similarity", signature = (text1, text2, normalize=None))]
fn py_similarity(text1: &str, text2: &str, normalize: Option<bool>) -> f64 {
    let normalize = normalize.unwrap_or(true);
    similarity(text1, text2, normalize)
}

/// Compute coverage ratio of transcribed text vs ayah text (Python binding)
#[pyfunction]
#[pyo3(name = "compute_coverage_ratio")]
fn py_compute_coverage_ratio(transcribed_text: &str, ayah_text: &str) -> f64 {
    compute_coverage_ratio(transcribed_text, ayah_text)
}

/// Compute alignment cost (Python binding)
#[pyfunction]
#[pyo3(name = "compute_alignment_cost", signature = (merged_text, ayah_text, critical_threshold=None, critical_penalty=None))]
fn py_compute_alignment_cost(
    merged_text: &str,
    ayah_text: &str,
    critical_threshold: Option<f64>,
    critical_penalty: Option<f64>,
) -> f64 {
    let critical_threshold = critical_threshold.unwrap_or(0.4);
    let critical_penalty = critical_penalty.unwrap_or(0.5);
    compute_alignment_cost(merged_text, ayah_text, critical_threshold, critical_penalty)
}

/// Batch similarity computation for parallel processing (Python binding)
#[pyfunction]
#[pyo3(name = "batch_similarity", signature = (pairs, normalize=None))]
fn py_batch_similarity(pairs: Vec<(String, String)>, normalize: Option<bool>) -> Vec<f64> {
    let normalize = normalize.unwrap_or(true);
    pairs
        .par_iter()
        .map(|(t1, t2)| similarity(t1, t2, normalize))
        .collect()
}

/// Segment data structure for Python interop
#[derive(Clone, Debug)]
pub struct Segment {
    pub id: i32,
    pub surah_id: i32,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub seg_type: String,
}

/// Ayah data structure for Python interop
#[derive(Clone, Debug)]
pub struct Ayah {
    pub id: i32,
    pub surah_id: i32,
    pub ayah_number: i32,
    pub text: String,
}

/// Alignment result data structure for Python interop
#[derive(Clone, Debug)]
pub struct AlignmentResult {
    pub ayah_id: i32,
    pub ayah_number: i32,
    pub start_time: f64,
    pub end_time: f64,
    pub transcribed_text: String,
    pub similarity_score: f64,
    pub overlap_detected: bool,
}

/// Full DP alignment (Python binding)
#[pyfunction]
#[pyo3(name = "align_segments_dp", signature = (segments, ayahs, max_segments_per_ayah=None))]
fn py_align_segments_dp(
    segments: Vec<(i32, i32, f64, f64, String, String)>,  // (id, surah_id, start, end, text, type)
    ayahs: Vec<(i32, i32, i32, String)>,  // (id, surah_id, ayah_number, text)
    max_segments_per_ayah: Option<usize>,
) -> Vec<(i32, i32, f64, f64, String, f64, bool)> {
    let max_segments_per_ayah = max_segments_per_ayah.unwrap_or(15);

    let segments: Vec<Segment> = segments
        .into_iter()
        .map(|(id, surah_id, start, end, text, seg_type)| Segment {
            id,
            surah_id,
            start,
            end,
            text,
            seg_type,
        })
        .collect();

    let ayahs: Vec<Ayah> = ayahs
        .into_iter()
        .map(|(id, surah_id, ayah_number, text)| Ayah {
            id,
            surah_id,
            ayah_number,
            text,
        })
        .collect();

    let results = align_segments_dp(&segments, &ayahs, max_segments_per_ayah);

    results
        .into_iter()
        .map(|r| (
            r.ayah_id,
            r.ayah_number,
            r.start_time,
            r.end_time,
            r.transcribed_text,
            r.similarity_score,
            r.overlap_detected,
        ))
        .collect()
}
