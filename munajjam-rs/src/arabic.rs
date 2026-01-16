//! Arabic text normalization utilities
//!
//! Provides fast normalization of Arabic text for comparison:
//! - Normalize alef variants to plain alef
//! - Normalize alef maqsura to ya
//! - Normalize ta marbuta to ha
//! - Remove diacritics
//! - Remove punctuation
//! - Collapse whitespace

/// Arabic diacritics Unicode range (U+064B to U+0652)
const DIACRITICS: &[char] = &[
    '\u{064B}', // Fathatan
    '\u{064C}', // Dammatan
    '\u{064D}', // Kasratan
    '\u{064E}', // Fatha
    '\u{064F}', // Damma
    '\u{0650}', // Kasra
    '\u{0651}', // Shadda
    '\u{0652}', // Sukun
    '\u{0653}', // Maddah
    '\u{0654}', // Hamza above
    '\u{0655}', // Hamza below
    '\u{0656}', // Subscript alef
    '\u{0657}', // Inverted damma
    '\u{0658}', // Mark noon ghunna
    '\u{0659}', // Zwarakay
    '\u{065A}', // Vowel sign small v above
    '\u{065B}', // Vowel sign inverted small v above
    '\u{065C}', // Vowel sign dot below
    '\u{065D}', // Reversed damma
    '\u{065E}', // Fatha with two dots
    '\u{065F}', // Wavy hamza below
    '\u{0670}', // Superscript alef
];

/// Alef variants that should be normalized to plain alef (ا)
const ALEF_VARIANTS: &[char] = &[
    '\u{0623}', // Alef with hamza above (أ)
    '\u{0625}', // Alef with hamza below (إ)
    '\u{0622}', // Alef with madda (آ)
    '\u{0627}', // Plain alef (ا)
    '\u{0671}', // Alef wasla (ٱ)
];

/// Normalize Arabic text for comparison
///
/// Performs the following normalizations:
/// - Replace all alef variants (أ إ آ ا ٱ) with plain alef (ا)
/// - Replace alef maqsura (ى) with ya (ي)
/// - Replace ta marbuta (ة) with ha (ه)
/// - Remove diacritics (tashkeel)
/// - Remove punctuation
/// - Collapse multiple spaces
pub fn normalize_arabic(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(text.len());
    let mut last_was_space = true; // Start true to skip leading spaces

    for c in text.chars() {
        // Skip diacritics
        if DIACRITICS.contains(&c) {
            continue;
        }

        // Normalize alef variants
        if ALEF_VARIANTS.contains(&c) {
            result.push('\u{0627}'); // Plain alef
            last_was_space = false;
            continue;
        }

        // Normalize alef maqsura to ya
        if c == '\u{0649}' {
            result.push('\u{064A}'); // Ya
            last_was_space = false;
            continue;
        }

        // Normalize ta marbuta to ha
        if c == '\u{0629}' {
            result.push('\u{0647}'); // Ha
            last_was_space = false;
            continue;
        }

        // Handle whitespace - collapse multiple spaces
        if c.is_whitespace() {
            if !last_was_space && !result.is_empty() {
                result.push(' ');
                last_was_space = true;
            }
            continue;
        }

        // Skip punctuation (keep only alphanumeric and Arabic letters)
        if c.is_alphanumeric() || is_arabic_letter(c) {
            result.push(c);
            last_was_space = false;
        }
    }

    // Remove trailing space
    if result.ends_with(' ') {
        result.pop();
    }

    result
}

/// Check if a character is an Arabic letter
#[inline]
fn is_arabic_letter(c: char) -> bool {
    let code = c as u32;
    // Arabic block: U+0600 to U+06FF
    // Arabic Supplement: U+0750 to U+077F
    // Arabic Extended-A: U+08A0 to U+08FF
    (0x0600..=0x06FF).contains(&code)
        || (0x0750..=0x077F).contains(&code)
        || (0x08A0..=0x08FF).contains(&code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_arabic() {
        // Test basic normalization
        let input = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ";
        let expected = "بسم الله الرحمن الرحيم";
        assert_eq!(normalize_arabic(input), expected);
    }

    #[test]
    fn test_normalize_alef_variants() {
        // Test alef variant normalization
        assert_eq!(normalize_arabic("أَعُوذُ"), "اعوذ");
        assert_eq!(normalize_arabic("إِلَيْهِ"), "اليه");
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize_arabic(""), "");
    }
}
