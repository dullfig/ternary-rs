//! f16 × f32 dot product — the inner loop of the tied-embedding output
//! projection.
//!
//! The embedding table arrives as f16 in GGUF. Expanding it to f32 at load
//! doubles the bytes streamed by the single hottest op in decode (the tied
//! output projection reads the whole table every token), so the table stays
//! f16 in memory and converts on the fly here.
//!
//! Conversion of each value is exact — f16 → f32 is lossless — but the SIMD
//! path accumulates 8-wide, so sums can differ from a scalar left-to-right
//! accumulation in the last ulp.


use crate::gguf::f16_to_f32;

/// Dot product of f32 activations against an f16 weight row.
///
/// Uses F16C hardware conversion (8 halves per instruction) with FMA
/// accumulation where available, falling back to scalar conversion
/// elsewhere. `a` and `b` must be the same length.
#[inline]
pub fn dot_f16(a: &[f32], b: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_f16 length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        // std caches the cpuid result, so this is a relaxed atomic load.
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature detection above.
            return unsafe { dot_f16_avx(a, b) };
        }
    }

    dot_f16_scalar(a, b)
}

/// Portable fallback: convert each half, multiply, accumulate in order.
fn dot_f16_scalar(a: &[f32], b: &[u16]) -> f32 {
    a.iter().zip(b).map(|(&x, &h)| x * f16_to_f32(h)).sum()
}

/// F16C + FMA path: 8 halves converted per `vcvtph2ps`.
///
/// # Safety
/// Caller must ensure AVX2, F16C and FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_f16_avx(a: &[f32], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        // 8 halves (128 bits) -> 8 f32 in one instruction
        let halves = _mm_loadu_si128(b[i..].as_ptr() as *const __m128i);
        let weights = _mm256_cvtph_ps(halves);
        let acts = _mm256_loadu_ps(a[i..].as_ptr());
        acc = _mm256_fmadd_ps(acts, weights, acc);
        i += 8;
    }

    let mut sum = hsum256_ps(acc);

    // Scalar tail for the remaining < 8 values
    while i < n {
        sum += a[i] * f16_to_f32(b[i]);
        i += 1;
    }

    sum
}

/// Horizontal sum of 8 packed f32 lanes.
///
/// # Safety
/// Requires AVX.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::f16_to_f32;

    /// Naive reference: convert then multiply-accumulate in order.
    fn reference_dot(a: &[f32], b: &[u16]) -> f32 {
        a.iter().zip(b).map(|(&x, &h)| x * f16_to_f32(h)).sum()
    }

    /// Deterministic f16 bit patterns, avoiding exp==31 (Inf/NaN).
    fn half_values(n: usize) -> Vec<u16> {
        let mut state: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let bits = (state >> 8) as u16;
                let exp = (bits >> 10) & 0x1F;
                if exp == 0x1F { bits & 0xE3FF } else { bits }
            })
            .collect()
    }

    #[test]
    fn dot_f16_matches_reference_across_lengths() {
        // Lengths straddle the 8-wide SIMD boundary so the scalar tail is covered.
        for n in [1, 7, 8, 9, 15, 16, 31, 64, 2560] {
            let b = half_values(n);
            let a: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();

            let got = dot_f16(&a, &b);
            let want = reference_dot(&a, &b);

            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "n={n}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn dot_f16_zero_length_is_zero() {
        assert_eq!(dot_f16(&[], &[]), 0.0);
    }

    #[test]
    fn dot_f16_handles_negatives() {
        // -1.0 and +2.0 in f16 bits
        let b = vec![0xBC00u16, 0x4000u16];
        let a = vec![3.0f32, 5.0f32];
        assert!((dot_f16(&a, &b) - 7.0).abs() < 1e-5);
    }
}
