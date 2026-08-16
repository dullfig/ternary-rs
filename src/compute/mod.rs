//! Compute backend abstraction for ternary inference.
//!
//! Provides a `ComputeBackend` trait that abstracts over hardware-specific
//! implementations (scalar, AVX2, future GPU). The `detect()` function
//! auto-selects the fastest backend for the current CPU.
//!
//! ## Architecture
//!
//! ```text
//!   ComputeBackend (trait)
//!       ├── ScalarBackend     — portable, no SIMD
//!       ├── Avx2Backend       — x86-64 AVX2 (32-wide i8 ops)
//!       └── WgpuBackend       — GPU via wgpu/Vulkan/DX12/Metal (future)
//! ```

pub mod scalar;
pub mod device;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;
#[cfg(feature = "gpu")]
pub mod gpu_engine;

use crate::tensor::TernaryTensor;

/// Hardware-independent compute interface for ternary inference kernels.
///
/// Each method corresponds to a hot-path operation in the transformer forward
/// pass. Backends implement whichever operations they can accelerate; the
/// trait provides default implementations that fall back to scalar code.
pub trait ComputeBackend: Send + Sync + std::fmt::Debug {
    /// Human-readable name for logging (e.g., "scalar", "avx2").
    fn name(&self) -> &str;

    /// Ternary matrix-vector product: y = W · x.
    ///
    /// `weights`: packed ternary (out_features × in_features).
    /// `input`: quantized 8-bit activations (in_features), must be in [-127, 127].
    ///          (Absmax quantization guarantees this. The -128 value is excluded
    ///          because SIMD `sign_epi8` can't negate it without overflow.)
    /// Returns: i32 accumulator per output feature.
    fn ternary_matvec(&self, weights: &TernaryTensor, input: &[i8]) -> Vec<i32>;

    /// RMS normalization: x_i * (w_i / rms), where rms = sqrt(mean(x²) + eps).
    fn rmsnorm(&self, input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        input.iter().zip(weight).map(|(&x, &w)| x * inv_rms * w).collect()
    }

    /// Softmax over a slice (in-place would be ideal, but returning Vec is fine
    /// for now — the bottleneck is matmul, not softmax).
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Element-wise multiply: out[i] = a[i] * b[i].
    fn elementwise_mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(&x, &y)| x * y).collect()
    }
}

/// Capability flags detected at runtime.
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU features using `std::arch::is_x86_feature_detected` (x86)
    /// or compile-time target detection (ARM).
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                avx512f: false,
                neon: true, // NEON is mandatory on aarch64
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx2: false,
                avx512f: false,
                neon: false,
            }
        }
    }
}

/// Environment variable that pins the backend, bypassing measurement.
pub const BACKEND_ENV: &str = "TERNARY_BACKEND";

/// Probe matrix dimensions used by [`fastest_of`]. Big enough to be
/// representative of a real projection, small enough that the whole
/// selection costs a few milliseconds at startup.
const PROBE_ROWS: usize = 1024;
const PROBE_COLS: usize = 1024;
const PROBE_ITERS: usize = 3;

/// Look up a backend by name. Returns `None` for an unknown name, or for a
/// backend that isn't available on this machine (no AVX2, no working GPU).
///
/// Accepted: `scalar`, `avx2`, `wgpu` (alias `gpu`). Case-insensitive.
pub fn backend_by_name(name: &str) -> Option<std::sync::Arc<dyn ComputeBackend>> {
    match name.trim().to_ascii_lowercase().as_str() {
        "scalar" => Some(std::sync::Arc::new(scalar::ScalarBackend)),

        #[cfg(target_arch = "x86_64")]
        "avx2" => std::arch::is_x86_feature_detected!("avx2")
            .then(|| std::sync::Arc::new(avx2::Avx2Backend) as std::sync::Arc<dyn ComputeBackend>),

        #[cfg(feature = "gpu")]
        "wgpu" | "gpu" => wgpu_backend::WgpuBackend::try_new()
            .map(|g| std::sync::Arc::new(g) as std::sync::Arc<dyn ComputeBackend>),

        _ => None,
    }
}

/// Build a deterministic ternary matrix + activation vector for backend probing.
fn probe_workload() -> (TernaryTensor, Vec<i8>) {
    use crate::tensor::Ternary;

    let n = PROBE_ROWS * PROBE_COLS;
    let mut values = Vec::with_capacity(n);
    // Cheap LCG — we want a fixed, non-degenerate weight pattern, not entropy.
    let mut state: u32 = 0x2545_F491;
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(match (state >> 16) % 3 {
            0 => Ternary::Neg,
            1 => Ternary::Zero,
            _ => Ternary::Pos,
        });
    }

    let weights = TernaryTensor::pack(&values, PROBE_ROWS, PROBE_COLS);
    // Absmax-quantized activations live in [-127, 127].
    let input: Vec<i8> = (0..PROBE_COLS)
        .map(|i| ((i % 255) as i32 - 127) as i8)
        .collect();

    (weights, input)
}

/// Time each candidate on a representative matvec and return the fastest.
///
/// This is the measurement that replaces the old presence-driven heuristic.
/// The previous logic assumed "discrete GPU beats AVX2", which is false for
/// the current per-call-upload GPU kernel — on an i9-14900HX + RTX 4080 it
/// picked a backend ~5x slower than the CPU one.
pub fn fastest_of(
    candidates: Vec<std::sync::Arc<dyn ComputeBackend>>,
) -> std::sync::Arc<dyn ComputeBackend> {
    match candidates.len() {
        0 => return std::sync::Arc::new(scalar::ScalarBackend),
        1 => return candidates.into_iter().next().expect("len checked"),
        _ => {}
    }

    let (weights, input) = probe_workload();
    let mut best: Option<(std::sync::Arc<dyn ComputeBackend>, f64)> = None;

    for backend in candidates {
        // Warmup: absorbs GPU shader compilation and first-touch page faults,
        // which would otherwise be charged to whichever backend ran first.
        let _ = backend.ternary_matvec(&weights, &input);

        let start = std::time::Instant::now();
        for _ in 0..PROBE_ITERS {
            let _ = backend.ternary_matvec(&weights, &input);
        }
        let secs = start.elapsed().as_secs_f64();

        tracing::debug!(backend = backend.name(), secs, "backend probe");
        if best.as_ref().is_none_or(|(_, b)| secs < *b) {
            best = Some((backend, secs));
        }
    }

    let (backend, _) = best.expect("at least two candidates");
    backend
}

/// Auto-detect the fastest available backend by measuring it.
///
/// Set `TERNARY_BACKEND` (`scalar` | `avx2` | `wgpu`) to pin one and skip
/// the probe. An unknown or unavailable name falls through to measurement.
pub fn detect() -> std::sync::Arc<dyn ComputeBackend> {
    let features = CpuFeatures::detect();
    tracing::info!(?features, "detecting compute backend");

    if let Ok(name) = std::env::var(BACKEND_ENV) {
        match backend_by_name(&name) {
            Some(backend) => {
                tracing::info!(backend = backend.name(), "backend pinned via {BACKEND_ENV}");
                return backend;
            }
            None => tracing::warn!("{BACKEND_ENV}={name} is unknown or unavailable; measuring"),
        }
    }

    // Only the best CPU kernel competes — scalar never beats AVX2, so racing
    // it would just add startup cost.
    let mut candidates: Vec<std::sync::Arc<dyn ComputeBackend>> = Vec::new();

    #[cfg(target_arch = "x86_64")]
    if features.avx2 {
        candidates.push(std::sync::Arc::new(avx2::Avx2Backend));
    }
    if candidates.is_empty() {
        candidates.push(std::sync::Arc::new(scalar::ScalarBackend));
    }

    #[cfg(feature = "gpu")]
    if let Some(gpu) = wgpu_backend::WgpuBackend::try_new() {
        candidates.push(std::sync::Arc::new(gpu));
    }

    let backend = fastest_of(candidates);
    tracing::info!(backend = backend.name(), "selected compute backend");
    backend
}

/// Auto-detect the fastest CPU-only backend (skip GPU).
///
/// Useful when you want deterministic CPU results or when the GPU
/// is reserved for other work.
pub fn detect_cpu_only() -> std::sync::Arc<dyn ComputeBackend> {
    let features = CpuFeatures::detect();

    #[cfg(target_arch = "x86_64")]
    if features.avx2 {
        return std::sync::Arc::new(avx2::Avx2Backend);
    }

    std::sync::Arc::new(scalar::ScalarBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_backend() {
        let backend = detect();
        let name = backend.name();
        assert!(!name.is_empty());
        eprintln!("detected backend: {name}");
    }

    #[test]
    fn backend_by_name_returns_named_backend() {
        let backend = backend_by_name("scalar").expect("scalar is always available");
        assert_eq!(backend.name(), "scalar");
    }

    #[test]
    fn backend_by_name_rejects_unknown() {
        assert!(backend_by_name("definitely-not-a-backend").is_none());
    }

    #[test]
    fn backend_by_name_is_case_insensitive() {
        let backend = backend_by_name("SCALAR").expect("case should not matter");
        assert_eq!(backend.name(), "scalar");
    }

    #[test]
    fn fastest_of_single_candidate_returns_it() {
        let only: Vec<std::sync::Arc<dyn ComputeBackend>> =
            vec![std::sync::Arc::new(scalar::ScalarBackend)];
        assert_eq!(fastest_of(only).name(), "scalar");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fastest_of_prefers_avx2_over_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return; // nothing to compare on this host
        }
        // AVX2 measures ~90x faster than scalar on ternary matvec, so this
        // margin is far too wide to flake on a noisy machine.
        let candidates: Vec<std::sync::Arc<dyn ComputeBackend>> = vec![
            std::sync::Arc::new(scalar::ScalarBackend),
            std::sync::Arc::new(avx2::Avx2Backend),
        ];
        assert_eq!(fastest_of(candidates).name(), "avx2");
    }

    #[test]
    fn cpu_features_detect() {
        let f = CpuFeatures::detect();
        eprintln!("CPU features: avx2={}, avx512f={}, neon={}", f.avx2, f.avx512f, f.neon);
    }

    #[test]
    fn default_rmsnorm() {
        let backend = scalar::ScalarBackend;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let result = backend.rmsnorm(&input, &weight, 1e-6);
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32 + 1e-6).sqrt();
        for (i, &v) in result.iter().enumerate() {
            let expected = input[i] / rms;
            assert!((v - expected).abs() < 1e-5, "rmsnorm[{i}]: expected {expected}, got {v}");
        }
    }

    #[test]
    fn default_softmax() {
        let backend = scalar::ScalarBackend;
        let result = backend.softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }
}
