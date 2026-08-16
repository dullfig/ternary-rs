# TurboQuant KV Cache for BitNet — Design Document

**Date:** 2026-03-26
**Source:** [turboquant.net](https://turboquant.net) (Google Research)
**Target crate:** ~~`crates/bitnet`~~ → **this repo** (`src/layers/kv_cache.rs`)

> **Rescued 2026-08-16.** Written against `AgentOS/crates/bitnet`, which was deleted that day as
> a five-month-stale duplicate of this repo (ternary-rs contains every one of its public items
> plus the GPU substrate). This file was **gitignored** in AgentOS, so it existed only on disk
> and would have been destroyed with the crate — it is the one thing in there that was not
> recoverable from history. Line references below point at the old crate but the module layout
> is identical here, so they still resolve.
>
> Related: the hub's `project_compression_substrate_quality_bar` pin, which sets the two quality
> bars any KV-compression scheme must be measured against separately (retrieval ranking vs.
> autoregressive generation) — read it before implementing this.

---

## Problem

The bitnet crate has 1.58-bit ternary weights (16x smaller than f32), but the
KV cache is **full f32**. During autoregressive generation, the cache becomes
the dominant memory consumer — often larger than the model weights themselves.

Current KV cache (`layers/kv_cache.rs`):
```rust
pub struct KvCache {
    k_cache: Vec<f32>,  // [max_seq_len, n_kv_heads * head_dim] as flat f32
    v_cache: Vec<f32>,  // same
    // ...
}
```

**Example: 22-layer model, 4 KV heads, head_dim=64, max_seq=2048**
- KV cache: 22 layers x 2 buffers x 2048 x 256 x 4 bytes = **92 MB**
- Model weights (ternary): roughly **~10-20 MB** for a 2B parameter model

The cache is 5-10x larger than the weights. This defeats the purpose of running
a 1.58-bit model on edge hardware.

---

## What TurboQuant Does

Two-stage KV cache compression — no training, no calibration, fully online:

### Stage 1: PolarQuant

1. **Random rotation**: multiply K/V vectors by a fixed random orthogonal
   matrix R (head_dim x head_dim). This spreads information uniformly across
   dimensions, eliminating outlier channels that break naive quantization.

2. **Polar coordinate transform**: convert each pair of rotated dimensions
   (x, y) into polar form (r, theta). The radius r concentrates tightly
   (low variance), so only the angle theta needs fine-grained quantization.

3. **Quantize angles to 3-bit** (8 buckets around the unit circle). The radius
   gets a coarse 1-2 bit encoding or is approximated by a per-head constant.

Net effect: 32-bit floats per dimension -> ~3 bits per dimension, with the
distortion approaching the information-theoretic limit.

### Stage 2: QJL (Quantized Johnson-Lindenstrauss)

A 1-bit residual correction that provides **unbiased inner-product estimation**.
After PolarQuant compresses the cache, QJL stores sign bits of a random
projection of the quantization residual. When computing Q*K dot products, the
QJL correction term is added back, keeping attention scores accurate without
full dequantization.

### Combined Result

| Metric           | Value                         |
|------------------|-------------------------------|
| Bits per element | ~3 (down from 32)             |
| Memory reduction | ~10.7x per cache element      |
| Quality          | Matches f32 on LongBench      |
| Overhead         | One rotation matvec per append |
| Requirements     | No training, no calibration   |

---

## Current Architecture (What Exists)

Files involved:

```
crates/bitnet/src/
  layers/
    kv_cache.rs      -- KvCache + ModelKvCache (f32 storage)
    attention.rs     -- MultiHeadAttention::forward_cached() uses KvCache
  ops/
    quantize.rs      -- absmax 8-bit activation quantization (reusable patterns)
```

### KvCache API surface (kv_cache.rs)

```rust
KvCache::new(n_kv_heads, head_dim, max_seq_len) -> Self
KvCache::append(&mut self, keys: &[f32], values: &[f32])
KvCache::key_at(&self, pos: usize, kv_head: usize) -> &[f32]  // returns [head_dim]
KvCache::value_at(&self, pos: usize, kv_head: usize) -> &[f32] // returns [head_dim]
KvCache::keys(&self) -> &[f32]      // flat slice, all cached positions
KvCache::values(&self) -> &[f32]    // flat slice, all cached positions
KvCache::clear(&mut self)
KvCache::memory_bytes(&self) -> usize
```

### How attention.rs uses the cache (forward_cached, ~line 262-359)

1. Project new tokens through Q/K/V BitLinear projections
2. Apply RoPE to Q and new K
3. **`cache.append(&k_new, &v_new)`** — store new K/V as f32
4. For each Q head, for each Q position:
   - Loop over all cached positions s=0..attend_len:
     - **`cache.key_at(s, kv_h)`** — read K vector, compute dot product with Q
   - Softmax over scores
   - Weighted sum: loop over s again:
     - **`cache.value_at(s, kv_h)`** — read V vector, accumulate
5. Output projection

The hot path is steps 4's inner loops — every cached position is read per query
token. This is where compressed storage + fast dot-product estimation pays off.

---

## Proposed Design: QuantizedKvCache

### Storage Layout

```rust
/// 3-bit quantized KV cache using TurboQuant (PolarQuant + QJL).
pub struct QuantizedKvCache {
    // -- PolarQuant compressed storage --
    /// Quantized angle indices for K cache.
    /// Each element is 3 bits (stored packed: 8 angles per 3 bytes, or
    /// simpler: Vec<u8> with values 0..7, optimize packing later).
    /// Shape: [max_seq_len, n_kv_heads, head_dim/2] (pairs of dims)
    k_angles: Vec<u8>,

    /// Quantized angle indices for V cache (same layout).
    v_angles: Vec<u8>,

    /// Per-position, per-head radius scale for K.
    /// Shape: [max_seq_len, n_kv_heads]
    k_radius: Vec<f32>,

    /// Per-position, per-head radius scale for V.
    v_radius: Vec<f32>,

    // -- QJL correction bits --
    /// 1-bit sign of random projection of quantization residual (K).
    /// Packed as bits: ceil(head_dim / 8) bytes per (position, head).
    k_qjl_signs: Vec<u8>,

    /// Same for V.
    v_qjl_signs: Vec<u8>,

    // -- Fixed random matrices (generated from seed) --
    /// Orthogonal rotation matrix for PolarQuant.
    /// Shape: [head_dim, head_dim]. One per cache instance (shared across positions).
    rotation_matrix: Vec<f32>,

    /// Random projection vectors for QJL correction.
    /// Shape: [n_qjl_projections, head_dim].
    qjl_projection: Vec<f32>,

    // -- Dimensions --
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    len: usize,

    // -- Precomputed angle->unit-vector lookup (8 entries for 3-bit) --
    angle_cos: [f32; 8],  // cos(2*pi*i/8) for i=0..7
    angle_sin: [f32; 8],  // sin(2*pi*i/8) for i=0..7
}
```

### Key Operations

#### Append (write path)

Called once per new token during decode. Cost is bounded and small.

```
fn append(&mut self, keys: &[f32], values: &[f32]):
    for each new position, for each kv_head:
        kv_vec = keys[pos, head]           // f32 [head_dim]

        // 1. Rotate: spread information uniformly
        rotated = rotation_matrix @ kv_vec  // matvec: head_dim x head_dim

        // 2. Pair dimensions and convert to polar coordinates
        for i in 0..head_dim/2:
            x = rotated[2*i]
            y = rotated[2*i + 1]
            r = sqrt(x*x + y*y)
            theta = atan2(y, x)            // range [-pi, pi]

            // 3. Quantize angle to 3-bit bucket (0..7)
            bucket = round((theta + pi) / (2*pi) * 8) % 8
            store bucket in k_angles

            // Accumulate radius for per-head scale
            radius_sum += r

        // 4. Store per-head average radius
        k_radius[pos, head] = radius_sum / (head_dim / 2)

        // 5. QJL correction: compute residual, project, store signs
        reconstructed = dequantize_polar(angles, radius)
        residual = rotated - reconstructed
        for each qjl projection vector p:
            sign_bit = (dot(residual, p) >= 0) as u1
            pack into k_qjl_signs
```

#### Dot product (read path — replaces key_at + manual dot)

Instead of dequantizing K back to f32 and then dotting with Q, we can compute
the dot product **directly** in compressed domain. But the simpler v1 approach
is dequantize-on-read:

```
fn key_at_dequant(&self, pos: usize, kv_head: usize) -> Vec<f32>:
    radius = k_radius[pos, kv_head]
    for i in 0..head_dim/2:
        bucket = k_angles[pos, head, i]
        x = radius * angle_cos[bucket]
        y = radius * angle_sin[bucket]
        rotated[2*i] = x
        rotated[2*i+1] = y

    // Inverse rotation to get back to original space
    original = rotation_matrix^T @ rotated

    // QJL correction (optional, improves accuracy)
    correction = qjl_decode(k_qjl_signs[pos, head], qjl_projection)
    original += rotation_matrix^T @ correction

    return original
```

**V1 (simple):** Dequantize on read, drop into existing attention loop unchanged.
**V2 (fast):** Compute Q*K dot product in rotated domain without materializing
the full f32 vector. Since rotation is orthogonal, `dot(Q, K) = dot(R@Q, R@K)`,
so rotate Q once and dot against compressed K directly.

### Integration into attention.rs

#### V1 — Drop-in replacement (minimal changes)

Make `QuantizedKvCache` implement the same API as `KvCache`, but `key_at` /
`value_at` return owned `Vec<f32>` instead of `&[f32]`. This requires changing
the attention loop to not borrow the cache across iterations:

```rust
// attention.rs — forward_cached, inner loop change

// BEFORE (borrows cache):
let k_vec = cache.key_at(s, kv_h);

// AFTER (owned, dequantized on the fly):
let k_vec = cache.key_at_dequant(s, kv_h);
```

Or better, use a trait:

```rust
pub trait KvStore {
    fn append(&mut self, keys: &[f32], values: &[f32]);
    fn dot_key(&self, pos: usize, kv_head: usize, query: &[f32]) -> f32;
    fn weighted_value_sum(&self, kv_head: usize, weights: &[f32], out: &mut [f32]);
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn memory_bytes(&self) -> usize;
}
```

This lets attention code work with either `KvCache` (f32) or
`QuantizedKvCache` (3-bit) through the same interface, and lets the quantized
version compute dot products without full dequantization.

#### V2 — Fused attention kernel

For maximum performance, fuse the dequantize + dot product:

```rust
impl QuantizedKvCache {
    /// Compute dot(query, cached_key[pos, head]) without full dequantization.
    ///
    /// Since R is orthogonal: dot(q, k) = dot(Rq, Rk).
    /// Rk is stored as polar angles + radius. Rq is computed once per query.
    fn dot_key_fast(&self, pos: usize, kv_head: usize, rotated_query: &[f32]) -> f32 {
        let radius = self.k_radius[pos * self.n_kv_heads + kv_head];
        let mut sum = 0.0f32;

        for i in 0..self.head_dim / 2 {
            let bucket = self.k_angles[/* index */] as usize;
            let rq_x = rotated_query[2 * i];
            let rq_y = rotated_query[2 * i + 1];
            // dot contribution = radius * (rq_x * cos(theta) + rq_y * sin(theta))
            sum += rq_x * self.angle_cos[bucket] + rq_y * self.angle_sin[bucket];
        }
        sum *= radius;

        // Add QJL correction term
        sum += self.qjl_correction(pos, kv_head, rotated_query);
        sum
    }
}
```

This avoids materializing the full f32 K vector entirely. The inner loop is
just a table lookup + 2 multiplies + 1 add per dimension pair.

---

## Generating the Rotation Matrix

The rotation matrix R must be orthogonal (preserves dot products). Standard
approach: generate a random matrix from a seeded RNG, then QR-decompose it.

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn generate_rotation_matrix(head_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = head_dim;

    // Generate random Gaussian matrix
    let mut mat = vec![0.0f32; n * n];
    for v in mat.iter_mut() {
        // Box-Muller or use rand_distr::Normal
        *v = sample_normal(&mut rng);
    }

    // QR decomposition (Gram-Schmidt) to get orthogonal Q
    gram_schmidt_inplace(&mut mat, n);
    mat
}
```

This is computed once at cache creation time. For head_dim=64, it's a 64x64
matrix = 16 KB — trivial. The seed should be fixed (e.g., derived from
layer index) so results are reproducible.

**Note:** We could also use a structured random rotation (Hadamard + random
sign flips) which is O(n log n) to apply instead of O(n^2) for the dense
matvec. For head_dim=64 the dense matvec is only 4096 FMAs — probably not
worth the complexity for v1, but worth considering if profiling shows the
rotation is a bottleneck during prefill of long prompts.

---

## QJL Correction Detail

QJL stores the sign of random projections of the quantization error:

```
residual = R @ original - dequantize(quantized)    // quantization error in rotated space
for j in 0..n_projections:
    bit_j = sign(dot(residual, projection[j]))     // 1 bit per projection
```

To correct a dot product `dot(q, k)`:

```
correction = (1 / n_projections) * sum_j(
    sign_bit_j * |dot(residual_estimate, projection[j])| * dot(rotated_q, projection[j])
)
```

The paper shows that even a small number of projection vectors (16-32) gives
meaningful correction. Each adds only 1 bit per position per head, so 32
projections = 4 bytes per (position, head, K-or-V).

**For v1:** Skip QJL entirely. PolarQuant alone at 3-bit gives most of the
benefit. Add QJL as a refinement in v2 if quality isn't sufficient.

---

## Memory Savings

### Per-position, per-head storage

| Component       | f32 baseline | PolarQuant 3-bit | PolarQuant + QJL |
|-----------------|-------------|------------------|------------------|
| K or V vector   | 64 x 4B = 256B | 32 angles x 3bit + 4B radius = 16B | 16B + 4B QJL = 20B |
| Compression     | 1x          | **16x**          | **12.8x**        |

### Full model (22 layers, 4 KV heads, head_dim=64, max_seq=2048)

| Config           | KV cache size | Notes |
|------------------|--------------|-------|
| f32 (current)    | 92 MB        | 22 x 2 x 2048 x 256 x 4 |
| PolarQuant 3-bit | ~5.8 MB      | angles + radius per head |
| PolarQuant + QJL | ~7.2 MB      | + 32 projection sign bits |
| Rotation matrices| ~0.35 MB     | 22 layers x 16KB each (one-time) |

**Total with TurboQuant: ~7.5 MB vs 92 MB current = 12x reduction.**

The KV cache drops from being the dominant allocation to being smaller than
the model weights. This is exactly what's needed for edge/mobile deployment
of ternary models.

---

## Implementation Plan

### Phase 1 — PolarQuant only (no QJL)

1. **`ops/polar.rs`** (new file)
   - `generate_rotation_matrix(head_dim, seed) -> Vec<f32>`
   - `rotate(matrix: &[f32], vec: &[f32], out: &mut [f32])` (matvec)
   - `rotate_transpose(matrix: &[f32], vec: &[f32], out: &mut [f32])` (R^T @ v)
   - `to_polar_quantized(rotated: &[f32]) -> (Vec<u8>, f32)` (angles + radius)
   - `from_polar_quantized(angles: &[u8], radius: f32, head_dim: usize) -> Vec<f32>`
   - Gram-Schmidt orthogonalization (or use Householder for numerical stability)

2. **`layers/quantized_kv_cache.rs`** (new file)
   - `QuantizedKvCache` struct as described above (skip QJL fields for now)
   - Implement same logical API as `KvCache`
   - `append()`: rotate + polar quantize + store
   - `key_at_dequant()` / `value_at_dequant()`: reconstruct f32 on the fly
   - `dot_key_fast()`: fused dot product in rotated domain
   - `ModelQuantizedKvCache` wrapper (Vec of per-layer caches)

3. **`layers/attention.rs` changes**
   - Add `forward_cached_quantized()` method or make `forward_cached()` generic
     over a `KvStore` trait
   - Wire `dot_key_fast()` into the attention score loop
   - Wire `weighted_value_sum()` for the value aggregation

4. **Tests**
   - Roundtrip: quantize -> dequantize, check error is small relative to magnitude
   - Dot product preservation: `|dot(q, k_original) - dot(q, k_dequantized)| < epsilon`
   - Rotation orthogonality: `R^T @ R ≈ I`
   - Full attention output: compare `forward_cached` (f32) vs quantized version,
     measure max/mean error across a batch of random inputs
   - Memory reporting: verify `memory_bytes()` is ~12x smaller

### Phase 2 — QJL correction

5. **`ops/qjl.rs`** (new file)
   - `generate_projections(n_proj, head_dim, seed) -> Vec<f32>`
   - `encode_signs(residual: &[f32], projections: &[f32]) -> Vec<u8>` (packed bits)
   - `correction_dot(signs: &[u8], projections: &[f32], query: &[f32]) -> f32`

6. Integrate into `QuantizedKvCache::append()` and `dot_key_fast()`

7. **Benchmark:** measure perplexity / LongBench quality with and without QJL
   to see if the extra bits are worth it for ternary-sourced KV values

### Phase 3 — Optimizations

8. **Bit-pack angles**: 3 bits x 32 pairs = 96 bits = 12 bytes per (pos, head)
   instead of 32 bytes with u8-per-angle. Saves another ~40% on angle storage.

9. **SIMD dot_key_fast**: the inner loop (lookup cos/sin, 2 FMAs per pair) is
   very SIMD-friendly. AVX2 version: process 4 angle pairs per 256-bit register.

10. **Structured rotation** (Hadamard + random signs): O(n log n) apply instead
    of O(n^2). Matters for prefill where we rotate many tokens at once.

---

## Open Questions

1. **Ternary-sourced KV distributions**: K and V come from BitLinear (ternary
   weights, 8-bit absmax activations). The resulting distributions may have
   lower dynamic range than standard models. This could mean:
   - PolarQuant works even better (less outlier mass to rotate away)
   - Or: values cluster near zero, making polar coordinates less efficient
   - **Action:** Histogram K/V values from a real bitnet model run, check distribution shape

2. **Radius encoding**: The paper uses per-head average radius. For ternary-
   sourced KV, the per-pair radius variance might be low enough that a single
   scalar suffices. If not, consider 4-bit per-pair radius (still much cheaper
   than f32).

3. **Value cache strategy**: TurboQuant's dot-product-in-compressed-domain trick
   works cleanly for K (we compute Q*K dot products). For V, we need weighted
   sums (`sum(weight_s * V_s)`), which is harder to do without dequantization.
   Options:
   - Dequantize V on the fly (still saves memory, just not compute)
   - Keep V at 8-bit instead of 3-bit (simpler, still 4x savings over f32)
   - Batch-dequantize only the top-k positions by attention weight

4. **Dependency situation**: Gram-Schmidt / QR needs no external crate for
   head_dim=64. We can write a simple in-crate version. RNG: `rand` +
   `rand_chacha` are likely already in the dependency tree; if not, a simple
   xoshiro from a seed would work.

5. **Interaction with GQA**: With n_kv_heads < n_heads, the cache is already
   smaller. TurboQuant still helps proportionally — 4 KV heads at 3-bit is
   still 12x smaller than 4 KV heads at f32.

---

## References

- [TurboQuant](https://turboquant.net) — Google Research, 2024
- Current KV cache: `crates/bitnet/src/layers/kv_cache.rs`
- Current attention: `crates/bitnet/src/layers/attention.rs`
- Existing quantization patterns: `crates/bitnet/src/ops/quantize.rs`
