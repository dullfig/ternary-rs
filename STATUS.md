# ternary-rs status

Last refresh: 2026-08-15

## Built

- [x] Core ternary kernels — `src/ops/{matmul,lut,quantize}.rs` (I2S add/sub/skip, TL1 lookup-table, absmax 8-bit activation quant)
- [x] Tensor types (TernaryTensor, ActivationTensor, FloatTensor) — `src/tensor.rs`
- [x] Full layer stack — `src/layers/{bitlinear,rmsnorm,rope,attention,swiglu,transformer,model}.rs` (BitLinear, RMSNorm, RoPE, GQA attention, SwiGLU FFN, TransformerBlock, TransformerModel with tied/untied embedding)
- [x] KV cache for autoregressive generation — `src/layers/kv_cache.rs`
- [x] Sampler (temp, top-k, top-p, repetition penalty) — `src/layers/sampler.rs`
- [x] BPE tokenizer from GGUF metadata — `src/tokenizer.rs`
- [x] GGUF loader (TQ1_0, TQ2_0, I2S, F32, F16, BF16 for norms+embed) — `src/gguf.rs`
- [x] `load_model()` helper wiring GGUF → TransformerModel — `src/loader.rs`
- [x] Hardware detection + boot banner — `src/compute/device.rs`
- [x] AVX2 ternary kernel, multi-threaded over rows via rayon — `src/compute/avx2.rs`
- [x] Scalar fallback kernel — `src/compute/scalar.rs`
- [x] wgpu GPU backend (initial substrate, ternary matmul shaders) — `src/compute/wgpu_backend.rs`
- [x] Measurement-driven backend selection — races candidates on a probe matvec at startup; `--backend` / `TERNARY_BACKEND` override — `src/compute/mod.rs`
- [x] `bitnet-chat` conversational TUI — `src/bin/bitnet-chat.rs`
- [x] `gguf-info`, `bitnet-bench`, `bitnet-diag`, `bitnet-i2s-test` — `src/bin/*`
- [x] GPU substrate absorbed from cortex (Stage 1) — `src/compute/gpu_engine.rs`, `src/layers/gpu_bitlinear.rs`, `src/compute/shaders/*.wgsl`. `WgpuBackend` split into a shared `GpuDevice` (Arc'd device + buffer helpers + compiled `Pipelines`) so layers can share one device and hold weights resident. **Substrate only — not on the inference path** (see In flight).
- [x] 280 library tests passing, 1 unused-unsafe warning (cpuid wrapper)
- [x] Block Attention Residuals listed in roadmap (per commit a84a318)

## In flight

- [~] **`GpuBitLinear` is built and tested but not wired into `TransformerModel`.** Its only references are `gpu_engine.rs` (itself uncalled) and doc comments. Inference still runs `BitLinear` against the per-call-upload `WgpuBackend`, which re-uploads the full weight matrix every call. This is why the GPU path is slow — see Perf baseline.

## Next

Priority order, by measured impact:

- [ ] **Reduce bytes moved per token.** Decode is now memory-bound (see Perf baseline), so this is where the remaining headroom is, not in more threads. Options not yet explored: fusing the three FFN projections to avoid re-streaming activations, keeping the output head's 82 MB out of the per-token path, blocking the weight walk for cache reuse.
- [ ] **Close the gap between kernel and end-to-end.** The matvec microbenchmark projects 37.2 tok/s; real decode is 20.8. The difference is non-matmul work — attention, RMSNorm, RoPE, softmax, sampling — none of which is SIMD or threaded. Profile before optimizing.
- [ ] **Wire `GpuBitLinear` into `TransformerModel`.** Still unwired. Note the bar moved: the GPU path must now beat 20.8 tok/s, not 2.5. Per-call overhead (~380 µs fixed × 210 matvecs/token) means resident weights alone won't do it — it needs the whole decode step resident, with one command buffer per token and only logits read back.
- [ ] Port `gpu_kv_cache.rs` from cortex (DELTA_REPORT step 3) — per-layer resident K/V buffers, no per-token upload/readback.
- [ ] AVX-512 and ARM NEON ternary kernels
- [ ] Heuristic stop conditions for base models (chat currently relies on role-marker scan, no rambling detection)
- [ ] Block Attention Residuals (MoonshotAI/Attention-Residuals) — learned depth-attention at block boundaries
- [ ] Stage 2: perf threshold framework + phantom-work audit + benchmark suite (modeled on cortex's `project_cortex_v1_perf_threshold`)

## Discussed only

- [?] Stage 3+: composable with cortex via Q-Former adapter (per `project_unified_memory_architecture` pin in ringhub-integration) — not specced, not active
- [?] Stage 4+: FPGA deployment via Zynqberry, eventually cascaded FPGA pipeline (per `project_zynqberry_bitnet_memex` pin) — not specced, not active
- [?] Engine trait matching agentos's SharedEngine interface — pinned in CLAUDE.md history but not currently scoped; agentos-claude will drive the API when it's needed

## Perf baseline

BitNet b1.58 2B (Microsoft), 30 layers, 2560 embed, 128256 vocab, 1.58 GB ternary weights, 4096 ctx.
Hardware: Intel i9-14900HX + RTX 4080 Laptop (discrete). Model load: 5.3 s.

**Measured 2026-08-15**, `--release`, 64-token decode. All figures measured, none projected —
force a backend with `--backend <scalar|avx2|wgpu>` or `TERNARY_BACKEND`.

End-to-end decode (`bitnet-chat`):

| Backend | Decode | vs. previous |
|---|---|---|
| wgpu — old presence-driven selection | 2.5 tok/s | baseline |
| avx2 — measurement-driven selection, serial kernel | 9.2 tok/s | 3.7× |
| avx2 — parallel kernel *(current auto-selection)* | **20.8 tok/s** | 2.3× |

**8.3× total.** Two changes: `detect()` now races candidates instead of assuming a discrete GPU
wins, and the AVX2 matvec parallelizes over rows instead of running on one core of 32.

Per-matvec (`bitnet-bench`, every backend cross-checked against scalar — all VERIFIED):

| Backend | Q proj 2560×2560 | FFN gate 6912×2560 | Full-model est. |
|---|---|---|---|
| scalar | 20004 µs | 64701 µs | 0.1 tok/s |
| avx2 | 57 µs | 223 µs | 37.2 tok/s |
| wgpu | 1381 µs | 5455 µs | 1.5 tok/s |

**Reading — decode is now memory-bound, not compute-bound.** The model streams ~603 MB of packed
weights per token (521 MB across 30 layers, GQA-corrected, plus 82 MB for the output head). At
20.8 tok/s that is ~12.5 GB/s; the gate matvec in isolation hits ~20 GB/s, which is about what this
laptop's DDR5 sustains. That is why threading returned 2.3× and not core-count, and it means
further gains come from moving fewer bytes, not from more threads.

Two caveats worth keeping in view:

- The bench is **cache-warm** (Q proj is 1.6 MB, fits L2), so its full-model estimate flatters CPU:
  it projected 12.4 tok/s for the serial kernel against 9.2 measured, and projects 37.2 against
  20.8 now. Treat it as an upper bound on the kernel, not a prediction of decode.
- The bench bills K and V at full Q size, but GQA makes them ¼ (`head_count=20`,
  `head_count_kv=5`), overstating per-layer work by ~14%.

## Architectural invariants

These are load-bearing and shouldn't be relaxed without going through the integration pin process:

- **Ternary only.** f16/Q4_K_M dequantization belongs in cortex, not here. Don't re-add a `LinearLayer` trait, `FloatLinear`, or K-quant dequant module.
- **F32 activations end-to-end.** Never pack activations to f16/bf16. (Per `project_f32_activations_invariant` — learned from cortex's NaN saga during the merger period.)
- **Plain f32 at layer boundaries.** No custom tensor framework lock-in; layers communicate via `&[f32]`.
- **Zero `unsafe` in hot paths.** The AVX2 kernel uses `#[target_feature]` + intrinsics, which are unsafe by construction; those are the only `unsafe` blocks and each is guarded by runtime feature detection. Matches the wording in CLAUDE.md. (The stricter "zero `unsafe`, SIMD via safe abstractions only" claim this replaces was never true of `avx2.rs`.)
- **Ternary encoding:** `0b00 = -1`, `0b01 = 0`, `0b10 = +1`, `0b11` unused. GGUF TQ2_0 uses a different convention (0=neg, 1=zero, 2=pos) and is remapped on load.

## Background

Ternary-rs was briefly subsumed into cortex during a "swiss army knife of transformers" phase. On 2026-05-29 that merger was reversed (the un-merge): cross-cutting cortex changes (PolarQuant matmul) kept breaking BitNet, requiring revert cycles. The architectural decision is to keep cortex (Qwen-class GPU) and ternary-rs (BitNet 1.58-bit) as sibling systems and eventually compose via a Q-Former adapter, rather than as variants of one transformer. The grounding pin is `project_training_time_representation` in `C:\src\ringhub-integration\memory\`.
