# ternary-rs status

Last refresh: 2026-08-11

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
- [x] AVX2 ternary kernel — `src/compute/avx2.rs`
- [x] Scalar fallback kernel — `src/compute/scalar.rs`
- [x] wgpu GPU backend (initial substrate, ternary matmul shaders) — `src/compute/wgpu_backend.rs`
- [x] Smart backend selection (CPU vs GPU based on detection) — `src/compute/mod.rs`
- [x] `bitnet-chat` conversational TUI — `src/bin/bitnet-chat.rs`
- [x] `gguf-info`, `bitnet-bench`, `bitnet-diag`, `bitnet-i2s-test` — `src/bin/*`
- [x] GPU substrate absorbed from cortex (Stage 1) — `src/compute/gpu_engine.rs`, `src/layers/gpu_bitlinear.rs`, `src/compute/shaders/*.wgsl`. `WgpuBackend` split into a shared `GpuDevice` (Arc'd device + buffer helpers + compiled `Pipelines`) so layers can share one device and hold weights resident. **Substrate only — not on the inference path** (see In flight).
- [x] 274 library tests passing, 1 unused-unsafe warning (cpuid wrapper)
- [x] Block Attention Residuals listed in roadmap (per commit a84a318)

## In flight

- [~] **`GpuBitLinear` is built and tested but not wired into `TransformerModel`.** Its only references are `gpu_engine.rs` (itself uncalled) and doc comments. Inference still runs `BitLinear` against the per-call-upload `WgpuBackend`, which re-uploads the full weight matrix every call. This is why the GPU path is slow — see Perf baseline.

## Next

Priority order, by measured impact:

- [ ] **Fix backend selection — it currently costs ~5×.** `compute::detect()` prefers any *discrete* GPU over AVX2. On this box that picks wgpu (2.6 tok/s) over AVX2 (12.4 tok/s est). The heuristic's premise — "discrete GPU beats AVX2" — is false for the current per-call-upload kernel. Make it measurement-driven, not presence-driven (DELTA_REPORT step 6). Cheapest available win.
- [ ] **Wire `GpuBitLinear` into `TransformerModel`.** The resident-weight path is the reason the substrate was imported; until the model uses it, the import buys nothing at runtime. This is the change that can make the GPU path genuinely win.
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

**Measured 2026-08-11**, after the Stage 1 substrate import, `--release`:

End-to-end (`bitnet-chat`, auto-selected backend = wgpu):

| Phase | Tokens | Time | Rate |
|---|---|---|---|
| Prefill | 8 | 3192 ms | 3 tok/s |
| Decode | 64 | 25820 ms | **2.5 tok/s** |

Per-matvec (`bitnet-bench`, all backends cross-checked against scalar — all VERIFIED):

| Backend | Q proj 2560×2560 | FFN gate 6912×2560 | Full-model est. |
|---|---|---|---|
| scalar | 18548 µs | 59466 µs | 0.1 tok/s |
| avx2 | 211 µs | 614 µs | **12.4 tok/s** |
| wgpu | 1191 µs | 2617 µs | 2.6 tok/s |

**Reading:** the import changed nothing on the hot path, so decode is unmoved from the 2026-05-29
baseline of 2 tok/s — expected, not a regression. The bench's wgpu estimate (2.6) landed within 4%
of measured decode (2.5), which is what makes the AVX2 estimate (12.4 tok/s) credible: the engine is
currently running ~5× slower than its own CPU kernel because `detect()` prefers the discrete GPU.

AVX2 end-to-end is an estimate, not a measurement — there is no CLI flag or env override to force
CPU-only, so it was not directly measured. Worth adding one.

## Architectural invariants

These are load-bearing and shouldn't be relaxed without going through the integration pin process:

- **Ternary only.** f16/Q4_K_M dequantization belongs in cortex, not here. Don't re-add a `LinearLayer` trait, `FloatLinear`, or K-quant dequant module.
- **F32 activations end-to-end.** Never pack activations to f16/bf16. (Per `project_f32_activations_invariant` — learned from cortex's NaN saga during the merger period.)
- **Plain f32 at layer boundaries.** No custom tensor framework lock-in; layers communicate via `&[f32]`.
- **Zero `unsafe`.** SIMD via safe abstractions only.
- **Ternary encoding:** `0b00 = -1`, `0b01 = 0`, `0b10 = +1`, `0b11` unused. GGUF TQ2_0 uses a different convention (0=neg, 1=zero, 2=pos) and is remapped on load.

## Background

Ternary-rs was briefly subsumed into cortex during a "swiss army knife of transformers" phase. On 2026-05-29 that merger was reversed (the un-merge): cross-cutting cortex changes (PolarQuant matmul) kept breaking BitNet, requiring revert cycles. The architectural decision is to keep cortex (Qwen-class GPU) and ternary-rs (BitNet 1.58-bit) as sibling systems and eventually compose via a Q-Former adapter, rather than as variants of one transformer. The grounding pin is `project_training_time_representation` in `C:\src\ringhub-integration\memory\`.
