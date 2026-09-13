# ternary-rs status

Last refresh: 2026-09-12

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
- [x] f16 embedding table kept at source precision; F16C `dot_f16` for the tied output projection — `src/tensor.rs` (`EmbeddingTable`), `src/compute/half.rs`
- [x] Scalar fallback kernel — `src/compute/scalar.rs`
- [x] wgpu GPU backend (initial substrate, ternary matmul shaders) — `src/compute/wgpu_backend.rs`
- [x] Measurement-driven backend selection — races candidates on a probe matvec at startup; `--backend` / `TERNARY_BACKEND` override — `src/compute/mod.rs`
- [x] `bitnet-chat` conversational TUI — `src/bin/bitnet-chat.rs`
- [x] `gguf-info`, `bitnet-bench`, `bitnet-diag`, `bitnet-i2s-test` — `src/bin/*`
- [x] GPU substrate absorbed from cortex (Stage 1) — `src/compute/gpu_engine.rs`, `src/layers/gpu_bitlinear.rs`, `src/compute/shaders/*.wgsl`. `WgpuBackend` split into a shared `GpuDevice` (Arc'd device + buffer helpers + compiled `Pipelines`) so layers can share one device and hold weights resident. **Substrate only — not on the inference path** (see In flight).
- [x] 288 library tests passing, 1 unused-unsafe warning (cpuid wrapper)
- [x] Block Attention Residuals listed in roadmap (per commit a84a318)

## In flight

- [~] **`GpuBitLinear` is built and tested but not wired into `TransformerModel`.** Its only references are `gpu_engine.rs` (itself uncalled) and doc comments. Inference still runs `BitLinear` against the per-call-upload `WgpuBackend`, which re-uploads the full weight matrix every call. This is why the GPU path is slow — see Perf baseline.

## Next

Priority order, by measured impact. **Profile first** — the last two rounds of
guessing both picked the wrong target.

- [ ] **Profile inside attention.** It streams only 123 MB but costs 12.5 ms/tok, 3× less
  byte-efficient than FFN. Roughly 9 ms/tok — ~19% of decode — is something other than weight
  streaming. Suspect per-head `Vec` allocation (20 heads × 30 layers × every token), but that is a
  guess and needs measuring.
- [ ] **Ternary kernel has real headroom.** FFN achieves 29.6 GB/s and attention 9.8 GB/s, against
  65 GB/s demonstrated achievable on this box by the output projection. The blocks are
  compute-bound, not bandwidth-bound — the opposite of what this file claimed before 2026-09-12.
- [ ] **Teach `bitnet-bench` about the output projection.** It models 4×Q + 3×gate per layer and
  stops, so it misses the largest single op in decode entirely. Its full-model estimates were
  misleading for exactly that reason.
- [ ] **Wire `GpuBitLinear` into `TransformerModel`.** Still unwired, and the bar keeps rising: the
  GPU path must now beat 27.2 tok/s, not 2.5. Per-call overhead (~380 µs fixed × 210 matvecs/token)
  means resident weights alone won't do it — it needs the whole decode step resident, one command
  buffer per token, only logits read back.
- [ ] Port `gpu_kv_cache.rs` from cortex (DELTA_REPORT step 3) — per-layer resident K/V buffers.
- [ ] AVX-512 and ARM NEON ternary kernels
- [ ] Heuristic stop conditions for base models (chat currently relies on role-marker scan, no rambling detection)
- [ ] Block Attention Residuals (MoonshotAI/Attention-Residuals) — learned depth-attention at block boundaries
- [ ] Stage 2: perf threshold framework + phantom-work audit + benchmark suite (modeled on cortex's `project_cortex_v1_perf_threshold`)

## Discussed only

- [?] Stage 3+: composable with cortex via Q-Former adapter (per `project_unified_memory_architecture` pin in ringhub-integration) — not specced, not active
- [?] Stage 4+: FPGA deployment via Zynqberry, eventually cascaded FPGA pipeline (per `project_zynqberry_bitnet_memex` pin) — not specced, not active
- [?] Engine trait matching agentos's SharedEngine interface — pinned in CLAUDE.md history but not currently scoped; agentos-claude will drive the API when it's needed

## Perf baseline

BitNet b1.58 2B (Microsoft), 30 layers, 2560 embed, 128256 vocab, 4096 ctx.
Hardware: Intel i9-14900HX + RTX 4080 Laptop (discrete).

**Measured 2026-09-12**, `--release`, 64-token decode. Force a backend with
`--backend <scalar|avx2|wgpu>` or `TERNARY_BACKEND`.

| Configuration | Decode | Step |
|---|---|---|
| wgpu — presence-driven selection | 2.5 tok/s | baseline |
| avx2 — measurement-driven selection | 9.2 tok/s | 3.7× |
| avx2 — kernel threaded over rows | 21.5 tok/s | 2.3× |
| avx2 — tied embedding kept f16 *(current)* | **27.2 tok/s** | 1.27× |

**10.9× total.** Peak working set dropped 2017 → 1392 MB with the f16 embedding.

### Where decode time goes

Instrumented profile at the 21.5 tok/s state (46.5 ms/token):

| Phase | ms/tok | Share | Weight bytes | Achieved |
|---|---|---|---|---|
| output_proj (tied embedding) | 20.22 | 43.5% | 1.313 GB | 65 GB/s |
| ffn | 13.47 | 29.0% | 398 MB | 29.6 GB/s |
| attention | 12.51 | 26.9% | 123 MB | 9.8 GB/s |
| norms + residuals + embedding lookup | 0.26 | 0.6% | — | — |

RMSNorm, RoPE, softmax and residuals are **0.6% combined** — optimizing them buys nothing.
The f16 change halves the output projection's bytes, which is where the 27.2 tok/s came from.

### Correction (2026-09-12)

The bandwidth analysis recorded here on 2026-08-15 was wrong, and the profile is what exposed it:

- It claimed ~603 MB/token. Real figure was **1.83 GB/token** — it omitted the output projection
  entirely, then mis-sized it as an 82 MB ternary matrix when it was 1.31 GB of f32.
- It claimed ~20 GB/s was about all this box sustains. The output projection demonstrably reaches
  **65 GB/s**.
- It concluded "decode is memory-bound; gains come from moving fewer bytes, not more threads."
  That held for the output projection only. The ternary blocks run **2–6× below** the achievable
  bandwidth and are still compute-bound.

Per-matvec figures from `bitnet-bench` remain useful for comparing backends, but treat its
full-model estimate as an upper bound on the *kernel*, not a prediction of decode: it is cache-warm
(Q proj is 1.6 MB, fits L2), it bills K/V at full Q size when GQA makes them ¼
(`head_count=20`, `head_count_kv=5`), and it does not model the output projection at all.

## Architectural invariants

These are load-bearing and shouldn't be relaxed without going through the integration pin process:

- **Ternary only.** f16/Q4_K_M dequantization belongs in cortex, not here. Don't re-add a `LinearLayer` trait, `FloatLinear`, or K-quant dequant module.
- **F32 activations end-to-end.** Never pack activations to f16/bf16. (Per `project_f32_activations_invariant` — learned from cortex's NaN saga during the merger period.)
  This constrains *activations*, not weight storage. Keeping the embedding table in its source f16 (see `EmbeddingTable`) is not a breach: the weights were already f16 on disk, conversion to f32 is exact, and every activation on the path stays f32. Downcasting an f32 source to f16 *would* be a breach — `load_embedding` never does it.
- **Plain f32 at layer boundaries.** No custom tensor framework lock-in; layers communicate via `&[f32]`.
- **Zero `unsafe` in hot paths.** The AVX2 kernel uses `#[target_feature]` + intrinsics, which are unsafe by construction; those are the only `unsafe` blocks and each is guarded by runtime feature detection. Matches the wording in CLAUDE.md. (The stricter "zero `unsafe`, SIMD via safe abstractions only" claim this replaces was never true of `avx2.rs`.)
- **Ternary encoding:** `0b00 = -1`, `0b01 = 0`, `0b10 = +1`, `0b11` unused. GGUF TQ2_0 uses a different convention (0=neg, 1=zero, 2=pos) and is remapped on load.

## Background

Ternary-rs was briefly subsumed into cortex during a "swiss army knife of transformers" phase. On 2026-05-29 that merger was reversed (the un-merge): cross-cutting cortex changes (PolarQuant matmul) kept breaking BitNet, requiring revert cycles. The architectural decision is to keep cortex (Qwen-class GPU) and ternary-rs (BitNet 1.58-bit) as sibling systems and eventually compose via a Q-Former adapter, rather than as variants of one transformer. The grounding pin is `project_training_time_representation` in `C:\src\ringhub-integration\memory\`.
