# ternary-rs

Pure-Rust 1.58-bit inference engine for ternary LLMs. Zero multiplication — the matmul hot path reduces to conditional addition: add, subtract, or skip.

Built from scratch. No PyTorch, no ONNX, no C++ bindings. Just `std` + `thiserror` + `rayon`.

## What is ternary inference?

Ternary models like [BitNet b1.58](https://arxiv.org/abs/2402.17764) quantize weights to {-1, 0, +1}. This eliminates all floating-point multiplication in matrix-vector products — the dominant cost in transformer inference. Instead of `weight * activation`, you get:

- **+1**: add the activation
- **-1**: subtract the activation
- **0**: skip entirely

The result: 16x weight compression (2 bits vs 32), dramatically reduced memory bandwidth, and arithmetic that maps naturally to integer ALUs.

## Features

- **GGUF model loading** — loads BitNet models in GGUF format (TQ1_0, TQ2_0, F16, BF16, F32)
- **Two matmul kernels**: I2S (conditional add/sub/skip) and LUT (TL1 lookup table)
- **Full transformer stack** — embedding, RoPE, GQA attention, SwiGLU FFN, RMSNorm
- **Tokenizer** — BPE tokenizer from GGUF metadata (no sentencepiece dependency)
- **Sampler** — temperature, top-k, top-p, repetition penalty
- **GPU detection** — wgpu-based hardware enumeration (compute shaders coming)
- **259 tests**, zero clippy warnings, zero `unsafe`

## Quick start

### Get a model

Download the BitNet b1.58-2B model in GGUF format:

```bash
# From HuggingFace (requires git-lfs)
git clone https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf
# The model file is BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf (~500MB)
```

Or download directly from: https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf

### Chat with the model

```bash
cargo run --release --bin bitnet-chat -- path/to/ggml-model-i2_s.gguf
```

### Inspect a GGUF file

```bash
cargo run --release --bin gguf-info -- path/to/model.gguf
```

### Run benchmarks

```bash
cargo run --release --bin bitnet-bench -- path/to/model.gguf
```

## Architecture

```
Token IDs
    │
    ▼
┌──────────┐
│ Embedding │  (float lookup table)
└──────────┘
    │
    ▼
┌──────────────────────────┐
│ TransformerBlock × N      │
│  ┌─────────────────────┐ │
│  │ RMSNorm → Attention │ │  ← Ternary Q/K/V/O via BitLinear
│  │   (GQA + RoPE)      │ │     + causal mask + softmax
│  │ + residual           │ │
│  ├─────────────────────┤ │
│  │ RMSNorm → SwiGLU    │ │  ← Ternary gate/up/down via BitLinear
│  │ + residual           │ │
│  └─────────────────────┘ │
└──────────────────────────┘
    │
    ▼
┌──────────┐
│ RMSNorm   │
│ → Logits  │  (output projection)
└──────────┘
```

### Kernel strategies

- **I2S**: 2-bit packed weights, scalar add/sub/skip loop. Byte-level unpacking with sub-byte row offset support.
- **LUT (TL1)**: Pairs 2 weights → 4-bit index into 9-entry precomputed table. Zero arithmetic in the hot loop.

Both kernels produce identical results (tested exhaustively).

### GGUF support

| Type | Description |
|------|-------------|
| TQ2_0 | 2-bit ternary, 256-element blocks |
| TQ1_0 | Base-3 packed ternary, 256-element blocks |
| F32 | 32-bit float |
| F16 | IEEE 754 half-precision |
| BF16 | Brain floating point |

## Using as a library

```toml
[dependencies]
ternary-rs = "0.1"
```

```rust
use ternary_rs::{GgufFile, ModelConfig};
use ternary_rs::loader::load_model;

// Load model
let gguf = GgufFile::open("model.gguf")?;
let config = gguf.model_config()?;
let loaded = load_model(&gguf, &config)?;

// Run inference
let tokens = vec![1u32, 15043, 29871]; // <s> Hello
let logits = loaded.model.forward(&tokens, 0);
```

## Binaries

| Binary | Description |
|--------|-------------|
| `bitnet-chat` | Interactive chat with a BitNet model |
| `bitnet-bench` | Benchmark matmul kernels on real model weights |
| `bitnet-diag` | Diagnostic tool — weight distributions, attention probes |
| `gguf-info` | Inspect GGUF file metadata and tensor info |

## Roadmap

- [x] GGUF model loader (TQ1_0, TQ2_0, F16, BF16, F32)
- [x] Full transformer stack (RoPE, GQA, SwiGLU, RMSNorm)
- [x] BPE tokenizer from GGUF metadata
- [x] Sampler (temperature, top-k, top-p, repetition penalty)
- [x] Hardware detection + boot banner
- [ ] KV cache for autoregressive generation
- [ ] SIMD kernels (x86 AVX2/512, ARM NEON)
- [ ] wgpu compute shaders for GPU inference
- [ ] Float matmul path (standard GGUF models alongside ternary)
- [ ] Heuristic stop conditions for base models
- [ ] Block Attention Residuals ([MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)) — learned depth-attention at block boundaries, drop-in quality boost for small models

## License

MIT
