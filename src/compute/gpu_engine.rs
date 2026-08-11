//! GPU forward-pass engine — sub-allocated scratch buffers and pipeline
//! dispatch helpers for the ternary path.
//!
//! ## What lives here
//!
//! - [`TernaryScratch`] — scratch buffers consumed by the batched ternary
//!   matmul path: i8 packed activations + per-token f32 scales. Allocated
//!   once per session (or per shape change), reused across every forward.
//!   This is the sub-allocator pattern cortex paid for the hard way — see
//!   the project memory pin on the `vkFreeMemory` cliff.
//!
//! - [`GpuEngine`] — thin orchestrator holding `Arc<GpuDevice>`. Provides
//!   the `dispatch_*` helpers that build command-buffer compute passes for
//!   the ternary matmul + activation quant pipelines.
//!
//! Eventually (substage 1.5) the full forward orchestrator lives here too:
//! `GpuEngine::forward_block(...)` will run a whole transformer block
//! (attention + FFN) in a single command buffer.
//!
//! ## Sub-allocator discipline
//!
//! Per project memory and cortex's empirical data: NVIDIA Vulkan's
//! `vkFreeMemory` does internal allocator bookkeeping that scales with
//! total allocator state — cortex hit a ~1.4s/buffer cliff after their
//! allocator state grew large, costing ~17s/request from a 12-buffer
//! scratch drop. Mitigation here: allocate scratch buffers **once** at
//! session start via `TernaryScratch::allocate()`, then reuse them across
//! every forward. Never drop and re-create per-request.

use std::sync::Arc;

use super::wgpu_backend::GpuDevice;
use crate::layers::gpu_bitlinear::GpuBitLinear;

// ---------------------------------------------------------------------------
// TernaryScratch — reusable per-block scratch buffers
// ---------------------------------------------------------------------------

/// Scratch buffers consumed by the batched ternary matmul path.
///
/// `activations_i8` holds 4 i8 values per u32 (matches the layout
/// `ternary_matmul_batch.wgsl` expects). `scales` holds one f32 per token.
/// Both are sized so any of the six linear-projection call sites in a
/// transformer block fits (Q, K, V, O attention projections + gate, up, down
/// FFN projections).
pub struct TernaryScratch {
    pub activations_i8: wgpu::Buffer,
    pub scales: wgpu::Buffer,
}

impl TernaryScratch {
    /// Allocate scratch buffers sized for a single forward of `n_tokens`.
    ///
    /// Sizing: the largest input dim across the six linear call sites in a
    /// transformer block is `max(embed_dim, intermediate)`. The `down_proj`
    /// input is `intermediate`; everything else is `embed_dim`. One pair of
    /// buffers fits all sites.
    pub fn allocate(
        gpu: &GpuDevice,
        n_tokens: usize,
        embed_dim: usize,
        intermediate: usize,
    ) -> Self {
        let max_in = embed_dim.max(intermediate);
        let act_q_u32_count = (n_tokens * ((max_in + 3) / 4)) as u64;
        let u32_bytes = std::mem::size_of::<u32>() as u64;
        let f32_bytes = std::mem::size_of::<f32>() as u64;

        Self {
            activations_i8: gpu.create_empty_buffer(
                act_q_u32_count * u32_bytes,
                "scratch.ternary.activations_i8",
            ),
            scales: gpu.create_empty_buffer(
                (n_tokens as u64) * f32_bytes,
                "scratch.ternary.scales",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuEngine — orchestrator + dispatch helpers
// ---------------------------------------------------------------------------

/// Thin orchestrator holding the shared `GpuDevice`. Provides the dispatch
/// helpers that build compute passes for the ternary pipeline set.
pub struct GpuEngine {
    pub gpu: Arc<GpuDevice>,
}

impl std::fmt::Debug for GpuEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuEngine")
    }
}

impl GpuEngine {
    /// Wrap a shared GPU device.
    pub fn new(gpu: Arc<GpuDevice>) -> Self {
        Self { gpu }
    }

    // ----------------------------------------------------------------------
    // dispatch_quantize_absmax_batch — f32 hidden → i8 packed + per-token scales
    // ----------------------------------------------------------------------

    /// Per-token absmax quantization of f32 hidden → packed i8 activations
    /// + per-token scales. Standalone pass variant: creates the compute pass
    /// and dispatches. Use the `_in_pass` variant when chaining dispatches
    /// inside an existing pass.
    ///
    /// `act_q_buf` must be sized at least `n_tokens * ceil(cols / 4)` u32s.
    /// `act_scales_buf` must be sized at least `n_tokens` f32s.
    pub fn dispatch_quantize_absmax_batch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_f32_buf: &wgpu::Buffer,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        cols: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.quantize_absmax_batch.pass"),
            timestamp_writes: None,
        });
        self.dispatch_quantize_absmax_batch_in_pass(
            &mut pass, input_f32_buf, act_q_buf, act_scales_buf, cols, n_tokens,
        );
    }

    /// In-pass variant of [`dispatch_quantize_absmax_batch_into`].
    pub fn dispatch_quantize_absmax_batch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        input_f32_buf: &wgpu::Buffer,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        cols: usize,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct QuantParams { cols: u32, n_tokens: u32 }
        let params = QuantParams { cols: cols as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.quantize_absmax_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[input_f32_buf, act_q_buf, act_scales_buf, &params_buf],
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    // ----------------------------------------------------------------------
    // dispatch_ternary_matmul_batch — i8 activations + scales → f32 output
    // ----------------------------------------------------------------------

    /// Batched shared-memory-tiled ternary matmul. Reads resident 2-bit packed
    /// weights from `GpuBitLinear`, i8 packed activations + per-token scales
    /// (output of [`dispatch_quantize_absmax_batch_into`]), and writes the f32
    /// output with per-token activation scale and the layer's weight_scale
    /// applied inline.
    pub fn dispatch_ternary_matmul_batch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &GpuBitLinear,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        out_f32_buf: &wgpu::Buffer,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.ternary_matmul_batch.pass"),
            timestamp_writes: None,
        });
        self.dispatch_ternary_matmul_batch_in_pass(
            &mut pass, layer, act_q_buf, act_scales_buf, out_f32_buf, n_tokens,
        );
    }

    /// In-pass variant of [`dispatch_ternary_matmul_batch_into`].
    pub fn dispatch_ternary_matmul_batch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &GpuBitLinear,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        out_f32_buf: &wgpu::Buffer,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TmmParams {
            rows: u32,
            cols: u32,
            n_tokens: u32,
            weight_scale_bits: u32,
        }
        let params = TmmParams {
            rows: layer.out_features() as u32,
            cols: layer.in_features() as u32,
            n_tokens: n_tokens as u32,
            weight_scale_bits: layer.weight_scale().to_bits(),
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.ternary_matmul_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[layer.weight_buffer(), act_q_buf, act_scales_buf, out_f32_buf, &params_buf],
        );

        // Phase 5 tiled dispatch: 32×16 output tile, 2 outputs per thread
        // stride-16 in the M dimension. Matches the float matmul_shared layout.
        let rows = layer.out_features();
        const TILE_M: usize = 32;
        const TILE_N: usize = 16;
        let dx = ((rows + TILE_M - 1) / TILE_M) as u32;
        let dy = ((n_tokens + TILE_N - 1) / TILE_N) as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }
}
