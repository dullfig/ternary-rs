//! wgpu compute backend — shared GPU device, compiled pipelines, and the
//! drop-in `ComputeBackend` impl for ternary matvec.
//!
//! ## Two-level architecture
//!
//! 1. **`GpuDevice`** — the shared GPU context: `wgpu::Device`, `wgpu::Queue`,
//!    and all compiled `Pipelines`. Created once at startup, shared via
//!    `Arc<GpuDevice>` across every layer that touches the GPU. This is the
//!    central handle that `GpuBitLinear` and the `gpu_engine` dispatch
//!    functions hold. (A resident `GpuKvCache` is planned but not yet
//!    ported — see DELTA_REPORT step 3.)
//!
//! 2. **`WgpuBackend`** — wraps an `Arc<GpuDevice>` and implements
//!    `ComputeBackend` for the per-call ternary matvec drop-in path. This is
//!    the *slow* path: it uploads the weight matrix on every call. Use this
//!    only for testing or fall-back scenarios. For production, use
//!    `GpuBitLinear` which keeps weights resident.
//!
//! ## Pipelines
//!
//! Four ternary-relevant shaders, loaded from `src/compute/shaders/`:
//! - `ternary_matvec` — single-token decode path (used by both the
//!   `ComputeBackend` impl above and by `GpuBitLinear` for decode).
//! - `ternary_matmul_batch` — shared-memory tiled batched matmul for prefill.
//! - `quantize_absmax_batch` — per-token i8 absmax activation quantization.
//! - `relu2_mul_batch` — BitNet b1.58 SwiGLU activation (ReLU²(gate) * up).
//!
//! Bind group layouts are inferred from the shaders (`layout: None`).

use std::sync::Arc;

use crate::tensor::TernaryTensor;
use super::ComputeBackend;

// ---------------------------------------------------------------------------
// Pipelines — compiled compute pipelines for the ternary path
// ---------------------------------------------------------------------------

/// The set of compiled compute pipelines used by the ternary GPU path.
///
/// Compiled once at `GpuDevice::try_new()` time. Each `wgpu::ComputePipeline`
/// is cheap to clone-reference and bind in a compute pass.
pub struct Pipelines {
    /// Single-token ternary matvec. Entry point `ternary_matvec`.
    /// Bindings (4): weights ro, activations ro, output rw, params uniform.
    pub ternary_matvec: wgpu::ComputePipeline,

    /// Batched, shared-memory tiled ternary matmul. Entry point `main`.
    /// Bindings (5): weights ro, activations ro, act_scales ro, output rw, params uniform.
    pub ternary_matmul_batch: wgpu::ComputePipeline,

    /// Per-token absmax i8 activation quantization. Entry point `main`.
    /// Bindings (4): input ro, output_q rw, output_scales rw, params uniform.
    pub quantize_absmax_batch: wgpu::ComputePipeline,

    /// BitNet ReLU²(gate) * up activation. Entry point `main`.
    /// Bindings (4): gate ro, up ro, output rw, params uniform.
    pub relu2_mul_batch: wgpu::ComputePipeline,
}

impl Pipelines {
    /// Compile all pipelines from the WGSL shader files.
    pub fn compile(device: &wgpu::Device) -> Self {
        let make = |src: &str, label: &str, entry: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Inferred from shader reflection.
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            ternary_matvec: make(
                include_str!("shaders/ternary_matvec.wgsl"),
                "ternary_matvec",
                "ternary_matvec",
            ),
            ternary_matmul_batch: make(
                include_str!("shaders/ternary_matmul_batch.wgsl"),
                "ternary_matmul_batch",
                "main",
            ),
            quantize_absmax_batch: make(
                include_str!("shaders/quantize_absmax_batch.wgsl"),
                "quantize_absmax_batch",
                "main",
            ),
            relu2_mul_batch: make(
                include_str!("shaders/relu2_mul_batch.wgsl"),
                "relu2_mul_batch",
                "main",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDevice — shared device, queue, and pipelines
// ---------------------------------------------------------------------------

/// Shared GPU context: device, queue, and compiled pipelines.
///
/// Created once at startup, shared across all GPU-resident layers via
/// `Arc<GpuDevice>`. Holds the compiled pipelines so they don't have to be
/// re-bound per call.
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: Pipelines,
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuDevice")
    }
}

impl GpuDevice {
    /// Try to create a GPU device. Returns `None` if no suitable adapter
    /// is found or device creation fails.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let info = adapter.get_info();
        tracing::info!(
            name = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            "wgpu adapter selected"
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ternary-rs"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let pipelines = Pipelines::compile(&device);

        Some(Self { device, queue, pipelines })
    }

    /// Create a bind group from a pipeline and a list of buffers.
    ///
    /// Buffers are bound to `@binding(0)`, `@binding(1)`, etc., in order.
    pub fn make_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    }

    /// Create a uniform buffer from a bytemuck-able params struct.
    ///
    /// Uses `queue.write_buffer` (not `create_buffer_init`) because the
    /// latter's internal staging belt did not recycle reliably across
    /// hundreds of per-dispatch params buffers in cortex — they hit a
    /// "staging buffer in bind group" validation error around the 200th
    /// call. `queue.write_buffer` manages its own staging at the queue
    /// level and is the wgpu-recommended pattern for frequent small writes.
    pub fn create_params_buffer<T: bytemuck::Pod>(&self, params: &T) -> wgpu::Buffer {
        let size = std::mem::size_of::<T>() as u64;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(params));
        buf
    }

    /// Create a storage buffer with initial data.
    pub fn create_storage_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    /// Create an empty storage buffer of a given size.
    pub fn create_empty_buffer(&self, size: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for GPU→CPU readback.
    pub fn create_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

// ---------------------------------------------------------------------------
// WgpuBackend — drop-in ComputeBackend impl
// ---------------------------------------------------------------------------

/// Drop-in GPU compute backend wrapping a shared `GpuDevice`.
///
/// Implements `ComputeBackend::ternary_matvec` by uploading the weight
/// matrix, activation vector, and reading back the i32 output **on every
/// call**. This is the slow path — use `GpuBitLinear` for resident weights
/// in production.
pub struct WgpuBackend {
    gpu: Arc<GpuDevice>,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuBackend")
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecParams {
    rows: u32,
    cols: u32,
}

impl WgpuBackend {
    /// Try to create a wgpu backend. Returns `None` if GPU initialization fails.
    pub fn try_new() -> Option<Self> {
        let gpu = GpuDevice::try_new()?;
        Some(Self { gpu: Arc::new(gpu) })
    }

    /// Construct from an already-built `Arc<GpuDevice>` (sharing the device
    /// with `GpuBitLinear` / other resident-weight layers).
    pub fn from_device(gpu: Arc<GpuDevice>) -> Self {
        Self { gpu }
    }

    /// Access the shared `GpuDevice` for other layers to clone-reference.
    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.gpu
    }

    /// Pad a byte slice to 4-byte alignment (wgpu requirement).
    fn pad_to_u32(data: &[u8]) -> Vec<u8> {
        let remainder = data.len() % 4;
        if remainder == 0 {
            data.to_vec()
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len() + (4 - remainder), 0);
            padded
        }
    }

    /// Pack i8 activations into a u32-aligned byte buffer.
    fn pack_activations(input: &[i8]) -> Vec<u8> {
        let mut bytes: Vec<u8> = input.iter().map(|&v| v as u8).collect();
        let remainder = bytes.len() % 4;
        if remainder != 0 {
            bytes.resize(bytes.len() + (4 - remainder), 0);
        }
        bytes
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &str { "wgpu" }

    fn ternary_matvec(&self, weights: &TernaryTensor, input: &[i8]) -> Vec<i32> {
        assert_eq!(weights.cols(), input.len(), "dimension mismatch");

        let rows = weights.rows();
        let cols = weights.cols();

        if rows == 0 || cols == 0 {
            return vec![0i32; rows];
        }

        let weight_bytes = Self::pad_to_u32(weights.packed_data());
        let act_bytes = Self::pack_activations(input);
        let output_size = (rows * std::mem::size_of::<i32>()) as u64;

        let weight_buf = self.gpu.create_storage_buffer(&weight_bytes, "weights");
        let act_buf = self.gpu.create_storage_buffer(&act_bytes, "activations");
        let output_buf = self.gpu.create_empty_buffer(output_size, "output");
        let staging_buf = self.gpu.create_staging_buffer(output_size);
        let params_buf = self.gpu.create_params_buffer(&MatvecParams {
            rows: rows as u32,
            cols: cols as u32,
        });

        let pipeline = &self.gpu.pipelines.ternary_matvec;
        let bind_group = self.gpu.make_bind_group(
            pipeline,
            &[&weight_buf, &act_buf, &output_buf, &params_buf],
        );

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ternary_matvec_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ternary_matvec_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("GPU readback failed").expect("buffer map failed");

        let data = slice.get_mapped_range();
        let result: Vec<i32> = data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        drop(data);
        staging_buf.unmap();

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Ternary, TernaryTensor};

    fn weights_from_i8(values: &[i8], rows: usize, cols: usize) -> TernaryTensor {
        let ternary: Vec<Ternary> = values.iter().map(|&v| match v {
            -1 => Ternary::Neg,
             0 => Ternary::Zero,
             1 => Ternary::Pos,
             _ => panic!("not ternary"),
        }).collect();
        TernaryTensor::pack(&ternary, rows, cols)
    }

    /// Skip tests if no GPU is available (CI, headless, etc.).
    fn get_backend() -> Option<WgpuBackend> {
        WgpuBackend::try_new()
    }

    #[test]
    fn identity_matvec() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[1, 0, 0, 1], 2, 2);
        let x = vec![42i8, -17i8];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![42, -17]);
    }

    #[test]
    fn negation_matvec() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[-1, 0, 0, -1], 2, 2);
        let x = vec![42i8, -17i8];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![-42, 17]);
    }

    #[test]
    fn mixed_weights() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[1, -1, 0, 1], 1, 4);
        let x = vec![10i8, 20, 30, 40];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![30]); // 10 - 20 + 0 + 40
    }

    #[test]
    fn all_zeros() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[0, 0, 0, 0], 1, 4);
        let x = vec![100i8, -50, 25, -12];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![0]);
    }

    #[test]
    fn non_aligned_cols() {
        let Some(backend) = get_backend() else { return };
        // 5 columns — not a multiple of 4
        let w = weights_from_i8(&[1, -1, 1, 0, -1], 1, 5);
        let x = vec![10i8, 20, 30, 40, 50];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![-30]); // 10 - 20 + 30 + 0 - 50
    }

    #[test]
    fn multi_row() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[
             1,  0,  0,  0,
             0,  1,  0,  0,
            -1, -1, -1, -1,
        ], 3, 4);
        let x = vec![5i8, 10, 15, 20];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![5, 10, -50]);
    }

    #[test]
    fn matches_scalar_random() {
        let Some(backend) = get_backend() else { return };
        let scalar = crate::compute::scalar::ScalarBackend;

        let rows = 64;
        let cols = 128;
        let mut weights_i8 = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 0xDEAD_BEEF;
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 33) % 3) as i8 - 1;
            weights_i8.push(v);
        }
        let w = weights_from_i8(&weights_i8, rows, cols);

        let mut activations = Vec::with_capacity(cols);
        for _ in 0..cols {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 33) % 255) as i8;
            activations.push(v);
        }

        let gpu_result = backend.ternary_matvec(&w, &activations);
        let cpu_result = scalar.ternary_matvec(&w, &activations);

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert_eq!(g, c, "mismatch at row {i}: gpu={g}, cpu={c}");
        }
    }

    #[test]
    fn large_matrix() {
        let Some(backend) = get_backend() else { return };
        let scalar = crate::compute::scalar::ScalarBackend;

        let rows = 256;
        let cols = 2048;
        let mut weights_i8 = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 42;
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            weights_i8.push(((rng >> 33) % 3) as i8 - 1);
        }
        let w = weights_from_i8(&weights_i8, rows, cols);

        let mut activations = Vec::with_capacity(cols);
        for _ in 0..cols {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            activations.push(((rng >> 33) % 127) as i8);
        }

        let gpu_result = backend.ternary_matvec(&w, &activations);
        let cpu_result = scalar.ternary_matvec(&w, &activations);

        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert_eq!(g, c, "mismatch at row {i}: gpu={g}, cpu={c}");
        }
    }

    #[test]
    fn backend_name() {
        let Some(backend) = get_backend() else { return };
        assert_eq!(backend.name(), "wgpu");
    }
}
