// Single-token ternary matvec — naive scalar loop, one workgroup per row.
//
// Encoding: 0b00 = -1 (Neg), 0b01 = 0 (Zero), 0b10 = +1 (Pos).
// Activations are i8 packed 4-per-u32 (little-endian byte order).
//
// One workgroup per output row (256 threads). Each thread strides
// across the column dimension accumulating partial sums, then a
// tree reduction in shared memory produces the final i32 result.
//
// Used by the resident-weight `GpuBitLinear` decode path and by the
// `ComputeBackend::ternary_matvec` drop-in. For prefill (batched),
// use `ternary_matmul_batch.wgsl` instead — it's tiled and much faster.

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> activations: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
var<workgroup> shared_acc: array<i32, 256>;

@compute @workgroup_size(256)
fn ternary_matvec(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.rows) { return; }

    let cols = params.cols;
    var acc: i32 = 0;

    // Each thread strides across columns by WG_SIZE
    var col = lid;
    while (col < cols) {
        // --- Unpack 2-bit weight ---
        // Global flat index in the ternary tensor
        let flat = row * cols + col;
        // 4 ternary values per byte, so byte index = flat / 4
        let w_byte_idx = flat / 4u;
        let w_bit_shift = (flat % 4u) * 2u;
        // Weights buffer is array<u32>, extract the byte
        let w_u32 = weights[w_byte_idx / 4u];
        let w_byte = (w_u32 >> ((w_byte_idx % 4u) * 8u)) & 0xFFu;
        let w_bits = (w_byte >> w_bit_shift) & 3u;

        // --- Unpack i8 activation ---
        let act_u32 = activations[col / 4u];
        let act_byte = (act_u32 >> ((col % 4u) * 8u)) & 0xFFu;
        // Sign-extend i8 → i32
        var act_val: i32 = i32(act_byte);
        if (act_val > 127) { act_val = act_val - 256; }

        // --- Conditional add/sub/skip ---
        if (w_bits == 0u) {        // Neg (-1)
            acc -= act_val;
        } else if (w_bits == 2u) { // Pos (+1)
            acc += act_val;
        }
        // 1u (Zero) and 3u (unused) → skip

        col += WG_SIZE;
    }

    // --- Workgroup tree reduction ---
    shared_acc[lid] = acc;
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            shared_acc[lid] += shared_acc[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        output[row] = shared_acc[0];
    }
}
