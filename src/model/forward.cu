// FSR 4 All — Model Forward Pass Implementation
// Copyright (c) 2026 TheWizardsForge. MIT License.

#include <fsr4/model.h>
#include <forge/nn/int8_conv2d.cuh>
#include <forge/nn/int8_conv_transpose2d.cuh>
#include <cstdio>
#include <algorithm>

namespace fsr4 {

// =========================================================================
// Helper kernels
// =========================================================================

static __global__ void relu_kernel(const __half* __restrict__ in,
                                    __half* __restrict__ out, int n) {
    const __half zero = __float2half(0.0f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = __hgt(in[i], zero) ? in[i] : zero;
    }
}

void fp16_relu(const __half* input, __half* output, int count, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (count + BLOCK - 1) / BLOCK;
    relu_kernel<<<grid, BLOCK, 0, stream>>>(input, output, count);
}

static __global__ void add_kernel(const __half* __restrict__ a,
                                   const __half* __restrict__ b,
                                   __half* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = __hadd(a[i], b[i]);
    }
}

void fp16_add(const __half* a, const __half* b, __half* output, int count,
              cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (count + BLOCK - 1) / BLOCK;
    add_kernel<<<grid, BLOCK, 0, stream>>>(a, b, output, count);
}

// FP16 strided conv2d, k=2 s=2. Each output pixel reads one 2x2 input patch.
// Weight: [C_out, C_in, 2, 2] in NCHW (already converted).
static __global__ void fp16_conv2d_k2s2_kernel(
    const __half* __restrict__ input,   // [C_in, H, W]
    const __half* __restrict__ weight,  // [C_out, C_in*4]
    const __half* __restrict__ bias,    // [C_out]
    __half* __restrict__ output,        // [C_out, H/2, W/2]
    int C_in, int H, int W, int C_out)
{
    int H_out = H / 2;
    int W_out = W / 2;
    int total = C_out * H_out * W_out;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        int co = idx / (H_out * W_out);
        int rem = idx % (H_out * W_out);
        int ho = rem / W_out;
        int wo = rem % W_out;

        float sum = bias ? __half2float(bias[co]) : 0.0f;

        int hi = ho * 2;
        int wi = wo * 2;

        const __half* w_row = weight + co * C_in * 4;

        for (int ci = 0; ci < C_in; ci++) {
            const __half* in_ch = input + ci * H * W;
            float w00 = __half2float(w_row[ci * 4 + 0]);
            float w01 = __half2float(w_row[ci * 4 + 1]);
            float w10 = __half2float(w_row[ci * 4 + 2]);
            float w11 = __half2float(w_row[ci * 4 + 3]);

            float i00 = __half2float(in_ch[hi * W + wi]);
            float i01 = __half2float(in_ch[hi * W + wi + 1]);
            float i10 = __half2float(in_ch[(hi + 1) * W + wi]);
            float i11 = __half2float(in_ch[(hi + 1) * W + wi + 1]);

            sum += i00 * w00 + i01 * w01 + i10 * w10 + i11 * w11;
        }

        output[idx] = __float2half(sum);
    }
}

void fp16_conv2d_k2s2(const __half* input, const __half* weight, const __half* bias,
                       __half* output, int C_in, int H, int W, int C_out,
                       cudaStream_t stream) {
    int H_out = H / 2;
    int W_out = W / 2;
    int total = C_out * H_out * W_out;
    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    fp16_conv2d_k2s2_kernel<<<grid, BLOCK, 0, stream>>>(
        input, weight, bias, output, C_in, H, W, C_out);
}

// Extract first C_first channels from C_total-channel tensor.
static __global__ void channel_slice_kernel(
    const __half* __restrict__ src,  // [C_total, H, W]
    __half* __restrict__ dst,        // [C_first, H, W]
    int C_total, int C_first, int spatial)
{
    int total = C_first * spatial;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        int c = idx / spatial;
        int s = idx % spatial;
        dst[idx] = src[c * spatial + s];
    }
}

void channel_slice(const __half* src, __half* dst,
                   int C_total, int C_first, int H, int W, cudaStream_t stream) {
    int spatial = H * W;
    int total = C_first * spatial;
    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    channel_slice_kernel<<<grid, BLOCK, 0, stream>>>(src, dst, C_total, C_first, spatial);
}

// Concatenate a[C_a, H, W] and b[C_b, H, W] -> out[C_a+C_b, H, W].
static __global__ void channel_concat_kernel(
    const __half* __restrict__ a,    // [C_a, H*W]
    const __half* __restrict__ b,    // [C_b, H*W]
    __half* __restrict__ out,        // [(C_a+C_b), H*W]
    int C_a, int C_b, int spatial)
{
    int C_total = C_a + C_b;
    int total = C_total * spatial;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        int c = idx / spatial;
        int s = idx % spatial;
        if (c < C_a)
            out[idx] = a[c * spatial + s];
        else
            out[idx] = b[(c - C_a) * spatial + s];
    }
}

void channel_concat(const __half* a, const __half* b, __half* output,
                    int C_a, int C_b, int H, int W, cudaStream_t stream) {
    int spatial = H * W;
    int total = (C_a + C_b) * spatial;
    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    channel_concat_kernel<<<grid, BLOCK, 0, stream>>>(a, b, output, C_a, C_b, spatial);
}

// Copy tensor (just cudaMemcpy, but with a cleaner interface).
static void copy_tensor(const __half* src, __half* dst, size_t count,
                        cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, count * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
}

// =========================================================================
// INT8 Conv2d wrapper (uses Forge kernel)
// =========================================================================

static void run_int8_conv2d(
    const __half* input, const ConvLayer& layer, __half* output,
    int H, int W, void* workspace, cudaStream_t stream)
{
    forge::nn::int8_conv2d_forward(
        input, layer.weight, layer.w_scale, layer.bias, output,
        1, layer.C_in, H, W, layer.C_out,
        layer.kH, layer.kW,
        (layer.kH == 3) ? 1 : 0,  // pad=1 for 3x3, pad=0 for others
        (layer.kW == 3) ? 1 : 0,
        (layer.kH == 2) ? 2 : 1,  // stride=2 for k=2 (downscale), stride=1 otherwise
        (layer.kW == 2) ? 2 : 1,
        1, 1,  // dilation
        workspace, stream);
}

static void run_int8_conv_transpose(
    const __half* input, const ConvTransposeLayer& layer, __half* output,
    int H, int W, void* workspace, cudaStream_t stream)
{
    forge::nn::int8_conv_transpose2d_k2s2_forward(
        input, layer.weight, layer.w_scale, layer.bias, output,
        1, layer.C_in, H, W, layer.C_out,
        workspace, stream);
}

// =========================================================================
// Block forward passes
// =========================================================================

// ConvNextBlock: conv3x3 -> pw_expand -> ReLU -> pw_contract + skip
// Input/output: [1, C, H, W] where C = layer channel count.
// Needs tmp buffer for intermediates (at least max(C, 2C) * H * W elements).
static void convnext_forward(
    const __half* input, __half* output,
    const ConvNextBlock& block,
    int H, int W,
    __half* tmp,      // temporary buffer (>= 2C * H * W)
    void* workspace,
    cudaStream_t stream)
{
    int C = block.conv.C_out;    // Stage channel count (e.g., 16)
    int C2 = block.pw_expand.C_out;  // Expanded channels (e.g., 32)
    int spatial = H * W;

    // Stage 1: Conv 3×3 (C→C), output to tmp
    run_int8_conv2d(input, block.conv, tmp, H, W, workspace, stream);

    // Stage 2: PW expand (C→2C) + ReLU, output to output buffer
    run_int8_conv2d(tmp, block.pw_expand, output, H, W, workspace, stream);
    fp16_relu(output, output, C2 * spatial, stream);  // in-place ReLU

    // Stage 3: PW contract (2C→C), output to tmp
    run_int8_conv2d(output, block.pw_contract, tmp, H, W, workspace, stream);

    // Residual add: output = tmp + input
    fp16_add(tmp, input, output, C * spatial, stream);
}

// FasterNetBlock: split → partial conv → cat → expand → ReLU → contract + skip
// Input/output: [1, C_stage, H, W] where C_stage = total channel count.
static void fasternet_forward(
    const __half* input, __half* output,
    const FasterNetBlock& block,
    int C_stage, int H, int W,
    __half* tmp_a,    // temporary buffer A
    __half* tmp_b,    // temporary buffer B
    void* workspace,
    cudaStream_t stream)
{
    int C_partial = block.spatial.C_in;   // First N channels processed (always 16)
    int C_spatial_out = block.spatial.C_out;
    int C_expand = block.pw_expand.C_out;
    int spatial = H * W;

    // Step 1: Extract first C_partial channels for spatial conv
    // tmp_a = input[:C_partial, :, :] (16 channels)
    channel_slice(input, tmp_a, C_stage, C_partial, H, W, stream);

    // Step 2: Spatial conv 3×3 on partial channels
    // tmp_b = conv3x3(tmp_a)  [C_spatial_out, H, W]
    run_int8_conv2d(tmp_a, block.spatial, tmp_b, H, W, workspace, stream);

    // Step 3: Concatenate spatial output + remaining channels from input
    // For FasterNet: concat(spatial_out[C_spatial_out], input[C_partial:C_stage])
    // remaining channels: input offset by C_partial channels
    const __half* remaining = input + C_partial * spatial;
    int C_remaining = C_stage - C_partial;

    // tmp_a = cat(spatial_out, remaining) [C_spatial_out + C_remaining, H, W]
    // Note: C_spatial_out + C_remaining should equal C_stage
    channel_concat(tmp_b, remaining, tmp_a, C_spatial_out, C_remaining, H, W, stream);

    // Step 4: PW expand + ReLU
    // tmp_b = relu(pw_expand(tmp_a))  [C_expand, H, W]
    run_int8_conv2d(tmp_a, block.pw_expand, tmp_b, H, W, workspace, stream);
    fp16_relu(tmp_b, tmp_b, C_expand * spatial, stream);

    // Step 5: PW contract
    // tmp_a = pw_contract(tmp_b)  [C_stage, H, W]
    run_int8_conv2d(tmp_b, block.pw_contract, tmp_a, H, W, workspace, stream);

    // Step 6: Residual add
    // output = tmp_a + input
    fp16_add(tmp_a, input, output, C_stage * spatial, stream);
}

// =========================================================================
// Workspace / buffer computation
// =========================================================================

static size_t compute_max_workspace(const FSR4Weights& w, int H_in, int W_in) {
    size_t max_ws = 0;

    auto check_conv = [&](const ConvLayer& l, int H, int W) {
        int padH = (l.kH == 3) ? 1 : 0;
        int padW = (l.kW == 3) ? 1 : 0;
        int strH = (l.kH == 2) ? 2 : 1;
        int strW = (l.kW == 2) ? 2 : 1;
        size_t ws = forge::nn::int8_conv2d_workspace_size(
            l.C_in, H, W, l.C_out, l.kH, l.kW, padH, padW, strH, strW);
        max_ws = std::max(max_ws, ws);
    };

    auto check_ct = [&](const ConvTransposeLayer& l, int H, int W) {
        size_t ws = forge::nn::int8_conv_transpose2d_k2s2_workspace_size(
            l.C_in, H, W, l.C_out);
        max_ws = std::max(max_ws, ws);
    };

    int H2 = H_in / 2, W2 = W_in / 2;
    int H4 = H_in / 4, W4 = W_in / 4;
    int H8 = H_in / 8, W8 = W_in / 8;

    // Encoder 2 blocks + downscale (H/2 x W/2)
    for (auto& b : w.enc2_res) {
        check_conv(b.conv, H2, W2);
        check_conv(b.pw_expand, H2, W2);
        check_conv(b.pw_contract, H2, W2);
    }
    check_conv(w.enc2_downscale, H2, W2);

    // Encoder 3 blocks + downscale (H/4 x W/4)
    for (auto& b : w.enc3_res) {
        check_conv(b.spatial, H4, W4);
        check_conv(b.pw_expand, H4, W4);
        check_conv(b.pw_contract, H4, W4);
    }
    check_conv(w.enc3_downscale, H4, W4);

    // Bottleneck (H/8 x W/8)
    for (auto& b : w.bn_res) {
        check_conv(b.spatial, H8, W8);
        check_conv(b.pw_expand, H8, W8);
        check_conv(b.pw_contract, H8, W8);
    }
    check_ct(w.bn_upscale, H8, W8);

    // Decoder 3 (H/4 x W/4)
    for (auto& b : w.dec3_res) {
        check_conv(b.spatial, H4, W4);
        check_conv(b.pw_expand, H4, W4);
        check_conv(b.pw_contract, H4, W4);
    }
    check_ct(w.dec3_upscale, H4, W4);

    // Decoder 2 (H/2 x W/2)
    for (auto& b : w.dec2_res) {
        check_conv(b.conv, H2, W2);
        check_conv(b.pw_expand, H2, W2);
        check_conv(b.pw_contract, H2, W2);
    }
    check_ct(w.dec2_upscale, H2, W2);

    return max_ws;
}

// =========================================================================
// FSR4Buffers
// =========================================================================

bool FSR4Buffers::allocate(int H_in, int W_in, bool nhwc_only) {
    cudaError_t err;

    if (nhwc_only) {
        // NHWC fused kernels keep PW expand in registers — buffers only hold
        // layer I/O. Max = 16ch × H/2 × W/2 = 4*H*W at enc2/dec2 level.
        // 3-buffer rotation: skip2 aliases buf_a, skip3 aliases buf_c.
        // Small bn_scratch for bottleneck ping-pong (64ch × H/8 × W/8).
        size_t layer_io_elems = (size_t)4 * H_in * W_in;
        size_t bn_elems = (size_t)(H_in / 8) * (W_in / 8) * 64;

        buf_a_bytes = layer_io_elems * sizeof(__half);
        buf_b_bytes = layer_io_elems * sizeof(__half);
        buf_c_bytes = layer_io_elems * sizeof(__half);
        bn_scratch_bytes = bn_elems * sizeof(__half);
        skip2_bytes = 0;  // aliased to buf_b
        skip3_bytes = 0;  // aliased to buf_a

        err = cudaMalloc(&buf_a, buf_a_bytes);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&buf_b, buf_b_bytes);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&buf_c, buf_c_bytes);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&bn_scratch, bn_scratch_bytes);
        if (err != cudaSuccess) return false;

        skip2 = buf_b;  // alias — enc2_mega writes skip2 to buf_b
        skip3 = buf_a;  // alias — enc3_mega writes skip3 to buf_a
    } else {
        // NCHW path: full-size buffers + separate skip allocations.
        size_t max_elems = (size_t)8 * H_in * W_in;
        buf_a_bytes = max_elems * sizeof(__half);
        buf_b_bytes = max_elems * sizeof(__half);
        buf_c_bytes = max_elems * sizeof(__half);
        skip2_bytes = (size_t)16 * (H_in / 2) * (W_in / 2) * sizeof(__half);
        skip3_bytes = (size_t)32 * (H_in / 4) * (W_in / 4) * sizeof(__half);

        err = cudaMalloc(&buf_a, buf_a_bytes);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&buf_b, buf_b_bytes);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&buf_c, buf_c_bytes);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&skip2, skip2_bytes);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&skip3, skip3_bytes);
        if (err != cudaSuccess) return false;
    }

    return true;
}

void FSR4Buffers::free() {
    if (buf_a)      { cudaFree(buf_a);      buf_a = nullptr; }
    if (buf_b)      { cudaFree(buf_b);      buf_b = nullptr; }
    if (buf_c)      { cudaFree(buf_c);      buf_c = nullptr; }
    if (bn_scratch) { cudaFree(bn_scratch); bn_scratch = nullptr; }
    // skip2/skip3 may be aliases (NHWC mode) — only free if separately allocated
    if (skip2_bytes > 0 && skip2) { cudaFree(skip2); }
    skip2 = nullptr;
    if (skip3_bytes > 0 && skip3) { cudaFree(skip3); }
    skip3 = nullptr;
    if (workspace) { cudaFree(workspace); workspace = nullptr; }
}

size_t FSR4Buffers::total_bytes() const {
    return buf_a_bytes + buf_b_bytes + buf_c_bytes + bn_scratch_bytes
         + skip2_bytes + skip3_bytes + workspace_bytes;
}

// =========================================================================
// FSR4Model
// =========================================================================

bool FSR4Model::init(const std::string& weight_dir, int H_in, int W_in,
                     bool nhwc_only) {
    if (!weights.load(weight_dir)) return false;

    if (!buffers.allocate(H_in, W_in, nhwc_only)) {
        std::fprintf(stderr, "Failed to allocate activation buffers\n");
        return false;
    }

    // Allocate workspace for INT8 conv operations.
    // NHWC path uses all fused/direct kernels — zero workspace needed.
    // NCHW path uses im2col → large workspace. Only allocated if needed.
    size_t nhwc_ws = fsr4_nhwc_workspace_size(weights, H_in, W_in);
    if (nhwc_only) {
        buffers.workspace_bytes = nhwc_ws;  // 0 for current fused NHWC path
    } else {
        size_t nchw_ws = compute_max_workspace(weights, H_in, W_in);
        buffers.workspace_bytes = std::max(nhwc_ws, nchw_ws);
    }

    if (buffers.workspace_bytes > 0) {
        cudaError_t err = cudaMalloc(&buffers.workspace, buffers.workspace_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Failed to allocate workspace (%zu bytes)\n",
                         buffers.workspace_bytes);
            return false;
        }
    }

    std::printf("FSR4 model initialized:\n");
    std::printf("  Weights:    %.1f KB GPU\n", weights.total_gpu_bytes() / 1024.0);
    std::printf("  Buffers:    %.1f MB GPU\n", buffers.total_bytes() / (1024.0 * 1024.0));
    std::printf("  Workspace:  %.1f MB GPU%s\n",
                buffers.workspace_bytes / (1024.0 * 1024.0),
                nhwc_only ? " (NHWC-only)" : "");
    std::printf("  Resolution: %dx%d -> %dx%d\n", W_in, H_in, W_in, H_in);
    return true;
}

void FSR4Model::forward(const __half* input, __half* output,
                         int H_in, int W_in, cudaStream_t stream,
                         FSR4Profile* prof)
{
    // Profiling macros — zero-cost when prof == nullptr
    #define PROF_BEGIN(name) if (prof) prof->begin(name, stream)
    #define PROF_END()       if (prof) prof->end(stream)

    // Three working buffers for ping-pong + scratch.
    __half* a = buffers.buf_a;
    __half* b = buffers.buf_b;
    __half* c = buffers.buf_c;
    void* ws = buffers.workspace;

    int H2 = H_in / 2, W2 = W_in / 2;
    int H4 = H_in / 4, W4 = W_in / 4;
    int H8 = H_in / 8, W8 = W_in / 8;

    // =====================================================================
    // Encoder 1: FP16 Conv2d(7->16, k=2, s=2)
    // =====================================================================
    PROF_BEGIN("enc1_downscale (7->16, k2s2 FP16)");
    fp16_conv2d_k2s2(input, weights.enc1_downscale.weight,
                     weights.enc1_downscale.bias, a,
                     7, H_in, W_in, 16, stream);
    PROF_END();

    // =====================================================================
    // Encoder 2: 2x ConvNextBlock(16ch) + downscale(16->32)
    // =====================================================================
    for (int i = 0; i < 2; i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "enc2_res%d (ConvNext 16ch)", i);
        PROF_BEGIN(name);
        convnext_forward(a, b, weights.enc2_res[i], H2, W2, c, ws, stream);
        PROF_END();
        std::swap(a, b);
    }

    PROF_BEGIN("enc2_skip_copy");
    copy_tensor(a, buffers.skip2, 16 * H2 * W2, stream);
    PROF_END();

    PROF_BEGIN("enc2_downscale (16->32, k2s2)");
    run_int8_conv2d(a, weights.enc2_downscale, b, H2, W2, ws, stream);
    PROF_END();
    std::swap(a, b);

    // =====================================================================
    // Encoder 3: 2x FasterNetBlock(32ch) + downscale(32->64)
    // =====================================================================
    for (int i = 0; i < 2; i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "enc3_res%d (FasterNet 32ch)", i);
        PROF_BEGIN(name);
        fasternet_forward(a, b, weights.enc3_res[i], 32, H4, W4, b, c, ws, stream);
        PROF_END();
        std::swap(a, b);
    }

    PROF_BEGIN("enc3_skip_copy");
    copy_tensor(a, buffers.skip3, 32 * H4 * W4, stream);
    PROF_END();

    PROF_BEGIN("enc3_downscale (32->64, k2s2)");
    run_int8_conv2d(a, weights.enc3_downscale, b, H4, W4, ws, stream);
    PROF_END();
    std::swap(a, b);

    // =====================================================================
    // Bottleneck: 3x FasterNetBlock(64ch) + upscale(64->32)
    // =====================================================================
    for (int i = 0; i < 3; i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "bn_res%d (FasterNet 64ch)", i);
        PROF_BEGIN(name);
        fasternet_forward(a, b, weights.bn_res[i], 64, H8, W8, b, c, ws, stream);
        PROF_END();
        std::swap(a, b);
    }

    PROF_BEGIN("bn_upscale (64->32, CT k2s2)");
    run_int8_conv_transpose(a, weights.bn_upscale, b, H8, W8, ws, stream);
    PROF_END();
    std::swap(a, b);

    // =====================================================================
    // Decoder 3: skip add + 2x FasterNetBlock(32ch) + upscale(32->16)
    // =====================================================================
    PROF_BEGIN("dec3_skip_add");
    fp16_add(a, buffers.skip3, a, 32 * H4 * W4, stream);
    PROF_END();

    for (int i = 0; i < 2; i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "dec3_res%d (FasterNet 32ch)", i);
        PROF_BEGIN(name);
        fasternet_forward(a, b, weights.dec3_res[i], 32, H4, W4, b, c, ws, stream);
        PROF_END();
        std::swap(a, b);
    }

    PROF_BEGIN("dec3_upscale (32->16, CT k2s2)");
    run_int8_conv_transpose(a, weights.dec3_upscale, b, H4, W4, ws, stream);
    PROF_END();
    std::swap(a, b);

    // =====================================================================
    // Decoder 2: skip add + 2x ConvNextBlock(16ch) + upscale(16->8)
    // =====================================================================
    PROF_BEGIN("dec2_skip_add");
    fp16_add(a, buffers.skip2, a, 16 * H2 * W2, stream);
    PROF_END();

    for (int i = 0; i < 2; i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "dec2_res%d (ConvNext 16ch)", i);
        PROF_BEGIN(name);
        convnext_forward(a, b, weights.dec2_res[i], H2, W2, c, ws, stream);
        PROF_END();
        std::swap(a, b);
    }

    PROF_BEGIN("dec2_upscale (16->8, CT k2s2)");
    run_int8_conv_transpose(a, weights.dec2_upscale, output, H2, W2, ws, stream);
    PROF_END();

    #undef PROF_BEGIN
    #undef PROF_END
}

void FSR4Model::free() {
    if (nhwc_graph_exec_) {
        cudaGraphExecDestroy(nhwc_graph_exec_);
        nhwc_graph_exec_ = nullptr;
    }
    if (nhwc_capture_stream_) {
        cudaStreamDestroy(nhwc_capture_stream_);
        nhwc_capture_stream_ = nullptr;
    }
    nhwc_graph_input_ = nullptr;
    nhwc_graph_output_ = nullptr;
    nhwc_graph_H_ = 0;
    nhwc_graph_W_ = 0;
    weights.free();
    buffers.free();
}

} // namespace fsr4
