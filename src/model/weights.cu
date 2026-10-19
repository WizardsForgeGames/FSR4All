// FSR 4 All — Weight Loader Implementation
// Copyright (c) 2026 TheWizardsForge. MIT License.

#include <fsr4/weights.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace fsr4 {

// =========================================================================
// Minimal JSON parser (manifest.json only — no dependencies)
// =========================================================================

struct ManifestLayer {
    std::string name;
    std::string dtype;
    std::string bin_file;
    std::string layout;
    int element_count = 0;
    int byte_size = 0;
    std::vector<int> shape;
    float quant_scale = 0.0f;
    bool has_scale = false;
};

static std::string extract_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\": \"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    auto end = json.find('"', pos);
    return json.substr(pos, end - pos);
}

static int extract_int(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\": ";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return 0;
    pos += needle.size();
    return std::atoi(json.c_str() + pos);
}

static float extract_float(const std::string& json, const std::string& key, bool* found = nullptr) {
    std::string needle = "\"" + key + "\": ";
    auto pos = json.find(needle);
    if (pos == std::string::npos) {
        if (found) *found = false;
        return 0.0f;
    }
    if (found) *found = true;
    pos += needle.size();
    return std::strtof(json.c_str() + pos, nullptr);
}

static std::vector<int> extract_int_array(const std::string& json, const std::string& key) {
    std::vector<int> result;
    std::string needle = "\"" + key + "\": [";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return result;
    pos += needle.size();
    auto end = json.find(']', pos);
    std::string arr = json.substr(pos, end - pos);
    std::istringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        int val = std::atoi(token.c_str());
        result.push_back(val);
    }
    return result;
}

static std::vector<ManifestLayer> parse_manifest(const std::string& json_text) {
    std::vector<ManifestLayer> layers;

    // Split into layer objects
    std::string search = "\"name\": \"";
    size_t pos = 0;
    while ((pos = json_text.find(search, pos)) != std::string::npos) {
        // Find the enclosing { }
        auto brace_start = json_text.rfind('{', pos);
        auto brace_end = json_text.find('}', pos);
        if (brace_start == std::string::npos || brace_end == std::string::npos) break;

        std::string obj = json_text.substr(brace_start, brace_end - brace_start + 1);

        ManifestLayer layer;
        layer.name = extract_string(obj, "name");
        layer.dtype = extract_string(obj, "dtype");
        layer.bin_file = extract_string(obj, "bin_file");
        layer.layout = extract_string(obj, "layout");
        layer.element_count = extract_int(obj, "element_count");
        layer.byte_size = extract_int(obj, "byte_size");
        layer.shape = extract_int_array(obj, "shape");
        bool has_scale = false;
        layer.quant_scale = extract_float(obj, "quant_scale", &has_scale);
        layer.has_scale = has_scale;

        if (!layer.name.empty()) {
            layers.push_back(std::move(layer));
        }
        pos = brace_end + 1;
    }
    return layers;
}

// =========================================================================
// Weight format conversion
// =========================================================================

// Convert INT8 weights from NHWC storage [C_out, kW, kH, C_in] to NCHW [C_out, C_in, kH, kW].
// Source layout verified from HLSL storageByteStrides: C_in stride=1, kH stride=C_in,
// kW stride=C_in*kH, C_out stride=C_in*kH*kW.  (kH/kW swap irrelevant for square kernels.)
// Output is [C_out, C_in*kH*kW] for GEMM B matrix.
static void nhwc_to_nchw_int8(
    const int8_t* src, int8_t* dst,
    int kH, int kW, int C_in, int C_out)
{
    for (int co = 0; co < C_out; co++) {
        for (int ci = 0; ci < C_in; ci++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    // src: [C_out][kH][kW][C_in]  (kH/kW interchangeable for square kernels)
                    int src_idx = co * (kH * kW * C_in)
                                + kh * (kW * C_in)
                                + kw * C_in
                                + ci;
                    // dst: [C_out][C_in][kH][kW]
                    int dst_idx = co * (C_in * kH * kW)
                                + ci * (kH * kW)
                                + kh * kW
                                + kw;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// Convert FP16 weights from NHWC [C_out, kH, kW, C_in_padded] to NCHW [C_out, C_in, kH, kW].
// C_in_padded accounts for storage padding (e.g., 7→8 in encoder1).
static void nhwc_to_nchw_fp16(
    const __half* src, __half* dst,
    int kH, int kW, int C_in, int C_out, int C_in_padded)
{
    for (int co = 0; co < C_out; co++) {
        for (int ci = 0; ci < C_in; ci++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    // src: [C_out][kH][kW][C_in_padded]
                    int src_idx = co * (kH * kW * C_in_padded)
                                + kh * (kW * C_in_padded)
                                + kw * C_in_padded
                                + ci;
                    // dst: [C_out][C_in][kH][kW]
                    int dst_idx = co * (C_in * kH * kW)
                                + ci * (kH * kW)
                                + kh * kW
                                + kw;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// Repack HWCN ConvTranspose2d weight (kH=2,kW=2,C_out,C_in) to Forge format [4*C_out, C_in].
// Row = co*4 + kh*2 + kw, Col = ci.
static void hwcn_to_repacked(
    const int8_t* src, int8_t* dst,
    int C_out, int C_in)
{
    constexpr int kH = 2, kW = 2;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            for (int co = 0; co < C_out; co++) {
                for (int ci = 0; ci < C_in; ci++) {
                    // src[kh][kw][co][ci] in HWCN
                    int src_idx = kh * kW * C_out * C_in
                                + kw * C_out * C_in
                                + co * C_in
                                + ci;
                    // dst[(co*4 + kh*2 + kw) * C_in + ci]
                    int row = co * 4 + kh * 2 + kw;
                    int dst_idx = row * C_in + ci;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// =========================================================================
// File I/O
// =========================================================================

static std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// =========================================================================
// Layer loading helpers
// =========================================================================

// Allocate a [C_out] GPU array filled with a single broadcast scale value.
static __half* alloc_broadcast_scale(float scale, int C_out) {
    std::vector<__half> host(C_out, __float2half(scale));
    __half* gpu = nullptr;
    cudaMalloc(&gpu, C_out * sizeof(__half));
    cudaMemcpy(gpu, host.data(), C_out * sizeof(__half), cudaMemcpyHostToDevice);
    return gpu;
}

using LayerMap = std::unordered_map<std::string, const ManifestLayer*>;

static bool load_conv_layer(
    ConvLayer& layer, const LayerMap& map,
    const std::string& weight_dir,
    const std::string& weight_name,
    const std::string& bias_name)
{
    auto wit = map.find(weight_name);
    if (wit == map.end()) {
        std::fprintf(stderr, "ERROR: weight '%s' not found in manifest\n", weight_name.c_str());
        return false;
    }
    const ManifestLayer& wm = *wit->second;

    if (wm.shape.size() < 4) {
        std::fprintf(stderr, "ERROR: weight '%s' has unexpected shape size %zu\n",
                     weight_name.c_str(), wm.shape.size());
        return false;
    }

    // NHWC shape: (kH, kW, C_in, C_out)
    layer.kH = wm.shape[0];
    layer.kW = wm.shape[1];
    layer.C_in = wm.shape[2];
    layer.C_out = wm.shape[3];

    if (!wm.has_scale) {
        std::fprintf(stderr, "ERROR: weight '%s' missing quant_scale\n", weight_name.c_str());
        return false;
    }
    layer.quant_scale = wm.quant_scale;
    layer.w_scale = alloc_broadcast_scale(wm.quant_scale, layer.C_out);

    // Load and convert weight
    auto raw = read_binary(weight_dir + "/" + wm.bin_file);
    if ((int)raw.size() != wm.byte_size) {
        std::fprintf(stderr, "ERROR: weight '%s' size mismatch: got %zu, expected %d\n",
                     weight_name.c_str(), raw.size(), wm.byte_size);
        return false;
    }

    int nchw_size = layer.C_out * layer.C_in * layer.kH * layer.kW;
    std::vector<int8_t> nchw(nchw_size);
    nhwc_to_nchw_int8(reinterpret_cast<const int8_t*>(raw.data()), nchw.data(),
                       layer.kH, layer.kW, layer.C_in, layer.C_out);

    layer.weight_bytes = nchw_size;
    cudaMalloc(&layer.weight, nchw_size);
    cudaMemcpy(layer.weight, nchw.data(), nchw_size, cudaMemcpyHostToDevice);

    // Load bias (optional)
    auto bit = map.find(bias_name);
    if (bit != map.end()) {
        const ManifestLayer& bm = *bit->second;
        auto bias_data = read_binary(weight_dir + "/" + bm.bin_file);
        layer.bias_bytes = (int)bias_data.size();
        cudaMalloc(&layer.bias, bias_data.size());
        cudaMemcpy(layer.bias, bias_data.data(), bias_data.size(), cudaMemcpyHostToDevice);
    }

    return true;
}

static bool load_fp16_conv_layer(
    FP16ConvLayer& layer, const LayerMap& map,
    const std::string& weight_dir,
    const std::string& weight_name,
    const std::string& bias_name)
{
    auto wit = map.find(weight_name);
    if (wit == map.end()) {
        std::fprintf(stderr, "ERROR: weight '%s' not found\n", weight_name.c_str());
        return false;
    }
    const ManifestLayer& wm = *wit->second;

    layer.kH = wm.shape[0];
    layer.kW = wm.shape[1];
    layer.C_in = wm.shape[2];
    layer.C_out = wm.shape[3];

    auto raw = read_binary(weight_dir + "/" + wm.bin_file);
    int nchw_elems = layer.C_out * layer.C_in * layer.kH * layer.kW;
    int nchw_bytes = nchw_elems * sizeof(__half);
    std::vector<__half> nchw(nchw_elems);
    // Compute C_in_padded from actual storage size (element_count includes padding)
    int C_in_padded = wm.element_count / (layer.C_out * layer.kH * layer.kW);
    nhwc_to_nchw_fp16(reinterpret_cast<const __half*>(raw.data()), nchw.data(),
                       layer.kH, layer.kW, layer.C_in, layer.C_out, C_in_padded);

    layer.weight_bytes = nchw_bytes;
    cudaMalloc(&layer.weight, nchw_bytes);
    cudaMemcpy(layer.weight, nchw.data(), nchw_bytes, cudaMemcpyHostToDevice);

    auto bit = map.find(bias_name);
    if (bit != map.end()) {
        auto bias_data = read_binary(weight_dir + "/" + bit->second->bin_file);
        layer.bias_bytes = (int)bias_data.size();
        cudaMalloc(&layer.bias, bias_data.size());
        cudaMemcpy(layer.bias, bias_data.data(), bias_data.size(), cudaMemcpyHostToDevice);
    }
    return true;
}

static bool load_conv_transpose_layer(
    ConvTransposeLayer& layer, const LayerMap& map,
    const std::string& weight_dir,
    const std::string& weight_name,
    const std::string& bias_name)
{
    auto wit = map.find(weight_name);
    if (wit == map.end()) {
        std::fprintf(stderr, "ERROR: weight '%s' not found\n", weight_name.c_str());
        return false;
    }
    const ManifestLayer& wm = *wit->second;

    // HWCN shape: (kH=2, kW=2, C_out, C_in) for ConvTranspose
    layer.C_out = wm.shape[2];
    layer.C_in = wm.shape[3];

    if (!wm.has_scale) {
        std::fprintf(stderr, "ERROR: weight '%s' missing quant_scale\n", weight_name.c_str());
        return false;
    }
    layer.quant_scale = wm.quant_scale;
    layer.w_scale = alloc_broadcast_scale(wm.quant_scale, layer.C_out);

    auto raw = read_binary(weight_dir + "/" + wm.bin_file);
    int repacked_size = 4 * layer.C_out * layer.C_in;
    std::vector<int8_t> repacked(repacked_size);
    hwcn_to_repacked(reinterpret_cast<const int8_t*>(raw.data()), repacked.data(),
                      layer.C_out, layer.C_in);

    layer.weight_bytes = repacked_size;
    cudaMalloc(&layer.weight, repacked_size);
    cudaMemcpy(layer.weight, repacked.data(), repacked_size, cudaMemcpyHostToDevice);

    auto bit = map.find(bias_name);
    if (bit != map.end()) {
        auto bias_data = read_binary(weight_dir + "/" + bit->second->bin_file);
        layer.bias_bytes = (int)bias_data.size();
        cudaMalloc(&layer.bias, bias_data.size());
        cudaMemcpy(layer.bias, bias_data.data(), bias_data.size(), cudaMemcpyHostToDevice);
    }
    return true;
}

static bool load_convnext_block(
    ConvNextBlock& block, const LayerMap& map,
    const std::string& weight_dir,
    const std::string& prefix)
{
    return load_conv_layer(block.conv, map, weight_dir,
               prefix + "_conv_dw_weight", prefix + "_conv_dw_bias")
        && load_conv_layer(block.pw_expand, map, weight_dir,
               prefix + "_conv_pw_expand_weight", prefix + "_conv_pw_expand_bias")
        && load_conv_layer(block.pw_contract, map, weight_dir,
               prefix + "_conv_pw_contract_weight", prefix + "_conv_pw_contract_bias");
}

static bool load_fasternet_block(
    FasterNetBlock& block, const LayerMap& map,
    const std::string& weight_dir,
    const std::string& prefix)
{
    return load_conv_layer(block.spatial, map, weight_dir,
               prefix + "_spatial_mixing_partial_conv_weight",
               prefix + "_spatial_mixing_partial_conv_bias")
        && load_conv_layer(block.pw_expand, map, weight_dir,
               prefix + "_pw_expand_weight", prefix + "_pw_expand_bias")
        && load_conv_layer(block.pw_contract, map, weight_dir,
               prefix + "_pw_contract_weight", prefix + "_pw_contract_bias");
}

// =========================================================================
// FSR4Weights public API
// =========================================================================

bool FSR4Weights::load(const std::string& weight_dir) {
    std::string manifest_path = weight_dir + "/manifest.json";
    std::string json = read_text(manifest_path);
    if (json.empty()) {
        std::fprintf(stderr, "ERROR: cannot read %s\n", manifest_path.c_str());
        return false;
    }

    auto layers = parse_manifest(json);
    std::printf("Loaded manifest: %zu layers\n", layers.size());

    // Build lookup map
    LayerMap map;
    for (auto& l : layers) {
        map[l.name] = &l;
    }

    bool ok = true;

    // --- Encoder 1 (FP16) ---
    ok = ok && load_fp16_conv_layer(enc1_downscale, map, weight_dir,
        "encoder1_DownscaleStridedConv2x2_downscale_conv_weight",
        "encoder1_DownscaleStridedConv2x2_downscale_conv_bias");

    // --- Encoder 2 (ConvNextBlocks + downscale) ---
    ok = ok && load_convnext_block(enc2_res[0], map, weight_dir,
        "encoder2_ResidualBlock_0_body");
    ok = ok && load_convnext_block(enc2_res[1], map, weight_dir,
        "encoder2_ResidualBlock_1_body");
    ok = ok && load_conv_layer(enc2_downscale, map, weight_dir,
        "encoder2_DownscaleStridedConv2x2_downscale_conv_weight",
        "encoder2_DownscaleStridedConv2x2_downscale_conv_bias");

    // --- Encoder 3 (FasterNetBlocks + downscale) ---
    ok = ok && load_fasternet_block(enc3_res[0], map, weight_dir,
        "encoder3_ResidualBlock_0_body");
    ok = ok && load_fasternet_block(enc3_res[1], map, weight_dir,
        "encoder3_ResidualBlock_1_body");
    ok = ok && load_conv_layer(enc3_downscale, map, weight_dir,
        "encoder3_DownscaleStridedConv2x2_downscale_conv_weight",
        "encoder3_DownscaleStridedConv2x2_downscale_conv_bias");

    // --- Bottleneck (FasterNetBlocks + upscale) ---
    ok = ok && load_fasternet_block(bn_res[0], map, weight_dir,
        "bottleneck_ResidualBlock_0_body");
    ok = ok && load_fasternet_block(bn_res[1], map, weight_dir,
        "bottleneck_ResidualBlock_1_body");
    ok = ok && load_fasternet_block(bn_res[2], map, weight_dir,
        "bottleneck_ResidualBlock_2_body");
    ok = ok && load_conv_transpose_layer(bn_upscale, map, weight_dir,
        "hwcn__bottleneck_UpscaleConvTranspose2x2_upscale_conv_weight",
        "bottleneck_UpscaleConvTranspose2x2_upscale_conv_bias");

    // --- Decoder 3 (FasterNetBlocks + upscale) ---
    // Note: manifest uses ResidualBlock_1 and _2 (not 0,1)
    ok = ok && load_fasternet_block(dec3_res[0], map, weight_dir,
        "decoder3_ResidualBlock_1_body");
    ok = ok && load_fasternet_block(dec3_res[1], map, weight_dir,
        "decoder3_ResidualBlock_2_body");
    ok = ok && load_conv_transpose_layer(dec3_upscale, map, weight_dir,
        "hwcn__decoder3_UpscaleConvTranspose2x2_upscale_conv_weight",
        "decoder3_UpscaleConvTranspose2x2_upscale_conv_bias");

    // --- Decoder 2 (ConvNextBlocks + upscale) ---
    ok = ok && load_convnext_block(dec2_res[0], map, weight_dir,
        "decoder2_ResidualBlock_1_body");
    ok = ok && load_convnext_block(dec2_res[1], map, weight_dir,
        "decoder2_ResidualBlock_2_body");
    ok = ok && load_conv_transpose_layer(dec2_upscale, map, weight_dir,
        "hwcn__decoder2_UpscaleConvTranspose2x2_upscale_conv_weight",
        "decoder2_UpscaleConvTranspose2x2_upscale_conv_bias");

    if (ok) {
        std::printf("All weights loaded successfully (%.1f KB GPU)\n",
                    total_gpu_bytes() / 1024.0f);
    }
    return ok;
}

static void free_conv(ConvLayer& l) {
    if (l.weight)  { cudaFree(l.weight);  l.weight = nullptr; }
    if (l.bias)    { cudaFree(l.bias);    l.bias = nullptr; }
    if (l.w_scale) { cudaFree(l.w_scale); l.w_scale = nullptr; }
}

static void free_conv_transpose(ConvTransposeLayer& l) {
    if (l.weight)  { cudaFree(l.weight);  l.weight = nullptr; }
    if (l.bias)    { cudaFree(l.bias);    l.bias = nullptr; }
    if (l.w_scale) { cudaFree(l.w_scale); l.w_scale = nullptr; }
}

static void free_fp16_conv(FP16ConvLayer& l) {
    if (l.weight) { cudaFree(l.weight); l.weight = nullptr; }
    if (l.bias)   { cudaFree(l.bias);   l.bias = nullptr; }
}

static void free_convnext(ConvNextBlock& b) {
    free_conv(b.conv);
    free_conv(b.pw_expand);
    free_conv(b.pw_contract);
}

static void free_fasternet(FasterNetBlock& b) {
    free_conv(b.spatial);
    free_conv(b.pw_expand);
    free_conv(b.pw_contract);
}

void FSR4Weights::free() {
    free_fp16_conv(enc1_downscale);
    for (auto& b : enc2_res) free_convnext(b);
    free_conv(enc2_downscale);
    for (auto& b : enc3_res) free_fasternet(b);
    free_conv(enc3_downscale);
    for (auto& b : bn_res) free_fasternet(b);
    free_conv_transpose(bn_upscale);
    for (auto& b : dec3_res) free_fasternet(b);
    free_conv_transpose(dec3_upscale);
    for (auto& b : dec2_res) free_convnext(b);
    free_conv_transpose(dec2_upscale);
}

static size_t conv_bytes(const ConvLayer& l) {
    return l.weight_bytes + l.bias_bytes + l.C_out * sizeof(__half);
}

static size_t conv_transpose_bytes(const ConvTransposeLayer& l) {
    return l.weight_bytes + l.bias_bytes + l.C_out * sizeof(__half);
}

static size_t fp16_conv_bytes(const FP16ConvLayer& l) {
    return l.weight_bytes + l.bias_bytes;
}

size_t FSR4Weights::total_gpu_bytes() const {
    size_t total = fp16_conv_bytes(enc1_downscale);
    for (auto& b : enc2_res) {
        total += conv_bytes(b.conv) + conv_bytes(b.pw_expand) + conv_bytes(b.pw_contract);
    }
    total += conv_bytes(enc2_downscale);
    for (auto& b : enc3_res) {
        total += conv_bytes(b.spatial) + conv_bytes(b.pw_expand) + conv_bytes(b.pw_contract);
    }
    total += conv_bytes(enc3_downscale);
    for (auto& b : bn_res) {
        total += conv_bytes(b.spatial) + conv_bytes(b.pw_expand) + conv_bytes(b.pw_contract);
    }
    total += conv_transpose_bytes(bn_upscale);
    for (auto& b : dec3_res) {
        total += conv_bytes(b.spatial) + conv_bytes(b.pw_expand) + conv_bytes(b.pw_contract);
    }
    total += conv_transpose_bytes(dec3_upscale);
    for (auto& b : dec2_res) {
        total += conv_bytes(b.conv) + conv_bytes(b.pw_expand) + conv_bytes(b.pw_contract);
    }
    total += conv_transpose_bytes(dec2_upscale);
    return total;
}

void FSR4Weights::print_summary() const {
    auto print_conv = [](const char* label, const ConvLayer& l) {
        std::printf("  %-45s INT8 [%3d->%3d] k=%dx%d  scale=%.6f  %6d B\n",
                    label, l.C_in, l.C_out, l.kH, l.kW,
                    (double)l.quant_scale, l.weight_bytes + l.bias_bytes);
    };
    auto print_ct = [](const char* label, const ConvTransposeLayer& l) {
        std::printf("  %-45s INT8 [%3d->%3d] k=2x2^  scale=%.6f  %6d B\n",
                    label, l.C_in, l.C_out,
                    (double)l.quant_scale, l.weight_bytes + l.bias_bytes);
    };

    std::printf("\n=== FSR4 Model Weights ===\n");
    std::printf("  %-45s FP16 [%3d->%3d] k=%dx%d              %6d B\n",
                "encoder1_downscale", enc1_downscale.C_in, enc1_downscale.C_out,
                enc1_downscale.kH, enc1_downscale.kW,
                enc1_downscale.weight_bytes + enc1_downscale.bias_bytes);

    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "encoder2_res%d_conv", i);
        print_conv(buf, enc2_res[i].conv);
        std::snprintf(buf, sizeof(buf), "encoder2_res%d_pw_expand", i);
        print_conv(buf, enc2_res[i].pw_expand);
        std::snprintf(buf, sizeof(buf), "encoder2_res%d_pw_contract", i);
        print_conv(buf, enc2_res[i].pw_contract);
    }
    print_conv("encoder2_downscale", enc2_downscale);

    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "encoder3_res%d_spatial", i);
        print_conv(buf, enc3_res[i].spatial);
        std::snprintf(buf, sizeof(buf), "encoder3_res%d_pw_expand", i);
        print_conv(buf, enc3_res[i].pw_expand);
        std::snprintf(buf, sizeof(buf), "encoder3_res%d_pw_contract", i);
        print_conv(buf, enc3_res[i].pw_contract);
    }
    print_conv("encoder3_downscale", enc3_downscale);

    for (int i = 0; i < 3; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "bottleneck_res%d_spatial", i);
        print_conv(buf, bn_res[i].spatial);
        std::snprintf(buf, sizeof(buf), "bottleneck_res%d_pw_expand", i);
        print_conv(buf, bn_res[i].pw_expand);
        std::snprintf(buf, sizeof(buf), "bottleneck_res%d_pw_contract", i);
        print_conv(buf, bn_res[i].pw_contract);
    }
    print_ct("bottleneck_upscale", bn_upscale);

    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "decoder3_res%d_spatial", i);
        print_conv(buf, dec3_res[i].spatial);
        std::snprintf(buf, sizeof(buf), "decoder3_res%d_pw_expand", i);
        print_conv(buf, dec3_res[i].pw_expand);
        std::snprintf(buf, sizeof(buf), "decoder3_res%d_pw_contract", i);
        print_conv(buf, dec3_res[i].pw_contract);
    }
    print_ct("decoder3_upscale", dec3_upscale);

    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "decoder2_res%d_conv", i);
        print_conv(buf, dec2_res[i].conv);
        std::snprintf(buf, sizeof(buf), "decoder2_res%d_pw_expand", i);
        print_conv(buf, dec2_res[i].pw_expand);
        std::snprintf(buf, sizeof(buf), "decoder2_res%d_pw_contract", i);
        print_conv(buf, dec2_res[i].pw_contract);
    }
    print_ct("decoder2_upscale", dec2_upscale);

    std::printf("\nTotal GPU memory: %.1f KB\n", total_gpu_bytes() / 1024.0f);
    std::printf("==========================\n\n");
}

} // namespace fsr4
