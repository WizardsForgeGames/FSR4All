// FSR 4 All — Test: Forward Pass
// Loads weights, runs the U-Net on synthetic input, validates output shape and values.

#include <fsr4/model.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

// Default test resolution (divisible by 8 for 3 downsample stages).
// Can override with --res WxH, e.g. --res 1920x1080
static int TEST_H = 64;
static int TEST_W = 64;
static constexpr int C_IN  = 7;
static constexpr int C_OUT = 8;

static bool check_cuda(const char* label) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
        return false;
    }
    return true;
}

static void parse_resolution(const char* arg) {
    // Parse WxH format
    int w = 0, h = 0;
    if (std::sscanf(arg, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
        TEST_W = w;
        TEST_H = h;
    }
}

int main(int argc, char** argv) {
    const char* weight_dir = "weights/quality";
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--res") == 0 && i + 1 < argc) {
            parse_resolution(argv[++i]);
        } else {
            weight_dir = argv[i];
        }
    }

    std::printf("=== FSR4 Forward Pass Test ===\n");
    std::printf("Resolution: %dx%d  (input: %dch, output: %dch)\n\n",
                TEST_W, TEST_H, C_IN, C_OUT);

    // Initialize model
    fsr4::FSR4Model model;
    if (!model.init(weight_dir, TEST_H, TEST_W)) {
        std::fprintf(stderr, "FAILED to initialize model\n");
        return 1;
    }

    // Create synthetic FP16 input on GPU: [1, 7, H, W]
    // Fill with small random-ish values (deterministic pattern)
    int input_elems = C_IN * TEST_H * TEST_W;
    int output_elems = C_OUT * TEST_H * TEST_W;

    std::vector<__half> h_input(input_elems);
    for (int i = 0; i < input_elems; i++) {
        // Simple deterministic pattern: small values in [-0.5, 0.5]
        float v = std::sin((float)i * 0.01f) * 0.5f;
        h_input[i] = __float2half(v);
    }

    __half* d_input = nullptr;
    __half* d_output = nullptr;
    cudaMalloc(&d_input, input_elems * sizeof(__half));
    cudaMalloc(&d_output, output_elems * sizeof(__half));
    cudaMemcpy(d_input, h_input.data(), input_elems * sizeof(__half), cudaMemcpyHostToDevice);

    if (!check_cuda("input allocation")) return 1;

    // Run forward pass
    std::printf("Running forward pass...\n");
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    model.forward(d_input, d_output, TEST_H, TEST_W);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!check_cuda("forward pass")) {
        std::fprintf(stderr, "FAILED: CUDA error during forward pass\n");
        model.free();
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Forward pass completed in %.2f ms\n\n", ms);

    // Read back output
    std::vector<__half> h_output(output_elems);
    cudaMemcpy(h_output.data(), d_output, output_elems * sizeof(__half), cudaMemcpyDeviceToHost);

    // Validate output: check for NaN/Inf and compute statistics per channel
    int errors = 0;
    std::printf("Output statistics per channel [1, %d, %d, %d]:\n", C_OUT, TEST_H, TEST_W);
    int spatial = TEST_H * TEST_W;

    for (int c = 0; c < C_OUT; c++) {
        float min_val = 1e9f, max_val = -1e9f;
        double sum = 0.0;
        int nan_count = 0;
        int inf_count = 0;

        for (int s = 0; s < spatial; s++) {
            float v = __half2float(h_output[c * spatial + s]);
            if (std::isnan(v)) { nan_count++; continue; }
            if (std::isinf(v)) { inf_count++; continue; }
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
            sum += v;
        }

        float mean = (float)(sum / spatial);
        std::printf("  ch%d: min=%8.4f  max=%8.4f  mean=%8.4f",
                    c, min_val, max_val, mean);
        if (nan_count > 0) {
            std::printf("  NaN=%d", nan_count);
            errors++;
        }
        if (inf_count > 0) {
            std::printf("  Inf=%d", inf_count);
            errors++;
        }
        std::printf("\n");

        // Check that values are in a reasonable range (not all zeros, not exploded)
        if (max_val - min_val < 1e-10f && nan_count == 0) {
            std::printf("    WARNING: channel %d appears constant (%.6f)\n", c, min_val);
        }
        if (std::abs(max_val) > 1000.0f || std::abs(min_val) > 1000.0f) {
            std::printf("    WARNING: channel %d has large values\n", c);
        }
    }

    // Run a few more iterations for timing
    std::printf("\nBenchmark (10 iterations):\n");
    cudaDeviceSynchronize();
    auto tb0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        model.forward(d_input, d_output, TEST_H, TEST_W);
    }
    cudaDeviceSynchronize();
    auto tb1 = std::chrono::high_resolution_clock::now();

    if (!check_cuda("benchmark")) {
        std::fprintf(stderr, "FAILED: CUDA error during benchmark\n");
        errors++;
    }

    double total_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
    std::printf("  Total: %.2f ms  (%.2f ms/iter)\n", total_ms, total_ms / 10.0);

    // Cleanup
    model.free();
    cudaFree(d_input);
    cudaFree(d_output);

    std::printf("\n=== %s (%d errors) ===\n",
                errors == 0 ? "FORWARD PASS OK" : "ERRORS DETECTED", errors);
    return errors;
}
