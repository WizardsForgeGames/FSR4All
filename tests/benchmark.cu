// FSR 4 All — Model Throughput Benchmark (FEAT-9)
// CUDA event-based per-layer profiling. No admin privileges required.
//
// Usage:
//   benchmark [--res WxH] [--iters N] [--warmup N] [--weights dir] [--nhwc]
//
// Presets: --720p, --1080p, --1440p, --4k
// --nhwc: Use NHWC-throughout forward pass (eliminates layout transposes)

#include <fsr4/model.h>
#include <fsr4/profile.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

struct Config {
    std::string weight_dir = "weights/quality";
    int width  = 1920;
    int height = 1080;
    int warmup = 5;
    int iters  = 20;
    bool nhwc  = false;
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--weights") == 0 && i + 1 < argc)
            cfg.weight_dir = argv[++i];
        else if (std::strcmp(argv[i], "--res") == 0 && i + 1 < argc)
            std::sscanf(argv[++i], "%dx%d", &cfg.width, &cfg.height);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            cfg.iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            cfg.warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--720p") == 0)
            { cfg.width = 1280; cfg.height = 720; }
        else if (std::strcmp(argv[i], "--1080p") == 0)
            { cfg.width = 1920; cfg.height = 1080; }
        else if (std::strcmp(argv[i], "--1440p") == 0)
            { cfg.width = 2560; cfg.height = 1440; }
        else if (std::strcmp(argv[i], "--4k") == 0)
            { cfg.width = 3840; cfg.height = 2160; }
        else if (std::strcmp(argv[i], "--nhwc") == 0)
            cfg.nhwc = true;
    }
    // Align to 8
    cfg.width = (cfg.width / 8) * 8;
    cfg.height = (cfg.height / 8) * 8;
    return cfg;
}

static std::string get_gpu_name() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.name;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    int H = cfg.height, W = cfg.width;
    int spatial = H * W;

    std::printf("=== FSR4 Inference Benchmark ===\n");
    std::printf("GPU:        %s\n", get_gpu_name().c_str());
    std::printf("Resolution: %dx%d\n", W, H);
    std::printf("Layout:     %s\n", cfg.nhwc ? "NHWC (optimized)" : "NCHW (baseline)");
    std::printf("Warmup:     %d iterations\n", cfg.warmup);
    std::printf("Benchmark:  %d iterations\n\n", cfg.iters);

    // Initialize model
    fsr4::FSR4Model model;
    if (!model.init(cfg.weight_dir, H, W, cfg.nhwc)) {
        std::fprintf(stderr, "ERROR: Failed to initialize model\n");
        return 1;
    }

    // Allocate synthetic input/output
    __half* d_input = nullptr;
    __half* d_output = nullptr;
    size_t input_bytes = (size_t)7 * spatial * sizeof(__half);
    size_t output_bytes = (size_t)8 * spatial * sizeof(__half);
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_input, 0, input_bytes);  // zero input (content doesn't matter for timing)

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Warmup iterations (no profiling, lets CUDA caches settle)
    std::printf("Warming up (%d iters)...\n", cfg.warmup);
    for (int i = 0; i < cfg.warmup; i++) {
        if (cfg.nhwc)
            model.forward_nhwc(d_input, d_output, H, W);
        else
            model.forward(d_input, d_output, H, W);
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR during warmup: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // --- Per-layer profiled run (single iteration for breakdown) ---
    std::printf("Profiling per-layer breakdown...\n\n");
    fsr4::FSR4Profile profile;
    if (cfg.nhwc)
        model.forward_nhwc(d_input, d_output, H, W, nullptr, &profile);
    else
        model.forward(d_input, d_output, H, W, nullptr, &profile);
    profile.resolve();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR during profiling: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::printf("Per-Layer Breakdown (single iteration):\n");
    profile.print();
    profile.free();

    // --- Total throughput (N iterations, wall clock) ---
    std::printf("\nThroughput benchmark (%d iterations):\n", cfg.iters);

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.iters; i++) {
        if (cfg.nhwc)
            model.forward_nhwc(d_input, d_output, H, W);
        else
            model.forward(d_input, d_output, H, W);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / cfg.iters;

    std::printf("  Total:      %.2f ms\n", total_ms);
    std::printf("  Average:    %.2f ms/frame\n", avg_ms);
    std::printf("  Throughput: %.1f FPS\n", 1000.0 / avg_ms);

    // --- Averaged per-layer profile (multiple iterations) ---
    std::printf("\nAveraged Per-Layer (%d iterations):\n", cfg.iters);
    fsr4::FSR4Profile avg_profile;
    for (int i = 0; i < cfg.iters; i++) {
        fsr4::FSR4Profile iter_prof;
        if (cfg.nhwc)
            model.forward_nhwc(d_input, d_output, H, W, nullptr, &iter_prof);
        else
            model.forward(d_input, d_output, H, W, nullptr, &iter_prof);
        iter_prof.resolve();
        avg_profile.accumulate(iter_prof);
        iter_prof.free();
    }
    avg_profile.divide(cfg.iters);
    avg_profile.print();

    // --- Memory summary ---
    std::printf("\nMemory Usage:\n");
    std::printf("  Weights:    %.1f KB\n", model.weights.total_gpu_bytes() / 1024.0);
    std::printf("  Buffers:    %.1f MB\n",
                (model.buffers.total_bytes() - model.buffers.workspace_bytes) / (1024.0 * 1024.0));
    std::printf("  Workspace:  %.1f MB\n",
                model.buffers.workspace_bytes / (1024.0 * 1024.0));
    std::printf("  Total VRAM: %.1f MB\n",
                (model.weights.total_gpu_bytes() + model.buffers.total_bytes()) / (1024.0 * 1024.0));

    // Cleanup
    avg_profile.free();
    model.free();
    cudaFree(d_input);
    cudaFree(d_output);

    std::printf("\n=== BENCHMARK COMPLETE ===\n");
    return 0;
}
