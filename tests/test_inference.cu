// FSR 4 All — Standalone Inference Test (FEAT-4 + FEAT-5)
// Full pipeline: load BMP → preprocess → model → postprocess → save upscaled BMP.
//
// Usage:
//   test_inference --input image.bmp [--output dir] [--weights dir] [--scale 2]
//   test_inference --res WxH [--output dir] [--weights dir]    (procedural test pattern)
//   test_inference --input image.bmp --raw                      (raw model, no pipeline)
//
// Pipeline mode (default): produces actual 2x upscaled RGB output.
// Raw mode (--raw): feeds dummy 7ch input directly to model, saves per-channel outputs.

#include <fsr4/pipeline.h>
#include <fsr4/model.h>
#include <fsr4/bmp.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>

struct Config {
    std::string input_path;
    std::string weight_dir = "weights/quality";
    std::string output_dir = "output";
    int width  = 0;
    int height = 0;
    int scale  = 2;
    bool raw_mode = false;
    bool save_channels = false;
    bool bypass_model = false;  // Zero out model output → neutral filter params
    int temporal = 1;  // Number of frames (1=single-shot, >1=temporal accumulation)

    // Sequence mode
    std::string sequence_dir;   // Directory containing frame_NNNNNN.bmp
    int start_frame = 0;        // First frame index to process
    int frame_count = 0;        // 0 = all frames in directory
    int save_every = 1;         // Save every N-th output frame

    // Motion vector support
    std::string mv_dir;             // Directory containing mv_NNNNNN.bin (raw FP16 [2,H,W])
    std::string synthetic_mv;       // Type: translate-x, translate-y, zoom, rotate
    float synthetic_mv_speed = 5.0f; // Pixels per frame for synthetic MVs
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            cfg.input_path = argv[++i];
        else if (std::strcmp(argv[i], "--weights") == 0 && i + 1 < argc)
            cfg.weight_dir = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            cfg.output_dir = argv[++i];
        else if (std::strcmp(argv[i], "--res") == 0 && i + 1 < argc)
            std::sscanf(argv[++i], "%dx%d", &cfg.width, &cfg.height);
        else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc)
            cfg.scale = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--raw") == 0)
            cfg.raw_mode = true;
        else if (std::strcmp(argv[i], "--channels") == 0)
            cfg.save_channels = true;
        else if (std::strcmp(argv[i], "--temporal") == 0 && i + 1 < argc)
            cfg.temporal = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bypass-model") == 0)
            cfg.bypass_model = true;
        else if (std::strcmp(argv[i], "--sequence") == 0 && i + 1 < argc)
            cfg.sequence_dir = argv[++i];
        else if (std::strcmp(argv[i], "--start-frame") == 0 && i + 1 < argc)
            cfg.start_frame = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--frame-count") == 0 && i + 1 < argc)
            cfg.frame_count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--save-every") == 0 && i + 1 < argc)
            cfg.save_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mv-dir") == 0 && i + 1 < argc)
            cfg.mv_dir = argv[++i];
        else if (std::strcmp(argv[i], "--synthetic-mv") == 0 && i + 1 < argc)
            cfg.synthetic_mv = argv[++i];
        else if (std::strcmp(argv[i], "--mv-speed") == 0 && i + 1 < argc)
            cfg.synthetic_mv_speed = (float)std::atof(argv[++i]);
    }
    return cfg;
}

static void align_dims(int& w, int& h) {
    w = (w / 8) * 8;
    h = (h / 8) * 8;
    if (w < 8) w = 8;
    if (h < 8) h = 8;
}

// Convert BMP sRGB uint8 to FP16 NCHW [3, H, W], values in [0,1] (still sRGB gamma).
static std::vector<__half> bmp_to_srgb_fp16(const fsr4::BmpImage& img) {
    int H = img.height, W = img.width;
    std::vector<__half> tensor(3 * H * W);
    for (int y = 0; y < H; y++) {
        const uint8_t* row = img.row(y);
        for (int x = 0; x < W; x++) {
            tensor[0 * H * W + y * W + x] = __float2half(row[x*3+0] / 255.0f);
            tensor[1 * H * W + y * W + x] = __float2half(row[x*3+1] / 255.0f);
            tensor[2 * H * W + y * W + x] = __float2half(row[x*3+2] / 255.0f);
        }
    }
    return tensor;
}

// Convert FP16 NCHW [3, H, W] sRGB values to BMP.
static fsr4::BmpImage fp16_srgb_to_bmp(const __half* data, int W, int H) {
    fsr4::BmpImage img;
    img.allocate(W, H);
    for (int y = 0; y < H; y++) {
        uint8_t* row = img.row(y);
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < 3; c++) {
                float v = __half2float(data[c * H * W + y * W + x]);
                v = std::fmax(0.0f, std::fmin(1.0f, v));
                row[x*3+c] = (uint8_t)(v * 255.0f + 0.5f);
            }
        }
    }
    return img;
}

// ==== Raw model mode (original FEAT-4 behavior) ====
static int run_raw_mode(const Config& cfg, fsr4::BmpImage& input_img) {
    int H = cfg.height, W = cfg.width;
    int spatial = H * W;

    fsr4::FSR4Model model;
    if (!model.init(cfg.weight_dir, H, W)) {
        std::fprintf(stderr, "ERROR: Failed to initialize model\n");
        return 1;
    }

    // Assemble 7ch input: RGB + zeros
    std::vector<__half> h_input(7 * spatial, __float2half(0.0f));
    for (int y = 0; y < H; y++) {
        const uint8_t* row = input_img.row(y);
        for (int x = 0; x < W; x++) {
            h_input[0 * spatial + y * W + x] = __float2half(row[x*3+0] / 255.0f);
            h_input[1 * spatial + y * W + x] = __float2half(row[x*3+1] / 255.0f);
            h_input[2 * spatial + y * W + x] = __float2half(row[x*3+2] / 255.0f);
        }
    }

    __half* d_input = nullptr;
    __half* d_output = nullptr;
    cudaMalloc(&d_input, 7 * spatial * sizeof(__half));
    cudaMalloc(&d_output, 8 * spatial * sizeof(__half));
    cudaMemcpy(d_input, h_input.data(), 7 * spatial * sizeof(__half),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    model.forward(d_input, d_output, H, W);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Raw inference: %.2f ms\n", ms);

    std::vector<__half> h_output(8 * spatial);
    cudaMemcpy(h_output.data(), d_output, 8 * spatial * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // Per-channel stats
    std::printf("\nOutput [1, 8, %d, %d]:\n", H, W);
    std::vector<float> output_f(8 * spatial);
    for (int i = 0; i < 8 * spatial; i++)
        output_f[i] = __half2float(h_output[i]);

    for (int c = 0; c < 8; c++) {
        float* ch = output_f.data() + c * spatial;
        float vmin = ch[0], vmax = ch[0];
        double sum = 0;
        for (int i = 0; i < spatial; i++) {
            vmin = std::min(vmin, ch[i]);
            vmax = std::max(vmax, ch[i]);
            sum += ch[i];
        }
        std::printf("  ch%d: min=%8.4f  max=%8.4f  mean=%8.4f\n",
                    c, vmin, vmax, (float)(sum / spatial));
    }

    // Save channel BMPs
    std::string mkdir_cmd = "mkdir \"" + cfg.output_dir + "\" 2>nul";
    std::system(mkdir_cmd.c_str());

    for (int c = 0; c < 8; c++) {
        std::string path = cfg.output_dir + "/raw_ch" + std::to_string(c) + ".bmp";
        fsr4::bmp_write_grayscale(path, output_f.data() + c * spatial, W, H);
        std::printf("Saved: %s\n", path.c_str());
    }

    model.free();
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

// ==== Pipeline mode (FEAT-5: full upscaling) ====
static int run_pipeline_mode(const Config& cfg, fsr4::BmpImage& input_img) {
    int H_lr = cfg.height, W_lr = cfg.width;
    int H_out = H_lr * cfg.scale;
    int W_out = W_lr * cfg.scale;

    // Align output to 8
    H_out = (H_out / 8) * 8;
    W_out = (W_out / 8) * 8;

    std::printf("Upscaling: %dx%d → %dx%d (%dx)\n\n", W_lr, H_lr, W_out, H_out, cfg.scale);

    // Initialize pipeline
    fsr4::FSR4Pipeline pipeline;
    if (!pipeline.init(cfg.weight_dir, H_lr, W_lr, cfg.scale)) {
        std::fprintf(stderr, "ERROR: Failed to initialize pipeline\n");
        return 1;
    }

    // Convert input BMP to sRGB FP16 and upload
    auto h_srgb = bmp_to_srgb_fp16(input_img);
    __half* d_input_srgb = nullptr;
    cudaMalloc(&d_input_srgb, 3 * H_lr * W_lr * sizeof(__half));
    cudaMemcpy(d_input_srgb, h_srgb.data(), 3 * H_lr * W_lr * sizeof(__half),
               cudaMemcpyHostToDevice);

    // Allocate output buffer
    __half* d_output_srgb = nullptr;
    cudaMalloc(&d_output_srgb, 3 * H_out * W_out * sizeof(__half));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA alloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::string mkdir_cmd = "mkdir \"" + cfg.output_dir + "\" 2>nul";
    std::system(mkdir_cmd.c_str());

    // ---- Temporal mode: run N frames with recurrent feedback ----
    if (cfg.temporal > 1) {
        std::printf("Running %d temporal frames (same input, recurrent feedback)...\n\n",
                    cfg.temporal);

        // Snapshot frames: 1, 5, 10, 20, N
        auto should_save = [&](int f) {
            return f == 1 || f == 5 || f == 10 || f == 20 || f == cfg.temporal;
        };

        for (int frame = 1; frame <= cfg.temporal; frame++) {
            cudaDeviceSynchronize();
            auto ft0 = std::chrono::high_resolution_clock::now();
            pipeline.upscale(d_input_srgb, d_output_srgb);
            cudaDeviceSynchronize();
            auto ft1 = std::chrono::high_resolution_clock::now();

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::fprintf(stderr, "CUDA ERROR at frame %d: %s\n",
                             frame, cudaGetErrorString(err));
                return 1;
            }

            double fms = std::chrono::duration<double, std::milli>(ft1 - ft0).count();

            if (should_save(frame)) {
                std::vector<__half> h_output(3 * H_out * W_out);
                cudaMemcpy(h_output.data(), d_output_srgb,
                           3 * H_out * W_out * sizeof(__half), cudaMemcpyDeviceToHost);

                char fname[64];
                std::snprintf(fname, sizeof(fname), "/frame_%03d.bmp", frame);
                fsr4::BmpImage snap = fp16_srgb_to_bmp(h_output.data(), W_out, H_out);
                fsr4::bmp_write(cfg.output_dir + fname, snap);
                std::printf("  Frame %3d: %6.2f ms  [saved %s]\n",
                            frame, fms, fname + 1);
            } else {
                std::printf("  Frame %3d: %6.2f ms\n", frame, fms);
            }
        }

        std::printf("\nTemporal run complete. Snapshots saved to %s/\n",
                    cfg.output_dir.c_str());
    }

    // ---- Single-frame mode (default) ----
    if (cfg.temporal <= 1) {
        std::printf("Running pipeline...\n");
        fsr4::FSR4Profile profile;

        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        if (cfg.bypass_model) {
            std::printf("\n*** BYPASS MODE: zeroing model output → neutral filter params ***\n");
            pipeline.bypass_model = true;
        }
        pipeline.upscale(d_input_srgb, d_output_srgb, nullptr, 0.0f, 0.0f, nullptr, &profile);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(err));
            return 1;
        }

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("\nPipeline completed in %.2f ms\n", ms);

        profile.resolve();
        std::printf("\nPipeline breakdown:\n");
        profile.print();
        profile.free();
    }

    // Download and save final output
    std::vector<__half> h_output(3 * H_out * W_out);
    cudaMemcpy(h_output.data(), d_output_srgb, 3 * H_out * W_out * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // Save upscaled output
    fsr4::BmpImage output_bmp = fp16_srgb_to_bmp(h_output.data(), W_out, H_out);
    std::string out_path = cfg.output_dir + "/upscaled.bmp";
    fsr4::bmp_write(out_path, output_bmp);
    std::printf("\nSaved: %s (%dx%d)\n", out_path.c_str(), W_out, H_out);

    // Save input copy for comparison
    std::string in_path = cfg.output_dir + "/input.bmp";
    fsr4::bmp_write(in_path, input_img);
    std::printf("Saved: %s (%dx%d)\n", in_path.c_str(), W_lr, H_lr);

    // --- Model output analysis (always print stats, save BMPs with --channels) ---
    {
        int out_spatial = H_out * W_out;
        std::vector<__half> h_model_out(8 * out_spatial);
        cudaMemcpy(h_model_out.data(), pipeline.d_model_output,
                   8 * out_spatial * sizeof(__half), cudaMemcpyDeviceToHost);

        const char* ch_names[] = {
            "correlation", "scale_x", "scale_y", "blend",
            "recurrent0", "recurrent1", "recurrent2", "recurrent3"
        };

        // Model output is NHWC: [H*W, 8]  →  pixel_idx * 8 + channel
        std::vector<float> ch_f(out_spatial);

        std::printf("\nModel output statistics (NHWC, raw values):\n");
        for (int c = 0; c < 8; c++) {
            float vmin = 1e30f, vmax = -1e30f;
            double sum = 0, sum2 = 0;
            for (int i = 0; i < out_spatial; i++) {
                float v = __half2float(h_model_out[i * 8 + c]);  // NHWC
                ch_f[i] = v;
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
                sum += v;
                sum2 += (double)v * v;
            }
            float mean = (float)(sum / out_spatial);
            float std_dev = (float)std::sqrt(sum2 / out_spatial - mean * mean);
            std::printf("  %-12s: min=%8.4f  max=%8.4f  mean=%8.4f  std=%7.4f\n",
                        ch_names[c], vmin, vmax, mean, std_dev);

            if (cfg.save_channels) {
                std::string path = cfg.output_dir + "/ch_" + ch_names[c] + ".bmp";
                fsr4::bmp_write_grayscale(path, ch_f.data(), W_out, H_out);
                std::printf("    → %s\n", path.c_str());
            }
        }

        // Print activated value stats for filter parameters
        std::printf("\nActivated filter parameter stats:\n");
        double corr_sum = 0, sx_sum = 0, sy_sum = 0, bl_sum = 0;
        double corr_min = 1e30, sx_min = 1e30, sy_min = 1e30, bl_min = 1e30;
        double corr_max = -1e30, sx_max = -1e30, sy_max = -1e30, bl_max = -1e30;
        for (int i = 0; i < out_spatial; i++) {
            float r0 = __half2float(h_model_out[i * 8 + 0]);
            float r1 = __half2float(h_model_out[i * 8 + 1]);
            float r2 = __half2float(h_model_out[i * 8 + 2]);
            float r3 = __half2float(h_model_out[i * 8 + 3]);

            float corr = std::tanh(r0);
            float sx = 2.0f / (1.0f + std::exp(-r1));
            float sy = 2.0f / (1.0f + std::exp(-r2));
            float bl = 1.0f / (1.0f + std::exp(-r3));

            corr_sum += corr; corr_min = std::min(corr_min, (double)corr); corr_max = std::max(corr_max, (double)corr);
            sx_sum += sx; sx_min = std::min(sx_min, (double)sx); sx_max = std::max(sx_max, (double)sx);
            sy_sum += sy; sy_min = std::min(sy_min, (double)sy); sy_max = std::max(sy_max, (double)sy);
            bl_sum += bl; bl_min = std::min(bl_min, (double)bl); bl_max = std::max(bl_max, (double)bl);
        }
        std::printf("  correlation [-1,1]: min=%7.4f  max=%7.4f  mean=%7.4f\n",
                    (float)corr_min, (float)corr_max, (float)(corr_sum / out_spatial));
        std::printf("  scale_x     [0,2]:  min=%7.4f  max=%7.4f  mean=%7.4f\n",
                    (float)sx_min, (float)sx_max, (float)(sx_sum / out_spatial));
        std::printf("  scale_y     [0,2]:  min=%7.4f  max=%7.4f  mean=%7.4f\n",
                    (float)sy_min, (float)sy_max, (float)(sy_sum / out_spatial));
        std::printf("  blend       [0,1]:  min=%7.4f  max=%7.4f  mean=%7.4f\n",
                    (float)bl_min, (float)bl_max, (float)(bl_sum / out_spatial));
    }

    // Output statistics
    std::printf("\nOutput RGB statistics:\n");
    for (int c = 0; c < 3; c++) {
        const char* names[] = {"R", "G", "B"};
        float vmin = 1.0f, vmax = 0.0f;
        double sum = 0;
        int out_spatial = H_out * W_out;
        for (int i = 0; i < out_spatial; i++) {
            float v = __half2float(h_output[c * out_spatial + i]);
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
            sum += v;
        }
        std::printf("  %s: min=%.4f  max=%.4f  mean=%.4f\n",
                    names[c], vmin, vmax, (float)(sum / out_spatial));
    }

    // Memory summary
    std::printf("\nMemory usage:\n");
    std::printf("  Model weights: %.1f KB\n",
                pipeline.model.weights.total_gpu_bytes() / 1024.0);
    std::printf("  Model buffers: %.1f MB\n",
                pipeline.model.buffers.total_bytes() / (1024.0 * 1024.0));
    size_t pipe_bytes = (size_t)(3 * H_lr * W_lr + 7 * H_out * W_out
                                 + 8 * H_out * W_out + 3 * H_out * W_out
                                 + 4 * H_out * W_out + 3 * H_out * W_out
                                 + 3 * H_out * W_out + 4 * H_out * W_out) * sizeof(__half);
    std::printf("  Pipeline:      %.1f MB\n", pipe_bytes / (1024.0 * 1024.0));

    // Cleanup
    pipeline.free();
    cudaFree(d_input_srgb);
    cudaFree(d_output_srgb);
    return 0;
}

// ==== Motion vector utilities ====

// Load motion vectors from binary file: raw FP16 [2, H, W]
static std::vector<__half> load_mv_binary(const std::string& path, int H, int W) {
    std::vector<__half> mvs(2 * H * W);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    size_t read = std::fread(mvs.data(), sizeof(__half), 2 * H * W, f);
    std::fclose(f);
    if (read != (size_t)(2 * H * W)) return {};
    return mvs;
}

// Generate synthetic motion vectors at output resolution
static std::vector<__half> generate_synthetic_mv(
    const std::string& type, int H, int W, int frame_idx, float speed)
{
    std::vector<__half> mvs(2 * H * W);
    float cx = W * 0.5f, cy = H * 0.5f;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            float dx = 0.0f, dy = 0.0f;

            if (type == "translate-x") {
                // Uniform horizontal motion: each pixel came from (x - speed) last frame
                dx = -speed;
            } else if (type == "translate-y") {
                dy = -speed;
            } else if (type == "zoom") {
                // Radial outward from center
                float rx = (float)x - cx;
                float ry = (float)y - cy;
                float r = std::sqrt(rx * rx + ry * ry);
                if (r > 1e-3f) {
                    float scale = speed / (r + 1e-3f);
                    dx = -rx * scale;
                    dy = -ry * scale;
                }
            } else if (type == "rotate") {
                // Small rotation around center (speed = degrees per frame)
                float rx = (float)x - cx;
                float ry = (float)y - cy;
                float theta = speed * 3.14159f / 180.0f;  // degrees to radians
                float cos_t = std::cos(theta);
                float sin_t = std::sin(theta);
                // Current pos (x,y) was at (x',y') = R^-1 * (x,y) in previous frame
                float prev_x = rx * cos_t + ry * sin_t;
                float prev_y = -rx * sin_t + ry * cos_t;
                dx = (prev_x + cx) - (float)x;  // (prev_x - x) = motion
                dy = (prev_y + cy) - (float)y;
            }

            mvs[0 * H * W + idx] = __float2half(dx);
            mvs[1 * H * W + idx] = __float2half(dy);
        }
    }
    return mvs;
}

// ==== Sequence mode: process frame sequence with temporal accumulation ====
struct FrameEntry {
    int index;          // Numeric index from filename (e.g., 1 from frame_000001.bmp)
    std::string path;
};

static std::vector<FrameEntry> scan_frame_sequence(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<FrameEntry> entries;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        // Match frame_NNNNNN.bmp (any number of digits)
        if (name.size() > 10 && name.substr(0, 6) == "frame_" &&
            name.substr(name.size() - 4) == ".bmp") {
            // Extract numeric index
            std::string num_str = name.substr(6, name.size() - 10);
            int idx = std::atoi(num_str.c_str());
            entries.push_back({idx, entry.path().string()});
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const FrameEntry& a, const FrameEntry& b) { return a.index < b.index; });
    return entries;
}

static int run_sequence_mode(const Config& cfg) {
    // Scan directory for frame BMPs
    auto all_entries = scan_frame_sequence(cfg.sequence_dir);
    if (all_entries.empty()) {
        std::fprintf(stderr, "ERROR: No frame_*.bmp files found in %s\n",
                     cfg.sequence_dir.c_str());
        return 1;
    }
    std::printf("Found %zu frames in %s\n", all_entries.size(), cfg.sequence_dir.c_str());

    // Apply start_frame and frame_count
    int start = cfg.start_frame;
    if (start >= (int)all_entries.size()) {
        std::fprintf(stderr, "ERROR: start_frame %d >= total frames %zu\n",
                     start, all_entries.size());
        return 1;
    }

    int count = cfg.frame_count > 0
        ? std::min(cfg.frame_count, (int)all_entries.size() - start)
        : (int)all_entries.size() - start;

    std::vector<FrameEntry> frames(all_entries.begin() + start,
                                    all_entries.begin() + start + count);
    std::printf("Processing frames %d-%d (%d frames, indices %d-%d)\n\n",
                start, start + count - 1, count,
                frames.front().index, frames.back().index);

    // Load first frame to get dimensions
    fsr4::BmpImage first_img;
    if (!fsr4::bmp_read(frames[0].path, first_img)) {
        std::fprintf(stderr, "ERROR: Failed to load %s\n", frames[0].path.c_str());
        return 1;
    }
    int H_lr = first_img.height;
    int W_lr = first_img.width;
    // Align to 8
    H_lr = (H_lr / 8) * 8;
    W_lr = (W_lr / 8) * 8;
    int H_out = H_lr * cfg.scale;
    int W_out = W_lr * cfg.scale;
    H_out = (H_out / 8) * 8;
    W_out = (W_out / 8) * 8;

    std::printf("Resolution: %dx%d -> %dx%d (%dx)\n\n", W_lr, H_lr, W_out, H_out, cfg.scale);

    // Initialize pipeline
    fsr4::FSR4Pipeline pipeline;
    if (!pipeline.init(cfg.weight_dir, H_lr, W_lr, cfg.scale)) {
        std::fprintf(stderr, "ERROR: Failed to initialize pipeline\n");
        return 1;
    }
    if (cfg.bypass_model) pipeline.bypass_model = true;

    // Allocate GPU I/O buffers
    __half* d_input_srgb = nullptr;
    __half* d_output_srgb = nullptr;
    __half* d_motion_vectors = nullptr;
    cudaMalloc(&d_input_srgb, 3 * H_lr * W_lr * sizeof(__half));
    cudaMalloc(&d_output_srgb, 3 * H_out * W_out * sizeof(__half));

    bool use_mvs = !cfg.mv_dir.empty() || !cfg.synthetic_mv.empty();
    if (use_mvs) {
        cudaMalloc(&d_motion_vectors, 2 * H_out * W_out * sizeof(__half));
        if (!cfg.mv_dir.empty())
            std::printf("Motion vectors: loading from %s\n", cfg.mv_dir.c_str());
        else
            std::printf("Motion vectors: synthetic '%s' (speed=%.1f px/frame)\n",
                        cfg.synthetic_mv.c_str(), cfg.synthetic_mv_speed);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA alloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Create output directory
    std::string mkdir_cmd = "mkdir \"" + cfg.output_dir + "\" 2>nul";
    std::system(mkdir_cmd.c_str());

    // Scan MV directory if provided
    std::vector<std::string> mv_paths;
    if (!cfg.mv_dir.empty()) {
        namespace fs = std::filesystem;
        for (auto& entry : fs::directory_iterator(cfg.mv_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string name = entry.path().filename().string();
            if (name.size() > 7 && name.substr(0, 3) == "mv_" &&
                name.substr(name.size() - 4) == ".bin") {
                mv_paths.push_back(entry.path().string());
            }
        }
        std::sort(mv_paths.begin(), mv_paths.end());
        std::printf("Found %zu MV files\n", mv_paths.size());
    }

    // Process frames
    double total_ms = 0;
    int save_every = std::max(1, cfg.save_every);

    for (int i = 0; i < count; i++) {
        // Load frame
        int frame_idx = frames[i].index;  // Original frame number from filename
        fsr4::BmpImage frame_img;
        if (!fsr4::bmp_read(frames[i].path, frame_img)) {
            std::fprintf(stderr, "WARNING: Failed to load %s, skipping\n",
                         frames[i].path.c_str());
            continue;
        }

        // Handle dimension mismatch by cropping
        if (frame_img.width != W_lr || frame_img.height != H_lr) {
            if (i == 0) {
                std::fprintf(stderr, "ERROR: First frame already set dims to %dx%d, "
                             "but frame %d is %dx%d\n",
                             W_lr, H_lr, i, frame_img.width, frame_img.height);
                break;
            }
            fsr4::BmpImage cropped;
            int cw = std::min(W_lr, frame_img.width);
            int ch = std::min(H_lr, frame_img.height);
            cropped.allocate(W_lr, H_lr);
            for (int y = 0; y < ch; y++) {
                const uint8_t* src = frame_img.row(y);
                uint8_t* dst = cropped.row(y);
                std::memcpy(dst, src, cw * 3);
            }
            frame_img = std::move(cropped);
        }

        // Convert to FP16 sRGB and upload
        auto h_srgb = bmp_to_srgb_fp16(frame_img);
        cudaMemcpy(d_input_srgb, h_srgb.data(), 3 * H_lr * W_lr * sizeof(__half),
                   cudaMemcpyHostToDevice);

        // Load or generate motion vectors for this frame
        const __half* frame_mvs = nullptr;
        if (use_mvs && i > 0) {  // No MVs for first frame (no history to reproject)
            if (!cfg.mv_dir.empty() && i - 1 < (int)mv_paths.size()) {
                auto h_mvs = load_mv_binary(mv_paths[i - 1], H_out, W_out);
                if (!h_mvs.empty()) {
                    cudaMemcpy(d_motion_vectors, h_mvs.data(),
                               2 * H_out * W_out * sizeof(__half), cudaMemcpyHostToDevice);
                    frame_mvs = d_motion_vectors;
                }
            } else if (!cfg.synthetic_mv.empty()) {
                auto h_mvs = generate_synthetic_mv(
                    cfg.synthetic_mv, H_out, W_out, i, cfg.synthetic_mv_speed);
                cudaMemcpy(d_motion_vectors, h_mvs.data(),
                           2 * H_out * W_out * sizeof(__half), cudaMemcpyHostToDevice);
                frame_mvs = d_motion_vectors;
            }
        }

        // Run pipeline
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        pipeline.upscale(d_input_srgb, d_output_srgb, frame_mvs);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "CUDA ERROR at frame %d: %s\n",
                         i, cudaGetErrorString(err));
            break;
        }

        double fms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += fms;

        // Save output frame (use original frame index for filename alignment)
        bool should_save = (i % save_every == 0) || (i == count - 1);
        if (should_save) {
            std::vector<__half> h_output(3 * H_out * W_out);
            cudaMemcpy(h_output.data(), d_output_srgb,
                       3 * H_out * W_out * sizeof(__half), cudaMemcpyDeviceToHost);

            char fname[64];
            std::snprintf(fname, sizeof(fname), "/frame_%06d.bmp", frame_idx);
            fsr4::BmpImage out_bmp = fp16_srgb_to_bmp(h_output.data(), W_out, H_out);
            fsr4::bmp_write(cfg.output_dir + fname, out_bmp);
            std::printf("  Frame %4d (#%d): %6.2f ms  [saved %s]%s\n",
                        i, frame_idx, fms, fname + 1, frame_mvs ? " +MV" : "");
        } else {
            std::printf("  Frame %4d (#%d): %6.2f ms%s\n",
                        i, frame_idx, fms, frame_mvs ? " +MV" : "");
        }
    }

    // Summary
    std::printf("\n--- Sequence Summary ---\n");
    std::printf("  Frames processed: %d\n", count);
    std::printf("  Total time: %.1f ms\n", total_ms);
    std::printf("  Avg per frame: %.2f ms (%.1f FPS)\n",
                total_ms / count, 1000.0 * count / total_ms);
    std::printf("  Output: %s/\n", cfg.output_dir.c_str());
    if (use_mvs)
        std::printf("  Motion vectors: %s\n",
                    cfg.mv_dir.empty() ? cfg.synthetic_mv.c_str() : cfg.mv_dir.c_str());

    pipeline.free();
    cudaFree(d_input_srgb);
    cudaFree(d_output_srgb);
    if (d_motion_vectors) cudaFree(d_motion_vectors);
    return 0;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    std::printf("=== FSR4 Inference Test ===\n");

    // Sequence mode: process frame directory
    if (!cfg.sequence_dir.empty()) {
        std::printf("Mode: sequence\n\n");
        int result = run_sequence_mode(cfg);
        std::printf("\n=== %s ===\n", result == 0 ? "TEST COMPLETE" : "TEST FAILED");
        return result;
    }

    std::printf("Mode: %s\n\n", cfg.raw_mode ? "raw model" : "full pipeline");

    // Load or generate input
    fsr4::BmpImage input_img;
    if (!cfg.input_path.empty()) {
        std::printf("Loading: %s\n", cfg.input_path.c_str());
        if (!fsr4::bmp_read(cfg.input_path, input_img)) {
            std::fprintf(stderr, "ERROR: Failed to load %s\n", cfg.input_path.c_str());
            return 1;
        }
        if (cfg.width == 0) cfg.width = input_img.width;
        if (cfg.height == 0) cfg.height = input_img.height;
    } else {
        if (cfg.width == 0) cfg.width = 256;
        if (cfg.height == 0) cfg.height = 256;
        std::printf("Generating %dx%d test pattern\n", cfg.width, cfg.height);
    }

    align_dims(cfg.width, cfg.height);
    std::printf("Input resolution: %dx%d\n", cfg.width, cfg.height);

    // Handle size mismatch
    if (input_img.width != cfg.width || input_img.height != cfg.height) {
        if (cfg.input_path.empty() || input_img.width == 0) {
            input_img = fsr4::bmp_generate_test(cfg.width, cfg.height);
        } else {
            // Crop to target size
            fsr4::BmpImage cropped;
            int cw = std::min(cfg.width, input_img.width);
            int ch = std::min(cfg.height, input_img.height);
            cropped.allocate(cfg.width, cfg.height);
            for (int y = 0; y < ch; y++) {
                const uint8_t* src = input_img.row(y);
                uint8_t* dst = cropped.row(y);
                std::memcpy(dst, src, cw * 3);
            }
            input_img = std::move(cropped);
            std::printf("Cropped to %dx%d\n", cfg.width, cfg.height);
        }
    }

    int result;
    if (cfg.raw_mode) {
        result = run_raw_mode(cfg, input_img);
    } else {
        result = run_pipeline_mode(cfg, input_img);
    }

    std::printf("\n=== %s ===\n", result == 0 ? "TEST COMPLETE" : "TEST FAILED");
    return result;
}
