// FSR 4 All - D3D12 Compute Pipeline Benchmark
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Standalone benchmark: creates a D3D12 device, dummy game resources, and
// measures the full FSR4 dispatch pipeline (Pre + Model×12 + Post + RCAS)
// using GPU timestamp queries. No game required.
//
// Build: cmake --build build --config Release --target fsr4_d3d12_benchmark

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include <fsr4/ffx_api_compat.h>
#include "dll/fsr4_d3d12_pipeline.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

// D3D12 Agility SDK: override system D3D12 runtime with bundled D3D12Core.dll
// This ensures SM 6.4+ is reported on Windows 10 (system runtime caps at SM 6.3)
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 614; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = u8".\\D3D12\\"; }

// =========================================================================
// D3D12 boilerplate
// =========================================================================

struct D3D12Bench {
    IDXGIFactory4*              factory = nullptr;
    IDXGIAdapter1*              adapter = nullptr;
    ID3D12Device*               device  = nullptr;
    ID3D12CommandQueue*         queue   = nullptr;
    ID3D12CommandAllocator*     alloc   = nullptr;
    ID3D12GraphicsCommandList*  cmd     = nullptr;
    ID3D12Fence*                fence   = nullptr;
    HANDLE                      fence_event = nullptr;
    uint64_t                    fence_value = 0;

    // GPU timestamp query heap
    ID3D12QueryHeap*            ts_heap = nullptr;
    ID3D12Resource*             ts_readback = nullptr;
    double                      gpu_tick_ms = 0.0;

    // Dummy game resources
    ID3D12Resource*             color_tex = nullptr;
    ID3D12Resource*             depth_tex = nullptr;
    ID3D12Resource*             mv_tex = nullptr;
    ID3D12Resource*             output_tex = nullptr;

    bool init();
    void create_dummy_resources(uint32_t render_w, uint32_t render_h,
                                 uint32_t upscale_w, uint32_t upscale_h);
    void execute_and_wait();
    void destroy();
};

static ID3D12Resource* create_tex2d(ID3D12Device* dev, uint32_t w, uint32_t h,
                                      DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags,
                                      D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = flags;

    ID3D12Resource* res = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state,
                                  nullptr, IID_PPV_ARGS(&res));
    return res;
}

bool D3D12Bench::init() {
    HRESULT hr;

    // Factory
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { printf("ERROR: CreateDXGIFactory1 failed: 0x%x\n", hr); return false; }

    // Enumerate adapters, pick first hardware GPU
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter->Release();
            adapter = nullptr;
            continue;
        }
        printf("GPU: %ls\n", desc.Description);
        printf("  VRAM: %llu MB\n", desc.DedicatedVideoMemory / (1024 * 1024));
        break;
    }
    if (!adapter) { printf("ERROR: no hardware GPU found\n"); return false; }

    // Device
    hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) { printf("ERROR: D3D12CreateDevice failed: 0x%x\n", hr); return false; }

    // Check SM 6.6 support
    D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_6 };
    hr = device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm));
    if (FAILED(hr) || sm.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
        printf("WARNING: GPU reports SM %.1f (need 6.6). PSO creation may fail.\n",
               (float)(sm.HighestShaderModel >> 4) + (float)(sm.HighestShaderModel & 0xF) / 16.0f);
    } else {
        printf("  Shader Model: 6.%d\n", sm.HighestShaderModel & 0xF);
    }

    // Command queue
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) { printf("ERROR: CreateCommandQueue failed: 0x%x\n", hr); return false; }

    // Command allocator + list
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                         IID_PPV_ARGS(&alloc));
    if (FAILED(hr)) return false;

    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                    alloc, nullptr, IID_PPV_ARGS(&cmd));
    if (FAILED(hr)) return false;
    cmd->Close();  // Start closed — we'll reset per iteration

    // Fence
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) return false;
    fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // GPU timestamp frequency
    uint64_t freq;
    queue->GetTimestampFrequency(&freq);
    gpu_tick_ms = 1000.0 / (double)freq;
    printf("  GPU timestamp freq: %llu Hz (%.3f us/tick)\n", freq, gpu_tick_ms * 1000.0);

    // Timestamp query heap (2 queries per measurement: begin + end)
    D3D12_QUERY_HEAP_DESC qhd = {};
    qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = 64;  // Plenty for multiple measurements
    hr = device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&ts_heap));
    if (FAILED(hr)) return false;

    // Readback buffer for timestamps
    D3D12_HEAP_PROPERTIES rhp = {};
    rhp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rbd = {};
    rbd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rbd.Width = 64 * sizeof(uint64_t);
    rbd.Height = 1;
    rbd.DepthOrArraySize = 1;
    rbd.MipLevels = 1;
    rbd.SampleDesc.Count = 1;
    rbd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    hr = device->CreateCommittedResource(&rhp, D3D12_HEAP_FLAG_NONE, &rbd,
                                          D3D12_RESOURCE_STATE_COPY_DEST,
                                          nullptr, IID_PPV_ARGS(&ts_readback));
    if (FAILED(hr)) return false;

    return true;
}

void D3D12Bench::create_dummy_resources(uint32_t render_w, uint32_t render_h,
                                          uint32_t upscale_w, uint32_t upscale_h) {
    // Release previous
    if (color_tex) { color_tex->Release(); color_tex = nullptr; }
    if (depth_tex) { depth_tex->Release(); depth_tex = nullptr; }
    if (mv_tex)    { mv_tex->Release();    mv_tex = nullptr; }
    if (output_tex){ output_tex->Release(); output_tex = nullptr; }

    auto srv = D3D12_RESOURCE_FLAG_NONE;
    auto uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // Color: game renders at render res, R16G16B16A16_FLOAT (common HDR format)
    color_tex = create_tex2d(device, render_w, render_h,
                              DXGI_FORMAT_R16G16B16A16_FLOAT, srv,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Depth: render res, R32_FLOAT
    depth_tex = create_tex2d(device, render_w, render_h,
                              DXGI_FORMAT_R32_FLOAT, srv,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Motion vectors: render res, R16G16_FLOAT
    mv_tex = create_tex2d(device, render_w, render_h,
                            DXGI_FORMAT_R16G16_FLOAT, srv,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Output: upscale res, R16G16B16A16_FLOAT
    output_tex = create_tex2d(device, upscale_w, upscale_h,
                               DXGI_FORMAT_R16G16B16A16_FLOAT, uav,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    printf("  Dummy resources: color=%ux%u depth=%ux%u mv=%ux%u output=%ux%u\n",
           render_w, render_h, render_w, render_h, render_w, render_h,
           upscale_w, upscale_h);
}

void D3D12Bench::execute_and_wait() {
    cmd->Close();

    ID3D12CommandList* lists[] = { cmd };
    queue->ExecuteCommandLists(1, lists);

    fence_value++;
    queue->Signal(fence, fence_value);
    if (fence->GetCompletedValue() < fence_value) {
        fence->SetEventOnCompletion(fence_value, fence_event);
        WaitForSingleObject(fence_event, INFINITE);
    }

    alloc->Reset();
    cmd->Reset(alloc, nullptr);
}

void D3D12Bench::destroy() {
    if (color_tex)    color_tex->Release();
    if (depth_tex)    depth_tex->Release();
    if (mv_tex)       mv_tex->Release();
    if (output_tex)   output_tex->Release();
    if (ts_readback)  ts_readback->Release();
    if (ts_heap)      ts_heap->Release();
    if (fence_event)  CloseHandle(fence_event);
    if (fence)        fence->Release();
    if (cmd)          cmd->Release();
    if (alloc)        alloc->Release();
    if (queue)        queue->Release();
    if (device)       device->Release();
    if (adapter)      adapter->Release();
    if (factory)      factory->Release();
}

// =========================================================================
// Benchmark runner
// =========================================================================

struct BenchConfig {
    const char* name;
    uint32_t render_w, render_h;
    uint32_t upscale_w, upscale_h;
};

static void run_benchmark(D3D12Bench& bench, fsr4::FSR4D3D12Pipeline& pipeline,
                            const BenchConfig& cfg, int warmup, int iterations) {
    printf("\n--- %s: %ux%u -> %ux%u (%.1fx) ---\n",
           cfg.name, cfg.render_w, cfg.render_h, cfg.upscale_w, cfg.upscale_h,
           (float)cfg.upscale_w / cfg.render_w);

    // Create dummy resources for this resolution
    bench.create_dummy_resources(cfg.render_w, cfg.render_h,
                                  cfg.upscale_w, cfg.upscale_h);

    // Build dispatch descriptor
    ffxDispatchDescUpscale desc = {};
    desc.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    desc.commandList = bench.cmd;
    desc.color.resource = bench.color_tex;
    desc.color.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
    desc.depth.resource = bench.depth_tex;
    desc.depth.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
    desc.motionVectors.resource = bench.mv_tex;
    desc.motionVectors.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
    desc.output.resource = bench.output_tex;
    desc.output.state = FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
    desc.jitterOffset = { 0.25f, -0.375f };
    desc.motionVectorScale = { (float)cfg.render_w, (float)cfg.render_h };
    desc.renderSize = { cfg.render_w, cfg.render_h };
    desc.upscaleSize = { cfg.upscale_w, cfg.upscale_h };
    desc.enableSharpening = false;
    desc.sharpness = 0.0f;
    desc.frameTimeDelta = 16.667f;
    desc.preExposure = 1.0f;
    desc.reset = false;
    desc.cameraNear = 0.1f;
    desc.cameraFar = 10000.0f;
    desc.cameraFovAngleVertical = 1.0472f;  // 60 degrees

    uint32_t create_flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                            FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                            FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;

    // Warmup: run without timing to stabilize GPU clocks & caches
    printf("  Warming up (%d frames)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        desc.reset = (i == 0);

        pipeline.bind_game_resources(bench.device,
            bench.color_tex, DXGI_FORMAT_R16G16B16A16_FLOAT,
            bench.depth_tex, DXGI_FORMAT_R32_FLOAT,
            bench.mv_tex, DXGI_FORMAT_R16G16_FLOAT,
            nullptr,  // No exposure texture — use default
            bench.output_tex, DXGI_FORMAT_R16G16B16A16_FLOAT,
            false);

        pipeline.update_constants(&desc, cfg.upscale_w, cfg.upscale_h,
                                    desc.reset, create_flags);
        pipeline.update_default_exposure(bench.cmd, 1.0f);
        pipeline.dispatch(bench.cmd, cfg.render_w, cfg.render_h,
                           cfg.upscale_w, cfg.upscale_h, false);

        bench.execute_and_wait();
    }

    // Timed iterations: GPU timestamps around each dispatch
    printf("  Benchmarking (%d iterations)...\n", iterations);
    std::vector<double> times_ms;
    times_ms.reserve(iterations);

    // Per-pass accumulators (up to 16 pass intervals from pipeline timestamps)
    static const int MAX_PASSES = 16;
    double pass_sum[MAX_PASSES] = {};
    double pass_min[MAX_PASSES];
    double pass_max[MAX_PASSES] = {};
    int pass_count = 0;
    for (int p = 0; p < MAX_PASSES; p++) pass_min[p] = 1e9;

    for (int i = 0; i < iterations; i++) {
        desc.reset = false;
        desc.jitterOffset.x = 0.5f * sinf((float)i * 0.1f);
        desc.jitterOffset.y = 0.5f * cosf((float)i * 0.1f);

        pipeline.bind_game_resources(bench.device,
            bench.color_tex, DXGI_FORMAT_R16G16B16A16_FLOAT,
            bench.depth_tex, DXGI_FORMAT_R32_FLOAT,
            bench.mv_tex, DXGI_FORMAT_R16G16_FLOAT,
            nullptr,
            bench.output_tex, DXGI_FORMAT_R16G16B16A16_FLOAT,
            false);

        pipeline.update_constants(&desc, cfg.upscale_w, cfg.upscale_h,
                                    false, create_flags);
        pipeline.update_default_exposure(bench.cmd, 1.0f);

        // GPU timestamp: begin
        bench.cmd->EndQuery(bench.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);

        // Dispatch full pipeline (pipeline writes its own per-pass timestamps)
        pipeline.frame_index = warmup + i;
        pipeline.dispatch(bench.cmd, cfg.render_w, cfg.render_h,
                           cfg.upscale_w, cfg.upscale_h, false);

        // GPU timestamp: end
        bench.cmd->EndQuery(bench.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        // Resolve our begin/end timestamps
        bench.cmd->ResolveQueryData(bench.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP,
                                     0, 2, bench.ts_readback, 0);

        bench.execute_and_wait();

        // Read our begin/end timestamps
        uint64_t* ts = nullptr;
        D3D12_RANGE range = { 0, 2 * sizeof(uint64_t) };
        bench.ts_readback->Map(0, &range, (void**)&ts);
        double elapsed = (double)(ts[1] - ts[0]) * bench.gpu_tick_ms;
        D3D12_RANGE no_write = { 0, 0 };
        bench.ts_readback->Unmap(0, &no_write);
        times_ms.push_back(elapsed);

        // Read pipeline's per-pass timestamps
        if (pipeline.timestamp_pending && pipeline.timestamp_readback) {
            uint64_t* pass_ts = nullptr;
            D3D12_RANGE pass_range = { 0, pipeline.timestamp_pass_count * sizeof(uint64_t) };
            pipeline.timestamp_readback->Map(0, &pass_range, (void**)&pass_ts);
            if (pass_ts && pipeline.timestamp_pass_count > 1) {
                int num_intervals = (int)pipeline.timestamp_pass_count - 1;
                pass_count = num_intervals;
                double tick_ms = 1000.0 / (double)pipeline.gpu_timestamp_frequency;
                for (int p = 0; p < num_intervals && p < MAX_PASSES; p++) {
                    double ms = (double)(pass_ts[p + 1] - pass_ts[p]) * tick_ms;
                    pass_sum[p] += ms;
                    if (ms < pass_min[p]) pass_min[p] = ms;
                    if (ms > pass_max[p]) pass_max[p] = ms;
                }
                D3D12_RANGE empty = { 0, 0 };
                pipeline.timestamp_readback->Unmap(0, &empty);
            }
            pipeline.timestamp_pending = false;
        }
    }

    // Statistics
    std::sort(times_ms.begin(), times_ms.end());
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean = sum / times_ms.size();
    double median = times_ms[times_ms.size() / 2];
    double p1 = times_ms[(size_t)(times_ms.size() * 0.01)];
    double p99 = times_ms[(size_t)(times_ms.size() * 0.99)];
    double min_t = times_ms.front();
    double max_t = times_ms.back();

    // Variance
    double var = 0;
    for (double t : times_ms) var += (t - mean) * (t - mean);
    var /= times_ms.size();
    double stddev = sqrt(var);

    printf("\n  Results (%d iterations):\n", iterations);
    printf("    Mean:    %7.3f ms  (%5.1f FPS overhead)\n", mean, 1000.0 / mean);
    printf("    Median:  %7.3f ms\n", median);
    printf("    Min:     %7.3f ms\n", min_t);
    printf("    Max:     %7.3f ms\n", max_t);
    printf("    P1:      %7.3f ms\n", p1);
    printf("    P99:     %7.3f ms\n", p99);
    printf("    Stddev:  %7.3f ms (%.1f%%)\n", stddev, 100.0 * stddev / mean);
    printf("    Budget:  %.1f%% of 16.67ms (60fps)\n", 100.0 * mean / 16.667);
    printf("    Budget:  %.1f%% of 33.33ms (30fps)\n", 100.0 * mean / 33.333);

    // Per-pass breakdown
    if (pass_count > 0) {
        static const char* pass_names[] = {
            "Pre", "Model1", "Model2", "Model3", "Model4",
            "Model5", "Model6", "Model7", "Model8",
            "Model9", "Model10", "Model11", "Model12",
            "Post", "RCAS"
        };
        printf("\n  Per-pass breakdown (avg over %d iterations):\n", iterations);
        double pass_total = 0;
        for (int p = 0; p < pass_count && p < MAX_PASSES; p++) {
            double avg = pass_sum[p] / iterations;
            pass_total += avg;
            const char* name = (p < 15) ? pass_names[p] : "???";
            printf("    %-10s  avg=%6.3f  min=%6.3f  max=%6.3f ms\n",
                   name, avg, pass_min[p], pass_max[p]);
        }
        printf("    %-10s  avg=%6.3f ms\n", "TOTAL", pass_total);
    }
}

// =========================================================================
// Main
// =========================================================================

int main(int argc, char* argv[]) {
    printf("=== FSR4 D3D12 Compute Pipeline Benchmark ===\n\n");

    int warmup = 50;
    int iterations = 200;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: fsr4_d3d12_benchmark [--warmup N] [--iterations N]\n");
            return 0;
        }
    }

    printf("Config: warmup=%d, iterations=%d\n\n", warmup, iterations);

    // Init D3D12
    D3D12Bench bench;
    if (!bench.init()) {
        printf("ERROR: D3D12 initialization failed\n");
        return 1;
    }

    // Benchmark configurations
    BenchConfig configs[] = {
        // 1080p output tier (weakest GPUs — GTX 1660, RX 6600)
        { "540p -> 1080p (Perf)",     960,  540,  1920,  1080 },
        { "720p -> 1080p (Quality)",  1280,  720,  1920,  1080 },

        // 1440p output (mid-range — uses 2160p tier shaders)
        { "720p -> 1440p (Perf)",    1280,  720,  2560,  1440 },

        // 4K output (high-end)
        { "1080p -> 4K (Perf)",      1920, 1080,  3840,  2160 },
        { "1440p -> 4K (Quality)",   2560, 1440,  3840,  2160 },
    };

    int num_configs = sizeof(configs) / sizeof(configs[0]);

    // For each config, create pipeline, benchmark, destroy
    for (int c = 0; c < num_configs; c++) {
        auto& cfg = configs[c];

        // Create pipeline for this resolution
        fsr4::FSR4D3D12Pipeline pipeline;
        pipeline.log = stdout;

        // Set GPU timestamp frequency for per-pass profiling
        uint64_t freq;
        bench.queue->GetTimestampFrequency(&freq);
        pipeline.gpu_timestamp_frequency = freq;

        printf("\nCreating pipeline for %s...\n", cfg.name);
        bool ok = pipeline.create(bench.device,
                                   cfg.render_w, cfg.render_h,
                                   cfg.upscale_w, cfg.upscale_h,
                                   FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                                   FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                                   FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION);
        if (!ok) {
            printf("ERROR: Pipeline creation failed for %s\n", cfg.name);
            printf("  (This GPU may not support SM 6.6 / dot4add_i8packed)\n");
            continue;
        }

        // Reset command list for first use
        bench.alloc->Reset();
        bench.cmd->Reset(bench.alloc, nullptr);

        run_benchmark(bench, pipeline, cfg, warmup, iterations);

        pipeline.destroy();
    }

    // Summary
    printf("\n=== Benchmark Complete ===\n");
    printf("Key metric: mean dispatch time is the GPU cost FSR4 adds per frame.\n");
    printf("For 60fps, the upscaler must stay under ~5ms to leave headroom.\n");
    printf("For 30fps targets, up to ~15ms is acceptable.\n");

    bench.destroy();
    return 0;
}
