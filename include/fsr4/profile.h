// FSR 4 All — CUDA Event Profiling for Forward Pass
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Zero-overhead when disabled (nullptr). Pass FSR4Profile* to forward() to record.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <vector>

namespace fsr4 {

struct FSR4Profile {
    struct Event {
        std::string name;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        float ms = 0.0f;
    };

    std::vector<Event> events;

    // Start timing a named region.
    void begin(const char* name, cudaStream_t stream) {
        Event e;
        e.name = name;
        cudaEventCreate(&e.start);
        cudaEventCreate(&e.stop);
        cudaEventRecord(e.start, stream);
        events.push_back(std::move(e));
    }

    // End timing the most recently started region.
    void end(cudaStream_t stream) {
        if (events.empty()) return;
        cudaEventRecord(events.back().stop, stream);
    }

    // Synchronize and compute all timing results.
    void resolve() {
        cudaDeviceSynchronize();
        for (auto& e : events) {
            if (e.start && e.stop) {
                cudaEventElapsedTime(&e.ms, e.start, e.stop);
            }
        }
    }

    // Print a formatted table.
    void print() const {
        float total = 0;
        for (auto& e : events) total += e.ms;

        std::printf("  %-42s %8s  %6s\n", "Layer", "Time(ms)", "%Total");
        std::printf("  %-42s %8s  %6s\n",
                    "------------------------------------------",
                    "--------", "------");
        for (auto& e : events) {
            float pct = total > 0 ? (e.ms / total * 100.0f) : 0;
            std::printf("  %-42s %8.3f  %5.1f%%\n", e.name.c_str(), e.ms, pct);
        }
        std::printf("  %-42s %8s  %6s\n",
                    "------------------------------------------",
                    "--------", "------");
        std::printf("  %-42s %8.3f  100.0%%\n", "Total", total);
        if (total > 0) {
            std::printf("  %-42s %8.1f\n", "Throughput (FPS)", 1000.0f / total);
        }
    }

    // Clean up CUDA events.
    void free() {
        for (auto& e : events) {
            if (e.start) cudaEventDestroy(e.start);
            if (e.stop)  cudaEventDestroy(e.stop);
        }
        events.clear();
    }

    // Reset for next iteration (reuse events).
    void reset() {
        free();
    }

    // Accumulate timings from another profile (for averaging).
    void accumulate(const FSR4Profile& other) {
        if (events.empty()) {
            events = other.events;
            // Null the event handles (owned by other)
            for (auto& e : events) { e.start = nullptr; e.stop = nullptr; }
            return;
        }
        for (size_t i = 0; i < events.size() && i < other.events.size(); i++) {
            events[i].ms += other.events[i].ms;
        }
    }

    // Divide all timings by N (for averaging).
    void divide(int n) {
        for (auto& e : events) e.ms /= n;
    }
};

} // namespace fsr4
