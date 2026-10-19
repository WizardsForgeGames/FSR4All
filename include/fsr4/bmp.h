// FSR 4 All — Simple BMP I/O (no external dependencies)
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Supports 24-bit uncompressed BMPs only. Top-down storage.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fsr4 {

struct BmpImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // RGB, row-major, 3 bytes per pixel

    int stride() const { return width * 3; }
    size_t byte_size() const { return (size_t)width * height * 3; }

    uint8_t* row(int y) { return pixels.data() + y * stride(); }
    const uint8_t* row(int y) const { return pixels.data() + y * stride(); }

    void allocate(int w, int h) {
        width = w;
        height = h;
        pixels.resize(byte_size(), 0);
    }
};

#pragma pack(push, 1)
struct BmpFileHeader {
    uint16_t type = 0x4D42;  // "BM"
    uint32_t file_size = 0;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t offset = 54;
};

struct BmpInfoHeader {
    uint32_t size = 40;
    int32_t  width = 0;
    int32_t  height = 0;     // negative = top-down
    uint16_t planes = 1;
    uint16_t bpp = 24;
    uint32_t compression = 0;
    uint32_t image_size = 0;
    int32_t  x_ppm = 2835;   // 72 DPI
    int32_t  y_ppm = 2835;
    uint32_t colors_used = 0;
    uint32_t colors_important = 0;
};
#pragma pack(pop)

inline bool bmp_write(const std::string& path, const BmpImage& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    int row_bytes = img.width * 3;
    int row_pad = (4 - (row_bytes % 4)) % 4;
    int padded_row = row_bytes + row_pad;

    BmpFileHeader fh;
    fh.file_size = 54 + padded_row * img.height;

    BmpInfoHeader ih;
    ih.width = img.width;
    ih.height = -img.height;  // negative = top-down (natural order)
    ih.image_size = padded_row * img.height;

    std::fwrite(&fh, sizeof(fh), 1, f);
    std::fwrite(&ih, sizeof(ih), 1, f);

    uint8_t pad[3] = {0, 0, 0};
    for (int y = 0; y < img.height; y++) {
        const uint8_t* row = img.row(y);
        // BMP stores BGR, our image is RGB
        for (int x = 0; x < img.width; x++) {
            uint8_t bgr[3] = { row[x*3+2], row[x*3+1], row[x*3+0] };
            std::fwrite(bgr, 3, 1, f);
        }
        if (row_pad > 0) std::fwrite(pad, row_pad, 1, f);
    }

    std::fclose(f);
    return true;
}

inline bool bmp_read(const std::string& path, BmpImage& img) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    BmpFileHeader fh;
    BmpInfoHeader ih;
    std::fread(&fh, sizeof(fh), 1, f);
    std::fread(&ih, sizeof(ih), 1, f);

    if (fh.type != 0x4D42 || ih.bpp != 24) {
        std::fclose(f);
        return false;  // only 24-bit BMPs
    }

    bool top_down = (ih.height < 0);
    img.width = ih.width;
    img.height = top_down ? -ih.height : ih.height;
    img.allocate(img.width, img.height);

    int row_bytes = img.width * 3;
    int row_pad = (4 - (row_bytes % 4)) % 4;

    std::fseek(f, fh.offset, SEEK_SET);

    for (int i = 0; i < img.height; i++) {
        int y = top_down ? i : (img.height - 1 - i);
        uint8_t* row = img.row(y);
        // BMP stores BGR, convert to RGB
        for (int x = 0; x < img.width; x++) {
            uint8_t bgr[3];
            std::fread(bgr, 3, 1, f);
            row[x*3+0] = bgr[2];  // R
            row[x*3+1] = bgr[1];  // G
            row[x*3+2] = bgr[0];  // B
        }
        if (row_pad > 0) std::fseek(f, row_pad, SEEK_CUR);
    }

    std::fclose(f);
    return true;
}

// Generate a procedural test image (gradient + checkerboard pattern).
inline BmpImage bmp_generate_test(int w, int h) {
    BmpImage img;
    img.allocate(w, h);

    for (int y = 0; y < h; y++) {
        uint8_t* row = img.row(y);
        for (int x = 0; x < w; x++) {
            float u = (float)x / w;
            float v = (float)y / h;

            // Smooth color gradient
            float r = u;
            float g = v;
            float b = 1.0f - (u + v) * 0.5f;

            // Checkerboard overlay (16px tiles)
            int cx = x / 16;
            int cy = y / 16;
            if ((cx + cy) % 2 == 0) {
                r *= 0.8f; g *= 0.8f; b *= 0.8f;
            }

            // Some circular highlights
            float dx = u - 0.3f, dy = v - 0.4f;
            float d1 = dx*dx + dy*dy;
            if (d1 < 0.02f) { r += 0.5f; g += 0.3f; }

            dx = u - 0.7f; dy = v - 0.6f;
            float d2 = dx*dx + dy*dy;
            if (d2 < 0.03f) { g += 0.4f; b += 0.5f; }

            auto clamp = [](float v) -> uint8_t {
                return v < 0 ? 0 : (v > 1 ? 255 : (uint8_t)(v * 255));
            };
            row[x*3+0] = clamp(r);
            row[x*3+1] = clamp(g);
            row[x*3+2] = clamp(b);
        }
    }
    return img;
}

// Write a single-channel float buffer as a grayscale BMP.
// Values are auto-scaled to [0, 255] using min/max.
inline bool bmp_write_grayscale(const std::string& path,
                                 const float* data, int w, int h) {
    float vmin = data[0], vmax = data[0];
    for (int i = 1; i < w * h; i++) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
    }
    float range = vmax - vmin;
    if (range < 1e-8f) range = 1.0f;

    BmpImage img;
    img.allocate(w, h);
    for (int y = 0; y < h; y++) {
        uint8_t* row = img.row(y);
        for (int x = 0; x < w; x++) {
            float v = (data[y * w + x] - vmin) / range;
            uint8_t b = (uint8_t)(v * 255);
            row[x*3+0] = b;
            row[x*3+1] = b;
            row[x*3+2] = b;
        }
    }
    return bmp_write(path, img);
}

} // namespace fsr4
