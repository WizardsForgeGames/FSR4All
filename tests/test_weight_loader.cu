// FSR 4 All — Test: Weight Loader
// Loads all FSR4 weights and validates shapes/sizes.

#include <fsr4/weights.h>
#include <cstdio>
#include <cstdlib>

static int check_conv(const char* name, const fsr4::ConvLayer& l,
                       int exp_cin, int exp_cout, int exp_kh, int exp_kw) {
    int errors = 0;
    if (l.weight == nullptr) {
        std::fprintf(stderr, "  FAIL: %s weight is null\n", name);
        return 1;
    }
    if (l.C_in != exp_cin || l.C_out != exp_cout || l.kH != exp_kh || l.kW != exp_kw) {
        std::fprintf(stderr, "  FAIL: %s shape mismatch: got (%d,%d,%d,%d) expected (%d,%d,%d,%d)\n",
                     name, l.C_in, l.C_out, l.kH, l.kW, exp_cin, exp_cout, exp_kh, exp_kw);
        errors++;
    }
    int expected_bytes = exp_cout * exp_cin * exp_kh * exp_kw;
    if (l.weight_bytes != expected_bytes) {
        std::fprintf(stderr, "  FAIL: %s weight_bytes=%d expected=%d\n",
                     name, l.weight_bytes, expected_bytes);
        errors++;
    }
    return errors;
}

static int check_ct(const char* name, const fsr4::ConvTransposeLayer& l,
                     int exp_cin, int exp_cout) {
    int errors = 0;
    if (l.weight == nullptr) {
        std::fprintf(stderr, "  FAIL: %s weight is null\n", name);
        return 1;
    }
    if (l.C_in != exp_cin || l.C_out != exp_cout) {
        std::fprintf(stderr, "  FAIL: %s shape mismatch: got C_in=%d C_out=%d expected (%d,%d)\n",
                     name, l.C_in, l.C_out, exp_cin, exp_cout);
        errors++;
    }
    int expected_bytes = 4 * exp_cout * exp_cin;
    if (l.weight_bytes != expected_bytes) {
        std::fprintf(stderr, "  FAIL: %s weight_bytes=%d expected=%d\n",
                     name, l.weight_bytes, expected_bytes);
        errors++;
    }
    return errors;
}

int main(int argc, char** argv) {
    const char* weight_dir = "weights/quality";
    if (argc > 1) weight_dir = argv[1];

    std::printf("=== FSR4 Weight Loader Test ===\n");
    std::printf("Weight directory: %s\n\n", weight_dir);

    fsr4::FSR4Weights weights;
    if (!weights.load(weight_dir)) {
        std::fprintf(stderr, "FAILED to load weights\n");
        return 1;
    }

    weights.print_summary();

    // Validate shapes
    int errors = 0;

    // Encoder 1: FP16 Conv2d(7→16, k=2, s=2)
    if (weights.enc1_downscale.C_in != 7 || weights.enc1_downscale.C_out != 16) {
        std::fprintf(stderr, "  FAIL: enc1_downscale shape\n");
        errors++;
    }

    // Encoder 2: ConvNextBlocks (16ch)
    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "enc2_res[%d].conv", i);
        errors += check_conv(buf, weights.enc2_res[i].conv, 16, 16, 3, 3);
        std::snprintf(buf, sizeof(buf), "enc2_res[%d].pw_expand", i);
        errors += check_conv(buf, weights.enc2_res[i].pw_expand, 16, 32, 1, 1);
        std::snprintf(buf, sizeof(buf), "enc2_res[%d].pw_contract", i);
        errors += check_conv(buf, weights.enc2_res[i].pw_contract, 32, 16, 1, 1);
    }
    errors += check_conv("enc2_downscale", weights.enc2_downscale, 16, 32, 2, 2);

    // Encoder 3: FasterNetBlocks (32ch, spatial uses first 16)
    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "enc3_res[%d].spatial", i);
        errors += check_conv(buf, weights.enc3_res[i].spatial, 16, 16, 3, 3);
        std::snprintf(buf, sizeof(buf), "enc3_res[%d].pw_expand", i);
        errors += check_conv(buf, weights.enc3_res[i].pw_expand, 32, 64, 1, 1);
        std::snprintf(buf, sizeof(buf), "enc3_res[%d].pw_contract", i);
        errors += check_conv(buf, weights.enc3_res[i].pw_contract, 64, 32, 1, 1);
    }
    errors += check_conv("enc3_downscale", weights.enc3_downscale, 32, 64, 2, 2);

    // Bottleneck: FasterNetBlocks (64ch, spatial C_in=16 C_out=32)
    for (int i = 0; i < 3; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "bn_res[%d].spatial", i);
        errors += check_conv(buf, weights.bn_res[i].spatial, 16, 32, 3, 3);
        std::snprintf(buf, sizeof(buf), "bn_res[%d].pw_expand", i);
        errors += check_conv(buf, weights.bn_res[i].pw_expand, 64, 128, 1, 1);
        std::snprintf(buf, sizeof(buf), "bn_res[%d].pw_contract", i);
        errors += check_conv(buf, weights.bn_res[i].pw_contract, 128, 64, 1, 1);
    }
    // Bottleneck upscale: ConvTranspose(C_in=64, C_out=32)
    errors += check_ct("bn_upscale", weights.bn_upscale, 64, 32);

    // Decoder 3: FasterNetBlocks (32ch)
    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "dec3_res[%d].spatial", i);
        errors += check_conv(buf, weights.dec3_res[i].spatial, 16, 16, 3, 3);
        std::snprintf(buf, sizeof(buf), "dec3_res[%d].pw_expand", i);
        errors += check_conv(buf, weights.dec3_res[i].pw_expand, 32, 64, 1, 1);
        std::snprintf(buf, sizeof(buf), "dec3_res[%d].pw_contract", i);
        errors += check_conv(buf, weights.dec3_res[i].pw_contract, 64, 32, 1, 1);
    }
    // Decoder 3 upscale: ConvTranspose(C_in=32, C_out=16)
    errors += check_ct("dec3_upscale", weights.dec3_upscale, 32, 16);

    // Decoder 2: ConvNextBlocks (16ch)
    for (int i = 0; i < 2; i++) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "dec2_res[%d].conv", i);
        errors += check_conv(buf, weights.dec2_res[i].conv, 16, 16, 3, 3);
        std::snprintf(buf, sizeof(buf), "dec2_res[%d].pw_expand", i);
        errors += check_conv(buf, weights.dec2_res[i].pw_expand, 16, 32, 1, 1);
        std::snprintf(buf, sizeof(buf), "dec2_res[%d].pw_contract", i);
        errors += check_conv(buf, weights.dec2_res[i].pw_contract, 32, 16, 1, 1);
    }
    // Decoder 2 upscale: ConvTranspose(C_in=16, C_out=8)
    errors += check_ct("dec2_upscale", weights.dec2_upscale, 16, 8);

    std::printf("\n=== %s (%d errors) ===\n",
                errors == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED", errors);

    weights.free();
    return errors;
}
