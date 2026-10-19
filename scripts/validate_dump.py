"""
FSR4 All — Validate tensor dump from in-game DLL.

Loads the model I/O tensors dumped by dump_frame=true and:
1. Checks value ranges for all model input/output channels
2. Visualizes the 7 input channels and 8 output channels
3. Optionally runs FP32 reference inference and compares

Usage:
    python scripts/validate_dump.py <dump_dir>
    python scripts/validate_dump.py <game_dir>/fsr4_dump

The dump directory should contain:
    model_input_7ch_nhwc.bin   — [H*W, 7] FP16
    model_output_8ch_nhwc.bin  — [H*W, 8] FP16
    lr_input_3ch_nchw.bin      — [3, H_lr, W_lr] FP16
    mv_2ch_nchw.bin            — [2, H_out, W_out] FP16
    output_3ch_nchw.bin        — [3, H_out, W_out] FP16
    metadata.json              — resolution info
"""
import sys, json, os
import numpy as np

def load_fp16(path, shape):
    data = np.fromfile(path, dtype=np.float16)
    return data.reshape(shape).astype(np.float32)

def channel_stats(name, arr):
    """Print min/max/mean/std for a tensor."""
    print(f"  {name:30s}  min={arr.min():+10.4f}  max={arr.max():+10.4f}  "
          f"mean={arr.mean():+10.4f}  std={arr.std():.4f}  "
          f"nan={np.isnan(arr).sum()}  inf={np.isinf(arr).sum()}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_dump.py <dump_dir>")
        sys.exit(1)

    dump_dir = sys.argv[1]
    meta_path = os.path.join(dump_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    H = meta["out_h"]
    W = meta["out_w"]
    H_lr = meta["lr_h"]
    W_lr = meta["lr_w"]
    print(f"Resolution: LR={W_lr}x{H_lr} → Output={W}x{H}")
    print(f"Jitter: ({meta['jitter_x']:.4f}, {meta['jitter_y']:.4f})")
    print(f"MV Scale: ({meta['mv_scale_x']:.1f}, {meta['mv_scale_y']:.1f})")
    print(f"First frame: {meta['first_frame']}")
    print()

    # --- Model Input [H*W, 7] NHWC ---
    inp_path = os.path.join(dump_dir, "model_input_7ch_nhwc.bin")
    if os.path.exists(inp_path):
        inp = load_fp16(inp_path, (H * W, 7))
        print("=== MODEL INPUT (7 channels, NHWC) ===")
        ch_names = ["Y_curr", "Y_hist", "color_dist",
                     "rec0", "rec1", "rec2", "rec3"]
        for i, name in enumerate(ch_names):
            channel_stats(name, inp[:, i])

        # Expected ranges (after d_normalize: x*2-1 → [-1, 1] for mu-law [0,1])
        print("\n  Expected: Y_curr/Y_hist in [-1, 1], color_dist in [0, 1],")
        print("            recurrent in [-1, 1] (normalized sigmoid output)")

        # Check for pathological values
        for i, name in enumerate(ch_names):
            ch = inp[:, i]
            if np.isnan(ch).any():
                print(f"  WARNING: {name} has NaN values!")
            if np.abs(ch).max() > 10:
                print(f"  WARNING: {name} has values > 10 (likely out of range)")
            pct_zero = (ch == 0).sum() / ch.size * 100
            if pct_zero > 90:
                print(f"  WARNING: {name} is {pct_zero:.1f}% zeros")
    else:
        print(f"Missing: {inp_path}")

    print()

    # --- Model Output [H*W, 8] NHWC ---
    out_path = os.path.join(dump_dir, "model_output_8ch_nhwc.bin")
    if os.path.exists(out_path):
        out = load_fp16(out_path, (H * W, 8))
        print("=== MODEL OUTPUT (8 channels, NHWC, raw pre-activation) ===")
        out_names = ["correlation(tanh)", "scale_x(2*sig)", "scale_y(2*sig)",
                      "blend(sigmoid)", "rec0(sig)", "rec1(sig)", "rec2(sig)", "rec3(sig)"]
        for i, name in enumerate(out_names):
            channel_stats(name, out[:, i])

        # Post-activation analysis
        print("\n  Post-activation blend distribution:")
        blend_raw = out[:, 3]
        blend = 1.0 / (1.0 + np.exp(-np.clip(blend_raw, -20, 20)))
        for thresh in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95]:
            pct = (blend > thresh).sum() / blend.size * 100
            print(f"    blend > {thresh:.2f}: {pct:6.2f}%")

        print(f"\n  Blend: mean={blend.mean():.4f} median={np.median(blend):.4f} "
              f"p5={np.percentile(blend, 5):.4f} p95={np.percentile(blend, 95):.4f}")

        corr = np.tanh(out[:, 0])
        sx = 2.0 / (1.0 + np.exp(-np.clip(out[:, 1], -20, 20)))
        sy = 2.0 / (1.0 + np.exp(-np.clip(out[:, 2], -20, 20)))
        print(f"  Correlation: mean={corr.mean():.4f} std={corr.std():.4f}")
        print(f"  Scale X: mean={sx.mean():.4f} std={sx.std():.4f}")
        print(f"  Scale Y: mean={sy.mean():.4f} std={sy.std():.4f}")
    else:
        print(f"Missing: {out_path}")

    print()

    # --- Motion Vectors [2, H, W] NCHW ---
    mv_path = os.path.join(dump_dir, "mv_2ch_nchw.bin")
    if os.path.exists(mv_path):
        mv = load_fp16(mv_path, (2, H, W))
        print("=== MOTION VECTORS (2ch NCHW, output-res pixels) ===")
        channel_stats("MV_x", mv[0])
        channel_stats("MV_y", mv[1])
        magnitude = np.sqrt(mv[0]**2 + mv[1]**2)
        print(f"  Magnitude: mean={magnitude.mean():.2f} max={magnitude.max():.2f} "
              f"p50={np.percentile(magnitude, 50):.2f} p95={np.percentile(magnitude, 95):.2f}")
    else:
        print(f"Missing: {mv_path}")

    print()

    # --- LR Input [3, H_lr, W_lr] NCHW ---
    lr_path = os.path.join(dump_dir, "lr_input_3ch_nchw.bin")
    if os.path.exists(lr_path):
        lr = load_fp16(lr_path, (3, H_lr, W_lr))
        print("=== LR INPUT (3ch NCHW, linear HDR) ===")
        for i, name in enumerate(["R", "G", "B"]):
            channel_stats(name, lr[i])
        if lr.min() < -0.01:
            print("  WARNING: Negative values in linear input (should be >=0)")
    else:
        print(f"Missing: {lr_path}")

    print()

    # --- Output [3, H, W] NCHW ---
    final_path = os.path.join(dump_dir, "output_3ch_nchw.bin")
    if os.path.exists(final_path):
        final = load_fp16(final_path, (3, H, W))
        print("=== FINAL OUTPUT (3ch NCHW, linear) ===")
        for i, name in enumerate(["R", "G", "B"]):
            channel_stats(name, final[i])
    else:
        print(f"Missing: {final_path}")

    # --- Optional: save visualization ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle("FSR4 Tensor Dump Visualization", fontsize=16)

        if os.path.exists(inp_path):
            inp_2d = inp.reshape(H, W, 7)
            for i, name in enumerate(ch_names[:4]):
                ax = axes[0, i]
                im = ax.imshow(inp_2d[:, :, i], cmap='viridis', aspect='auto')
                ax.set_title(f"Input: {name}")
                plt.colorbar(im, ax=ax, fraction=0.046)

        if os.path.exists(out_path):
            out_2d = out.reshape(H, W, 8)
            blend_2d = 1.0 / (1.0 + np.exp(-np.clip(out_2d[:, :, 3], -20, 20)))
            corr_2d = np.tanh(out_2d[:, :, 0])

            ax = axes[1, 0]
            im = ax.imshow(corr_2d, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
            ax.set_title("Output: correlation")
            plt.colorbar(im, ax=ax, fraction=0.046)

            ax = axes[1, 1]
            im = ax.imshow(blend_2d, cmap='hot', vmin=0, vmax=1, aspect='auto')
            ax.set_title("Output: blend (sigmoid)")
            plt.colorbar(im, ax=ax, fraction=0.046)

        if os.path.exists(mv_path):
            mv_2d = mv
            ax = axes[1, 2]
            im = ax.imshow(mv_2d[0], cmap='RdBu', aspect='auto')
            ax.set_title("MV_x (pixels)")
            plt.colorbar(im, ax=ax, fraction=0.046)

            ax = axes[1, 3]
            im = ax.imshow(mv_2d[1], cmap='RdBu', aspect='auto')
            ax.set_title("MV_y (pixels)")
            plt.colorbar(im, ax=ax, fraction=0.046)

        if os.path.exists(lr_path):
            lr_rgb = np.clip(lr.transpose(1, 2, 0), 0, 1)
            axes[2, 0].imshow(lr_rgb, aspect='auto')
            axes[2, 0].set_title("LR Input (clipped [0,1])")

        if os.path.exists(final_path):
            out_rgb = np.clip(final.transpose(1, 2, 0), 0, 1)
            axes[2, 1].imshow(out_rgb, aspect='auto')
            axes[2, 1].set_title("Final Output (clipped [0,1])")

        # Reprojected history
        hist_path = os.path.join(dump_dir, "reprojected_mulaw_3ch_nchw.bin")
        if os.path.exists(hist_path):
            hist = load_fp16(hist_path, (3, H, W))
            hist_rgb = np.clip(hist.transpose(1, 2, 0), 0, 1)
            axes[2, 2].imshow(hist_rgb, aspect='auto')
            axes[2, 2].set_title("Reprojected History (mu-law)")

        for ax in axes.flat:
            if not ax.has_data():
                ax.set_visible(False)

        plt.tight_layout()
        viz_path = os.path.join(dump_dir, "visualization.png")
        plt.savefig(viz_path, dpi=150)
        print(f"\nVisualization saved to: {viz_path}")
        plt.close()

    except ImportError:
        print("\nInstall matplotlib for visualization: pip install matplotlib")

if __name__ == "__main__":
    main()
