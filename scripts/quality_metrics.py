#!/usr/bin/env python3
"""FSR4 Quality Metrics — PSNR, SSIM, and per-channel analysis.

Usage:
    python scripts/quality_metrics.py <upscaled> <reference> [--crop N]
    python scripts/quality_metrics.py output/1080p_test/upscaled.bmp test_images/outer_worlds_4k.bmp
    python scripts/quality_metrics.py output/temporal_1080p/frame_*.bmp --ref test_images/outer_worlds_4k.bmp

Sequence mode (frame-by-frame comparison):
    python scripts/quality_metrics.py --seq-upscaled output/video/ --seq-ref ref_frames/ [--csv out.csv]

Options:
    --ref FILE          Reference image (if comparing multiple upscaled frames)
    --crop N            Crop N pixels from each edge before comparison
    --save FILE         Save difference heatmap as BMP
    --seq-upscaled DIR  Upscaled frame sequence directory
    --seq-ref DIR       Reference frame sequence directory
    --csv FILE          Save per-frame metrics as CSV
"""

import sys
import os
import re
import glob
import argparse
import csv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def load_bmp(path):
    """Load BMP as float32 RGB array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def crop_border(img, n):
    """Crop n pixels from each edge."""
    if n <= 0:
        return img
    return img[n:-n, n:-n]


def compute_metrics(upscaled, reference):
    """Compute PSNR and SSIM between two images."""
    assert upscaled.shape == reference.shape, (
        f"Shape mismatch: {upscaled.shape} vs {reference.shape}"
    )

    # Overall PSNR (treats all channels together)
    psnr_val = psnr(reference, upscaled, data_range=1.0)

    # Overall SSIM (multichannel)
    ssim_val = ssim(reference, upscaled, data_range=1.0, channel_axis=2)

    # Per-channel PSNR
    ch_names = ["R", "G", "B"]
    ch_psnr = {}
    ch_ssim = {}
    for i, name in enumerate(ch_names):
        ch_psnr[name] = psnr(reference[:, :, i], upscaled[:, :, i], data_range=1.0)
        ch_ssim[name] = ssim(reference[:, :, i], upscaled[:, :, i], data_range=1.0)

    # Error statistics
    diff = np.abs(upscaled - reference)
    mae = np.mean(diff)
    max_err = np.max(diff)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "ch_psnr": ch_psnr,
        "ch_ssim": ch_ssim,
        "mae": mae,
        "max_error": max_err,
        "diff": diff,
    }


def print_metrics(name, metrics):
    """Print formatted metrics."""
    print(f"  {'PSNR':>8s}: {metrics['psnr']:.2f} dB")
    print(f"  {'SSIM':>8s}: {metrics['ssim']:.4f}")
    print(f"  {'MAE':>8s}: {metrics['mae']:.6f} ({metrics['mae']*255:.2f}/255)")
    print(f"  {'Max Err':>8s}: {metrics['max_error']:.4f} ({metrics['max_error']*255:.1f}/255)")
    print(f"  Per-channel PSNR:  R={metrics['ch_psnr']['R']:.2f}  "
          f"G={metrics['ch_psnr']['G']:.2f}  B={metrics['ch_psnr']['B']:.2f}")
    print(f"  Per-channel SSIM:  R={metrics['ch_ssim']['R']:.4f}  "
          f"G={metrics['ch_ssim']['G']:.4f}  B={metrics['ch_ssim']['B']:.4f}")


def save_heatmap(diff, path):
    """Save error heatmap as BMP (amplified 10x for visibility)."""
    # Mean across channels, amplify, clamp
    gray = np.mean(diff, axis=2)
    amplified = np.clip(gray * 10.0, 0, 1)
    # Apply colormap: black → red → yellow → white
    h, w = amplified.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(amplified * 3.0, 0, 1) * 255          # R ramps first
    rgb[:, :, 1] = np.clip((amplified - 0.33) * 3.0, 0, 1) * 255  # G ramps second
    rgb[:, :, 2] = np.clip((amplified - 0.66) * 3.0, 0, 1) * 255  # B ramps last
    Image.fromarray(rgb).save(path)
    print(f"  Saved heatmap: {path}")


def grade(psnr_val, ssim_val):
    """Simple quality grade."""
    if psnr_val >= 35 and ssim_val >= 0.95:
        return "EXCELLENT"
    elif psnr_val >= 30 and ssim_val >= 0.90:
        return "GOOD"
    elif psnr_val >= 25 and ssim_val >= 0.80:
        return "FAIR"
    else:
        return "POOR"


def scan_frames(directory):
    """Scan directory for frame_NNNNNN.bmp files, return sorted paths."""
    pattern = re.compile(r"^frame_(\d+)\.bmp$")
    frames = []
    for f in os.listdir(directory):
        m = pattern.match(f)
        if m:
            frames.append((int(m.group(1)), os.path.join(directory, f)))
    frames.sort(key=lambda x: x[0])
    return frames


def run_sequence_comparison(upscaled_dir, ref_dir, crop_n, csv_path):
    """Compare frame sequences and compute per-frame + temporal metrics."""
    up_frames = scan_frames(upscaled_dir)
    ref_frames = scan_frames(ref_dir)

    if not up_frames:
        print(f"ERROR: No frame_*.bmp files in {upscaled_dir}", file=sys.stderr)
        sys.exit(1)
    if not ref_frames:
        print(f"ERROR: No frame_*.bmp files in {ref_dir}", file=sys.stderr)
        sys.exit(1)

    # Match frames by index
    ref_dict = {idx: path for idx, path in ref_frames}
    matched = []
    for idx, up_path in up_frames:
        if idx in ref_dict:
            matched.append((idx, up_path, ref_dict[idx]))

    if not matched:
        # If indices don't match, try pairing by order
        n = min(len(up_frames), len(ref_frames))
        for i in range(n):
            matched.append((up_frames[i][0], up_frames[i][1], ref_frames[i][1]))

    print(f"Upscaled: {upscaled_dir} ({len(up_frames)} frames)")
    print(f"Reference: {ref_dir} ({len(ref_frames)} frames)")
    print(f"Matched: {len(matched)} frame pairs")
    print()

    results = []
    prev_up_img = None

    for idx, up_path, ref_path in matched:
        up_img = load_bmp(up_path)
        ref_img = load_bmp(ref_path)

        if crop_n > 0:
            up_img = crop_border(up_img, crop_n)
            ref_img = crop_border(ref_img, crop_n)

        if up_img.shape != ref_img.shape:
            print(f"  Frame {idx}: shape mismatch {up_img.shape} vs {ref_img.shape}, skipping")
            continue

        psnr_val = psnr(ref_img, up_img, data_range=1.0)
        ssim_val = ssim(ref_img, up_img, data_range=1.0, channel_axis=2)
        mae_val = np.mean(np.abs(up_img - ref_img))

        # Temporal consistency: PSNR between consecutive upscaled frames
        temporal_psnr = None
        temporal_diff = None
        if prev_up_img is not None and prev_up_img.shape == up_img.shape:
            temporal_psnr = psnr(prev_up_img, up_img, data_range=1.0)
            temporal_diff = np.mean(np.abs(up_img - prev_up_img))

        results.append({
            "frame": idx,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "mae": mae_val,
            "temporal_psnr": temporal_psnr,
            "temporal_diff": temporal_diff,
        })

        tp_str = f"  t_psnr={temporal_psnr:.2f}" if temporal_psnr else ""
        print(f"  Frame {idx:6d}: PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  MAE={mae_val:.6f}{tp_str}")

        prev_up_img = up_img

    # Summary
    if results:
        psnrs = [r["psnr"] for r in results]
        ssims = [r["ssim"] for r in results]
        maes = [r["mae"] for r in results]
        t_psnrs = [r["temporal_psnr"] for r in results if r["temporal_psnr"] is not None]

        print()
        print("=" * 60)
        print("SEQUENCE SUMMARY")
        print("=" * 60)
        print(f"  Frames: {len(results)}")
        print(f"  PSNR:   avg={np.mean(psnrs):.2f}  min={np.min(psnrs):.2f}  max={np.max(psnrs):.2f}")
        print(f"  SSIM:   avg={np.mean(ssims):.4f}  min={np.min(ssims):.4f}  max={np.max(ssims):.4f}")
        print(f"  MAE:    avg={np.mean(maes):.6f}")
        if t_psnrs:
            print(f"  Temporal PSNR (inter-frame): avg={np.mean(t_psnrs):.2f}  "
                  f"min={np.min(t_psnrs):.2f}  max={np.max(t_psnrs):.2f}")
        avg_g = grade(np.mean(psnrs), np.mean(ssims))
        print(f"  Grade: {avg_g}")

    # CSV output
    if csv_path and results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame", "psnr", "ssim", "mae", "temporal_psnr", "temporal_diff"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "frame": r["frame"],
                    "psnr": f"{r['psnr']:.4f}",
                    "ssim": f"{r['ssim']:.6f}",
                    "mae": f"{r['mae']:.8f}",
                    "temporal_psnr": f"{r['temporal_psnr']:.4f}" if r["temporal_psnr"] else "",
                    "temporal_diff": f"{r['temporal_diff']:.8f}" if r["temporal_diff"] else "",
                })
        print(f"\nCSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="FSR4 Quality Metrics")
    parser.add_argument("images", nargs="*", help="Upscaled image(s) to evaluate")
    parser.add_argument("--ref", help="Reference image path")
    parser.add_argument("--crop", type=int, default=0, help="Crop N border pixels")
    parser.add_argument("--save", help="Save difference heatmap")
    parser.add_argument("--seq-upscaled", help="Upscaled frame sequence directory")
    parser.add_argument("--seq-ref", help="Reference frame sequence directory")
    parser.add_argument("--csv", help="Save per-frame metrics as CSV")
    args = parser.parse_args()

    # Sequence mode
    if args.seq_upscaled:
        if not args.seq_ref:
            print("ERROR: --seq-ref required with --seq-upscaled", file=sys.stderr)
            sys.exit(1)
        run_sequence_comparison(args.seq_upscaled, args.seq_ref, args.crop, args.csv)
        return

    if not args.images:
        parser.print_help()
        sys.exit(1)

    # Resolve image list (expand globs on Windows)
    upscaled_paths = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        if expanded:
            upscaled_paths.extend(sorted(expanded))
        else:
            upscaled_paths.append(pattern)

    # If exactly 2 positional args and no --ref, treat second as reference
    if len(upscaled_paths) == 2 and args.ref is None:
        ref_path = upscaled_paths[1]
        upscaled_paths = [upscaled_paths[0]]
    elif args.ref:
        ref_path = args.ref
    else:
        print("ERROR: Need --ref <reference> when comparing multiple images", file=sys.stderr)
        sys.exit(1)

    # Load reference
    print(f"Reference: {ref_path}")
    ref_img = load_bmp(ref_path)
    print(f"  Size: {ref_img.shape[1]}x{ref_img.shape[0]}")

    if args.crop > 0:
        ref_img = crop_border(ref_img, args.crop)
        print(f"  Cropped to: {ref_img.shape[1]}x{ref_img.shape[0]}")

    print()

    # Compare each upscaled image
    all_results = []
    for i, up_path in enumerate(upscaled_paths):
        name = os.path.basename(up_path)
        print(f"--- {name} ---")

        up_img = load_bmp(up_path)
        if up_img.shape != ref_img.shape and args.crop == 0:
            print(f"  Size: {up_img.shape[1]}x{up_img.shape[0]} (ref: {ref_img.shape[1]}x{ref_img.shape[0]})")
            print(f"  ERROR: Dimension mismatch. Use --crop or ensure same resolution.")
            print()
            continue

        if args.crop > 0:
            up_img = crop_border(up_img, args.crop)

        metrics = compute_metrics(up_img, ref_img)
        print_metrics(name, metrics)
        g = grade(metrics["psnr"], metrics["ssim"])
        print(f"  Grade: {g}")

        # Save heatmap for single image or if explicitly requested
        if args.save and len(upscaled_paths) == 1:
            save_heatmap(metrics["diff"], args.save)
        elif args.save and len(upscaled_paths) > 1:
            base, ext = os.path.splitext(args.save)
            save_heatmap(metrics["diff"], f"{base}_{i:03d}{ext}")

        all_results.append((name, metrics))
        print()

    # Summary table for multi-image comparison
    if len(all_results) > 1:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  {'Image':<25s} {'PSNR':>8s} {'SSIM':>8s} {'MAE':>8s} {'Grade'}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for name, m in all_results:
            g = grade(m["psnr"], m["ssim"])
            print(f"  {name:<25s} {m['psnr']:>7.2f}  {m['ssim']:>7.4f}  {m['mae']:>7.5f}  {g}")


if __name__ == "__main__":
    main()
