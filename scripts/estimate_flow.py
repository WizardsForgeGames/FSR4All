#!/usr/bin/env python3
"""Estimate optical flow (motion vectors) between frame pairs using RAFT.

Produces binary MV files compatible with test_inference --mv-dir.

Usage:
    python scripts/estimate_flow.py frames/ --output mvs/ --scale 2
    python scripts/estimate_flow.py frames/ --output mvs/ --scale 2 --gpu 0 --visualize
    python scripts/estimate_flow.py frames/ --output mvs/ --model raft-large --scale 2

Options:
    --output DIR     Output directory for MV binary files (default: mvs/)
    --model NAME     RAFT model: raft-small or raft-large (default: raft-small)
    --scale N        Upscale flow by N (frames at input res, MVs needed at output res)
    --gpu N          CUDA device index (default: 0)
    --visualize      Save flow visualization PNGs alongside MV binaries
    --max-frames N   Process at most N frame pairs

Requirements:
    pip install torch torchvision numpy Pillow
"""

import argparse
import os
import re
import sys
import numpy as np
from PIL import Image

# Defer torch imports for faster --help
torch = None
torchvision = None


def lazy_import_torch():
    """Import torch lazily (takes a few seconds)."""
    global torch, torchvision
    if torch is None:
        import torch as _torch
        import torchvision as _tv
        torch = _torch
        torchvision = _tv


def scan_frames(directory):
    """Scan directory for frame_NNNNNN.bmp files, return sorted paths."""
    pattern = re.compile(r"^frame_(\d+)\.(bmp|png)$")
    frames = []
    for f in os.listdir(directory):
        m = pattern.match(f)
        if m:
            frames.append((int(m.group(1)), os.path.join(directory, f)))
    frames.sort(key=lambda x: x[0])
    return frames


def load_frame_tensor(path, device):
    """Load image as [1, 3, H, W] float32 tensor in [0, 255]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)  # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(device)


def flow_to_color(flow_uv, clip_flow=None):
    """Convert optical flow to HSV color visualization.

    flow_uv: [2, H, W] numpy array (dx, dy)
    Returns: [H, W, 3] uint8 RGB array
    """
    u = flow_uv[0]
    v = flow_uv[1]
    mag = np.sqrt(u * u + v * v)
    if clip_flow is not None:
        mag = np.clip(mag, 0, clip_flow)
    max_mag = np.max(mag) + 1e-5

    # HSV: angle → hue, magnitude → saturation/value
    angle = np.arctan2(-v, -u) / np.pi  # [-1, 1]
    hue = (angle + 1.0) / 2.0  # [0, 1]
    sat = np.clip(mag / max_mag, 0, 1)
    val = np.ones_like(hue)

    # HSV to RGB
    hsv = np.stack([hue * 179, sat * 255, val * 255], axis=-1).astype(np.uint8)
    from PIL import ImageDraw
    # Manual HSV→RGB conversion
    h = hue * 6.0
    c = sat
    x = c * (1.0 - np.abs(h % 2.0 - 1.0))

    rgb = np.zeros((*u.shape, 3), dtype=np.float32)
    mask = (h < 1)
    rgb[mask] = np.stack([c[mask], x[mask], np.zeros_like(c[mask])], axis=-1)
    mask = (h >= 1) & (h < 2)
    rgb[mask] = np.stack([x[mask], c[mask], np.zeros_like(c[mask])], axis=-1)
    mask = (h >= 2) & (h < 3)
    rgb[mask] = np.stack([np.zeros_like(c[mask]), c[mask], x[mask]], axis=-1)
    mask = (h >= 3) & (h < 4)
    rgb[mask] = np.stack([np.zeros_like(c[mask]), x[mask], c[mask]], axis=-1)
    mask = (h >= 4) & (h < 5)
    rgb[mask] = np.stack([x[mask], np.zeros_like(c[mask]), c[mask]], axis=-1)
    mask = (h >= 5)
    rgb[mask] = np.stack([c[mask], np.zeros_like(c[mask]), x[mask]], axis=-1)

    m = val[..., None] - c[..., None]
    rgb = ((rgb + m) * 255).clip(0, 255).astype(np.uint8)
    return rgb


def main():
    parser = argparse.ArgumentParser(description="Estimate optical flow for FSR4 MV input")
    parser.add_argument("frame_dir", help="Directory containing frame_NNNNNN.bmp/png")
    parser.add_argument("--output", default="mvs", help="Output directory (default: mvs/)")
    parser.add_argument("--model", default="raft-small", choices=["raft-small", "raft-large"],
                        help="RAFT model variant (default: raft-small)")
    parser.add_argument("--scale", type=int, default=1,
                        help="Upscale flow by N (e.g., 2 for 2x FSR4)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device (default: 0)")
    parser.add_argument("--visualize", action="store_true", help="Save flow visualization PNGs")
    parser.add_argument("--max-frames", type=int, help="Max frame pairs to process")
    args = parser.parse_args()

    if not os.path.isdir(args.frame_dir):
        print(f"ERROR: Directory not found: {args.frame_dir}", file=sys.stderr)
        sys.exit(1)

    # Import torch
    print("Loading PyTorch and RAFT model...")
    lazy_import_torch()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load RAFT model
    if args.model == "raft-small":
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
    else:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()

    print(f"Model: {args.model}")

    # Scan frames
    frames = scan_frames(args.frame_dir)
    if len(frames) < 2:
        print(f"ERROR: Need at least 2 frames, found {len(frames)}", file=sys.stderr)
        sys.exit(1)

    n_pairs = len(frames) - 1
    if args.max_frames:
        n_pairs = min(n_pairs, args.max_frames)
    print(f"Found {len(frames)} frames, processing {n_pairs} pairs")

    os.makedirs(args.output, exist_ok=True)

    # Process consecutive pairs
    # RAFT convention: flow from frame1 to frame2 maps pixels in frame2 to frame1
    # Our MV convention: current_pos + mv = previous_pos
    # So flow(frame_N, frame_N+1) gives us MVs for frame_N+1
    with torch.no_grad():
        for i in range(n_pairs):
            idx_curr = frames[i + 1][0]  # Current frame (needs MVs)
            path_prev = frames[i][1]
            path_curr = frames[i + 1][1]

            # Load frame pair
            img_prev = load_frame_tensor(path_prev, device)
            img_curr = load_frame_tensor(path_curr, device)

            # RAFT needs images in [0, 255] range and divisible by 8
            H, W = img_curr.shape[2], img_curr.shape[3]
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h > 0 or pad_w > 0:
                img_prev = torch.nn.functional.pad(img_prev, [0, pad_w, 0, pad_h])
                img_curr = torch.nn.functional.pad(img_curr, [0, pad_w, 0, pad_h])

            # Estimate flow: maps pixels in img_curr back to img_prev
            # Returns list of flow predictions (last is best)
            flow_predictions = model(img_curr, img_prev)
            flow = flow_predictions[-1]  # [1, 2, H+pad, W+pad]

            # Remove padding
            flow = flow[:, :, :H, :W]

            # Convert to numpy [2, H, W]
            flow_np = flow[0].cpu().numpy()  # [2, H, W]

            # Upscale if needed
            if args.scale > 1:
                s = args.scale
                # Resize flow field and scale values
                flow_tensor = torch.from_numpy(flow_np).unsqueeze(0)  # [1, 2, H, W]
                flow_up = torch.nn.functional.interpolate(
                    flow_tensor, scale_factor=s, mode="bilinear", align_corners=False
                )
                flow_np = flow_up[0].numpy() * s  # Scale flow values by upscale factor

            # Save as raw FP16 binary [2, H_out, W_out]
            flow_fp16 = flow_np.astype(np.float16)
            out_path = os.path.join(args.output, f"mv_{idx_curr:06d}.bin")
            flow_fp16.tofile(out_path)

            # Optionally save visualization
            if args.visualize:
                vis = flow_to_color(flow_np)
                vis_path = os.path.join(args.output, f"flow_{idx_curr:06d}.png")
                Image.fromarray(vis).save(vis_path)

            mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            print(f"  Pair {i+1}/{n_pairs}: frame {idx_curr}  "
                  f"flow mag: mean={np.mean(mag):.2f} max={np.max(mag):.1f}  "
                  f"→ {out_path}")

    print(f"\nDone. {n_pairs} MV files saved to {args.output}/")
    if args.scale > 1:
        out_H = flow_np.shape[1]
        out_W = flow_np.shape[2]
        print(f"  Output resolution: {out_W}x{out_H} ({args.scale}x upscaled)")
    print(f"  Format: raw FP16 [2, H, W], convention: current_pos + mv = previous_pos")


if __name__ == "__main__":
    main()
