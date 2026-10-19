#!/usr/bin/env python3
"""Assemble BMP/PNG frame sequence into video using FFmpeg.

Usage:
    python scripts/assemble_video.py output/video_test/ --fps 30 --output result.mp4
    python scripts/assemble_video.py output/video_test/ --fps 30 --crf 18

Options:
    --output FILE    Output video path (default: <input_dir>/output.mp4)
    --fps N          Frame rate (default: 30, or from manifest.json)
    --crf N          Quality (0-51, lower=better, default: 18)
    --format FMT     Frame format: bmp or png (default: auto-detect)
"""

import argparse
import json
import os
import re
import subprocess
import sys


def find_ffmpeg():
    """Find ffmpeg executable."""
    for name in ["ffmpeg", "ffmpeg.exe"]:
        try:
            result = subprocess.run(
                [name, "-version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def detect_format(frame_dir):
    """Detect frame format and numbering from directory contents."""
    for ext in ["bmp", "png"]:
        pattern = re.compile(rf"^frame_(\d+)\.{ext}$")
        frames = []
        for f in os.listdir(frame_dir):
            m = pattern.match(f)
            if m:
                frames.append(int(m.group(1)))
        if frames:
            frames.sort()
            return ext, frames
    return None, []


def main():
    parser = argparse.ArgumentParser(description="Assemble frames into video")
    parser.add_argument("frame_dir", help="Directory containing frame_NNNNNN.bmp/png files")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--fps", type=float, help="Frame rate (default: 30 or from manifest)")
    parser.add_argument("--crf", type=int, default=18, help="Quality 0-51 (default: 18)")
    parser.add_argument("--format", choices=["bmp", "png"], help="Frame format (auto-detect)")
    args = parser.parse_args()

    if not os.path.isdir(args.frame_dir):
        print(f"ERROR: Directory not found: {args.frame_dir}", file=sys.stderr)
        sys.exit(1)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ERROR: ffmpeg not found. Install FFmpeg and add to PATH.", file=sys.stderr)
        sys.exit(1)

    # Try to read manifest
    fps = args.fps
    manifest_path = os.path.join(args.frame_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if not fps:
            fps = manifest.get("fps", 30.0)
        print(f"Read manifest: {manifest.get('frame_count', '?')} frames, "
              f"{manifest.get('resolution', '?')}, {fps} fps")
    if not fps:
        fps = 30.0

    # Detect format
    fmt = args.format
    if not fmt:
        fmt, frames = detect_format(args.frame_dir)
        if not fmt:
            print("ERROR: No frame_*.bmp or frame_*.png files found!", file=sys.stderr)
            sys.exit(1)
    else:
        _, frames = detect_format(args.frame_dir)

    if not frames:
        print("ERROR: No frames found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(frames)} {fmt.upper()} frames (index {frames[0]}-{frames[-1]})")

    # Output path
    output = args.output or os.path.join(args.frame_dir, "output.mp4")

    # Build ffmpeg command
    input_pattern = os.path.join(args.frame_dir, f"frame_%06d.{fmt}")
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-start_number", str(frames[0]),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"\nOutput: {output} ({size_mb:.1f} MB)")
    print(f"  {len(frames)} frames @ {fps} fps = {len(frames)/fps:.1f}s")


if __name__ == "__main__":
    main()
