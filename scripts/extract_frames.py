#!/usr/bin/env python3
"""Extract frames from video for FSR4 temporal testing.

Usage:
    python scripts/extract_frames.py video.mp4 --output frames/test/ --res 960x540 --count 60
    python scripts/extract_frames.py video.mp4 --output frames/test/ --start 10 --duration 2 --fps 30

Options:
    --output DIR     Output directory for frames (default: frames/)
    --res WxH        Target resolution (low-res input). Omit to keep original.
    --count N        Extract first N frames
    --start SEC      Start time in seconds
    --duration SEC   Duration in seconds
    --fps N          Override frame rate extraction
    --format FMT     Frame format: bmp or png (default: bmp)
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


def get_video_info(ffmpeg, path):
    """Get video resolution and frame rate using ffprobe/ffmpeg."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-of", "json",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            stream = info["streams"][0]
            w = int(stream["width"])
            h = int(stream["height"])
            # Parse frame rate (e.g., "30000/1001" or "30/1")
            fps_str = stream.get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den else 30.0
            return {"width": w, "height": h, "fps": fps}
    except Exception:
        pass
    return None


def extract_frames(ffmpeg, video_path, output_dir, args):
    """Extract frames using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [ffmpeg, "-i", video_path]

    # Start time
    if args.start:
        cmd = [ffmpeg, "-ss", str(args.start), "-i", video_path]

    # Duration
    if args.duration:
        cmd.extend(["-t", str(args.duration)])

    # Frame count
    if args.count:
        cmd.extend(["-vframes", str(args.count)])

    # Build video filter chain
    vf_parts = []
    if args.fps:
        vf_parts.append(f"fps={args.fps}")
    if args.res:
        vf_parts.append(f"scale={args.res.replace('x', ':')}")
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    # Output format
    ext = args.format
    if ext == "bmp":
        cmd.extend(["-pix_fmt", "bgr24"])

    cmd.append(os.path.join(output_dir, f"frame_%06d.{ext}"))

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}", file=sys.stderr)
        return False
    return True


def count_frames(output_dir, ext):
    """Count extracted frames."""
    pattern = re.compile(rf"^frame_(\d+)\.{ext}$")
    frames = []
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            frames.append(int(m.group(1)))
    return sorted(frames)


def main():
    parser = argparse.ArgumentParser(description="Extract video frames for FSR4 testing")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--output", default="frames", help="Output directory (default: frames/)")
    parser.add_argument("--res", help="Target resolution WxH (e.g., 960x540)")
    parser.add_argument("--count", type=int, help="Number of frames to extract")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--duration", type=float, help="Duration in seconds")
    parser.add_argument("--fps", type=float, help="Override frame rate")
    parser.add_argument("--format", choices=["bmp", "png"], default="bmp",
                        help="Frame format (default: bmp)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ERROR: ffmpeg not found. Install FFmpeg and add to PATH.", file=sys.stderr)
        sys.exit(1)

    # Get source video info
    info = get_video_info(ffmpeg, args.video)
    if info:
        print(f"Source: {args.video}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  Frame rate: {info['fps']:.2f} fps")
    else:
        print(f"Source: {args.video} (could not read metadata)")

    if args.res:
        print(f"  Target res: {args.res}")

    # Extract
    print()
    if not extract_frames(ffmpeg, args.video, args.output, args):
        sys.exit(1)

    # Count and report
    frames = count_frames(args.output, args.format)
    if not frames:
        print("ERROR: No frames extracted!", file=sys.stderr)
        sys.exit(1)

    # FFmpeg numbers frames starting at 1
    n_frames = len(frames)
    print(f"\nExtracted {n_frames} frames to {args.output}/")

    # Determine actual resolution from first frame
    first_frame = os.path.join(args.output, f"frame_{frames[0]:06d}.{args.format}")
    actual_res = args.res
    if not actual_res and info:
        actual_res = f"{info['width']}x{info['height']}"

    # Write manifest
    manifest = {
        "source": os.path.abspath(args.video),
        "frame_count": n_frames,
        "first_index": frames[0],
        "resolution": actual_res,
        "fps": args.fps or (info["fps"] if info else 30.0),
        "format": args.format,
        "pattern": f"frame_%06d.{args.format}",
    }
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
