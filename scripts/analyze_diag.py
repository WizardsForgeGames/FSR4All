"""Analyze FSR4 diagnostic readback log for MV patterns and anomalies."""
import re
import sys
from collections import Counter

if len(sys.argv) < 2:
    sys.exit("Usage: python analyze_diag.py <path/to/fsr4_debug.log>")
log_path = sys.argv[1]

# Parse all DIAG lines
upscale_re = re.compile(
    r"DIAG upscale_center\((\d+),(\d+)\): "
    r"history=\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\) "
    r"reproj=\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\) "
    r"recurrent=\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)"
)

mv_re = re.compile(
    r"DIAG mv_(\w+): hex=\(0x([0-9A-Fa-f]+),0x([0-9A-Fa-f]+)\) "
    r"float=\(([-\d.]+),([-\d.]+)\)"
)

output_re = re.compile(
    r"DIAG output_center: \(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)"
)

input_color_re = re.compile(
    r"DIAG input_color_center: \(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)"
)

exposure_re = re.compile(
    r"DIAG exposure: r16=([-\d.]+) r32=([-\d.]+) raw16=0x([0-9A-Fa-f]+)"
)

dilated_mv_re = re.compile(
    r"DIAG dilated_mv_center: \(([-\d.]+),([-\d.]+)\)"
)

rcas_temp_re = re.compile(
    r"DIAG rcas_temp_center: \(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)"
)

frame_re = re.compile(r"Readback issued for frame (\d+)")

upscale_data = []  # list of dicts
mv_data = {"center": [], "quarter": [], "3quarter": [], "corner": []}
output_data = []   # list of [r,g,b,a]
input_color_data = []  # list of [r,g,b,a]
exposure_data = []  # list of {"r16": float, "r32": float, "raw16": int}
dilated_mv_data = []  # list of [x, y]
rcas_temp_data = []  # list of [r,g,b,a]
frame_numbers = []

current_upscale = None

with open(log_path, "r") as f:
    for line in f:
        m = upscale_re.search(line)
        if m:
            g = m.groups()
            current_upscale = {
                "cx": int(g[0]), "cy": int(g[1]),
                "hist": [float(g[2]), float(g[3]), float(g[4]), float(g[5])],
                "repr": [float(g[6]), float(g[7]), float(g[8]), float(g[9])],
                "rec":  [float(g[10]), float(g[11]), float(g[12]), float(g[13])],
            }
            upscale_data.append(current_upscale)
            continue

        m = mv_re.search(line)
        if m:
            pos = m.group(1)
            hx, hy = int(m.group(2), 16), int(m.group(3), 16)
            fx, fy = float(m.group(4)), float(m.group(5))
            if pos in mv_data:
                mv_data[pos].append({"hx": hx, "hy": hy, "fx": fx, "fy": fy})
            continue

        m = output_re.search(line)
        if m:
            output_data.append([float(m.group(i)) for i in range(1, 5)])
            continue

        m = input_color_re.search(line)
        if m:
            input_color_data.append([float(m.group(i)) for i in range(1, 5)])
            continue

        m = exposure_re.search(line)
        if m:
            exposure_data.append({"r16": float(m.group(1)), "r32": float(m.group(2)),
                                  "raw16": int(m.group(3), 16)})
            continue

        m = dilated_mv_re.search(line)
        if m:
            dilated_mv_data.append([float(m.group(1)), float(m.group(2))])
            continue

        m = rcas_temp_re.search(line)
        if m:
            rcas_temp_data.append([float(m.group(i)) for i in range(1, 5)])
            continue

        m = frame_re.search(line)
        if m:
            frame_numbers.append(int(m.group(1)))

print(f"Log: {log_path}")
print(f"Total diagnostic frames: {len(upscale_data)}")
print(f"Frame range: {frame_numbers[0]}..{frame_numbers[-1]}" if frame_numbers else "no frame numbers")
print()

# === MV Analysis ===
print("=" * 70)
print("MOTION VECTOR ANALYSIS")
print("=" * 70)

for pos in ["center", "quarter", "3quarter", "corner"]:
    samples = mv_data[pos]
    if not samples:
        print(f"\n  {pos}: no data")
        continue

    n = len(samples)
    # Classify: true zero (0x0000), neg zero (0x8000), or nonzero
    true_zero = sum(1 for s in samples if s["hx"] == 0 and s["hy"] == 0)
    neg_zero_only = sum(1 for s in samples
                        if (s["hx"] in (0, 0x8000)) and (s["hy"] in (0, 0x8000))
                        and not (s["hx"] == 0 and s["hy"] == 0))
    nonzero = n - true_zero - neg_zero_only

    # Stats on nonzero values
    nz = [s for s in samples if not ((s["hx"] in (0, 0x8000)) and (s["hy"] in (0, 0x8000)))]

    print(f"\n  {pos} ({n} samples):")
    print(f"    True zero (0x0000,0x0000): {true_zero} ({100*true_zero/n:.1f}%)")
    print(f"    Neg-zero only (±0):        {neg_zero_only} ({100*neg_zero_only/n:.1f}%)")
    print(f"    Nonzero (actual MV data):  {nonzero} ({100*nonzero/n:.1f}%)")

    if nz:
        fx_vals = [s["fx"] for s in nz]
        fy_vals = [s["fy"] for s in nz]
        print(f"    X range: [{min(fx_vals):.6f}, {max(fx_vals):.6f}]  mean={sum(fx_vals)/len(fx_vals):.6f}")
        print(f"    Y range: [{min(fy_vals):.6f}, {max(fy_vals):.6f}]  mean={sum(fy_vals)/len(fy_vals):.6f}")

        # Magnitude
        mags = [(s["fx"]**2 + s["fy"]**2)**0.5 for s in nz]
        print(f"    Magnitude: min={min(mags):.6f} max={max(mags):.6f} mean={sum(mags)/len(mags):.6f}")

        # Convert to pixel space (UV * render_res)
        px_mags = [m * 1920 for m in mags]  # approximate using render width
        print(f"    Pixel-space mag (×1920): min={min(px_mags):.1f} max={max(px_mags):.1f} mean={sum(px_mags)/len(px_mags):.1f}")

        # Check for suspicious values
        huge = [s for s in nz if abs(s["fx"]) > 0.5 or abs(s["fy"]) > 0.5]
        if huge:
            print(f"    WARNING: {len(huge)} samples with |MV| > 0.5 (half-screen+ motion)")

        tiny = [s for s in nz if 0 < (s["fx"]**2 + s["fy"]**2)**0.5 < 1e-5]
        if tiny:
            print(f"    WARNING: {len(tiny)} samples with near-zero but nonzero (subnormal?)")

        # Hex value distribution - check for patterns
        hx_unique = len(set(s["hx"] for s in nz))
        hy_unique = len(set(s["hy"] for s in nz))
        print(f"    Unique hex X values: {hx_unique}  Y values: {hy_unique}")

# === Temporal Analysis: where are the zero runs? ===
print("\n" + "=" * 70)
print("TEMPORAL PATTERN (center MV)")
print("=" * 70)

center = mv_data["center"]
if center:
    # Find runs of zeros and nonzeros
    runs = []
    is_zero = lambda s: (s["hx"] in (0, 0x8000)) and (s["hy"] in (0, 0x8000))

    current_type = is_zero(center[0])
    run_start = 0
    for i in range(1, len(center)):
        z = is_zero(center[i])
        if z != current_type:
            runs.append(("ZERO" if current_type else "DATA", run_start, i - 1, i - run_start))
            current_type = z
            run_start = i
    runs.append(("ZERO" if current_type else "DATA", run_start, len(center) - 1, len(center) - run_start))

    print(f"\n  Run analysis ({len(runs)} runs):")
    for rtype, start, end, length in runs:
        if length >= 3 or rtype == "ZERO":
            frame_s = frame_numbers[start] if start < len(frame_numbers) else "?"
            frame_e = frame_numbers[end] if end < len(frame_numbers) else "?"
            print(f"    {rtype:4s}: frames {frame_s}-{frame_e} ({length} frames)")

# === History vs Reprojected comparison ===
print("\n" + "=" * 70)
print("HISTORY vs REPROJECTED COMPARISON")
print("=" * 70)

if upscale_data:
    diffs = []
    for d in upscale_data:
        diff = sum(abs(h - r) for h, r in zip(d["hist"], d["repr"]))
        diffs.append(diff)

    print(f"\n  Total frames: {len(diffs)}")
    print(f"  Sum-of-abs-diff (hist vs repr):")
    print(f"    min={min(diffs):.6f}  max={max(diffs):.6f}  mean={sum(diffs)/len(diffs):.6f}")

    identical = sum(1 for d in diffs if d < 1e-6)
    close = sum(1 for d in diffs if d < 0.01)
    different = sum(1 for d in diffs if d >= 0.01)
    print(f"    Identical (<1e-6): {identical} ({100*identical/len(diffs):.1f}%)")
    print(f"    Close (<0.01):     {close} ({100*close/len(diffs):.1f}%)")
    print(f"    Different (>=0.01): {different} ({100*different/len(diffs):.1f}%)")

    # History value ranges
    hist_r = [d["hist"][0] for d in upscale_data]
    hist_g = [d["hist"][1] for d in upscale_data]
    hist_b = [d["hist"][2] for d in upscale_data]
    hist_a = [d["hist"][3] for d in upscale_data]
    print(f"\n  History value ranges:")
    print(f"    R: [{min(hist_r):.4f}, {max(hist_r):.4f}]")
    print(f"    G: [{min(hist_g):.4f}, {max(hist_g):.4f}]")
    print(f"    B: [{min(hist_b):.4f}, {max(hist_b):.4f}]")
    print(f"    A: [{min(hist_a):.4f}, {max(hist_a):.4f}]")

# === Recurrent state analysis ===
print("\n" + "=" * 70)
print("RECURRENT STATE ANALYSIS")
print("=" * 70)

if upscale_data:
    for ch, name in enumerate(["R", "G", "B", "A"]):
        vals = [d["rec"][ch] for d in upscale_data]
        print(f"  {name}: [{min(vals):.3f}, {max(vals):.3f}] mean={sum(vals)/len(vals):.3f}")

    # Check if recurrent is stuck (same value every frame)
    rec_changes = 0
    for i in range(1, len(upscale_data)):
        if upscale_data[i]["rec"] != upscale_data[i-1]["rec"]:
            rec_changes += 1
    print(f"  Frame-to-frame changes: {rec_changes}/{len(upscale_data)-1}")

# === Exposure Analysis ===
print("\n" + "=" * 70)
print("EXPOSURE TEXTURE ANALYSIS")
print("=" * 70)

if exposure_data:
    r16_vals = [e["r16"] for e in exposure_data]
    r32_vals = [e["r32"] for e in exposure_data]
    print(f"\n  Exposure frames: {len(exposure_data)}")
    print(f"  As R16_FLOAT: [{min(r16_vals):.6f}, {max(r16_vals):.6f}] mean={sum(r16_vals)/len(r16_vals):.6f}")
    print(f"  As R32_FLOAT: [{min(r32_vals):.6f}, {max(r32_vals):.6f}] mean={sum(r32_vals)/len(r32_vals):.6f}")
    # Show first few raw values
    print(f"  First 5 raw16: {[hex(e['raw16']) for e in exposure_data[:5]]}")
    if r16_vals[0] > 100:
        print(f"  >>> EXPOSURE IS HUGE ({r16_vals[0]:.1f}) - this amplifies all colors!")
    elif r16_vals[0] < 0.01:
        print(f"  >>> EXPOSURE IS TINY ({r16_vals[0]:.6f}) - colors will be very dark")
else:
    print("\n  No exposure data (rebuild DLL)")

# === Dilated MV Analysis ===
print("\n" + "=" * 70)
print("DILATED MV (Pre shader output at upscale center)")
print("=" * 70)

if dilated_mv_data:
    dx = [d[0] for d in dilated_mv_data]
    dy = [d[1] for d in dilated_mv_data]
    nz_dmv = [d for d in dilated_mv_data if abs(d[0]) > 1e-6 or abs(d[1]) > 1e-6]
    print(f"\n  Dilated MV frames: {len(dilated_mv_data)}")
    print(f"  Nonzero: {len(nz_dmv)}/{len(dilated_mv_data)}")
    if nz_dmv:
        nz_dx = [d[0] for d in nz_dmv]
        nz_dy = [d[1] for d in nz_dmv]
        mags = [(d[0]**2+d[1]**2)**0.5 for d in nz_dmv]
        print(f"  X: [{min(nz_dx):.8f}, {max(nz_dx):.8f}]")
        print(f"  Y: [{min(nz_dy):.8f}, {max(nz_dy):.8f}]")
        print(f"  Magnitude: [{min(mags):.8f}, {max(mags):.8f}] mean={sum(mags)/len(mags):.8f}")
        # Compare to raw game MVs
        if mv_data["center"]:
            raw_nz = [s for s in mv_data["center"] if not ((s["hx"] in (0,0x8000)) and (s["hy"] in (0,0x8000)))]
            if raw_nz:
                raw_mags = [(s["fx"]**2+s["fy"]**2)**0.5 for s in raw_nz]
                print(f"  Raw MV mean mag: {sum(raw_mags)/len(raw_mags):.8f}")
                print(f"  Dilated/Raw ratio: {sum(mags)/len(mags) / (sum(raw_mags)/len(raw_mags)):.4f}")
else:
    print("\n  No dilated MV data (rebuild DLL)")

# === RCAS Temp vs Output ===
print("\n" + "=" * 70)
print("RCAS ANALYSIS (pre-sharpening vs final output)")
print("=" * 70)

if rcas_temp_data:
    print(f"\n  RCAS temp frames: {len(rcas_temp_data)}")
    for ch, name in enumerate(["R", "G", "B", "A"]):
        vals = [d[ch] for d in rcas_temp_data]
        print(f"  RCAS temp {name}: [{min(vals):.4f}, {max(vals):.4f}] mean={sum(vals)/len(vals):.4f}")
    if output_data and len(rcas_temp_data) == len(output_data):
        rcas_out_diffs = []
        for r, o in zip(rcas_temp_data, output_data):
            diff = sum(abs(rv - ov) for rv, ov in zip(r, o))
            rcas_out_diffs.append(diff)
        identical = sum(1 for d in rcas_out_diffs if d < 1e-6)
        print(f"  RCAS temp vs Output: identical={identical}/{len(rcas_out_diffs)} "
              f"mean_diff={sum(rcas_out_diffs)/len(rcas_out_diffs):.6f}")
        if identical == len(rcas_out_diffs):
            print("  >>> RCAS is NOT modifying output (disabled or passthrough)")
else:
    print("\n  No RCAS temp data (rebuild DLL)")

# === Output vs Input Color ===
print("\n" + "=" * 70)
print("OUTPUT vs INPUT COLOR (end-to-end pipeline check)")
print("=" * 70)

if output_data:
    print(f"\n  Output frames: {len(output_data)}")
    for ch, name in enumerate(["R", "G", "B", "A"]):
        vals = [d[ch] for d in output_data]
        print(f"  Output {name}: [{min(vals):.4f}, {max(vals):.4f}] mean={sum(vals)/len(vals):.4f}")
else:
    print("\n  No output pixel data (rebuild DLL with output readback)")

if input_color_data:
    print(f"\n  Input color frames: {len(input_color_data)}")
    for ch, name in enumerate(["R", "G", "B", "A"]):
        vals = [d[ch] for d in input_color_data]
        print(f"  Input  {name}: [{min(vals):.4f}, {max(vals):.4f}] mean={sum(vals)/len(vals):.4f}")
else:
    print("\n  No input color data (rebuild DLL with color readback)")

if output_data and input_color_data and len(output_data) == len(input_color_data):
    # Compare output to input - are they similar? (ratio check)
    ratios_r = []
    for o, i in zip(output_data, input_color_data):
        if abs(i[0]) > 1e-6:
            ratios_r.append(o[0] / i[0])
    if ratios_r:
        print(f"\n  Output/Input ratio (R channel): min={min(ratios_r):.4f} max={max(ratios_r):.4f} mean={sum(ratios_r)/len(ratios_r):.4f}")

    # Compare output to history - if output == history, temporal isn't blending with reconstruction
    if upscale_data and len(output_data) == len(upscale_data):
        out_hist_diffs = []
        for o, d in zip(output_data, upscale_data):
            diff = sum(abs(ov - hv) for ov, hv in zip(o, d["hist"]))
            out_hist_diffs.append(diff)
        print(f"  Output vs History diff: min={min(out_hist_diffs):.6f} max={max(out_hist_diffs):.6f} mean={sum(out_hist_diffs)/len(out_hist_diffs):.6f}")

    # Frame-to-frame output stability (should change with temporal accumulation)
    if len(output_data) > 1:
        frame_diffs = []
        for i in range(1, len(output_data)):
            diff = sum(abs(output_data[i][ch] - output_data[i-1][ch]) for ch in range(4))
            frame_diffs.append(diff)
        stable = sum(1 for d in frame_diffs if d < 1e-6)
        print(f"  Frame-to-frame output changes: {len(frame_diffs) - stable}/{len(frame_diffs)} "
              f"(stable={stable})")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
nz_center = sum(1 for s in mv_data["center"]
                if not ((s["hx"] in (0, 0x8000)) and (s["hy"] in (0, 0x8000))))
total = len(mv_data["center"])
print(f"  MV data present: {nz_center}/{total} frames ({100*nz_center/total:.1f}%)" if total else "  No MV data")
if upscale_data:
    avg_diff = sum(abs(d["hist"][0] - d["repr"][0]) for d in upscale_data) / len(upscale_data)
    print(f"  Avg history-reproj diff (R channel): {avg_diff:.6f}")
    if avg_diff < 0.001:
        print("  >>> History ≈ Reprojected — reprojection is sampling from ~same position")
        print("  >>> This means MVs are either zero OR very small relative to pixel grid")
