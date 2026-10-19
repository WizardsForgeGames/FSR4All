"""
Analyze fsr4_debug.log for motion vector patterns, anomalies, and regimes.

Parses all RAW MV diagnostic blocks and performs:
1. Regime detection (zero, low, medium, high motion)
2. Transition analysis (zero↔nonzero snapping patterns)
3. Impossible value detection (NaN, Inf, sign inconsistencies)
4. Temporal coherence (frame-to-frame acceleration/deceleration)
5. Spatial coherence (cross-probe consistency within each frame)
6. Statistical summary with histograms
"""

import re
import sys
import math
from collections import defaultdict

# Usage: python analyze_mv_log.py <path/to/fsr4_debug.log>
# (fsr4_debug.log is written next to the debug DLL in the game's binary directory)
if len(sys.argv) < 2:
    sys.exit("Usage: python analyze_mv_log.py <path/to/fsr4_debug.log>")
LOG_PATH = sys.argv[1]

# ── Parse log ──────────────────────────────────────────────────────────

PROBE_NAMES = ["left-mid", "quarter", "center", "3quarter", "right-mid"]

# Match lines like:
#     left-mid   (   0, 540): raw=(-0.171387, -0.088135)  scaled=(-658.1, -190.4)  ...
PROBE_RE = re.compile(
    r'^\s+(left-mid|quarter|center|3quarter|right-mid)\s+'
    r'\(\s*(\d+),\s*(\d+)\):\s*'
    r'raw=\(([^,]+),\s*([^)]+)\)\s+'
    r'scaled=\(([^,]+),\s*([^)]+)\)'
)

# Match dispatch header to get frame context
DISPATCH_RE = re.compile(
    r'^Dispatch:\s+([\d.]+)ms\s+render=(\d+)x(\d+)\s+'
    r'jitter=\(([^,]+),([^)]+)\)\s+'
    r'mvScale=\(([^,]+),([^)]+)\)'
)

RAW_MV_HEADER_RE = re.compile(r'^\s+RAW MV \(pre-scale')

def parse_log(path):
    """Parse all frames with MV probe data from the debug log."""
    frames = []
    current_frame = None
    current_dispatch = None

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            # Track dispatch headers
            m = DISPATCH_RE.match(line)
            if m:
                current_dispatch = {
                    'line': line_no,
                    'dispatch_ms': float(m.group(1)),
                    'render_w': int(m.group(2)),
                    'render_h': int(m.group(3)),
                    'jitter_x': float(m.group(4)),
                    'jitter_y': float(m.group(5)),
                    'mv_scale_x': float(m.group(6)),
                    'mv_scale_y': float(m.group(7)),
                }
                continue

            # Start of RAW MV block
            if RAW_MV_HEADER_RE.match(line):
                current_frame = {
                    'line': line_no,
                    'dispatch': current_dispatch,
                    'probes': {},
                }
                continue

            # Probe line
            m = PROBE_RE.match(line)
            if m and current_frame is not None:
                name = m.group(1)
                px, py = int(m.group(2)), int(m.group(3))
                raw_x, raw_y = float(m.group(4)), float(m.group(5))
                scaled_x, scaled_y = float(m.group(6)), float(m.group(7))

                current_frame['probes'][name] = {
                    'px': px, 'py': py,
                    'raw_x': raw_x, 'raw_y': raw_y,
                    'scaled_x': scaled_x, 'scaled_y': scaled_y,
                    'raw_mag': math.sqrt(raw_x**2 + raw_y**2),
                    'scaled_mag': math.sqrt(scaled_x**2 + scaled_y**2),
                }

                # If we have all 5 probes, frame is complete
                if len(current_frame['probes']) == 5:
                    frames.append(current_frame)
                    current_frame = None

    return frames


def classify_frame(frame, zero_thresh=1e-6, low_thresh=0.005, high_thresh=0.05):
    """Classify a frame's motion regime based on center probe magnitude."""
    center = frame['probes'].get('center')
    if center is None:
        return 'unknown'
    mag = center['raw_mag']
    if mag < zero_thresh:
        return 'zero'
    elif mag < low_thresh:
        return 'low'
    elif mag < high_thresh:
        return 'medium'
    else:
        return 'high'


def all_probes_zero(frame, thresh=1e-6):
    """Check if ALL probes are zero (not just center)."""
    return all(p['raw_mag'] < thresh for p in frame['probes'].values())


def any_probe_zero(frame, thresh=1e-6):
    """Check if ANY probe is zero while others aren't."""
    mags = [p['raw_mag'] for p in frame['probes'].values()]
    has_zero = any(m < thresh for m in mags)
    has_nonzero = any(m >= thresh for m in mags)
    return has_zero and has_nonzero


# ── Analysis ───────────────────────────────────────────────────────────

def main():
    print(f"Parsing {LOG_PATH}...")
    frames = parse_log(LOG_PATH)
    print(f"Found {len(frames)} frames with MV probe data\n")

    if not frames:
        print("No frames found!")
        return

    # ── 1. Basic regime breakdown ──
    print("=" * 80)
    print("1. MOTION REGIME BREAKDOWN")
    print("=" * 80)

    regimes = defaultdict(list)
    for i, f in enumerate(frames):
        regime = classify_frame(f)
        regimes[regime].append(i)

    for regime in ['zero', 'low', 'medium', 'high', 'unknown']:
        count = len(regimes[regime])
        pct = 100.0 * count / len(frames)
        print(f"  {regime:>8s}: {count:6d} frames ({pct:5.1f}%)")

    # Also check "all probes zero" vs "some probes zero"
    all_zero_count = sum(1 for f in frames if all_probes_zero(f))
    mixed_zero_count = sum(1 for f in frames if any_probe_zero(f))
    print(f"\n  All 5 probes exactly zero: {all_zero_count} ({100*all_zero_count/len(frames):.1f}%)")
    print(f"  Mixed (some zero, some not): {mixed_zero_count} ({100*mixed_zero_count/len(frames):.1f}%)")

    # ── 2. Regime runs (consecutive frames in same regime) ──
    print("\n" + "=" * 80)
    print("2. REGIME RUNS (consecutive frames in same regime)")
    print("=" * 80)

    runs = []
    current_regime = classify_frame(frames[0])
    run_start = 0
    for i in range(1, len(frames)):
        regime = classify_frame(frames[i])
        if regime != current_regime:
            runs.append((current_regime, run_start, i - 1, i - run_start))
            current_regime = regime
            run_start = i
    runs.append((current_regime, run_start, len(frames) - 1, len(frames) - run_start))

    # Summarize run lengths by regime
    run_lengths = defaultdict(list)
    for regime, start, end, length in runs:
        run_lengths[regime].append(length)

    for regime in ['zero', 'low', 'medium', 'high']:
        lengths = run_lengths.get(regime, [])
        if lengths:
            avg = sum(lengths) / len(lengths)
            print(f"  {regime:>8s}: {len(lengths):4d} runs, "
                  f"len avg={avg:.1f} min={min(lengths)} max={max(lengths)} "
                  f"median={sorted(lengths)[len(lengths)//2]}")

    # Show the last 50 runs (gameplay portion)
    print(f"\n  Last 40 regime transitions (gameplay):")
    for regime, start, end, length in runs[-40:]:
        line = frames[start]['line']
        print(f"    frames {start:5d}-{end:5d} ({length:3d}): {regime:>8s}  (log line ~{line})")

    # ── 3. Transition analysis: zero→nonzero and nonzero→zero ──
    print("\n" + "=" * 80)
    print("3. ZERO<->NONZERO TRANSITIONS")
    print("=" * 80)

    transitions = []
    for i in range(1, len(frames)):
        prev_zero = all_probes_zero(frames[i-1])
        curr_zero = all_probes_zero(frames[i])
        if prev_zero != curr_zero:
            # Get center magnitude of the non-zero frame
            if curr_zero:
                # Was nonzero, now zero
                prev_center = frames[i-1]['probes']['center']
                transitions.append({
                    'frame': i,
                    'line': frames[i]['line'],
                    'type': 'nonzero→zero',
                    'last_nonzero_mag': prev_center['raw_mag'],
                    'last_nonzero_raw': (prev_center['raw_x'], prev_center['raw_y']),
                })
            else:
                # Was zero, now nonzero
                curr_center = frames[i]['probes']['center']
                transitions.append({
                    'frame': i,
                    'line': frames[i]['line'],
                    'type': 'zero→nonzero',
                    'first_nonzero_mag': curr_center['raw_mag'],
                    'first_nonzero_raw': (curr_center['raw_x'], curr_center['raw_y']),
                })

    print(f"  Total transitions: {len(transitions)}")
    nz_to_z = [t for t in transitions if t['type'] == 'nonzero→zero']
    z_to_nz = [t for t in transitions if t['type'] == 'zero→nonzero']
    print(f"    nonzero→zero: {len(nz_to_z)}")
    print(f"    zero→nonzero: {len(z_to_nz)}")

    # Check if MVs gradually approach zero before snapping, or snap abruptly
    print(f"\n  Last MV magnitude before snapping to zero (center probe):")
    if nz_to_z:
        mags = [t['last_nonzero_mag'] for t in nz_to_z]
        print(f"    min={min(mags):.6f}  max={max(mags):.6f}  mean={sum(mags)/len(mags):.6f}")
        print(f"    Values (last 20):")
        for t in nz_to_z[-20:]:
            rx, ry = t['last_nonzero_raw']
            print(f"      frame {t['frame']:5d} (line {t['line']:5d}): "
                  f"center raw=({rx:.6f}, {ry:.6f}) mag={t['last_nonzero_mag']:.6f}")

    print(f"\n  First MV magnitude after leaving zero (center probe):")
    if z_to_nz:
        mags = [t['first_nonzero_mag'] for t in z_to_nz]
        print(f"    min={min(mags):.6f}  max={max(mags):.6f}  mean={sum(mags)/len(mags):.6f}")
        print(f"    Values (last 20):")
        for t in z_to_nz[-20:]:
            rx, ry = t['first_nonzero_raw']
            print(f"      frame {t['frame']:5d} (line {t['line']:5d}): "
                  f"center raw=({rx:.6f}, {ry:.6f}) mag={t['first_nonzero_mag']:.6f}")

    # ── 4. Impossible / suspicious values ──
    print("\n" + "=" * 80)
    print("4. IMPOSSIBLE / SUSPICIOUS VALUES")
    print("=" * 80)

    impossible = []
    for i, f in enumerate(frames):
        for pname, p in f['probes'].items():
            # NaN / Inf
            if math.isnan(p['raw_x']) or math.isnan(p['raw_y']):
                impossible.append((i, f['line'], pname, 'NaN', p))
            if math.isinf(p['raw_x']) or math.isinf(p['raw_y']):
                impossible.append((i, f['line'], pname, 'Inf', p))
            # Enormous values (> 1.0 in UV space = more than full screen per frame)
            if abs(p['raw_x']) > 1.0 or abs(p['raw_y']) > 1.0:
                impossible.append((i, f['line'], pname, 'magnitude>1', p))
            # Negative zero check (IEEE -0.0)
            if p['raw_x'] == 0.0 and math.copysign(1, p['raw_x']) < 0:
                if p['raw_y'] == 0.0:
                    pass  # negative zero X with zero Y, note it

    if impossible:
        print(f"  Found {len(impossible)} impossible values:")
        shown = 0
        for frame_idx, line, pname, reason, p in impossible:
            if shown < 50:
                print(f"    frame {frame_idx:5d} (line {line:5d}) {pname:>10s}: "
                      f"raw=({p['raw_x']:.6f}, {p['raw_y']:.6f}) [{reason}]")
                shown += 1
        if len(impossible) > 50:
            print(f"    ... and {len(impossible) - 50} more")
    else:
        print("  No NaN, Inf, or >1.0 UV values found.")

    # ── 5. Spatial coherence check ──
    print("\n" + "=" * 80)
    print("5. SPATIAL COHERENCE (within-frame probe consistency)")
    print("=" * 80)

    # For pure camera rotation, all probes should move in roughly the same
    # direction with similar magnitude (within ~4x due to perspective).
    # Flag frames where one probe disagrees with others.
    spatially_incoherent = []
    for i, f in enumerate(frames):
        if all_probes_zero(f):
            continue
        mags = {name: p['raw_mag'] for name, p in f['probes'].items()}
        nonzero_mags = {n: m for n, m in mags.items() if m > 1e-6}
        if len(nonzero_mags) < 2:
            continue

        vals = list(nonzero_mags.values())
        max_m = max(vals)
        min_m = min(vals)

        # Check magnitude ratio
        if min_m > 0 and max_m / min_m > 10:
            spatially_incoherent.append((i, f['line'], mags, max_m / min_m))

        # Check sign consistency (X direction should be consistent across probes)
        signs_x = {name: math.copysign(1, f['probes'][name]['raw_x'])
                    for name in nonzero_mags if abs(f['probes'][name]['raw_x']) > 1e-6}
        if signs_x:
            unique_signs = set(signs_x.values())
            if len(unique_signs) > 1:
                spatially_incoherent.append((i, f['line'], mags, 'sign_flip_x'))

    if spatially_incoherent:
        print(f"  Found {len(spatially_incoherent)} spatially incoherent frames:")
        for frame_idx, line, mags, reason in spatially_incoherent[:30]:
            mag_str = "  ".join(f"{n}={m:.6f}" for n, m in mags.items())
            print(f"    frame {frame_idx:5d} (line {line:5d}): {mag_str}  [{reason}]")
        if len(spatially_incoherent) > 30:
            print(f"    ... and {len(spatially_incoherent) - 30} more")
    else:
        print("  All non-zero frames have spatially coherent probes.")

    # ── 6. Temporal coherence (frame-to-frame) ──
    print("\n" + "=" * 80)
    print("6. TEMPORAL COHERENCE (frame-to-frame center probe)")
    print("=" * 80)

    # Extract center raw_x, raw_y for all frames
    center_series = []
    for f in frames:
        c = f['probes']['center']
        center_series.append((c['raw_x'], c['raw_y'], c['raw_mag']))

    # Compute frame-to-frame acceleration (change in MV)
    accelerations = []
    big_jumps = []
    for i in range(1, len(center_series)):
        dx = center_series[i][0] - center_series[i-1][0]
        dy = center_series[i][1] - center_series[i-1][1]
        accel = math.sqrt(dx*dx + dy*dy)
        accelerations.append(accel)

        # Flag huge jumps (MV changed by more than 0.1 UV in one frame)
        if accel > 0.05:
            big_jumps.append((i, frames[i]['line'],
                              center_series[i-1], center_series[i], accel))

    if accelerations:
        nonzero_accel = [a for a in accelerations if a > 1e-8]
        print(f"  Frame-to-frame MV change (center):")
        print(f"    Total frames: {len(accelerations)}")
        print(f"    Zero change: {len(accelerations) - len(nonzero_accel)}")
        if nonzero_accel:
            print(f"    Non-zero changes: {len(nonzero_accel)}")
            print(f"    Mean change: {sum(nonzero_accel)/len(nonzero_accel):.6f}")
            print(f"    Max change: {max(nonzero_accel):.6f}")

    if big_jumps:
        print(f"\n  Large MV jumps (>0.05 UV frame-to-frame, center):")
        for frame_idx, line, prev, curr, accel in big_jumps[-20:]:
            print(f"    frame {frame_idx:5d} (line {line:5d}): "
                  f"({prev[0]:+.6f},{prev[1]:+.6f}) → ({curr[0]:+.6f},{curr[1]:+.6f})  "
                  f"Δ={accel:.6f}")

    # ── 7. Alternating zero/nonzero pattern ──
    print("\n" + "=" * 80)
    print("7. ZERO/NONZERO ALTERNATION PATTERN (gameplay)")
    print("=" * 80)

    # Focus on gameplay frames (last 30% of log or where we first see non-zero)
    first_nonzero = None
    for i, f in enumerate(frames):
        if not all_probes_zero(f):
            first_nonzero = i
            break

    if first_nonzero is not None:
        print(f"  First non-zero frame: #{first_nonzero} (log line {frames[first_nonzero]['line']})")
        gameplay_frames = frames[first_nonzero:]
        gameplay_zero = sum(1 for f in gameplay_frames if all_probes_zero(f))
        gameplay_nonzero = len(gameplay_frames) - gameplay_zero
        print(f"  Gameplay frames: {len(gameplay_frames)}")
        print(f"    Zero: {gameplay_zero} ({100*gameplay_zero/len(gameplay_frames):.1f}%)")
        print(f"    Non-zero: {gameplay_nonzero} ({100*gameplay_nonzero/len(gameplay_frames):.1f}%)")

        # Check for regular alternation pattern
        # Encode as binary string: 0=zero, 1=nonzero
        pattern = ''.join('1' if not all_probes_zero(f) else '0' for f in gameplay_frames)

        # Count patterns of length 2-10
        print(f"\n  Pattern analysis (gameplay portion):")
        print(f"    Binary sequence (last 200 chars): ...{pattern[-200:]}")

        # Find common subsequences
        for plen in [2, 3, 4, 5, 6]:
            subseqs = defaultdict(int)
            for j in range(len(pattern) - plen + 1):
                subseqs[pattern[j:j+plen]] += 1
            top = sorted(subseqs.items(), key=lambda x: -x[1])[:5]
            print(f"    Length-{plen} patterns: " +
                  "  ".join(f"'{p}'={c}" for p, c in top))

    # ── 8. Distribution of non-zero MV magnitudes ──
    print("\n" + "=" * 80)
    print("8. NON-ZERO MV MAGNITUDE DISTRIBUTION (center probe)")
    print("=" * 80)

    nonzero_mags = [c[2] for c in center_series if c[2] > 1e-6]
    if nonzero_mags:
        nonzero_mags.sort()
        print(f"  Count: {len(nonzero_mags)}")
        print(f"  Min:    {nonzero_mags[0]:.6f}")
        print(f"  P5:     {nonzero_mags[int(len(nonzero_mags)*0.05)]:.6f}")
        print(f"  P25:    {nonzero_mags[int(len(nonzero_mags)*0.25)]:.6f}")
        print(f"  Median: {nonzero_mags[len(nonzero_mags)//2]:.6f}")
        print(f"  P75:    {nonzero_mags[int(len(nonzero_mags)*0.75)]:.6f}")
        print(f"  P95:    {nonzero_mags[int(len(nonzero_mags)*0.95)]:.6f}")
        print(f"  Max:    {nonzero_mags[-1]:.6f}")

        # Histogram
        buckets = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        print(f"\n  Histogram:")
        prev = 0
        for b in buckets:
            count = sum(1 for m in nonzero_mags if prev <= m < b)
            bar = '#' * min(60, count)
            print(f"    [{prev:.4f}, {b:.4f}): {count:5d} {bar}")
            prev = b
        count = sum(1 for m in nonzero_mags if m >= buckets[-1])
        if count:
            print(f"    [{buckets[-1]:.4f}, ∞):     {count:5d}")

    # ── 9. MV direction consistency for non-zero frames ──
    print("\n" + "=" * 80)
    print("9. MV DIRECTION ANALYSIS (non-zero center probe)")
    print("=" * 80)

    nonzero_dirs = []
    for i, c in enumerate(center_series):
        if c[2] > 1e-6:
            angle = math.atan2(c[1], c[0]) * 180 / math.pi
            nonzero_dirs.append((i, angle, c[0], c[1]))

    if nonzero_dirs:
        angles = [d[1] for d in nonzero_dirs]
        print(f"  Direction angles (degrees from +X):")
        print(f"    Count: {len(angles)}")
        print(f"    Min: {min(angles):.1f}°  Max: {max(angles):.1f}°")

        # Direction sign analysis
        pos_x = sum(1 for d in nonzero_dirs if d[2] > 0)
        neg_x = sum(1 for d in nonzero_dirs if d[2] < 0)
        pos_y = sum(1 for d in nonzero_dirs if d[3] > 0)
        neg_y = sum(1 for d in nonzero_dirs if d[3] < 0)
        print(f"    MV_X: +{pos_x} / -{neg_x}")
        print(f"    MV_Y: +{pos_y} / -{neg_y}")

        # Check for direction reversals (consistent with circular motion)
        reversals = 0
        for i in range(1, len(nonzero_dirs)):
            prev_x = nonzero_dirs[i-1][2]
            curr_x = nonzero_dirs[i][2]
            if prev_x * curr_x < 0 and abs(prev_x) > 0.001 and abs(curr_x) > 0.001:
                reversals += 1
        print(f"    X-direction reversals: {reversals}")

    # ── 10. Jitter analysis ──
    print("\n" + "=" * 80)
    print("10. JITTER VALUES")
    print("=" * 80)

    jitters = [(f['dispatch']['jitter_x'], f['dispatch']['jitter_y'])
               for f in frames if f['dispatch']]
    if jitters:
        jx = [j[0] for j in jitters]
        jy = [j[1] for j in jitters]
        print(f"  Jitter X: min={min(jx):.4f}  max={max(jx):.4f}  unique={len(set(jx))}")
        print(f"  Jitter Y: min={min(jy):.4f}  max={max(jy):.4f}  unique={len(set(jy))}")

        # Check if jitter ever repeats exactly (should cycle through Halton)
        jitter_pairs = [(round(j[0], 6), round(j[1], 6)) for j in jitters]
        unique_jitters = len(set(jitter_pairs))
        print(f"  Unique jitter pairs: {unique_jitters} / {len(jitter_pairs)} total")

    # ── 11. Timing analysis ──
    print("\n" + "=" * 80)
    print("11. DISPATCH TIMING vs MV STATUS")
    print("=" * 80)

    zero_dispatch_times = []
    nonzero_dispatch_times = []
    for f in frames:
        if f['dispatch']:
            t = f['dispatch']['dispatch_ms']
            if all_probes_zero(f):
                zero_dispatch_times.append(t)
            else:
                nonzero_dispatch_times.append(t)

    if zero_dispatch_times:
        print(f"  Zero-MV frames: avg dispatch={sum(zero_dispatch_times)/len(zero_dispatch_times):.1f}ms "
              f"(n={len(zero_dispatch_times)})")
    if nonzero_dispatch_times:
        print(f"  Non-zero frames: avg dispatch={sum(nonzero_dispatch_times)/len(nonzero_dispatch_times):.1f}ms "
              f"(n={len(nonzero_dispatch_times)})")

    # ── 12. Per-probe MV comparison (are all probes zero together?) ──
    print("\n" + "=" * 80)
    print("12. PER-PROBE ZERO ANALYSIS")
    print("=" * 80)

    for pname in PROBE_NAMES:
        zero_count = sum(1 for f in frames
                        if f['probes'][pname]['raw_mag'] < 1e-6)
        print(f"  {pname:>10s}: {zero_count:5d} zero ({100*zero_count/len(frames):.1f}%) "
              f"/ {len(frames)-zero_count:5d} non-zero")

    # Check if probes ever disagree (one zero, another not)
    disagreements = 0
    disagree_examples = []
    for i, f in enumerate(frames):
        zeros = {n for n, p in f['probes'].items() if p['raw_mag'] < 1e-6}
        nonzeros = {n for n, p in f['probes'].items() if p['raw_mag'] >= 1e-6}
        if zeros and nonzeros:
            disagreements += 1
            if len(disagree_examples) < 10:
                disagree_examples.append((i, f['line'], zeros, nonzeros,
                    {n: f['probes'][n]['raw_mag'] for n in f['probes']}))

    print(f"\n  Frames with mixed zero/nonzero probes: {disagreements}")
    for frame_idx, line, zeros, nonzeros, mags in disagree_examples:
        mag_str = "  ".join(f"{n}={mags[n]:.6f}" for n in PROBE_NAMES)
        print(f"    frame {frame_idx:5d} (line {line:5d}): zero={zeros}  nonzero={nonzeros}")
        print(f"      {mag_str}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
