#!/usr/bin/env python3
"""
Extract FSR4 INT8 model weights from leaked HLSL shader source.

Two weight storage mechanisms:
  1. Small tensors: `static const uint embedded_<name>_dwords[N]` in HLSL
  2. Large tensors: raw bytes in `initializers.bin`, referenced via
     InitializerBuffer with byte offsets declared in HLSL

Per-tensor INT8 quantization scales are parsed from QuantizedTensor declarations.
Tensor shapes (NHWC/HWCN) are extracted from logicalSize fields.

Usage:
    python extract_fsr4_weights.py <hlsl_dir> [--output-dir <dir>]

Example:
    python extract_fsr4_weights.py "$FSR4_SOURCE/Kits/FidelityFX/upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality"
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path


# =========================================================================
# Decoding helpers
# =========================================================================

def decode_fp16_pair(dword: int) -> tuple:
    lo = dword & 0xFFFF
    hi = (dword >> 16) & 0xFFFF
    lo_f = struct.unpack('e', struct.pack('H', lo))[0]
    hi_f = struct.unpack('e', struct.pack('H', hi))[0]
    return lo_f, hi_f


def decode_int8_quad(dword: int) -> tuple:
    b0 = dword & 0xFF
    b1 = (dword >> 8) & 0xFF
    b2 = (dword >> 16) & 0xFF
    b3 = (dword >> 24) & 0xFF
    return tuple(b - 256 if b > 127 else b for b in [b0, b1, b2, b3])


# =========================================================================
# HLSL parsing
# =========================================================================

def parse_embedded_arrays(hlsl_text: str) -> list:
    """Parse all static const uint embedded_*_dwords arrays."""
    pattern = re.compile(
        r'static\s+const\s+uint\s+(embedded_\w+_dwords)\[(\d+)\]\s*=\s*\{([^}]+)\};',
        re.DOTALL
    )
    arrays = []
    for match in pattern.finditer(hlsl_text):
        name = match.group(1)
        declared_size = int(match.group(2))
        body = match.group(3)

        hex_pattern = re.compile(r'0x([0-9a-fA-F]+)')
        dwords = [int(h.group(1), 16) for h in hex_pattern.finditer(body)]

        if len(dwords) != declared_size:
            print(f"  WARNING: {name}: declared {declared_size}, found {len(dwords)} dwords",
                  file=sys.stderr)

        comment_values = []
        for line in body.split('\n'):
            comment_match = re.match(r'\s*//\s*(.+)', line)
            if comment_match:
                for v in comment_match.group(1).split(','):
                    try:
                        comment_values.append(float(v.strip()))
                    except ValueError:
                        pass

        arrays.append({
            'name': name,
            'dword_count': len(dwords),
            'dwords': dwords,
            'comment_floats': comment_values,
        })
    return arrays


def parse_tensor_metadata(hlsl_text: str) -> dict:
    """Parse all QuantizedTensor4i8_NHWC/HWCN and Tensor4h_NHWC declarations.

    Returns dict: clean_tensor_name -> {shape, layout, scale, offset, source}
    """
    metadata = {}

    # Helper: match template param like ConstantBufferStorage<576> or BufferStorage
    tp = r'[^>]*(?:<[^>]*>)?[^>]*'

    # INT8 NHWC tensors (both ConstantBufferStorage and BufferStorage/InitializerBuffer)
    nhwc_pattern = re.compile(
        r'QuantizedTensor4i8_NHWC\s*<\s*' + tp + r'\s*>\s*'
        r'(\w+)\s*=\s*\{\s*'
        r'uint4\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\).*?'
        r'(\d+),\s*//\s*threadGroupStorageByteOffset\s*\n\s*'
        r'([\d.eE+-]+)',
        re.DOTALL
    )
    for m in nhwc_pattern.finditer(hlsl_text):
        name = m.group(1)
        shape = [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))]
        offset = int(m.group(6))
        scale = float(m.group(7))
        is_initializer = 'InitializerBuffer' in hlsl_text[max(0, m.start()-300):m.start()]
        metadata[name] = {
            'shape': shape, 'layout': 'NHWC', 'scale': scale,
            'offset': offset, 'source': 'initializers.bin' if is_initializer else 'hlsl_static',
        }

    # INT8 HWCN tensors (ConvTranspose2d weights)
    hwcn_pattern = re.compile(
        r'QuantizedTensor4i8_HWCN\s*<\s*' + tp + r'\s*>\s*'
        r'(\w+)\s*=\s*\{\s*'
        r'uint4\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\).*?'
        r'(\d+),\s*//\s*threadGroupStorageByteOffset\s*\n\s*'
        r'([\d.eE+-]+)',
        re.DOTALL
    )
    for m in hwcn_pattern.finditer(hlsl_text):
        name = m.group(1)
        shape = [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))]
        offset = int(m.group(6))
        scale = float(m.group(7))
        is_initializer = 'InitializerBuffer' in hlsl_text[max(0, m.start()-300):m.start()]
        metadata[name] = {
            'shape': shape, 'layout': 'HWCN', 'scale': scale,
            'offset': offset, 'source': 'initializers.bin' if is_initializer else 'hlsl_static',
        }

    # FP16 NHWC tensors (encoder1 weight)
    fp16_pattern = re.compile(
        r'Tensor4h_NHWC\s*<\s*' + tp + r'\s*>\s*'
        r'(\w+)\s*=\s*\{\s*'
        r'uint4\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
        re.DOTALL
    )
    for m in fp16_pattern.finditer(hlsl_text):
        name = m.group(1)
        shape = [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))]
        metadata[name] = {
            'shape': shape, 'layout': 'NHWC', 'scale': None,
            'offset': 0, 'source': 'hlsl_static',
        }

    # FP16 bias tensors (Tensor1h)
    bias_pattern = re.compile(
        r'Tensor1h\s*<\s*' + tp + r'\s*>\s*'
        r'(\w+)\s*=\s*\{\s*'
        r'(\d+),\s*//\s*logicalSize',
        re.DOTALL
    )
    for m in bias_pattern.finditer(hlsl_text):
        name = m.group(1)
        size = int(m.group(2))
        metadata[name] = {
            'shape': [size], 'layout': 'flat', 'scale': None,
            'offset': 0, 'source': 'hlsl_static',
        }

    return metadata


def infer_tensor_type(name: str) -> str:
    if 'quant' in name.lower() or 'quantize' in name.lower():
        return 'int8'
    if 'bias' in name.lower():
        return 'fp16'
    if 'weight' in name.lower():
        return 'fp16'
    return 'fp16'


def clean_layer_name(raw_name: str) -> str:
    """Simplify HLSL tensor names to clean identifiers."""
    name = raw_name
    # Remove common prefixes/suffixes
    name = name.replace('embedded_', '')
    name = name.replace('_quant_export_handler_QuantizeLinear_output_0', '')
    name = name.replace('_dwords', '')
    # Clean up leading underscores (from HLSL naming)
    name = name.lstrip('_')
    return name


def match_metadata(clean_name: str, metadata: dict) -> dict:
    """Find metadata entry matching a clean layer name."""
    # Try exact match first
    for key, val in metadata.items():
        key_clean = clean_layer_name(key)
        if key_clean == clean_name:
            return val

    # Try partial match (metadata names may have slightly different form)
    for key, val in metadata.items():
        key_clean = clean_layer_name(key)
        # Match without 'body_' prefix variations
        if clean_name.replace('body_', '') == key_clean.replace('body_', ''):
            return val

    return None


# =========================================================================
# Extraction
# =========================================================================

def extract_weights(hlsl_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'source': str(hlsl_dir),
        'model': 'FSR4 v07 INT8',
        'layers': [],
    }

    hlsl_files = sorted(hlsl_dir.glob('*.hlsl'))
    if not hlsl_files:
        print(f"No .hlsl files found in {hlsl_dir}", file=sys.stderr)
        return manifest

    # Collect all tensor metadata from all HLSL files
    all_metadata = {}
    all_hlsl_text = ""
    for hlsl_file in hlsl_files:
        text = hlsl_file.read_text(encoding='utf-8', errors='replace')
        all_hlsl_text += text
        meta = parse_tensor_metadata(text)
        all_metadata.update(meta)
    print(f"Parsed {len(all_metadata)} tensor metadata entries from HLSL")

    # Track already-extracted names to deduplicate across HLSL files
    seen_names = set()

    # --- Phase 1: Extract static const arrays from HLSL ---
    for hlsl_file in hlsl_files:
        print(f"\nProcessing {hlsl_file.name}...")
        text = hlsl_file.read_text(encoding='utf-8', errors='replace')
        arrays = parse_embedded_arrays(text)

        for arr in arrays:
            raw_name = arr['name'].replace('_dwords', '')
            clean = clean_layer_name(raw_name)

            if clean in seen_names:
                continue
            seen_names.add(clean)

            tensor_type = infer_tensor_type(raw_name)

            if tensor_type == 'fp16':
                values = []
                for dw in arr['dwords']:
                    lo, hi = decode_fp16_pair(dw)
                    values.extend([lo, hi])
                bin_data = struct.pack(f'{len(values)}e', *values)
                dtype = 'float16'
                elem_count = len(values)
            else:
                values = []
                for dw in arr['dwords']:
                    values.extend(decode_int8_quad(dw))
                bin_data = struct.pack(f'{len(values)}b', *values)
                dtype = 'int8'
                elem_count = len(values)

            bin_filename = f"{clean}.bin"
            bin_file = output_dir / bin_filename
            bin_file.write_bytes(bin_data)

            meta = match_metadata(clean, all_metadata)

            layer_info = {
                'name': clean,
                'dtype': dtype,
                'element_count': elem_count,
                'byte_size': len(bin_data),
                'bin_file': bin_filename,
                'source': 'hlsl_static',
            }

            if meta:
                layer_info['shape'] = meta['shape']
                layer_info['layout'] = meta['layout']
                if meta['scale'] is not None:
                    layer_info['quant_scale'] = meta['scale']

            # Validate FP16 against comment floats
            if arr['comment_floats'] and dtype == 'float16':
                max_err = 0
                checked = min(len(values), len(arr['comment_floats']))
                for i in range(checked):
                    err = abs(values[i] - arr['comment_floats'][i])
                    max_err = max(max_err, err)
                layer_info['validation_max_err'] = max_err
                if max_err > 0.01:
                    print(f"    WARNING: {clean} max_err={max_err:.6f}", file=sys.stderr)

            manifest['layers'].append(layer_info)
            print(f"  {clean}: {elem_count} {dtype} → {bin_filename}")

    # --- Phase 2: Extract weights from initializers.bin ---
    init_bin = hlsl_dir / 'initializers.bin'
    if init_bin.exists():
        print(f"\nProcessing initializers.bin ({init_bin.stat().st_size} bytes)...")
        init_data = init_bin.read_bytes()

        # Find all metadata entries that reference InitializerBuffer
        for tensor_name, meta in all_metadata.items():
            if meta['source'] != 'initializers.bin':
                continue

            clean = clean_layer_name(tensor_name)
            if clean in seen_names:
                continue
            seen_names.add(clean)

            shape = meta['shape']
            elem_count = 1
            for s in shape:
                elem_count *= s
            offset = meta['offset']

            if offset + elem_count > len(init_data):
                print(f"  WARNING: {clean} offset={offset} size={elem_count} "
                      f"exceeds file ({len(init_data)} bytes)", file=sys.stderr)
                continue

            bin_data = init_data[offset:offset + elem_count]

            bin_filename = f"{clean}.bin"
            bin_file = output_dir / bin_filename
            bin_file.write_bytes(bin_data)

            layer_info = {
                'name': clean,
                'dtype': 'int8',
                'element_count': elem_count,
                'byte_size': elem_count,
                'bin_file': bin_filename,
                'source': 'initializers.bin',
                'initializer_offset': offset,
                'shape': shape,
                'layout': meta['layout'],
                'quant_scale': meta['scale'],
            }
            manifest['layers'].append(layer_info)
            print(f"  {clean}: {elem_count} int8 @ offset {offset} → {bin_filename}")
    else:
        print(f"\nNo initializers.bin found (only static const weights extracted)")

    # --- Phase 3: Fill in missing metadata for static arrays ---
    # Some static arrays may not have had metadata matches; do a second pass
    for layer in manifest['layers']:
        if 'shape' not in layer:
            # Infer shape from element count for known patterns
            name = layer['name']
            ec = layer['element_count']
            if 'bias' in name:
                layer['shape'] = [ec]
                layer['layout'] = 'flat'

    # Sort layers by network order
    stage_order = {
        'encoder1': 0, 'encoder2': 1, 'encoder3': 2,
        'bottleneck': 3,
        'decoder3': 4, 'decoder2': 5, 'decoder1': 6,
    }

    def sort_key(layer):
        name = layer['name']
        for prefix, order in stage_order.items():
            if name.startswith(prefix):
                return (order, name)
        return (99, name)

    manifest['layers'].sort(key=sort_key)

    # Summary statistics
    total_params = sum(l['element_count'] for l in manifest['layers'])
    int8_params = sum(l['element_count'] for l in manifest['layers'] if l['dtype'] == 'int8')
    fp16_params = sum(l['element_count'] for l in manifest['layers'] if l['dtype'] == 'float16')
    total_bytes = sum(l['byte_size'] for l in manifest['layers'])

    manifest['summary'] = {
        'total_layers': len(manifest['layers']),
        'total_params': total_params,
        'int8_params': int8_params,
        'fp16_params': fp16_params,
        'total_bytes': total_bytes,
    }

    # Write manifest
    manifest_file = output_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Manifest: {manifest_file}")
    print(f"Layers:   {len(manifest['layers'])}")
    print(f"Params:   {total_params:,} ({int8_params:,} INT8 + {fp16_params:,} FP16)")
    print(f"Size:     {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"{'='*60}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description='Extract FSR4 weights from HLSL shaders')
    parser.add_argument('hlsl_dir', type=Path, help='Directory containing HLSL shader files')
    parser.add_argument('--output-dir', '-o', type=Path, default=None,
                        help='Output directory (default: weights/<variant>)')
    args = parser.parse_args()

    if not args.hlsl_dir.is_dir():
        print(f"Error: {args.hlsl_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    variant = args.hlsl_dir.name.replace('fsr4_model_v07_i8_', '')
    output_dir = args.output_dir or (Path(__file__).parent.parent / 'weights' / variant)
    extract_weights(args.hlsl_dir, output_dir)


if __name__ == '__main__':
    main()
