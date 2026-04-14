#!/usr/bin/env python3
"""GEMM Benchmark: test different GEMM implementations against collected ground truth.

Loads data from gemm_data/ directory (produced by collect_gemm_data.py),
runs 6 GEMM tests, computes accuracy and performance metrics, outputs table + JSON.

Usage:
    python bench_gemm.py --data_dir ./gemm_data [--layers q_proj,o_proj,gate_proj,down_proj] [--batch_sizes 1,2,4]
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F


# ============================================================
# Metrics
# ============================================================

def compute_accuracy_metrics(y_test, y_ref):
    """Compute accuracy metrics between test output and reference."""
    y_test = y_test.float().flatten()
    y_ref = y_ref.float().flatten()
    diff = y_test - y_ref

    max_abs_err = diff.abs().max().item()
    mae = diff.abs().mean().item()
    rmse = (diff.pow(2).mean()).sqrt().item()

    eps = 1e-8
    mape = (diff.abs() / (y_ref.abs() + eps)).mean().item()

    cos_sim = F.cosine_similarity(y_test.unsqueeze(0), y_ref.unsqueeze(0)).item()

    ref_power = y_ref.pow(2).sum().item()
    err_power = diff.pow(2).sum().item()
    if err_power > 0:
        snr_db = 10 * torch.log10(torch.tensor(ref_power / err_power)).item()
    else:
        snr_db = float('inf')

    return {
        'max_abs_err': max_abs_err,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'cosine_sim': cos_sim,
        'snr_db': snr_db,
    }


def compute_perf_metrics(func, args, M, N, K, warmup=10, repeat=100):
    """Compute performance metrics for a GEMM function."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat

    latency_ms = elapsed * 1000
    flops = 2 * M * N * K
    tflops = flops / elapsed / 1e12 if elapsed > 0 else 0

    return {
        'latency_ms': latency_ms,
        'tflops': tflops,
    }


# ============================================================
# GEMM Test Implementations
# ============================================================

def gemm_fp16_baseline(x_fp16, W_fp16):
    """Test 1: FP16 matmul, no quantization."""
    x_flat = x_fp16.reshape(-1, x_fp16.shape[-1])
    return (x_flat @ W_fp16.T).reshape(*x_fp16.shape[:-1], -1)


def gemm_fake_quant(output_fake):
    """Test 2: Return pre-collected fake quant output (no computation needed)."""
    return output_fake


def _split_and_dequant_activation(act_quant_main, act_quant_high=None):
    """Helper: dequantize activation integers back to fp16 (fake quant equivalent)."""
    q_m = act_quant_main['q_int'].cuda()
    s_m = act_quant_main['scale'].cuda()
    z_m = act_quant_main.get('zero', None)
    if z_m is not None:
        z_m = z_m.cuda()
        dq_m = s_m * (q_m.float() - z_m.float())
    else:
        dq_m = s_m * q_m.float()

    dq_h = None
    if act_quant_high is not None:
        q_h = act_quant_high['q_int'].cuda()
        s_h = act_quant_high['scale'].cuda()
        z_h = act_quant_high.get('zero', None)
        if z_h is not None:
            z_h = z_h.cuda()
            dq_h = s_h * (q_h.float() - z_h.float())
        else:
            dq_h = s_h * q_h.float()

    return dq_m, dq_h


def gemm_real_quant(act_quant_main, act_quant_high,
                    weight_int_main, weight_int_high,
                    accumulate_dtype=torch.float32):
    """Tests 3/4/5: Real quant GEMM with configurable accumulation.

    accumulate_dtype:
        torch.float32 -> Test 3 (fp32 accum)
        torch.float16 -> Test 4 (fp16 accum)
        torch.int32   -> Test 5 (int32 accum)
    """
    # Activation: get centered integers and scales
    q_x_m = act_quant_main['q_int'].cuda()  # int16
    s_x_m = act_quant_main['scale'].cuda()
    z_x_m = act_quant_main.get('zero', None)
    if z_x_m is not None:
        z_x_m = z_x_m.cuda()
        q_x_m_centered = q_x_m.float() - z_x_m.float()
    else:
        q_x_m_centered = q_x_m.float()

    # Weight: get integers and scale
    q_w_m = weight_int_main['q_int'].cuda()  # half (centered ints)
    s_w_m = weight_int_main['scale'].cuda()

    # Main group matmul
    init_shape = q_x_m_centered.shape
    q_x_flat = q_x_m_centered.reshape(-1, q_x_m_centered.shape[-1])

    if accumulate_dtype == torch.int32:
        y_m_int = q_x_flat.int() @ q_w_m.int().T
        y_m = y_m_int.float()
    elif accumulate_dtype == torch.float16:
        y_m = q_x_flat.half() @ q_w_m.half().T
        y_m = y_m.float()
    else:  # fp32
        y_m = q_x_flat.float() @ q_w_m.float().T

    # Apply scales: per-token s_x (M,1) * per-channel s_w (1,N)
    # s_x shape varies: could be (batch, seq, 1) or (batch, seq, ngroups, 1)
    # For simplicity, reshape s_x to (M, 1)
    s_x_m_flat = s_x_m.reshape(-1, s_x_m.shape[-1])  # (M, 1) or similar
    # Handle case where s_x has more dims (groupsize)
    if s_x_m_flat.shape[-1] != 1:
        # Per-group scale — average or broadcast. For benchmark, just use mean.
        s_x_m_flat = s_x_m_flat.mean(dim=-1, keepdim=True)
    y_m = s_x_m_flat * s_w_m.flatten().unsqueeze(0) * y_m

    # High group (if exists)
    y_h = torch.zeros_like(y_m)
    if act_quant_high is not None and weight_int_high is not None:
        q_x_h = act_quant_high['q_int'].cuda()
        s_x_h = act_quant_high['scale'].cuda()
        z_x_h = act_quant_high.get('zero', None)
        if z_x_h is not None:
            z_x_h = z_x_h.cuda()
            q_x_h_centered = q_x_h.float() - z_x_h.float()
        else:
            q_x_h_centered = q_x_h.float()

        q_w_h = weight_int_high['q_int'].cuda()
        s_w_h = weight_int_high['scale'].cuda()

        q_x_h_flat = q_x_h_centered.reshape(-1, q_x_h_centered.shape[-1])

        if accumulate_dtype == torch.int32:
            y_h_int = q_x_h_flat.int() @ q_w_h.int().T
            y_h = y_h_int.float()
        elif accumulate_dtype == torch.float16:
            y_h = q_x_h_flat.half() @ q_w_h.half().T
            y_h = y_h.float()
        else:
            y_h = q_x_h_flat.float() @ q_w_h.float().T

        s_x_h_flat = s_x_h.reshape(-1, s_x_h.shape[-1])
        if s_x_h_flat.shape[-1] != 1:
            s_x_h_flat = s_x_h_flat.mean(dim=-1, keepdim=True)
        y_h = s_x_h_flat * s_w_h.flatten().unsqueeze(0) * y_h

    y = (y_m + y_h).half()
    return y.reshape(*init_shape[:-1], -1)


def gemm_custom_kernel(data, kernel_path=None):
    """Test 6: Placeholder for custom CUDA kernel."""
    return None


# ============================================================
# Layer Benchmark
# ============================================================

ALL_TESTS = [
    ('FP16 baseline', 'fp16'),
    ('Fake quant', 'fake'),
    ('Real (fp32 acc)', 'real_fp32'),
    ('Real (fp16 acc)', 'real_fp16'),
    ('Real (int32 acc)', 'real_int32'),
    ('Custom kernel', 'custom'),
]


def load_layer_data(layer_dir, bs_key):
    """Load all data for a single layer + batch size."""
    data = {}

    # Input
    path = os.path.join(layer_dir, f'input_fp16_{bs_key}.pt')
    if os.path.exists(path):
        data['input_fp16'] = torch.load(path, map_location='cpu')

    # Weight
    path = os.path.join(layer_dir, 'weight_fp16.pt')
    if os.path.exists(path):
        data['weight_fp16'] = torch.load(path, map_location='cpu')

    # Weight integers
    for group in ['main', 'high', 'low']:
        path = os.path.join(layer_dir, f'weight_int_{group}.pt')
        if os.path.exists(path):
            data[f'weight_int_{group}'] = torch.load(path, map_location='cpu')

    # Activation quant
    for group in ['main', 'high', 'low']:
        path = os.path.join(layer_dir, f'act_quant_{group}_{bs_key}.pt')
        if os.path.exists(path):
            data[f'act_quant_{group}'] = torch.load(path, map_location='cpu')

    # Outputs
    for kind in ['fp16_baseline', 'fake_quant', 'real_quant']:
        path = os.path.join(layer_dir, f'output_{kind}_{bs_key}.pt')
        if os.path.exists(path):
            data[f'output_{kind}'] = torch.load(path, map_location='cpu')

    # Metadata
    path = os.path.join(layer_dir, 'metadata.json')
    if os.path.exists(path):
        with open(path) as f:
            data['metadata'] = json.load(f)

    return data


def bench_single_layer(layer_dir, bs_key):
    """Run all GEMM tests on a single layer for a single batch size."""
    data = load_layer_data(layer_dir, bs_key)

    if 'input_fp16' not in data or 'weight_fp16' not in data:
        return None

    x_fp16 = data['input_fp16'].cuda()
    W_fp16 = data['weight_fp16'].cuda()
    meta = data.get('metadata', {})
    N, K = W_fp16.shape
    M = x_fp16.reshape(-1, x_fp16.shape[-1]).shape[0]

    results = {}

    # Reference outputs
    ref_fp16 = data.get('output_fp16_baseline', gemm_fp16_baseline(x_fp16, W_fp16).cpu()).cuda()
    ref_fake = data.get('output_fake_quant', ref_fp16).cuda()

    for test_name, test_key in ALL_TESTS:
        result = {'accuracy_vs_fp16': None, 'accuracy_vs_fake': None, 'perf': None}

        try:
            if test_key == 'fp16':
                y = gemm_fp16_baseline(x_fp16, W_fp16)
                perf = compute_perf_metrics(gemm_fp16_baseline, (x_fp16, W_fp16), M, N, K)

            elif test_key == 'fake':
                y = ref_fake.clone()
                # For perf, simulate: dequant(x) @ W.T
                act_main = data.get('act_quant_main')
                if act_main is not None:
                    dq_m, dq_h = _split_and_dequant_activation(act_main, data.get('act_quant_high'))
                    dq_x = torch.cat([p for p in [dq_m, dq_h] if p is not None], dim=-1)
                    dq_x_flat = dq_x.reshape(-1, dq_x.shape[-1]).half().cuda()
                    perf = compute_perf_metrics(
                        lambda a, b: a @ b.T, (dq_x_flat, W_fp16), M, N, K)
                else:
                    perf = compute_perf_metrics(gemm_fp16_baseline, (x_fp16, W_fp16), M, N, K)

            elif test_key in ('real_fp32', 'real_fp16', 'real_int32'):
                acc_dtype = {
                    'real_fp32': torch.float32,
                    'real_fp16': torch.float16,
                    'real_int32': torch.int32,
                }[test_key]

                act_main = data.get('act_quant_main')
                act_high = data.get('act_quant_high')
                w_main = data.get('weight_int_main')
                w_high = data.get('weight_int_high')

                if act_main is None or w_main is None:
                    result['error'] = 'missing quant data'
                    results[test_name] = result
                    continue

                y = gemm_real_quant(act_main, act_high, w_main, w_high, acc_dtype)

                perf = compute_perf_metrics(
                    lambda: gemm_real_quant(act_main, act_high, w_main, w_high, acc_dtype),
                    (), M, N, K, warmup=5, repeat=20)

            elif test_key == 'custom':
                y = gemm_custom_kernel(data)
                if y is None:
                    result['error'] = 'not implemented'
                    results[test_name] = result
                    continue
                perf = {'latency_ms': 0, 'tflops': 0}
            else:
                continue

            y = y.cuda()
            result['accuracy_vs_fp16'] = compute_accuracy_metrics(y, ref_fp16)
            result['accuracy_vs_fake'] = compute_accuracy_metrics(y, ref_fake)
            result['perf'] = perf

        except Exception as e:
            result['error'] = str(e)

        results[test_name] = result

    # Add FP16 baseline perf as reference for speedup calculation
    fp16_lat = results.get('FP16 baseline', {}).get('perf', {}).get('latency_ms', 1)
    for test_name in results:
        r = results[test_name]
        if r.get('perf') and fp16_lat > 0:
            r['perf']['speedup_vs_fp16'] = fp16_lat / r['perf']['latency_ms']

    # Memory comparison
    w_fp16_mb = W_fp16.numel() * 2 / 1e6
    results['_memory'] = {
        'weight_fp16_mb': w_fp16_mb,
        'weight_int4_mb': W_fp16.numel() * 0.5 / 1e6,  # theoretical
        'weight_int8_mb': W_fp16.numel() * 1 / 1e6,      # theoretical
    }
    results['_meta'] = {'M': M, 'N': N, 'K': K, 'layer': meta.get('name', '')}

    return results


# ============================================================
# Output Formatting
# ============================================================

def print_results_table(layer_name, results, bs_key):
    """Print a formatted results table for a single layer + batch size."""
    meta = results.get('_meta', {})
    M, N, K = meta.get('M', '?'), meta.get('N', '?'), meta.get('K', '?')
    print(f"\nLayer: {meta.get('layer', layer_name)} (M={M}, N={N}, K={K}) [{bs_key}]")
    print("=" * 120)

    header = f"{'Test':<20} {'MaxAbsErr':>10} {'MAE':>10} {'RMSE':>10} {'CosSim':>10} {'SNR(dB)':>8} {'Lat(ms)':>8} {'TFLOPS':>8} {'vs FP16':>8}"
    print(header)
    print("-" * 120)

    for test_name, _ in ALL_TESTS:
        r = results.get(test_name, {})
        if 'error' in r:
            print(f"{test_name:<20} {'N/A':>10} {'':>10} {'':>10} {'':>10} {'':>8} {'':>8} {'':>8} {r['error']}")
            continue

        acc = r.get('accuracy_vs_fake', r.get('accuracy_vs_fp16', {}))
        perf = r.get('perf', {})

        if acc is None:
            acc = {}
        if perf is None:
            perf = {}

        max_err = f"{acc.get('max_abs_err', 0):.6f}" if acc else "N/A"
        mae = f"{acc.get('mae', 0):.6f}" if acc else "N/A"
        rmse = f"{acc.get('rmse', 0):.6f}" if acc else "N/A"
        cos = f"{acc.get('cosine_sim', 0):.6f}" if acc else "N/A"
        snr = f"{acc.get('snr_db', 0):.1f}" if acc and acc.get('snr_db') != float('inf') else "inf"
        lat = f"{perf.get('latency_ms', 0):.3f}" if perf else "N/A"
        tflops = f"{perf.get('tflops', 0):.2f}" if perf else "N/A"
        speedup = f"{perf.get('speedup_vs_fp16', 0):.2f}x" if perf.get('speedup_vs_fp16') else "N/A"

        print(f"{test_name:<20} {max_err:>10} {mae:>10} {rmse:>10} {cos:>10} {snr:>8} {lat:>8} {tflops:>8} {speedup:>8}")

    mem = results.get('_memory', {})
    if mem:
        print(f"\nMemory: FP16={mem.get('weight_fp16_mb', 0):.1f}MB, "
              f"INT8={mem.get('weight_int8_mb', 0):.1f}MB, "
              f"INT4={mem.get('weight_int4_mb', 0):.1f}MB")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='GEMM Benchmark')
    parser.add_argument('--data_dir', type=str, default='./gemm_data',
                        help='Directory containing collected GEMM data')
    parser.add_argument('--layers', type=str, default='',
                        help='Comma-separated layer name substrings to test (default: all)')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4',
                        help='Comma-separated batch sizes to test')
    parser.add_argument('--output', type=str, default='gemm_benchmark_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    layer_filters = [x.strip() for x in args.layers.split(',')] if args.layers else []

    # Find all layer directories
    if not os.path.exists(args.data_dir):
        print(f"Error: data directory {args.data_dir} not found. Run collect_gemm_data.py first.")
        return

    layer_dirs = []
    for d in sorted(os.listdir(args.data_dir)):
        full_path = os.path.join(args.data_dir, d)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'metadata.json')):
            if not layer_filters or any(f in d for f in layer_filters):
                layer_dirs.append((d, full_path))

    print(f"Found {len(layer_dirs)} layers to benchmark")
    if layer_filters:
        print(f"Filter: {layer_filters}")

    all_results = {}
    for layer_name, layer_path in layer_dirs:
        for bs in batch_sizes:
            bs_key = f"bs{bs}"
            # Check if data exists for this batch size
            if not os.path.exists(os.path.join(layer_path, f'input_fp16_{bs_key}.pt')):
                continue

            print(f"\nBenchmarking: {layer_name} [{bs_key}]...")
            results = bench_single_layer(layer_path, bs_key)
            if results is not None:
                key = f"{layer_name}/{bs_key}"
                all_results[key] = results
                print_results_table(layer_name, results, bs_key)

    # Save JSON
    # Convert non-serializable values
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, float):
            if obj == float('inf'):
                return "inf"
            elif obj == float('-inf'):
                return "-inf"
            return obj
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        return obj

    with open(args.output, 'w') as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
