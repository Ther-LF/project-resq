#!/usr/bin/env python3
"""GEMM Benchmark: test different GEMM implementations against collected ground truth.

Loads data from gemm_data/ directory (produced by collect_gemm_data.py),
runs GEMM tests, computes accuracy and performance metrics, outputs table + JSON.

Usage:
    python bench_gemm.py --data_dir ./gemm_data [--layers q_proj,o_proj,gate_proj,down_proj] [--batch_sizes 1,2,4]
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

# Triton and CUTLASS GEMM kernels removed — baseline uses fp32 simulation only.
# Real INT TC kernels will be added later as separate baselines.


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


def compute_perf_metrics(func, M, N, K, warmup=10, repeat=100):
    """Compute performance metrics for a GEMM function (no-arg callable)."""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        func()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat

    latency_ms = elapsed * 1000
    flops = 2 * M * N * K
    tflops = flops / elapsed / 1e12 if elapsed > 0 else 0

    return {'latency_ms': latency_ms, 'tflops': tflops}


# ============================================================
# Helpers
# ============================================================

def _extract_per_token_scale(scale_tensor):
    """Extract per-token scale as (M, 1) from potentially expanded scale.

    ActQuantizer._find_params expands scale to match x shape, so scale may be
    (batch, seq, K_dim) where all K_dim values are the same. We take [:, :, :1].
    """
    if scale_tensor.dim() == 3:
        return scale_tensor[..., :1]  # (batch, seq, 1)
    return scale_tensor



# ============================================================
# GEMM Implementations
# ============================================================

def gemm_fp16_baseline(x_fp16, W_fp16):
    """FP16 matmul, no quantization."""
    if x_fp16.dim() > 3:
        batch, seq = x_fp16.shape[0], x_fp16.shape[1]
        x_fp16 = x_fp16.reshape(batch, seq, -1)
    x_flat = x_fp16.reshape(-1, x_fp16.shape[-1])
    return (x_flat @ W_fp16.T).reshape(*x_fp16.shape[:-1], -1)


def _real_quant_single_group_main(q_x_raw, zero_x, s_x, q_w, s_w):
    """Main group (4-bit) quant matmul using shift-8 + bias formula (same as INT4 GEMM).

    Main group: q_int [0,15], needs shift-8 to fit int4 [-8,7].
    Shift-8 trick (same idea as high group's shift-128):
        q_shifted = q_int - 8             → [-8,7] fits int4
        bias = (8 - zero) * colsum(q_w)   → precomputed per (M, N)
        Y = s_x * s_w * (q_shifted @ q_w^T + bias)

    q_x_raw: (M, K) raw unsigned q_int as float (e.g., [0,15])
    zero_x:  (M, 1) or scalar zero point as float
    s_x:     (M, 1) per-token scale as float
    q_w:     (N, K) centered weight int as float (e.g., [-8,7])
    s_w:     (N, 1) per-channel weight scale
    """
    q_shifted = q_x_raw - 8.0  # [-8, 7], fits int4

    # Precompute bias: (8 - zero) * colsum(q_w)
    shift_minus_zero = 8.0 - zero_x  # (M, 1)
    w_colsum = q_w.float().sum(dim=1, keepdim=True).T  # (1, N)
    bias = shift_minus_zero @ w_colsum  # (M, N)

    # fp32 simulates: q_shifted_int4 @ q_w_int4^T
    y_int = q_shifted.float() @ q_w.float().T  # fp32 simulates int matmul

    y = s_x * s_w.flatten().unsqueeze(0) * (y_int + bias)
    return y


def _real_quant_single_group_high(q_x_raw, zero_x, s_x, q_w, s_w):
    """High group (8-bit) quant matmul using shift-128 + bias formula (same as INT GEMM).

    High group: q_int [0,255], centered overflows int8.
    Shift-128 trick:
        q_shifted = q_int - 128            → [-128,127] fits int8
        bias = (128 - zero) * colsum(q_w)  → precomputed per (M, N)
        Y = s_x * s_w * (q_shifted @ q_w^T + bias)

    q_x_raw: (M, K) raw unsigned q_int as float (e.g., [0,255])
    zero_x:  (M, 1) or scalar zero point as float
    s_x:     (M, 1) per-token scale as float
    q_w:     (N, K) centered weight int as float (e.g., [-128,127])
    s_w:     (N, 1) per-channel weight scale
    """
    q_shifted = q_x_raw - 128.0  # [-128, 127], same as INT GEMM .to(int8)

    # Precompute bias: (128 - zero) * colsum(q_w)
    shift_minus_zero = 128.0 - zero_x  # (M, 1)
    w_colsum = q_w.float().sum(dim=1, keepdim=True).T  # (1, N)
    bias = shift_minus_zero @ w_colsum  # (M, N)

    # fp32 simulates: q_shifted_int8 @ q_w_int8^T
    y_int = q_shifted.float() @ q_w.float().T  # fp32 simulates int matmul

    y = s_x * s_w.flatten().unsqueeze(0) * (y_int + bias)
    return y


def gemm_real_quant(act_quant_main, act_quant_high,
                    weight_int_main, weight_int_high,
                    column_order=None):
    """Quantization accuracy baseline — fp32 simulation of exact INT GEMM formula.

    Uses the SAME formula as the INT TC kernel (shift-8/shift-128 + bias),
    but computes matmul in fp32 instead of INT4/INT8 tensor core.
    If the INT kernel is implemented correctly, its output should match this exactly.

    For per-group layers with column_order (o_proj):
        Activation is quantized in original K space (per-group).
        Weight is in rearranged K space (GPTQ).
        We dequant activation per-group, flatten to original K, apply column_order,
        then do one matmul with the rearranged weight.
    """
    q_w_m = weight_int_main['q_int'].cuda().float()
    s_w_m = weight_int_main['scale'].cuda().float()

    q_w_h = weight_int_high['q_int'].cuda().float() if weight_int_high else None
    s_w_h = weight_int_high['scale'].cuda().float() if weight_int_high else None

    # Get raw (uncentered) activation data
    q_x_m_raw = act_quant_main['q_int'].cuda().float()
    s_x_m = act_quant_main['scale'].cuda().float()
    z_x_m = act_quant_main.get('zero', None)
    if z_x_m is not None:
        z_x_m = z_x_m.cuda().float()
    else:
        z_x_m = torch.zeros_like(q_x_m_raw[..., :1])

    grouped = (s_x_m.dim() > 3)  # (batch, seq, ngroups, 1) = grouped
    N = q_w_m.shape[0]

    if not grouped:
        # === Per-token ===
        s_x_tok = _extract_per_token_scale(s_x_m).reshape(-1, 1)
        z_x_tok = _extract_per_token_scale(z_x_m).reshape(-1, 1) if z_x_m.shape[-1] > 1 else z_x_m.reshape(-1, 1)
        q_x_flat = q_x_m_raw.reshape(-1, q_x_m_raw.shape[-1])
        M = q_x_flat.shape[0]

        y_m = _real_quant_single_group_main(q_x_flat, z_x_tok, s_x_tok, q_w_m, s_w_m)

        y_h = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        if act_quant_high is not None and q_w_h is not None:
            q_x_h_raw = act_quant_high['q_int'].cuda().float()
            s_x_h = act_quant_high['scale'].cuda().float()
            z_x_h = act_quant_high.get('zero', None)
            if z_x_h is not None:
                z_x_h = z_x_h.cuda().float()
            else:
                z_x_h = torch.zeros_like(q_x_h_raw[..., :1])

            s_x_h_tok = _extract_per_token_scale(s_x_h).reshape(-1, 1)
            z_x_h_tok = _extract_per_token_scale(z_x_h).reshape(-1, 1) if z_x_h.shape[-1] > 1 else z_x_h.reshape(-1, 1)
            q_x_h_flat = q_x_h_raw.reshape(-1, q_x_h_raw.shape[-1])

            y_h = _real_quant_single_group_high(q_x_h_flat, z_x_h_tok, s_x_h_tok, q_w_h, s_w_h)

        y = (y_m + y_h).half()
        return y

    else:
        # === Per-group (o_proj) ===
        # Both activation and weight are in the SAME K space (original or rearranged).
        # Activation per-group layout: [g0_main(56)|g0_high(8)|g1_main(56)|g1_high(8)|...]
        # We dequant per-group, reconstruct the interleaved layout, then matmul with
        # the full dequanted weight (also reconstructed in the same interleaved layout).
        batch, seq, ngroups, group_k_m = q_x_m_raw.shape
        M = batch * seq

        # Dequant main activation: scale * (q_int - zero) per group
        x_dq_m = s_x_m * (q_x_m_raw - z_x_m)  # (batch, seq, ngroups, group_k_m=56)

        # Dequant high activation and interleave with main
        if act_quant_high is not None and q_w_h is not None:
            q_x_h_raw = act_quant_high['q_int'].cuda().float()
            s_x_h = act_quant_high['scale'].cuda().float()
            z_x_h = act_quant_high.get('zero', None)
            if z_x_h is not None:
                z_x_h = z_x_h.cuda().float()
            else:
                z_x_h = torch.zeros_like(q_x_h_raw[..., :1])

            x_dq_h = s_x_h * (q_x_h_raw - z_x_h)  # (batch, seq, ngroups, group_k_h=8)

            # Interleave: [g0_main(56)|g0_high(8)|g1_main(56)|g1_high(8)|...]
            x_dq = torch.cat([x_dq_m, x_dq_h], dim=-1)  # (batch, seq, ngroups, 64)

            # Reconstruct weight in same interleaved layout
            group_k_h = q_x_h_raw.shape[-1]
            q_w_m_g = q_w_m.reshape(N, ngroups, group_k_m)  # (N, 32, 56)
            q_w_h_g = q_w_h.reshape(N, ngroups, group_k_h)  # (N, 32, 8)
            W_dq_g = torch.cat([
                s_w_m * q_w_m_g,  # main groups dequant
                s_w_h * q_w_h_g,  # high groups dequant
            ], dim=-1)  # (N, 32, 64)
            W_dq = W_dq_g.reshape(N, -1)  # (N, 2048) interleaved
        else:
            x_dq = x_dq_m
            W_dq = (s_w_m * q_w_m).reshape(N, -1)

        x_dq = x_dq.reshape(M, -1)  # (M, K)
        y = x_dq @ W_dq.T

        return y.half().reshape(batch, seq, N)
        y = x_dq.reshape(M, -1) @ W_dq.T

        return y.half().reshape(batch, seq, N)



# ============================================================
# Test definitions
# ============================================================

ALL_TESTS = [
    ('FP16 baseline', 'fp16'),
    ('Real (fp32 acc)', 'real_fp32'),
]


# ============================================================
# Data loading
# ============================================================

def load_layer_data(layer_dir, bs_key):
    """Load all data for a single layer + batch size."""
    data = {}

    path = os.path.join(layer_dir, f'input_fp16_{bs_key}.pt')
    if os.path.exists(path):
        data['input_fp16'] = torch.load(path, map_location='cpu')

    path = os.path.join(layer_dir, 'weight_fp16.pt')
    if os.path.exists(path):
        data['weight_fp16'] = torch.load(path, map_location='cpu')

    # Column order (o_proj has reordered K dimension)
    path = os.path.join(layer_dir, 'column_order.pt')
    if os.path.exists(path):
        data['column_order'] = torch.load(path, map_location='cpu')

    for group in ['main', 'high', 'low']:
        path = os.path.join(layer_dir, f'weight_int_{group}.pt')
        if os.path.exists(path):
            data[f'weight_int_{group}'] = torch.load(path, map_location='cpu')

    for group in ['main', 'high', 'low']:
        path = os.path.join(layer_dir, f'act_quant_{group}_{bs_key}.pt')
        if os.path.exists(path):
            data[f'act_quant_{group}'] = torch.load(path, map_location='cpu')

    path = os.path.join(layer_dir, 'metadata.json')
    if os.path.exists(path):
        with open(path) as f:
            data['metadata'] = json.load(f)

    return data


# ============================================================
# Layer benchmark
# ============================================================

def bench_single_layer(layer_dir, bs_key):
    """Run all GEMM tests on a single layer for a single batch size."""
    data = load_layer_data(layer_dir, bs_key)

    if 'input_fp16' not in data or 'weight_fp16' not in data:
        return None

    x_fp16 = data['input_fp16'].cuda()
    W_fp16 = data['weight_fp16'].cuda()
    meta = data.get('metadata', {})
    N, K = W_fp16.shape

    # Apply column reorder if present (o_proj).
    # Quant data (act_quant_*, weight_int_*) are stored in reordered K order.
    # FP16 data (input_fp16, weight_fp16) are in original order.
    # Reorder FP16 data to match, so FP16 baseline is a fair comparison.
    col_order = data.get('column_order', None)
    if col_order is not None:
        x_fp16 = x_fp16[..., col_order]
        W_fp16 = W_fp16[:, col_order]

    if x_fp16.dim() > 3:
        M = x_fp16.shape[0] * x_fp16.shape[1]
    else:
        M = x_fp16.reshape(-1, x_fp16.shape[-1]).shape[0]

    results = {}

    # Reference output: FP16 matmul in (possibly reordered) space.
    # Note: x_reord @ W_reord^T == x_orig @ W_orig^T (reorder cancels in matmul)
    ref_fp16 = gemm_fp16_baseline(x_fp16, W_fp16).cuda()

    # Prepare common quant data
    act_main = data.get('act_quant_main')
    act_high = data.get('act_quant_high')
    w_main = data.get('weight_int_main')
    w_high = data.get('weight_int_high')

    for test_name, test_key in ALL_TESTS:
        result = {'accuracy_vs_fp16': None, 'perf': None}

        try:
            if test_key == 'fp16':
                y = gemm_fp16_baseline(x_fp16, W_fp16)
                perf = compute_perf_metrics(
                    lambda: gemm_fp16_baseline(x_fp16, W_fp16), M, N, K)

            elif test_key == 'real_fp32':
                if act_main is None or w_main is None:
                    result['error'] = 'missing quant data'
                    results[test_name] = result
                    continue

                y = gemm_real_quant(act_main, act_high, w_main, w_high, column_order=col_order)
                perf = compute_perf_metrics(
                    lambda: gemm_real_quant(act_main, act_high, w_main, w_high, column_order=col_order),
                    M, N, K, warmup=5, repeat=20)

            else:
                continue

            y = y.cuda()
            result['accuracy_vs_fp16'] = compute_accuracy_metrics(y, ref_fp16)
            result['perf'] = perf

        except Exception as e:
            import traceback
            result['error'] = str(e)
            traceback.print_exc()

        results[test_name] = result

    # Speedup calculation
    fp16_lat = results.get('FP16 baseline', {}).get('perf', {}).get('latency_ms', 1)
    for test_name in results:
        r = results[test_name]
        if isinstance(r, dict) and r.get('perf') and fp16_lat > 0:
            r['perf']['speedup_vs_fp16'] = fp16_lat / r['perf']['latency_ms']

    results['_memory'] = {
        'weight_fp16_mb': W_fp16.numel() * 2 / 1e6,
        'weight_int8_mb': W_fp16.numel() * 1 / 1e6,
        'weight_int4_mb': W_fp16.numel() * 0.5 / 1e6,
    }
    results['_meta'] = {'M': M, 'N': N, 'K': K, 'layer': meta.get('name', '')}

    return results


# ============================================================
# Output formatting
# ============================================================

def print_results_table(layer_name, results, bs_key):
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

        acc = r.get('accuracy_vs_fp16', {})
        perf = r.get('perf', {})
        if not acc: acc = {}
        if not perf: perf = {}

        max_err = f"{acc.get('max_abs_err', 0):.6f}" if acc else "N/A"
        mae_v = f"{acc.get('mae', 0):.6f}" if acc else "N/A"
        rmse_v = f"{acc.get('rmse', 0):.6f}" if acc else "N/A"
        cos_v = f"{acc.get('cosine_sim', 0):.6f}" if acc else "N/A"
        snr_v = f"{acc.get('snr_db', 0):.1f}" if acc and acc.get('snr_db') != float('inf') else "inf"
        lat_v = f"{perf.get('latency_ms', 0):.3f}" if perf else "N/A"
        tflops_v = f"{perf.get('tflops', 0):.2f}" if perf else "N/A"
        speedup_v = f"{perf.get('speedup_vs_fp16', 0):.2f}x" if perf.get('speedup_vs_fp16') else "N/A"

        print(f"{test_name:<20} {max_err:>10} {mae_v:>10} {rmse_v:>10} {cos_v:>10} {snr_v:>8} {lat_v:>8} {tflops_v:>8} {speedup_v:>8}")

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
    parser.add_argument('--data_dir', type=str, default='./gemm_data')
    parser.add_argument('--layers', type=str, default='')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4')
    parser.add_argument('--output', type=str, default='gemm_benchmark_results.json')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    layer_filters = [x.strip() for x in args.layers.split(',')] if args.layers else []

    if not os.path.exists(args.data_dir):
        print(f"Error: {args.data_dir} not found. Run collect_gemm_data.py first.")
        return

    layer_dirs = []
    for d in sorted(os.listdir(args.data_dir)):
        full_path = os.path.join(args.data_dir, d)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'metadata.json')):
            if not layer_filters or any(f in d for f in layer_filters):
                layer_dirs.append((d, full_path))

    print(f"Found {len(layer_dirs)} layers to benchmark")

    all_results = {}
    for layer_name, layer_path in layer_dirs:
        for bs in batch_sizes:
            bs_key = f"bs{bs}"
            if not os.path.exists(os.path.join(layer_path, f'input_fp16_{bs_key}.pt')):
                continue

            print(f"\nBenchmarking: {layer_name} [{bs_key}]...")
            results = bench_single_layer(layer_path, bs_key)
            if results is not None:
                all_results[f"{layer_name}/{bs_key}"] = results
                print_results_table(layer_name, results, bs_key)

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, float):
            if obj == float('inf'): return "inf"
            elif obj == float('-inf'): return "-inf"
            return obj
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        return obj

    with open(args.output, 'w') as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()