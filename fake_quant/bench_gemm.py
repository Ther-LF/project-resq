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

# Try to import CUTLASS GEMM extension
try:
    import resq_gemm
    HAS_CUTLASS = True
    print("CUTLASS GEMM extension loaded successfully")
except ImportError:
    HAS_CUTLASS = False
    print("WARNING: resq_gemm not found. CUTLASS tests will be skipped.")
    print("  Build with: cd csrc && python setup.py install")


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


def _center_and_flatten(act_quant):
    """Get centered integer activations as flat (M, K_group) and per-token scale (M, 1)."""
    q = act_quant['q_int'].cuda().float()
    s = act_quant['scale'].cuda().float()
    z = act_quant.get('zero', None)
    if z is not None:
        z = z.cuda().float()
        q = q - z

    grouped = (s.dim() > 3)  # (batch, seq, ngroups, 1) = grouped

    if grouped:
        # Return as-is for grouped handling
        return q, s, True
    else:
        # Per-token: flatten to (M, K_group), scale to (M, 1)
        s_per_token = _extract_per_token_scale(s)
        q_flat = q.reshape(-1, q.shape[-1])
        s_flat = s_per_token.reshape(-1, 1)
        return q_flat, s_flat, False


def _pack_int4(vals):
    """Pack int4 values (range [-8,7]) into uint8: two values per byte.

    vals: (M, K) tensor with int4 values
    Returns: (M, K//2) uint8 tensor, CUTLASS int4b_t packing (low nibble first)
    """
    assert vals.shape[-1] % 2 == 0, "K must be even for int4 packing"
    v = vals.to(torch.int8)
    lo = v[..., 0::2] & 0xF  # even indices -> low nibble
    hi = v[..., 1::2] & 0xF  # odd indices -> high nibble
    packed = (hi << 4) | lo
    return packed.to(torch.uint8)


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


def _real_quant_single_group(q_x_flat, s_x_flat, q_w, s_w, use_cutlass=False):
    """Single-group real quant matmul: y = s_x * s_w * (q_x @ q_w^T).

    q_x_flat: (M, K) centered int values as float
    s_x_flat: (M, 1) per-token scale
    q_w: (N, K) centered int values as half
    s_w: (N, 1) per-channel weight scale
    """
    M, K = q_x_flat.shape
    N = q_w.shape[0]

    if use_cutlass and HAS_CUTLASS:
        # Determine which CUTLASS kernel to use based on value range
        val_max = max(q_x_flat.abs().max().item(), q_w.abs().max().item())
        if val_max <= 7:
            # INT4 range: use INT4 GEMM
            q_x_i4 = _pack_int4(q_x_flat)
            q_w_i4 = _pack_int4(q_w)
            y_int = resq_gemm.gemm_s4s4s32(q_x_i4, q_w_i4, K).float()
        else:
            # INT8 range: use INT8 GEMM
            q_x_i8 = q_x_flat.to(torch.int8).contiguous()
            q_w_i8 = q_w.to(torch.int8).contiguous()
            y_int = resq_gemm.gemm_s8s8s32(q_x_i8, q_w_i8).float()
    else:
        # Fallback: fp32 matmul
        y_int = q_x_flat.float() @ q_w.float().T

    # Apply scales: (M,1) * (1,N) * (M,N)
    y = s_x_flat * s_w.flatten().unsqueeze(0) * y_int
    return y


def gemm_real_quant(act_quant_main, act_quant_high,
                    weight_int_main, weight_int_high,
                    use_cutlass=False):
    """Real quant GEMM with per-token or per-group scale handling."""
    q_w_m = weight_int_main['q_int'].cuda()
    s_w_m = weight_int_main['scale'].cuda()

    q_w_h = weight_int_high['q_int'].cuda() if weight_int_high else None
    s_w_h = weight_int_high['scale'].cuda() if weight_int_high else None

    q_x_m, s_x_m, grouped = _center_and_flatten(act_quant_main)
    N = q_w_m.shape[0]

    if not grouped:
        # === Per-token scale: single matmul per precision group ===
        y_m = _real_quant_single_group(q_x_m, s_x_m, q_w_m, s_w_m, use_cutlass)

        y_h = torch.zeros_like(y_m)
        if act_quant_high is not None and q_w_h is not None:
            q_x_h, s_x_h, _ = _center_and_flatten(act_quant_high)
            y_h = _real_quant_single_group(q_x_h, s_x_h, q_w_h, s_w_h, use_cutlass)

        y = (y_m + y_h).half()
        return y

    else:
        # === Per-group scale: ngroups small GEMMs ===
        batch, seq, ngroups, group_k_m = q_x_m.shape
        M = batch * seq

        q_x_m_flat = q_x_m.reshape(M, ngroups, group_k_m)
        s_x_m_flat = s_x_m.reshape(M, ngroups, 1)
        q_w_m_grouped = q_w_m.reshape(N, ngroups, group_k_m)

        y = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        for g in range(ngroups):
            q_x_g = q_x_m_flat[:, g, :].contiguous()
            q_w_g = q_w_m_grouped[:, g, :].contiguous()
            s_x_g = s_x_m_flat[:, g, :]
            y += _real_quant_single_group(q_x_g, s_x_g, q_w_g, s_w_m, use_cutlass)

        # High group
        if act_quant_high is not None and q_w_h is not None:
            q_x_h, s_x_h, _ = _center_and_flatten(act_quant_high)
            group_k_h = q_x_h.shape[-1]
            q_x_h_flat = q_x_h.reshape(M, ngroups, group_k_h)
            s_x_h_flat = s_x_h.reshape(M, ngroups, 1)
            q_w_h_grouped = q_w_h.reshape(N, ngroups, group_k_h)

            for g in range(ngroups):
                q_x_g = q_x_h_flat[:, g, :].contiguous()
                q_w_g = q_w_h_grouped[:, g, :].contiguous()
                s_x_g = s_x_h_flat[:, g, :]
                y += _real_quant_single_group(q_x_g, s_x_g, q_w_g, s_w_h, use_cutlass)

        return y.half().reshape(batch, seq, N)


def gemm_cutlass_fp16(x_fp16, W_fp16):
    """CUTLASS FP16 x FP16 -> FP32 GEMM."""
    if x_fp16.dim() > 3:
        batch, seq = x_fp16.shape[0], x_fp16.shape[1]
        x_fp16 = x_fp16.reshape(batch, seq, -1)
    x_flat = x_fp16.reshape(-1, x_fp16.shape[-1]).contiguous()
    W = W_fp16.contiguous()
    y = resq_gemm.gemm_f16f16f32(x_flat.half(), W.half())  # returns fp32
    return y.half().reshape(*x_fp16.shape[:-1], -1)


# ============================================================
# Test definitions
# ============================================================

ALL_TESTS = [
    ('FP16 baseline', 'fp16'),
    ('Fake quant', 'fake'),
    ('Real (fp32 acc)', 'real_fp32'),
    ('CUTLASS INT GEMM', 'cutlass_int'),
    ('CUTLASS FP16', 'cutlass_fp16'),
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

    for group in ['main', 'high', 'low']:
        path = os.path.join(layer_dir, f'weight_int_{group}.pt')
        if os.path.exists(path):
            data[f'weight_int_{group}'] = torch.load(path, map_location='cpu')

    for group in ['main', 'high', 'low']:
        path = os.path.join(layer_dir, f'act_quant_{group}_{bs_key}.pt')
        if os.path.exists(path):
            data[f'act_quant_{group}'] = torch.load(path, map_location='cpu')

    for kind in ['fp16_baseline', 'fake_quant', 'real_quant']:
        path = os.path.join(layer_dir, f'output_{kind}_{bs_key}.pt')
        if os.path.exists(path):
            data[f'output_{kind}'] = torch.load(path, map_location='cpu')

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

    if x_fp16.dim() > 3:
        M = x_fp16.shape[0] * x_fp16.shape[1]
    else:
        M = x_fp16.reshape(-1, x_fp16.shape[-1]).shape[0]

    results = {}

    # Reference outputs
    ref_fp16 = data.get('output_fp16_baseline',
                        gemm_fp16_baseline(x_fp16, W_fp16).cpu()).cuda()
    ref_fake = data.get('output_fake_quant', ref_fp16).cuda()

    # Prepare common quant data
    act_main = data.get('act_quant_main')
    act_high = data.get('act_quant_high')
    w_main = data.get('weight_int_main')
    w_high = data.get('weight_int_high')

    for test_name, test_key in ALL_TESTS:
        result = {'accuracy_vs_fp16': None, 'accuracy_vs_fake': None, 'perf': None}

        try:
            if test_key == 'fp16':
                y = gemm_fp16_baseline(x_fp16, W_fp16)
                perf = compute_perf_metrics(
                    lambda: gemm_fp16_baseline(x_fp16, W_fp16), M, N, K)

            elif test_key == 'fake':
                y = ref_fake.clone()
                perf = compute_perf_metrics(
                    lambda: gemm_fp16_baseline(x_fp16, W_fp16), M, N, K)

            elif test_key == 'real_fp32':
                if act_main is None or w_main is None:
                    result['error'] = 'missing quant data'
                    results[test_name] = result
                    continue

                y = gemm_real_quant(act_main, act_high, w_main, w_high,
                                    use_cutlass=False)
                perf = compute_perf_metrics(
                    lambda: gemm_real_quant(act_main, act_high, w_main, w_high,
                                           use_cutlass=False),
                    M, N, K, warmup=5, repeat=20)

            elif test_key == 'cutlass_int':
                if not HAS_CUTLASS:
                    result['error'] = 'resq_gemm not installed'
                    results[test_name] = result
                    continue
                if act_main is None or w_main is None:
                    result['error'] = 'missing quant data'
                    results[test_name] = result
                    continue

                y = gemm_real_quant(act_main, act_high, w_main, w_high,
                                    use_cutlass=True)
                perf = compute_perf_metrics(
                    lambda: gemm_real_quant(act_main, act_high, w_main, w_high,
                                           use_cutlass=True),
                    M, N, K, warmup=5, repeat=20)

            elif test_key == 'cutlass_fp16':
                if not HAS_CUTLASS:
                    result['error'] = 'resq_gemm not installed'
                    results[test_name] = result
                    continue

                y = gemm_cutlass_fp16(x_fp16, W_fp16)
                perf = compute_perf_metrics(
                    lambda: gemm_cutlass_fp16(x_fp16, W_fp16), M, N, K)

            else:
                continue

            y = y.cuda()
            result['accuracy_vs_fp16'] = compute_accuracy_metrics(y, ref_fp16)
            result['accuracy_vs_fake'] = compute_accuracy_metrics(y, ref_fake)
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

        acc = r.get('accuracy_vs_fake', r.get('accuracy_vs_fp16', {}))
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
