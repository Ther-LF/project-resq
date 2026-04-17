#!/usr/bin/env python3
"""GEMM Benchmark: test different GEMM implementations against collected ground truth.

Loads data from gemm_data/ directory (produced by collect_gemm_data.py),
runs GEMM tests, computes accuracy and performance metrics, outputs table + JSON.

Timing convention: measure the full "module replacement" path —
from quantized int inputs to fp16 output (including dequant/scale/bias),
but NOT including data transfer, weight preprocessing, or activation quantization.

Usage:
    python bench_gemm.py --data_dir ./gemm_data [--layers q_proj,o_proj,gate_proj,down_proj] [--batch_sizes 1,2,4]
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

# CUTLASS INT8 TC kernel
try:
    import sys as _sys
    _csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')
    if _csrc_dir not in _sys.path:
        _sys.path.insert(0, _csrc_dir)
    import resq_gemm_v2
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("WARNING: resq_gemm_v2 not available — CUTLASS INT8 tests will be skipped")


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


def _pad_k_to_multiple(t, multiple=16):
    """Pad last dim of tensor to nearest multiple (for CUTLASS alignment)."""
    K = t.shape[-1]
    pad_k = (multiple - K % multiple) % multiple
    if pad_k == 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[-1] = pad_k
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=-1)


# ============================================================
# Data preparation — all preprocessing done ONCE, outside timing
# ============================================================

def prepare_per_token_data(act_quant_main, act_quant_high,
                           weight_int_main, weight_int_high):
    """Prepare all GPU tensors for per-token layers. Called once, outside timing.

    Returns a dict with everything needed by the compute functions,
    all on GPU, all in the right dtype.
    """
    d = {}

    # Weight (static, precomputed)
    q_w_m = weight_int_main['q_int'].cuda().float()
    s_w_m = weight_int_main['scale'].cuda().float()  # (N, 1)
    d['q_w_m'] = q_w_m
    d['q_w_m_int8'] = q_w_m.to(torch.int8).contiguous()
    d['s_w_m'] = s_w_m.flatten().unsqueeze(0)         # (1, N)
    d['w_colsum_m'] = d['q_w_m_int8'].float().sum(dim=1, keepdim=True).T  # (1, N)

    N = q_w_m.shape[0]
    d['N'] = N

    # Activation main
    q_x_m_raw = act_quant_main['q_int'].cuda().float()
    s_x_m = act_quant_main['scale'].cuda().float()
    z_x_m = act_quant_main.get('zero', None)
    if z_x_m is not None:
        z_x_m = z_x_m.cuda().float()
    else:
        z_x_m = torch.zeros_like(q_x_m_raw[..., :1])

    s_x_tok = _extract_per_token_scale(s_x_m).reshape(-1, 1)
    z_x_tok = _extract_per_token_scale(z_x_m).reshape(-1, 1) if z_x_m.shape[-1] > 1 else z_x_m.reshape(-1, 1)
    q_x_flat = q_x_m_raw.reshape(-1, q_x_m_raw.shape[-1])

    d['q_x_m_flat'] = q_x_flat                                # (M, K_m) float
    d['q_x_m_shifted_f32'] = q_x_flat - 8.0                   # (M, K_m) float, for B2
    d['q_x_m_shifted_i8'] = (q_x_flat - 8.0).to(torch.int8).contiguous()  # (M, K_m) int8, for B3
    d['s_x_m'] = s_x_tok                                       # (M, 1)
    d['z_x_m'] = z_x_tok                                       # (M, 1)
    d['shift_minus_z_m'] = 8.0 - z_x_tok                      # (M, 1) precomputed
    d['M'] = q_x_flat.shape[0]

    # High group
    d['has_high'] = (act_quant_high is not None and weight_int_high is not None)
    if d['has_high']:
        q_w_h = weight_int_high['q_int'].cuda().float()
        s_w_h = weight_int_high['scale'].cuda().float()
        d['q_w_h'] = q_w_h
        d['q_w_h_int8'] = q_w_h.to(torch.int8).contiguous()
        d['s_w_h'] = s_w_h.flatten().unsqueeze(0)
        d['w_colsum_h'] = d['q_w_h_int8'].float().sum(dim=1, keepdim=True).T

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

        d['q_x_h_flat'] = q_x_h_flat
        d['q_x_h_shifted_f32'] = q_x_h_flat - 128.0
        d['q_x_h_shifted_i8'] = (q_x_h_flat - 128.0).to(torch.int8).contiguous()
        d['s_x_h'] = s_x_h_tok
        d['z_x_h'] = z_x_h_tok
        d['shift_minus_z_h'] = 128.0 - z_x_h_tok

    return d


def prepare_grouped_data(act_quant_main, act_quant_high,
                         weight_int_main, weight_int_high):
    """Prepare all GPU tensors for per-group (o_proj) layers. Called once, outside timing.

    Returns a dict with everything needed, all on GPU, precomputed.
    """
    d = {}

    q_w_m = weight_int_main['q_int'].cuda().float()
    s_w_m = weight_int_main['scale'].cuda().float()
    N = q_w_m.shape[0]
    d['N'] = N
    d['s_w_m'] = s_w_m.flatten().unsqueeze(0)  # (1, N)

    q_x_m_raw = act_quant_main['q_int'].cuda().float()  # (B, S, G, K_g)
    s_x_m = act_quant_main['scale'].cuda().float()       # (B, S, G, 1)
    z_x_m = act_quant_main.get('zero', None)
    if z_x_m is not None:
        z_x_m = z_x_m.cuda().float()
    else:
        z_x_m = torch.zeros_like(q_x_m_raw[..., :1])

    batch, seq, G, K_g = q_x_m_raw.shape
    M = batch * seq
    d['batch'] = batch
    d['seq'] = seq
    d['G'] = G
    d['K_g'] = K_g
    d['M'] = M

    q_x_m_flat = q_x_m_raw.reshape(M, G, K_g)
    s_x_m_flat = s_x_m.reshape(M, G, 1)
    z_x_m_flat = z_x_m.reshape(M, G, 1)

    d['q_x_m_flat'] = q_x_m_flat
    d['s_x_m_flat'] = s_x_m_flat
    d['z_x_m_flat'] = z_x_m_flat

    # Precompute per-group weight data (static)
    q_w_m_int8 = q_w_m.to(torch.int8)
    d['q_w_m_int8'] = q_w_m_int8
    d['q_w_m'] = q_w_m

    # Per-group w_colsum: (G, 1, N) — precomputed once
    w_colsums_m = []
    for g in range(G):
        q_w_g = q_w_m_int8[:, g * K_g:(g + 1) * K_g].float()  # (N, K_g)
        w_colsums_m.append(q_w_g.sum(dim=1, keepdim=True).T)    # (1, N)
    d['w_colsums_m'] = torch.stack(w_colsums_m, dim=0)  # (G, 1, N)

    # For Baseline 2: per-group fp32 weight slices (G, N, K_g)
    w_slices_m = []
    for g in range(G):
        w_slices_m.append(q_w_m[:, g * K_g:(g + 1) * K_g])
    d['w_slices_m'] = torch.stack(w_slices_m, dim=0)  # (G, N, K_g)

    # For Baseline 3: pre-shifted activation + pre-padded weight (CUTLASS input)
    q_x_shifted = (q_x_m_flat - 8.0).to(torch.int8)            # (M, G, K_g)
    q_x_shifted = q_x_shifted.permute(1, 0, 2).contiguous()     # (G, M, K_g)
    d['A_grouped_m'] = _pad_k_to_multiple(q_x_shifted, 16)      # (G, M, K_pad)

    B_groups = []
    for g in range(G):
        B_groups.append(q_w_m_int8[:, g * K_g:(g + 1) * K_g])
    B_stacked = torch.stack(B_groups, dim=0).contiguous()        # (G, N, K_g)
    d['B_grouped_m'] = _pad_k_to_multiple(B_stacked, 16)        # (G, N, K_pad)

    # High group
    d['has_high'] = (act_quant_high is not None and weight_int_high is not None)
    if d['has_high']:
        q_w_h = weight_int_high['q_int'].cuda().float()
        s_w_h = weight_int_high['scale'].cuda().float()
        d['s_w_h'] = s_w_h.flatten().unsqueeze(0)
        q_w_h_int8 = q_w_h.to(torch.int8)
        d['q_w_h_int8'] = q_w_h_int8

        q_x_h_raw = act_quant_high['q_int'].cuda().float()  # (B, S, G, K_h)
        s_x_h = act_quant_high['scale'].cuda().float()
        z_x_h = act_quant_high.get('zero', None)
        if z_x_h is not None:
            z_x_h = z_x_h.cuda().float()
        else:
            z_x_h = torch.zeros_like(q_x_h_raw[..., :1])

        K_h = q_x_h_raw.shape[-1]
        d['K_h'] = K_h
        q_x_h_flat = q_x_h_raw.reshape(M, G, K_h)
        d['q_x_h_flat'] = q_x_h_flat
        d['s_x_h_flat'] = s_x_h.reshape(M, G, 1)
        d['z_x_h_flat'] = z_x_h.reshape(M, G, 1)

        w_colsums_h = []
        for g in range(G):
            q_w_hg = q_w_h_int8[:, g * K_h:(g + 1) * K_h].float()
            w_colsums_h.append(q_w_hg.sum(dim=1, keepdim=True).T)
        d['w_colsums_h'] = torch.stack(w_colsums_h, dim=0)  # (G, 1, N)

        w_slices_h = []
        for g in range(G):
            w_slices_h.append(q_w_h[:, g * K_h:(g + 1) * K_h])
        d['w_slices_h'] = torch.stack(w_slices_h, dim=0)

        q_x_h_shifted = (q_x_h_flat - 128.0).to(torch.int8)
        q_x_h_shifted = q_x_h_shifted.permute(1, 0, 2).contiguous()
        d['A_grouped_h'] = _pad_k_to_multiple(q_x_h_shifted, 16)

        B_h_groups = []
        for g in range(G):
            B_h_groups.append(q_w_h_int8[:, g * K_h:(g + 1) * K_h])
        B_h_stacked = torch.stack(B_h_groups, dim=0).contiguous()
        d['B_grouped_h'] = _pad_k_to_multiple(B_h_stacked, 16)

    return d


# ============================================================
# GEMM implementations — ONLY compute, all inputs already on GPU
# ============================================================

def gemm_fp16(x_fp16, W_fp16):
    """Baseline 1: FP16 matmul. Input fp16, output fp16."""
    x_flat = x_fp16.reshape(-1, x_fp16.shape[-1])
    return (x_flat @ W_fp16.T).half()


# --- Baseline 2: fp32 simulation of INT GEMM ---

def gemm_b2_per_token(d):
    """Baseline 2, per-token: fp32-simulated INT GEMM + dequant → fp16.

    Timed operations:
      1. fp32 matmul: q_x_shifted @ q_w^T                (simulates INT TC)
      2. bias:        (shift - zero) * w_colsum            (broadcast)
      3. scale:       s_x * s_w * (matmul + bias)          (elementwise)
      4. .half()                                            (fp16 output)
    """
    # Main group
    y_int_m = d['q_x_m_shifted_f32'] @ d['q_w_m'].T           # fp32 matmul
    bias_m = d['shift_minus_z_m'] * d['w_colsum_m']             # broadcast, not @
    Y_m = d['s_x_m'] * d['s_w_m'] * (y_int_m + bias_m)

    # High group
    if d['has_high']:
        y_int_h = d['q_x_h_shifted_f32'] @ d['q_w_h'].T
        bias_h = d['shift_minus_z_h'] * d['w_colsum_h']
        Y_h = d['s_x_h'] * d['s_w_h'] * (y_int_h + bias_h)
        return (Y_m + Y_h).half()

    return Y_m.half()


def gemm_b2_grouped(d):
    """Baseline 2, grouped (o_proj): fp32-simulated INT GEMM + dequant → fp16.

    Timed operations (per group × G groups):
      1. fp32 matmul: q_x_g @ q_w_g^T                  (simulates INT TC)
      2. bias:        (8 - z_g) * w_colsum_g             (broadcast)
      3. scale:       s_x_g * s_w * (matmul + bias)      (elementwise)
      4. accumulate across groups
      5. .half()
    """
    M, G, K_g = d['q_x_m_flat'].shape
    N = d['N']
    s_w_m = d['s_w_m']
    y = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    for g in range(G):
        q_x_g = d['q_x_m_flat'][:, g, :] - 8.0                # (M, K_g) shifted
        q_w_g = d['w_slices_m'][g]                               # (N, K_g)
        y_int = q_x_g @ q_w_g.T                                  # fp32 matmul
        bias_g = (8.0 - d['z_x_m_flat'][:, g, :]) * d['w_colsums_m'][g]  # broadcast
        y += d['s_x_m_flat'][:, g, :] * s_w_m * (y_int + bias_g)

    if d['has_high']:
        s_w_h = d['s_w_h']
        K_h = d['K_h']
        for g in range(G):
            q_x_g = d['q_x_h_flat'][:, g, :] - 128.0
            q_w_g = d['w_slices_h'][g]
            y_int = q_x_g @ q_w_g.T
            bias_g = (128.0 - d['z_x_h_flat'][:, g, :]) * d['w_colsums_h'][g]
            y += d['s_x_h_flat'][:, g, :] * s_w_h * (y_int + bias_g)

    return y.half()


# --- Baseline 3: CUTLASS INT8 TC ---

def gemm_b3_per_token(d):
    """Baseline 3, per-token: CUTLASS INT8 TC GEMM + dequant → fp16.

    Timed operations:
      1. CUTLASS INT8 GEMM: q_x_shifted_i8 @ q_w_i8^T   (INT8 TC, INT32 acc)
      2. bias:              (shift - zero) * w_colsum      (broadcast)
      3. scale:             s_x * s_w * (D + bias)         (elementwise)
      4. .half()                                            (fp16 output)
    """
    # Main group
    D_m = resq_gemm_v2.gemm_s8s8(d['q_x_m_shifted_i8'], d['q_w_m_int8'])
    bias_m = d['shift_minus_z_m'] * d['w_colsum_m']
    Y_m = d['s_x_m'] * d['s_w_m'] * (D_m + bias_m)

    # High group
    if d['has_high']:
        D_h = resq_gemm_v2.gemm_s8s8(d['q_x_h_shifted_i8'], d['q_w_h_int8'])
        bias_h = d['shift_minus_z_h'] * d['w_colsum_h']
        Y_h = d['s_x_h'] * d['s_w_h'] * (D_h + bias_h)
        return (Y_m + Y_h).half()

    return Y_m.half()


def gemm_b3_grouped(d):
    """Baseline 3, grouped (o_proj): CUTLASS grouped INT8 TC GEMM + dequant → fp16.

    Timed operations:
      1. CUTLASS grouped GEMM: G × (M,K)×(N,K)^T          (1 kernel launch)
      2. per-group bias + scale + accumulate                (Python loop)
      3. .half()
    """
    M = d['M']
    N = d['N']
    G = d['G']
    s_w_m = d['s_w_m']

    # 1 kernel launch for all G groups
    D_m = resq_gemm_v2.grouped_gemm_s8s8(d['A_grouped_m'], d['B_grouped_m'])  # (G, M, N)

    # Post-process
    y = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    for g in range(G):
        bias_g = (8.0 - d['z_x_m_flat'][:, g, :]) * d['w_colsums_m'][g]  # broadcast
        y += d['s_x_m_flat'][:, g, :] * s_w_m * (D_m[g] + bias_g)

    if d['has_high']:
        s_w_h = d['s_w_h']
        D_h = resq_gemm_v2.grouped_gemm_s8s8(d['A_grouped_h'], d['B_grouped_h'])
        for g in range(G):
            bias_g = (128.0 - d['z_x_h_flat'][:, g, :]) * d['w_colsums_h'][g]
            y += d['s_x_h_flat'][:, g, :] * s_w_h * (D_h[g] + bias_g)

    return y.half()


# ============================================================
# Test definitions
# ============================================================

ALL_TESTS = [
    ('FP16 baseline', 'fp16'),
    ('Real (fp32 acc)', 'real_fp32'),
    ('CUTLASS INT8 TC', 'cutlass_int8'),
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

    # Apply column reorder to activation if present (o_proj).
    col_order = data.get('column_order', None)
    if col_order is not None:
        x_fp16 = x_fp16[..., col_order]

    if x_fp16.dim() > 3:
        M = x_fp16.shape[0] * x_fp16.shape[1]
    else:
        M = x_fp16.reshape(-1, x_fp16.shape[-1]).shape[0]

    act_main = data.get('act_quant_main')
    act_high = data.get('act_quant_high')
    w_main = data.get('weight_int_main')
    w_high = data.get('weight_int_high')

    # Detect per-group vs per-token
    grouped = False
    if act_main is not None:
        s_test = act_main['scale']
        grouped = (s_test.dim() > 3)

    # === Precompute all data ONCE, outside timing ===
    prep = None
    if act_main is not None and w_main is not None:
        if grouped:
            prep = prepare_grouped_data(act_main, act_high, w_main, w_high)
        else:
            prep = prepare_per_token_data(act_main, act_high, w_main, w_high)

    # Reference output
    ref_fp16 = gemm_fp16(x_fp16, W_fp16).cuda()

    results = {}

    for test_name, test_key in ALL_TESTS:
        result = {'accuracy_vs_fp16': None, 'perf': None}

        try:
            if test_key == 'fp16':
                y = gemm_fp16(x_fp16, W_fp16)
                perf = compute_perf_metrics(
                    lambda: gemm_fp16(x_fp16, W_fp16), M, N, K)

            elif test_key == 'real_fp32':
                if prep is None:
                    result['error'] = 'missing quant data'
                    results[test_name] = result
                    continue
                if grouped:
                    y = gemm_b2_grouped(prep)
                    perf = compute_perf_metrics(
                        lambda: gemm_b2_grouped(prep), M, N, K, warmup=5, repeat=50)
                else:
                    y = gemm_b2_per_token(prep)
                    perf = compute_perf_metrics(
                        lambda: gemm_b2_per_token(prep), M, N, K, warmup=10, repeat=100)

            elif test_key == 'cutlass_int8':
                if not HAS_CUTLASS:
                    result['error'] = 'resq_gemm_v2 not available'
                    results[test_name] = result
                    continue
                if prep is None:
                    result['error'] = 'missing quant data'
                    results[test_name] = result
                    continue
                if grouped:
                    y = gemm_b3_grouped(prep)
                    perf = compute_perf_metrics(
                        lambda: gemm_b3_grouped(prep), M, N, K, warmup=5, repeat=50)
                else:
                    y = gemm_b3_per_token(prep)
                    perf = compute_perf_metrics(
                        lambda: gemm_b3_per_token(prep), M, N, K, warmup=10, repeat=100)

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
    results['_meta'] = {
        'M': M, 'N': N, 'K': K,
        'layer': meta.get('name', ''),
        'grouped': grouped,
    }

    return results


# ============================================================
# Output formatting
# ============================================================

def print_results_table(layer_name, results, bs_key):
    meta = results.get('_meta', {})
    M, N, K = meta.get('M', '?'), meta.get('N', '?'), meta.get('K', '?')
    tag = ' [grouped]' if meta.get('grouped') else ''
    print(f"\nLayer: {meta.get('layer', layer_name)} (M={M}, N={N}, K={K}) [{bs_key}]{tag}")
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
