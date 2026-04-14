"""Triton GEMM kernels for ResQ mixed-precision quantized inference.

Kernels:
  1. _scaled_int_gemm_kernel - INT8×INT8→INT32 with per-group/per-token act scale
     Per-group: K dimension padded so each BLOCK_K tile falls within one group,
     enabling INT8 tensor core for every tile (no fp32 dequant fallback).
  2. _fp16_gemm_kernel - FP16 baseline

High-level APIs:
  - scaled_int_gemm()  - Padded INT8 GEMM (handles per-group padding automatically)
  - triton_fp16_gemm() - FP16 × FP16 → FP32
"""

import torch
import triton
import triton.language as tl

# Alignment for per-group K padding (must be >= max BLOCK_K in autotune configs)
_K_ALIGN = 64


# ============================================================
# 1. Scaled INT8 × INT8 GEMM  (per-token and per-group)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _scaled_int_gemm_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    scale_a_ptr, scale_b_ptr,
    # Matrix dims
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sa_m, stride_sa_g,
    stride_sb_n,
    # Grouping
    QUANT_GROUP_SIZE: tl.constexpr,  # -1 for per-token, >0 for per-group (padded)
    # Tile sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """INT8 × INT8 → INT32 GEMM with per-group or per-token activation scale.

    A: (M, K) int8, B: (N, K) int8  →  C = A @ B^T
    scale_a: (M, num_groups) or (M,) float32
    scale_b: (N,) float32

    Per-group: QUANT_GROUP_SIZE > 0. K must be padded so that BLOCK_K divides
    QUANT_GROUP_SIZE. Then each tile's k-range falls within one group, and we
    can use INT8 tensor core dot + apply the group's scale as a scalar.

    Per-token: QUANT_GROUP_SIZE == -1. Pure INT8 dot, scale applied at the end.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    sb = tl.load(scale_b_ptr + offs_n * stride_sb_n, mask=offs_n < N, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (k_offs[None, :] < K), other=0)

        # INT8 tensor core: int8 × int8 → int32 (exact)
        dot_int32 = tl.dot(a, tl.trans(b), out_dtype=tl.int32)
        dot_f32 = dot_int32.to(tl.float32)

        if QUANT_GROUP_SIZE > 0:
            # Per-group: this tile is within one group (guaranteed by K padding)
            group_idx = k_start // QUANT_GROUP_SIZE
            sa = tl.load(scale_a_ptr + offs_m * stride_sa_m + group_idx * stride_sa_g,
                         mask=offs_m < M, other=1.0)
            acc += sa[:, None] * dot_f32
        else:
            acc += dot_f32

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if QUANT_GROUP_SIZE > 0:
        acc = acc * sb[None, :]
    else:
        sa = tl.load(scale_a_ptr + offs_m * stride_sa_m, mask=offs_m < M, other=1.0)
        acc = sa[:, None] * sb[None, :] * acc

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def _pad_groups(q, group_size):
    """Pad K dimension so each group is a multiple of _K_ALIGN.

    q: (rows, ngroups * group_k) or (rows, K)
    Returns: (rows, ngroups * padded_gk), padded_gk
    """
    rows, K = q.shape
    ngroups = K // group_size
    padded_gk = ((group_size + _K_ALIGN - 1) // _K_ALIGN) * _K_ALIGN
    if padded_gk == group_size:
        return q, group_size  # no padding needed
    # Reshape to (rows, ngroups, group_k), pad, flatten
    q_3d = q.reshape(rows, ngroups, group_size)
    pad_width = padded_gk - group_size
    q_padded = torch.nn.functional.pad(q_3d, (0, pad_width), value=0)  # pad last dim
    return q_padded.reshape(rows, ngroups * padded_gk), padded_gk


def scaled_int_gemm(q_act, q_weight, scale_act, scale_weight, group_size=-1):
    """Scaled integer GEMM with INT8 tensor core.

    For per-group: automatically pads K so BLOCK_K tiles align with groups.

    Args:
        q_act:    (M, K) int8 centered activations
        q_weight: (N, K) int8 centered weights
        scale_act:    (M, 1)/(M,) per-token, or (M, num_groups) per-group
        scale_weight: (N, 1)/(N,) per-channel
        group_size:   -1 per-token, >0 per-group (original, before padding)
    Returns:
        (M, N) float32
    """
    assert q_act.dim() == 2 and q_weight.dim() == 2
    M, K = q_act.shape
    N = q_weight.shape[0]
    assert q_weight.shape[1] == K

    q_act = q_act.to(torch.int8).contiguous()
    q_weight = q_weight.to(torch.int8).contiguous()

    scale_act = scale_act.float().contiguous()
    scale_weight = scale_weight.float().flatten().contiguous()
    assert scale_weight.shape[0] == N

    if group_size > 0:
        # Pad groups for BLOCK_K alignment
        q_act, padded_gk = _pad_groups(q_act, group_size)
        q_weight, _ = _pad_groups(q_weight, group_size)
        K = q_act.shape[1]  # updated K after padding

        if scale_act.dim() == 1:
            scale_act = scale_act.unsqueeze(1)
        assert scale_act.shape[0] == M
        stride_sa_m = scale_act.stride(0)
        stride_sa_g = scale_act.stride(1) if scale_act.dim() > 1 else 0
        qgs = padded_gk
    else:
        if scale_act.dim() == 2:
            scale_act = scale_act[:, 0]
        scale_act = scale_act.flatten().contiguous()
        assert scale_act.shape[0] == M
        stride_sa_m = 1
        stride_sa_g = 0
        qgs = -1

    C = torch.empty((M, N), device=q_act.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _scaled_int_gemm_kernel[grid](
        q_act, q_weight, C,
        scale_act, scale_weight,
        M, N, K,
        q_act.stride(0), q_act.stride(1),
        q_weight.stride(0), q_weight.stride(1),
        C.stride(0), C.stride(1),
        stride_sa_m, stride_sa_g,
        scale_weight.stride(0),
        QUANT_GROUP_SIZE=qgs,
    )
    return C


# ============================================================
# 2. FP16 × FP16 → FP32 Triton baseline
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp16_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """FP16 GEMM: C = A @ B^T, with FP32 accumulation."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (k_offs[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_fp16_gemm(A, B):
    """Triton FP16 × FP16 → FP32 GEMM.  C = A @ B^T."""
    assert A.dim() == 2 and B.dim() == 2
    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K

    A = A.half().contiguous()
    B = B.half().contiguous()
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    _fp16_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
