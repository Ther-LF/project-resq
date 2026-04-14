"""Triton GEMM kernels for ResQ mixed-precision quantized inference.

Provides scaled integer GEMM where activation has per-token or per-group scale
and weight has per-channel scale. Dequantization happens inside the kernel's
tiling loop, so we don't need to split into small GEMMs for grouped layers.

Kernels:
  1. matmul_s8s8s32_scaled  - INT8 act × INT8 weight, per-group act scale + per-channel weight scale
  2. matmul_s4s4s32_scaled  - INT4 (packed) act × INT4 weight, same scale support
  3. matmul_f16f16f32       - FP16 baseline with Triton auto-tune

All kernels compute: y[m,n] = Σ_k scale_a[m, k//G] * scale_w[n] * q_a[m,k] * q_w[n,k]
which is mathematically equivalent to: Y = diag(S_a_per_group) @ Q_a @ Q_w^T @ diag(S_w)

Usage:
    from triton_gemm import scaled_int_gemm, triton_fp16_gemm

    # INT8 with per-group act scale (e.g., o_proj with groupsize=64)
    y = scaled_int_gemm(q_act_int8, q_weight_int8, scale_act, scale_weight, group_size=64)

    # INT8 with per-token act scale (e.g., q_proj, groupsize=-1)
    y = scaled_int_gemm(q_act_int8, q_weight_int8, scale_act, scale_weight, group_size=-1)
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 1. Scaled INT8 × INT8 GEMM with per-group activation scale
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
    stride_sa_m, stride_sa_g,   # scale_a: (M, num_groups) or (M, 1)
    stride_sb_n,                # scale_b: (N,) or (N, 1)
    # Grouping
    QUANT_GROUP_SIZE: tl.constexpr,  # -1 for per-token, >0 for per-group
    # Tile sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Scaled integer GEMM: C[m,n] = Σ_k scale_a[m, k//G] * scale_b[n] * A[m,k] * B[n,k]

    A: (M, K) int8, row-major
    B: (N, K) int8, row-major  (C = A @ B^T)
    scale_a: (M, num_groups) float32, per-group activation scale
    scale_b: (N,) float32, per-channel weight scale
    C: (M, N) float32 output
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Swizzle for better L2 cache hit
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    # Load weight scale (per-channel, constant for this N-block)
    sb = tl.load(scale_b_ptr + offs_n * stride_sb_n, mask=offs_n < N, other=0.0)

    # Main GEMM loop over K tiles
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        # Load A tile (int8)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0)
        # Load B tile (int8)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (k_offs[None, :] < K), other=0)

        # Integer dot product: int8 × int8 → int32 (exact, uses tensor core)
        dot_int32 = tl.dot(a, tl.trans(b), out_dtype=tl.int32)
        dot_f32 = dot_int32.to(tl.float32)

        if QUANT_GROUP_SIZE > 0:
            # Per-group scale: each K-tile corresponds to one activation group
            # (assumes BLOCK_K == QUANT_GROUP_SIZE for best results)
            group_idx = k_start // QUANT_GROUP_SIZE
            sa = tl.load(scale_a_ptr + offs_m * stride_sa_m + group_idx * stride_sa_g,
                         mask=offs_m < M, other=1.0)
            # Apply per-group act scale: (BLOCK_M, 1) * (BLOCK_M, BLOCK_N)
            acc += sa[:, None] * dot_f32
        else:
            # Per-token scale: same scale for all K, apply once at the end
            acc += dot_f32

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if QUANT_GROUP_SIZE > 0:
        # Weight scale applied after accumulation: (1, BLOCK_N) * (BLOCK_M, BLOCK_N)
        acc = acc * sb[None, :]
    else:
        # Per-token: apply both act scale and weight scale
        sa = tl.load(scale_a_ptr + offs_m * stride_sa_m,
                     mask=offs_m < M, other=1.0)
        acc = sa[:, None] * sb[None, :] * acc

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def scaled_int_gemm(q_act, q_weight, scale_act, scale_weight, group_size=-1):
    """Scaled integer GEMM: Y = dequant(q_act, scale_act) @ dequant(q_weight, scale_weight)^T

    Args:
        q_act: (M, K) int8 quantized activation (centered integers)
        q_weight: (N, K) int8 quantized weight (centered integers, stored as half->int8)
        scale_act: (M, 1) for per-token, or (M, num_groups) for per-group
        scale_weight: (N, 1) or (N,) per-channel weight scale
        group_size: -1 for per-token, >0 for per-group (e.g., 64)

    Returns:
        Y: (M, N) float32 output
    """
    assert q_act.dim() == 2 and q_weight.dim() == 2
    M, K = q_act.shape
    N = q_weight.shape[0]
    assert q_weight.shape[1] == K

    # Ensure int8
    q_act = q_act.to(torch.int8).contiguous()
    q_weight = q_weight.to(torch.int8).contiguous()

    # Prepare scales
    scale_act = scale_act.float().contiguous()
    scale_weight = scale_weight.float().flatten().contiguous()
    assert scale_weight.shape[0] == N

    if group_size > 0:
        if scale_act.dim() == 1:
            scale_act = scale_act.unsqueeze(1)
        assert scale_act.shape[0] == M
        stride_sa_m = scale_act.stride(0)
        stride_sa_g = scale_act.stride(1) if scale_act.dim() > 1 else 0
    else:
        if scale_act.dim() == 2:
            scale_act = scale_act[:, 0]  # per-token: just one value per row
        scale_act = scale_act.flatten().contiguous()
        assert scale_act.shape[0] == M
        stride_sa_m = 1
        stride_sa_g = 0

    # Output
    C = torch.empty((M, N), device=q_act.device, dtype=torch.float32)

    # Grid
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
        QUANT_GROUP_SIZE=group_size if group_size > 0 else -1,
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
    """Triton FP16 × FP16 → FP32 GEMM. C = A @ B^T.

    A: (M, K) fp16, B: (N, K) fp16 -> C: (M, N) fp32
    """
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


# ============================================================
# 3. High-level API for ResQ benchmark
# ============================================================

def resq_mixed_precision_gemm(
    q_act_main, scale_act_main,
    q_act_high, scale_act_high,
    q_weight_main, scale_weight_main,
    q_weight_high, scale_weight_high,
    group_size=-1,
):
    """Full ResQ mixed-precision GEMM: main (4-bit) + high (8-bit).

    Y = scaled_gemm(q_act_main, q_w_main, s_a_main, s_w_main)
      + scaled_gemm(q_act_high, q_w_high, s_a_high, s_w_high)

    For per-group (o_proj): group_size=64, scale_act has shape (M, num_groups)
    For per-token (q_proj): group_size=-1, scale_act has shape (M, 1) or (M,)
    """
    y = scaled_int_gemm(q_act_main, q_weight_main,
                        scale_act_main, scale_weight_main, group_size)

    if q_act_high is not None and q_weight_high is not None:
        y_h = scaled_int_gemm(q_act_high, q_weight_high,
                              scale_act_high, scale_weight_high, group_size)
        y = y + y_h

    return y
