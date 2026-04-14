# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import transformers

from train_utils.quant_linear import QuantizeLinear
from utils import hadamard_utils
from utils.utils import HadamardTransform


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def quantize_weight_per_channel(W, bits, sym=True):
    """Per-output-channel weight quantization. Returns (q_int, scale, [zero]).

    W: (N, K) weight tensor
    Returns centered integer weights as fp16 and per-channel scale (N,1).
    """
    _, maxq = get_minq_maxq(bits, sym)
    maxq = maxq.to(W.device)

    if sym:
        # Symmetric: scale = max(|W|) / maxq per channel
        wmax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = wmax / maxq
        q = torch.clamp(torch.round(W / scale), -(maxq + 1), maxq)
        # Return centered int as fp16 (for sym, already centered at 0)
        return q, scale
    else:
        # Asymmetric: scale = (max - min) / maxq per channel
        wmax = W.amax(dim=1, keepdim=True)
        wmin = W.amin(dim=1, keepdim=True)
        tmp = (wmin == 0) & (wmax == 0)
        wmin[tmp] = -1
        wmax[tmp] = 1
        scale = (wmax - wmin) / maxq
        zero = torch.round(-wmin / scale)
        q = torch.clamp(torch.round(W / scale) + zero, 0, maxq)
        # Return centered int as fp16: (q - zero)
        q_centered = q - zero
        return q_centered, scale


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def stoch_round(tensor):
    """
    Applies stochastic rounding to the elements of the input tensor.
    """
    # Get the floor and ceil values of the tensor
    floor_values = tensor.floor()
    ceil_values = tensor.ceil()

    # Calculate the fractional part
    fractional_part = tensor - floor_values

    # Generate random numbers between 0 and 1
    random_values = torch.rand_like(tensor)

    # Determine the rounding direction based on the random values and fractional part
    rounded_tensor = torch.where(
        random_values < fractional_part, ceil_values, floor_values
    )

    return rounded_tensor


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq, stoch=False):
        scale = scale.to(x.device)
        if stoch:
            q = torch.clamp(stoch_round(x / scale), -(maxq + 1), maxq)
        else:
            q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq, stoch=False):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        if stoch:
            q = torch.clamp(stoch_round(x / scale) + zero, 0, maxq)
        else:
            q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        
        # for high prec
        self.register_buffer("maxq_h", torch.tensor(0))
        self.register_buffer("scale_h", torch.zeros(1))
        self.register_buffer("zero_h", torch.zeros(1))
        
        # for low prec
        self.register_buffer("maxq_l", torch.tensor(0))
        self.register_buffer("scale_l", torch.zeros(1))
        self.register_buffer("zero_l", torch.zeros(1))

        self.bits = 16
        self.high_bits = 16
        self.low_bits = 16

        self.high_bits_length = 0
        self.low_bits_length = 0
        

    def free(self) -> None:
        self.zero = None
        self.scale = None
        self.zero_h = None
        self.scale_h = None
        self.zero_l = None
        self.scale_l = None

    def forward(self, x):
        x_dtype = x.dtype

        if self.bits == 16:
            return x
        if self.groupsize > 0:
            init_shape = x.shape
            x = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize
            )

        low_dim, high_dim = self.low_bits_length, x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        if self.sym:
            x = STEQuantize.apply(x_m, self.scale, self.maxq)

            if self.high_bits_length != 0:
                x_h = STEQuantize.apply(x_h, self.scale_h, self.maxq_h)
                x = torch.cat([x, x_h], dim=-1).to(x_dtype)

            if self.low_bits_length != 0:
                x_l = STEQuantize.apply(x_l, self.scale_l, self.maxq_l)
                x = torch.cat([x_l, x], dim=-1).to(x_dtype)

        else:
            x = AsymSTEQuantize.apply(x_m, self.scale, self.zero, self.maxq)

            if self.high_bits_length != 0:
                x_h = AsymSTEQuantize.apply(x_h, self.scale_h, self.zero_h, self.maxq_h)
                x = torch.cat([x, x_h], dim=-1).to(x_dtype)

            if self.low_bits_length != 0:
                x_l = AsymSTEQuantize.apply(x_l, self.scale_l, self.zero_l, self.maxq_l)
                x = torch.cat([x_l, x], dim=-1).to(x_dtype)

        if self.groupsize > 0:
            x = x.reshape(init_shape)

        return x

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x, return_low=False, return_high=False):
        low_dim, high_dim = self.low_bits_length, x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        if self.sym:
            if return_low:
                return sym_quant(x_l, self.scale_l, self.maxq_l)
            elif return_high:
                return sym_quant(x_h, self.scale_h, self.maxq_h)
            else:
                return sym_quant(x_m, self.scale, self.maxq)
        else:
            if return_low:
                return asym_quant(x_l, self.scale_l, self.zero_l, self.maxq_l)
            elif return_high:
                return asym_quant(x_h, self.scale_h, self.zero_h, self.maxq_h)
            else:
                return asym_quant(x_m, self.scale, self.zero, self.maxq)

    def configure(
        self,
        bits: int,
        groupsize: int = -1,
        sym: bool = False,
        clip_ratio: float = 1.0,
        stoch: bool = False,
        high_bits_length: int = 0,
        high_bits: int = 16,
        low_bits_length: int = 0,
        low_bits: int = 16,
    ) -> None:
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.stoch = stoch
        
        self.high_bits_length = high_bits_length
        self.high_bits = high_bits
        _, self.maxq_h = get_minq_maxq(high_bits, sym)

        self.low_bits_length = low_bits_length
        self.low_bits = low_bits
        _, self.maxq_l = get_minq_maxq(low_bits, sym)

        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x, maxq):
        xmax = torch.amax(x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = xmax / maxq
            scale[tmp] = 1
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)

        return scale, zero

    def find_params(self, x, residual_dim=-1) -> None:
        if self.groupsize > 0:
            # per group mixed precision quantization
            init_shape = x.shape
            x_reshaped = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize
            )
            low_dim, high_dim = self.low_bits_length, x_reshaped.shape[-1] - self.high_bits_length
            x_l, x_m, x_h = x_reshaped[..., :low_dim], x_reshaped[..., low_dim:high_dim], x_reshaped[..., high_dim:]
            
            self.scale, self.zero = self.find_params_per_token_groupwise(x_m, self.maxq)

            if self.high_bits_length != 0:
                self.scale_h, self.zero_h = self.find_params_per_token_groupwise(x_h, self.maxq_h)

            if self.low_bits_length != 0:
                self.scale_l, self.zero_l = self.find_params_per_token_groupwise(x_l, self.maxq_l)

            return
        
        low_dim, high_dim = self.low_bits_length, x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        self.scale, self.zero = self._find_params(x_m, self.maxq)
        
        if self.high_bits_length != 0:
            self.scale_h, self.zero_h = self._find_params(x_h, self.maxq_h)
        
        if self.low_bits_length != 0:
            self.scale_l, self.zero_l = self._find_params(x_l, self.maxq_l)

        return

    def _find_params(self, x, maxq):
        if self.bits == 16:
            return

        dev = x.device

        init_shape = x.shape

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            scale[tmp] = 1
            scale = scale.reshape(init_shape)
            zero = torch.zeros_like(scale)

        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)

            scale = (
                scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            )

            zero = zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.hadK_quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.residual = 0
        self.no_had = False

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}, High Bits/Dim: {self.quantizer.high_bits}/{self.quantizer.high_bits_length}, Low Bits/Dim: {self.quantizer.low_bits}/{self.quantizer.low_bits_length}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}, High Bits/Dim: {self.out_quantizer.high_bits}/{self.out_quantizer.high_bits_length}, Low Bits/Dim: {self.out_quantizer.low_bits}/{self.out_quantizer.low_bits_length}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def forward(
        self,
        x,
        R1=None,
        R2=None,
        transpose=False,
        column_order=None,
    ):
        x_dtype = x.dtype
        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        if column_order is not None:
            x = x[..., column_order]

        if R1 is not None:
            assert column_order is None #column order only used when not training rotations
            x = self.module(
                x,
                R1,
                R2,
                transpose,
            ).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x

    def prepare_real_quant_weights(self, w_bits=None, w_quantizers=None, w_int_weights=None):
        """Pre-quantize weights for real integer GEMM inference.

        Splits weight along K dimension into precision groups matching
        the activation quantizer's configuration.

        Weight quantization strategy (matching ResQ paper):
        - main group: w_bits (default 4 with GPTQ, or 8 without)
        - high group: high_bits (8-bit)
        - low group: low_bits (2-bit)

        If w_int_weights tensor is provided (directly saved integers from GPTQ),
        uses them directly for zero-error integer weights.
        If only w_quantizers dict is provided (from GPTQ checkpoint), uses the
        exact GPTQ scale/zero for re-quantization.

        After calling this, use forward_real_quant() instead of forward().
        """
        W = self.module.weight.data  # (N, K) - already GPTQ dequant'd if w_bits<16
        K = W.shape[1]
        quantizer = self.quantizer

        if quantizer.bits >= 16:
            self._real_quant_ready = False
            return

        low_dim = quantizer.low_bits_length
        high_dim_start = K - quantizer.high_bits_length

        # Determine per-group weight precision
        W_BITS_M = w_bits if w_bits is not None else 8
        W_BITS_H = quantizer.high_bits if quantizer.high_bits_length > 0 else W_BITS_M
        W_BITS_L = quantizer.low_bits if quantizer.low_bits_length > 0 else W_BITS_M

        # Handle activation groupsize > 0 (e.g., o_proj with groupsize=64)
        # NOTE: GPTQ uses contiguous [low|main|high] layout regardless of activation groupsize.
        # The weight is already rearranged by rearrange_columns(), so globally:
        # [low_cols | main_cols | high_cols]. GPTQ quantizes in this layout.
        # We always use contiguous split for weight (matching GPTQ), not per-group split.
        if quantizer.groupsize > 0:
            N = W.shape[0]
            gs = quantizer.groupsize
            num_groups = K // gs

            # For GPTQ weights, use contiguous split matching GPTQ's layout
            # GPTQ's high_dim is based on global model_dim:
            # high_bits_length for grouped = per_group_high * num_groups
            global_high = quantizer.high_bits_length * num_groups
            global_low = quantizer.low_bits_length * num_groups
            high_dim_start_global = K - global_high

            W_m = W[:, global_low:high_dim_start_global]
            W_h = W[:, high_dim_start_global:] if global_high > 0 else None
            W_l = W[:, :global_low] if global_low > 0 else None

            if w_int_weights is not None:
                Q_int = w_int_weights.to(W.device)
                Q_m = Q_int[:, global_low:high_dim_start_global]
                Q_h = Q_int[:, high_dim_start_global:] if global_high > 0 else None
                Q_l = Q_int[:, :global_low] if global_low > 0 else None
            else:
                Q_m, Q_h, Q_l = None, None, None
        else:
            W_m = W[:, low_dim:high_dim_start]
            W_h = W[:, high_dim_start:] if quantizer.high_bits_length > 0 else None
            W_l = W[:, :low_dim] if quantizer.low_bits_length > 0 else None

            # Also split integer weights if provided
            if w_int_weights is not None:
                Q_int = w_int_weights.to(W.device)
                Q_m = Q_int[:, low_dim:high_dim_start]
                Q_h = Q_int[:, high_dim_start:] if quantizer.high_bits_length > 0 else None
                Q_l = Q_int[:, :low_dim] if quantizer.low_bits_length > 0 else None
            else:
                Q_m, Q_h, Q_l = None, None, None

        # Priority: direct int weights > GPTQ re-quantization > RTN fallback
        if w_int_weights is not None and w_quantizers is not None:
            # Best path: use direct integers from GPTQ + scale from quantizer
            scale_m = w_quantizers['main'].scale.to(W.device).half()
            self.register_buffer('W_m_int', Q_m.half())
            self.register_buffer('W_m_scale', scale_m)
            self._w_groupsize_m = W_m.shape[1]

            # DEBUG: verify reconstruction
            import logging
            W_m_recon = (Q_m.float() * scale_m.float()).half()
            diff = (W_m.float() - W_m_recon.float()).abs()
            logging.warning(f"[DEBUG prepare] W_m recon: max_diff={diff.max().item():.6f}, Q_m range=[{Q_m.min().item()}, {Q_m.max().item()}], scale_m shape={scale_m.shape}")

            if W_h is not None and 'high' in w_quantizers:
                scale_h = w_quantizers['high'].scale.to(W.device).half()
                self.register_buffer('W_h_int', Q_h.half())
                self.register_buffer('W_h_scale', scale_h)
                self._w_groupsize_h = W_h.shape[1]

            if W_l is not None and 'low' in w_quantizers:
                scale_l = w_quantizers['low'].scale.to(W.device).half()
                self.register_buffer('W_l_int', Q_l.half())
                self.register_buffer('W_l_scale', scale_l)
                self._w_groupsize_l = W_l.shape[1]

        elif w_quantizers is not None:
            # Fallback: re-quantize using GPTQ's scale (方案A, slight fp16 rounding error)
            q_m, s_m = self._requant_with_gptq_scale(W_m, w_quantizers['main'])
            self.register_buffer('W_m_int', q_m)
            self.register_buffer('W_m_scale', s_m)
            self._w_groupsize_m = W_m.shape[1]  # per-channel

            if W_h is not None and 'high' in w_quantizers:
                q_h, s_h = self._requant_with_gptq_scale(W_h, w_quantizers['high'])
                self.register_buffer('W_h_int', q_h)
                self.register_buffer('W_h_scale', s_h)
                self._w_groupsize_h = W_h.shape[1]

            if W_l is not None and 'low' in w_quantizers:
                q_l, s_l = self._requant_with_gptq_scale(W_l, w_quantizers['low'])
                self.register_buffer('W_l_int', q_l)
                self.register_buffer('W_l_scale', s_l)
                self._w_groupsize_l = W_l.shape[1]
        else:
            # No GPTQ quantizers: use per-group RTN (for w_bits=16 or standalone)
            w_groupsize = 128
            q_m, s_m = self._quantize_weight_per_group(W_m, W_BITS_M, w_groupsize)
            self.register_buffer('W_m_int', q_m)
            self.register_buffer('W_m_scale', s_m)
            self._w_groupsize_m = w_groupsize

            if W_h is not None:
                q_h, s_h = self._quantize_weight_per_group(W_h, W_BITS_H, w_groupsize)
                self.register_buffer('W_h_int', q_h)
                self.register_buffer('W_h_scale', s_h)
                self._w_groupsize_h = w_groupsize

            if W_l is not None:
                q_l, s_l = self._quantize_weight_per_group(W_l, W_BITS_L, w_groupsize)
                self.register_buffer('W_l_int', q_l)
                self.register_buffer('W_l_scale', s_l)
                self._w_groupsize_l = w_groupsize

        self._real_quant_ready = True

    @staticmethod
    def _requant_with_gptq_scale(W, wq):
        """Re-quantize using GPTQ's original scale (per-channel symmetric).

        W: (N, K) dequantized fp16 weight from GPTQ
        wq: WeightQuantizer with .scale, .zero, .maxq from GPTQ
        Returns integer values and scale matching GPTQ exactly.
        """
        scale = wq.scale.to(W.device)  # (N, 1) per-channel
        maxq = wq.maxq.to(W.device)
        if wq.sym:
            q = torch.clamp(torch.round(W / scale), -(maxq + 1), maxq)
        else:
            zero = wq.zero.to(W.device)
            q = torch.clamp(torch.round(W / scale) + zero, 0, maxq)
            q = q - zero  # center to get signed integers
        return q.half(), scale.half()

    @staticmethod
    def _quantize_weight_per_group(W, bits, groupsize):
        """Per-group symmetric weight quantization along K dimension.

        W: (N, K) weight tensor
        Returns:
            q: (N, K) centered integer values (as half for storage)
            scale: (N, num_groups) per-group scales
        """
        N, K_orig = W.shape
        _, maxq = get_minq_maxq(bits, sym=True)
        maxq = maxq.to(W.device)

        # Pad K to multiple of groupsize if needed
        if K_orig % groupsize != 0:
            pad = groupsize - K_orig % groupsize
            W = torch.nn.functional.pad(W, (0, pad))
        K = W.shape[1]
        num_groups = K // groupsize

        W_grouped = W.reshape(N, num_groups, groupsize)

        # Per-group symmetric: scale = max(|W_group|) / maxq
        wmax = W_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # (N, ng, 1)
        scale = wmax / maxq  # (N, ng, 1)

        q = torch.clamp(torch.round(W_grouped / scale), -(maxq + 1), maxq)
        q = q.reshape(N, K)[:, :K_orig]  # trim padding
        scale = scale.squeeze(2)  # (N, num_groups)

        return q.half(), scale.half()

        self._real_quant_ready = True

    def _apply_rotation(self, x):
        """Apply Hadamard rotation (extracted from forward for reuse)."""
        x_dtype = x.dtype
        if self.online_full_had:
            if self.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()
            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2)
                    ) / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)
            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)
        return x

    def forward_real_quant(
        self,
        x,
        R1=None,
        R2=None,
        transpose=False,
        column_order=None,
    ):
        """Real quantization forward: integer quantize -> integer-equivalent matmul -> dequantize.

        Strategy:
        - Quantize activation to integers (per-token or per-group)
        - Dequantize weight from stored int8 per-group representation
        - For groupsize>0: dequant activation per-group, reshape flat, single matmul
        - For groupsize=-1: split matmul by precision groups for future int kernel
        """
        x_dtype = x.dtype

        # 1. Rotation (identical to fake quant)
        x = self._apply_rotation(x)

        if self.quantizer.bits >= 16:
            if R1 is not None:
                return self.module(x, R1, R2, transpose).to(x_dtype)
            else:
                return self.module(x).to(x_dtype)

        # 2. Compute per-token activation quantization parameters
        self.quantizer.find_params(x)
        quantizer = self.quantizer

        # 3. Split activation and quantize to centered integers
        if quantizer.groupsize > 0:
            init_shape = x.shape  # (batch, seq, K)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // quantizer.groupsize, quantizer.groupsize)

        low_dim = quantizer.low_bits_length
        high_dim = x.shape[-1] - quantizer.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        # Quantize and dequant each precision group
        if quantizer.sym:
            dq_m = quantizer.scale * torch.clamp(torch.round(x_m / quantizer.scale), -(quantizer.maxq + 1), quantizer.maxq)
        else:
            q_m_int = torch.clamp(torch.round(x_m / quantizer.scale) + quantizer.zero, 0, quantizer.maxq)
            dq_m = quantizer.scale * (q_m_int - quantizer.zero)

        dq_h = None
        if quantizer.high_bits_length > 0:
            if quantizer.sym:
                dq_h = quantizer.scale_h * torch.clamp(torch.round(x_h / quantizer.scale_h), -(quantizer.maxq_h + 1), quantizer.maxq_h)
            else:
                q_h_int = torch.clamp(torch.round(x_h / quantizer.scale_h) + quantizer.zero_h, 0, quantizer.maxq_h)
                dq_h = quantizer.scale_h * (q_h_int - quantizer.zero_h)

        dq_l = None
        if quantizer.low_bits_length > 0:
            if quantizer.sym:
                dq_l = quantizer.scale_l * torch.clamp(torch.round(x_l / quantizer.scale_l), -(quantizer.maxq_l + 1), quantizer.maxq_l)
            else:
                q_l_int = torch.clamp(torch.round(x_l / quantizer.scale_l) + quantizer.zero_l, 0, quantizer.maxq_l)
                dq_l = quantizer.scale_l * (q_l_int - quantizer.zero_l)

        self.quantizer.free()

        # 4. Reassemble dequantized activation in original layout
        parts = []
        if dq_l is not None:
            parts.append(dq_l)
        parts.append(dq_m)
        if dq_h is not None:
            parts.append(dq_h)
        dq_x = torch.cat(parts, dim=-1)  # (..., groupsize) or (..., K)

        if quantizer.groupsize > 0:
            dq_x = dq_x.reshape(init_shape)  # (batch, seq, K)

        # 4b. Apply column_order permutation (critical for o_proj/down_proj)
        if column_order is not None:
            dq_x = dq_x[..., column_order]

        # 5. Dequantize weight and matmul
        # Weight was quantized to 8-bit per-group and stored in interleaved layout
        W_deq = self._dequant_weight()
        M_flat = dq_x.reshape(-1, dq_x.shape[-1])  # (M, K)

        # DEBUG: compare W_deq with original weight on first call
        if not hasattr(self, '_debug_printed'):
            W_orig = self.module.weight.data
            diff = (W_deq.float().to(W_orig.device) - W_orig.float()).abs()
            import logging
            logging.warning(f"[DEBUG] W_deq vs W_orig: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}, "
                          f"W_deq shape={W_deq.shape}, W_orig shape={W_orig.shape}, "
                          f"W_deq range=[{W_deq.min().item():.4f}, {W_deq.max().item():.4f}], "
                          f"W_orig range=[{W_orig.min().item():.4f}, {W_orig.max().item():.4f}]")
            self._debug_printed = True

        y = (M_flat.float() @ W_deq.float().T).half()  # fp32 matmul to avoid overflow

        # Bias
        if self.module.bias is not None:
            y = y + self.module.bias

        # Reshape back
        if quantizer.groupsize > 0:
            y = y.reshape(init_shape[0], init_shape[1], -1)
        else:
            y = y.reshape(*x_m.shape[:-1], -1)

        y = y.to(x_dtype)

        # 6. Output quantization (if needed)
        if self.out_quantizer.bits < 16:
            self.out_quantizer.find_params(y)
            y = self.out_quantizer(y).to(x_dtype)
            self.out_quantizer.free()

        return y

    def _dequant_weight(self):
        """Dequantize stored integer weights back to fp16 in original K layout."""
        quantizer = self.quantizer
        N = self.W_m_int.shape[0]
        K_m = self.W_m_int.shape[1]

        # Dequant each precision group
        W_m_deq = self._dequant_weight_group(self.W_m_int, self.W_m_scale, self._w_groupsize_m)

        W_h_deq = None
        if hasattr(self, 'W_h_int'):
            W_h_deq = self._dequant_weight_group(self.W_h_int, self.W_h_scale, self._w_groupsize_h)

        W_l_deq = None
        if hasattr(self, 'W_l_int'):
            W_l_deq = self._dequant_weight_group(self.W_l_int, self.W_l_scale, self._w_groupsize_l)

        # Reassemble in original K layout
        if quantizer.groupsize > 0:
            # GPTQ uses contiguous layout [low | main | high] globally,
            # so just concatenate contiguously (same as non-grouped)
            parts = []
            if W_l_deq is not None:
                parts.append(W_l_deq)
            parts.append(W_m_deq)
            if W_h_deq is not None:
                parts.append(W_h_deq)
            W_deq = torch.cat(parts, dim=1)
        else:
            # Contiguous: layout is [low | main | high]
            parts = []
            if W_l_deq is not None:
                parts.append(W_l_deq)
            parts.append(W_m_deq)
            if W_h_deq is not None:
                parts.append(W_h_deq)
            W_deq = torch.cat(parts, dim=1)

        return W_deq.half()

    @staticmethod
    def _dequant_weight_group(W_int, W_scale, w_groupsize):
        """Dequantize per-group quantized weights."""
        N, K = W_int.shape
        num_groups = W_scale.shape[1]
        gs = w_groupsize

        if K % gs != 0:
            pad = num_groups * gs - K
            W_int = torch.nn.functional.pad(W_int.float(), (0, pad))
        else:
            W_int = W_int.float()

        W_g = W_int.reshape(N, num_groups, gs)
        W_deq = (W_g * W_scale.float().unsqueeze(2)).reshape(N, num_groups * gs)[:, :K]
        return W_deq

    def get_rotated_weight(self, R1=None, R2=None, R4=None, transpose=False):

        return self.module.get_rotated_weight(R1, R2, R4, transpose)


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq, False).to(x_dtype)

            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq, False).to(
                x_dtype
            )
        return x

    def quantize_to_int(self, x):
        """Return the raw integer values (NOT dequantized) and the scale/zero used.

        Returns:
            q_int: integer tensor (same shape as x), dtype int16 or int8
            scale: per-channel scale used
            zero: per-channel zero point used (0 for symmetric)
        """
        if self.ready() and self.bits < 16:
            scale = self.scale.to(x.device)
            if self.sym:
                q = torch.clamp(torch.round(x / scale), -(self.maxq + 1), self.maxq)
                return q.to(torch.int16), scale, torch.zeros_like(scale)
            else:
                zero = self.zero.to(x.device)
                q = torch.clamp(torch.round(x / scale) + zero, 0, self.maxq)
                return q.to(torch.int16), scale, zero
        return None, None, None

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(
    module: ActQuantWrapper,
    name: str = "",
    layers=[
        torch.nn.Linear,
        QuantizeLinear,
        ActQuantWrapper,
        transformers.models.falcon.modeling_falcon.FalconLinear,
    ],
) -> None:
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper, QuantizeLinear],
    name: str = "",
):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
