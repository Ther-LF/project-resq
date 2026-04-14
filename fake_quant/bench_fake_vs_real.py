#!/usr/bin/env python3
"""Benchmark: Fake Quant vs Real Quant latency comparison.

Measures per-layer and end-to-end inference time for both modes.
Usage:
    source /vllm-workspace/plaquant/.venv/bin/activate
    cd /vllm-workspace/plaquant/project-resq/fake_quant
    python bench_fake_vs_real.py
"""

import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from transformers import AutoConfig, AutoTokenizer

from eval_utils.main import ptq_model
from eval_utils.modeling_llama_2 import LlamaForCausalLM
from utils import data_utils, utils
from utils.process_args import process_args_ptq
from utils.quant_utils import ActQuantWrapper


# ─────────────────────────── helpers ───────────────────────────

def timer_sync():
    """Synchronize CUDA and return current time."""
    torch.cuda.synchronize()
    return time.perf_counter()


def benchmark_layer_forward(layer, inp, attention_mask, position_ids,
                            position_embeddings, warmup=5, repeats=20):
    """Benchmark a single decoder layer's forward pass."""
    # Warmup
    for _ in range(warmup):
        _ = layer(inp, attention_mask=attention_mask,
                  position_ids=position_ids,
                  position_embeddings=position_embeddings)

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = timer_sync()
        _ = layer(inp, attention_mask=attention_mask,
                  position_ids=position_ids,
                  position_embeddings=position_embeddings)
        t1 = timer_sync()
        times.append((t1 - t0) * 1000)  # ms
    return times


def benchmark_full_inference(model, input_ids, warmup=3, repeats=10):
    """Benchmark full model forward (prefill) latency."""
    model.eval()
    dev = next(model.parameters()).device

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids.to(dev))

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = timer_sync()
        with torch.no_grad():
            _ = model(input_ids.to(dev))
        t1 = timer_sync()
        times.append((t1 - t0) * 1000)
    return times


def collect_wrapper_stats(model):
    """Collect info about ActQuantWrapper layers."""
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, ActQuantWrapper):
            is_real = hasattr(m, '_real_quant_ready') and m._real_quant_ready
            bits = m.quantizer.bits
            gs = m.quantizer.groupsize
            high_bits = m.quantizer.high_bits_length
            stats[name] = {
                'bits': bits, 'groupsize': gs, 'high_dim': high_bits,
                'real_quant': is_real,
                'weight_shape': tuple(m.module.weight.shape),
            }
    return stats


def benchmark_per_linear(layer, inp, warmup=3, repeats=15):
    """Benchmark each ActQuantWrapper within a decoder layer separately."""
    results = {}
    # Find all ActQuantWrappers in this layer
    wrappers = {}
    for name, m in layer.named_modules():
        if isinstance(m, ActQuantWrapper) and m.quantizer.bits < 16:
            wrappers[name] = m

    # For each wrapper, measure its forward independently
    for wname, wrapper in wrappers.items():
        N, K = wrapper.module.weight.shape
        # Create a suitable input
        if wrapper.quantizer.groupsize > 0:
            test_inp = torch.randn(1, 128, K, dtype=torch.float16,
                                   device=inp.device)
        else:
            test_inp = torch.randn(1, 128, K, dtype=torch.float16,
                                   device=inp.device)

        # Warmup
        for _ in range(warmup):
            _ = wrapper(test_inp)

        times = []
        for _ in range(repeats):
            t0 = timer_sync()
            _ = wrapper(test_inp)
            t1 = timer_sync()
            times.append((t1 - t0) * 1000)

        results[wname] = {
            'shape': f'({N}, {K})',
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
        }
    return results


# ─────────────────────────── main ───────────────────────────

def main():
    import sys
    import argparse

    # We'll build args manually to avoid torchrun
    # Parse the common PTQ args
    dev = torch.device('cuda:0')

    print("=" * 70)
    print("ResQ Fake Quant vs Real Quant Latency Benchmark")
    print("=" * 70)

    # ── Step 1: Load model with fake quant ──
    print("\n[1/6] Loading model with fake quant (PTQ)...")
    t0 = time.time()

    # We need to go through the same ptq flow as ptq.py
    # Simulate the args
    config = AutoConfig.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    dtype = torch.float16
    config.tie_word_embeddings = False

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path="unsloth/Llama-3.2-1B-Instruct",
        torch_dtype=dtype,
        config=config,
    )
    model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))

    # Build a minimal namespace for ptq_args
    class PTQArgs:
        w_bits = 16
        a_bits = 4
        k_bits = 4
        v_bits = 4
        high_bits = 8
        low_bits = 2
        w_clip = True
        a_asym = True
        k_asym = True
        v_asym = True
        k_groupsize = 64
        v_groupsize = 64
        high_fraction = 0.125
        low_fraction = 0.0
        rotate_mode = "resq"
        optimized_rotation_path = "./rotation/R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-1B.bin"
        optimized_basis_path = "./rotation/U-wikitext-512-Llama-3.2-1B.bin"
        rotation_granularity = "full_shared"
        rotate = True
        w_rtn = False
        w_gptq = False
        real_quant = False
        # lm_eval params
        bsz = 1
        capture_layer_io = False
        layer_idx = -1
        seed = 0
        lm_eval = False
        tasks = ""
        multigpu = False
        flash_attn = False
        model_max_length = 2048
        # Additional params that ptq_model might need
        a_clip_ratio = 1.0
        w_asym = False
        int8_down_proj = False
        residual_fraction = 0.0
        residual_bits = 16
        sparse_fraction = 0.0
        sparse_bits = 16
        optimized_sparse_rotation_path = ""

    class ModelArgs:
        input_model = "unsloth/Llama-3.2-1B-Instruct"

    ptq_args = PTQArgs()
    model_args = ModelArgs()

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = 2048
    print(f"    Model loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Prepare test inputs ──
    print("\n[2/6] Preparing test inputs...")
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct", use_fast=True)

    # Different sequence lengths to benchmark
    seq_lengths = [128, 512, 2048]
    test_inputs = {}
    for sl in seq_lengths:
        test_inputs[sl] = torch.randint(100, 30000, (1, sl), device=dev)

    # ── Step 3: Benchmark FAKE QUANT (full model on GPU) ──
    print("\n[3/6] Benchmarking FAKE QUANT (full model forward)...")
    model = model.to(dev)
    model.eval()

    fake_results = {}
    for sl in seq_lengths:
        times = benchmark_full_inference(model, test_inputs[sl],
                                         warmup=3, repeats=10)
        fake_results[sl] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'median_ms': np.median(times),
        }
        print(f"    seq_len={sl:>5d}: {np.mean(times):.2f} ± {np.std(times):.2f} ms "
              f"(min={np.min(times):.2f}, median={np.median(times):.2f})")

    # ── Step 4: Benchmark FAKE QUANT per-layer ──
    print("\n[4/6] Benchmarking FAKE QUANT per-layer (layer 0)...")
    layer0 = model.model.layers[0]

    # Get a representative hidden state
    with torch.no_grad():
        dummy_ids = test_inputs[512]
        emb = model.model.embed_tokens(dummy_ids)
        # Create attention mask and position ids
        batch_size, seq_len = dummy_ids.shape
        attention_mask = torch.ones(batch_size, seq_len, device=dev, dtype=dtype)
        # Expand attention mask for causal
        from transformers.modeling_attn_mask_utils import \
            _prepare_4d_causal_attention_mask
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask.long(), (batch_size, seq_len), emb, 0)
        position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)
        # Get position embeddings (rotary)
        pos_emb = model.model.rotary_emb(emb, position_ids)

    fake_layer_times = benchmark_layer_forward(
        layer0, emb, causal_mask, position_ids, pos_emb,
        warmup=5, repeats=20)
    print(f"    Layer 0 (seq=512): {np.mean(fake_layer_times):.2f} ± "
          f"{np.std(fake_layer_times):.2f} ms")

    # Per-linear breakdown
    print("\n    Per-linear breakdown (fake quant, seq=128):")
    fake_linear_results = benchmark_per_linear(layer0, emb, warmup=3, repeats=15)
    for name, res in fake_linear_results.items():
        print(f"      {name:>40s} {res['shape']:>15s}: "
              f"{res['mean_ms']:.3f} ± {res['std_ms']:.3f} ms")

    # ── Step 5: Enable REAL QUANT ──
    print("\n[5/6] Enabling REAL QUANT and benchmarking...")
    real_quant_count = 0
    for name, m in model.named_modules():
        if isinstance(m, ActQuantWrapper):
            m.prepare_real_quant_weights()
            if getattr(m, '_real_quant_ready', False):
                m.forward = m.forward_real_quant
                real_quant_count += 1
    print(f"    Real quant enabled for {real_quant_count} layers")

    # Full model forward benchmark
    real_results = {}
    for sl in seq_lengths:
        times = benchmark_full_inference(model, test_inputs[sl],
                                         warmup=3, repeats=10)
        real_results[sl] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'median_ms': np.median(times),
        }
        print(f"    seq_len={sl:>5d}: {np.mean(times):.2f} ± {np.std(times):.2f} ms "
              f"(min={np.min(times):.2f}, median={np.median(times):.2f})")

    # ── Step 6: Benchmark REAL QUANT per-layer ──
    print("\n[6/6] Benchmarking REAL QUANT per-layer (layer 0)...")
    layer0 = model.model.layers[0]
    real_layer_times = benchmark_layer_forward(
        layer0, emb, causal_mask, position_ids, pos_emb,
        warmup=5, repeats=20)
    print(f"    Layer 0 (seq=512): {np.mean(real_layer_times):.2f} ± "
          f"{np.std(real_layer_times):.2f} ms")

    # Per-linear breakdown
    print("\n    Per-linear breakdown (real quant, seq=128):")
    real_linear_results = benchmark_per_linear(layer0, emb, warmup=3, repeats=15)
    for name, res in real_linear_results.items():
        print(f"      {name:>40s} {res['shape']:>15s}: "
              f"{res['mean_ms']:.3f} ± {res['std_ms']:.3f} ms")

    # ─────────────────────── Summary ───────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Fake Quant vs Real Quant Latency")
    print("=" * 70)

    print(f"\n{'Mode':<15s} | ", end="")
    for sl in seq_lengths:
        print(f"{'seq=' + str(sl):>15s} | ", end="")
    print()
    print("-" * (18 + 18 * len(seq_lengths)))

    print(f"{'Fake Quant':<15s} | ", end="")
    for sl in seq_lengths:
        print(f"{fake_results[sl]['mean_ms']:>12.2f} ms | ", end="")
    print()

    print(f"{'Real Quant':<15s} | ", end="")
    for sl in seq_lengths:
        print(f"{real_results[sl]['mean_ms']:>12.2f} ms | ", end="")
    print()

    print(f"{'Ratio (R/F)':<15s} | ", end="")
    for sl in seq_lengths:
        ratio = real_results[sl]['mean_ms'] / fake_results[sl]['mean_ms']
        print(f"{ratio:>12.2f}x  | ", end="")
    print()

    print(f"\n{'Per-layer (seq=512)':<30s}")
    print(f"  Fake Quant: {np.mean(fake_layer_times):.2f} ms")
    print(f"  Real Quant: {np.mean(real_layer_times):.2f} ms")
    ratio = np.mean(real_layer_times) / np.mean(fake_layer_times)
    print(f"  Ratio:      {ratio:.2f}x")

    print(f"\n{'Per-linear breakdown (seq=128)':<30s}")
    print(f"  {'Name':>40s} | {'Fake (ms)':>10s} | {'Real (ms)':>10s} | {'Ratio':>6s}")
    print("  " + "-" * 78)
    for name in fake_linear_results:
        f_ms = fake_linear_results[name]['mean_ms']
        r_ms = real_linear_results.get(name, {}).get('mean_ms', float('nan'))
        ratio = r_ms / f_ms if f_ms > 0 else float('nan')
        print(f"  {name:>40s} | {f_ms:>10.3f} | {r_ms:>10.3f} | {ratio:>5.2f}x")

    print("\n" + "=" * 70)
    print("NOTE: Real quant is SLOWER because it does:")
    print("  - fp32 matmul (instead of fp16 cuBLAS GEMM)")
    print("  - Weight dequant from int8 per-group buffers each forward")
    print("  - Extra tensor ops (split, cat, reshape)")
    print("This is expected! The real speedup comes from a custom CUDA kernel")
    print("that does true INT4/INT8 GEMM on Tensor Cores.")
    print("=" * 70)


if __name__ == "__main__":
    main()
