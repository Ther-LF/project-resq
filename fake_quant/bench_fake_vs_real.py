#!/usr/bin/env python3
"""Benchmark: Fake Quant vs Real Quant latency comparison.

Measures per-layer and end-to-end inference time for both modes.
Usage:
    source /vllm-workspace/plaquant/.venv/bin/activate
    cd /vllm-workspace/plaquant/project-resq/fake_quant
    python bench_fake_vs_real.py
"""

import sys
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from transformers import AutoConfig, AutoTokenizer

from eval_utils.main import ptq_model
from eval_utils.modeling_llama_2 import LlamaForCausalLM
from utils import data_utils, utils
from utils.quant_utils import ActQuantWrapper


# ─────────────────────────── helpers ───────────────────────────

def timer_sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def benchmark_forward(fn, *args, warmup=5, repeats=20, **kwargs):
    """Generic benchmark: call fn(*args, **kwargs) and return times in ms."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = timer_sync()
        fn(*args, **kwargs)
        t1 = timer_sync()
        times.append((t1 - t0) * 1000)
    return times


def stats(times):
    return {
        'mean': np.mean(times), 'std': np.std(times),
        'min': np.min(times), 'median': np.median(times),
    }


def build_ptq_args():
    """Build PTQ args by injecting sys.argv and using the real parser."""
    saved_argv = sys.argv
    sys.argv = [
        "bench",
        "--rotate",
        "--rotate_mode", "resq",
        "--a_bits", "4",
        "--w_bits", "16",
        "--k_bits", "4",
        "--v_bits", "4",
        "--high_bits", "8",
        "--low_bits", "2",
        "--w_clip",
        "--a_asym",
        "--k_asym",
        "--v_asym",
        "--k_groupsize", "64",
        "--v_groupsize", "64",
        "--high_fraction", "0.125",
        "--low_fraction", "0.0",
        "--rotation_granularity", "full_shared",
        "--optimized_rotation_path", "./rotation/R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-1B.bin",
        "--optimized_basis_path", "./rotation/U-wikitext-512-Llama-3.2-1B.bin",
        # HF TrainingArguments
        "--input_model", "unsloth/Llama-3.2-1B-Instruct",
        "--per_device_eval_batch_size", "1",
        "--model_max_length", "2048",
        "--fp16", "True",
        "--bf16", "False",
    ]
    from utils.process_args import process_args_ptq
    model_args, training_args, ptq_args = process_args_ptq()
    sys.argv = saved_argv
    return model_args, training_args, ptq_args


def load_model_with_ptq(ptq_args, model_args):
    """Load and apply PTQ to model (fake quant)."""
    config = AutoConfig.from_pretrained(model_args.input_model)
    dtype = torch.float16
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    else:
        process_word_embeddings = False

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=dtype, config=config,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = 2048
    return model


def enable_real_quant(model):
    """Enable real quant on all applicable layers."""
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, ActQuantWrapper):
            m.prepare_real_quant_weights()
            if getattr(m, '_real_quant_ready', False):
                m.forward = m.forward_real_quant
                count += 1
    return count


def benchmark_per_linear(layer, device, seq_len=128, warmup=3, repeats=15):
    """Benchmark each ActQuantWrapper in a layer independently."""
    results = {}
    for name, m in layer.named_modules():
        if isinstance(m, ActQuantWrapper) and m.quantizer.bits < 16:
            K = m.module.weight.shape[1]
            x = torch.randn(1, seq_len, K, dtype=torch.float16, device=device)
            times = benchmark_forward(m, x, warmup=warmup, repeats=repeats)
            results[name] = {
                'shape': f'({m.module.weight.shape[0]}, {K})',
                **stats(times),
            }
    return results


# ─────────────────────────── main ───────────────────────────

@torch.no_grad()
def main():
    dev = torch.device('cuda:0')

    # Init distributed (required by rotation_utils barriers)
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29599')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', world_size=1, rank=0)

    print("=" * 70)
    print("  ResQ Fake Quant vs Real Quant Latency Benchmark")
    print("  Model: Llama-3.2-1B-Instruct | GPU: H20 | dtype: fp16")
    print("=" * 70)

    # ── Load model ──
    print("\n[1/7] Loading model with fake quant (PTQ)...")
    t0 = time.time()
    model_args, training_args, ptq_args = build_ptq_args()
    model = load_model_with_ptq(ptq_args, model_args)
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # ── Move full model to GPU ──
    print("\n[2/7] Moving model to GPU...")
    model = model.to(dev).eval()
    torch.cuda.empty_cache()

    # ── Prepare inputs ──
    print("\n[3/7] Preparing test inputs...")
    seq_lengths = [128, 512, 2048]
    test_inputs = {sl: torch.randint(100, 30000, (1, sl), device=dev)
                   for sl in seq_lengths}

    # ── Benchmark FAKE QUANT full forward ──
    print("\n[4/7] Benchmarking FAKE QUANT (full model forward)...")
    fake_full = {}
    for sl in seq_lengths:
        times = benchmark_forward(
            lambda ids: model(ids), test_inputs[sl], warmup=3, repeats=10)
        fake_full[sl] = stats(times)
        s = fake_full[sl]
        print(f"    seq={sl:>5d}: {s['mean']:>8.2f} ± {s['std']:.2f} ms  "
              f"(min={s['min']:.2f}, median={s['median']:.2f})")

    # ── Benchmark FAKE QUANT per-layer & per-linear ──
    print("\n[5/7] Benchmarking FAKE QUANT per-layer (layer 0, seq=512)...")
    layer0 = model.model.layers[0]
    # Get hidden states for layer input
    with torch.no_grad():
        emb = model.model.embed_tokens(test_inputs[512])
        bs, sl = 1, 512
        causal_mask = torch.triu(
            torch.full((sl, sl), float('-inf'), device=dev), diagonal=1
        ).unsqueeze(0).unsqueeze(0).to(torch.float16)
        position_ids = torch.arange(sl, device=dev).unsqueeze(0)
        pos_emb = model.model.rotary_emb(emb, position_ids)

    fake_layer_times = benchmark_forward(
        layer0, emb, attention_mask=causal_mask,
        position_ids=position_ids, position_embeddings=pos_emb,
        warmup=5, repeats=20)
    fake_layer = stats(fake_layer_times)
    print(f"    Layer 0: {fake_layer['mean']:.2f} ± {fake_layer['std']:.2f} ms")

    print("\n    Per-linear breakdown (fake quant, seq=128):")
    fake_linears = benchmark_per_linear(layer0, dev, seq_len=128)
    for name, r in fake_linears.items():
        print(f"      {name:>30s} {r['shape']:>15s}: "
              f"{r['mean']:.3f} ± {r['std']:.3f} ms")

    # ── Enable REAL QUANT ──
    print("\n[6/7] Enabling REAL QUANT...")
    count = enable_real_quant(model)
    print(f"    Enabled for {count} layers")
    torch.cuda.empty_cache()

    # ── Benchmark REAL QUANT full forward ──
    print("\n    Benchmarking REAL QUANT (full model forward)...")
    real_full = {}
    for sl in seq_lengths:
        times = benchmark_forward(
            lambda ids: model(ids), test_inputs[sl], warmup=3, repeats=10)
        real_full[sl] = stats(times)
        s = real_full[sl]
        print(f"    seq={sl:>5d}: {s['mean']:>8.2f} ± {s['std']:.2f} ms  "
              f"(min={s['min']:.2f}, median={s['median']:.2f})")

    # ── Benchmark REAL QUANT per-layer & per-linear ──
    print("\n[7/7] Benchmarking REAL QUANT per-layer (layer 0, seq=512)...")
    layer0 = model.model.layers[0]
    real_layer_times = benchmark_forward(
        layer0, emb, attention_mask=causal_mask,
        position_ids=position_ids, position_embeddings=pos_emb,
        warmup=5, repeats=20)
    real_layer = stats(real_layer_times)
    print(f"    Layer 0: {real_layer['mean']:.2f} ± {real_layer['std']:.2f} ms")

    print("\n    Per-linear breakdown (real quant, seq=128):")
    real_linears = benchmark_per_linear(layer0, dev, seq_len=128)
    for name, r in real_linears.items():
        print(f"      {name:>30s} {r['shape']:>15s}: "
              f"{r['mean']:.3f} ± {r['std']:.3f} ms")

    # ─────────────────────── Summary Tables ───────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: Full Model Forward Latency (ms)")
    print("=" * 70)
    hdr = f"  {'Mode':<15s}"
    for sl in seq_lengths:
        hdr += f" | {'seq=' + str(sl):>12s}"
    print(hdr)
    print("  " + "-" * (15 + 15 * len(seq_lengths)))

    row_f = f"  {'Fake Quant':<15s}"
    row_r = f"  {'Real Quant':<15s}"
    row_ratio = f"  {'Ratio (R/F)':<15s}"
    for sl in seq_lengths:
        row_f += f" | {fake_full[sl]['mean']:>9.2f} ms"
        row_r += f" | {real_full[sl]['mean']:>9.2f} ms"
        ratio = real_full[sl]['mean'] / fake_full[sl]['mean']
        row_ratio += f" | {ratio:>9.2f}x "
    print(row_f)
    print(row_r)
    print(row_ratio)

    print(f"\n  Per-layer (layer 0, seq=512):")
    print(f"    Fake Quant: {fake_layer['mean']:.2f} ms")
    print(f"    Real Quant: {real_layer['mean']:.2f} ms")
    print(f"    Ratio:      {real_layer['mean'] / fake_layer['mean']:.2f}x")

    print(f"\n  Per-linear breakdown (layer 0, seq=128):")
    print(f"    {'Name':>30s} | {'Fake(ms)':>9s} | {'Real(ms)':>9s} | {'Ratio':>6s}")
    print("    " + "-" * 62)
    total_fake = total_real = 0
    for name in fake_linears:
        f_ms = fake_linears[name]['mean']
        r_ms = real_linears.get(name, {}).get('mean', float('nan'))
        ratio = r_ms / f_ms if f_ms > 0 else float('nan')
        total_fake += f_ms
        total_real += r_ms
        print(f"    {name:>30s} | {f_ms:>9.3f} | {r_ms:>9.3f} | {ratio:>5.2f}x")
    print("    " + "-" * 62)
    print(f"    {'TOTAL':>30s} | {total_fake:>9.3f} | {total_real:>9.3f} | "
          f"{total_real / total_fake:.2f}x")

    print("\n" + "=" * 70)
    print("  NOTES:")
    print("  - Real quant is EXPECTED to be slower than fake quant because:")
    print("    1. fp32 matmul instead of cuBLAS fp16 GEMM")
    print("    2. Weight dequant from int8 buffers every forward pass")
    print("    3. Extra tensor ops (split, cat, reshape)")
    print("  - The actual speedup requires a custom CUDA kernel doing")
    print("    true INT4/INT8 GEMM on Tensor Cores (Phase 2-5)")
    print("  - This benchmark establishes the baseline to beat.")
    print("=" * 70)


if __name__ == "__main__":
    main()
