#!/usr/bin/env python3
"""Collect GEMM ground-truth data from ResQ model inference.

Hooks into each ActQuantWrapper during WikiText2 PPL evaluation to capture:
- FP16 input activation (post-rotation, pre-quantization)
- Quantized activation integers + scales (main/high groups)
- FP16 weight (GPTQ dequantized)
- Integer weight + scales (from checkpoint)
- FP16 baseline output (no quantization)
- Fake quant output
- Real quant output
- Layer metadata

Usage:
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=24560 collect_gemm_data.py \
        --input_model unsloth/Llama-3.2-1B-Instruct \
        --load_qmodel_path ./qmodels/W4A4KV4-Llama-3.2-1B-v2.pt \
        ... (same args as 3_eval_w4a4_real_v2.sh) \
        --gemm_output_dir ./gemm_data \
        --gemm_batch_sizes 1,2,4 \
        --gemm_num_batches 1
"""

import datetime
import json
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from eval_utils.main import ptq_model
from eval_utils.modeling_llama_2 import LlamaForCausalLM
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
from utils import data_utils, utils, quant_utils
from utils.process_args import process_args_ptq
from utils.quant_utils import ActQuantWrapper, ActQuantizer
from utils import hadamard_utils
from utils.utils import HadamardTransform
import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def sanitize_layer_name(name):
    """Convert module name to filesystem-safe directory name."""
    return name.replace('.', '_')


def collect_layer_metadata(name, wrapper):
    """Extract metadata dict from an ActQuantWrapper."""
    q = wrapper.quantizer
    meta = {
        'name': name,
        'weight_shape': list(wrapper.module.weight.shape),
        'has_bias': wrapper.module.bias is not None,
        'a_bits': q.bits,
        'a_sym': q.sym,
        'a_groupsize': q.groupsize,
        'a_high_bits': q.high_bits,
        'a_high_bits_length': q.high_bits_length,
        'a_low_bits': q.low_bits,
        'a_low_bits_length': q.low_bits_length,
        'online_full_had': wrapper.online_full_had,
        'online_partial_had': wrapper.online_partial_had,
        'has_column_order': hasattr(wrapper, '_column_order') and wrapper._column_order is not None,
        'out_quantizer_bits': wrapper.out_quantizer.bits,
        'has_real_quant': getattr(wrapper, '_real_quant_ready', False),
    }
    # Add weight quant info if available
    if hasattr(wrapper, 'W_m_int'):
        meta['w_m_int_shape'] = list(wrapper.W_m_int.shape)
        meta['w_m_scale_shape'] = list(wrapper.W_m_scale.shape)
    if hasattr(wrapper, 'W_h_int'):
        meta['w_h_int_shape'] = list(wrapper.W_h_int.shape)
        meta['w_h_scale_shape'] = list(wrapper.W_h_scale.shape)
    return meta


def quantize_activation(x, quantizer):
    """Quantize activation and return integer tensors + scales for each precision group.

    Returns dict with keys: main, high, low (each having q_int, scale, zero).
    """
    result = {}
    low_dim = quantizer.low_bits_length
    high_dim = x.shape[-1] - quantizer.high_bits_length

    x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

    # Main group
    if quantizer.sym:
        q_m, scale_m = quant_utils.sym_quant(x_m, quantizer.scale, quantizer.maxq)
        result['main'] = {'q_int': q_m.cpu().short(), 'scale': quantizer.scale.cpu().half(), 'zero': None}
    else:
        q_m, scale_m, zero_m = quant_utils.asym_quant(x_m, quantizer.scale, quantizer.zero, quantizer.maxq)
        result['main'] = {'q_int': q_m.cpu().short(), 'scale': scale_m.cpu().half(), 'zero': zero_m.cpu().half()}

    # High group
    if quantizer.high_bits_length > 0:
        if quantizer.sym:
            q_h, scale_h = quant_utils.sym_quant(x_h, quantizer.scale_h, quantizer.maxq_h)
            result['high'] = {'q_int': q_h.cpu().short(), 'scale': quantizer.scale_h.cpu().half(), 'zero': None}
        else:
            q_h, scale_h, zero_h = quant_utils.asym_quant(x_h, quantizer.scale_h, quantizer.zero_h, quantizer.maxq_h)
            result['high'] = {'q_int': q_h.cpu().short(), 'scale': scale_h.cpu().half(), 'zero': zero_h.cpu().half()}

    # Low group
    if quantizer.low_bits_length > 0:
        if quantizer.sym:
            q_l, scale_l = quant_utils.sym_quant(x_l, quantizer.scale_l, quantizer.maxq_l)
            result['low'] = {'q_int': q_l.cpu().short(), 'scale': quantizer.scale_l.cpu().half(), 'zero': None}
        else:
            q_l, scale_l, zero_l = quant_utils.asym_quant(x_l, quantizer.scale_l, quantizer.zero_l, quantizer.maxq_l)
            result['low'] = {'q_int': q_l.cpu().short(), 'scale': scale_l.cpu().half(), 'zero': zero_l.cpu().half()}

    return result


class GEMMDataCollector:
    """Hooks into model to collect GEMM data during inference."""

    def __init__(self, model, output_dir, batch_sizes=(1, 2, 4), num_batches=1):
        self.model = model
        self.output_dir = output_dir
        self.batch_sizes = batch_sizes
        self.num_batches = num_batches
        self.hooks = []
        self.collected = {}  # {layer_name: {bs: data}}

        os.makedirs(output_dir, exist_ok=True)

        # Find all ActQuantWrapper layers
        self.wrappers = {}
        for name, m in model.named_modules():
            if isinstance(m, ActQuantWrapper) and m.quantizer.bits < 16:
                self.wrappers[name] = m

        print(f"Found {len(self.wrappers)} quantized layers to collect")

    def _install_hooks(self, batch_size):
        """Install forward hooks to capture data for a specific batch size."""
        self.hooks = []
        self._current_bs = batch_size
        self._capture_data = {}

        for name, wrapper in self.wrappers.items():
            hook = self._make_hook(name, wrapper)
            handle = wrapper.register_forward_pre_hook(hook)
            self.hooks.append(handle)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _make_hook(self, name, wrapper):
        """Create a pre-forward hook that captures input for FP16 baseline only.
        Activation quantization data is captured from _last_quant (set by forward_real_quant).
        """
        collector = self

        def hook_fn(module, args):
            x = args[0]  # input activation

            # Apply rotation (same as forward does)
            x_rotated = wrapper._apply_rotation(x.clone())

            # Compute FP16 baseline: use column_order if present (o_proj)
            # Weight W is already rearranged by rearrange_columns(),
            # so we reorder x to match for the FP16 baseline matmul only.
            col_order = getattr(wrapper, '_column_order', None)
            if col_order is not None:
                x_for_baseline = x_rotated[..., col_order]
            else:
                x_for_baseline = x_rotated

            # Compute FP16 baseline: x @ W.T (no quantization)
            W = wrapper.module.weight.data
            x_flat = x_for_baseline.reshape(-1, x_for_baseline.shape[-1])
            output_fp16_baseline = (x_flat.float() @ W.float().T).half()
            output_fp16_baseline = output_fp16_baseline.reshape(*x_rotated.shape[:-1], -1)

            # Store partial data (activation quant will be grabbed from _last_quant after forward)
            bs_key = f"bs{collector._current_bs}"
            if name not in collector._capture_data:
                collector._capture_data[name] = {}

            collector._capture_data[name][bs_key] = {
                'input_fp16': x_rotated.cpu().half(),
                'output_fp16_baseline': output_fp16_baseline.cpu().half(),
            }

        return hook_fn

    def _install_output_hooks(self, batch_size):
        """Install output hooks (post-forward) to capture fake/real quant outputs + activation quant data."""
        self._output_hooks = []

        for name, wrapper in self.wrappers.items():
            def make_output_hook(layer_name, wrap):
                def hook_fn(module, args, output):
                    bs_key = f"bs{self._current_bs}"
                    if layer_name in self._capture_data and bs_key in self._capture_data[layer_name]:
                        # Determine if this is real quant based on readiness flag
                        # NOTE: comparing bound methods (module.forward == module.forward_real_quant)
                        # is unreliable because bound method objects are recreated on access.
                        if hasattr(module, '_real_quant_ready') and module._real_quant_ready:
                            self._capture_data[layer_name][bs_key]['output_real_quant'] = output.cpu().half()
                            # Grab quantized activation from _last_quant (computed inside forward_real_quant)
                            if hasattr(module, '_last_quant'):
                                lq = module._last_quant
                                act_quant = {}
                                act_quant['main'] = {
                                    'q_int': lq['q_m'].cpu().short(),
                                    'scale': lq['s_x_m'].cpu().half() if torch.is_tensor(lq['s_x_m']) else None,
                                    'zero': lq['z_x_m'].cpu().half() if torch.is_tensor(lq['z_x_m']) else None,
                                }
                                if lq['q_h'] is not None:
                                    act_quant['high'] = {
                                        'q_int': lq['q_h'].cpu().short(),
                                        'scale': lq['s_x_h'].cpu().half() if torch.is_tensor(lq['s_x_h']) else None,
                                        'zero': lq['z_x_h'].cpu().half() if torch.is_tensor(lq['z_x_h']) else None,
                                    }
                                self._capture_data[layer_name][bs_key]['act_quant'] = act_quant
                        else:
                            self._capture_data[layer_name][bs_key]['output_fake_quant'] = output.cpu().half()
                return hook_fn

            handle = wrapper.register_forward_hook(make_output_hook(name, wrapper))
            self._output_hooks.append(handle)

    def _remove_output_hooks(self):
        for h in self._output_hooks:
            h.remove()
        self._output_hooks = []

    def collect(self, tokenizer, device, seqlen=2048):
        """Run inference and collect data for all batch sizes."""

        # Get evaluation data
        testenc = data_utils.get_wikitext2(
            seed=0, seqlen=seqlen, tokenizer=tokenizer, eval_mode=True
        )
        test_ids = testenc.input_ids

        for bs in self.batch_sizes:
            print(f"\n=== Collecting batch_size={bs} ===")
            self._capture_data = {}
            self._current_bs = bs

            # Install hooks
            self._install_hooks(bs)
            self._install_output_hooks(bs)

            # Run inference on a few batches
            self.model.eval()
            self.model.to(device)

            for batch_idx in range(self.num_batches):
                start = batch_idx * bs * seqlen
                input_ids = []
                for b in range(bs):
                    s = start + b * seqlen
                    if s + seqlen > test_ids.shape[1]:
                        s = 0  # wrap around
                    input_ids.append(test_ids[:, s:s+seqlen])
                input_ids = torch.cat(input_ids, dim=0).to(device)  # (bs, seqlen)

                with torch.no_grad():
                    self.model(input_ids)

            # After forward: save _last_quant and _last_gemm_output directly to disk
            # (must be done immediately, before next bs iteration overwrites them)
            bs_key = f"bs{bs}"
            for name, wrapper in self.wrappers.items():
                layer_dir = os.path.join(self.output_dir, sanitize_layer_name(name))
                os.makedirs(layer_dir, exist_ok=True)

                if hasattr(wrapper, '_last_quant'):
                    lq = wrapper._last_quant
                    main_data = {
                        'q_int': lq['q_m'].cpu().short(),
                        'scale': lq['s_x_m'].cpu().half() if torch.is_tensor(lq['s_x_m']) else None,
                        'zero': lq['z_x_m'].cpu().half() if torch.is_tensor(lq['z_x_m']) else None,
                    }
                    torch.save(main_data, os.path.join(layer_dir, f'act_quant_main_{bs_key}.pt'))
                    if lq['q_h'] is not None:
                        high_data = {
                            'q_int': lq['q_h'].cpu().short(),
                            'scale': lq['s_x_h'].cpu().half() if torch.is_tensor(lq['s_x_h']) else None,
                            'zero': lq['z_x_h'].cpu().half() if torch.is_tensor(lq['z_x_h']) else None,
                        }
                        torch.save(high_data, os.path.join(layer_dir, f'act_quant_high_{bs_key}.pt'))

                if hasattr(wrapper, '_last_gemm_output'):
                    gemm_out = wrapper._last_gemm_output
                    # Verify shape matches current batch size (guard against stale data)
                    if gemm_out.shape[0] == bs:
                        torch.save(gemm_out.cpu().half(),
                                   os.path.join(layer_dir, f'output_gemm_only_{bs_key}.pt'))
                    # Clear to prevent stale data leaking into next bs iteration
                    del wrapper._last_gemm_output

            self._remove_hooks()
            self._remove_output_hooks()

            # Merge into collected
            for layer_name, bs_data in self._capture_data.items():
                if layer_name not in self.collected:
                    self.collected[layer_name] = {}
                self.collected[layer_name].update(bs_data)

            print(f"  Captured {len(self._capture_data)} layers")

    def save_weight_data(self):
        """Save weight data (only once, independent of batch size)."""
        for name, wrapper in self.wrappers.items():
            layer_dir = os.path.join(self.output_dir, sanitize_layer_name(name))
            os.makedirs(layer_dir, exist_ok=True)

            # FP16 weight
            torch.save(wrapper.module.weight.data.cpu().half(),
                       os.path.join(layer_dir, 'weight_fp16.pt'))

            # Integer weights (if available)
            if hasattr(wrapper, 'W_m_int'):
                torch.save({'q_int': wrapper.W_m_int.cpu(), 'scale': wrapper.W_m_scale.cpu()},
                           os.path.join(layer_dir, 'weight_int_main.pt'))
            if hasattr(wrapper, 'W_h_int'):
                torch.save({'q_int': wrapper.W_h_int.cpu(), 'scale': wrapper.W_h_scale.cpu()},
                           os.path.join(layer_dir, 'weight_int_high.pt'))
            if hasattr(wrapper, 'W_l_int'):
                torch.save({'q_int': wrapper.W_l_int.cpu(), 'scale': wrapper.W_l_scale.cpu()},
                           os.path.join(layer_dir, 'weight_int_low.pt'))

            # Column order (for o_proj)
            if hasattr(wrapper, '_column_order') and wrapper._column_order is not None:
                torch.save(wrapper._column_order.cpu(),
                           os.path.join(layer_dir, 'column_order.pt'))

    def save_all(self):
        """Save all collected data to disk."""
        # Save global metadata
        global_meta = {
            'batch_sizes': self.batch_sizes,
            'num_batches': self.num_batches,
            'num_layers': len(self.wrappers),
            'layer_names': list(self.wrappers.keys()),
        }
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(global_meta, f, indent=2)

        # Save weight data
        self.save_weight_data()

        # Save per-layer data
        for name, wrapper in self.wrappers.items():
            layer_dir = os.path.join(self.output_dir, sanitize_layer_name(name))
            os.makedirs(layer_dir, exist_ok=True)

            # Save layer metadata
            meta = collect_layer_metadata(name, wrapper)
            with open(os.path.join(layer_dir, 'metadata.json'), 'w') as f:
                json.dump(meta, f, indent=2)

            # Save per-batch-size data
            if name in self.collected:
                for bs_key, data in self.collected[name].items():
                    # Input
                    if 'input_fp16' in data:
                        torch.save(data['input_fp16'],
                                   os.path.join(layer_dir, f'input_fp16_{bs_key}.pt'))

                    # Activation quant data (already saved directly after forward)
                    # Just check if files exist, skip if already written
                    if 'act_quant' in data:
                        for group_name, group_data in data['act_quant'].items():
                            torch.save(group_data,
                                       os.path.join(layer_dir, f'act_quant_{group_name}_{bs_key}.pt'))

                    # Outputs
                    if 'output_fp16_baseline' in data:
                        torch.save(data['output_fp16_baseline'],
                                   os.path.join(layer_dir, f'output_fp16_baseline_{bs_key}.pt'))
                    if 'output_fake_quant' in data:
                        torch.save(data['output_fake_quant'],
                                   os.path.join(layer_dir, f'output_fake_quant_{bs_key}.pt'))
                    if 'output_real_quant' in data:
                        torch.save(data['output_real_quant'],
                                   os.path.join(layer_dir, f'output_real_quant_{bs_key}.pt'))
                    if 'output_gemm_only' in data:
                        torch.save(data['output_gemm_only'],
                                   os.path.join(layer_dir, f'output_gemm_only_{bs_key}.pt'))

        print(f"\nData saved to {self.output_dir}")
        # Print summary
        total_files = sum(len(os.listdir(os.path.join(self.output_dir, d)))
                         for d in os.listdir(self.output_dir)
                         if os.path.isdir(os.path.join(self.output_dir, d)))
        print(f"Total files: {total_files}")


def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=1))
    model_args, training_args, ptq_args = process_args_ptq()
    seed_everything(ptq_args.seed)

    # Parse extra args
    output_dir = getattr(ptq_args, 'gemm_output_dir', './gemm_data')
    batch_sizes_str = getattr(ptq_args, 'gemm_batch_sizes', '1,2,4')
    batch_sizes = [int(x) for x in batch_sizes_str.split(',')]
    num_batches = getattr(ptq_args, 'gemm_num_batches', 1)

    config = AutoConfig.from_pretrained(model_args.input_model)
    if ptq_args.flash_attn:
        config._attn_implementation = "flash_attention_2"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    if "llama" in model_args.input_model.lower():
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=dtype, config=config,
        )
    elif "qwen2" in model_args.input_model.lower():
        model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=dtype, config=config,
        )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))

    model = ptq_model(ptq_args, model, model_args)

    # Prepare real quant weights if checkpoint available
    if getattr(ptq_args, 'real_quant', False) and ptq_args.load_qmodel_path:
        w_bits = ptq_args.w_bits if ptq_args.w_bits < 16 else None
        qmodel = torch.load(ptq_args.load_qmodel_path, map_location='cpu')
        gptq_quantizers = qmodel.get('w_quantizers', None)
        gptq_int_weights = qmodel.get('w_int_weights', None)
        del qmodel

        for name, m in model.named_modules():
            if isinstance(m, ActQuantWrapper):
                wq_dict = None
                w_int_dict = None
                if gptq_quantizers is not None:
                    base_key = name + '.module'
                    wq_dict = {}
                    if base_key in gptq_quantizers:
                        wq_dict['main'] = gptq_quantizers[base_key]
                    if base_key + ',high_quantizer' in gptq_quantizers:
                        wq_dict['high'] = gptq_quantizers[base_key + ',high_quantizer']
                    if base_key + ',low_quantizer' in gptq_quantizers:
                        wq_dict['low'] = gptq_quantizers[base_key + ',low_quantizer']
                    if 'main' not in wq_dict:
                        wq_dict = None

                if gptq_int_weights is not None:
                    base_key = name + '.module'
                    if base_key in gptq_int_weights:
                        w_int_dict = gptq_int_weights[base_key]

                m.prepare_real_quant_weights(w_bits=w_bits, w_quantizers=wq_dict, w_int_weights=w_int_dict)

    # Find column_order for o_proj layers
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'new_column_order'):
            col_order = layer.self_attn.new_column_order
            # Find the o_proj wrapper
            for name, m in layer.named_modules():
                if 'o_proj' in name and isinstance(m, ActQuantWrapper):
                    m._column_order = col_order

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        model_max_length=training_args.model_max_length,
        padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False,
    )

    # === Phase 1: Collect fake quant outputs ===
    print("\n========== Phase 1: Collecting FAKE QUANT data ==========")
    collector = GEMMDataCollector(model, output_dir, batch_sizes, num_batches)
    collector.collect(tokenizer, 'cuda')

    # === Phase 2: Switch to real quant and collect ===
    if getattr(ptq_args, 'real_quant', False):
        print("\n========== Phase 2: Collecting REAL QUANT data ==========")
        real_count = 0
        for name, m in model.named_modules():
            if isinstance(m, ActQuantWrapper) and getattr(m, '_real_quant_ready', False):
                m.forward = m.forward_real_quant
                real_count += 1
        print(f"Switched {real_count} layers to real quant forward")

        # Create new collector for real quant
        collector_real = GEMMDataCollector(model, output_dir, batch_sizes, num_batches)
        collector_real.collect(tokenizer, 'cuda')

        # Merge real quant outputs into main collector
        for layer_name in collector_real.collected:
            if layer_name not in collector.collected:
                collector.collected[layer_name] = {}
            for bs_key, data in collector_real.collected[layer_name].items():
                if bs_key in collector.collected.get(layer_name, {}):
                    if 'output_real_quant' in data:
                        collector.collected[layer_name][bs_key]['output_real_quant'] = data['output_real_quant']
                    if 'act_quant' in data:
                        collector.collected[layer_name][bs_key]['act_quant'] = data['act_quant']

        # Grab GEMM-only output (pre output-quant) from wrappers
        # _last_gemm_output is the GEMM result BEFORE out_quantizer is applied
        for name, wrapper in collector_real.wrappers.items():
            if hasattr(wrapper, '_last_gemm_output'):
                for bs in batch_sizes:
                    bs_key = f"bs{bs}"
                    if name in collector.collected and bs_key in collector.collected[name]:
                        collector.collected[name][bs_key]['output_gemm_only'] = wrapper._last_gemm_output.cpu().half()

    # Save everything
    collector.save_all()

    dist.barrier()
    print("\n===== Collection complete! =====")


if __name__ == "__main__":
    main()
