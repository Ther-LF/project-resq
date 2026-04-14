"""Debug: compare fake quant vs real quant on a single layer."""
import torch
import sys
sys.path.insert(0, '.')

# Load model with real quant
from ptq import *

# Override sys.argv to simulate the eval script
sys.argv = [
    'ptq.py',
    '--input_model', 'unsloth/Llama-3.2-1B-Instruct',
    '--per_device_eval_batch_size', '1',
    '--model_max_length', '2048',
    '--fp16', 'True', '--bf16', 'False',
    '--w_bits', '4', '--a_bits', '4', '--k_bits', '4', '--v_bits', '4',
    '--high_bits', '8', '--low_bits', '2',
    '--w_clip', '--a_asym', '--k_asym', '--v_asym',
    '--k_groupsize', '64', '--v_groupsize', '64',
    '--high_fraction', '0.125', '--low_fraction', '0.0',
    '--rotate_mode', 'resq',
    '--optimized_rotation_path', './rotation/R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-1B.bin',
    '--optimized_basis_path', './rotation/U-wikitext-512-Llama-3.2-1B.bin',
    '--rotation_granularity', 'full_shared',
    '--rotate',
    '--load_qmodel_path', './qmodels/W4A4KV4-Llama-3.2-1B-v2.pt',
]

dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=1), world_size=1, rank=0)
model_args, training_args, ptq_args = process_args_ptq()

config = AutoConfig.from_pretrained(model_args.input_model)
dtype = torch.float16
if config.tie_word_embeddings:
    config.tie_word_embeddings = False

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.input_model,
    torch_dtype=dtype, config=config,
)
model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
for name, m in model.named_modules():
    if "basis_change" in name:
        m.weight.data.copy_(torch.eye(model.config.hidden_size))
model = ptq_model(ptq_args, model, model_args)

# Get a test layer (q_proj of layer 0)
layer_name = 'model.layers.0.self_attn.q_proj'
qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
wrapper = qlayers[layer_name]

print(f"\n=== Layer: {layer_name} ===")
print(f"quantizer bits={wrapper.quantizer.bits}, high_bits_length={wrapper.quantizer.high_bits_length}")
print(f"W shape: {wrapper.module.weight.shape}")

# Create a fake input
torch.manual_seed(42)
x = torch.randn(1, 4, 2048, dtype=dtype, device='cuda')
wrapper = wrapper.cuda()

# Run fake quant forward
out_fake = wrapper.forward(x.clone())
print(f"Fake quant output: shape={out_fake.shape}, mean={out_fake.float().mean():.6f}, std={out_fake.float().std():.6f}")

# Now prepare real quant
qmodel = torch.load('./qmodels/W4A4KV4-Llama-3.2-1B-v2.pt', map_location='cpu')
gptq_quantizers = qmodel['w_quantizers']
gptq_int_weights = qmodel.get('w_int_weights', None)

base_key = layer_name + '.module'
wq_dict = {}
wq_dict['main'] = gptq_quantizers[base_key]
if base_key + ',high_quantizer' in gptq_quantizers:
    wq_dict['high'] = gptq_quantizers[base_key + ',high_quantizer']
w_int_dict = gptq_int_weights.get(base_key, None) if gptq_int_weights else None

print(f"\nw_int_dict type: {type(w_int_dict)}, shape: {w_int_dict.shape if w_int_dict is not None else None}")

# Test with direct int weights (方案B)
wrapper_b = wrapper  # reuse
wrapper_b.prepare_real_quant_weights(w_bits=4, w_quantizers=wq_dict, w_int_weights=w_int_dict)
out_real_b = wrapper_b.forward_real_quant(x.clone())
print(f"Real quant (方案B) output: shape={out_real_b.shape}, mean={out_real_b.float().mean():.6f}, std={out_real_b.float().std():.6f}")

# Compare
diff = (out_fake.float() - out_real_b.float()).abs()
print(f"\nFake vs Real-B diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
cos = torch.nn.functional.cosine_similarity(out_fake.float().flatten(), out_real_b.float().flatten(), dim=0)
print(f"Cosine similarity: {cos.item():.8f}")
