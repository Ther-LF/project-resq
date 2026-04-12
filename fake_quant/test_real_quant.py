"""Quick single-layer test: compare fake quant vs real quant output.

Usage: cd project-resq/fake_quant && python test_real_quant.py
"""
import torch
import sys
sys.path.insert(0, '.')

from utils.quant_utils import ActQuantWrapper, ActQuantizer


def test_single_layer():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    # Create a simple linear layer
    K, N = 2048, 2048
    linear = torch.nn.Linear(K, N, bias=False).to(device).to(dtype)

    # Wrap with ActQuantWrapper
    wrapper = ActQuantWrapper(linear).to(device)

    # Configure quantizer: 4-bit main, 8-bit high (1/8), asym, per-token
    high_bits_length = K // 8  # 256
    wrapper.quantizer.configure(
        bits=4,
        groupsize=-1,
        sym=False,
        clip_ratio=1.0,
        high_bits_length=high_bits_length,
        high_bits=8,
        low_bits_length=0,
        low_bits=2,
    )

    # No output quantization for simplicity
    # No Hadamard rotation for simplicity

    # Test input: batch=2, seq=4, hidden=2048
    x = torch.randn(2, 4, K, device=device, dtype=dtype)

    # Run fake quant forward
    with torch.no_grad():
        y_fake = wrapper.forward(x)

    # Prepare real quant weights and run real quant forward
    wrapper.prepare_real_quant_weights()
    assert wrapper._real_quant_ready, "Real quant preparation failed"

    with torch.no_grad():
        y_real = wrapper.forward_real_quant(x)

    # Compare
    print(f"Fake quant output shape: {y_fake.shape}")
    print(f"Real quant output shape: {y_real.shape}")

    diff = (y_fake - y_real).abs()
    rel_diff = diff / (y_fake.abs() + 1e-8)

    print(f"Max absolute diff:  {diff.max().item():.6f}")
    print(f"Mean absolute diff: {diff.mean().item():.6f}")
    print(f"Max relative diff:  {rel_diff.max().item():.6f}")
    print(f"Mean relative diff: {rel_diff.mean().item():.6f}")

    # They won't be exactly equal because:
    # - fake quant uses fp16 weight directly, real quant quantizes weight to int then back
    # - But both should be close
    # The key question is: does the ACTIVATION quantization produce the same result?
    # Let's also test with identity weight to isolate activation quant
    print("\n--- Testing with identity-like weight (isolate activation quant) ---")
    # Use a small K to make identity feasible
    K_small, N_small = 64, 64
    linear_small = torch.nn.Linear(K_small, N_small, bias=False).to(device).to(dtype)
    # Set weight to identity
    with torch.no_grad():
        linear_small.weight.copy_(torch.eye(K_small, device=device, dtype=dtype))

    wrapper_small = ActQuantWrapper(linear_small).to(device)
    wrapper_small.quantizer.configure(
        bits=4, groupsize=-1, sym=False, clip_ratio=1.0,
        high_bits_length=K_small // 8, high_bits=8,
        low_bits_length=0, low_bits=2,
    )

    x_small = torch.randn(2, 4, K_small, device=device, dtype=dtype)

    with torch.no_grad():
        y_fake_small = wrapper_small.forward(x_small)

    wrapper_small.prepare_real_quant_weights()
    with torch.no_grad():
        y_real_small = wrapper_small.forward_real_quant(x_small)

    diff_small = (y_fake_small - y_real_small).abs()
    print(f"Max absolute diff:  {diff_small.max().item():.6f}")
    print(f"Mean absolute diff: {diff_small.mean().item():.6f}")

    # For identity weight with symmetric weight quant, the weight quantization
    # introduces some error. But the activation quantization path should be exact.

    print("\n--- PASS ---" if diff.max().item() < 5.0 else "\n--- NEEDS INVESTIGATION ---")


if __name__ == '__main__':
    test_single_layer()
