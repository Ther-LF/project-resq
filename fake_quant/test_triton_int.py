"""Test: can Triton do int8 dot product natively?"""
import triton
import triton.language as tl
import torch


@triton.jit
def test_int_dot_kernel(A_ptr, B_ptr, C_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(A_ptr + offs[:, None] * BLOCK + offs[None, :])  # int8
    b = tl.load(B_ptr + offs[:, None] * BLOCK + offs[None, :])  # int8
    # Try int8 dot product with int32 accumulation
    c = tl.dot(a, tl.trans(b))
    tl.store(C_ptr + offs[:, None] * BLOCK + offs[None, :], c)


@triton.jit
def test_int_dot_cast_kernel(A_ptr, B_ptr, C_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(A_ptr + offs[:, None] * BLOCK + offs[None, :])  # int8
    b = tl.load(B_ptr + offs[:, None] * BLOCK + offs[None, :])  # int8
    # Explicitly cast to int8 before dot
    c = tl.dot(a.to(tl.int8), tl.trans(b.to(tl.int8)), out_dtype=tl.int32)
    tl.store(C_ptr + offs[:, None] * BLOCK + offs[None, :], c)


def main():
    BLOCK = 32
    A = torch.randint(-8, 8, (BLOCK, BLOCK), dtype=torch.int8, device='cuda')
    B = torch.randint(-8, 8, (BLOCK, BLOCK), dtype=torch.int8, device='cuda')
    C_ref = (A.float() @ B.float().T).int()

    # Test 1: direct int8 dot
    C1 = torch.zeros(BLOCK, BLOCK, dtype=torch.int32, device='cuda')
    try:
        test_int_dot_kernel[(1,)](A, B, C1, BLOCK=BLOCK)
        torch.cuda.synchronize()
        diff1 = (C1 - C_ref).abs().max().item()
        print(f"Test 1 (direct int8 dot): OK, max diff={diff1}, C dtype={C1.dtype}")
    except Exception as e:
        print(f"Test 1 (direct int8 dot): FAILED - {e}")

    # Test 2: cast to int8 + out_dtype=int32
    C2 = torch.zeros(BLOCK, BLOCK, dtype=torch.int32, device='cuda')
    try:
        test_int_dot_cast_kernel[(1,)](A, B, C2, BLOCK=BLOCK)
        torch.cuda.synchronize()
        diff2 = (C2 - C_ref).abs().max().item()
        print(f"Test 2 (int8 dot out_dtype=int32): OK, max diff={diff2}, C dtype={C2.dtype}")
    except Exception as e:
        print(f"Test 2 (int8 dot out_dtype=int32): FAILED - {e}")

    # Test 3: what dtype does tl.dot produce from int8 inputs?
    print(f"\nRef dtype: {C_ref.dtype}, ref range: [{C_ref.min().item()}, {C_ref.max().item()}]")


if __name__ == '__main__':
    main()
