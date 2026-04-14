"""Test: Triton int4 support via tl.dot"""
import triton
import triton.language as tl
import torch


@triton.jit
def test_int4_dot_kernel(A_ptr, B_ptr, C_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    """A, B are packed int4 (uint8, 2 values per byte). K is logical int4 count."""
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K // 2)  # packed dimension

    # Load packed bytes
    a_packed = tl.load(A_ptr + offs_m[:, None] * (K // 2) + offs_k[None, :])  # uint8
    b_packed = tl.load(B_ptr + offs_n[:, None] * (K // 2) + offs_k[None, :])  # uint8

    # Unpack: low nibble = even index, high nibble = odd index
    # Sign extend 4-bit to 8-bit: value & 0xF, then sign extend if >= 8
    a_lo = (a_packed & 0xF).to(tl.int8)
    a_hi = ((a_packed >> 4) & 0xF).to(tl.int8)
    b_lo = (b_packed & 0xF).to(tl.int8)
    b_hi = ((b_packed >> 4) & 0xF).to(tl.int8)

    # Sign extend: if value >= 8, subtract 16
    a_lo = tl.where(a_lo >= 8, a_lo - 16, a_lo)
    a_hi = tl.where(a_hi >= 8, a_hi - 16, a_hi)
    b_lo = tl.where(b_lo >= 8, b_lo - 16, b_lo)
    b_hi = tl.where(b_hi >= 8, b_hi - 16, b_hi)

    # Manual dot: sum over K
    # a has shape (BLOCK, K/2) unpacked to (BLOCK, K) via lo/hi
    # We compute dot as sum of lo*lo + hi*hi contributions
    # Actually, interleaved: a[m, 2k] = a_lo[m,k], a[m, 2k+1] = a_hi[m,k]
    # dot = sum_k a[m,k]*b[n,k] = sum_j (a_lo[m,j]*b_lo[n,j] + a_hi[m,j]*b_hi[n,j])

    c_lo = tl.dot(a_lo, tl.trans(b_lo), out_dtype=tl.int32)
    c_hi = tl.dot(a_hi, tl.trans(b_hi), out_dtype=tl.int32)
    c = c_lo + c_hi

    tl.store(C_ptr + offs_m[:, None] * BLOCK + offs_n[None, :], c)


def pack_int4_signed(vals):
    """Pack signed int4 [-8,7] into uint8 (low nibble first)."""
    v = vals.to(torch.int8)
    lo = v[:, 0::2] & 0xF
    hi = v[:, 1::2] & 0xF
    return ((hi << 4) | lo).to(torch.uint8)


def main():
    BLOCK = 32
    K = 64  # logical int4 elements

    A_vals = torch.randint(-8, 7, (BLOCK, K), dtype=torch.int8, device='cuda')
    B_vals = torch.randint(-8, 7, (BLOCK, K), dtype=torch.int8, device='cuda')

    A_packed = pack_int4_signed(A_vals)  # (BLOCK, K//2) uint8
    B_packed = pack_int4_signed(B_vals)

    C = torch.zeros(BLOCK, BLOCK, dtype=torch.int32, device='cuda')
    C_ref = (A_vals.float() @ B_vals.float().T).int()

    try:
        test_int4_dot_kernel[(1,)](A_packed, B_packed, C, K=K, BLOCK=BLOCK)
        torch.cuda.synchronize()
        diff = (C - C_ref).abs().max().item()
        print(f"INT4 unpack+dot: max diff={diff}")
        if diff == 0:
            print("PERFECT: Triton int4 (unpack in kernel + int8 dot) works!")
        else:
            print(f"Mismatch! Example C[0,:4]={C[0,:4].tolist()} vs ref={C_ref[0,:4].tolist()}")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == '__main__':
    main()
