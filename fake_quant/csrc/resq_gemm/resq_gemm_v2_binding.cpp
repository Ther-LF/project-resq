/*
 * PyTorch binding for ResQ INT8 GEMM kernel.
 *
 * gemm_s8s8(A_int8, B_int8) -> D_float32
 *   A: (M, K) int8, activation (shifted)
 *   B: (N, K) int8, weight (centered)
 *   D: (M, N) float32, raw INT GEMM result (before scale/bias)
 *
 * Semantics: D[m,n] = float(Σ_k A[m,k] * B[n,k])  using INT8 TC + INT32 acc
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declaration
void run_gemm_s8s8(
    int M, int N, int K,
    const int8_t* A, const int8_t* B,
    float* D, cudaStream_t stream);

static void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cuda(), name, " must be on CUDA device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// gemm_s8s8(A, B) -> D
// A: (M, K) int8, B: (N, K) int8
// D: (M, N) float32 = A_int8 @ B_int8^T with INT32 accumulation
torch::Tensor gemm_s8s8(torch::Tensor A, torch::Tensor B)
{
    check_cuda_contiguous(A, "A");
    check_cuda_contiguous(B, "B");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A.K must match B.K, got ", A.size(1), " vs ", B.size(1));

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto D = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    run_gemm_s8s8(M, N, K,
        reinterpret_cast<const int8_t*>(A.data_ptr()),
        reinterpret_cast<const int8_t*>(B.data_ptr()),
        reinterpret_cast<float*>(D.data_ptr()),
        stream);

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ResQ GEMM — CUTLASS 3.x Hopper INT8×INT8→INT32 GEMM";
    m.def("gemm_s8s8", &gemm_s8s8,
          "INT8 x INT8 GEMM with INT32 accumulation. D_float = A_int8 @ B_int8^T",
          py::arg("A"), py::arg("B"));
}
