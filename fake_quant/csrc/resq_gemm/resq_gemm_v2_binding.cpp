/*
 * Minimal test binding for CUTLASS 3.x INT8 GEMM with group-wise scale.
 * Just tests compilation and basic correctness.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declaration
void run_gemm_s8s8_grouped_scale(
    int M, int N, int K, int group_size,
    const int8_t* A, const int8_t* B, const float* scale_B,
    float* D, cudaStream_t stream);

static void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cuda(), name, " must be on CUDA device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// gemm_s8s8_grouped_scale(A, B, scale_B, group_size) -> D
// A: (M, K) int8, B: (N, K) int8, scale_B: (N, scale_k) float
// D: (M, N) float32
// Semantics: D = A @ (scale_B * B)^T  with group-wise dequant of B
torch::Tensor gemm_s8s8_grouped_scale(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B, int64_t group_size)
{
    check_cuda_contiguous(A, "A");
    check_cuda_contiguous(B, "B");
    check_cuda_contiguous(scale_B, "scale_B");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(scale_B.dtype() == torch::kFloat32, "scale_B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A.K must match B.K");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    int scale_k = (K + group_size - 1) / group_size;
    // scale_B must be (scale_k, N) contiguous — CUTLASS MN-major layout
    // This means fastest-changing dim is N (columns), slowest is scale_k (groups)
    TORCH_CHECK(scale_B.size(0) == scale_k, "scale_B must be (scale_k, N), got row=", scale_B.size(0), " expected=", scale_k);
    TORCH_CHECK(scale_B.size(1) == N, "scale_B must be (scale_k, N), got col=", scale_B.size(1), " expected=", N);

    auto D = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    run_gemm_s8s8_grouped_scale(M, N, K, group_size,
        reinterpret_cast<const int8_t*>(A.data_ptr()),
        reinterpret_cast<const int8_t*>(B.data_ptr()),
        reinterpret_cast<const float*>(scale_B.data_ptr()),
        reinterpret_cast<float*>(D.data_ptr()),
        stream);

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ResQ GEMM v2 — CUTLASS 3.x Hopper with group-wise scale";
    m.def("gemm_s8s8_grouped_scale", &gemm_s8s8_grouped_scale,
          "INT8 x INT8 GEMM with group-wise weight scale. C = A @ (dequant(B, scale))^T",
          py::arg("A"), py::arg("B"), py::arg("scale_B"), py::arg("group_size"));
}
