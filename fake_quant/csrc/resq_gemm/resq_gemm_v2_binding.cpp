/*
 * PyTorch binding — thin wrapper, no CUTLASS headers.
 *
 * gemm_s8s8(A_int8, B_int8) -> D_float32
 * grouped_gemm_s8s8(A_groups, B_groups) -> D_stacked (G,M,N)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// From resq_gemm_v2.cu
void run_gemm_s8s8(int M, int N, int K,
    const int8_t* A, const int8_t* B, float* D, cudaStream_t stream);

// From resq_gemm_grouped.cu — takes raw torch tensors, builds metadata internally
torch::Tensor launch_grouped_gemm_s8s8(torch::Tensor A_groups, torch::Tensor B_groups);

static void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cuda(), name, " must be on CUDA device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor gemm_s8s8(torch::Tensor A, torch::Tensor B)
{
    check_cuda_contiguous(A, "A");
    check_cuda_contiguous(B, "B");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A.K must match B.K");

    const int M = A.size(0), K = A.size(1), N = B.size(0);
    auto D = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

    run_gemm_s8s8(M, N, K,
        reinterpret_cast<const int8_t*>(A.data_ptr()),
        reinterpret_cast<const int8_t*>(B.data_ptr()),
        reinterpret_cast<float*>(D.data_ptr()),
        at::cuda::getCurrentCUDAStream().stream());

    return D;
}

torch::Tensor grouped_gemm_s8s8(torch::Tensor A_groups, torch::Tensor B_groups)
{
    check_cuda_contiguous(A_groups, "A_groups");
    check_cuda_contiguous(B_groups, "B_groups");
    TORCH_CHECK(A_groups.dtype() == torch::kInt8, "A_groups must be int8");
    TORCH_CHECK(B_groups.dtype() == torch::kInt8, "B_groups must be int8");
    TORCH_CHECK(A_groups.dim() == 3 && B_groups.dim() == 3,
        "A_groups and B_groups must be 3D (G, M/N, K)");
    TORCH_CHECK(A_groups.size(0) == B_groups.size(0), "group count mismatch");
    TORCH_CHECK(A_groups.size(2) == B_groups.size(2), "K mismatch");

    return launch_grouped_gemm_s8s8(A_groups, B_groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ResQ GEMM — CUTLASS 3.x Hopper INT8 GEMM kernels";
    m.def("gemm_s8s8", &gemm_s8s8, "INT8 GEMM: D_f32 = A_i8 @ B_i8^T", py::arg("A"), py::arg("B"));
    m.def("grouped_gemm_s8s8", &grouped_gemm_s8s8,
          "Grouped INT8 GEMM: (G,M,K)x(G,N,K)^T -> (G,M,N) f32", py::arg("A_groups"), py::arg("B_groups"));
}
