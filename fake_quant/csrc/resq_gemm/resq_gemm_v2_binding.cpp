/*
 * PyTorch binding for ResQ INT8 GEMM kernels.
 *
 * gemm_s8s8(A_int8, B_int8) -> D_float32
 *   Single GEMM: D[m,n] = float(Σ_k A[m,k] * B[n,k])
 *
 * grouped_gemm_s8s8(A_groups, B_groups) -> D_stacked
 *   Grouped GEMM: G independent (M,K)×(N,K)^T → stacked (G,M,N)
 *   Single kernel launch for all G groups.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Forward declarations
void run_gemm_s8s8(
    int M, int N, int K,
    const int8_t* A, const int8_t* B,
    float* D, cudaStream_t stream);

void run_grouped_gemm_s8s8(
    int ngroups, int M, int N, int K,
    const int8_t** A_ptrs_host,
    const int8_t** B_ptrs_host,
    float** D_ptrs_host,
    cudaStream_t stream);

static void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cuda(), name, " must be on CUDA device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ============================================================
// Single GEMM: A(M,K) @ B(N,K)^T -> D(M,N)
// ============================================================
torch::Tensor gemm_s8s8(torch::Tensor A, torch::Tensor B)
{
    check_cuda_contiguous(A, "A");
    check_cuda_contiguous(B, "B");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A.K must match B.K");

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

// ============================================================
// Grouped GEMM: G independent (M,K)×(N,K)^T → (G,M,N)
// A_groups: (G, M, K) int8 — G activation groups, contiguous
// B_groups: (G, N, K) int8 — G weight slices, contiguous
//   OR list of G tensors each (N, K)
// Returns: (G, M, N) float32
// ============================================================
torch::Tensor grouped_gemm_s8s8(torch::Tensor A_groups, torch::Tensor B_groups)
{
    check_cuda_contiguous(A_groups, "A_groups");
    check_cuda_contiguous(B_groups, "B_groups");
    TORCH_CHECK(A_groups.dtype() == torch::kInt8, "A_groups must be int8");
    TORCH_CHECK(B_groups.dtype() == torch::kInt8, "B_groups must be int8");
    TORCH_CHECK(A_groups.dim() == 3 && B_groups.dim() == 3,
        "A_groups and B_groups must be 3D (G, M/N, K)");

    const int G = A_groups.size(0);
    const int M = A_groups.size(1);
    const int K = A_groups.size(2);
    const int N = B_groups.size(1);

    TORCH_CHECK(B_groups.size(0) == G, "A and B must have same number of groups");
    TORCH_CHECK(B_groups.size(2) == K, "A.K must match B.K");

    // Allocate output: (G, M, N)
    auto D = torch::empty({G, M, N}, torch::TensorOptions().device(A_groups.device()).dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Build pointer arrays (host side, will be copied to device inside kernel)
    std::vector<const int8_t*> A_ptrs(G);
    std::vector<const int8_t*> B_ptrs(G);
    std::vector<float*> D_ptrs(G);

    const int8_t* A_base = reinterpret_cast<const int8_t*>(A_groups.data_ptr());
    const int8_t* B_base = reinterpret_cast<const int8_t*>(B_groups.data_ptr());
    float* D_base = reinterpret_cast<float*>(D.data_ptr());

    for (int g = 0; g < G; g++) {
        A_ptrs[g] = A_base + (size_t)g * M * K;
        B_ptrs[g] = B_base + (size_t)g * N * K;
        D_ptrs[g] = D_base + (size_t)g * M * N;
    }

    run_grouped_gemm_s8s8(G, M, N, K,
        A_ptrs.data(), B_ptrs.data(), D_ptrs.data(),
        stream);

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ResQ GEMM — CUTLASS 3.x Hopper INT8 GEMM kernels";
    m.def("gemm_s8s8", &gemm_s8s8,
          "INT8 x INT8 GEMM with INT32 accumulation. D_float = A_int8 @ B_int8^T",
          py::arg("A"), py::arg("B"));
    m.def("grouped_gemm_s8s8", &grouped_gemm_s8s8,
          "Grouped INT8 GEMM: G independent (M,K)x(N,K)^T -> (G,M,N) float. Single kernel launch.",
          py::arg("A_groups"), py::arg("B_groups"));
}
