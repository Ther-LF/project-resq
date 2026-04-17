/*
 * PyTorch binding for ResQ INT8 GEMM kernels.
 *
 * gemm_s8s8(A_int8, B_int8) -> D_float32
 *   Single GEMM: D[m,n] = float(Σ_k A[m,k] * B[n,k])
 *
 * grouped_gemm_s8s8(A_groups, B_groups) -> D_stacked
 *   Grouped GEMM: G independent (M,K)×(N,K)^T → stacked (G,M,N)
 *   Single kernel launch. All metadata pre-built on device.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// From resq_gemm_v2.cu
void run_gemm_s8s8(
    int M, int N, int K,
    const int8_t* A, const int8_t* B,
    float* D, cudaStream_t stream);

// From resq_gemm_grouped.cu — forward-declare opaque stride types
// We need the actual types for building metadata, so include the needed headers.
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"

// Must match the types in resq_gemm_grouped.cu exactly
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
namespace grouped_types {
    using namespace cute;
    using ElementA = int8_t;
    using LayoutA  = cutlass::layout::RowMajor;
    using ElementB = int8_t;
    using LayoutB  = cutlass::layout::ColumnMajor;
    using ElementC = float;
    using LayoutC  = cutlass::layout::RowMajor;
    using ElementD = float;
    using LayoutD  = LayoutC;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using TileShape    = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
    constexpr int AlignmentA = 16;
    constexpr int AlignmentB = 16;
    constexpr int AlignmentC = 4;
    constexpr int AlignmentD = 4;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC *, AlignmentC,
        ElementD, LayoutD *, AlignmentD,
        EpilogueSchedule
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA *, AlignmentA,
        ElementB, LayoutB *, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename GemmKernel::InternalStrideA;
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
}
#endif

// From resq_gemm_grouped.cu
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
void run_grouped_gemm_s8s8(
    int ngroups,
    void* problem_sizes_dev,
    void* problem_sizes_host,
    const int8_t** A_ptrs_dev,
    grouped_types::StrideA* strides_A_dev,
    const int8_t** B_ptrs_dev,
    grouped_types::StrideB* strides_B_dev,
    grouped_types::StrideC* strides_C_dev,
    float** D_ptrs_dev,
    grouped_types::StrideD* strides_D_dev,
    void* workspace,
    cudaStream_t stream);

size_t get_grouped_gemm_workspace_size(int ngroups, int M, int N, int K);
#endif

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
//
// A_groups: (G, M, K) int8, contiguous
// B_groups: (G, N, K) int8, contiguous
// Returns:  (G, M, N) float32
//
// All metadata (problem sizes, pointer arrays, strides, workspace)
// built on device ONCE here, then pure kernel launch.
// ============================================================
torch::Tensor grouped_gemm_s8s8(torch::Tensor A_groups, torch::Tensor B_groups)
{
#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    TORCH_CHECK(false, "SM90 not supported");
#else
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

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = A_groups.device();

    // Output: (G, M, N)
    auto D = torch::empty({G, M, N}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

    // --- Build all metadata on device ---
    using namespace grouped_types;

    // 1. Problem sizes: all identical {M, N, K}
    std::vector<UnderlyingProblemShape> sizes_host(G, cute::make_shape(M, N, K));
    auto sizes_dev = torch::from_blob(sizes_host.data(),
        {(int64_t)(G * sizeof(UnderlyingProblemShape))}, torch::kUInt8)
        .to(device).contiguous();

    // 2. Pointer arrays
    const int8_t* A_base = reinterpret_cast<const int8_t*>(A_groups.data_ptr());
    const int8_t* B_base = reinterpret_cast<const int8_t*>(B_groups.data_ptr());
    float* D_base = reinterpret_cast<float*>(D.data_ptr());

    std::vector<const int8_t*> A_ptrs_h(G), B_ptrs_h(G);
    std::vector<float*> D_ptrs_h(G);
    for (int g = 0; g < G; g++) {
        A_ptrs_h[g] = A_base + (size_t)g * M * K;
        B_ptrs_h[g] = B_base + (size_t)g * N * K;
        D_ptrs_h[g] = D_base + (size_t)g * M * N;
    }

    auto A_ptrs_dev = torch::from_blob(A_ptrs_h.data(),
        {(int64_t)(G * sizeof(const int8_t*))}, torch::kUInt8)
        .to(device).contiguous();
    auto B_ptrs_dev = torch::from_blob(B_ptrs_h.data(),
        {(int64_t)(G * sizeof(const int8_t*))}, torch::kUInt8)
        .to(device).contiguous();
    auto D_ptrs_dev = torch::from_blob(D_ptrs_h.data(),
        {(int64_t)(G * sizeof(float*))}, torch::kUInt8)
        .to(device).contiguous();

    // 3. Strides: all identical
    auto sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto sc = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    std::vector<StrideA> sa_h(G, sa);
    std::vector<StrideB> sb_h(G, sb);
    std::vector<StrideC> sc_h(G, sc);
    std::vector<StrideD> sd_h(G, sd);

    auto sa_dev = torch::from_blob(sa_h.data(),
        {(int64_t)(G * sizeof(StrideA))}, torch::kUInt8).to(device).contiguous();
    auto sb_dev = torch::from_blob(sb_h.data(),
        {(int64_t)(G * sizeof(StrideB))}, torch::kUInt8).to(device).contiguous();
    auto sc_dev = torch::from_blob(sc_h.data(),
        {(int64_t)(G * sizeof(StrideC))}, torch::kUInt8).to(device).contiguous();
    auto sd_dev = torch::from_blob(sd_h.data(),
        {(int64_t)(G * sizeof(StrideD))}, torch::kUInt8).to(device).contiguous();

    // 4. Workspace
    size_t ws_size = get_grouped_gemm_workspace_size(G, M, N, K);
    torch::Tensor workspace;
    void* ws_ptr = nullptr;
    if (ws_size > 0) {
        workspace = torch::empty({(int64_t)ws_size}, torch::TensorOptions().device(device).dtype(torch::kUInt8));
        ws_ptr = workspace.data_ptr();
    }

    // --- Launch: pure kernel, no memcpy ---
    run_grouped_gemm_s8s8(
        G,
        sizes_dev.data_ptr(),
        sizes_host.data(),
        reinterpret_cast<const int8_t**>(A_ptrs_dev.data_ptr()),
        reinterpret_cast<StrideA*>(sa_dev.data_ptr()),
        reinterpret_cast<const int8_t**>(B_ptrs_dev.data_ptr()),
        reinterpret_cast<StrideB*>(sb_dev.data_ptr()),
        reinterpret_cast<StrideC*>(sc_dev.data_ptr()),
        reinterpret_cast<float**>(D_ptrs_dev.data_ptr()),
        reinterpret_cast<StrideD*>(sd_dev.data_ptr()),
        ws_ptr,
        stream);

    return D;
#endif
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
