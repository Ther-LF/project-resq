/*
 * ResQ Grouped GEMM Kernel — CUTLASS 3.x Hopper (SM90a)
 *
 * Grouped INT8×INT8 → INT32 accumulator GEMM using PtrArray schedule.
 * Each group g computes: D_g[M,N] = A_g[M,K] @ B_g[N,K]^T
 *
 * All metadata (problem_sizes, pointer arrays, strides) built on device
 * via torch tensors in launch_grouped_gemm_s8s8. No raw cudaMalloc/cudaMemcpy.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// ============================================================
// Type definitions — pure INT8×INT8 → INT32, Grouped mode
// ============================================================

using ElementA    = int8_t;
using LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

using ElementB    = int8_t;
using LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

using ElementC     = float;
using LayoutC      = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 4

using ElementD     = float;
using LayoutD      = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = int32_t;
using ElementCompute     = float;

using ArchTag        = cutlass::arch::Sm90;
using OperatorClass  = cutlass::arch::OpClassTensorOp;

using TileShape    = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using StrideC = typename GemmKernel::InternalStrideC;
using StrideD = typename GemmKernel::InternalStrideD;

#define CUTLASS_CHECK(status)                                         \
  {                                                                   \
    cutlass::Status error = status;                                   \
    if (error != cutlass::Status::kSuccess) {                         \
      throw std::runtime_error(                                       \
          std::string("CUTLASS error: ") +                            \
          cutlassGetStatusString(error) +                             \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
    }                                                                 \
  }

// ============================================================
// Run grouped GEMM — ALL pointers are already on device.
// No cudaMalloc, no cudaMemcpy, just launch.
// ============================================================

void run_grouped_gemm_s8s8(
    int ngroups,
    // All device pointers, pre-allocated by binding layer:
    void* problem_sizes_dev,     // device: ProblemShape::UnderlyingProblemShape[ngroups]
    void* problem_sizes_host,    // host: same data for CUTLASS host-side planning
    const int8_t** A_ptrs_dev,   // device: [ngroups] pointers into A data
    StrideA* strides_A_dev,      // device: [ngroups]
    const int8_t** B_ptrs_dev,   // device: [ngroups]
    StrideB* strides_B_dev,      // device: [ngroups]
    StrideC* strides_C_dev,      // device: [ngroups]
    float** D_ptrs_dev,          // device: [ngroups] pointers into D data
    StrideD* strides_D_dev,      // device: [ngroups]
    void* workspace,             // device: pre-allocated workspace (or nullptr)
    cudaStream_t stream)
{
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    typename Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {ngroups,
         static_cast<UnderlyingProblemShape*>(problem_sizes_dev),
         static_cast<UnderlyingProblemShape*>(problem_sizes_host)},
        {A_ptrs_dev, strides_A_dev, B_ptrs_dev, strides_B_dev},
        {{1.0f, 0.0f}, nullptr, strides_C_dev, D_ptrs_dev, strides_D_dev}
    };

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(args));
    CUTLASS_CHECK(gemm_op.initialize(args, workspace, stream));
    CUTLASS_CHECK(gemm_op(stream));
}

// Helper: query workspace size for a given config
size_t get_grouped_gemm_workspace_size(int ngroups, int M, int N, int K) {
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    // Build minimal args to query workspace
    std::vector<UnderlyingProblemShape> sizes_host(ngroups, make_shape(M, N, K));

    typename Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {ngroups, nullptr, sizes_host.data()},
        {nullptr, nullptr, nullptr, nullptr},
        {{1.0f, 0.0f}, nullptr, nullptr, nullptr, nullptr}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// Helper: compute stride for given dimensions
StrideA make_stride_a(int M, int K) {
    return cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
}
StrideB make_stride_b(int N, int K) {
    return cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
}
StrideC make_stride_c(int M, int N) {
    return cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
}
StrideD make_stride_d(int M, int N) {
    return cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
}

// ============================================================
// PyTorch entry point: build device metadata + launch
//
// A_groups: (G, M, K) int8 contiguous on CUDA
// B_groups: (G, N, K) int8 contiguous on CUDA
// Returns:  (G, M, N) float32
// ============================================================

torch::Tensor launch_grouped_gemm_s8s8(torch::Tensor A_groups, torch::Tensor B_groups)
{
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    const int G = A_groups.size(0);
    const int M = A_groups.size(1);
    const int K = A_groups.size(2);
    const int N = B_groups.size(1);
    auto device = A_groups.device();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Output
    auto D = torch::empty({G, M, N}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

    // 1. Problem sizes — host vector + device copy via torch
    std::vector<UnderlyingProblemShape> sizes_host(G, make_shape(M, N, K));
    auto sizes_dev = torch::from_blob(
        sizes_host.data(), {(int64_t)(G * sizeof(UnderlyingProblemShape))}, torch::kUInt8
    ).to(device);

    // 2. Pointer arrays — host vectors + device copy
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

    auto A_ptrs_dev = torch::from_blob(
        A_ptrs_h.data(), {(int64_t)(G * sizeof(void*))}, torch::kUInt8).to(device);
    auto B_ptrs_dev = torch::from_blob(
        B_ptrs_h.data(), {(int64_t)(G * sizeof(void*))}, torch::kUInt8).to(device);
    auto D_ptrs_dev = torch::from_blob(
        D_ptrs_h.data(), {(int64_t)(G * sizeof(void*))}, torch::kUInt8).to(device);

    // 3. Strides — all identical
    auto sa = make_stride_a(M, K);
    auto sb = make_stride_b(N, K);
    auto sc = make_stride_c(M, N);
    auto sd = make_stride_d(M, N);

    std::vector<StrideA> sa_h(G, sa);
    std::vector<StrideB> sb_h(G, sb);
    std::vector<StrideC> sc_h(G, sc);
    std::vector<StrideD> sd_h(G, sd);

    auto sa_dev = torch::from_blob(sa_h.data(), {(int64_t)(G * sizeof(StrideA))}, torch::kUInt8).to(device);
    auto sb_dev = torch::from_blob(sb_h.data(), {(int64_t)(G * sizeof(StrideB))}, torch::kUInt8).to(device);
    auto sc_dev = torch::from_blob(sc_h.data(), {(int64_t)(G * sizeof(StrideC))}, torch::kUInt8).to(device);
    auto sd_dev = torch::from_blob(sd_h.data(), {(int64_t)(G * sizeof(StrideD))}, torch::kUInt8).to(device);

    // 4. Workspace
    size_t ws_size = get_grouped_gemm_workspace_size(G, M, N, K);
    torch::Tensor workspace;
    void* ws_ptr = nullptr;
    if (ws_size > 0) {
        workspace = torch::empty({(int64_t)ws_size}, torch::TensorOptions().device(device).dtype(torch::kUInt8));
        ws_ptr = workspace.data_ptr();
    }

    // 5. Launch — pure kernel, all data on device
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
}

#else

void run_grouped_gemm_s8s8(int, void*, void*, const int8_t**, void*,
    const int8_t**, void*, void*, float**, void*, void*, cudaStream_t) {
    throw std::runtime_error("SM90 not supported");
}
size_t get_grouped_gemm_workspace_size(int, int, int, int) { return 0; }

torch::Tensor launch_grouped_gemm_s8s8(torch::Tensor, torch::Tensor) {
    throw std::runtime_error("SM90 not supported");
}

#endif
