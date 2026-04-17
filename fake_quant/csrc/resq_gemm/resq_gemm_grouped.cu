/*
 * ResQ Grouped GEMM Kernel — CUTLASS 3.x Hopper (SM90a)
 *
 * Grouped INT8×INT8 → INT32 accumulator GEMM using PtrArray schedule.
 * Each group g computes: D_g[M,N] = A_g[M,K] @ B_g[N,K]^T
 *
 * All metadata (problem_sizes, pointer arrays, strides) must be
 * pre-allocated on device by the caller. No cudaMalloc/cudaMemcpy here.
 */

#include <cuda_runtime.h>
#include <iostream>

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

#else

void run_grouped_gemm_s8s8(int, void*, void*, const int8_t**, void*,
    const int8_t**, void*, void*, float**, void*, void*, cudaStream_t) {
    throw std::runtime_error("SM90 not supported");
}
size_t get_grouped_gemm_workspace_size(int, int, int, int) { return 0; }

#endif
