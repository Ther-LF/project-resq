/*
 * ResQ Grouped GEMM Kernel — CUTLASS 3.x Hopper (SM90a)
 *
 * Grouped INT8×INT8 → INT32 accumulator GEMM using PtrArray schedule.
 * Each group g computes: D_g[M,N] = A_g[M,K] @ B_g[N,K]^T
 *
 * Used for per-group activation quantization (o_proj):
 *   - G groups, each with independent A pointer and B column slice
 *   - All groups share same M, N, K dimensions
 *   - Output: G separate (M,N) float32 matrices
 *
 * Scale/bias/shift handled in Python post-processing.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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

// TileShape: K=64 to handle K=56 padded to 64
using TileShape    = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

// Grouped/PtrArray schedules
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;

// Problem shape for grouped GEMM
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

// ============================================================
// Epilogue: convert INT32 acc to float output
// ============================================================

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC *, AlignmentC,       // pointer type for grouped
    ElementD, LayoutD *, AlignmentD,       // pointer type for grouped
    EpilogueSchedule
>::CollectiveOp;

// ============================================================
// Mainloop: INT8×INT8 → INT32, grouped
// ============================================================

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,       // pointer type for grouped
    ElementB, LayoutB *, AlignmentB,       // pointer type for grouped
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

// ============================================================
// Stride types
// ============================================================
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

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
// Run grouped GEMM: G independent (M,K)×(N,K)^T → (M,N) each
// ============================================================

void run_grouped_gemm_s8s8(
    int ngroups, int M, int N, int K,
    const int8_t** A_ptrs_host,  // [ngroups] pointers to (M, K) RowMajor
    const int8_t** B_ptrs_host,  // [ngroups] pointers to (N, K) RowMajor→ColMajor
    float** D_ptrs_host,         // [ngroups] pointers to (M, N) RowMajor output
    cudaStream_t stream = nullptr)
{
    // Build problem sizes (all identical)
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
    std::vector<UnderlyingProblemShape> problem_sizes_host(ngroups);
    for (int i = 0; i < ngroups; i++) {
        problem_sizes_host[i] = make_shape(M, N, K);
    }

    // Build strides (all identical since same M,N,K)
    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

    std::vector<StrideA> strides_A_host(ngroups, stride_a);
    std::vector<StrideB> strides_B_host(ngroups, stride_b);
    std::vector<StrideC> strides_C_host(ngroups, stride_c);
    std::vector<StrideD> strides_D_host(ngroups, stride_d);

    // Allocate device arrays
    UnderlyingProblemShape* problem_sizes_dev;
    const int8_t** A_ptrs_dev;
    const int8_t** B_ptrs_dev;
    float** D_ptrs_dev;
    StrideA* strides_A_dev;
    StrideB* strides_B_dev;
    StrideC* strides_C_dev;
    StrideD* strides_D_dev;

    cudaMalloc(&problem_sizes_dev, ngroups * sizeof(UnderlyingProblemShape));
    cudaMalloc(&A_ptrs_dev, ngroups * sizeof(const int8_t*));
    cudaMalloc(&B_ptrs_dev, ngroups * sizeof(const int8_t*));
    cudaMalloc(&D_ptrs_dev, ngroups * sizeof(float*));
    cudaMalloc(&strides_A_dev, ngroups * sizeof(StrideA));
    cudaMalloc(&strides_B_dev, ngroups * sizeof(StrideB));
    cudaMalloc(&strides_C_dev, ngroups * sizeof(StrideC));
    cudaMalloc(&strides_D_dev, ngroups * sizeof(StrideD));

    cudaMemcpyAsync(problem_sizes_dev, problem_sizes_host.data(),
        ngroups * sizeof(UnderlyingProblemShape), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(A_ptrs_dev, A_ptrs_host,
        ngroups * sizeof(const int8_t*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(B_ptrs_dev, B_ptrs_host,
        ngroups * sizeof(const int8_t*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(D_ptrs_dev, D_ptrs_host,
        ngroups * sizeof(float*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(strides_A_dev, strides_A_host.data(),
        ngroups * sizeof(StrideA), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(strides_B_dev, strides_B_host.data(),
        ngroups * sizeof(StrideB), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(strides_C_dev, strides_C_host.data(),
        ngroups * sizeof(StrideC), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(strides_D_dev, strides_D_host.data(),
        ngroups * sizeof(StrideD), cudaMemcpyHostToDevice, stream);

    typename Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {ngroups, problem_sizes_dev, problem_sizes_host.data()},
        {A_ptrs_dev, strides_A_dev, B_ptrs_dev, strides_B_dev},
        {{1.0f, 0.0f}, nullptr, strides_C_dev, D_ptrs_dev, strides_D_dev}
    };

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(args));

    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    CUTLASS_CHECK(gemm_op.initialize(args, workspace, stream));
    CUTLASS_CHECK(gemm_op(stream));

    // Sync before freeing temp arrays
    cudaStreamSynchronize(stream);

    cudaFree(problem_sizes_dev);
    cudaFree(A_ptrs_dev);
    cudaFree(B_ptrs_dev);
    cudaFree(D_ptrs_dev);
    cudaFree(strides_A_dev);
    cudaFree(strides_B_dev);
    cudaFree(strides_C_dev);
    cudaFree(strides_D_dev);
    if (workspace) cudaFree(workspace);
}

#else
void run_grouped_gemm_s8s8(int, int, int, int,
    const int8_t**, const int8_t**, float**, cudaStream_t) {
    throw std::runtime_error("SM90 not supported");
}
#endif
