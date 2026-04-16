/*
 * ResQ GEMM Kernel — CUTLASS 3.x Hopper (SM90a)
 *
 * Pure INT8×INT8 → INT32 accumulator GEMM using standard CollectiveBuilder.
 * No MixedInput (which converts to float). Uses real INT8 Tensor Core.
 *
 * Semantics:
 *   D[m,n] = A_int8[m,:] @ B_int8[n,:]^T   (INT32 accumulation)
 *   Output is float32 (converted from int32 in epilogue).
 *
 * Scale/bias/shift are handled in Python post-processing for now.
 * The kernel is intentionally minimal: just INT8 GEMM with correct TC usage.
 */

#include <cuda_runtime.h>
#include <iostream>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// ============================================================
// Type definitions — pure INT8×INT8 → INT32
// ============================================================

// A = activation: int8, RowMajor
using ElementA    = int8_t;
using LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

// B = weight: int8, ColumnMajor (stored as RowMajor N×K, passed as ColMajor K×N)
using ElementB    = int8_t;
using LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

// Output: float32 (epilogue converts int32 accumulator to float)
using ElementC     = float;
using LayoutC      = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 4

using ElementD     = float;
using LayoutD      = LayoutC;
constexpr int AlignmentD = AlignmentC;

// INT32 accumulator — this is the whole point: use INT8 TC with INT32 acc
using ElementAccumulator = int32_t;
using ElementCompute     = float;

// Architecture
using ArchTag        = cutlass::arch::Sm90;
using OperatorClass  = cutlass::arch::OpClassTensorOp;

// Tile shape — TileShapeK=128 is typical for INT8 GEMM
using TileShape    = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// Standard schedules (NOT MixedInput)
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// ============================================================
// Epilogue: convert INT32 acc to float output, alpha=1 beta=0
// ============================================================

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// ============================================================
// Mainloop: standard INT8×INT8 → INT32
// ============================================================

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
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
// Run: pure INT8 GEMM, D_float = int32_to_float(A_int8 @ B_int8^T)
// ============================================================

void run_gemm_s8s8(
    int M, int N, int K,
    const int8_t* A,   // (M, K) RowMajor
    const int8_t* B,   // (N, K) RowMajor (passed as ColMajor K×N)
    float* D,          // (M, N) RowMajor output
    cudaStream_t stream = nullptr)
{
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {A, stride_A, B, stride_B},
        {{1.0f, 0.0f}, nullptr, stride_C, D, stride_D}
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

    if (workspace) cudaFree(workspace);
}

#else
void run_gemm_s8s8(int, int, int, const int8_t*, const int8_t*, float*, cudaStream_t) {
    throw std::runtime_error("SM90 not supported");
}
#endif
