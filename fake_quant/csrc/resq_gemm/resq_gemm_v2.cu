/*
 * ResQ GEMM Kernels — CUTLASS 3.x Hopper (SM90)
 *
 * Based on CUTLASS example 55 (hopper_mixed_dtype_gemm).
 * Attempts same-type INT8×INT8 with group-wise activation scale.
 *
 * Step 1: Minimal test — does CollectiveBuilder accept tuple<int8_t, float> for
 *         same-type GEMM with group-wise scale?
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
// Type definitions
// ============================================================

// A = activation: int8, RowMajor — the "wide" type
using ElementA    = int8_t;
using LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

// B = weight: int8, ColumnMajor (stored as RowMajor N×K, passed as ColMajor K×N)
// B is the operand that carries the group-wise scale
using ElementB    = int8_t;
using LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

// Scale and zero types for B's group-wise dequant
using ElementScale = float;
using ElementZero  = float;

// Output
using ElementC     = float;
using LayoutC      = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 4

using ElementD     = float;
using LayoutD      = LayoutC;
constexpr int AlignmentD = AlignmentC;

// Accumulator and compute
using ElementAccumulator = int32_t;
using ElementCompute     = float;

// Architecture
using ArchTag       = cutlass::arch::Sm90;
using OperatorClass  = cutlass::arch::OpClassTensorOp;

// Tile shapes
constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<ElementA>::value;  // 128 for int8
using TileShape    = Shape<_128, _128, cute::Int<TileShapeK>>;
using ClusterShape = Shape<_1, _1, _1>;

// Schedules
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// For the swap trick (same as example 55)
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// ============================================================
// Epilogue: simple alpha*acc (no bias for now)
// ============================================================

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    // Transpose C/D layouts due to swap trick
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// ============================================================
// Mainloop: INT8 × (INT8 + Scale) with group-wise dequant
// ============================================================
// Following example 55: B is the "narrow" type with scale.
// We swap A and B so that B (with scale) goes through register file.
// tuple<ElementB, ElementScale> tells the builder to do group-wise dequant on B.

using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    // Swapped: B (with scale) first, A second
    cute::tuple<ElementB, ElementScale>, LayoutB_Transpose, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloopScaleOnly,
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

// Scale stride: (N, scale_k) row-major
using StrideS = cutlass::detail::TagToStrideB_t<cutlass::layout::RowMajor>;

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
// Run function
// ============================================================

void run_gemm_s8s8_grouped_scale(
    int M, int N, int K, int group_size,
    const int8_t* A,       // (M, K) activation, RowMajor
    const int8_t* B,       // (N, K) weight, RowMajor (passed as ColMajor)
    const float* scale_B,  // (N, scale_k) per-group weight scale, scale_k = ceil(K/group_size)
    float* D,              // (M, N) output
    cudaStream_t stream = nullptr)
{
    int scale_k = (K + group_size - 1) / group_size;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    // Transpose strides for C/D due to swap
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, 1));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(N, scale_k, 1));

    // Arguments: swapped problem shape (N, M, K) due to explicit swap
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K, 1},  // problem shape: swapped M and N
        {B, stride_B, A, stride_A, scale_B, stride_S, group_size},
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
// Fallback for non-SM90
void run_gemm_s8s8_grouped_scale(int, int, int, int, const int8_t*, const int8_t*, const float*, float*, cudaStream_t) {
    throw std::runtime_error("SM90 not supported");
}
#endif
