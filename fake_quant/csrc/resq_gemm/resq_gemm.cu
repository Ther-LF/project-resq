/*
 * ResQ GEMM Kernels using CUTLASS
 *
 * Provides three GEMM operations for the ResQ mixed-precision benchmark:
 * 1. INT8  x INT8  -> INT32 accumulation (high precision group)
 * 2. INT4  x INT4  -> INT32 accumulation (main precision group)
 * 3. FP16  x FP16  -> FP32  accumulation (baseline)
 *
 * All kernels target SM90 (Hopper) Tensor Cores.
 * Uses CUTLASS 2.x device-level API for simplicity.
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

// ============================================================
// Helper: check CUTLASS status
// ============================================================
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
// 1. INT8 x INT8 -> INT32  (SM80 Tensor Core, works on SM90)
// ============================================================
// A: RowMajor (M, K) int8
// B: ColumnMajor (K, N) int8  [i.e. B^T is RowMajor (N, K)]
// C: RowMajor (M, N) int32

using Gemm_s8s8s32 = cutlass::gemm::device::Gemm<
    int8_t,                                // ElementA
    cutlass::layout::RowMajor,             // LayoutA
    int8_t,                                // ElementB
    cutlass::layout::ColumnMajor,          // LayoutB
    int32_t,                               // ElementC
    cutlass::layout::RowMajor,             // LayoutC
    int32_t,                               // ElementAccumulator
    cutlass::arch::OpClassTensorOp,        // OperatorClass
    cutlass::arch::Sm80                    // ArchTag (SM80 works on SM90)
>;

void run_gemm_s8s8s32(
    int M, int N, int K,
    const int8_t* A, const int8_t* B, int32_t* C,
    cudaStream_t stream = nullptr)
{
    Gemm_s8s8s32 gemm_op;
    Gemm_s8s8s32::Arguments args(
        {M, N, K},           // problem size
        {A, K},              // A (RowMajor stride = K)
        {B, K},              // B (ColumnMajor stride = K)
        {C, N},              // C (RowMajor stride = N)
        {C, N},              // D = C (in-place)
        {1, 0}               // alpha=1, beta=0
    );

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

// ============================================================
// 2. INT4 x INT4 -> INT32  (SM80 Tensor Core)
// ============================================================
// Uses cutlass::int4b_t (packed 4-bit integers)
// A: RowMajor (M, K) int4b_t  — K elements, stored as K/2 bytes
// B: ColumnMajor (K, N) int4b_t
// C: RowMajor (M, N) int32_t

using Gemm_s4s4s32 = cutlass::gemm::device::Gemm<
    cutlass::int4b_t,                      // ElementA
    cutlass::layout::RowMajor,             // LayoutA
    cutlass::int4b_t,                      // ElementB
    cutlass::layout::ColumnMajor,          // LayoutB
    int32_t,                               // ElementC
    cutlass::layout::RowMajor,             // LayoutC
    int32_t,                               // ElementAccumulator
    cutlass::arch::OpClassTensorOp,        // OperatorClass
    cutlass::arch::Sm80                    // ArchTag
>;

void run_gemm_s4s4s32(
    int M, int N, int K,
    const cutlass::int4b_t* A, const cutlass::int4b_t* B, int32_t* C,
    cudaStream_t stream = nullptr)
{
    Gemm_s4s4s32 gemm_op;
    Gemm_s4s4s32::Arguments args(
        {M, N, K},
        {A, K},              // A stride
        {B, K},              // B stride (ColumnMajor)
        {C, N},
        {C, N},
        {1, 0}
    );

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

// ============================================================
// 3. FP16 x FP16 -> FP32  (Tensor Core baseline)
// ============================================================

using Gemm_f16f16f32 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                       // ElementA
    cutlass::layout::RowMajor,             // LayoutA
    cutlass::half_t,                       // ElementB
    cutlass::layout::ColumnMajor,          // LayoutB
    float,                                 // ElementC
    cutlass::layout::RowMajor,             // LayoutC
    float,                                 // ElementAccumulator
    cutlass::arch::OpClassTensorOp,        // OperatorClass
    cutlass::arch::Sm80                    // ArchTag
>;

void run_gemm_f16f16f32(
    int M, int N, int K,
    const cutlass::half_t* A, const cutlass::half_t* B, float* C,
    cudaStream_t stream = nullptr)
{
    Gemm_f16f16f32 gemm_op;
    Gemm_f16f16f32::Arguments args(
        {M, N, K},
        {A, K},
        {B, K},
        {C, N},
        {C, N},
        {1.0f, 0.0f}
    );

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
