/*
 * PyBind11 bindings for ResQ GEMM kernels.
 *
 * Exposes three functions to Python:
 *   resq_gemm.gemm_s8s8s32(A, B)  -> C  (int8 x int8 -> int32)
 *   resq_gemm.gemm_s4s4s32(A, B)  -> C  (int4 packed x int4 packed -> int32)
 *   resq_gemm.gemm_f16f16f32(A, B) -> C (fp16 x fp16 -> fp32)
 *
 * Convention: C = A @ B^T  (matching PyTorch's nn.Linear: y = x @ W^T)
 * A is (M, K), B is (N, K) [stored as row-major], result C is (M, N)
 *
 * Internally, we pass B as ColumnMajor to CUTLASS (which is equivalent
 * to B^T as RowMajor), achieving the A @ B^T semantics.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/numeric_types.h>

// Forward declarations of kernel launchers (defined in resq_gemm.cu)
void run_gemm_s8s8s32(int M, int N, int K,
                      const int8_t* A, const int8_t* B, int32_t* C,
                      cudaStream_t stream);

void run_gemm_s4s4s32(int M, int N, int K,
                      const cutlass::int4b_t* A, const cutlass::int4b_t* B, int32_t* C,
                      cudaStream_t stream);

void run_gemm_f16f16f32(int M, int N, int K,
                        const cutlass::half_t* A, const cutlass::half_t* B, float* C,
                        cudaStream_t stream);

// ============================================================
// Input validation helpers
// ============================================================

static void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cuda(), name, " must be on CUDA device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ============================================================
// INT8 x INT8 -> INT32
// ============================================================
// A: (M, K) int8, B: (N, K) int8  =>  C: (M, N) int32
// Semantics: C = A @ B^T

torch::Tensor gemm_s8s8s32(torch::Tensor A, torch::Tensor B) {
    check_cuda_contiguous(A, "A");
    check_cuda_contiguous(B, "B");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A.K must match B.K");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kInt32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    run_gemm_s8s8s32(M, N, K,
                     reinterpret_cast<const int8_t*>(A.data_ptr()),
                     reinterpret_cast<const int8_t*>(B.data_ptr()),
                     reinterpret_cast<int32_t*>(C.data_ptr()),
                     stream);

    return C;
}

// ============================================================
// INT4 x INT4 -> INT32  (packed format)
// ============================================================
// A: (M, K/2) uint8 [packed int4], B: (N, K/2) uint8 [packed int4]
// K is the logical number of int4 elements (must be even)
// Semantics: C = A @ B^T  where A, B are unpacked int4 matrices

torch::Tensor gemm_s4s4s32(torch::Tensor A_packed, torch::Tensor B_packed, int64_t K) {
    check_cuda_contiguous(A_packed, "A_packed");
    check_cuda_contiguous(B_packed, "B_packed");
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8 || A_packed.dtype() == torch::kInt8,
                "A_packed must be uint8 or int8 (packed int4)");
    TORCH_CHECK(B_packed.dtype() == torch::kUInt8 || B_packed.dtype() == torch::kInt8,
                "B_packed must be uint8 or int8 (packed int4)");
    TORCH_CHECK(A_packed.dim() == 2 && B_packed.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(K % 2 == 0, "K must be even for int4 packing");
    TORCH_CHECK(A_packed.size(1) == K / 2, "A_packed cols must be K/2");
    TORCH_CHECK(B_packed.size(1) == K / 2, "B_packed cols must be K/2");

    const int M = A_packed.size(0);
    const int N = B_packed.size(0);

    auto C = torch::empty({M, N}, torch::TensorOptions().device(A_packed.device()).dtype(torch::kInt32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    run_gemm_s4s4s32(M, N, K,
                     reinterpret_cast<const cutlass::int4b_t*>(A_packed.data_ptr()),
                     reinterpret_cast<const cutlass::int4b_t*>(B_packed.data_ptr()),
                     reinterpret_cast<int32_t*>(C.data_ptr()),
                     stream);

    return C;
}

// ============================================================
// FP16 x FP16 -> FP32
// ============================================================
// A: (M, K) fp16, B: (N, K) fp16  =>  C: (M, N) fp32

torch::Tensor gemm_f16f16f32(torch::Tensor A, torch::Tensor B) {
    check_cuda_contiguous(A, "A");
    check_cuda_contiguous(B, "B");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A.K must match B.K");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    run_gemm_f16f16f32(M, N, K,
                       reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
                       reinterpret_cast<const cutlass::half_t*>(B.data_ptr()),
                       reinterpret_cast<float*>(C.data_ptr()),
                       stream);

    return C;
}

// ============================================================
// PyBind11 module definition
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ResQ GEMM kernels using CUTLASS (INT4, INT8, FP16)";

    m.def("gemm_s8s8s32", &gemm_s8s8s32,
          "INT8 x INT8 -> INT32 GEMM. C = A @ B^T. A:(M,K) int8, B:(N,K) int8 -> C:(M,N) int32",
          py::arg("A"), py::arg("B"));

    m.def("gemm_s4s4s32", &gemm_s4s4s32,
          "INT4 x INT4 -> INT32 GEMM. C = A @ B^T. A:(M,K/2) packed, B:(N,K/2) packed, K=logical -> C:(M,N) int32",
          py::arg("A_packed"), py::arg("B_packed"), py::arg("K"));

    m.def("gemm_f16f16f32", &gemm_f16f16f32,
          "FP16 x FP16 -> FP32 GEMM. C = A @ B^T. A:(M,K) fp16, B:(N,K) fp16 -> C:(M,N) fp32",
          py::arg("A"), py::arg("B"));
}
