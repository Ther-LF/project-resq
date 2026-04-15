"""Build script for ResQ GEMM v2 — CUTLASS 3.x Hopper test."""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cutlass_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cutlass')
if not os.path.exists(os.path.join(cutlass_dir, 'include', 'cutlass', 'cutlass.h')):
    raise RuntimeError(f"CUTLASS not found at {cutlass_dir}.")

cutlass_include_dirs = [
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools', 'util', 'include'),
]

setup(
    name='resq_gemm_v2',
    version='0.2.0',
    ext_modules=[
        CUDAExtension(
            name='resq_gemm_v2',
            sources=[
                'resq_gemm/resq_gemm_v2.cu',
                'resq_gemm/resq_gemm_v2_binding.cpp',
            ],
            include_dirs=cutlass_include_dirs,
            extra_compile_args={
                'cxx': ['-std=c++17', '-O3'],
                'nvcc': [
                    '-arch=sm_90',
                    '-std=c++17',
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '-DCUTLASS_VERSIONS_GENERATED',
                    '-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
