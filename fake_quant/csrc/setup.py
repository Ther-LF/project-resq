"""Build script for ResQ GEMM CUTLASS extension.

Usage:
    cd fake_quant/csrc
    pip install -e .

Or:
    python setup.py install
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUTLASS should be cloned at csrc/cutlass/
cutlass_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cutlass')
if not os.path.exists(os.path.join(cutlass_dir, 'include', 'cutlass', 'cutlass.h')):
    raise RuntimeError(
        f"CUTLASS not found at {cutlass_dir}. "
        "Please run: git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git"
    )

cutlass_include_dirs = [
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools', 'util', 'include'),
]

setup(
    name='resq_gemm',
    version='0.1.0',
    description='ResQ GEMM kernels using CUTLASS (INT4, INT8, FP16)',
    ext_modules=[
        CUDAExtension(
            name='resq_gemm',
            sources=[
                'resq_gemm/resq_gemm.cu',
                'resq_gemm/resq_gemm_binding.cpp',
            ],
            include_dirs=cutlass_include_dirs,
            extra_compile_args={
                'cxx': ['-std=c++17', '-O3'],
                'nvcc': [
                    '-arch=sm_80',          # SM80 GEMM works on SM90
                    '-std=c++17',
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '-DCUTLASS_VERSIONS_GENERATED',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
