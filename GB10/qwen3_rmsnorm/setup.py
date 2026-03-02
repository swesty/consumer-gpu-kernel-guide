"""
Build script for Qwen3 RMSNorm CUDA kernels.

Targets NVIDIA GB10 (DGX Spark, sm_120 / sm_121).

Build:
    pip install -e .
    # or
    uv pip install -e .
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qwen3_kernels",
    version="0.1.0",
    description="Vectorized RMSNorm CUDA kernel for Qwen3-8B on NVIDIA GB10 (DGX Spark)",
    packages=["qwen3_kernels"],
    package_dir={"qwen3_kernels": "torch-ext/qwen3_kernels"},
    ext_modules=[
        CUDAExtension(
            name="qwen3_kernels._C",
            sources=[
                "torch-ext/torch_binding.cpp",
                "kernel_src/rmsnorm.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    # GB10 DGX Spark: sm_120 base, forward-compatible with sm_121
                    "-gencode=arch=compute_120,code=sm_120",
                    "-gencode=arch=compute_120,code=compute_120",  # PTX for JIT on sm_121
                    "--use_fast_math",
                    "-lineinfo",             # For profiling with ncu/nsys
                    "--threads=4",           # Parallel compilation
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.6.0"],
)
