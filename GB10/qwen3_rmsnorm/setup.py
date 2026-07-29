"""
Build script for Qwen3 RMSNorm CUDA kernels.

Targets NVIDIA GB10 (DGX Spark, sm_121a).

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
                    # GB10 DGX Spark: sm_121a native + sm_120f family fallback (CUDA 12.9+)
                    "-gencode=arch=compute_121a,code=sm_121a",   # CC 12.1 only
                    "-gencode=arch=compute_120f,code=sm_120f",   # Any CC 12.x, no JIT
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
