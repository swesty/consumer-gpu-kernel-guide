"""
Build script for Qwen3 RMSNorm CUDA kernels.

Targets RTX 3090 Ampere (sm_86).

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
    description="Vectorized RMSNorm CUDA kernel for Qwen3-8B on RTX 3090 Ampere",
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
                    "-arch=sm_86",           # Ampere
                    "-gencode=arch=compute_86,code=sm_86",
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
