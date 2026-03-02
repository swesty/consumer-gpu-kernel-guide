/*
 * PyTorch C++ bindings for Qwen3 RMSNorm kernel
 * Dispatches to BF16/FP16/FP32 vectorized CUDA kernels
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA kernel entry points (defined in rmsnorm.cu)
extern "C" {
void rmsnorm_forward_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, int num_rows, int hidden_size,
    float eps, cudaStream_t stream);

void rmsnorm_forward_fp16(
    __half* output, const __half* input,
    const __half* weight, int num_rows, int hidden_size,
    float eps, cudaStream_t stream);

void rmsnorm_forward_fp32(
    float* output, const float* input,
    const float* weight, int num_rows, int hidden_size,
    float eps, cudaStream_t stream);
}

void rmsnorm_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int hidden_size = input.size(-1);
    const int num_rows = input.numel() / hidden_size;

    TORCH_CHECK(weight.numel() == hidden_size,
        "weight size (", weight.numel(), ") must match hidden_size (", hidden_size, ")");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == at::kBFloat16) {
        rmsnorm_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            num_rows, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kHalf) {
        rmsnorm_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            num_rows, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        rmsnorm_forward_fp32(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            static_cast<const float*>(weight.data_ptr()),
            num_rows, hidden_size, eps, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", input.scalar_type(),
            ". Supported: bfloat16, float16, float32");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward,
          "Vectorized RMSNorm forward (BF16/FP16/FP32) for RTX 5090 Blackwell",
          py::arg("output"), py::arg("input"), py::arg("weight"), py::arg("eps"));
}
