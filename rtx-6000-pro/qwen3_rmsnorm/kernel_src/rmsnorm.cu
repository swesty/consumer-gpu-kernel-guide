/*
 * Vectorized RMSNorm CUDA Kernel — RTX 6000 Pro Blackwell (sm_120)
 *
 * Optimized for Qwen3-8B model in HuggingFace transformers:
 *   - hidden_size = 4096  (layer norms)
 *   - head_dim    = 128   (QK norms)
 *   - eps         = 1e-6
 *   - dtype       = bfloat16 / float16 / float32
 *
 * RTX 6000 Pro Blackwell (GB202) target parameters:
 *   - Compute capability: sm_120
 *   - SMs: 188
 *   - Memory bandwidth: 1.79 TB/s (GDDR7, 512-bit)
 *   - L2 cache: 128 MB
 *   - Shared memory/SM: 128 KB (configurable up to 228 KB)
 *   - Max threads/SM: 2048
 *
 * Vectorization strategy:
 *   - BF16: __nv_bfloat162 (32-bit loads, 2 elements)
 *   - FP16: __half2         (32-bit loads, 2 elements)
 *   - FP32: float4          (128-bit loads, 4 elements)
 *
 * Formula: y = (x / sqrt(mean(x^2) + eps)) * weight
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Constants tuned for RTX 6000 Pro Blackwell
// ---------------------------------------------------------------------------
constexpr int WARP_SIZE   = 32;
constexpr int MAX_THREADS = 1024;

// Qwen3-8B primary dimensions (for static unroll hints)
constexpr int QWEN3_HIDDEN = 4096;  // hidden_size
constexpr int QWEN3_HEAD   = 128;   // head_dim (QK norm)

// ---------------------------------------------------------------------------
// Type conversion helpers
// Required because PyTorch compiles with -D__CUDA_NO_HALF_OPERATORS__
// ---------------------------------------------------------------------------
__device__ __forceinline__ float to_float(float x)            { return x; }
__device__ __forceinline__ float to_float(__half x)           { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x)    { return __bfloat162float(x); }

__device__ __forceinline__ float       from_float(float x, float*)            { return x; }
__device__ __forceinline__ __half      from_float(float x, __half*)           { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// ---------------------------------------------------------------------------
// Warp-level reduction via shuffle
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction: warp shuffle → shared memory → warp shuffle
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    const int num_warps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

// ===========================================================================
// Scalar fallback kernel (for odd hidden_size or small dimensions)
// ===========================================================================
template <typename scalar_t>
__global__ void rmsnorm_kernel_scalar(
    scalar_t*       __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t* row_in  = input  + row * hidden_size;
    scalar_t*       row_out = output + row * hidden_size;

    // Phase 1: sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = to_float(row_in[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();
    const float rms_inv = s_rms_inv;

    // Phase 2: normalize and scale
    for (int i = tid; i < hidden_size; i += stride) {
        float v = to_float(row_in[i]);
        float w = to_float(weight[i]);
        row_out[i] = from_float(v * rms_inv * w, (scalar_t*)nullptr);
    }
}

// ===========================================================================
// Vectorized BF16 kernel (__nv_bfloat162 — 2 elements per 32-bit load)
// ===========================================================================
__global__ void rmsnorm_kernel_bf16_vec(
    __nv_bfloat16*       __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* row_in  = input  + row * hidden_size;
    __nv_bfloat16*       row_out = output + row * hidden_size;

    const int vec_hidden = hidden_size / 2;
    const __nv_bfloat162* vec_in = reinterpret_cast<const __nv_bfloat162*>(row_in);

    // Phase 1: vectorized sum-of-squares
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v = vec_in[i];
        float v0 = __bfloat162float(v.x);
        float v1 = __bfloat162float(v.y);
        sum_sq += v0 * v0 + v1 * v1;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    // RMS inverse
    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();
    const float rms_inv = s_rms_inv;

    // Phase 2: vectorized normalize + scale
    const __nv_bfloat162* vec_w   = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162*       vec_out = reinterpret_cast<__nv_bfloat162*>(row_out);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v = vec_in[i];
        __nv_bfloat162 w = vec_w[i];

        float v0 = __bfloat162float(v.x);
        float v1 = __bfloat162float(v.y);
        float w0 = __bfloat162float(w.x);
        float w1 = __bfloat162float(w.y);

        __nv_bfloat162 result;
        result.x = __float2bfloat16(v0 * rms_inv * w0);
        result.y = __float2bfloat16(v1 * rms_inv * w1);
        vec_out[i] = result;
    }
}

// ===========================================================================
// Vectorized FP16 kernel (__half2 — 2 elements per 32-bit load)
// ===========================================================================
__global__ void rmsnorm_kernel_fp16_vec(
    __half*       __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    const __half* row_in  = input  + row * hidden_size;
    __half*       row_out = output + row * hidden_size;

    const int vec_hidden = hidden_size / 2;
    const __half2* vec_in = reinterpret_cast<const __half2*>(row_in);

    // Phase 1: vectorized sum-of-squares
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __half2 v = vec_in[i];
        float v0 = __half2float(v.x);
        float v1 = __half2float(v.y);
        sum_sq += v0 * v0 + v1 * v1;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();
    const float rms_inv = s_rms_inv;

    // Phase 2: vectorized normalize + scale
    const __half2* vec_w   = reinterpret_cast<const __half2*>(weight);
    __half2*       vec_out = reinterpret_cast<__half2*>(row_out);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __half2 v = vec_in[i];
        __half2 w = vec_w[i];

        float v0 = __half2float(v.x);
        float v1 = __half2float(v.y);
        float w0 = __half2float(w.x);
        float w1 = __half2float(w.y);

        __half2 result;
        result.x = __float2half(v0 * rms_inv * w0);
        result.y = __float2half(v1 * rms_inv * w1);
        vec_out[i] = result;
    }
}

// ===========================================================================
// Vectorized FP32 kernel (float4 — 4 elements per 128-bit load)
// ===========================================================================
__global__ void rmsnorm_kernel_fp32_vec(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    const float* row_in  = input  + row * hidden_size;
    float*       row_out = output + row * hidden_size;

    const int vec_hidden = hidden_size / 4;
    const float4* vec_in = reinterpret_cast<const float4*>(row_in);

    // Phase 1: vectorized sum-of-squares (4 floats at a time)
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        float4 v = vec_in[i];
        sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    // Handle remainder (hidden_size not divisible by 4)
    for (int i = vec_hidden * 4 + tid; i < hidden_size; i += stride) {
        float v = row_in[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();
    const float rms_inv = s_rms_inv;

    // Phase 2: vectorized normalize + scale
    const float4* vec_w   = reinterpret_cast<const float4*>(weight);
    float4*       vec_out = reinterpret_cast<float4*>(row_out);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        float4 v = vec_in[i];
        float4 w = vec_w[i];

        float4 result;
        result.x = v.x * rms_inv * w.x;
        result.y = v.y * rms_inv * w.y;
        result.z = v.z * rms_inv * w.z;
        result.w = v.w * rms_inv * w.w;
        vec_out[i] = result;
    }
    // Handle remainder
    for (int i = vec_hidden * 4 + tid; i < hidden_size; i += stride) {
        float v = row_in[i];
        float w = weight[i];
        row_out[i] = v * rms_inv * w;
    }
}

// ===========================================================================
// Launch helpers — compute thread count and shared memory, dispatch kernel
// ===========================================================================

static inline int compute_threads_vec2(int hidden_size) {
    // For bf16/fp16: 2 elements per vector load
    int threads = min(hidden_size / 2, MAX_THREADS);
    threads = max(threads, WARP_SIZE);
    // Round up to warp boundary
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return threads;
}

static inline int compute_threads_vec4(int hidden_size) {
    // For fp32: 4 elements per vector load
    int threads = min(hidden_size / 4, MAX_THREADS);
    threads = max(threads, WARP_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return threads;
}

static inline int compute_threads_scalar(int hidden_size) {
    int threads = min(hidden_size, MAX_THREADS);
    threads = max(threads, WARP_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return threads;
}

static inline size_t smem_size(int threads) {
    return ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
}

// ===========================================================================
// C++ entry points (called from torch_binding.cpp)
// ===========================================================================
extern "C" {

void rmsnorm_forward_bf16(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    if (hidden_size % 2 == 0 && hidden_size >= 64) {
        int threads = compute_threads_vec2(hidden_size);
        rmsnorm_kernel_bf16_vec<<<num_rows, threads, smem_size(threads), stream>>>(
            output, input, weight, hidden_size, eps
        );
    } else {
        int threads = compute_threads_scalar(hidden_size);
        rmsnorm_kernel_scalar<__nv_bfloat16><<<num_rows, threads, smem_size(threads), stream>>>(
            output, input, weight, hidden_size, eps
        );
    }
}

void rmsnorm_forward_fp16(
    __half*       output,
    const __half* input,
    const __half* weight,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    if (hidden_size % 2 == 0 && hidden_size >= 64) {
        int threads = compute_threads_vec2(hidden_size);
        rmsnorm_kernel_fp16_vec<<<num_rows, threads, smem_size(threads), stream>>>(
            output, input, weight, hidden_size, eps
        );
    } else {
        int threads = compute_threads_scalar(hidden_size);
        rmsnorm_kernel_scalar<__half><<<num_rows, threads, smem_size(threads), stream>>>(
            output, input, weight, hidden_size, eps
        );
    }
}

void rmsnorm_forward_fp32(
    float*       output,
    const float* input,
    const float* weight,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    if (hidden_size % 4 == 0 && hidden_size >= 64) {
        int threads = compute_threads_vec4(hidden_size);
        rmsnorm_kernel_fp32_vec<<<num_rows, threads, smem_size(threads), stream>>>(
            output, input, weight, hidden_size, eps
        );
    } else {
        int threads = compute_threads_scalar(hidden_size);
        rmsnorm_kernel_scalar<float><<<num_rows, threads, smem_size(threads), stream>>>(
            output, input, weight, hidden_size, eps
        );
    }
}

} // extern "C"
