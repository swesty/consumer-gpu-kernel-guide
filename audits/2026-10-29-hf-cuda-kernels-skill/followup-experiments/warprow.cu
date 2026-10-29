// Prototype: one warp per row RMSNorm for bf16, h <= 1024, h % 8 == 0, 16-byte aligned.
// Row is held in registers (<= 4 x uint4 per lane), so x is read from DRAM once.
// No shared memory, no __syncthreads: reduction is warp shuffles only.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

template <int MAXV>
__global__ void rmsnorm_warp_row_bf16(__nv_bfloat16* __restrict__ out,
                                      const __nv_bfloat16* __restrict__ in,
                                      const __nv_bfloat16* __restrict__ w,
                                      int64_t rows, int h, float eps) {
  const int lane = threadIdx.x & 31;
  const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int nvec = h >> 3;  // 8 bf16 per uint4
  const uint4* x4 = reinterpret_cast<const uint4*>(in + row * h);
  uint4 v[MAXV];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < MAXV; ++k) {
    const int i = lane + k * 32;
    if (i < nvec) {
      v[k] = x4[i];
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v[k]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(p[j]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float r = rsqrtf(ss / (float)h + eps);
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  uint4* o4 = reinterpret_cast<uint4*>(out + row * h);
#pragma unroll
  for (int k = 0; k < MAXV; ++k) {
    const int i = lane + k * 32;
    if (i < nvec) {
      uint4 wv = w4[i], res;
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&v[k]);
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
      __nv_bfloat162* rp = reinterpret_cast<__nv_bfloat162*>(&res);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 a = __bfloat1622float2(xp[j]), b = __bfloat1622float2(wp[j]);
        rp[j] = __floats2bfloat162_rn(a.x * r * b.x, a.y * r * b.y);
      }
      o4[i] = res;
    }
  }
}

void warp_row(torch::Tensor out, torch::Tensor x, torch::Tensor w, double eps, int64_t warps_per_block) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
  const int h = x.size(-1);
  const int64_t rows = x.numel() / h;
  TORCH_CHECK(h % 8 == 0 && h <= 1024 && out.numel() == x.numel() && w.numel() == h);
  TORCH_CHECK(((uintptr_t)x.data_ptr() | (uintptr_t)w.data_ptr() | (uintptr_t)out.data_ptr()) % 16 == 0);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int threads = 32 * warps_per_block;
  const int64_t blocks = (rows + warps_per_block - 1) / warps_per_block;
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto* i = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* ww = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  if (h <= 256)      rmsnorm_warp_row_bf16<1><<<blocks, threads, 0, st>>>(o, i, ww, rows, h, (float)eps);
  else if (h <= 512) rmsnorm_warp_row_bf16<2><<<blocks, threads, 0, st>>>(o, i, ww, rows, h, (float)eps);
  else               rmsnorm_warp_row_bf16<4><<<blocks, threads, 0, st>>>(o, i, ww, rows, h, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
