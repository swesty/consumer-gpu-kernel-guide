// Prototype: L lanes per row (L = 32, 16 or 8), so 32/L rows per warp. bf16, h <= 1024, h % 8 == 0.
// Row held in registers; shuffle-only reduction within the L-lane group (xor offsets < L stay in-group).
// Out-of-range groups do no memory work but still execute the shuffles, so the full mask stays valid.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

template <int L, int MAXV>
__global__ void rmsnorm_subwarp_bf16(__nv_bfloat16* __restrict__ out,
                                     const __nv_bfloat16* __restrict__ in,
                                     const __nv_bfloat16* __restrict__ w,
                                     int64_t rows, int h, float eps) {
  const int sub = threadIdx.x % L;
  const int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / L;
  const bool valid = row < rows;
  const int nvec = h >> 3;
  const uint4* x4 = reinterpret_cast<const uint4*>(in + (valid ? row : 0) * h);
  uint4 v[MAXV];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < MAXV; ++k) {
    const int i = sub + k * L;
    if (valid && i < nvec) {
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
  for (int o = L / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (!valid) return;
  const float r = rsqrtf(ss / (float)h + eps);
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  uint4* o4 = reinterpret_cast<uint4*>(out + row * h);
#pragma unroll
  for (int k = 0; k < MAXV; ++k) {
    const int i = sub + k * L;
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

template <int L>
static void launch(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w, int64_t rows, int h,
                   float eps, int threads, cudaStream_t st) {
  const int64_t blocks = (rows * L + threads - 1) / threads;
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto* i = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* ww = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  const int per_lane = (h / 8 + L - 1) / L;  // uint4 vectors each lane holds
  if (per_lane <= 1)      rmsnorm_subwarp_bf16<L, 1><<<blocks, threads, 0, st>>>(o, i, ww, rows, h, eps);
  else if (per_lane <= 2) rmsnorm_subwarp_bf16<L, 2><<<blocks, threads, 0, st>>>(o, i, ww, rows, h, eps);
  else if (per_lane <= 4) rmsnorm_subwarp_bf16<L, 4><<<blocks, threads, 0, st>>>(o, i, ww, rows, h, eps);
  else                    rmsnorm_subwarp_bf16<L, 16><<<blocks, threads, 0, st>>>(o, i, ww, rows, h, eps);
}

void subwarp(torch::Tensor out, torch::Tensor x, torch::Tensor w, double eps, int64_t lanes, int64_t threads) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
  const int h = x.size(-1);
  const int64_t rows = x.numel() / h;
  TORCH_CHECK(h % 8 == 0 && h <= 1024 && out.numel() == x.numel() && w.numel() == h);
  TORCH_CHECK(((uintptr_t)x.data_ptr() | (uintptr_t)w.data_ptr() | (uintptr_t)out.data_ptr()) % 16 == 0);
  TORCH_CHECK(threads % 32 == 0);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  if (lanes == 32)      launch<32>(out, x, w, rows, h, (float)eps, threads, st);
  else if (lanes == 16) launch<16>(out, x, w, rows, h, (float)eps, threads, st);
  else if (lanes == 8)  launch<8>(out, x, w, rows, h, (float)eps, threads, st);
  else TORCH_CHECK(false, "lanes must be 32, 16 or 8");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
