# GB10 GPU Optimization Guide for CUDA Kernels

Deep dive into GB10 (DGX Spark) specific optimizations for LLM inference CUDA kernels, with focus on Qwen3-8B.

## GB10 Blackwell Architecture Overview

### Key Specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| Compute Capability | 12.1 (sm_121a) | Arch-specific target; pair with `sm_120f` for fleet-portable builds |
| SMs | 48 | ~3.9x fewer than RTX 6000 Pro (188) |
| CUDA Cores | 6,144 | 128 per SM |
| Tensor Cores | 192 | 5th gen, FP4/FP8/FP16/BF16 |
| L2 Cache | 24 MB | Weight-friendly, not KV-cache-friendly |
| Shared Memory | 128 KB/SM | Configurable |
| Registers | 64K 32-bit/SM | 255 per thread max |
| Memory | 128 GB LPDDR5X | Unified with Grace CPU |
| Memory Bandwidth | 273 GB/s | **Shared** with CPU |
| NVLink-C2C | 600 GB/s bidirectional | CPU↔GPU interconnect |
| Max Threads/SM | 2048 | 64 warps |
| Max Threads/Block | 1024 | 32 warps |
| Warp Size | 32 | Unchanged |
| TDP | 140W (SoC) | Includes CPU + GPU |
| Process | TSMC 3nm | |

### GB10 vs Other Blackwell GPUs

| Spec | GB10 (DGX Spark) | RTX 6000 Pro | RTX 5090 |
|------|-------------------|--------------|----------|
| SMs | 48 | 188 | 170 |
| Memory BW | 273 GB/s | 1,790 GB/s | 1,792 GB/s |
| Memory Type | LPDDR5X (unified) | GDDR7 | GDDR7 |
| Memory Size | 128 GB | 96 GB | 32 GB |
| L2 Cache | 24 MB | 128 MB | 96 MB |
| TDP | 140W (SoC) | 600W | 575W |
| Compute Cap | sm_121a | sm_120 | sm_120 |

### What Makes GB10 Unique

1. **Unified memory** — No dedicated GPU VRAM. CPU and GPU share 128 GB LPDDR5X.
2. **Bandwidth-constrained** — 273 GB/s shared between CPU and GPU. Every byte of memory traffic counts.
3. **Large capacity** — 128 GB means even 70B parameter models can fit without quantization.
4. **Low power** — 140W SoC. Thermal throttling is less of a concern but sustained power is limited.
5. **High latency memory** — LPDDR5X has higher access latency than GDDR7/HBM3. Latency hiding is critical.

## Memory Hierarchy Optimization

### The Bandwidth Bottleneck

At 273 GB/s, the GB10 is severely bandwidth-constrained compared to discrete GPUs. This makes bandwidth-bound kernels (normalization, element-wise ops, attention) the primary optimization targets.

**Roofline model for GB10:**
```
Peak compute (FP32): ~31 TFLOPS
Peak bandwidth:      273 GB/s
Arithmetic intensity crossover: 31000 / 273 ≈ 113 FLOP/byte

Operations below 113 FLOP/byte are memory-bound on GB10.
RMSNorm: ~6 FLOP/byte → deeply memory-bound
```

For comparison, H100's crossover is ~200 FLOP/byte, meaning more operations are compute-bound there. On GB10, almost everything except dense matrix multiply is memory-bound.

### Vectorized Memory Access (Critical)

Vectorized loads are the single most important optimization on GB10. They reduce the number of memory transactions and improve bus utilization:

**BF16 vectorization (2 elements per 32-bit load):**
```cuda
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 8  // Higher unroll for LPDDR5X latency hiding
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
    // Process v0, v1...
}
```

**FP16 vectorization:**
```cuda
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_input[i];
float v0 = __half2float(v.x);
float v1 = __half2float(v.y);
```

**FP32 vectorization (4 elements per 128-bit load):**
```cuda
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 v = vec_input[i];
// v.x, v.y, v.z, v.w — 4 consecutive floats in one transaction
```

**Transaction sizes:**
- Minimum: 32 bytes
- Optimal: 128 bytes (full warp coalesced, FP32)
- Always align to 128-byte boundaries when possible

### L2 Cache Strategy

With only 24 MB of L2, careful cache management matters more than on larger GPUs:

**What fits in L2:**
| Data | Size (Qwen3-8B, BF16) | Fits in 24 MB L2? |
|------|------------------------|-------------------|
| RMSNorm weight (dim=4096) | 8 KB | Yes (trivially) |
| RMSNorm weight (dim=128) | 256 B | Yes (trivially) |
| One transformer layer's norms | ~16 KB | Yes |
| All 145 RMSNorm weights | ~1.2 MB | Yes |
| Single attention head KV (seq=2048) | 1 MB | Yes |
| Full KV cache (seq=2048) | ~2.4 GB | No (100x too large) |

**Implications:**
- RMSNorm weights will stay hot in L2 across all row iterations — good
- KV cache will constantly evict and reload from LPDDR5X — attention is the real bottleneck
- Sequential kernel launches that share data (e.g., RMSNorm → linear projection) can benefit from L2 residency of the normalized output, if it's small enough

**L2 residency hint (advanced):**
```cuda
// Pin hot data in L2 across kernel launches
cudaStreamAttrValue stream_attr = {};
stream_attr.accessPolicyWindow.base_ptr = weight_ptr;
stream_attr.accessPolicyWindow.num_bytes = weight_bytes;
stream_attr.accessPolicyWindow.hitRatio = 1.0f;
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

### Shared Memory Configuration

GB10 supports 128 KB of shared memory per SM, configurable between shared memory and L1 cache. For reduction kernels like RMSNorm, shared memory usage is minimal (only a few floats for warp reduction results), so the default split is fine.

For attention kernels with large tiles:
```cuda
// Request more shared memory if needed
cudaFuncSetAttribute(
    your_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    128 * 1024  // 128 KB max on GB10
);
```

### Bank Conflicts

Shared memory has 32 banks (4 bytes per bank). Same rules as all NVIDIA GPUs:
```cuda
// BAD: stride-32 access, all threads hit same bank
float val = shared[threadIdx.x * 32];

// GOOD: consecutive access, no conflicts
float val = shared[threadIdx.x];

// GOOD: Padding to avoid conflicts in 2D access
__shared__ float tile[32][33];  // 33 instead of 32
```

## Occupancy and Thread Configuration

### Block Size Selection for GB10

With only 48 SMs, the goal is to maximize occupancy per SM while keeping enough threads for efficient vectorized loops.

| Kernel Type | Threads/Block | Blocks/SM | Rationale |
|-------------|---------------|-----------|-----------|
| RMSNorm (hidden=4096) | **512** | 4 | 4 × 512 = 2048 threads/SM (100% occupancy) |
| RMSNorm (head_dim=128) | **64** | 32 | Small dim, 1 warp enough; many blocks for SM fill |
| Element-wise (RoPE, GELU) | **256** | 8 | Simple ops, high occupancy |
| Attention | **256** | 4-8 | Balance shared mem + registers |

**Why 512 instead of 1024 for RMSNorm on GB10:**

With hidden_size=4096 (BF16 vectorized → 2048 vec elements):
- **1024 threads**: 2048 / 1024 = 2 vec elements/thread, 2 blocks/SM max → if one block stalls on memory, the other can't fully hide the latency
- **512 threads**: 2048 / 512 = 4 vec elements/thread, 4 blocks/SM → more blocks to overlap memory latency, better for high-latency LPDDR5X

### Calculating Occupancy

```
Occupancy = Active Warps per SM / Max Warps per SM (64)

Limiting factors (GB10):
1. Registers: 65536 registers / (threads_per_block × regs_per_thread)
2. Shared Memory: 128 KB / smem_per_block
3. Threads: 2048 / threads_per_block
```

For our RMSNorm kernel at 512 threads:
- Registers: ~24 per thread × 512 = 12,288 → fits 5 blocks (but limited by threads)
- Shared memory: 64 bytes per block → negligible
- Threads: 2048 / 512 = 4 blocks/SM
- **Occupancy: 4 × 16 warps = 64 warps → 100%**

### Grid Size and SM Utilization

With 48 SMs, you want `num_blocks >= 48` to keep all SMs busy, and ideally `num_blocks` is a multiple of 48 for even distribution.

For RMSNorm, grid size = num_rows (one block per row):
```
Single token decode:  1 row   → 1/48 SMs active (2% utilization)
seq_len=128:          128 rows → 128/48 = 2.67 blocks/SM (good)
seq_len=512:          512 rows → 512/48 = 10.67 blocks/SM (saturated)
seq_len=2048:         2048 rows → fully saturated
```

Short sequences underutilize the GPU significantly. For single-token decode, consider batching multiple norm operations or using persistent kernels.

## Latency Hiding for LPDDR5X

### The LPDDR5X Latency Problem

LPDDR5X has higher access latency than GDDR7 or HBM3:

| Memory Type | Typical Latency | Bandwidth |
|-------------|----------------|-----------|
| HBM3 (H100) | ~100 ns | 3,350 GB/s |
| GDDR7 (RTX 6000 Pro) | ~120 ns | 1,790 GB/s |
| LPDDR5X (GB10) | ~150-200 ns | 273 GB/s |

Higher latency means more in-flight memory requests are needed to saturate bandwidth. The techniques:

### 1. Loop Unrolling

Deeper unrolling generates more independent memory requests:
```cuda
// GB10: unroll 8 (vs 4 on high-bandwidth GPUs)
#pragma unroll 8
for (int i = tid; i < vec_hidden; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    // ...
}
```

The compiler generates 8 load instructions before any dependent operations, allowing the memory controller to pipeline them.

### 2. Higher Occupancy

More resident blocks per SM = more warps available to execute while others wait on memory:
```
512 threads/block → 4 blocks/SM → 64 warps → always a warp ready to execute
1024 threads/block → 2 blocks/SM → 32 warps → may stall if both wait on memory
```

### 3. Instruction-Level Parallelism

Structure compute to be independent of memory loads:
```cuda
// GOOD: independent accumulation
float sum0 = 0.0f, sum1 = 0.0f;  // Two independent accumulators
for (int i = tid; i < vec_hidden; i += stride * 2) {
    __nv_bfloat162 v0 = vec_input[i];
    __nv_bfloat162 v1 = vec_input[i + stride];
    // v0 and v1 loads are independent → can pipeline
    float a = __bfloat162float(v0.x);
    float b = __bfloat162float(v1.x);
    sum0 += a * a;
    sum1 += b * b;
}
float sum_sq = sum0 + sum1;
```

## Precision and Type Handling

### BF16 vs FP16 on GB10

GB10 (Blackwell) fully supports both BF16 and FP16. BF16 is preferred for LLM inference:

```
FP16:  1 sign + 5 exponent + 10 mantissa → better precision, smaller range
BF16:  1 sign + 8 exponent + 7 mantissa  → same range as FP32, less precision
```

For Qwen3-8B at BF16:
- Model size: ~16 GB
- Fits easily in 128 GB unified memory
- No quantization needed (unlike 32 GB discrete GPUs where you might need INT8/INT4)

### PyTorch Type Conversion Requirement

PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__`, disabling implicit FP16/BF16 conversions. Always include explicit helpers:

```cuda
__device__ __forceinline__ float to_float(float x)            { return x; }
__device__ __forceinline__ float to_float(__half x)           { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x)    { return __bfloat162float(x); }

__device__ __forceinline__ float          from_float(float x, float*)            { return x; }
__device__ __forceinline__ __half         from_float(float x, __half*)           { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16  from_float(float x, __nv_bfloat16*)   { return __float2bfloat16(x); }
```

### Mixed Precision Accumulation

Always accumulate reductions in FP32:
```cuda
float sum_sq = 0.0f;  // FP32 accumulator
for (...) {
    float val = __bfloat162float(input[i]);  // Convert to FP32
    sum_sq += val * val;                      // Accumulate in FP32
}
// Reduction in FP32
sum_sq = block_reduce_sum(sum_sq, shared);
// Apply and convert back to BF16
output[i] = __float2bfloat16(result);
```

## Warp-Level Operations

### Shuffle Reductions

Fastest intra-warp communication — identical across all NVIDIA GPUs:
```cuda
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### Block Reduction Pattern

Warp shuffle → shared memory → warp shuffle:
```cuda
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}
```

Shared memory needed: `ceil(threads / 32) × 4` bytes. For 512 threads: 64 bytes.

## Qwen3-8B on GB10: What to Optimize

### Model Profile

| Component | % of Inference Time (approx) | Memory-bound? | Custom Kernel Value |
|-----------|------------------------------|---------------|---------------------|
| Linear projections (QKV, O, FFN) | ~60% | No (compute-bound) | Low (use cuBLAS) |
| Attention (softmax + score @ V) | ~20% | Mixed | Medium (use Flash Attention) |
| RMSNorm (all 145 modules) | ~8% | **Yes** | **High** |
| RoPE | ~3% | Yes | Medium |
| Activation (SiLU) | ~2% | Yes | Low (already fast) |
| Sampling + overhead | ~7% | CPU | N/A |

On GB10, the memory-bound kernels (RMSNorm, RoPE, activation) are proportionally more expensive because of the lower bandwidth. Custom kernels have higher relative value here.

### RMSNorm Module Breakdown (Qwen3-8B)

| Module | Count | Dimension | Threads | Vec Elements/Thread |
|--------|-------|-----------|---------|---------------------|
| input_layernorm | 36 | 4096 | 512 | 4 |
| post_attention_layernorm | 36 | 4096 | 512 | 4 |
| model.norm | 1 | 4096 | 512 | 4 |
| self_attn.q_norm | 36 | 128 | 64 | 1 |
| self_attn.k_norm | 36 | 128 | 64 | 1 |

### Bandwidth Efficiency Targets

For a well-optimized RMSNorm kernel on GB10:
```
Theoretical minimum time = data_size / peak_bandwidth

For [1, 2048, 4096] BF16:
  Read input:  2048 × 4096 × 2 bytes = 16 MB
  Read weight: 4096 × 2 bytes        = 8 KB
  Write output: 16 MB
  Total: ~32 MB

  Theoretical minimum: 32 MB / 273 GB/s = 0.117 ms

Achieving 40-60% of peak bandwidth is realistic:
  Expected: 0.20 - 0.29 ms
```

## Compilation

### NVCC Flags for GB10

```bash
nvcc \
    -O3 \
    -gencode=arch=compute_121a,code=sm_121a \  # GB10-native, CC 12.1 only
    -gencode=arch=compute_120f,code=sm_120f \  # Family SASS, any CC 12.x
    --use_fast_math \
    -lineinfo \
    --threads=4 \
    your_kernel.cu
```

**Key flags:**
| Flag | Purpose |
|------|---------|
| `-gencode=arch=compute_121a,code=sm_121a` | Native SASS, GB10-only arch-specific features |
| `-gencode=arch=compute_120f,code=sm_120f` | Family SASS, loads on any CC 12.x — no JIT |
| `--use_fast_math` | Fast `rsqrtf`, `__expf`, etc. |
| `-lineinfo` | Debug info for ncu/nsys without performance loss |
| `--threads=4` | Parallel ptxas compilation |
| `-maxrregcount=N` | Limit registers (rarely needed) |

**Requires CUDA Toolkit 12.9+** — compute capability 12.1 targets (`sm_121`, `sm_121a`) and the
family-specific `sm_120f` target were all added in CUDA 12.9. Check with:
```bash
nvcc --list-gpu-arch | grep 121
```

**Why `sm_120f` and not a PTX fallback.** `sm_121a` is architecture-specific: it loads on GB10 and
nowhere else, failing on CC 12.0 parts with `no kernel image is available for execution on the
device`. The second gencode line covers everything else. Shipping `sm_120f` SASS beats shipping
`compute_120` PTX because the family target loads directly on any CC 12.x device with no JIT step,
so there is no first-launch compile pause and no dependency on the end user's driver being able to
compile the PTX. Drop the `sm_121a` line entirely if you do not need GB10-exclusive features —
`sm_120f` alone runs on all three Blackwell GPUs in this repo.

See [Blackwell target suffixes](../../guides/tuning-guide.md) in the tuning guide for the measured
compatibility matrix.

### setup.py Configuration

```python
extra_compile_args={
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-gencode=arch=compute_121a,code=sm_121a",
        "-gencode=arch=compute_120f,code=sm_120f",
        "--use_fast_math",
        "-lineinfo",
        "--threads=4",
    ],
},
```

## Profiling on GB10

### NVIDIA Nsight Systems (nsys)

System-wide profiling to see CPU/GPU interaction:
```bash
nsys profile -o gb10_profile python your_script.py

# Key things to look for on GB10:
# - CPU memory traffic during GPU kernels (contention)
# - GPU idle time between kernels (launch overhead)
# - Page migration events (unified memory)
```

### NVIDIA Nsight Compute (ncu)

Kernel-level analysis:
```bash
# Full metrics
ncu --set full -o metrics.ncu-rep python your_script.py

# GB10-specific metrics to watch:
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed \
    python your_script.py
```

**Critical metrics for bandwidth-bound kernels on GB10:**
| Metric | Target | What it means |
|--------|--------|---------------|
| `dram__throughput` | >40% of peak | Are you saturating LPDDR5X? |
| `lts__throughput` (L2) | High | L2 cache doing its job |
| `sm__throughput` | Low (for mem-bound) | Expected — compute isn't the bottleneck |
| `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | >60% | Occupancy / latency hiding |

### Common Performance Issues on GB10

1. **Low DRAM throughput (<20% of peak)**
   - Cause: Non-coalesced access or low occupancy
   - Fix: Vectorize loads, increase block count

2. **CPU contention reducing GPU bandwidth**
   - Cause: Python/framework CPU work during kernel execution
   - Fix: Use `torch.inference_mode()`, CUDA graphs, minimize allocations

3. **L2 thrashing**
   - Cause: Working set exceeds 24 MB L2
   - Fix: Tile data to fit in L2, use streaming access for non-reusable data

4. **Low occupancy on short sequences**
   - Cause: num_rows < 48, most SMs idle
   - Fix: Batch multiple operations, use persistent kernels

5. **High kernel launch overhead**
   - Cause: Many small kernels with Python dispatch in between
   - Fix: CUDA graphs, kernel fusion, `torch.compile`

## Best Practices Summary for GB10

1. **Vectorize everything** — 273 GB/s is precious. Use `__nv_bfloat162`, `__half2`, `float4`.
2. **Unroll aggressively** — `#pragma unroll 8` to hide LPDDR5X latency.
3. **Prefer 512 threads/block** for reduction kernels — allows 4 blocks/SM for latency hiding.
4. **Keep weights in L2** — small weight tensors (< 24 MB total) benefit from persistent caching.
5. **Accumulate in FP32** — always reduce in float, convert back at the end.
6. **Minimize CPU work during inference** — shared bandwidth means CPU steals from GPU.
7. **Warmup before benchmarking** — page migration needs a few iterations to settle.
8. **Profile with nsys first** — identify CPU contention before optimizing kernel code.
9. **Use explicit type conversions** — `to_float()` / `from_float()` helpers for PyTorch compat.
10. **BF16 is preferred** — 128 GB unified memory means no need to quantize Qwen3-8B.

## torch.compile Compatibility

### The Problem

Custom CUDA kernels that access tensor data pointers are not compatible with `torch.compile`'s graph tracing by default. The compiler traces using "fake tensors" that don't have real data:

```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
```

### Solution: Register as a PyTorch Custom Op

Use `torch.library.custom_op` to make your kernel visible to the compiler:

```python
import torch
from qwen3_kernels._C import rmsnorm_forward as _rmsnorm_forward

@torch.library.custom_op("qwen3_kernels::rmsnorm", mutates_args={"out"})
def rmsnorm_op(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    _rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)

@rmsnorm_op.register_fake
def _(out, input, weight, eps):
    pass  # No shape/dtype changes — output is written in-place to 'out'
```

Then use the custom op in your patching code:

```python
def make_forward(mod, epsilon):
    def forward(hidden_states):
        out = torch.empty_like(hidden_states)
        torch.ops.qwen3_kernels.rmsnorm(out, hidden_states, mod.weight, epsilon)
        return out
    return forward
```

### Alternative: TORCH_LIBRARY_EXPAND (C++ side)

For Hub-publishable kernels, use the newer binding pattern in `torch_binding.cpp`:

```cpp
#include <torch/library.h>
#include "registration.h"  // From kernel-builder

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("rmsnorm_forward(Tensor! out, Tensor input, Tensor weight, float eps) -> ()");
    ops.impl("rmsnorm_forward", torch::kCUDA, &rmsnorm_forward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

This pattern replaces `PYBIND11_MODULE` and is required for both torch.compile and HuggingFace Kernels Hub publishing.

### Performance Comparison

| Configuration | Speedup | Notes |
|:---|:---:|:---|
| Custom kernels only | ~1.06-1.08x | Works immediately, no compilation overhead |
| torch.compile only | ~1.30-1.40x | Requires warmup compilation pass |
| Both (custom op registered) | Best of both | Requires `torch.library` registration |

On GB10, the torch.compile benefit is proportionally larger because it eliminates CPU-side framework overhead that competes for shared LPDDR5X bandwidth.

## CUDA Graphs

CUDA graphs capture a sequence of kernel launches and replay them with near-zero CPU overhead. Particularly valuable on GB10 where CPU dispatch traffic competes for LPDDR5X bandwidth.

### Basic Pattern

```python
# Capture phase (run once)
static_input = torch.randn(1, 2048, 4096, device="cuda", dtype=torch.bfloat16)
static_output = torch.empty_like(static_input)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay phase (run many times — near-zero CPU overhead)
static_input.copy_(real_input)  # Update input in-place
g.replay()
# static_output now contains the result
```

### With torch.compile (recommended)

```python
# torch.compile with reduce-overhead mode uses CUDA graphs internally
model = torch.compile(model, mode="reduce-overhead")
```

### GB10-Specific Benefit

On discrete GPUs, kernel launch overhead is ~5-10 μs per kernel, mostly hidden by GPU execution. On GB10, each launch also generates CPU memory traffic that steals from the GPU's 273 GB/s. CUDA graphs eliminate this contention entirely during replay.

## HuggingFace Kernels Hub

### Using Pre-Compiled Kernels (No Local Build)

Instead of compiling kernels locally, you can load pre-built optimized kernels from the HuggingFace Hub:

```bash
pip install kernels torch
```

```python
from kernels import get_kernel, has_kernel

# Check availability for your environment
if has_kernel("kernels-community/triton-layer-norm"):
    layer_norm = get_kernel("kernels-community/triton-layer-norm")

    # Use directly
    x = torch.randn(2, 1024, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(4096, dtype=torch.bfloat16, device="cuda")
    out = layer_norm.rms_norm(x, weight, eps=1e-6)
```

### Qwen3's Built-In Hub Kernel Support

Qwen3RMSNorm in transformers already has a `@use_kernel_forward_from_hub("RMSNorm")` decorator. If a compatible Hub kernel is available and the `kernels` library is installed, transformers will **automatically** use the optimized kernel — no manual patching needed:

```python
# This may already use Hub kernels automatically
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
# Check if Hub kernels are active:
# pip install kernels && python -c "from kernels import has_kernel; print(has_kernel('kernels-community/triton-layer-norm'))"
```

### Publishing Your Own Kernel to Hub

To make your GB10-optimized kernel available to others:

1. Use the `TORCH_LIBRARY_EXPAND` binding pattern (see torch.compile section above)
2. Set `cuda-capabilities = ["12.1"]` in `build.toml`
3. Build and upload:

```bash
pip install kernel-builder
kernel-builder build
huggingface-cli upload your-username/qwen3-rmsnorm-gb10 ./dist
```

### Available Community Kernels

| Kernel | Repo ID | Functions |
|--------|---------|-----------|
| Layer/RMS Norm | `kernels-community/triton-layer-norm` | `rms_norm`, `layer_norm` |
| Activations | `kernels-community/activation` | `gelu_fast`, `silu` |
| Flash Attention | `kernels-community/flash-attn` | Flash Attention 2 |

### When HF Kernels Hub Does NOT Apply

The HF Kernels Hub (`get_kernel()`, `kernelize()`, `@use_kernel_forward_from_hub`) only works with **vanilla HuggingFace transformers** running in standard PyTorch. It does **not** apply to production inference servers:

| Framework | Uses HF `forward()` | Hub Kernels Work? | Custom Kernel Method |
|-----------|---------------------|-------------------|----------------------|
| **HF Transformers** (vanilla) | Yes | Yes | `kernelize()`, monkey-patch |
| **vLLM** | No (own model implementations) | No | CustomOp plugin system (`forward_oot()`) |
| **TensorRT-LLM** | No (compiled TRT engine) | No | C++ TRT plugins |

**vLLM** has its own complete model implementations in `vllm/model_executor/models/` with custom CUDA kernels (e.g., `csrc/layernorm_kernels.cu` for RMSNorm). It only imports HuggingFace's config class — never the model forward methods. vLLM also provides fused kernels (AllReduce + RMSNorm + Quantize via FlashInfer) that wouldn't be possible with individual Hub kernel replacements.

**TensorRT-LLM** compiles models into optimized TRT engines. At inference time, there are no Python `forward()` methods — the engine executes a pre-compiled CUDA execution plan. Custom kernels must be written as C++ TRT plugins.

**Monkey-patching `module.forward`** is also unreliable with these frameworks:
- **vLLM**: Runs the model in a separate process (V1 architecture). Patches in the main process have no effect. Additionally, `torch.compile` and CUDA graph capture bypass Python-level patches.
- **TensorRT-LLM**: No Python forward at runtime. Pre-compilation patches would be overwritten by TRT's optimizer.

**When to use what:**
- **Prototyping / research / small-scale inference** → HF Transformers + Hub kernels or manual injection
- **Production serving** → vLLM (CustomOp plugins) or TRT-LLM (C++ plugins) — these have their own optimized kernel stacks and the overhead of using vanilla transformers would negate any Hub kernel benefit

## Troubleshooting

### Build Errors

**1. Type conversion errors with FP16/BF16**
```
error: no suitable conversion function from "__half" to "float" exists
```
Cause: PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__`.
Fix: Add `to_float()` / `from_float()` helpers (see Precision section above).

**2. Missing CUDA headers in torch_binding.cpp**
```
error: undeclared identifier '__half'
```
Fix: Include all required headers:
```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>      // Required for __half
#include <cuda_bf16.h>      // Required for __nv_bfloat16
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
```

**3. sm_121a not recognized**
```
nvcc fatal: Unsupported gpu architecture 'compute_121a'
```
Fix: Requires CUDA Toolkit 12.9+. Check version:
```bash
nvcc --version
```

### Integration Errors

**4. `isinstance()` misses model's RMSNorm**
```python
# WRONG — misses Qwen3RMSNorm, LlamaRMSNorm, etc.
if isinstance(module, torch.nn.RMSNorm):

# CORRECT — catches all variants
if 'RMSNorm' in type(module).__name__:
```

**5. Wrong epsilon attribute name**
```python
# LLaMA / Qwen: 'variance_epsilon'
# Others: 'eps'
# Safe pattern:
eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)
```

**6. Kernel patching doesn't survive `enable_model_cpu_offload()`**

Fix: Inject kernels AFTER `.to("cuda")`, BEFORE `enable_model_cpu_offload()`.

### Verification

```python
# Check CUDA architecture is detected correctly
python -c "import torch; print(torch.cuda.get_device_capability())"
# GB10 should print (12, 1)

# Verify kernel import works
python -c "from qwen3_kernels import rmsnorm; print('OK')"

# Test a forward pass through a patched module
for name, module in model.named_modules():
    if 'RMSNorm' in type(module).__name__:
        x = torch.randn(1, 10, 4096, device='cuda', dtype=torch.bfloat16)
        out = module(x)
        print(f"{name}: {x.shape} -> {out.shape}")
        break
```

## Multi-Node Operation (Spark Stacking)

### Interconnect Architecture

Two DGX Sparks connect via **ConnectX-7 SmartNIC over Ethernet/RoCE** — there is no NVLink bridge between nodes. NVLink-C2C (600 GB/s) is strictly an intra-node interconnect (CPU↔GPU within one SoC).

```
┌──────────────────────┐   ConnectX-7 RoCE   ┌──────────────────────┐
│   DGX Spark Node 0   │◄── 200 Gbps ──────►│   DGX Spark Node 1   │
│                       │    (~25 GB/s)       │                       │
│  48 SMs  │  128 GB    │                     │  48 SMs  │  128 GB    │
│  273 GB/s (local)     │                     │  273 GB/s (local)     │
│  NVLink-C2C (internal)│                     │  NVLink-C2C (internal)│
└──────────────────────┘                      └──────────────────────┘
```

Each DGX Spark has **2x QSFP ports** on the ConnectX-7. With one cable: 200 Gbps (~25 GB/s). With two cables: 400 Gbps (~50 GB/s). NVIDIA sells an official **DGX Spark Bundle** with cable included.

### Combined Specifications

| Spec | Single Node | Two Nodes (Stacked) | Notes |
|------|-------------|---------------------|-------|
| SMs | 48 | 96 | Split across nodes |
| CUDA Cores | 6,144 | 12,288 | |
| Memory | 128 GB | 256 GB | NOT shared across nodes |
| Memory BW (per node) | 273 GB/s | 273 GB/s each | Does NOT double for a single kernel |
| Inter-node BW | N/A | ~25 GB/s (1 cable) | ~10x slower than local LPDDR5X |
| L2 Cache | 24 MB | 24 MB each | NOT shared |
| TDP | 140W | 280W | |
| Max model (BF16) | ~60B params | ~120B params | |
| Max model (FP4) | ~200B params | ~405B params | |

**Critical:** Memory bandwidth does NOT double. Each kernel runs on one GPU and sees 273 GB/s. The inter-node link at ~25 GB/s is the synchronization bottleneck.

### NCCL Configuration

NCCL v2.28+ is required. Key environment variables:

```bash
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export TP_SOCKET_IFNAME=enp1s0f1np1
```

NCCL must be compiled for Blackwell aarch64:
```bash
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121a,code=sm_121a"
```

Measured NCCL bandwidth: **~22-23 GB/s** (all_gather_perf), consistent with 200 Gbps line rate minus protocol overhead.

### Parallelism Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **Tensor Parallelism (TP=2)** | Shard weight matrices across both GPUs. AllReduce/AllGather at each layer. | Model too large for one node (>60B BF16) |
| **Pipeline Parallelism (PP=2)** | Split layers across nodes (e.g., layers 0-17 on node 0, 18-35 on node 1). Point-to-point at pipeline boundaries. | Large models, lower communication overhead than TP |
| **Data Parallelism** | Run independent model copies, one per node. No inter-node communication during inference. | Throughput scaling for models that fit on one node |

**For inference serving:**
```bash
# vLLM with tensor parallelism
vllm serve "model-name" --tensor-parallel-size 2

# TensorRT-LLM
trtllm-build --tp_size 2
```

### Kernel Implications for Multi-Node

**Individual kernel code does NOT change.** A CUDA kernel runs identically whether on a single node or as part of a TP=2 setup. What changes is the surrounding framework and the effective problem decomposition:

1. **Communication-compute overlap**: While one kernel executes, NCCL can be transferring data for the next operation. This is a framework-level optimization (TensorRT-LLM, Megatron-LM, DeepSpeed all implement this). Kernel authors don't need to handle communication directly.

2. **Reduced per-node problem size**: With TP=2, each node processes half the hidden dimension. Kernels that dynamically size their thread blocks (e.g., based on input dimension) automatically adapt:
   ```
   Single node: [batch, seq, hidden_dim]     → full dimension
   TP=2:        [batch, seq, hidden_dim / 2]  → half dimension per node
   ```
   Reduction kernels may end up with fewer threads and different occupancy characteristics. Element-wise kernels simply process half the elements.

3. **AllReduce is the bottleneck**: After TP-sharded operations (linear projections, attention), results must be synchronized across nodes. The cost scales with tensor size:
   ```
   AllReduce payload = batch × seq_len × hidden_dim × bytes_per_element

   Example: [1, 2048, 4096] BF16 = 16 MB
   At 25 GB/s inter-node: ~0.64 ms per AllReduce

   For comparison, a well-optimized bandwidth-bound kernel on the same
   data takes ~0.1-0.3 ms locally. Communication dominates.
   ```

4. **Not all operations are sharded**: Per-head operations (QK norm in attention, head-local projections) and layer norms applied before sharding run on the full local dimension with no cross-node communication.

### When to Use Each Parallelism Strategy

| Model Size (BF16) | Fits Single Node? | Recommended Strategy |
|:---|:---:|:---|
| < 60B params (~120 GB) | Yes | Data parallelism — run independent instances for throughput |
| 60-120B params | Needs 2 nodes | PP=2 preferred (lower communication) or TP=2 |
| > 120B params | Needs 2+ nodes | TP=2 + PP if needed, or FP8/FP4 quantization to fit |

**General guidance:**
- If a model fits on one node, **don't shard it**. Run independent instances for throughput scaling. TP=2 adds ~25 GB/s AllReduce overhead at every layer, which usually costs more than it saves.
- **Pipeline parallelism (PP)** has less communication than TP — only point-to-point transfers at pipeline stage boundaries, not AllReduce at every layer. Prefer PP when latency tolerance allows micro-batching.
- **Tensor parallelism (TP)** gives lower per-request latency than PP (no pipeline bubbles) but requires AllReduce/AllGather at every sharded operation. Best for latency-sensitive serving of models that don't fit on one node.

### Kernel Design Considerations for Multi-Node

For models that *require* TP=2, kernel optimization takes on a different character:

1. **Faster local kernels expose the communication bottleneck.** Paradoxically, making a kernel 2x faster doesn't make the end-to-end step 2x faster — it makes the AllReduce a larger fraction of total time. Amdahl's law applies: if communication is 60% of step time, halving compute time only gives ~1.25x speedup.

2. **Kernel fusion is more valuable in multi-node.** Fusing multiple operations between communication points reduces the number of kernel launches, intermediate memory traffic, and total compute-window time. This maximizes the compute:communication ratio:
   ```
   Unfused: LayerNorm → AllReduce → Linear → GELU → Linear → AllReduce
   Fused:   LayerNorm+Linear+GELU+Linear → AllReduce
   (Fewer kernels, less memory traffic, more work per communication step)
   ```

3. **Communication-compute overlap requires kernel awareness.** Advanced frameworks (TensorRT-LLM, Megatron-LM) split operations to overlap NCCL transfers with independent local compute. Custom kernels that support partial execution (processing a slice of the input while another slice is in transit) enable finer-grained overlap.

4. **Persistent kernels** that stay resident and process a work queue eliminate per-kernel launch overhead. In multi-node settings, launch overhead compounds — each unnecessary microsecond is multiplied by the number of kernels per step (often hundreds). CUDA graphs address this at the framework level; persistent kernels address it at the kernel level.

5. **Minimize synchronization points.** Each `__syncthreads()` or global memory fence adds latency. In multi-node settings where the GPU is frequently waiting for NCCL anyway, reducing intra-kernel synchronization allows warps to make progress on other work while waiting.

### Profiling Multi-Node Workloads

```bash
# Profile both nodes simultaneously with nsys
# Node 0:
nsys profile -o node0_profile --trace=cuda,nvtx,osrt python your_script.py

# Node 1:
nsys profile -o node1_profile --trace=cuda,nvtx,osrt python your_script.py

# Key things to look for:
# - NCCL collective duration vs kernel duration (communication ratio)
# - GPU idle time between NCCL completion and next kernel launch
# - NCCL-kernel overlap (are they running concurrently?)
# - Imbalanced computation between nodes (one node idle while the other computes)
```

**NCCL performance validation:**
```bash
# Test raw inter-node bandwidth with nccl-tests
all_reduce_perf -b 1M -e 256M -f 2 -g 1
all_gather_perf -b 1M -e 256M -f 2 -g 1

# Expected: ~22-23 GB/s bus bandwidth (200 Gbps line rate minus overhead)
# If significantly lower, check cable seating, network config, or NIC firmware
```

### Memory Visibility Across Nodes

Each node's 128 GB is **local only**. There is no unified address space across nodes. Cross-node data movement happens exclusively through NCCL collectives over the ConnectX-7 link.

However, the unified memory architecture provides an indirect benefit: the ConnectX-7 NIC writes incoming RDMA data directly into the same LPDDR5X that the GPU accesses. There's no staging through a separate "host memory" buffer — the GPU can immediately read NCCL-received data without an additional copy. This is effectively **GPUDirect-like behavior without nvidia-peermem** (which is not supported on DGX Spark).

Key implications for kernel/buffer design:
- **`cudaMalloc` vs `cudaHostAlloc`** is less meaningful on GB10 — both allocate from the same LPDDR5X. NCCL buffers are GPU-visible regardless.
- **No explicit staging copies** — on discrete GPUs, NCCL must stage through pinned host memory or use GPUDirect RDMA. On GB10, the NIC and GPU share memory natively.
- **Communication adds to shared bandwidth pressure** — the NIC's ~25 GB/s traffic competes with GPU kernel memory access and CPU framework overhead on the same 273 GB/s LPDDR5X bus.

## Improving Kernels in vLLM and TensorRT-LLM

Both vLLM and TensorRT-LLM have significant kernel optimization opportunities, particularly on Blackwell GPUs. Neither framework has fully exploited Blackwell-specific hardware features, and many high-value kernel fusions remain unimplemented.

### Where the Gaps Are

**Blackwell hardware features underutilized by current kernels:**

| Feature | What It Does | Current Status |
|---------|-------------|----------------|
| **TCGen05 (5th-gen Tensor Cores)** | 256×256×16 MMA spanning 2 SMs via CTA pairs | Most kernels use Hopper-era 128×128 tiles |
| **Tensor Memory (TMEM)** | 128×512 × 32-bit on-chip buffer dedicated to tensor core data | Kernels not using TMEM leave bandwidth on the table |
| **TMA with multicast** | Load from global memory to shared memory of multiple SMs in one op | Kernels using per-thread address calculation miss this |
| **CTA pair cooperation** | Two CTAs in a cluster execute a single tensor core instruction | Fundamentally new execution model, not in Hopper kernels |
| **Native FP4/FP6/FP8** | Block-scaled variants (NVFP4, MXFP4, MXFP6) in hardware | NVFP4 supported but with large performance gaps |

Kernels tuned for SM90 (Hopper) that use `wgmma` with 128×128 tiles, standard shared memory, and per-warp TMA are not exploiting Blackwell's architectural advances.

**SM120-specific gaps (GB10, RTX 5090, RTX 6000 Pro):**
- SM120 shares FP4/FP8 tensor core capabilities with SM100 but was not recognized in some backend selection logic (e.g., NVFP4 MoE kernel selection in vLLM originally only checked SM90 and SM100)
- FP8 CUTLASS group GEMM had to fall back to Triton on SM120
- SM120 kernel parity with SM100 is still catching up

### vLLM: Known Optimization Opportunities

vLLM tracks missing kernel fusions in [issue #25179](https://github.com/vllm-project/vllm/issues/25179) (30 subtasks, 0 completed as of early 2026). High-impact opportunities:

**Communication-compute overlap (highest value for multi-node):**
- `all_gather + gemm` — overlap collective with matrix multiply
- `gemm + reduce_scatter` — overlap matrix multiply with collective
- These eliminate the sequential communication→compute pattern that makes the inter-node link the bottleneck

**Fused normalization + quantization:**
- `rms_norm + nvfp4` — missing on Blackwell
- `all_reduce + rms_norm + dynamic fp8 quantization` — combines collective, normalization, and quantization in one kernel launch
- `rope + kvcache` and `rope + fp8_quant + kvcache` — available in FlashInfer but not yet integrated

**MoE (Mixture of Experts):**
- 145 TFLOPS gap between vLLM and SGLang on FP4 MoE on B200, attributed to kernel fusion differences (7 memory passes vs 5), missing Blackwell-specific CUTLASS schedules, and adaptive grid sizing
- MoE padding/quantization + finalize/slice fusion — expected ~6% end-to-end gain

**Decode-phase optimization:**
- MLA decode + quantization fusion (FP8 and NVFP4 variants)
- Speculative decoding `prepare_inputs_padded` — could be a Triton kernel instead of Python

### TensorRT-LLM: Known Optimization Opportunities

**Auto-fusion limitations:**
- Complex fusions (FlashAttention, MLA variants, sparse attention) cannot be auto-discovered by TRT's graph optimizer — they require explicit plugins
- Any novel attention pattern requires manual C++ plugin development

**Quantization maturity:**
- NVFP4 and FP8 kernel implementations are explicitly noted as "evolving"
- NVFP4 KV cache quantization (50% reduction vs FP8) is a recent addition with room for tuning
- Pipeline parallelism with FP8/NVFP4 weights has known issues on some model families

### General Principles for Kernel Optimization

#### 1. Profile Before Optimizing

```bash
# vLLM: system-wide profiling
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    python -m vllm.entrypoints.openai.api_server --model your-model

# Per-kernel analysis
ncu --clock-control None --kernel-id :::1 --set full \
    python your_benchmark.py

# Key metrics:
# - "GPU Speed of Light" — are you hitting peak throughput?
# - "Occupancy" — are SMs fully utilized?
# - "Memory Workload Analysis" — where are the memory bottlenecks?
```

Look for: GPU idle gaps between kernels (fusion opportunities), low DRAM throughput (coalescing issues), high kernel launch overhead (CUDA graph candidates).

#### 2. Fuse Across Communication Boundaries

The highest-value optimization on multi-node systems is overlapping communication with compute. Instead of:
```
Kernel A → wait → AllReduce → wait → Kernel B
```
Fuse to:
```
Kernel A + AllReduce (overlapped) → Kernel B
```

On GB10 with 25 GB/s inter-node bandwidth, communication takes 10× longer per byte than local compute. Every millisecond of overlap saves real time.

#### 3. Fuse Memory-Bound Operations

Memory-bound kernels (normalization, activation, quantization) are fusion targets because they're limited by memory bandwidth, not compute. Fusing them eliminates intermediate memory round-trips:

```
Unfused (3 kernel launches, 3 memory reads + writes):
  RMSNorm → SiLU → FP8 Quantize

Fused (1 kernel launch, 1 read + 1 write):
  RMSNorm_SiLU_FP8Quant
```

On GB10 at 273 GB/s, eliminating two intermediate read-write pairs for a [1, 2048, 4096] BF16 tensor saves ~0.23 ms per fused operation. Across 36 layers, that's ~8 ms per forward pass.

#### 4. Exploit Blackwell-Specific Hardware

For GEMM-class kernels on Blackwell:
- Use **CTA-pair 256×256 tiles** instead of Hopper's 128×128 — doubles the work per tensor core instruction
- Use **TMEM** for tensor core input staging instead of shared memory — dedicated on-chip buffer with higher throughput
- Use **TMA multicast** to load data into multiple SMs' shared memory simultaneously — reduces redundant global memory reads
- Target **CUTLASS 3.8+** which has Blackwell-specific schedules

For bandwidth-bound kernels (normalization, activation, RoPE):
- Blackwell's hardware changes don't fundamentally alter these kernels — they're still limited by DRAM bandwidth
- The wins come from **fusion** (reducing the number of separate kernel launches) and **quantization integration** (fusing quantize/dequantize with compute)

#### 5. Design for CUDA Graph Compatibility

Both vLLM and TRT-LLM use CUDA graphs aggressively. Your kernel must work within this constraint:

- **No dynamic memory allocation** inside the kernel (no `cudaMalloc` during execution)
- **No host-device synchronization** (no `cudaMemcpy` or `cudaDeviceSynchronize` in the hot path)
- **Fixed grid/block dimensions** for a given input shape (or use a small set of pre-captured shapes)
- **No conditional kernel launches** based on runtime values — the graph is a fixed execution plan

vLLM uses three CUDA graph modes:
- `PIECEWISE` — attention stays eager, other ops in graph
- `FULL` — everything in graph (only FlashAttention 3 supports this)
- `FULL_AND_PIECEWISE` — default, best for low latency

#### 6. Design for Tensor Parallelism Compatibility

Kernels in TP-sharded models see reduced dimensions:
- Linear projections: weight matrices are split across devices
- Normalization: may run on full or partial dimension depending on placement
- Communication collectives (AllReduce, AllGather) are inserted between sharded operations

Your kernel should:
- Accept arbitrary dimension sizes (not hardcode to a specific hidden_dim)
- Handle dimensions that aren't power-of-2 (TP sharding may produce odd sizes)
- Not assume it's the only kernel running — other streams may be executing NCCL collectives concurrently

#### 7. Design for Quantization Orthogonality

Modern inference servers mix precision formats throughout the model. A well-designed kernel should work with multiple quantization formats:
- BF16/FP16 inputs (standard)
- FP8 (E4M3, E5M2) inputs and outputs
- NVFP4 block-scaled inputs
- Mixed: BF16 compute with FP8 KV cache

Fusing quantization into compute kernels (e.g., `rms_norm + fp8_quant`) eliminates a separate quantization kernel launch and its associated memory traffic.

### Contributing Kernels

**To vLLM:**

1. **CustomOp / PluggableLayer**: Register with `@CustomOp.register("name")` (or the newer PluggableLayer + vLLM IR). Implement `forward_cuda()` for CUDA, `forward_native()` for PyTorch fallback.

2. **Helion / Triton**: vLLM is moving toward [Helion](https://github.com/vllm-project/vllm/issues/32962), a higher-level DSL that compiles to Triton. 24 of 29 tracked kernels are not yet started — contributions welcome.

3. **Testing**: Use pytest with CUDA dependencies. Run warmup before collecting performance numbers. Micro-benchmark framework runs tuning outside the runtime.

4. **Build**: Use the [incremental compilation workflow](https://docs.vllm.ai/en/v0.10.1/contributing/incremental_build.html) for faster iteration.

**To TensorRT-LLM:**

1. **C++ Plugins**: Implement `IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`. The `enqueue()` method launches your CUDA kernel.

2. **Autotuning**: Extend `gemmPluginProfiler` to profile different tactics (tile sizes, thread configs) for given problem dimensions. The profiler selects the fastest tactic per shape.

3. **PyTorch-first backend** (TRT-LLM 1.0+): Custom kernels can serve as both TRT plugins and PyTorch custom ops, simplifying development.

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│          GB10 DGX Spark Quick Ref       │
├─────────────────────────────────────────┤
│ Arch:       sm_121a (Blackwell)         │
│ SMs:        48                          │
│ CUDA Cores: 6,144                       │
│ Memory:     128 GB LPDDR5X (unified)    │
│ Bandwidth:  273 GB/s (shared w/ CPU)    │
│ L2 Cache:   24 MB                       │
│ Shared/SM:  128 KB                      │
│ Regs/SM:    65,536                      │
│ Threads/SM: 2,048                       │
│ TDP:        140W (SoC)                  │
│ CUDA:       >= 12.8 required            │
├─────────────────────────────────────────┤
│ Compile:  -gencode compute_121a,sm_121a │
│ Threads:  512 (reduction), 256 (elem)   │
│ Unroll:   #pragma unroll 8              │
│ Vectorize: bf162 / half2 / float4       │
│ Peak BW:  273 GB/s (realistic: ~110)    │
└─────────────────────────────────────────┘
```
