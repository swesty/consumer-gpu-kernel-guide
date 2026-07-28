# RTX 6000 Pro Blackwell Optimization Guide for CUDA Kernels

Deep dive into RTX 6000 Pro Blackwell (GB202) specific optimizations for LLM inference CUDA kernels, with focus on Qwen3-8B.

## RTX 6000 Pro Blackwell Architecture Overview

### Key Specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| Compute Capability | 12.1 (sm_121) | Target in build.toml / setup.py |
| GPU Die | GB202 | TSMC 5nm, 760 mm², 92.2B transistors |
| SMs | 188 | Most of any current Blackwell GPU |
| CUDA Cores | 24,064 | 128 per SM |
| Tensor Cores | 752 | 5th gen, FP4/FP8/FP16/BF16 |
| RT Cores | 188 | 4th gen |
| L2 Cache | 128 MB | Enormous — larger than most GPUs' VRAM |
| Shared Memory | 128 KB/SM | Configurable up to 228 KB |
| Registers | 64K 32-bit/SM | 255 per thread max |
| Memory | 96 GB GDDR7 | 512-bit interface, ECC |
| Memory Bandwidth | 1.79 TB/s | Dedicated, no contention |
| Max Threads/SM | 2048 | 64 warps |
| Max Threads/Block | 1024 | 32 warps |
| Warp Size | 32 | Unchanged |
| TDP | 600W | Active cooling required |
| PCIe | Gen 5 x16 | ~64 GB/s bidirectional |
| Base/Boost Clock | 1590 / 2617 MHz | |

### Blackwell Family Comparison

| Spec | RTX 6000 Pro | RTX 5090 | RTX 5080 | GB10 (DGX Spark) | B200 (Data Center) |
|------|-------------|----------|----------|-------------------|---------------------|
| SMs | **188** | 170 | 84 | 48 | 160 |
| Memory BW | **1.79 TB/s** | 1.79 TB/s | 960 GB/s | 273 GB/s | 8.0 TB/s |
| Memory Type | GDDR7 | GDDR7 | GDDR7 | LPDDR5X | HBM3e |
| Memory Size | **96 GB** | 32 GB | 16 GB | 128 GB | 192 GB |
| L2 Cache | **128 MB** | 96 MB | 48 MB | 24 MB | 96 MB |
| Compute Cap | sm_121 | sm_120 | sm_120 | sm_121a | sm_100 |
| TDP | 600W | 575W | 360W | 140W | 1000W |

### Blackwell Architecture Highlights

1. **5th Gen Tensor Cores** — FP4 support, higher throughput FP8/BF16
2. **Neural Shader Execution** — hardware-accelerated inference paths
3. **GDDR7 memory** — higher per-pin bandwidth than GDDR6X, 512-bit bus
4. **Massive L2** — 128 MB enables caching of entire working sets
5. **PCIe Gen 5** — 2x host bandwidth vs Gen 4

## Memory Hierarchy Optimization

### The RTX 6000 Pro Memory Stack

```
Register File:  64K × 32-bit per SM  │  ~0 latency   │  Per-thread private
Shared Memory:  128 KB per SM         │  ~20 cycles   │  Per-block shared
L1 Cache:       Configurable w/ smem  │  ~30 cycles   │  Per-SM
L2 Cache:       128 MB unified        │  ~200 cycles  │  Global shared
GDDR7:          96 GB                 │  ~400 cycles  │  Global
PCIe (host):    System RAM            │  ~10K cycles  │  Cross-device
```

### Bandwidth at Each Level

| Level | Bandwidth | Latency | Size |
|-------|-----------|---------|------|
| Registers | ~100+ TB/s (aggregate) | 0 cycles | 64K/SM |
| Shared Memory | ~40 TB/s (aggregate) | ~20 cycles | 128 KB/SM |
| L2 Cache | ~8-12 TB/s | ~200 cycles | 128 MB |
| GDDR7 | **1.79 TB/s** | ~400 cycles | 96 GB |
| PCIe 5 | ~64 GB/s | ~10K cycles | System RAM |

The 1.79 TB/s GDDR7 bandwidth is the key number for memory-bound kernels.

### Vectorized Memory Access

Vectorized loads remain important even at 1.79 TB/s — they reduce instruction count and improve coalescing:

**BF16 vectorization (2 elements per 32-bit load):**
```cuda
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 4
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
// v.x, v.y, v.z, v.w — 4 consecutive floats
```

**Coalescing requirements:**
- 32 bytes minimum transaction
- 128 bytes optimal (full warp, FP32)
- Align to 128-byte boundaries

### L2 Cache Exploitation (128 MB)

The 128 MB L2 is a major advantage over smaller GPUs. It can hold:

| Working Set | Size | Fits in 128 MB L2? |
|-------------|------|---------------------|
| All Qwen3-8B RMSNorm weights | ~1.2 MB | Yes |
| Full transformer layer weights | ~450 MB | No |
| RMSNorm input [4, 2048, 4096] BF16 | 64 MB | Yes |
| RMSNorm input [4, 8192, 4096] BF16 | 256 MB | No |
| KV cache (seq=512, all layers) | ~600 MB | No |
| Attention scores (one layer, seq=2048) | ~32 MB | Yes |

**Optimization strategies:**

1. **Weight caching**: Small tensors like RMSNorm weights (~8 KB each) live permanently in L2.

2. **Inter-kernel reuse**: When RMSNorm output feeds a linear projection, the normalized activations may still be in L2:
```
RMSNorm kernel writes [batch, seq, 4096] → sits in L2
Linear projection reads [batch, seq, 4096] → L2 hit (if it fits)
```

3. **L2 persistence hints** (for critical data):
```cuda
cudaStreamAttrValue stream_attr = {};
stream_attr.accessPolicyWindow.base_ptr = ptr;
stream_attr.accessPolicyWindow.num_bytes = bytes;
stream_attr.accessPolicyWindow.hitRatio = 1.0f;
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

4. **Streaming access for large data**: Mark data that won't be reused as streaming to avoid polluting L2:
```cuda
// For one-shot reads that won't be reused
// Use __ldcs (load cached streaming) intrinsic
float val = __ldcs(&input[idx]);
```

### Shared Memory and Bank Conflicts

128 KB shared memory per SM, 32 banks (4 bytes per bank):

```cuda
// BAD: stride-32 access — all threads hit same bank
float val = shared[threadIdx.x * 32];

// GOOD: consecutive access
float val = shared[threadIdx.x];

// GOOD: padding for 2D arrays
__shared__ float tile[32][33];  // 33 avoids bank conflicts
```

For RMSNorm, shared memory usage is minimal (warp reduction scratch: 64-128 bytes), so bank conflicts are not a concern.

## Occupancy and Thread Configuration

### Block Size Selection for RTX 6000 Pro

With 188 SMs and high bandwidth, maximize per-block throughput:

| Kernel Type | Threads/Block | Blocks/SM | Occupancy | Rationale |
|-------------|---------------|-----------|-----------|-----------|
| RMSNorm (hidden=4096) | **1024** | 2 | 100% | Max threads, 2 blocks sufficient at low latency |
| RMSNorm (head_dim=128) | **64** | 32 | 100% | Small dim, many blocks fill SM |
| Element-wise (RoPE, GELU) | **256** | 8 | 100% | Simple ops, high occupancy |
| Attention | **256** | 4-8 | 50-100% | Balance shared mem + registers |

**Why 1024 works on RTX 6000 Pro (but not GB10):**
- GDDR7 latency (~100-120 ns) is lower than LPDDR5X (~150-200 ns)
- 2 blocks/SM with 1024 threads each = 2048 threads (100% occupancy)
- Fewer blocks means less scheduling overhead
- With high bandwidth, each block completes faster, freeing the SM sooner

### Grid Sizing for 188 SMs

For maximum GPU utilization, `num_blocks` should be >= 188, ideally a multiple:

```
RMSNorm grid = num_rows (one block per row):
  seq_len=1:     1 row    → 1/188 SMs (0.5% util) — poor
  seq_len=128:   128 rows → 68% SM utilization
  seq_len=188:   188 rows → 100% SM utilization (ideal)
  seq_len=512:   512 rows → 2.7 blocks/SM (saturated)
  seq_len=2048:  2048 rows → 10.9 blocks/SM (fully saturated)
```

Short sequences (< 188 rows) underutilize the GPU. For single-token decode, the RTX 6000 Pro's 188 SMs are mostly idle on a per-kernel basis — overall throughput comes from pipelining across the model's many layers.

### Occupancy Calculator

```
Occupancy = Active Warps per SM / Max Warps per SM (64)

Limiting factors:
1. Registers: 65536 / (threads_per_block × regs_per_thread)
2. Shared Memory: 128 KB / smem_per_block
3. Threads: 2048 / threads_per_block
4. Blocks: max 32 blocks/SM

For our RMSNorm at 1024 threads:
  Registers: ~20/thread × 1024 = 20480 → fits ~3 blocks (but threads limit to 2)
  Shared: 128 bytes → negligible
  Threads: 2048 / 1024 = 2 blocks/SM → 32 warps/SM → 100% of max with 2 blocks
```

Check register usage:
```bash
nvcc --ptxas-options=-v kernel_src/rmsnorm.cu
# Shows: "Used X registers, Y bytes smem, Z bytes cmem"
```

## Roofline Analysis

### RTX 6000 Pro Roofline

```
Peak FP32: ~100 TFLOPS (estimated)
Peak BW:   1.79 TB/s
Crossover: 100000 / 1790 ≈ 56 FLOP/byte

Operations below 56 FLOP/byte are memory-bound.
```

| Kernel | FLOP/byte | Bound | Notes |
|--------|-----------|-------|-------|
| RMSNorm | ~6 | **Memory** | sum_sq + rsqrt + scale |
| GELU/SiLU | ~2 | **Memory** | Element-wise activation |
| RoPE | ~8 | **Memory** | sin/cos + rotate |
| Attention (no flash) | ~10-20 | **Memory** | Softmax + score @ V |
| Linear (GEMM) | ~100-1000 | **Compute** | Depends on dimensions |
| Flash Attention | ~50-200 | **Mixed** | Tiled, memory-efficient |

Most operations except GEMM and Flash Attention are memory-bound. Custom kernels for RMSNorm, RoPE, and activation functions can help by:
- Minimizing memory traffic (vectorized loads)
- Reducing kernel launch overhead (fusion)
- Eliminating intermediate materializations

### Bandwidth Efficiency Targets

For a well-optimized RMSNorm kernel on RTX 6000 Pro:

```
For [1, 2048, 4096] BF16:
  Read input:  2048 × 4096 × 2 bytes = 16 MB
  Read weight: 4096 × 2 bytes        = 8 KB (cached in L2)
  Write output: 16 MB
  Total: ~32 MB

  Theoretical minimum: 32 MB / 1790 GB/s = 0.018 ms
  Realistic (35-45% efficiency): 0.040 - 0.051 ms
```

For larger batches:
```
For [4, 2048, 4096] BF16:
  Total: ~128 MB
  Theoretical minimum: 128 MB / 1790 GB/s = 0.072 ms
  Realistic: 0.16 - 0.21 ms
```

## Precision and Type Handling

### BF16 vs FP16 on Blackwell

Both are first-class citizens on Blackwell. BF16 is preferred for LLM inference:

```
FP16:  1 sign + 5 exponent + 10 mantissa
  - Better precision (10 bits mantissa)
  - Smaller range (±65504)
  - Risk of overflow in attention scores

BF16:  1 sign + 8 exponent + 7 mantissa
  - Same range as FP32 (±3.4e38)
  - Lower precision (7 bits mantissa)
  - No overflow risk in attention
  - Preferred for training and inference
```

### FP8 and FP4 on Blackwell

Blackwell's 5th-gen Tensor Cores support FP8 (E4M3, E5M2) and FP4 natively. While not used in our RMSNorm kernel (normalization needs FP32 accumulation), they're relevant for:
- Quantized linear projections (FP8 GEMM)
- Quantized KV cache (FP8 storage)
- Future quantized attention kernels

### PyTorch Type Conversion Requirement

Always include explicit conversion helpers:
```cuda
__device__ __forceinline__ float to_float(float x)            { return x; }
__device__ __forceinline__ float to_float(__half x)           { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x)    { return __bfloat162float(x); }

__device__ __forceinline__ float          from_float(float x, float*)            { return x; }
__device__ __forceinline__ __half         from_float(float x, __half*)           { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16  from_float(float x, __nv_bfloat16*)   { return __float2bfloat16(x); }
```

### Mixed Precision Accumulation

Always reduce in FP32:
```cuda
float sum_sq = 0.0f;  // FP32 accumulator
for (...) {
    float val = __bfloat162float(input[i]);
    sum_sq += val * val;
}
sum_sq = block_reduce_sum(sum_sq, shared);
output[i] = __float2bfloat16(normalized * weight_val);
```

## Warp-Level Operations

### Shuffle Reductions

Identical across all NVIDIA GPUs:
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

Shared memory for 1024 threads: `ceil(1024/32) × 4 = 128 bytes`.

## Qwen3-8B on RTX 6000 Pro: Optimization Priorities

### Model Fit

Qwen3-8B in BF16 (~16 GB) leaves 80 GB of GDDR7 free for:
- KV caches (long context inference)
- Batch processing (high-throughput serving)
- Gradient storage (fine-tuning)

### Where Custom Kernels Help Most

| Component | % of Inference Time (approx) | Bound | Custom Kernel Value |
|-----------|------------------------------|-------|---------------------|
| Linear projections | ~65% | Compute | Low (cuBLAS is near-optimal) |
| Attention | ~15% | Mixed | Medium (Flash Attention 2) |
| **RMSNorm** | **~5%** | **Memory** | **High** (vectorized kernel) |
| RoPE | ~3% | Memory | Medium |
| Activation (SiLU) | ~2% | Memory | Low |
| Framework overhead | ~10% | CPU | Use torch.compile / CUDA graphs |

On RTX 6000 Pro, RMSNorm is a smaller fraction of total time than on GB10 (because linear projections run 6.5x faster with 6.5x more bandwidth, shifting the bottleneck toward compute). Custom kernels have relatively less end-to-end impact, but the kernel itself still runs proportionally faster.

### RMSNorm Module Breakdown (Qwen3-8B)

| Module | Count | Dimension | Threads | Vec Elements/Thread |
|--------|-------|-----------|---------|---------------------|
| input_layernorm | 36 | 4096 | 1024 | 2 |
| post_attention_layernorm | 36 | 4096 | 1024 | 2 |
| model.norm | 1 | 4096 | 1024 | 2 |
| self_attn.q_norm | 36 | 128 | 64 | 1 |
| self_attn.k_norm | 36 | 128 | 64 | 1 |

### Combining with torch.compile

The RTX 6000 Pro is an excellent target for `torch.compile`:
```python
# Option 1: torch.compile only (no custom kernels)
model = torch.compile(model, mode="reduce-overhead")
# Expected: ~30-40% end-to-end speedup

# Option 2: Custom kernels only
inject_optimized_kernels(model)
# Expected: ~5-8% end-to-end speedup (RMSNorm is small fraction)

# Option 3: Both (requires custom op registration)
# See torch.library.custom_op pattern in SKILL.md
```

Note: Custom kernels and `torch.compile` are mutually exclusive without custom op registration via `torch.library`.

## Compilation

### NVCC Flags for RTX 6000 Pro

```bash
nvcc \
    -O3 \
    -arch=sm_121 \
    -gencode=arch=compute_121,code=sm_121 \
    --use_fast_math \
    -lineinfo \
    --threads=4 \
    your_kernel.cu
```

| Flag | Purpose |
|------|---------|
| `-arch=sm_121` | Native SASS for Blackwell sm_121 |
| `-gencode=arch=compute_121,code=sm_121` | Explicit gencode for sm_121 |
| `--use_fast_math` | Fast `rsqrtf`, `__expf`, etc. (~10% faster math) |
| `-lineinfo` | Debug info for ncu/nsys (no perf impact) |
| `--threads=4` | Parallel ptxas compilation |
| `-maxrregcount=N` | Limit registers if needed for occupancy |

**Requires CUDA Toolkit 12.8+** for sm_121 support.

### setup.py Configuration

```python
extra_compile_args={
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-arch=sm_121",
        "-gencode=arch=compute_121,code=sm_121",
        "--use_fast_math",
        "-lineinfo",
        "--threads=4",
    ],
},
```

## Profiling on RTX 6000 Pro

### NVIDIA Nsight Systems (nsys)

```bash
nsys profile -o rtx6000_profile python your_script.py

# What to look for:
# - Kernel duration (should be short with 1.79 TB/s)
# - PCIe transfer time (model loading)
# - GPU idle gaps between kernels (framework overhead)
# - torch.compile warmup cost
```

### NVIDIA Nsight Compute (ncu)

```bash
# Full analysis
ncu --set full -o metrics.ncu-rep python your_script.py

# Key metrics for RTX 6000 Pro:
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    python your_script.py
```

**Target metrics for bandwidth-bound kernels:**
| Metric | Target | Notes |
|--------|--------|-------|
| `dram__throughput` | >35% of peak | GDDR7 utilization |
| `lts__throughput` (L2) | Varies | High if data fits in L2 |
| `sm__throughput` | Low (10-20%) | Expected for memory-bound |
| `sm__warps_active` | >50% | Occupancy indicator |
| `l1tex__throughput` | Moderate | Shared mem + L1 traffic |

### Common Performance Issues

1. **Low DRAM throughput**
   - Non-coalesced access → vectorize loads
   - Low occupancy → check register/shared memory pressure

2. **High kernel launch overhead**
   - Many small kernels → fuse, use CUDA graphs, torch.compile
   - Python dispatch → use `torch.inference_mode()`

3. **PCIe bottleneck on model load**
   - One-time cost, not a kernel issue
   - Use `device_map="cuda"` for efficient loading

4. **Register spilling**
   - Too many local variables → reduce register usage or accept lower occupancy
   - Check with `nvcc --ptxas-options=-v`

5. **L2 eviction on large batches**
   - Batch × seq × hidden > 128 MB → input data doesn't fit in L2
   - Streaming access for non-reusable data (`__ldcs`)

## Best Practices Summary for RTX 6000 Pro

1. **Vectorize loads** — `__nv_bfloat162`, `__half2`, `float4` for coalesced 32/128-bit transactions.
2. **Use 1024 threads/block** for reduction kernels — 2 blocks/SM, full occupancy, low latency.
3. **Standard unroll** (`#pragma unroll 4`) — GDDR7 latency is manageable without aggressive unrolling.
4. **Exploit 128 MB L2** — weight tensors and moderate-sized activations stay cached.
5. **Accumulate in FP32** — always reduce in float, convert back at the end.
6. **Profile with ncu** — target >35% DRAM throughput for bandwidth-bound kernels.
7. **Consider torch.compile** — 30-40% end-to-end speedup, complementary to custom kernels.
8. **BF16 is the default** — 96 GB GDDR7 means no quantization needed for 8B models.
9. **Use explicit type conversions** — `to_float()` / `from_float()` helpers for PyTorch compat.
10. **Grid size >= 188** — keep all SMs busy; short sequences underutilize the GPU.

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
| Custom kernels only | ~1.05-1.08x | Works immediately, no compilation overhead |
| torch.compile only | ~1.30-1.40x | Requires warmup compilation pass |
| Both (custom op registered) | Best of both | Requires `torch.library` registration |

On RTX 6000 Pro, torch.compile's benefit comes primarily from fusing Python dispatch overhead and eliminating kernel launch gaps — the 1.79 TB/s bandwidth means individual kernels are already fast, so reducing the gaps between them matters.

## CUDA Graphs

CUDA graphs capture a sequence of kernel launches and replay them with near-zero CPU overhead. On RTX 6000 Pro, individual kernels complete very quickly (high bandwidth), making the relative cost of per-kernel CPU dispatch proportionally higher.

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

### RTX 6000 Pro-Specific Benefit

With 1.79 TB/s bandwidth, a small RMSNorm kernel ([1, 1, 4096] BF16) executes in ~10-20 μs. CPU launch overhead of ~5-10 μs per kernel is a significant fraction. CUDA graphs eliminate this overhead entirely, which matters most for:
- Single-token decode (small kernels, many launches per step)
- Short sequences where kernel time is minimal
- Serving workloads requiring consistent low latency

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

To make your RTX 6000 Pro-optimized kernel available to others:

1. Use the `TORCH_LIBRARY_EXPAND` binding pattern (see torch.compile section above)
2. Set `cuda-capabilities = ["12.1"]` in `build.toml`
3. Build and upload:

```bash
pip install kernel-builder
kernel-builder build
huggingface-cli upload your-username/qwen3-rmsnorm-rtx6000pro ./dist
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

**3. sm_121 not recognized**
```
nvcc fatal: Unsupported gpu architecture 'compute_121'
```
Fix: Requires CUDA Toolkit 12.8+. Check version:
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
# RTX 6000 Pro should print (10, 0)

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

## Improving Kernels in vLLM and TensorRT-LLM

Both vLLM and TensorRT-LLM have significant kernel optimization opportunities on Blackwell GPUs. Neither framework has fully exploited Blackwell-specific hardware features, and many high-value kernel fusions remain unimplemented.

### Where the Gaps Are

**Blackwell hardware features underutilized by current kernels:**

| Feature | What It Does | Current Status |
|---------|-------------|----------------|
| **TCGen05 (5th-gen Tensor Cores)** | 256×256×16 MMA spanning 2 SMs via CTA pairs | Most kernels use Hopper-era 128×128 tiles |
| **Tensor Memory (TMEM)** | 128×512 × 32-bit on-chip buffer dedicated to tensor core data | Kernels not using TMEM leave bandwidth on the table |
| **TMA with multicast** | Load from global memory to shared memory of multiple SMs in one op | Kernels using per-thread address calculation miss this |
| **CTA pair cooperation** | Two CTAs in a cluster execute a single tensor core instruction | Fundamentally new execution model, not in Hopper kernels |
| **Native FP4/FP6/FP8** | Block-scaled variants (NVFP4, MXFP4, MXFP6) in hardware | NVFP4 supported but with large performance gaps |

Kernels tuned for SM90 (Hopper) that use `wgmma` with 128×128 tiles, standard shared memory, and per-warp TMA are not exploiting Blackwell's architectural advances. The RTX 6000 Pro with 188 SMs has even more CTA pairs available than data center Blackwell GPUs.

### vLLM: Known Optimization Opportunities

vLLM tracks missing kernel fusions in [issue #25179](https://github.com/vllm-project/vllm/issues/25179) (30 subtasks, 0 completed as of early 2026). High-impact opportunities:

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

**Helion kernels:** 24 of 29 tracked kernels ([issue #32962](https://github.com/vllm-project/vllm/issues/32962)) are not yet started — including `rms_norm`, `silu_and_mul`, and all collective fusions.

### TensorRT-LLM: Known Optimization Opportunities

- Complex fusions (FlashAttention, MLA variants, sparse attention) cannot be auto-discovered by TRT's graph optimizer — they require explicit C++ plugins
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

# Key sections to check:
# - "GPU Speed of Light" — are you hitting peak throughput?
# - "Occupancy" — are SMs fully utilized?
# - "Memory Workload Analysis" — where are the memory bottlenecks?
```

On RTX 6000 Pro, look for: GPU idle gaps between kernels (fusion opportunities), low DRAM throughput below 35% of 1.79 TB/s (coalescing issues), high kernel launch overhead relative to fast kernel execution (CUDA graph candidates).

#### 2. Fuse Memory-Bound Operations

Memory-bound kernels (normalization, activation, quantization) are the prime fusion targets. Fusing eliminates intermediate memory round-trips:

```
Unfused (3 kernel launches, 3 memory reads + writes):
  RMSNorm → SiLU → FP8 Quantize

Fused (1 kernel launch, 1 read + 1 write):
  RMSNorm_SiLU_FP8Quant
```

On RTX 6000 Pro at 1.79 TB/s, individual kernels complete very quickly — but the launch overhead between them becomes proportionally significant. Fusing 3 kernels into 1 eliminates ~10-20 μs of launch overhead plus two intermediate memory round-trips.

#### 3. Exploit Blackwell-Specific Hardware

For GEMM-class kernels on Blackwell:
- Use **CTA-pair 256×256 tiles** instead of Hopper's 128×128 — doubles the work per tensor core instruction. The RTX 6000 Pro's 188 SMs provide 94 CTA pairs.
- Use **TMEM** for tensor core input staging instead of shared memory — dedicated on-chip buffer with higher throughput
- Use **TMA multicast** to load data into multiple SMs' shared memory simultaneously — reduces redundant global memory reads. Particularly effective with 128 MB L2 cache serving as the multicast source.
- Target **CUTLASS 3.8+** which has Blackwell-specific schedules

For bandwidth-bound kernels (normalization, activation, RoPE):
- Blackwell's tensor core advances don't fundamentally alter these kernels — they're still limited by DRAM bandwidth
- The wins come from **fusion** (reducing separate kernel launches) and **quantization integration** (fusing quantize/dequantize with compute)
- The RTX 6000 Pro's 128 MB L2 makes inter-kernel data reuse more likely — fused kernels that keep intermediate results in registers are still better, but the penalty for unfused sequential kernels is lower than on GPUs with smaller L2

#### 4. Design for CUDA Graph Compatibility

Both vLLM and TRT-LLM use CUDA graphs aggressively. Your kernel must work within this constraint:

- **No dynamic memory allocation** inside the kernel (no `cudaMalloc` during execution)
- **No host-device synchronization** (no `cudaMemcpy` or `cudaDeviceSynchronize` in the hot path)
- **Fixed grid/block dimensions** for a given input shape (or use a small set of pre-captured shapes)
- **No conditional kernel launches** based on runtime values — the graph is a fixed execution plan

vLLM uses three CUDA graph modes:
- `PIECEWISE` — attention stays eager, other ops in graph
- `FULL` — everything in graph (only FlashAttention 3 supports this)
- `FULL_AND_PIECEWISE` — default, best for low latency

#### 5. Design for Tensor Parallelism Compatibility

Kernels in TP-sharded models see reduced dimensions:
- Linear projections: weight matrices are split across devices
- Normalization: may run on full or partial dimension depending on placement
- Communication collectives (AllReduce, AllGather) are inserted between sharded operations

Your kernel should:
- Accept arbitrary dimension sizes (not hardcode to a specific hidden_dim)
- Handle dimensions that aren't power-of-2 (TP sharding may produce odd sizes)
- Not assume it's the only kernel running — other streams may be executing concurrently

#### 6. Design for Quantization Orthogonality

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

3. **Testing**: Use pytest with CUDA dependencies. Run warmup before collecting performance numbers. vLLM's micro-benchmark framework runs tuning outside the runtime.

4. **Build**: Use the [incremental compilation workflow](https://docs.vllm.ai/en/v0.10.1/contributing/incremental_build.html) for faster iteration.

**To TensorRT-LLM:**

1. **C++ Plugins**: Implement `IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`. The `enqueue()` method launches your CUDA kernel.

2. **Autotuning**: Extend `gemmPluginProfiler` to profile different tactics (tile sizes, thread configs) for given problem dimensions. The profiler selects the fastest tactic per shape.

3. **PyTorch-first backend** (TRT-LLM 1.0+): Custom kernels can serve as both TRT plugins and PyTorch custom ops, simplifying development.

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│     RTX 6000 Pro Blackwell Quick Ref    │
├─────────────────────────────────────────┤
│ Arch:       sm_121 (Blackwell GB202)    │
│ SMs:        188                         │
│ CUDA Cores: 24,064                      │
│ Memory:     96 GB GDDR7 (dedicated)     │
│ Bandwidth:  1.79 TB/s (exclusive)       │
│ L2 Cache:   128 MB                      │
│ Shared/SM:  128 KB (up to 228 KB)       │
│ Regs/SM:    65,536                      │
│ Threads/SM: 2,048                       │
│ TDP:        600W                        │
│ CUDA:       >= 12.8 required            │
├─────────────────────────────────────────┤
│ Compile:  -arch=sm_121                  │
│ Threads:  1024 (reduction), 256 (elem)  │
│ Unroll:   #pragma unroll 4              │
│ Vectorize: bf162 / half2 / float4       │
│ Peak BW:  1790 GB/s (realistic: ~650)   │
└─────────────────────────────────────────┘
```
