# RTX 3090 Ampere Optimization Guide for CUDA Kernels

Deep dive into RTX 3090 Ampere (GA102) specific optimizations for LLM inference CUDA kernels, with focus on Qwen3-8B.

## RTX 3090 Ampere Architecture Overview

### Key Specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| Compute Capability | 8.6 (sm_86) | Target in build.toml / setup.py |
| GPU Die | GA102 | Samsung 8nm, 628 mm², 28.3B transistors |
| SMs | 82 | Full GA102 die |
| CUDA Cores | 10,496 | 128 per SM |
| Tensor Cores | 328 | 3rd gen, FP16/BF16/TF32/INT8/INT4 |
| RT Cores | 82 | 2nd gen |
| L2 Cache | 6 MB | Much smaller than Blackwell-era GPUs |
| Shared Memory | 100 KB/SM | Configurable (128 KB combined L1+shared) |
| Registers | 64K 32-bit/SM | 255 per thread max |
| Memory | 24 GB GDDR6X | 384-bit interface |
| Memory Bandwidth | 936 GB/s | Dedicated, no contention |
| Max Threads/SM | 1,536 | 48 warps (not 2048 like Blackwell) |
| Max Threads/Block | 1,024 | 32 warps |
| Warp Size | 32 | Unchanged |
| TDP | 350W | Active cooling required |
| PCIe | Gen 4 x16 | ~32 GB/s bidirectional |
| NVLink | 3rd gen (2-way) | 112.5 GB/s bidirectional |
| Base/Boost Clock | 1395 / 1695 MHz | |

### Ampere Family Comparison

| Spec | RTX 3090 | RTX 3090 Ti | RTX 3080 | A100 (Data Center) |
|------|----------|-------------|----------|---------------------|
| SMs | **82** | 84 | 68 | 108 |
| Memory BW | **936 GB/s** | 1,008 GB/s | 760 GB/s | 2,039 GB/s (80GB) |
| Memory Type | GDDR6X | GDDR6X | GDDR6X | HBM2e |
| Memory Size | **24 GB** | 24 GB | 10/12 GB | 40/80 GB |
| L2 Cache | **6 MB** | 6 MB | 5 MB | 40 MB |
| Compute Cap | sm_86 | sm_86 | sm_86 | sm_80 |
| TDP | 350W | 450W | 320W | 300/400W |
| NVLink | 2-way | No | No | Yes (600 GB/s) |

### Cross-Generation Comparison (Ampere vs Blackwell)

| Spec | RTX 3090 (Ampere) | RTX 5090 (Blackwell) | RTX 6000 Pro (Blackwell) |
|------|-------------------|----------------------|--------------------------|
| SMs | 82 | 170 | 188 |
| Memory BW | 936 GB/s | 1,792 GB/s | 1,790 GB/s |
| Memory Size | 24 GB GDDR6X | 32 GB GDDR7 | 96 GB GDDR7 |
| L2 Cache | **6 MB** | 96 MB | 128 MB |
| Max Threads/SM | 1,536 | 2,048 | 2,048 |
| Compute Cap | sm_86 | sm_120 | sm_120 |
| TDP | 350W | 575W | 600W |
| FP8/FP4 | No | Yes (5th gen TC) | Yes (5th gen TC) |

### Ampere Architecture Highlights

1. **3rd Gen Tensor Cores** — FP16, BF16, TF32, INT8, INT4 (no FP8/FP4)
2. **BF16 support** — first consumer generation with native BF16 Tensor Core support
3. **GDDR6X memory** — PAM4 signaling for higher per-pin bandwidth
4. **NVLink 3rd gen** — 2-way GPU linking (112.5 GB/s), last consumer GPU with NVLink
5. **PCIe Gen 4** — 32 GB/s bidirectional (half of Gen 5)

## Memory Hierarchy Optimization

### The RTX 3090 Memory Stack

```
Register File:  64K × 32-bit per SM  │  ~0 latency   │  Per-thread private
Shared Memory:  100 KB per SM         │  ~20 cycles   │  Per-block shared
L1 Cache:       Configurable w/ smem  │  ~30 cycles   │  Per-SM (128 KB combined)
L2 Cache:       6 MB unified          │  ~200 cycles  │  Global shared
GDDR6X:         24 GB                 │  ~120 cycles  │  Global
PCIe (host):    System RAM            │  ~10K cycles  │  Cross-device
```

### Bandwidth at Each Level

| Level | Bandwidth | Latency | Size |
|-------|-----------|---------|------|
| Registers | ~40+ TB/s (aggregate) | 0 cycles | 64K/SM |
| Shared Memory | ~15 TB/s (aggregate) | ~20 cycles | 100 KB/SM |
| L2 Cache | ~3-4 TB/s | ~200 cycles | 6 MB |
| GDDR6X | **936 GB/s** | ~120 cycles | 24 GB |
| PCIe 4 | ~32 GB/s | ~10K cycles | System RAM |

The 936 GB/s GDDR6X bandwidth is the key number for memory-bound kernels — about half of Blackwell-era GDDR7 GPUs.

### Vectorized Memory Access

Vectorized loads are critical at 936 GB/s — they reduce instruction count and improve coalescing:

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

### L2 Cache Reality (6 MB)

The 6 MB L2 is **dramatically smaller** than Blackwell GPUs (96-128 MB). This fundamentally changes caching behavior:

| Working Set | Size | Fits in 6 MB L2? | Fits in 96+ MB L2? (Blackwell) |
|-------------|------|-------------------|-------------------------------|
| All Qwen3-8B RMSNorm weights | ~1.2 MB | Yes | Yes |
| Single RMSNorm weight (dim=4096, BF16) | 8 KB | Yes | Yes |
| RMSNorm input [1, 128, 4096] BF16 | 1 MB | Yes | Yes |
| RMSNorm input [1, 512, 4096] BF16 | 4 MB | Barely | Yes |
| RMSNorm input [1, 2048, 4096] BF16 | 16 MB | **No** | Yes |
| RMSNorm input [4, 512, 4096] BF16 | 16 MB | **No** | Yes |
| Attention scores (one layer, seq=2048) | ~32 MB | **No** | Yes |

**Key implications compared to Blackwell:**
- **Weight caching still works**: RMSNorm weights (~8 KB each, ~1.2 MB total) fit comfortably in 6 MB L2.
- **Activation caching mostly does NOT work**: Even moderate-sized activations (seq>512) spill from L2. Inter-kernel data reuse through L2 is much less effective.
- **L2 persistence hints have limited value**: With only 6 MB, there's little room to pin data.
- **Streaming access is more important**: Mark non-reusable data with `__ldcs` to avoid polluting the small L2.

**Optimization strategies for 6 MB L2:**

1. **Weight caching**: Still effective — weight tensors are small enough to stay hot in L2 across all rows.

2. **Streaming loads for activations**: Since activations won't stay in L2 anyway, use streaming loads to avoid evicting weights:
```cuda
float val = __ldcs(&input[idx]);  // Load cached streaming
```

3. **Minimize working set**: Keep kernel-local data as small as possible. The warp reduction scratch (64-128 bytes) is negligible.

4. **Don't rely on inter-kernel L2 reuse**: Unlike Blackwell, you cannot assume that RMSNorm output will still be in L2 when the next kernel reads it. Each kernel effectively does a full GDDR6X round-trip.

### Shared Memory and Bank Conflicts

100 KB shared memory per SM (configurable, 128 KB combined with L1), 32 banks:

```cuda
// BAD: stride-32 access — all threads hit same bank
float val = shared[threadIdx.x * 32];

// GOOD: consecutive access
float val = shared[threadIdx.x];

// GOOD: padding for 2D arrays
__shared__ float tile[32][33];  // 33 avoids bank conflicts
```

For RMSNorm, shared memory usage is minimal (warp reduction scratch: 64 bytes for 512 threads), so bank conflicts are not a concern.

## Occupancy and Thread Configuration

### Block Size Selection for RTX 3090

The RTX 3090's sm_86 supports a maximum of **1,536 threads per SM** (48 warps), not 2,048 like Blackwell. This changes optimal block sizes:

| Threads/Block | Blocks/SM | Threads/SM | Occupancy | Notes |
|---------------|-----------|------------|-----------|-------|
| **512** | 3 | 1,536 | **100%** | Optimal for RTX 3090 |
| 768 | 2 | 1,536 | 100% | Also valid, less common |
| 1024 | 1 | 1,024 | 66.7% | Suboptimal occupancy |
| 256 | 6 | 1,536 | 100% | Good for simple element-wise ops |

**Recommended block sizes:**

| Kernel Type | Threads/Block | Blocks/SM | Occupancy | Rationale |
|-------------|---------------|-----------|-----------|-----------|
| RMSNorm (hidden=4096) | **512** | 3 | 100% | Full occupancy, 4 vec elements/thread |
| RMSNorm (head_dim=128) | **64** | 24 | 100% | Small dim, many blocks fill SM |
| Element-wise (RoPE, GELU) | **256** | 6 | 100% | Simple ops, high occupancy |
| Attention | **256** | 4-6 | 67-100% | Balance shared mem + registers |

**Why 512 (not 1024) on RTX 3090:**
- 1024 threads/block = only 1 block/SM = 66.7% occupancy
- 512 threads/block = 3 blocks/SM = 100% occupancy
- With 936 GB/s bandwidth and moderate latency, 3 blocks/SM provides better latency hiding
- For hidden_size=4096 BF16: 2048 vec elements / 512 threads = 4 elements/thread — well-balanced

**Contrast with Blackwell GPUs (RTX 5090/6000 Pro):**
- Blackwell's sm_120 supports 2,048 threads/SM
- 1024 threads/block = 2 blocks/SM = 100% occupancy on Blackwell
- But only 66.7% occupancy on RTX 3090

### Grid Sizing for 82 SMs

For maximum GPU utilization, `num_blocks` should be >= 82, ideally a multiple:

```
RMSNorm grid = num_rows (one block per row):
  seq_len=1:     1 row    → 1/82 SMs (1.2% util) — poor
  seq_len=82:    82 rows  → 100% SM utilization (ideal)
  seq_len=128:   128 rows → 1.6 blocks/SM (partial wave)
  seq_len=512:   512 rows → 6.2 blocks/SM (well-saturated)
  seq_len=2048:  2048 rows → 25.0 blocks/SM (fully saturated)
```

Short sequences (< 82 rows) underutilize the GPU. The RTX 3090 has fewer SMs (82) than Blackwell GPUs (170/188), so it reaches full utilization at shorter sequence lengths — a small advantage for low-batch inference.

### Occupancy Calculator

```
Occupancy = Active Warps per SM / Max Warps per SM (48)

Limiting factors (sm_86):
1. Registers: 65536 / (threads_per_block × regs_per_thread)
2. Shared Memory: 100 KB / smem_per_block
3. Threads: 1536 / threads_per_block
4. Blocks: max 16 blocks/SM (sm_86 limit)

For our RMSNorm at 512 threads:
  Registers: ~20/thread × 512 = 10240 → fits ~6 blocks (threads limit to 3)
  Shared: 64 bytes → negligible
  Threads: 1536 / 512 = 3 blocks/SM → 48 warps/SM → 100% occupancy
```

Check register usage:
```bash
nvcc --ptxas-options=-v kernel_src/rmsnorm.cu
# Shows: "Used X registers, Y bytes smem, Z bytes cmem"
```

## Roofline Analysis

### RTX 3090 Roofline

```
Peak FP32: ~35.6 TFLOPS
Peak BW:   936 GB/s
Crossover: 35600 / 936 ≈ 38 FLOP/byte

Operations below 38 FLOP/byte are memory-bound.
```

| Kernel | FLOP/byte | Bound | Notes |
|--------|-----------|-------|-------|
| RMSNorm | ~6 | **Memory** | sum_sq + rsqrt + scale |
| GELU/SiLU | ~2 | **Memory** | Element-wise activation |
| RoPE | ~8 | **Memory** | sin/cos + rotate |
| Attention (no flash) | ~10-20 | **Memory** | Softmax + score @ V |
| Linear (GEMM) | ~100-1000 | **Compute** | Depends on dimensions |
| Flash Attention | ~50-200 | **Mixed** | Tiled, memory-efficient |

Most operations except GEMM and Flash Attention are memory-bound — same as Blackwell, but at lower absolute bandwidth. Custom kernels for RMSNorm, RoPE, and activation functions help by:
- Minimizing memory traffic (vectorized loads)
- Reducing kernel launch overhead (fusion)
- Eliminating intermediate materializations

### Bandwidth Efficiency Targets

For a well-optimized RMSNorm kernel on RTX 3090:

```
For [1, 2048, 4096] BF16:
  Read input:  2048 × 4096 × 2 bytes = 16 MB
  Read weight: 4096 × 2 bytes        = 8 KB (cached in L2)
  Write output: 16 MB
  Total: ~32 MB

  Theoretical minimum: 32 MB / 936 GB/s = 0.034 ms
  Realistic (35-45% efficiency): 0.076 - 0.098 ms

Compare to Blackwell (1790 GB/s): 0.018 ms theoretical — ~1.9x faster
```

For larger batches:
```
For [4, 2048, 4096] BF16:
  Total: ~128 MB
  Theoretical minimum: 128 MB / 936 GB/s = 0.137 ms
  Realistic: 0.30 - 0.39 ms
```

## VRAM Constraints: 24 GB

The 24 GB GDDR6X limits which models fit, similar to (but tighter than) the RTX 5090's 32 GB.

### Model Fit Analysis

| Model | BF16 Size | Fits in 24 GB? | Remaining for KV/Activations | Notes |
|-------|-----------|----------------|------------------------------|-------|
| Qwen3-8B BF16 | ~16 GB | Yes | ~8 GB | Tight but workable |
| Qwen3-8B INT8 | ~8 GB | Yes | ~16 GB | Good headroom |
| Qwen3-8B INT4 (GPTQ/AWQ) | ~4 GB | Yes | ~20 GB | Maximum headroom |
| Qwen3-14B BF16 | ~28 GB | **No** | — | Does not fit |
| Qwen3-14B INT4 | ~7 GB | Yes | ~17 GB | Quantization required |
| Qwen3-30B BF16 | ~60 GB | **No** | — | Does not fit |
| Qwen3-30B INT4 | ~15 GB | Yes | ~9 GB | Tight, INT4 required |

**No FP8 support**: Unlike Blackwell's 5th-gen Tensor Cores, the RTX 3090's 3rd-gen Tensor Cores do not support FP8. Quantization options are limited to INT8 and INT4.

**Key takeaways:**
- Qwen3-8B BF16 fits but with only ~8 GB headroom — sufficient for moderate-length inference
- For longer contexts or batched inference, INT8/INT4 quantization frees significant VRAM
- Models >8B parameters require quantization

### KV Cache Budget at 24 GB

For Qwen3-8B BF16 (~16 GB model weights), ~8 GB remains for KV cache:

```
KV cache per token per layer (Qwen3-8B):
  num_kv_heads=8, head_dim=128, BF16 (2 bytes)
  Per layer: 2 × 8 × 128 × 2 = 4,096 bytes/token
  36 layers: 36 × 4,096 = 147,456 bytes/token (~144 KB)

Available: ~8 GB = ~8,192 MB
Max tokens: 8,192 MB / 0.144 MB ≈ 56,889 tokens

With activations overhead (~2-3 GB):
  Available for KV: ~5-6 GB
  Max tokens: ~34,700 - 41,600 tokens
```

For longer contexts, use INT4 model weights (~4 GB) to free ~20 GB for KV cache, or use INT8 KV cache (halves KV memory). FP8 KV cache is not available on Ampere.

## Precision and Type Handling

### BF16 vs FP16 on Ampere

BF16 is supported on Ampere (sm_86) via 3rd-gen Tensor Cores. BF16 is preferred for LLM inference:

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

### Tensor Core Support on Ampere (3rd Gen)

| Format | Tensor Core Support | Notes |
|--------|-------------------|-------|
| FP16 | Yes | Full support |
| BF16 | Yes | Added in Ampere |
| TF32 | Yes | FP32 inputs, automatic |
| INT8 | Yes | For quantized GEMM |
| INT4 | Yes | For quantized GEMM |
| FP8 | **No** | Blackwell/Hopper only |
| FP4 | **No** | Blackwell only |

For our RMSNorm kernel, Tensor Cores are not used (it's a memory-bound element-wise operation). But the lack of FP8/FP4 means quantized GEMMs on RTX 3090 are limited to INT8/INT4, which affects end-to-end model optimization.

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

Warp shuffle -> shared memory -> warp shuffle:
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

Shared memory for 512 threads: `ceil(512/32) × 4 = 64 bytes`.

## Qwen3-8B on RTX 3090: Optimization Priorities

### Model Fit

Qwen3-8B in BF16 (~16 GB) leaves ~8 GB of GDDR6X free for:
- KV caches (moderate context lengths)
- Activations during inference

This is much tighter than Blackwell GPUs. For high-throughput serving or long contexts, use INT8/INT4 quantized weights.

### Where Custom Kernels Help Most

| Component | % of Inference Time (approx) | Bound | Custom Kernel Value |
|-----------|------------------------------|-------|---------------------|
| Linear projections | ~65% | Compute | Low (cuBLAS is near-optimal) |
| Attention | ~15% | Mixed | Medium (Flash Attention 2) |
| **RMSNorm** | **~8%** | **Memory** | **High** (vectorized kernel) |
| RoPE | ~4% | Memory | Medium |
| Activation (SiLU) | ~3% | Memory | Low |
| Framework overhead | ~5% | CPU | Use torch.compile / CUDA graphs |

On RTX 3090, memory-bound kernels like RMSNorm represent a **larger fraction** of total inference time compared to Blackwell GPUs. This is because the 936 GB/s bandwidth takes roughly 2x longer per kernel, while compute-bound linear projections don't scale proportionally. Custom kernels therefore have **relatively more impact** on RTX 3090.

### RMSNorm Module Breakdown (Qwen3-8B)

| Module | Count | Dimension | Threads | Vec Elements/Thread |
|--------|-------|-----------|---------|---------------------|
| input_layernorm | 36 | 4096 | 512 | 4 |
| post_attention_layernorm | 36 | 4096 | 512 | 4 |
| model.norm | 1 | 4096 | 512 | 4 |
| self_attn.q_norm | 36 | 128 | 64 | 1 |
| self_attn.k_norm | 36 | 128 | 64 | 1 |

### Combining with torch.compile

The RTX 3090 works well with `torch.compile`:
```python
# Option 1: torch.compile only (no custom kernels)
model = torch.compile(model, mode="reduce-overhead")
# Expected: ~25-35% end-to-end speedup

# Option 2: Custom kernels only
inject_optimized_kernels(model)
# Expected: ~8-12% end-to-end speedup (RMSNorm is a bigger fraction on RTX 3090)

# Option 3: Both (requires custom op registration)
# See torch.library.custom_op pattern below
```

Note: Custom kernels and `torch.compile` are mutually exclusive without custom op registration via `torch.library`.

## Compilation

### NVCC Flags for RTX 3090

```bash
nvcc \
    -O3 \
    -arch=sm_86 \
    -gencode=arch=compute_86,code=sm_86 \
    --use_fast_math \
    -lineinfo \
    --threads=4 \
    your_kernel.cu
```

| Flag | Purpose |
|------|---------|
| `-arch=sm_86` | Native SASS for Ampere sm_86 |
| `-gencode=arch=compute_86,code=sm_86` | Explicit gencode for sm_86 |
| `--use_fast_math` | Fast `rsqrtf`, `__expf`, etc. (~10% faster math) |
| `-lineinfo` | Debug info for ncu/nsys (no perf impact) |
| `--threads=4` | Parallel ptxas compilation |
| `-maxrregcount=N` | Limit registers if needed for occupancy |

**Requires CUDA Toolkit 11.1+** for sm_86 support. CUDA 12.x recommended for best performance.

### setup.py Configuration

```python
extra_compile_args={
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-arch=sm_86",
        "-gencode=arch=compute_86,code=sm_86",
        "--use_fast_math",
        "-lineinfo",
        "--threads=4",
    ],
},
```

## Profiling on RTX 3090

### NVIDIA Nsight Systems (nsys)

```bash
nsys profile -o rtx3090_profile python your_script.py

# What to look for:
# - Kernel duration (longer than Blackwell due to lower bandwidth)
# - PCIe Gen 4 transfer time (model loading — slower than Gen 5)
# - GPU idle gaps between kernels (framework overhead)
# - torch.compile warmup cost
```

### NVIDIA Nsight Compute (ncu)

```bash
# Full analysis
ncu --set full -o metrics.ncu-rep python your_script.py

# Key metrics for RTX 3090:
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
| `dram__throughput` | >35% of peak | GDDR6X utilization |
| `lts__throughput` (L2) | Low-moderate | 6 MB L2 limits caching benefit |
| `sm__throughput` | Low (10-20%) | Expected for memory-bound |
| `sm__warps_active` | >50% | Occupancy indicator |
| `l1tex__throughput` | Moderate | Shared mem + L1 traffic |

### Common Performance Issues

1. **Low DRAM throughput**
   - Non-coalesced access -> vectorize loads
   - Low occupancy -> use 512 threads/block, check register pressure

2. **High kernel launch overhead**
   - Many small kernels -> fuse, use CUDA graphs, torch.compile
   - Python dispatch -> use `torch.inference_mode()`

3. **PCIe Gen 4 bottleneck on model load**
   - PCIe Gen 4 x16: ~25-30 GB/s effective (vs ~50-60 GB/s on Gen 5)
   - Qwen3-8B BF16 load time: ~0.5-0.6s (one-time cost)

4. **Register spilling**
   - Too many local variables -> reduce register usage
   - sm_86 allows max 16 blocks/SM — register spilling reduces block count

5. **L2 thrashing**
   - With only 6 MB L2, even moderate working sets cause evictions
   - Use `__ldcs` for data that won't be reused
   - Don't over-tune for L2 hits — assume GDDR6X round-trip for most data

6. **VRAM exhaustion**
   - 24 GB fills up quickly with BF16 models
   - Monitor with `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
   - Use INT8/INT4 quantization for models >8B parameters
   - Reduce batch size or sequence length if needed

## Best Practices Summary for RTX 3090

1. **Vectorize loads** — `__nv_bfloat162`, `__half2`, `float4` for coalesced 32/128-bit transactions.
2. **Use 512 threads/block** for reduction kernels — 3 blocks/SM, full occupancy on sm_86.
3. **Standard unroll** (`#pragma unroll 4`) — GDDR6X latency is manageable with 3 blocks/SM for latency hiding.
4. **Don't over-rely on L2** — 6 MB means most activations won't stay cached; optimize for GDDR6X throughput.
5. **Accumulate in FP32** — always reduce in float, convert back at the end.
6. **Profile with ncu** — target >35% DRAM throughput for bandwidth-bound kernels.
7. **Consider torch.compile** — 25-35% end-to-end speedup, complementary to custom kernels.
8. **BF16 is the default for 8B models** — 24 GB fits Qwen3-8B; use INT8/INT4 for larger models (no FP8).
9. **Use explicit type conversions** — `to_float()` / `from_float()` helpers for PyTorch compat.
10. **Grid size >= 82** — keep all SMs busy; short sequences underutilize the GPU.

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

### Performance Comparison

| Configuration | Speedup | Notes |
|:---|:---:|:---|
| Custom kernels only | ~1.08-1.12x | More impact on RTX 3090 than Blackwell |
| torch.compile only | ~1.25-1.35x | Requires warmup compilation pass |
| Both (custom op registered) | Best of both | Requires `torch.library` registration |

On RTX 3090, custom kernels provide relatively more benefit than on Blackwell because memory-bound operations take longer (lower bandwidth), so the optimization gains per kernel are larger in absolute time.

## CUDA Graphs

CUDA graphs capture a sequence of kernel launches and replay them with near-zero CPU overhead. On RTX 3090, individual kernels take longer than on Blackwell (lower bandwidth), so CUDA graph overhead reduction is proportionally smaller — but still valuable.

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

### RTX 3090-Specific Considerations

With 936 GB/s bandwidth, a small RMSNorm kernel ([1, 1, 4096] BF16) executes in ~20-40 us (vs ~10-20 us on Blackwell). CPU launch overhead of ~5-10 us per kernel is a smaller fraction of total time, so CUDA graphs provide proportionally less benefit than on faster GPUs. Still, for decode-phase inference with many small kernels, CUDA graphs are valuable.

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

To make your RTX 3090-optimized kernel available to others:

1. Use the `TORCH_LIBRARY_EXPAND` binding pattern (see torch.compile section above)
2. Set `cuda-capabilities = ["8.6"]` in `build.toml`
3. Build and upload:

```bash
pip install kernel-builder
kernel-builder build
huggingface-cli upload your-username/qwen3-rmsnorm-rtx3090 ./dist
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

**When to use what:**
- **Prototyping / research / small-scale inference** -> HF Transformers + Hub kernels or manual injection
- **Production serving** -> vLLM (CustomOp plugins) or TRT-LLM (C++ plugins) — these have their own optimized kernel stacks

## RTX 3090 as a Legacy Consumer GPU: Practical Considerations

### NVLink Support (Unique Among Consumer GPUs)

The RTX 3090 is one of the last consumer GPUs to support NVLink (3rd gen, 2-way). Two RTX 3090s can be connected with an NVLink bridge:
- **112.5 GB/s** bidirectional bandwidth (vs PCIe Gen 4's ~32 GB/s)
- Enables more efficient tensor parallelism than PCIe-only GPUs (RTX 4090, RTX 5090)
- NCCL/AllReduce operations benefit from NVLink

However, 2-way NVLink is still much less capable than data center NVLink (A100's 600 GB/s, B200's 1,800 GB/s).

### No ECC Memory

The RTX 3090 does not have ECC GDDR6X. For inference workloads, this is rarely an issue.

### PCIe Gen 4 (Not Gen 5)

PCIe Gen 4 x16 provides ~32 GB/s bidirectional (~25-30 GB/s effective), half the bandwidth of Gen 5. This primarily affects:
- Model loading time (one-time cost)
- CPU-GPU data transfers during inference (typically small)
- Multi-GPU communication when not using NVLink

### Power and Cooling

At 350W TDP, the RTX 3090 requires:
- A high-quality PSU (750W+ recommended for system)
- Good case airflow (blower or dual-fan cooler)
- Adequate PCIe slot spacing (typically 3-slot card)

### Driver and Software Support

- CUDA Toolkit 11.1+ required for sm_86 (CUDA 12.x recommended)
- Well-supported with mature drivers
- PyTorch has excellent sm_86 support
- Flash Attention 2 supports sm_86
- Most inference frameworks (vLLM, TRT-LLM) support sm_86

### Age Considerations

The RTX 3090 is an Ampere-generation GPU (2020). While still highly capable for inference:
- Missing FP8/FP4 Tensor Core support limits quantization options
- No Blackwell hardware features (TMA, TMEM, CTA clusters)
- Newer software optimizations may target sm_90+ (Hopper) or sm_120+ (Blackwell)
- Still widely deployed and cost-effective for 8B-class model inference

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

**3. sm_86 not recognized**
```
nvcc fatal: Unsupported gpu architecture 'compute_86'
```
Fix: Requires CUDA Toolkit 11.1+. Check version:
```bash
nvcc --version
```

**4. Out of memory errors**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```
With only 24 GB VRAM, this is common. Solutions:
- Use INT8 or INT4 quantized models (no FP8 on Ampere)
- Reduce batch size
- Reduce max sequence length (smaller KV cache)
- Use `torch.cuda.empty_cache()` between operations

### Integration Errors

**5. `isinstance()` misses model's RMSNorm**
```python
# WRONG — misses Qwen3RMSNorm, LlamaRMSNorm, etc.
if isinstance(module, torch.nn.RMSNorm):

# CORRECT — catches all variants
if 'RMSNorm' in type(module).__name__:
```

**6. Wrong epsilon attribute name**
```python
# LLaMA / Qwen: 'variance_epsilon'
# Others: 'eps'
# Safe pattern:
eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)
```

**7. Kernel patching doesn't survive `enable_model_cpu_offload()`**

Fix: Inject kernels AFTER `.to("cuda")`, BEFORE `enable_model_cpu_offload()`.

### Verification

```python
# Check CUDA architecture is detected correctly
python -c "import torch; print(torch.cuda.get_device_capability())"
# RTX 3090 should print (8, 6)

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

Both vLLM and TensorRT-LLM support the RTX 3090 (sm_86). While Ampere-specific kernel optimization opportunities are more limited than Blackwell (no new hardware features to exploit), there are still gains to be had.

### Ampere-Specific Considerations

**What's available on Ampere (but not older):**
- Async copy (`cp.async`) — hardware-accelerated shared memory loads
- TF32 — transparent FP32 acceleration in Tensor Cores
- BF16 Tensor Cores — first generation with native BF16
- Fine-grained structured sparsity (2:4) for Tensor Cores

**What's NOT available on Ampere (Hopper/Blackwell only):**
- TMA (Tensor Memory Accelerator) — no hardware-managed DMA
- WGMMA / CTA clusters — no cross-SM cooperation
- TMEM — no dedicated tensor memory
- FP8/FP4 — limits quantization options
- Distributed shared memory — no cross-SM shared memory access

### General Principles for Kernel Optimization

#### 1. Profile Before Optimizing

```bash
# vLLM: system-wide profiling
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    python -m vllm.entrypoints.openai.api_server --model your-model

# Per-kernel analysis
ncu --clock-control None --kernel-id :::1 --set full \
    python your_benchmark.py
```

On RTX 3090, look for: low DRAM throughput below 35% of 936 GB/s (coalescing issues), poor occupancy (should be 100% with 512 threads/block), L2 miss rate (expected to be high with 6 MB L2).

#### 2. Fuse Memory-Bound Operations

Kernel fusion is even more valuable on RTX 3090 than on Blackwell:

```
Unfused (3 kernel launches, 3 memory reads + writes):
  RMSNorm -> SiLU -> INT8 Quantize

Fused (1 kernel launch, 1 read + 1 write):
  RMSNorm_SiLU_INT8Quant
```

At 936 GB/s, each memory round-trip takes ~2x longer than on Blackwell. Eliminating two intermediate round-trips via fusion provides ~2x more absolute time savings per fusion. The small 6 MB L2 also means unfused kernels cannot rely on intermediate data staying cached between launches.

#### 3. Design for CUDA Graph Compatibility

Both vLLM and TRT-LLM use CUDA graphs on Ampere. Your kernel must:

- **No dynamic memory allocation** inside the kernel
- **No host-device synchronization** in the hot path
- **Fixed grid/block dimensions** for a given input shape
- **No conditional kernel launches** based on runtime values

#### 4. Design for Tensor Parallelism Compatibility

With NVLink available on RTX 3090 (unlike RTX 4090/5090), 2-GPU tensor parallelism is more practical:
- NVLink provides ~3.5x the bandwidth of PCIe Gen 4 for AllReduce
- Your kernel should accept arbitrary dimension sizes (TP sharding produces non-standard sizes)

#### 5. Quantization: INT8/INT4 Only

Without FP8, the RTX 3090's quantization options are:
- **INT8**: W8A8 or W8A16 — good balance of accuracy and performance
- **INT4**: GPTQ, AWQ — most aggressive compression, some accuracy loss
- **No FP8/FP4**: Don't write kernels assuming FP8 Tensor Core support

### Contributing Kernels

**To vLLM:**

1. **CustomOp / PluggableLayer**: Register with `@CustomOp.register("name")`. Implement `forward_cuda()` for CUDA, `forward_native()` for PyTorch fallback.

2. **Testing**: Use pytest with CUDA dependencies. Run warmup before collecting performance numbers.

**To TensorRT-LLM:**

1. **C++ Plugins**: Implement `IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`.

2. **Autotuning**: Profile different tactics for sm_86 specifically (tile sizes, thread configs).

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│       RTX 3090 Ampere Quick Ref         │
├─────────────────────────────────────────┤
│ Arch:       sm_86 (Ampere GA102)        │
│ SMs:        82                          │
│ CUDA Cores: 10,496                      │
│ Memory:     24 GB GDDR6X (dedicated)    │
│ Bandwidth:  936 GB/s (exclusive)        │
│ L2 Cache:   6 MB                        │
│ Shared/SM:  100 KB (128 KB L1+shared)   │
│ Regs/SM:    65,536                      │
│ Threads/SM: 1,536 (not 2048!)           │
│ TDP:        350W                        │
│ NVLink:     2-way (112.5 GB/s)          │
│ CUDA:       >= 11.1 required            │
├─────────────────────────────────────────┤
│ Compile:  -arch=sm_86                   │
│ Threads:  512 (reduction), 256 (elem)   │
│ Unroll:   #pragma unroll 4              │
│ Vectorize: bf162 / half2 / float4       │
│ Peak BW:  936 GB/s (realistic: ~350)    │
├─────────────────────────────────────────┤
│ VRAM:     24 GB — plan model fit!       │
│ Qwen3-8B BF16: ~16 GB (fits, tight)    │
│ Qwen3-14B: use INT4 (~7 GB)            │
│ No FP8 — INT8/INT4 only                │
└─────────────────────────────────────────┘
```
