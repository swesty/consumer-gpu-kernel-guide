# RTX 3090 Ampere: Memory Architecture and Kernel Implications

Notes on how the RTX 3090's discrete GDDR6X memory architecture affects CUDA kernel design, with specific focus on RMSNorm for Qwen3-8B.

## Architecture Overview

The RTX 3090 is a consumer discrete GPU based on the GA102 die (Ampere architecture, Samsung 8nm). The GPU has its own dedicated 24 GB of GDDR6X memory connected via a 384-bit interface, completely separate from the host system's DDR4/DDR5 memory. Data moves between CPU and GPU over PCIe Gen 4.

```
┌─────────────────────┐         PCIe 4.0         ┌──────────────────────┐
│      Host CPU       │◄───── ~32 GB/s BD ──────►│  RTX 3090 (350W)     │
│  (DDR4/5 sys RAM)   │                           │  GA102 Ampere        │
└─────────────────────┘                           │                      │
                                                  │  82 SMs              │
                                                  │  10,496 CUDA cores   │
                                                  │                      │
                                                  │  ┌──────────────┐    │
                                                  │  │  24 GB GDDR6X│    │
                                                  │  │  384-bit bus  │    │
                                                  │  │  936 GB/s     │    │
                                                  │  └──────────────┘    │
                                                  │                      │
                                                  │  6 MB L2 cache       │
                                                  └──────────────────────┘
```

### Comparison with Other GPUs

| | RTX 3090 (Ampere) | RTX 5090 (Blackwell) | RTX 6000 Pro (Blackwell) | GB10 (DGX Spark) |
|---|---|---|---|---|
| Memory type | GDDR6X (dedicated) | GDDR7 (dedicated) | GDDR7 (dedicated, ECC) | LPDDR5X (unified) |
| Memory size | 24 GB | 32 GB | 96 GB | 128 GB |
| GPU bandwidth | 936 GB/s | 1,792 GB/s | 1,790 GB/s | 273 GB/s |
| CPU-GPU transfer | PCIe 4.0 (~32 GB/s) | PCIe 5.0 (~64 GB/s) | PCIe 5.0 (~64 GB/s) | Zero-copy (C2C) |
| L2 cache | **6 MB** | 96 MB | 128 MB | 24 MB |
| BW ratio vs RTX 3090 | 1x (baseline) | 1.9x | 1.9x | 0.29x |

### The 6 MB L2 Cache: The Defining Characteristic

The RTX 3090's 6 MB L2 cache is **16-21x smaller** than Blackwell GPUs (96-128 MB) and **4x smaller** than GB10 (24 MB). This is the single most important architectural difference for kernel design. On Blackwell, many intermediate activations stay in L2 between kernel launches; on RTX 3090, virtually nothing does.

## What the Discrete Memory Architecture Means for Kernels

### 1. Exclusive Bandwidth — No Contention

The 936 GB/s is available exclusively to the GPU. Unlike the GB10, where CPU memory traffic reduces GPU bandwidth, the RTX 3090's GDDR6X bandwidth is fully dedicated to GPU workloads regardless of CPU activity.

**Implication:** No need to worry about CPU-side memory traffic during kernel execution. The kernel can rely on consistent bandwidth.

### 2. GDDR6X Latency Characteristics

GDDR6X uses PAM4 signaling for higher per-pin bandwidth. Its access latency is moderate:

| Memory Type | Typical Latency | Bandwidth | Bus Width |
|-------------|----------------|-----------|-----------|
| GDDR6X (RTX 3090) | ~100-150 ns | 936 GB/s | 384-bit |
| GDDR7 (RTX 5090/6000 Pro) | ~100-120 ns | 1,790 GB/s | 512-bit |
| LPDDR5X (GB10) | ~150-200 ns | 273 GB/s | — |
| HBM2e (A100) | ~80-100 ns | 2,039 GB/s | 5120-bit |

**Implication:** GDDR6X latency is similar to GDDR7, so standard unroll depth (`#pragma unroll 4`) is sufficient. The memory controller can saturate bandwidth without aggressive unrolling, especially with 3 blocks/SM providing ample latency hiding.

### 3. The 6 MB L2 Cache — Fundamentally Different from Blackwell

The 6 MB L2 cache changes kernel design philosophy:

**What fits in L2 on RTX 3090:**
| Data | Size (Qwen3-8B, BF16) | Fits in 6 MB L2? | Fits in 96+ MB L2? (Blackwell) |
|------|------------------------|-------------------|-------------------------------|
| All RMSNorm weights | ~1.2 MB | Yes | Yes |
| Single RMSNorm weight (dim=4096) | 8 KB | Yes (trivially) | Yes |
| RMSNorm input [1, 128, 4096] | 1 MB | Yes | Yes |
| RMSNorm input [1, 256, 4096] | 2 MB | Yes | Yes |
| RMSNorm input [1, 512, 4096] | 4 MB | Barely | Yes |
| RMSNorm input [1, 2048, 4096] | 16 MB | **No** | Yes |
| RMSNorm input [4, 512, 4096] | 16 MB | **No** | Yes |
| Single layer weights | ~150 MB | **No** | Partially |
| KV cache (any practical size) | >10 MB | **No** | Partially |

**Key design implications:**

1. **Weights cache well**: RMSNorm weights (~8 KB each, ~1.2 MB total) fit entirely in 6 MB L2. Weight broadcasting from L2 is effective even with 82 SMs reading concurrently.

2. **Activations do NOT cache**: For any practical batch/sequence beyond very short sequences, activation tensors spill from L2. Every kernel launch effectively reads from and writes to GDDR6X.

3. **Inter-kernel L2 reuse is minimal**: On Blackwell, the output of RMSNorm might still be in L2 when the next kernel (linear projection) reads it. On RTX 3090, this rarely happens. Each kernel must assume a full GDDR6X round-trip.

4. **L2 persistence hints have limited utility**: With only 6 MB, pinning data in L2 for one kernel evicts data needed by another. Use sparingly (primarily for broadcast weights).

5. **Streaming loads are more important**: Since most data won't stay in L2 anyway, using `__ldcs` (cached streaming) for non-reusable data helps avoid evicting the small amount that IS cached (weights).

### 4. PCIe Gen 4 Transfer Cost

Loading Qwen3-8B onto the RTX 3090 requires an explicit PCIe transfer:

```
Qwen3-8B in BF16: ~16 GB
PCIe 4.0 x16 effective: ~25 GB/s
Transfer time: ~0.6-0.7s

Compare PCIe Gen 5 (Blackwell): ~0.3-0.4s — roughly 2x faster
```

This is a one-time startup cost. Once weights are in GDDR6X, all subsequent kernel execution is at full 936 GB/s. The 24 GB of GDDR6X holds Qwen3-8B with some room for KV caches and activations.

### 5. No Page Migration / NUMA Effects

On discrete GPUs, memory is either on the GPU or not. There are no page migration heuristics, no NUMA effects, no gradual warming of page placement. Once `model.to("cuda")` completes, everything is at full speed immediately.

**Implication:** Benchmarking is straightforward — no warmup needed for memory subsystem (still need kernel cache warmup for instruction caches though).

## Impact on RMSNorm Kernel Design

### What Benefits from 936 GB/s Bandwidth

The RTX 3090's 936 GB/s means RMSNorm kernels take roughly 2x longer than on Blackwell, but the kernel is still bandwidth-bound (RMSNorm is ~6 FLOP/byte, far below the compute-memory crossover at ~38 FLOP/byte):

```
For [1, 2048, 4096] BF16:
  Total data: ~32 MB (read input + weight, write output)

  RTX 3090:   32 MB / 936 GB/s  = 0.034 ms theoretical minimum
  RTX 5090:   32 MB / 1792 GB/s = 0.018 ms theoretical minimum
  RTX 6000:   32 MB / 1790 GB/s = 0.018 ms theoretical minimum
  GB10:       32 MB / 273 GB/s  = 0.117 ms theoretical minimum

  ~1.9x slower than Blackwell, ~3.4x faster than GB10
```

Achieving 30-40% of peak bandwidth is realistic with vectorized loads.

### Why 512 Threads/Block (Not 1024)

The RTX 3090's sm_86 has a maximum of 1,536 threads per SM (48 warps), unlike Blackwell's 2,048 (64 warps). This drives the block size choice:

- **512 threads/block**: 3 blocks/SM -> 1,536 threads/SM (100% occupancy)
- **1024 threads/block**: 1 block/SM -> 1,024 threads/SM (66.7% occupancy)
- 100% occupancy provides better latency hiding with 3 independent blocks per SM
- For hidden_size=4096 BF16: 2048 vec elements / 512 threads = 4 elements/thread — well-balanced

**Contrast with Blackwell (sm_120 / sm_121):**
- 1024 threads/block = 2 blocks/SM = 2048 threads = 100% occupancy on Blackwell
- But only 66.7% occupancy on RTX 3090's sm_86

### Why Unroll 4

GDDR6X latency (~100-150 ns) is similar to GDDR7, and the 3 blocks/SM at 100% occupancy provide ample warps for latency hiding:
- The memory controller can achieve high throughput with 4 outstanding loads per thread
- Deeper unrolling would increase register pressure without proportional benefit
- With 48 warps active per SM, the scheduler has plenty of independent work to issue

### L2 Cache for Weight Broadcast

Despite the tiny 6 MB L2, weight tensor caching is still very effective. The weight tensor (~8 KB for BF16 dim=4096) is:
- Read from GDDR6X once by the first row
- Served from L2 for all subsequent rows (fits trivially in 6 MB)
- Even with 82 SMs issuing concurrent weight reads, L2 bandwidth (~3-4 TB/s) far exceeds what's needed

For a batch of [4, 2048, 4096]: 8192 rows all reading the same 8 KB weight — the weight is read from GDDR6X once and served from L2 for the remaining 8191 rows.

## VRAM Management: The 24 GB Constraint

### Model Fit Analysis

The 24 GB VRAM is the RTX 3090's primary constraint for LLM inference:

| Scenario | Model Weights | KV Cache (est.) | Activations | Total | Fits? |
|----------|--------------|-----------------|-------------|-------|-------|
| Qwen3-8B BF16, short ctx | 16 GB | 1 GB | 2 GB | ~19 GB | Yes |
| Qwen3-8B BF16, long ctx (32K) | 16 GB | 4.5 GB | 3 GB | ~23.5 GB | Barely |
| Qwen3-8B BF16, very long ctx (128K) | 16 GB | 18 GB | 4 GB | ~38 GB | **No** |
| Qwen3-8B INT8, long ctx (32K) | 8 GB | 4.5 GB | 3 GB | ~15.5 GB | Yes |
| Qwen3-8B INT4, very long ctx | 4 GB | 9 GB | 3 GB | ~16 GB | Yes |
| Qwen3-14B INT4 | 7 GB | 2 GB | 3 GB | ~12 GB | Yes |
| Qwen3-30B INT4 | 15 GB | 3 GB | 3 GB | ~21 GB | Yes |

**No FP8 quantization available**: The RTX 3090's 3rd-gen Tensor Cores do not support FP8. Quantization options are limited to INT8 (via INT8 Tensor Cores) and INT4 (GPTQ/AWQ).

**Strategies for maximizing VRAM utilization:**
1. **INT8/INT4 quantized weights** — reduce model footprint (no FP8 on Ampere)
2. **INT8 KV cache** — reduces KV cache size (no FP8 KV cache on Ampere)
3. **Paged attention** (vLLM-style) — eliminates wasted KV cache allocation
4. **Sliding window attention** — caps KV cache growth for long sequences
5. **Gradient checkpointing** — essential if fine-tuning on RTX 3090

### Comparison with Blackwell GPUs

| Model | RTX 3090 (24 GB) | RTX 5090 (32 GB) | RTX 6000 Pro (96 GB) |
|-------|-------------------|-------------------|----------------------|
| Qwen3-8B BF16 | Fits (tight) | Fits (comfortable) | Fits (80 GB free) |
| Qwen3-14B BF16 | **No** | **No** (tight) | Fits |
| Qwen3-30B BF16 | **No** | **No** | Fits |
| Qwen3-8B FP8 | **N/A** (no FP8) | Fits | Fits |
| Qwen3-14B INT4 | Fits | Fits | Fits |
| Qwen3-30B INT4 | Fits (tight) | Fits | Fits |

The RTX 3090 is best suited for Qwen3-8B class models. For larger models, aggressive quantization is required.

## NVLink: A Unique RTX 3090 Feature

The RTX 3090 supports NVLink 3rd gen (2-way), the last consumer GPU to do so:

```
┌──────────────┐    NVLink 3.0     ┌──────────────┐
│  RTX 3090 #1 │◄═══ 112.5 ════►  │  RTX 3090 #2 │
│  24 GB GDDR6X│    GB/s BD        │  24 GB GDDR6X│
└──────────────┘                   └──────────────┘
      Combined: 48 GB VRAM (with NCCL/TP)
      NVLink BW: 112.5 GB/s (vs PCIe 4: 32 GB/s)
```

**Benefits for LLM inference:**
- 2x RTX 3090 via NVLink = 48 GB combined VRAM — fits Qwen3-14B BF16 or Qwen3-30B INT4 with headroom
- AllReduce operations for tensor parallelism are ~3.5x faster than PCIe
- Lower latency for TP communication

**Limitations:**
- Only 2-way (not 4+ like data center GPUs)
- 112.5 GB/s is still much less than A100 NVLink (600 GB/s) or B200 (1,800 GB/s)
- Requires NVLink bridge hardware
- RTX 4090 and later consumer GPUs dropped NVLink support

## When Discrete Memory Architecture Matters More

### Multi-Stream Execution

The exclusive bandwidth enables running multiple CUDA streams concurrently without interference:
```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    out1 = rmsnorm(x1, w1)
with torch.cuda.stream(stream2):
    out2 = rmsnorm(x2, w2)
```

On GB10, multi-stream execution would compete for the same 273 GB/s shared pipe. On RTX 3090, the 936 GB/s has headroom for concurrent kernels.

### torch.compile and CUDA Graphs

The RTX 3090 benefits from `torch.compile` and CUDA graphs:

```python
model = torch.compile(model, mode="reduce-overhead")
```

Since individual kernels take ~2x longer than on Blackwell, the relative benefit of eliminating launch overhead is smaller. But for decode-phase inference with many small kernels, CUDA graphs still provide meaningful latency reduction.

## Summary

| Concern | Impact on RMSNorm Kernel | Notes |
|---------|--------------------------|-------|
| Exclusive bandwidth | Positive — consistent 936 GB/s | No CPU contention |
| GDDR6X latency | Moderate — similar to GDDR7 | Standard unroll (4) sufficient |
| **6 MB L2 cache** | **Limiting** — activations don't cache | Design for GDDR6X throughput, not L2 hits |
| 24 GB VRAM | Constraining — limits model size | Qwen3-8B fits; larger need INT8/INT4 |
| PCIe Gen 4 | Slower model loading than Gen 5 | One-time ~0.6s cost |
| 1,536 max threads/SM | Affects block sizing | Use 512 threads (not 1024) for 100% occupancy |
| No FP8/FP4 | Limits quantization options | INT8/INT4 only |
| NVLink (2-way) | Unique advantage | Enables 2-GPU TP, 48 GB combined |
| Page migration | N/A | No unified memory effects |
| Kernel code vs Blackwell | `MAX_THREADS=512`, `sm_86`, same vectorization | Different tuning, same algorithm |
| Kernel code vs GB10 | `MAX_THREADS=512`, `unroll 4` (not 8) | Similar threads, less unrolling needed |
