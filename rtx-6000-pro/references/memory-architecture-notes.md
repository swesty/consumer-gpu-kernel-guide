# RTX 6000 Pro Blackwell: Memory Architecture and Kernel Implications

Notes on how the RTX 6000 Pro's discrete GDDR7 memory architecture affects CUDA kernel design, with specific focus on RMSNorm for Qwen3-8B.

## Architecture Overview

The RTX 6000 Pro Blackwell is a discrete workstation GPU based on the GB202 die. The GPU has its own dedicated 96 GB of GDDR7 memory connected via a 512-bit interface, completely separate from the host system's DDR5 memory. Data moves between CPU and GPU over PCIe Gen 5.

```
┌─────────────────────┐         PCIe 5.0         ┌──────────────────────┐
│      Host CPU       │◄───── ~64 GB/s BD ──────►│  RTX 6000 Pro (600W) │
│  (DDR5 system RAM)  │                           │  GB202 Blackwell     │
└─────────────────────┘                           │                      │
                                                  │  188 SMs             │
                                                  │  24,064 CUDA cores   │
                                                  │                      │
                                                  │  ┌──────────────┐    │
                                                  │  │  96 GB GDDR7 │    │
                                                  │  │  512-bit bus  │    │
                                                  │  │  1.79 TB/s    │    │
                                                  │  └──────────────┘    │
                                                  │                      │
                                                  │  128 MB L2 cache     │
                                                  └──────────────────────┘
```

### Comparison with GB10 (DGX Spark)

| | RTX 6000 Pro Blackwell | GB10 (DGX Spark) |
|---|---|---|
| Memory type | GDDR7 (dedicated) | LPDDR5X (unified with CPU) |
| Memory size | 96 GB | 128 GB |
| GPU bandwidth | 1,790 GB/s (GPU exclusive) | 273 GB/s (shared with CPU) |
| CPU↔GPU transfer | PCIe 5.0 copy (~64 GB/s) | Zero-copy (NVLink-C2C) |
| Memory contention | None — GPU has exclusive bandwidth | CPU and GPU compete |
| L2 cache | 128 MB | 24 MB |
| Bandwidth ratio | 6.6x higher | 1x (baseline) |

## What the Discrete Memory Architecture Means for Kernels

### 1. Exclusive Bandwidth — No Contention

The 1.79 TB/s is available exclusively to the GPU. Unlike the GB10, where CPU memory traffic reduces GPU bandwidth, the RTX 6000 Pro's GDDR7 bandwidth is fully dedicated to GPU workloads regardless of what the CPU is doing.

**Implication:** No need to worry about CPU-side memory traffic during kernel execution. The kernel can rely on consistent bandwidth.

### 2. Lower Memory Latency than LPDDR5X

GDDR7 has lower access latency than LPDDR5X:

| Memory Type | Typical Latency | Bandwidth |
|-------------|----------------|-----------|
| GDDR7 (RTX 6000 Pro) | ~100-120 ns | 1,790 GB/s |
| LPDDR5X (GB10) | ~150-200 ns | 273 GB/s |
| HBM3 (H100) | ~80-100 ns | 3,350 GB/s |

**Implication:** Standard unroll depth (`#pragma unroll 4`) is sufficient. The memory controller can saturate bandwidth with fewer in-flight requests compared to LPDDR5X.

### 3. Massive L2 Cache (128 MB)

The 128 MB L2 cache is a significant asset:

**What fits in L2 on RTX 6000 Pro:**
| Data | Size (Qwen3-8B, BF16) | Fits in 128 MB L2? |
|------|------------------------|---------------------|
| All RMSNorm weights | ~1.2 MB | Yes (trivially) |
| Single layer's full weights | ~150 MB | Partially |
| KV cache (seq=2048, all heads) | ~2.4 GB | No |
| KV cache (seq=256, all heads) | ~300 MB | No |
| RMSNorm input (batch=1, seq=2048, dim=4096) | 16 MB | Yes |
| RMSNorm input (batch=4, seq=2048, dim=4096) | 64 MB | Yes |

The L2 cache is large enough to hold the entire RMSNorm input for most practical batch/sequence combinations. This means:
- **Weight data**: Permanently hot in L2 across all rows
- **Input/output data**: For moderate batches, the normalized output may still be in L2 when the next kernel (e.g., a linear projection) reads it
- **Cross-kernel reuse**: Sequential kernels within a transformer layer can benefit from L2 residency of intermediate activations

### 4. PCIe Transfer Cost

Unlike the GB10's zero-copy unified memory, loading Qwen3-8B onto the RTX 6000 Pro requires an explicit PCIe transfer:

```
Qwen3-8B in BF16: ~16 GB
PCIe 5.0 x16 effective: ~25-30 GB/s
Transfer time: ~0.5-0.6s
```

This is a one-time startup cost. Once weights are in GDDR7, all subsequent kernel execution is at full 1.79 TB/s. The 96 GB of GDDR7 comfortably holds Qwen3-8B with room for KV caches and activations.

### 5. No Page Migration / NUMA Effects

On discrete GPUs, memory is either on the GPU or not. There are no page migration heuristics, no NUMA effects, no gradual warming of page placement. Once `model.to("cuda")` completes, everything is at full speed immediately.

**Implication:** Benchmarking is more straightforward — less warmup needed for memory subsystem (still need kernel cache warmup for instruction caches though).

## Impact on RMSNorm Kernel Design

### What Benefits from High Bandwidth

The RTX 6000 Pro's 1.79 TB/s means RMSNorm kernels complete faster in absolute terms, but the kernel is still bandwidth-bound (RMSNorm is ~6 FLOP/byte, far below the compute-memory crossover):

```
For [1, 2048, 4096] BF16:
  Total data: ~32 MB (read input + weight, write output)

  RTX 6000 Pro: 32 MB / 1790 GB/s = 0.018 ms theoretical minimum
  GB10:         32 MB / 273 GB/s  = 0.117 ms theoretical minimum

  ~6.5x faster theoretical floor on RTX 6000 Pro
```

Achieving 30-40% of peak bandwidth is realistic with vectorized loads.

### Why 1024 Threads/Block (vs 512 on GB10)

With 188 SMs and abundant bandwidth, the RTX 6000 Pro can use larger blocks:

- **1024 threads/block**: 2048 vec elements / 1024 threads = 2 vec elements/thread. 2 blocks/SM → 2048 threads/SM (100% occupancy).
- The higher bandwidth means latency hiding is less critical — 2 blocks/SM is sufficient because memory stalls are shorter (GDDR7 latency < LPDDR5X).
- Larger blocks also mean fewer blocks total → less launch overhead for the scheduler.

### Why Unroll 4 (vs 8 on GB10)

GDDR7's lower latency means fewer in-flight memory requests are needed to saturate bandwidth:
- The memory controller can achieve high throughput with 4 outstanding loads per thread
- Deeper unrolling would increase register pressure without proportional benefit
- The compiler may over-unroll with higher hints, spilling registers to local memory

### L2 Cache for Weight Broadcast

With 128 MB L2, the weight tensor (~8 KB for BF16 dim=4096) is cached so aggressively that it's effectively free after the first row reads it. Even with 188 SMs issuing concurrent weight reads, the L2 can serve them all without any GDDR7 traffic.

For a batch of [4, 2048, 4096]: 8192 rows all reading the same 8 KB weight — the weight is read from GDDR7 once and served from L2 for the remaining 8191 rows.

## When Discrete Memory Architecture Matters More

### Large Model Inference

The RTX 6000 Pro's 96 GB allows running larger models without quantization:
- Qwen3-8B BF16: ~16 GB → fits easily
- Qwen3-30B BF16: ~60 GB → fits
- Qwen3-72B BF16: ~144 GB → does NOT fit (need quantization or multi-GPU)

The GB10's 128 GB advantage for very large models is offset by its lower bandwidth — a 70B model runs on GB10 but 6.5x slower per token on bandwidth-bound operations.

### Multi-Stream Execution

The exclusive bandwidth enables running multiple CUDA streams concurrently without interference:
```python
# Overlap RMSNorm for different layers on different streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    out1 = rmsnorm(x1, w1)
with torch.cuda.stream(stream2):
    out2 = rmsnorm(x2, w2)
```

On GB10, multi-stream execution would compete for the same 273 GB/s shared pipe. On RTX 6000 Pro, the 1.79 TB/s has enough headroom for concurrent kernels.

### torch.compile and CUDA Graphs

The RTX 6000 Pro's high bandwidth and no-contention memory make it an ideal target for `torch.compile` and CUDA graphs, which eliminate Python/CPU overhead between kernels:

```python
# CUDA graphs eliminate per-kernel launch overhead
# Particularly effective on RTX 6000 Pro where kernel execution is fast
# and launch overhead becomes proportionally larger
model = torch.compile(model, mode="reduce-overhead")
```

## Summary

| Concern | Impact on RMSNorm Kernel | Notes |
|---------|--------------------------|-------|
| Exclusive bandwidth | Positive — consistent 1.79 TB/s | No CPU contention |
| GDDR7 latency | Positive — lower than LPDDR5X | Standard unroll (4) sufficient |
| 128 MB L2 cache | Positive — weights + inputs cached | Cross-kernel reuse for activations |
| PCIe transfer cost | None at runtime | One-time ~0.5s model load |
| Page migration | N/A | No unified memory effects |
| Kernel code differences from GB10 | `MAX_THREADS=1024`, `unroll 4` | Minor tuning, same algorithm |
