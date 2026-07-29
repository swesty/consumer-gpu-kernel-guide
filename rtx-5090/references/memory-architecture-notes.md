# RTX 5090 Blackwell: Memory Architecture and Kernel Implications

Notes on how the RTX 5090's discrete GDDR7 memory architecture affects CUDA kernel design, with specific focus on RMSNorm for Qwen3-8B.

## Architecture Overview

The RTX 5090 is a consumer discrete GPU based on the GB202 die (same die as RTX 6000 Pro, with fewer active SMs). The GPU has its own dedicated 32 GB of GDDR7 memory connected via a 512-bit interface, completely separate from the host system's DDR5 memory. Data moves between CPU and GPU over PCIe Gen 5.

```
┌─────────────────────┐         PCIe 5.0         ┌──────────────────────┐
│      Host CPU       │◄───── ~64 GB/s BD ──────►│  RTX 5090 (575W)     │
│  (DDR5 system RAM)  │                           │  GB202 Blackwell     │
└─────────────────────┘                           │                      │
                                                  │  170 SMs             │
                                                  │  21,760 CUDA cores   │
                                                  │                      │
                                                  │  ┌──────────────┐    │
                                                  │  │  32 GB GDDR7 │    │
                                                  │  │  512-bit bus  │    │
                                                  │  │  1.79 TB/s    │    │
                                                  │  └──────────────┘    │
                                                  │                      │
                                                  │  96 MB L2 cache      │
                                                  └──────────────────────┘
```

### Comparison with Other Blackwell GPUs

| | RTX 5090 | RTX 6000 Pro Blackwell | GB10 (DGX Spark) |
|---|---|---|---|
| Memory type | GDDR7 (dedicated) | GDDR7 (dedicated, ECC) | LPDDR5X (unified with CPU) |
| Memory size | 32 GB | 96 GB | 128 GB |
| GPU bandwidth | 1,792 GB/s (GPU exclusive) | 1,790 GB/s (GPU exclusive) | 273 GB/s (shared with CPU) |
| CPU-GPU transfer | PCIe 5.0 copy (~64 GB/s) | PCIe 5.0 copy (~64 GB/s) | Zero-copy (NVLink-C2C) |
| Memory contention | None — GPU has exclusive bandwidth | None — GPU has exclusive bandwidth | CPU and GPU compete |
| L2 cache | 96 MB | 128 MB | 24 MB |
| Bandwidth ratio vs GB10 | 6.6x higher | 6.6x higher | 1x (baseline) |

### RTX 5090 vs RTX 6000 Pro: Key Differences

The RTX 5090 and RTX 6000 Pro share the same GB202 die and sm_120 compute capability, but differ in important ways:

| Aspect | RTX 5090 | RTX 6000 Pro | Impact |
|--------|----------|-------------|--------|
| **VRAM** | 32 GB | 96 GB | **Major** — limits model size, KV cache, batch size |
| **SMs** | 170 | 188 | Minor — ~10% fewer SMs, grid sizing changes |
| **L2 Cache** | 96 MB | 128 MB | Minor — 96 MB still very large |
| **ECC** | No | Yes | Matters for HPC/scientific, not for inference |
| **NVLink** | No | No | Both use PCIe only |
| **TDP** | 575W | 600W | Comparable |
| **Positioning** | Consumer flagship | Workstation | Price, driver certification, support |

**The 32 GB VRAM constraint is the single most important architectural difference for LLM inference.**

## What the Discrete Memory Architecture Means for Kernels

### 1. Exclusive Bandwidth — No Contention

The 1.79 TB/s is available exclusively to the GPU. Unlike the GB10, where CPU memory traffic reduces GPU bandwidth, the RTX 5090's GDDR7 bandwidth is fully dedicated to GPU workloads regardless of what the CPU is doing.

**Implication:** No need to worry about CPU-side memory traffic during kernel execution. The kernel can rely on consistent bandwidth.

### 2. Lower Memory Latency than LPDDR5X

GDDR7 has lower access latency than LPDDR5X:

| Memory Type | Typical Latency | Bandwidth |
|-------------|----------------|-----------|
| GDDR7 (RTX 5090) | ~100-120 ns | 1,792 GB/s |
| GDDR7 (RTX 6000 Pro) | ~100-120 ns | 1,790 GB/s |
| LPDDR5X (GB10) | ~150-200 ns | 273 GB/s |
| HBM3 (H100) | ~80-100 ns | 3,350 GB/s |

**Implication:** Standard unroll depth (`#pragma unroll 4`) is sufficient. The memory controller can saturate bandwidth with fewer in-flight requests compared to LPDDR5X.

### 3. L2 Cache (96 MB) — Large but Smaller than RTX 6000 Pro

The 96 MB L2 cache is a significant asset, though 32 MB smaller than RTX 6000 Pro's 128 MB:

**What fits in L2 on RTX 5090:**
| Data | Size (Qwen3-8B, BF16) | Fits in 96 MB L2? | Fits in 128 MB L2? (6000 Pro) |
|------|------------------------|---------------------|-------------------------------|
| All RMSNorm weights | ~1.2 MB | Yes (trivially) | Yes |
| Single layer's full weights | ~150 MB | No | No |
| RMSNorm input (batch=1, seq=2048, dim=4096) | 16 MB | Yes | Yes |
| RMSNorm input (batch=4, seq=2048, dim=4096) | 64 MB | Yes | Yes |
| RMSNorm input (batch=4, seq=4096, dim=4096) | 128 MB | **No** | Yes |
| KV cache (seq=2048, all heads) | ~2.4 GB | No | No |
| KV cache (seq=256, all heads) | ~300 MB | No | No |

The 96 MB L2 cache is large enough to hold the entire RMSNorm input for most practical batch/sequence combinations. The main difference from RTX 6000 Pro is at the edge: working sets between 96-128 MB that fit on RTX 6000 Pro but spill on RTX 5090. This affects large-batch, long-sequence scenarios.

**Key implications:**
- **Weight data**: Permanently hot in L2 across all rows (same as RTX 6000 Pro)
- **Input/output data**: For moderate batches, the normalized output may still be in L2 when the next kernel reads it
- **Cross-kernel reuse**: Sequential kernels within a transformer layer can benefit from L2 residency of intermediate activations
- **Edge cases**: Batch=4, seq=4096+ scenarios may see slightly more L2 misses than on RTX 6000 Pro

### 4. PCIe Transfer Cost

Unlike the GB10's zero-copy unified memory, loading Qwen3-8B onto the RTX 5090 requires an explicit PCIe transfer:

```
Qwen3-8B in BF16: ~16 GB
PCIe 5.0 x16 effective: ~25-30 GB/s
Transfer time: ~0.5-0.6s
```

This is a one-time startup cost. Once weights are in GDDR7, all subsequent kernel execution is at full 1.79 TB/s. The 32 GB of GDDR7 holds Qwen3-8B with room for KV caches and activations, but larger models may not fit.

### 5. No Page Migration / NUMA Effects

On discrete GPUs, memory is either on the GPU or not. There are no page migration heuristics, no NUMA effects, no gradual warming of page placement. Once `model.to("cuda")` completes, everything is at full speed immediately.

**Implication:** Benchmarking is more straightforward — less warmup needed for memory subsystem (still need kernel cache warmup for instruction caches though).

## Impact on RMSNorm Kernel Design

### What Benefits from High Bandwidth

The RTX 5090's 1.79 TB/s means RMSNorm kernels complete faster in absolute terms, but the kernel is still bandwidth-bound (RMSNorm is ~6 FLOP/byte, far below the compute-memory crossover):

```
For [1, 2048, 4096] BF16:
  Total data: ~32 MB (read input + weight, write output)

  RTX 5090:   32 MB / 1792 GB/s = 0.018 ms theoretical minimum
  RTX 6000:   32 MB / 1790 GB/s = 0.018 ms theoretical minimum (essentially identical)
  GB10:       32 MB / 273 GB/s  = 0.117 ms theoretical minimum

  ~6.5x faster theoretical floor on RTX 5090 vs GB10
  ~1.0x vs RTX 6000 Pro (same bandwidth class)
```

Achieving 30-40% of peak bandwidth is realistic with vectorized loads.

### Why 1024 Threads/Block (vs 512 on GB10)

With 170 SMs and abundant bandwidth, the RTX 5090 can use larger blocks:

- **1024 threads/block**: 2048 vec elements / 1024 threads = 2 vec elements/thread. 2 blocks/SM -> 2048 threads/SM (100% occupancy).
- The higher bandwidth means latency hiding is less critical — 2 blocks/SM is sufficient because memory stalls are shorter (GDDR7 latency < LPDDR5X).
- Larger blocks also mean fewer blocks total -> less launch overhead for the scheduler.

### Why Unroll 4 (vs 8 on GB10)

GDDR7's lower latency means fewer in-flight memory requests are needed to saturate bandwidth:
- The memory controller can achieve high throughput with 4 outstanding loads per thread
- Deeper unrolling would increase register pressure without proportional benefit
- The compiler may over-unroll with higher hints, spilling registers to local memory

### L2 Cache for Weight Broadcast

With 96 MB L2, the weight tensor (~8 KB for BF16 dim=4096) is cached so aggressively that it's effectively free after the first row reads it. Even with 170 SMs issuing concurrent weight reads, the L2 can serve them all without any GDDR7 traffic.

For a batch of [4, 2048, 4096]: 8192 rows all reading the same 8 KB weight — the weight is read from GDDR7 once and served from L2 for the remaining 8191 rows.

## VRAM Management: The 32 GB Constraint

### Model Fit Analysis

The 32 GB VRAM is the RTX 5090's primary constraint for LLM inference. Here's what fits:

| Scenario | Model Weights | KV Cache (est.) | Activations | Total | Fits? |
|----------|--------------|-----------------|-------------|-------|-------|
| Qwen3-8B BF16, short ctx | 16 GB | 1 GB | 2 GB | ~19 GB | Yes |
| Qwen3-8B BF16, long ctx (32K) | 16 GB | 4.5 GB | 3 GB | ~23.5 GB | Yes |
| Qwen3-8B BF16, very long ctx (128K) | 16 GB | 18 GB | 4 GB | ~38 GB | **No** |
| Qwen3-8B FP8, very long ctx (128K) | 8 GB | 9 GB | 3 GB | ~20 GB | Yes |
| Qwen3-14B FP8 | 14 GB | 2 GB | 3 GB | ~19 GB | Yes |
| Qwen3-30B INT4 | 15 GB | 3 GB | 3 GB | ~21 GB | Yes |

**Strategies for maximizing VRAM utilization:**
1. **Quantized weights** (FP8, INT4) — reduce model footprint
2. **FP8 KV cache** — halves KV cache size vs BF16
3. **Paged attention** (vLLM-style) — eliminates wasted KV cache allocation
4. **Sliding window attention** — caps KV cache growth for long sequences

### Comparison with RTX 6000 Pro (96 GB)

The RTX 6000 Pro's 96 GB means:
- Qwen3-8B BF16 fits with 80 GB to spare — no VRAM concerns at all
- Qwen3-30B BF16 (~60 GB) fits comfortably
- Even Qwen3-72B BF16 (~144 GB) doesn't fit, but FP8 (~72 GB) does
- No need for quantization for models up to ~30B parameters

The RTX 5090 requires more careful VRAM planning, making quantization a standard part of the deployment strategy for anything beyond 8B parameters.

## When Discrete Memory Architecture Matters More

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

On GB10, multi-stream execution would compete for the same 273 GB/s shared pipe. On RTX 5090, the 1.79 TB/s has enough headroom for concurrent kernels.

### torch.compile and CUDA Graphs

The RTX 5090's high bandwidth and no-contention memory make it an ideal target for `torch.compile` and CUDA graphs, which eliminate Python/CPU overhead between kernels:

```python
# CUDA graphs eliminate per-kernel launch overhead
# Particularly effective on RTX 5090 where kernel execution is fast
# and launch overhead becomes proportionally larger
model = torch.compile(model, mode="reduce-overhead")
```

## Summary

| Concern | Impact on RMSNorm Kernel | Notes |
|---------|--------------------------|-------|
| Exclusive bandwidth | Positive — consistent 1.79 TB/s | No CPU contention |
| GDDR7 latency | Positive — lower than LPDDR5X | Standard unroll (4) sufficient |
| 96 MB L2 cache | Positive — weights + inputs cached | Slightly less headroom than 128 MB |
| 32 GB VRAM | **Constraining** — limits model size | Qwen3-8B fits; larger models need quantization |
| PCIe transfer cost | None at runtime | One-time ~0.5s model load |
| Page migration | N/A | No unified memory effects |
| Kernel code differences from GB10 | `MAX_THREADS=1024`, `unroll 4` | Minor tuning, same algorithm |
| Kernel code differences from RTX 6000 Pro | **None** | Same sm_120, same GDDR7, same tuning |
