# GB10 Unified Memory: Kernel Optimization Implications

Notes on how the GB10's 128 GB unified LPDDR5X memory architecture affects (and doesn't affect) CUDA kernel design, with specific focus on RMSNorm for Qwen3-8B.

## Architecture Overview

The GB10 is a System-on-Chip combining a 20-core Grace ARM CPU and a 48-SM Blackwell GPU on a single package. They share 128 GB of LPDDR5X memory over NVLink-C2C (600 GB/s bidirectional interconnect). There is no dedicated GPU VRAM — every byte of memory is physically the same LPDDR5X, accessed through a unified virtual address space.

```
┌──────────────────────────────────────────────────┐
│                  GB10 SoC (140W)                 │
│                                                  │
│  ┌──────────────┐    NVLink-C2C    ┌───────────┐ │
│  │  Grace CPU   │◄──600 GB/s BD──►│ Blackwell │ │
│  │  20 ARM v9.2 │                  │ GPU 48 SM │ │
│  │  cores       │                  │ 6144 CUDA │ │
│  └──────┬───────┘                  └─────┬─────┘ │
│         │                                │       │
│         └────────────┬───────────────────┘       │
│                      │                           │
│              ┌───────▼────────┐                  │
│              │ 128 GB LPDDR5X │                  │
│              │   273 GB/s     │                  │
│              │  (shared)      │                  │
│              └────────────────┘                  │
└──────────────────────────────────────────────────┘
```

This is fundamentally different from discrete GPUs:

| | Discrete GPU (e.g. RTX 6000 Pro) | GB10 (DGX Spark) |
|---|---|---|
| GPU memory | 96 GB dedicated GDDR7 | 128 GB shared LPDDR5X |
| GPU bandwidth | 1.79 TB/s (GPU exclusive) | 273 GB/s (shared with CPU) |
| CPU↔GPU transfer | PCIe 5.0 copy required | Zero-copy (same physical memory) |
| Memory contention | None (separate pools) | CPU and GPU compete for bandwidth |

## What Doesn't Change at the Kernel Level

For a bandwidth-bound kernel like RMSNorm, the GPU cores see memory through the same cache hierarchy regardless of the backing memory type. These kernel patterns are identical:

- **Coalesced vectorized loads** (`__nv_bfloat162`, `__half2`, `float4`) — still the right approach
- **Warp shuffle reductions** (`__shfl_xor_sync`) — register-level, memory-agnostic
- **Shared memory for block reductions** — on-chip SRAM, unaffected
- **Thread/block configuration** — determined by SM count and occupancy, not memory type

The RMSNorm kernel code itself needs zero changes for unified vs. dedicated memory.

## What Does Matter

### 1. Shared Bandwidth — CPU Contention

The 273 GB/s is not exclusively available to the GPU. During `model.generate()`, the CPU is simultaneously:

- Running Python interpreter and tokenizer
- Managing KV cache allocations
- Performing sampling (top-k, top-p) on CPU tensors
- Executing PyTorch dispatcher overhead

Each of these generates LPDDR5X traffic that competes with GPU kernel memory accesses. Effective GPU bandwidth may drop below 273 GB/s during active inference.

**Mitigation (application-level, not kernel-level):**
- Use `torch.inference_mode()` to minimize autograd bookkeeping
- Avoid unnecessary tensor copies in the Python hot path
- Prefer in-place operations where possible to reduce allocation pressure
- Profile with `nsys` to identify CPU memory traffic during GPU kernel execution

### 2. L2 Cache Is Your Real Fast Memory

With only 273 GB/s from LPDDR5X but 24 MB of L2 cache, the relative cost of a cache miss is much higher than on a high-bandwidth discrete GPU:

| | RTX 6000 Pro | GB10 |
|---|---|---|
| DRAM bandwidth | 1,790 GB/s | 273 GB/s |
| L2 cache | 128 MB | 24 MB |
| Cache miss penalty (relative) | 1x | ~6.5x worse |

For RMSNorm, this works in our favor:
- Weight tensor: 4096 × 2 bytes (BF16) = **8 KB** — fits trivially in L2
- Weight tensor stays hot across all row iterations within a batch
- QK norm weights: 128 × 2 bytes = **256 bytes** — essentially free

Where L2 pressure *would* matter:
- Fused kernels with large intermediate buffers
- Attention kernels touching full KV caches (Qwen3-8B at seq_len=32k ≈ 2 GB KV cache, far exceeds 24 MB L2)
- Multiple concurrent kernels evicting each other's cached data

### 3. Zero-Copy Model Loading

On a discrete GPU, loading Qwen3-8B in BF16 (~16 GB) requires:
1. CPU reads weights from disk into host memory
2. Framework copies weights to GPU memory over PCIe (~16 GB at ~25 GB/s ≈ 0.6s)

On GB10, step 2 is essentially free — the weights are already in the memory the GPU can access. The `device_map="cuda"` call sets up page table mappings, not physical data movement.

This doesn't affect kernel performance, but it significantly improves model load time and eliminates the "double memory" problem (no need for host + device copies simultaneously).

### 4. Page Placement and NUMA Effects

While the memory is unified, the NVLink-C2C interconnect means pages can be "closer" to either the CPU or GPU. CUDA's page migration engine handles this automatically — after a few kernel invocations, frequently-accessed pages migrate to the GPU side.

For long-running inference this is invisible. For benchmarking, it means:
- **Always do warmup runs** before timing (pages need to migrate)
- First invocation may be slower due to page faults and migration
- Subsequent invocations will be at steady-state bandwidth

You can also give explicit hints:
```python
# Framework-level hint (not commonly needed)
# torch.cuda.mem_advise(tensor, torch.cuda.MemAdvise.SET_PREFERRED_LOCATION, device)
```

### 5. No Benefit from Pinned Memory / Async Transfers

On discrete GPUs, pinned (page-locked) host memory enables async CPU↔GPU transfers overlapped with kernel execution. On GB10, this concept is irrelevant — there's nothing to transfer. `cuda.memcpy` between "host" and "device" addresses that map to the same physical LPDDR5X is essentially a no-op or a page table update.

## Impact Assessment for RMSNorm

| Concern | Impact on RMSNorm Kernel | Impact on System/Framework |
|---------|--------------------------|---------------------------|
| Shared bandwidth (CPU contention) | None (kernel code unchanged) | Moderate — minimize CPU work during inference |
| L2 cache pressure | Negligible (weights are 8 KB) | High for larger kernels |
| Page placement / NUMA | None | Minor — auto-migrates after warmup |
| Zero-copy model loading | None | Saves ~0.6s on Qwen3-8B load |
| Kernel code changes needed | **None** | — |

## Where Unified Memory Would Change Kernel Design

For a different, more complex kernel (not RMSNorm), unified memory could motivate:

1. **Persistent kernels** that stay resident and consume a work queue — avoids repeated launch overhead and keeps L2 warm. More valuable when DRAM bandwidth is limited.

2. **Explicit L2 residency controls** (`cudaAccessPropertyPersisting`) to pin hot data in L2 across kernel launches. Useful for KV cache in attention.

3. **CPU-GPU cooperative kernels** where the CPU preprocesses data that the GPU immediately consumes without an explicit transfer. Possible with unified memory, impossible with discrete.

4. **Reduced kernel fusion pressure** — on a discrete GPU you fuse aggressively to minimize DRAM round-trips. On GB10, the calculus is similar (bandwidth is scarce), but the shared L2 between sequential kernels may reduce the fusion benefit slightly if intermediate data fits in cache.

## Multi-Node (Spark Stacking): Memory Implications

When two DGX Sparks are connected ("Spark Stacking"), each node retains its own 128 GB LPDDR5X. There is no shared address space across nodes — inter-node data movement happens via NCCL collectives over ConnectX-7 Ethernet/RoCE at ~25 GB/s.

### Unified Memory Advantage for NCCL

The unified memory architecture provides an indirect benefit for multi-node communication:

```
Discrete GPU (e.g. RTX 6000 Pro):
  NIC receives data → Host DDR5 → PCIe copy → GPU GDDR7
  (Two hops, requires nvidia-peermem for GPUDirect RDMA)

DGX Spark (GB10):
  NIC receives data → LPDDR5X ← GPU already has access
  (One hop — NIC and GPU share the same physical memory)
```

The ConnectX-7 NIC writes incoming RDMA data directly into LPDDR5X, which the GPU can immediately read. This is effectively GPUDirect-like behavior without `nvidia-peermem` (which is not supported on DGX Spark). No staging buffer, no extra copy.

### What This Means for Kernels

The kernel itself is still unchanged. But the data flow around the kernel differs in multi-node:

1. **NCCL AllReduce buffers sit in unified LPDDR5X** — the GPU can read/write them at 273 GB/s, and the NIC can DMA to/from them at ~25 GB/s. No explicit `cudaMemcpy` needed.

2. **Communication competes for LPDDR5X bandwidth** — when NCCL is actively transferring data, the NIC's ~25 GB/s traffic adds to the shared memory pressure. This is on top of CPU contention.

3. **Buffer placement**: NCCL communication buffers allocated via `cudaHostAlloc` are already GPU-visible. The `cudaMalloc` vs `cudaHostAlloc` distinction is less meaningful on GB10 since both point to the same LPDDR5X.

### Bandwidth Budget (Two-Node Inference)

During tensor-parallel inference with active NCCL communication:

```
Available LPDDR5X bandwidth: 273 GB/s (per node)

Consumers:
  - GPU kernel execution:   ~100-150 GB/s (realistic kernel throughput)
  - CPU framework overhead:  ~5-10 GB/s
  - NCCL NIC traffic:        ~25 GB/s (ConnectX-7)

Total demand: ~130-185 GB/s → fits within 273 GB/s
```

In practice, NCCL transfers and kernel execution are pipelined (not fully overlapping), so the bandwidth contention is manageable. The real bottleneck is the 25 GB/s inter-node link latency, not bandwidth contention within a node.

## Bottom Line

The unified memory architecture is transparent to individual CUDA kernels — their code needs zero changes versus discrete GPU implementations. The hardware adaptations that matter are SM count, memory latency, and bandwidth (512 threads/block for occupancy, `#pragma unroll 8` for LPDDR5X latency hiding), not the unified memory model itself. Unified memory effects are at the system and framework level: zero-copy model loading, shared bandwidth contention with the CPU, and page migration during warmup. In multi-node (Spark Stacking) configurations, unified memory provides a natural advantage for NCCL communication by eliminating the host-to-device staging copy that discrete GPUs require, but individual kernels remain unchanged.
