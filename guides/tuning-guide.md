# GPU Kernel Tuning Guide
### From First Principles to Per-GPU Optimization

A comprehensive guide to writing fast CUDA kernels for LLM inference, covering GPU architecture fundamentals, the four tuning knobs that matter, per-GPU deep dives for four GPUs, and a real-world case study of a cooperative megakernel.

**Target GPUs:** RTX 3090 (Ampere), RTX 5090 (Blackwell), RTX 6000 Pro (Blackwell), GB10/DGX Spark (Blackwell)
**Target Model:** Qwen3-8B (hidden_size=4096, BF16)
**Running Example:** RMSNorm kernel (`y = x / sqrt(mean(x^2) + eps) * weight`)

---

## Table of Contents

- [Part 1: GPU Fundamentals](#part-1-gpu-fundamentals)
- [Part 2: The Four Knobs That Matter](#part-2-the-four-knobs-that-matter)
- [Part 3: Per-GPU Deep Dives](#part-3-per-gpu-deep-dives)
- [Part 4: MegaQwen Case Study](#part-4-megaqwen-case-study)
- [Part 5: Model Size & Architecture Decision Framework](#part-5-model-size--architecture-decision-framework)
- [Part 6: Practical Tooling](#part-6-practical-tooling)

---

## Part 1: GPU Fundamentals

### What a GPU Actually Is

A CPU is designed to execute one thread of instructions as fast as possible. It has a huge cache, complex branch prediction, speculative execution, and out-of-order pipelines. It is optimized for latency: getting the answer to one question as quickly as possible.

A GPU is the opposite. It is designed to execute thousands of threads simultaneously, each doing a little work. It sacrifices single-thread performance for massive parallelism. The individual threads are slow and simple, but there are so many of them that the aggregate throughput dwarfs what a CPU can do.

```
CPU: 20 fast lanes, each doing complex work
    ┌──────────────────────────────────────────────────────────┐
    │  Core 0: [======fetch=====][==decode==][===execute====]  │
    │  Core 1: [======fetch=====][==decode==][===execute====]  │
    │  ...20 cores, deep pipelines, big caches                 │
    └──────────────────────────────────────────────────────────┘

GPU: 10,000+ simple lanes, all doing the same thing
    ┌──────────────────────────────────────────────────────────┐
    │  Thread    0: [ld][mul][add]                             │
    │  Thread    1: [ld][mul][add]                             │
    │  Thread    2: [ld][mul][add]                             │
    │  ...                                                     │
    │  Thread 9999: [ld][mul][add]                             │
    └──────────────────────────────────────────────────────────┘
```

This design exists because many computational workloads — graphics rendering, matrix multiplication, normalization over tensors — involve applying the same operation to thousands or millions of data elements. A GPU can process all of them in parallel.

### SIMT: Single Instruction, Multiple Threads

NVIDIA GPUs use the SIMT (Single Instruction, Multiple Threads) execution model. Groups of 32 threads, called a **warp**, execute the same instruction at the same time. Every thread in a warp is at the same program counter. When you write `sum_sq += v * v`, all 32 threads in the warp execute that multiply-add simultaneously — on different data.

This has a critical consequence: if threads in a warp take different branches of an `if` statement, the hardware must execute both paths serially, masking off the threads that shouldn't participate in each path. This is called **warp divergence**, and it halves (or worse) your throughput. For reduction kernels like RMSNorm, this is rarely an issue because all threads follow the same control flow.

### The Execution Hierarchy: Threads, Warps, Blocks, Grids

CUDA organizes parallel work into a hierarchy:

```
Grid (the entire kernel launch)
├── Block 0                          ← runs on one SM
│   ├── Warp 0  (threads 0-31)      ← always execute in lockstep
│   ├── Warp 1  (threads 32-63)
│   ├── ...
│   └── Warp 15 (threads 480-511)
├── Block 1                          ← runs on another SM (or the same)
│   ├── Warp 0
│   └── ...
├── Block 2
│   └── ...
└── Block N-1
```

- **Thread**: The smallest unit of execution. Has its own registers and program counter (within a warp). Each thread processes one or more data elements.
- **Warp**: 32 threads that execute in lockstep. This is the true unit of scheduling — the hardware doesn't schedule individual threads; it schedules warps.
- **Block** (also called a CTA — Cooperative Thread Array): A group of threads (32 to 1024) that can cooperate via shared memory and `__syncthreads()`. All threads in a block run on the same SM.
- **Grid**: The collection of all blocks in a kernel launch. Blocks in a grid are independent — they cannot synchronize with each other (with rare exceptions like cooperative groups).

For RMSNorm, each block processes one row of the input tensor:
```
Input: [num_rows, hidden_size]       Grid: num_rows blocks
       ─────────────────             Each block: 512 threads
       row 0: [x0, x1, ..., x4095]  → Block 0
       row 1: [x0, x1, ..., x4095]  → Block 1
       ...
       row N: [x0, x1, ..., x4095]  → Block N
```

### Streaming Multiprocessors (SMs)

The SM is the fundamental compute unit of a GPU. Each SM contains:

```
┌──────────────────────────────────────────────────────┐
│                  Streaming Multiprocessor             │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Warp Schedulers (4 per SM on Blackwell)       │  │
│  │  Select which warps to execute each cycle      │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ INT32 units  │  │ FP32 units   │  128 CUDA cores  │
│  └──────────────┘  └──────────────┘                  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ Tensor Cores │  │  Load/Store  │                  │
│  │ (matrix ops) │  │    units     │                  │
│  └──────────────┘  └──────────────┘                  │
│                                                      │
│  ┌──────────────────────────────────────────────────┐│
│  │  Register File: 64K x 32-bit registers          ││
│  │  (partitioned among all resident threads)        ││
│  └──────────────────────────────────────────────────┘│
│                                                      │
│  ┌──────────────────────────────────────────────────┐│
│  │  Shared Memory / L1 Cache: 100-128 KB           ││
│  │  (shared among all threads in a block)           ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

The number of SMs determines how much work the GPU can do in parallel:

| GPU | SMs | What this means |
|-----|-----|-----------------|
| RTX 3090 | 82 | Need >= 82 blocks to use all SMs |
| GB10 | 48 | Need >= 48 blocks to use all SMs |
| RTX 5090 | 170 | Need >= 170 blocks to use all SMs |
| RTX 6000 Pro | 188 | Need >= 188 blocks to use all SMs |

The warp scheduler is the key to latency hiding. When a warp issues a memory load and stalls waiting for data, the scheduler switches to a different warp that is ready to execute. This is why occupancy matters — more resident warps means more warps available to fill the gap while others wait on memory.

### The Memory Hierarchy

A GPU has multiple levels of memory, each trading capacity for speed:

```
                    Bandwidth        Latency    Size
                    ─────────        ───────    ────
Registers           ~100 TB/s        0 cycles   64K x 32-bit per SM
        │                                       (~256 KB per SM)
        ▼
Shared Memory       ~15-40 TB/s      ~20 cyc    100-128 KB per SM
(on-chip SRAM)                                   (shared per block)
        │
        ▼
L1 Cache            (configurable    ~30 cyc    Combined with
                     with shared)               shared memory
        │
        ▼
L2 Cache            ~3-12 TB/s       ~200 cyc   6-128 MB
(on-chip, global)                               (shared across all SMs)
        │
        ▼
DRAM                273-1790 GB/s    100-200    24-128 GB
(off-chip)                           cycles     (the "memory bandwidth"
                                                 everyone quotes)
        │
        ▼
PCIe / Host         32-64 GB/s       ~10K cyc   System RAM
```

**Registers** are the fastest. Each thread can use up to 255 32-bit registers. The register file is partitioned among all resident threads on an SM, so using more registers per thread means fewer threads can be resident (reducing occupancy).

**Shared memory** is an on-chip SRAM scratchpad visible to all threads within a block. It's fast (~20 cycles) but small (100-128 KB per SM). It has 32 banks — if multiple threads access the same bank simultaneously (a "bank conflict"), accesses are serialized.

**L2 cache** is a global cache shared by all SMs. Its size varies dramatically across GPUs and fundamentally changes optimization strategy:

| GPU | L2 Cache | Implication |
|-----|----------|-------------|
| RTX 3090 | 6 MB | Activations don't fit. Assume DRAM round-trip. |
| GB10 | 24 MB | Weight caching works. Most activations spill. |
| RTX 5090 | 96 MB | Moderate activations fit. Inter-kernel reuse possible. |
| RTX 6000 Pro | 128 MB | Large working sets cached. Best inter-kernel reuse. |

**DRAM** is the main GPU memory — GDDR6X, GDDR7, or LPDDR5X depending on the GPU. Its bandwidth (273-1790 GB/s) is the number that determines performance for memory-bound kernels.

### Occupancy: What It Means and Why It Matters

Occupancy is the ratio of active warps on an SM to the maximum number of warps the SM can hold. It's a measure of how much latency-hiding potential you have.

```
Occupancy = Active Warps per SM / Max Warps per SM

Example (RTX 3090, 512 threads/block):
  512 threads/block = 16 warps/block
  3 blocks fit per SM = 48 active warps
  Max warps per SM = 48 (sm_86)
  Occupancy = 48/48 = 100%

Example (RTX 3090, 1024 threads/block):
  1024 threads/block = 32 warps/block
  Only 1 block fits per SM (1536 thread limit, not 2048)
  32 active warps / 48 max = 66.7% occupancy
```

Three things limit occupancy:
1. **Thread count**: Max threads per SM (1536 for sm_86, 2048 for sm_100/sm_120)
2. **Register usage**: 65536 registers / (threads_per_block x registers_per_thread) = max blocks
3. **Shared memory**: Shared memory per SM / shared memory per block = max blocks

For memory-bound kernels, higher occupancy means more warps available to execute while others wait on memory. This is essential for hiding the 100-200 cycle DRAM latency.

### Memory-Bound vs Compute-Bound: The Roofline Model

Every kernel is either limited by how fast it can move data (memory-bound) or how fast it can do math (compute-bound). The **roofline model** tells you which:

```
Performance                    Compute
(GFLOPS)                       ceiling
    │                         ┌────────────
    │                        ╱
    │                       ╱
    │                      ╱  <-- Your kernel lives
    │                     ╱       somewhere on this line
    │        Memory      ╱
    │        ceiling     ╱
    │               ╱   ╱
    │              ╱   ╱
    │             ╱   ╱
    │            ╱   ╱
    │           ╱   ╱
    │          ╱   ╱
    │         ╱   ╱
    └────────╱───╱─────────────────────
             ▲       Arithmetic Intensity
             │       (FLOPs per byte)
          Crossover
          point
```

The **crossover point** = Peak FLOPS / Peak Bandwidth. Below this point, you're memory-bound; above it, you're compute-bound.

| GPU | Peak FP32 | Peak BW | Crossover | RMSNorm (~6 FLOP/byte) |
|-----|-----------|---------|-----------|------------------------|
| GB10 | ~10 TFLOPS | 273 GB/s | ~37 | Memory-bound |
| RTX 3090 | ~35.6 TFLOPS | 936 GB/s | ~38 | Memory-bound |
| RTX 5090 | ~90 TFLOPS | 1.79 TB/s | ~50 | Memory-bound |
| RTX 6000 Pro | ~100 TFLOPS | 1.79 TB/s | ~56 | Memory-bound |

RMSNorm performs ~6 FLOPs per byte of data moved — far below the crossover for any GPU. It is **deeply memory-bound on all four GPUs**. This means the optimization goal is to maximize memory bandwidth utilization, not compute throughput.

For comparison, GEMM (matrix multiplication) operates at ~100-1000 FLOP/byte and is compute-bound on all GPUs. Flash Attention operates at ~50-200 FLOP/byte and sits near the crossover.

### Coalescing and Vectorization

When a warp of 32 threads reads memory, the hardware combines those 32 reads into as few memory transactions as possible — if the addresses are contiguous. This is called **coalescing**.

```
GOOD: Coalesced access (threads read consecutive addresses)
  Thread 0: addr[0]    ┐
  Thread 1: addr[1]    │── One 128-byte transaction
  Thread 2: addr[2]    │   (32 threads x 4 bytes = 128 bytes)
  ...                   │
  Thread 31: addr[31]  ┘

BAD: Strided access (threads read with gaps)
  Thread 0: addr[0]     → transaction 1
  Thread 1: addr[128]   → transaction 2
  Thread 2: addr[256]   → transaction 3
  ...                      32 separate transactions!
```

Coalesced access moves data in one 128-byte transaction. Strided access can cause 32 separate transactions for the same amount of data — a 32x penalty.

**Vectorization** goes further: instead of loading one element per thread, you load 2 or 4 at once:

```cuda
// Scalar: 1 element per load (16-bit BF16)
__nv_bfloat16 v = input[i];

// Vectorized: 2 elements per 32-bit load
__nv_bfloat162 v = vec_input[i];  // loads input[2i] and input[2i+1]

// FP32 vectorized: 4 elements per 128-bit load
float4 v = vec_input[i];          // loads input[4i] through input[4i+3]
```

Vectorized loads reduce the number of load instructions and guarantee coalesced access. For memory-bound kernels, this is not optional — it's the difference between 20% and 40%+ bandwidth utilization.

### Warp Divergence and Reductions

When threads in a warp need to combine their results (e.g., summing all partial sums), they use **warp shuffle** instructions:

```
Warp of 32 threads, each holding a partial sum:
  Step 1: Each thread adds with neighbor 16 away  (16 adds in parallel)
  Step 2: Each thread adds with neighbor 8 away   (16 adds in parallel)
  Step 3: Each thread adds with neighbor 4 away   (16 adds in parallel)
  Step 4: Each thread adds with neighbor 2 away   (16 adds in parallel)
  Step 5: Each thread adds with neighbor 1 away   (16 adds in parallel)
  Result: Thread 0 holds the sum of all 32 values

  [a0] [a1] [a2] ... [a15] [a16] [a17] ... [a31]
    ↕                         ↕                     Step 1: XOR 16
  [a0+a16] ...              [a16+a0] ...
    ↕           ↕                                   Step 2: XOR 8
  ...
  Final: [total_sum] in all threads
```

This is implemented with `__shfl_xor_sync()` — a register-to-register operation within a warp with zero memory traffic. Five shuffle steps reduce 32 values to one.

For blocks larger than 32 threads, we use a two-level reduction:
1. Each warp reduces within itself (shuffle)
2. Warp leaders write to shared memory
3. First warp reads shared memory and does a final shuffle reduction

This is the `block_reduce_sum` pattern used in all four of our RMSNorm kernels.

### Kernel Launch Overhead

Every kernel launch has CPU-side overhead: the driver must set up the launch parameters, copy them to the GPU, and trigger execution. This takes approximately 5-10 microseconds.

For large kernels processing megabytes of data, this overhead is negligible. But for small kernels (single-token decode with a 4096-element vector), the kernel itself may execute in 10-20 microseconds — making launch overhead a significant fraction of total time.

Solutions:
- **CUDA Graphs**: Capture a sequence of launches once, replay with near-zero overhead
- **Kernel Fusion**: Combine multiple small kernels into one
- **`torch.compile`**: Automatically fuses operations and uses CUDA graphs

---

## Part 2: The Four Knobs That Matter

When tuning a CUDA kernel for a specific GPU, there are four parameters that determine most of the performance: block size, unroll depth, grid size, and vectorization width. The optimal values for each are determined by the GPU's hardware constraints.

### Knob 1: Block Size (Threads Per Block)

Block size determines how many threads cooperate within a block, and therefore how many blocks can co-reside on an SM. The goal is to achieve 100% occupancy — filling every warp slot on every SM.

The constraint is **max threads per SM**, which differs across architectures:

```
sm_86 (RTX 3090):   1,536 threads/SM (48 warps)
sm_100 (RTX 5090):  2,048 threads/SM (64 warps)
sm_100 (RTX 6000):  2,048 threads/SM (64 warps)
sm_120 (GB10):      2,048 threads/SM (64 warps)
```

For our RMSNorm kernel with hidden_size=4096 BF16 (2048 vectorized elements):

| Block Size | RTX 3090 (1536 max) | 5090/6000/GB10 (2048 max) |
|------------|---------------------|---------------------------|
| 256 | 6 blocks/SM = 1536 = 100% | 8 blocks/SM = 2048 = 100% |
| 512 | 3 blocks/SM = 1536 = **100%** | 4 blocks/SM = 2048 = **100%** |
| 1024 | 1 block/SM = 1024 = **66.7%** | 2 blocks/SM = 2048 = **100%** |

**RTX 3090** must use 512 (or 256) threads. Using 1024 threads wastes a third of the SM's capacity.

**RTX 5090 and RTX 6000 Pro** can use 1024 threads because 2 blocks of 1024 still fills the SM. Fewer, larger blocks mean less scheduling overhead.

**GB10** uses 512 threads despite supporting 2048 threads/SM. Why? LPDDR5X has higher latency (~150-200 ns vs ~100-120 ns for GDDR7). More blocks per SM (4 vs 2) provides more warps to switch between when waiting on memory, improving latency hiding.

### Knob 2: Unroll Depth

`#pragma unroll N` tells the compiler to replicate the loop body N times, generating N independent load instructions before any dependent computation. This creates more in-flight memory requests, helping saturate memory bandwidth despite DRAM latency.

```cuda
// Without unroll: one load, wait for it, compute, next load
for (int i = tid; i < vec_hidden; i += stride) {
    val = vec_input[i];    // load (100-200 ns wait)
    sum += val.x * val.x;  // compute (depends on load)
}

// With #pragma unroll 4: four loads issued before computing
// The memory controller can pipeline all four requests
```

The optimal unroll depth is determined by memory latency:

| GPU | Memory Type | Latency | Unroll | Why |
|-----|-------------|---------|--------|-----|
| RTX 3090 | GDDR6X | ~100-150 ns | 4 | Moderate latency, 3 blocks/SM for hiding |
| RTX 5090 | GDDR7 | ~100-120 ns | 4 | Low latency, 2 blocks sufficient |
| RTX 6000 Pro | GDDR7 | ~100-120 ns | 4 | Low latency, 2 blocks sufficient |
| GB10 | LPDDR5X | ~150-200 ns | **8** | High latency needs more requests in flight |

The GB10's higher unroll depth compensates for LPDDR5X's sluggish response time. With 512 threads and hidden_size=4096: each thread processes 4 vectorized elements, so `#pragma unroll 8` fully unrolls the loop (the compiler generates all 4 iterations explicitly). The extra instruction-level parallelism keeps the memory controller busy.

### Knob 3: Grid Size

Grid size is the number of blocks launched. For RMSNorm, grid size = number of rows in the input tensor. The goal is to keep every SM busy:

```
Minimum useful grid size = number of SMs

RTX 3090:    >= 82 blocks   (82 SMs)
GB10:        >= 48 blocks   (48 SMs)
RTX 5090:    >= 170 blocks  (170 SMs)
RTX 6000 Pro: >= 188 blocks (188 SMs)
```

When grid size < SM count, some SMs sit idle:

```
Grid = 48 blocks on RTX 5090 (170 SMs):
  SMs 0-47:  busy
  SMs 48-169: idle (72% waste)
```

For sequence length=1 (single-token decode), all GPUs underutilize their SMs. The RTX 3090 and GB10 have fewer SMs, so they reach full utilization at shorter sequences — a small advantage for low-batch inference.

The grid size is usually dictated by the problem (number of rows), not the programmer. What you can control is whether to use one block per row (standard) or multiple blocks per row (for very short sequences where SM underutilization is severe).

### Knob 4: Vectorization Width

Vectorization is universal — always use the widest vector load that your data type supports:

| Data Type | Scalar Load | Vector Load | Width | Bytes/Load |
|-----------|-------------|-------------|-------|------------|
| BF16 | `__nv_bfloat16` (16-bit) | `__nv_bfloat162` (32-bit) | 2 elements | 4 bytes |
| FP16 | `__half` (16-bit) | `__half2` (32-bit) | 2 elements | 4 bytes |
| FP32 | `float` (32-bit) | `float4` (128-bit) | 4 elements | 16 bytes |

Vectorized loads reduce load instruction count and guarantee coalesced access. This is the same on all four GPUs — it is not a GPU-specific tuning parameter.

### Build Target: cuda-capabilities

The `cuda-capabilities` field in `build.toml` tells the compiler which GPU architecture to generate code for:

| GPU | cuda-capabilities | NVCC Flag | Min CUDA Toolkit |
|-----|-------------------|-----------|------------------|
| RTX 3090 | `"8.6"` | `-arch=sm_86` | 11.1+ |
| RTX 5090 | `"10.0"` | `-arch=sm_100` | 12.8+ |
| RTX 6000 Pro | `"10.0"` | `-arch=sm_100` | 12.8+ |
| GB10 | `"12.0"` | `-gencode=arch=compute_120,code=sm_120` | 12.8+ |

Targeting the wrong architecture means the GPU either can't run the code (too new) or runs it suboptimally (missing hardware features). Always match exactly.

### Summary: All Four Knobs by GPU

| Knob | RTX 3090 | RTX 5090 | RTX 6000 Pro | GB10 |
|------|----------|----------|--------------|------|
| **Block size** | 512 | 1024 | 1024 | 512 |
| **Unroll depth** | 4 | 4 | 4 | 8 |
| **Min grid size** | >= 82 | >= 170 | >= 188 | >= 48 |
| **Vectorization** | bf162/half2/float4 | bf162/half2/float4 | bf162/half2/float4 | bf162/half2/float4 |
| **cuda-capabilities** | `"8.6"` | `"10.0"` | `"10.0"` | `"12.0"` |

---

## Part 3: Per-GPU Deep Dives

### RTX 3090 (Ampere, sm_86)

#### Hardware Quick Reference

```
┌─────────────────────────────────────────┐
│        RTX 3090 Ampere (GA102)          │
├─────────────────────────────────────────┤
│ Compute:  sm_86, 82 SMs, 10496 cores   │
│ Memory:   24 GB GDDR6X, 936 GB/s       │
│ L2:       6 MB                          │
│ Shared:   100 KB/SM                     │
│ Threads:  1536/SM (48 warps) — NOT 2048 │
│ Tensor:   3rd gen (BF16, INT8, INT4)    │
│ TDP:      350W                          │
│ NVLink:   2-way (112.5 GB/s)           │
│ PCIe:     Gen 4 x16                     │
│ CUDA:     >= 11.1 required              │
└─────────────────────────────────────────┘
```

#### The 3090's Unique Constraint Profile

The RTX 3090 has **three constraints** that no other GPU in this set shares:

1. **1536 threads/SM limit.** This is the defining characteristic of sm_86 for kernel tuning. While all other GPUs support 2048 threads/SM, the 3090 caps at 1536. Using 1024 threads/block gives only 1 block/SM = 66.7% occupancy. You must use 512 (or 768, or 256).

2. **6 MB L2 cache.** The RTX 3090's L2 is 16x smaller than the RTX 5090's (96 MB) and 21x smaller than the RTX 6000 Pro's (128 MB). Activations for sequences longer than ~512 tokens don't fit. You cannot rely on inter-kernel L2 reuse — assume every kernel does a full DRAM round-trip.

3. **No FP8/FP4 support.** The 3rd-gen Tensor Cores support BF16, FP16, INT8, and INT4 — but not FP8 or FP4. Quantization options are limited to INT8 and INT4 (GPTQ/AWQ).

The 3090 also has a **unique advantage**: it's one of the last consumer GPUs with NVLink (2-way, 112.5 GB/s). Two 3090s connected via NVLink bridge get 3.5x the inter-GPU bandwidth of PCIe Gen 4, enabling more practical tensor parallelism than any PCIe-only setup.

#### Optimal Kernel Configuration

```cuda
// rtx-3090/qwen3_rmsnorm/kernel_src/rmsnorm.cu
constexpr int MAX_THREADS = 512;

// Vectorized BF16 loop
#pragma unroll 4
for (int i = tid; i < vec_hidden; i += stride) {
    __nv_bfloat162 v = vec_in[i];
    // ...
}
```

**Why 512 threads**: 3 blocks x 512 = 1536 threads/SM = 100% occupancy. Each thread processes 4 vectorized BF16 elements (2048 vec elements / 512 threads). Three blocks provide adequate warp-level latency hiding for GDDR6X's ~100-150 ns latency.

**Why unroll 4**: GDDR6X has moderate latency. With 3 resident blocks providing 48 warps for latency hiding, `unroll 4` generates enough in-flight memory requests without excessive register pressure.

#### Model Fit Analysis

| Model | BF16 Size | Fits? | Strategy |
|-------|-----------|-------|----------|
| Qwen3-8B | ~16 GB | Yes (8 GB free) | BF16, tight on KV cache |
| Qwen3-14B | ~28 GB | No | INT4 (~7 GB) |
| Qwen3-30B | ~60 GB | No | INT4 (~15 GB), tight |
| Qwen3-72B | ~144 GB | No | Does not fit |

With 24 GB, the 3090 fits Qwen3-8B BF16 with ~8 GB remaining for KV cache and activations. For larger models, INT8/INT4 quantization is mandatory (no FP8 option).

#### Optimization Priority Order

1. **Vectorize loads** — coalesced `__nv_bfloat162` access is the biggest single win
2. **Use 512 threads/block** — 100% occupancy on sm_86
3. **Fuse memory-bound kernels** — each DRAM round-trip at 936 GB/s takes ~2x longer than on Blackwell; fusion saves more absolute time
4. **Don't over-rely on L2** — 6 MB means most activations spill; use streaming loads (`__ldcs`) for non-reusable data
5. **Consider INT4 quantization** — for models >8B, this is the only way to fit

#### Code-Level Differences

The key code difference versus other GPUs is `MAX_THREADS = 512` and its impact on the launch helper:

```cuda
// RTX 3090: cap at 512 for sm_86 occupancy
constexpr int MAX_THREADS = 512;

static inline int compute_threads_vec2(int hidden_size) {
    int threads = min(hidden_size / 2, MAX_THREADS);  // capped at 512
    threads = max(threads, WARP_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return threads;
}
```

---

### RTX 5090 (Blackwell, sm_100)

#### Hardware Quick Reference

```
┌─────────────────────────────────────────┐
│       RTX 5090 Blackwell (GB202)        │
├─────────────────────────────────────────┤
│ Compute:  sm_100, 170 SMs, 21760 cores  │
│ Memory:   32 GB GDDR7, 1.79 TB/s       │
│ L2:       96 MB                         │
│ Shared:   128 KB/SM (up to 228 KB)      │
│ Threads:  2048/SM (64 warps)            │
│ Tensor:   5th gen (FP4/FP8/BF16)       │
│ TDP:      575W                          │
│ NVLink:   None                          │
│ PCIe:     Gen 5 x16                     │
│ CUDA:     >= 12.8 required              │
└─────────────────────────────────────────┘
```

#### The 5090's Unique Constraint Profile

The RTX 5090 is the **highest bandwidth consumer GPU** in this set. Its constraint profile is dominated by one thing: **32 GB VRAM**.

While the kernel execution hardware is nearly identical to the RTX 6000 Pro (same sm_100, same GDDR7, same 1.79 TB/s), the 32 GB capacity limits what models fit. The 5090 is the GPU where model fit strategy matters most — FP8/FP4 quantization (supported by 5th-gen Tensor Cores) is the key to fitting larger models.

The 5090 has 170 SMs — more than the 3090's 82 or the GB10's 48, but fewer than the RTX 6000 Pro's 188. Short sequences (< 170 rows) underutilize the GPU. For single-token decode, individual kernels are extremely fast (1.79 TB/s), making kernel launch overhead a proportionally larger fraction of total time.

#### Optimal Kernel Configuration

```cuda
// rtx-5090/qwen3_rmsnorm/kernel_src/rmsnorm.cu
constexpr int MAX_THREADS = 1024;

#pragma unroll 4
for (int i = tid; i < vec_hidden; i += stride) {
    __nv_bfloat162 v = vec_in[i];
    // ...
}
```

**Why 1024 threads**: 2 blocks x 1024 = 2048 threads/SM = 100% occupancy. GDDR7's low latency (~100-120 ns) means 2 blocks provide sufficient warps for hiding. Fewer, larger blocks reduce scheduling overhead.

**Why unroll 4**: GDDR7 latency is low. Two blocks of 1024 threads = 64 warps already provides excellent latency hiding without deep unrolling.

#### Model Fit Analysis

| Model | BF16 Size | Fits? | Strategy |
|-------|-----------|-------|----------|
| Qwen3-8B | ~16 GB | Yes (16 GB free) | BF16, comfortable |
| Qwen3-14B | ~28 GB | Tight (4 GB free) | FP8 (~14 GB) recommended |
| Qwen3-30B | ~60 GB | No | INT4 (~15 GB) |
| Qwen3-72B | ~144 GB | No | Does not fit |

The 5090 is the sweet spot for Qwen3-8B BF16. For Qwen3-14B, FP8 is the right strategy (5th-gen Tensor Cores make this efficient). For Qwen3-30B, INT4 is required.

#### Optimization Priority Order

1. **Vectorize loads** — still the foundation at 1.79 TB/s
2. **Use 1024 threads/block** — 100% occupancy on sm_100
3. **Exploit 96 MB L2** — moderate activations stay cached between kernels
4. **Reduce launch overhead** — kernels are fast; the gaps between them matter. Use CUDA graphs or `torch.compile`
5. **Use FP8/FP4 quantization** for models >8B — the 5th-gen Tensor Cores make this efficient

---

### RTX 6000 Pro (Blackwell, sm_100)

#### Hardware Quick Reference

```
┌─────────────────────────────────────────┐
│     RTX 6000 Pro Blackwell (GB202)      │
├─────────────────────────────────────────┤
│ Compute:  sm_100, 188 SMs, 24064 cores  │
│ Memory:   96 GB GDDR7, 1.79 TB/s       │
│ L2:       128 MB                        │
│ Shared:   128 KB/SM (up to 228 KB)      │
│ Threads:  2048/SM (64 warps)            │
│ Tensor:   5th gen (FP4/FP8/BF16)       │
│ TDP:      600W                          │
│ NVLink:   None                          │
│ PCIe:     Gen 5 x16                     │
│ ECC:      Yes                           │
│ CUDA:     >= 12.8 required              │
└─────────────────────────────────────────┘
```

#### The 6000 Pro's Unique Constraint Profile

The RTX 6000 Pro is the **least constrained** GPU in this set. It has the most SMs (188), the largest VRAM (96 GB), the largest L2 (128 MB), ECC memory, and the same 1.79 TB/s bandwidth as the RTX 5090.

Its constraints are:
- **Power**: 600W TDP — the highest in the set. Requires workstation-class cooling.
- **Cost**: Workstation-class pricing.
- **No NVLink**: Multi-GPU scaling is limited to PCIe Gen 5.

The 96 GB VRAM means most models fit without quantization — Qwen3-8B, 14B, and 30B all run in BF16. The 128 MB L2 cache means even large activation tensors ([4, 4096, 4096] BF16 = 128 MB) can stay cached between kernel launches.

#### Optimal Kernel Configuration

```cuda
// rtx-6000-pro/qwen3_rmsnorm/kernel_src/rmsnorm.cu
constexpr int MAX_THREADS = 1024;

#pragma unroll 4
for (int i = tid; i < vec_hidden; i += stride) {
    __nv_bfloat162 v = vec_in[i];
    // ...
}
```

**Identical to the RTX 5090.** Same sm_100 architecture, same GDDR7 memory, same thread/block configuration. The kernel code is the same — the difference is what you can do at the system level (larger models, bigger batches, longer contexts).

**Why 188 SMs matters**: You need >= 188 blocks (rows) to fully utilize the GPU. For batch=1, seq_len=128, only 68% of SMs are active. The RTX 6000 Pro reaches full utilization at longer sequences than any other GPU in this set.

#### Model Fit Analysis

| Model | BF16 Size | Fits? | Strategy |
|-------|-----------|-------|----------|
| Qwen3-8B | ~16 GB | Yes (80 GB free) | BF16, massive headroom |
| Qwen3-14B | ~28 GB | Yes (68 GB free) | BF16, comfortable |
| Qwen3-30B | ~60 GB | Yes (36 GB free) | BF16, good |
| Qwen3-72B | ~144 GB | No | Does not fit even quantized |

The 96 GB VRAM is the RTX 6000 Pro's key differentiator. While the kernel runs at the same speed as the RTX 5090, you can run Qwen3-30B in BF16 without any quantization — preserving full model quality.

#### Optimization Priority Order

1. **Vectorize loads** — foundation
2. **Use 1024 threads/block** — 100% occupancy
3. **Maximize L2 reuse** — 128 MB L2 is your secret weapon; inter-kernel data often stays cached
4. **Run BF16 whenever possible** — 96 GB means you rarely need to quantize
5. **Use CUDA graphs / torch.compile** — same launch overhead concerns as RTX 5090

---

### GB10 / DGX Spark (Blackwell, sm_120)

#### Hardware Quick Reference

```
┌─────────────────────────────────────────┐
│       GB10 DGX Spark (Blackwell)        │
├─────────────────────────────────────────┤
│ Compute:  sm_120, 48 SMs, 6144 cores   │
│ Memory:   128 GB LPDDR5X (unified)     │
│ BW:       273 GB/s (shared with CPU)   │
│ L2:       24 MB                         │
│ Shared:   128 KB/SM                     │
│ Threads:  2048/SM (64 warps)           │
│ Tensor:   5th gen (FP4/FP8/BF16)      │
│ TDP:      140W (entire SoC)            │
│ NVLink:   C2C (CPU-GPU, 600 GB/s)     │
│ CUDA:     >= 12.8 required             │
└─────────────────────────────────────────┘
```

#### The GB10's Unique Constraint Profile

The GB10 is **fundamentally different** from the other three GPUs. It is not a discrete GPU card — it is a System-on-Chip combining a 20-core Grace ARM CPU and a 48-SM Blackwell GPU sharing 128 GB of LPDDR5X memory.

```
┌──────────────────────────────────────────────────┐
│                  GB10 SoC (140W)                 │
│                                                  │
│  ┌──────────────┐    NVLink-C2C    ┌───────────┐ │
│  │  Grace CPU   │◄──600 GB/s BD──►│ Blackwell │ │
│  │  20 ARM v9.2 │                  │ GPU 48 SM │ │
│  └──────┬───────┘                  └─────┬─────┘ │
│         └────────────┬───────────────────┘       │
│              ┌───────▼────────┐                   │
│              │ 128 GB LPDDR5X │                   │
│              │   273 GB/s     │                   │
│              │  (shared)      │                   │
│              └────────────────┘                   │
└──────────────────────────────────────────────────┘
```

Three constraints dominate:

1. **273 GB/s shared bandwidth.** This is 6.5x lower than the RTX 5090/6000 Pro and 3.4x lower than the RTX 3090. Every byte of memory traffic is precious. Worse, this bandwidth is shared with the CPU — Python framework overhead, tokenization, and sampling all compete for the same memory bus.

2. **LPDDR5X latency (~150-200 ns).** Higher than GDDR6X (~100-150 ns) or GDDR7 (~100-120 ns). More warps and deeper unrolling are needed to keep the memory pipeline full.

3. **Only 48 SMs.** The smallest GPU in the set. Even moderate batch sizes (>48 rows) saturate SM utilization, which is an advantage — but single-token decode uses only 1/48th of the GPU.

The GB10's **unique advantage** is 128 GB of unified memory. No other GPU can run Qwen3-72B (even quantized). The GB10 with FP8 quantization (~72 GB) can. Models up to ~60B parameters run in BF16 without any quantization at all.

#### Optimal Kernel Configuration

```cuda
// GB10/qwen3_rmsnorm/kernel_src/rmsnorm.cu
constexpr int MAX_THREADS_GB10 = 512;

// Deep unroll for LPDDR5X latency hiding
#pragma unroll 8
for (int i = tid; i < vec_hidden; i += stride) {
    __nv_bfloat162 v = vec_in[i];
    // ...
}
```

**Why 512 threads (not 1024)**: Even though sm_120 supports 2048 threads/SM, using 512 threads/block allows 4 blocks per SM. Four blocks provide 64 warps — when one block stalls on a slow LPDDR5X load, three others can run. With 1024 threads, only 2 blocks fit, giving fewer opportunities to hide the higher memory latency.

**Why unroll 8**: LPDDR5X's ~150-200 ns latency means the memory pipeline needs more outstanding requests to stay full. With 512 threads and hidden_size=4096 (2048 vec elements / 512 threads = 4 iterations), `unroll 8` ensures the compiler fully unrolls the loop, issuing all load instructions before any dependent computation.

#### Model Fit Analysis

| Model | BF16 Size | Fits? | Strategy |
|-------|-----------|-------|----------|
| Qwen3-8B | ~16 GB | Yes (112 GB free) | BF16, enormous headroom |
| Qwen3-14B | ~28 GB | Yes (100 GB free) | BF16, comfortable |
| Qwen3-30B | ~60 GB | Yes (68 GB free) | BF16, good |
| Qwen3-72B | ~144 GB | No in BF16 | FP8 (~72 GB), fits |

The GB10 can run models that no other GPU in this set can touch. Qwen3-72B in FP8 with room for KV cache is a capability unique to the 128 GB unified memory.

#### Optimization Priority Order

1. **Vectorize everything** — 273 GB/s is precious; coalesced vectorized loads are the single biggest optimization
2. **Unroll aggressively** — `#pragma unroll 8` hides LPDDR5X latency
3. **Use 512 threads/block** — more blocks/SM for latency hiding
4. **Minimize CPU work during inference** — CPU traffic steals from the GPU's shared 273 GB/s; use `torch.inference_mode()`, CUDA graphs, minimize allocations
5. **Keep weights in L2** — RMSNorm weights (8 KB) trivially fit in the 24 MB L2 and stay hot across all rows
6. **BF16 by default** — 128 GB means you rarely need to quantize (except for 72B+ models)
7. **Warmup before benchmarking** — unified memory page migration needs a few iterations to settle

#### Code-Level Differences

The GB10 kernel differs from the others in two places:

```cuda
// 1. Thread count constant uses a different name (clarity)
constexpr int MAX_THREADS_GB10 = 512;  // vs MAX_THREADS on others

// 2. Unroll depth doubled
#pragma unroll 8  // vs #pragma unroll 4 on all other GPUs
```

The kernel structure, vectorization pattern, reduction logic, and launch helpers are otherwise identical.

---

### Side-by-Side Comparison: The Same Kernel, Four GPUs

Here are the three lines that actually change across the four kernel files:

| File | `MAX_THREADS` | `#pragma unroll` | `cuda-capabilities` |
|------|---------------|------------------|----------------------|
| `rtx-3090/.../rmsnorm.cu` | 512 | 4 | `"8.6"` |
| `rtx-5090/.../rmsnorm.cu` | 1024 | 4 | `"10.0"` |
| `rtx-6000-pro/.../rmsnorm.cu` | 1024 | 4 | `"10.0"` |
| `GB10/.../rmsnorm.cu` | 512 | 8 | `"12.0"` |

Everything else — the vectorization strategy, the two-phase reduce-then-normalize algorithm, the warp shuffle reduction, the block reduce via shared memory, the type conversion helpers — is identical across all four GPUs. The algorithm doesn't change. The tuning does.

---

## Part 4: MegaQwen Case Study

This section analyzes the MegaQwen project (documented at [elliotarledge.com/blog/megaqwen](https://elliotarledge.com/blog/megaqwen)), which fuses an entire Qwen3-0.6B transformer into a single cooperative CUDA kernel on the RTX 3090. It is a case study in how identifying the right bottleneck determines everything.

### What the Megakernel Is

A standard transformer inference pipeline launches hundreds of individual CUDA kernels per forward pass: one for each RMSNorm, one for each linear projection, one for each attention computation. Each kernel reads from and writes to global memory. Each has CPU-side launch overhead.

The MegaQwen megakernel takes the opposite approach: **all 28 transformer layers run as a single kernel launch**. Intermediate activations stay in registers and L2 cache instead of being written to and read from DRAM between operations. This eliminates:
- ~100+ kernel launches per forward pass (saving ~5-10 us each)
- ~100+ global memory round-trips for intermediate activations
- CPU dispatch overhead between kernels

For Qwen3-0.6B (a small model), this produced a 3.9x speedup over HuggingFace's baseline: **530 tok/s** vs 136 tok/s on the RTX 3090.

### How Cooperative Groups and grid.sync() Work

Normal CUDA kernels cannot synchronize across blocks — blocks are independent. Cooperative groups (`cg::grid_group`) change this by enabling **grid-wide barriers** where all blocks wait until every block reaches the same point.

```
Normal kernel: blocks are independent
  Block 0: [compute] [done]
  Block 1: [compute] [done]        ← no coordination between blocks
  Block 2: [compute] [done]

Cooperative kernel: blocks synchronize
  Block 0: [phase 1] |sync| [phase 2] |sync| [phase 3]
  Block 1: [phase 1] |sync| [phase 2] |sync| [phase 3]
  Block 2: [phase 1] |sync| [phase 2] |sync| [phase 3]
                      ▲                ▲
                 grid.sync()      grid.sync()
                 all blocks       all blocks
                 wait here        wait here
```

The MegaQwen kernel uses this to implement the sequential dependency chain in a transformer: RMSNorm output feeds into QKV projection, which feeds into attention, which feeds into MLP. Each phase ends with a `grid.sync()` so all blocks see the completed results before starting the next phase.

The RTX 3090 with 82 SMs runs 82 blocks in the cooperative kernel. Each decode step requires approximately **225 `grid.sync()` calls** (8 per layer x 28 layers + extras).

### The Key Findings

**Peak performance**: 530 tok/s at short context, batch=1, BF16.

**The bottleneck is synchronization, not memory bandwidth:**

```
Measured bandwidth utilization: ~47 GB/s
Peak bandwidth:                 936 GB/s
Utilization:                    ~5%
```

The kernel is using only 5% of the RTX 3090's memory bandwidth. It is not waiting on data — it is waiting on synchronization. Each `grid.sync()` takes approximately 0.7 microseconds. With 140+ sync calls per token, that's ~100 microseconds of pure waiting per token — a significant fraction of the ~1.9 ms per-token time at 530 tok/s.

**Key optimizations and their impact:**

| Optimization | Effect | Why |
|---|---|---|
| Block divergence + L2 prefetch | +2x | During attention (only 16 of 82 blocks active), idle blocks prefetch MLP weights into L2 via `__ldg()` |
| Redundant RMSNorm | +42% (short ctx) | Each block computes RMSNorm independently, eliminating 56 `grid.sync()` calls |
| 128-bit vectorized loads | +3.5% | Wider loads reduce instruction count |
| Warp producer/consumer split | 0% | Reducing compute warps hurt more than prefetching helped |
| Shared memory caching | 0% | L1/L2 already effective; extra `__syncthreads()` added overhead |
| `cp.async` double-buffering | +1% | Insufficient compute-memory overlap to justify complexity |

**The 530 tok/s ceiling**: The blog concludes this is the architectural ceiling for batch=1 BF16 cooperative megakernels on the RTX 3090. Further gains require fundamentally different approaches: INT4 quantization (~4x model size reduction), speculative decoding (~2-4x tokens per forward pass), or leaving the cooperative megakernel paradigm entirely.

### Per-GPU Analysis: Would MegaQwen Work on Other GPUs?

The megakernel approach was designed for the RTX 3090's specific constraints. Would it work on the other three GPUs?

**RTX 5090 (170 SMs, 1.79 TB/s, 96 MB L2):**
- **More SMs = more sync overhead.** With 170 blocks instead of 82, each `grid.sync()` must wait for 2x more blocks to arrive at the barrier. Sync cost per call likely increases.
- **Higher bandwidth changes the calculus.** At 1.79 TB/s (vs 936 GB/s), conventional separate-kernel execution is much faster. The overhead of separate kernel launches (~5-10 us each) is a smaller fraction of the faster kernel execution time.
- **96 MB L2 enables inter-kernel caching.** The intermediate activations that MegaQwen keeps in registers/L2 would naturally stay in the RTX 5090's L2 between separate kernel launches anyway.
- **Verdict**: The megakernel approach provides less benefit. The sync overhead is higher, and the baseline it's competing against is faster. Conventional kernels + CUDA graphs + torch.compile are likely the better path.

**RTX 6000 Pro (188 SMs, 1.79 TB/s, 128 MB L2):**
- Same analysis as RTX 5090, but **worse** — 188 SMs means even more blocks at each sync barrier.
- **128 MB L2** makes inter-kernel reuse even more effective with separate kernels.
- **Verdict**: Not recommended. The conventional approach wins.

**GB10 (48 SMs, 273 GB/s, 24 MB L2):**
- **Fewer SMs = cheaper sync.** Only 48 blocks need to synchronize, which should be faster than the 3090's 82.
- **Low bandwidth makes fusion valuable.** At 273 GB/s, every eliminated DRAM round-trip saves more time. The megakernel's register/L2 residency approach is highly attractive.
- **But CPU contention matters.** The cooperative launch still involves CPU interaction, and on GB10, CPU memory traffic competes with GPU memory traffic on the shared bus.
- **Verdict**: Most promising alternative to the RTX 3090. The low bandwidth makes fusion critical, and fewer SMs mean cheaper synchronization. However, the 273 GB/s ceiling means absolute performance will be much lower.

### The Key Lesson: Identify Your Bottleneck First

MegaQwen demonstrates a universal principle: **you must identify which bottleneck you're hitting before optimizing**.

The developers spent significant effort on memory bandwidth optimizations (vectorized loads, shared memory caching, double-buffering) — and got minimal returns. The bottleneck was synchronization latency from `grid.sync()`, not memory bandwidth. The biggest win came from **eliminating sync calls** (redundant RMSNorm: +42%) and **hiding sync latency** (block divergence + L2 prefetch: +2x).

On a different GPU, the bottleneck might be different:
- On GB10, it would likely be **memory bandwidth** (273 GB/s is the hard limit)
- On RTX 5090, it would likely be **kernel launch overhead** (fast kernels make the gaps between them proportionally larger)
- On RTX 6000 Pro, it might be **SM underutilization** for short sequences (188 SMs with batch=1)

The right optimization strategy depends entirely on which constraint you hit first. Profile before you optimize.

---

## Part 5: Model Size & Architecture Decision Framework

### How Model Size Changes What Matters

The optimization landscape shifts dramatically as models scale:

**Small models (< 3B parameters):**
- Entire model fits in L2 cache region. Latency-dominated.
- Kernel launch overhead is a large fraction of total time.
- CUDA graphs and kernel fusion have the highest relative impact.
- The MegaQwen megakernel approach is most viable here.

**Medium models (3B-14B parameters):**
- Model weights in DRAM, but fit on most GPUs in BF16.
- Memory bandwidth is the primary constraint.
- Vectorized loads and high bandwidth utilization matter most.
- This is where our RMSNorm kernel optimization has the most impact.

**Large models (14B-72B parameters):**
- VRAM capacity becomes the constraint.
- Quantization strategy determines what's even possible.
- The optimization shifts from "how fast" to "can it fit at all."
- GB10 (128 GB) and RTX 6000 Pro (96 GB) have unique advantages.

**Very large models (72B+):**
- Multi-GPU or multi-node required on all GPUs except GB10 (with heavy quantization).
- Communication overhead dominates optimization.
- Kernel optimization matters less than parallelism strategy.

### Dense vs MoE vs Long-Context vs Batched Serving

**Dense models** (Qwen3-8B, LLaMA-70B): Standard case. All parameters are active for every token. Memory bandwidth determines throughput for memory-bound ops; compute throughput determines throughput for GEMMs.

**Mixture-of-Experts (MoE)** models (Mixtral, DeepSeek): Only a subset of parameters (experts) is active per token, but all parameters must be in memory. The key constraint is VRAM capacity for weights + routing overhead. MoE models are especially friendly to the GB10 (128 GB capacity) and RTX 6000 Pro (96 GB) because the total parameter count is high but per-token compute is moderate.

**Long-context inference** (32K-128K+ tokens): KV cache grows linearly with context length and dominates VRAM usage. At 128K context with Qwen3-8B, KV cache alone is ~18 GB. Long context pushes VRAM-limited GPUs (3090: 24 GB, 5090: 32 GB) toward KV cache quantization (INT8 or FP8) or shorter contexts.

**Batched serving** (multiple concurrent requests): Batching increases arithmetic intensity of GEMMs (more efficient use of tensor cores) but multiplies memory requirements. A batch of 8 requires 8x the activation memory and 8x the KV cache. VRAM capacity and memory bandwidth both matter.

### Quantization Strategy Per GPU

| Format | RTX 3090 | RTX 5090 | RTX 6000 Pro | GB10 |
|--------|----------|----------|--------------|------|
| BF16 | Yes | Yes | Yes | Yes |
| FP16 | Yes | Yes | Yes | Yes |
| INT8 | Yes (TC) | Yes (TC) | Yes (TC) | Yes (TC) |
| INT4 | Yes (TC) | Yes (TC) | Yes (TC) | Yes (TC) |
| FP8 | **No** | Yes (TC) | Yes (TC) | Yes (TC) |
| FP4/NVFP4 | **No** | Yes (TC) | Yes (TC) | Yes (TC) |

**RTX 3090**: Limited to INT8 and INT4. Use GPTQ or AWQ for INT4 weight-only quantization. No FP8 means less granular quantization options — you jump from BF16 (2 bytes) directly to INT8 (1 byte) or INT4 (0.5 bytes).

**RTX 5090**: Full quantization support. FP8 is the recommended first step when BF16 doesn't fit — it halves model size with minimal quality loss. NVFP4 is available for aggressive compression.

**RTX 6000 Pro**: Same quantization support as RTX 5090, but 96 GB VRAM means you rarely need it for models up to ~30B.

**GB10**: Full quantization support. 128 GB means BF16 is viable for most models. Use FP8 for Qwen3-72B to fit within 128 GB with room for KV cache.

### Decision Flowchart

```
Start: What model are you running?
  │
  ├─ Model fits in BF16 with room for KV cache?
  │   ├─ YES → Run in BF16. Optimize kernel bandwidth utilization.
  │   └─ NO ─┐
  │           │
  │   ├─ GPU supports FP8?
  │   │   ├─ YES → Does model fit in FP8?
  │   │   │   ├─ YES → Use FP8. Minimal quality loss.
  │   │   │   └─ NO → Use INT4 (GPTQ/AWQ).
  │   │   └─ NO (RTX 3090) → Use INT4 (GPTQ/AWQ).
  │   │
  │   └─ Still doesn't fit?
  │       ├─ Consider a GPU with more VRAM (GB10: 128 GB, RTX 6000 Pro: 96 GB)
  │       └─ Or multi-GPU tensor parallelism
  │
  └─ What's the primary workload?
      ├─ Single-request, low-latency → Optimize kernel launch overhead (CUDA graphs)
      ├─ Batched serving → Optimize VRAM efficiency (quantization + KV cache management)
      └─ Long context (>32K) → Optimize KV cache (FP8 KV, paged attention)
```

---

## Part 6: Practical Tooling

### Profiling Commands

**System-wide profiling with Nsight Systems (nsys):**
```bash
# See the full timeline: kernels, launches, CPU activity, memory transfers
nsys profile -o my_profile python my_inference_script.py

# Open in Nsight Systems GUI or analyze on command line:
nsys stats my_profile.nsys-rep
```

What to look for:
- **GPU idle gaps** between kernels (fusion or CUDA graph opportunities)
- **CPU activity during GPU execution** (especially important on GB10 with shared bandwidth)
- **Kernel duration** (compare across GPUs to validate bandwidth utilization)

**Per-kernel analysis with Nsight Compute (ncu):**
```bash
# Full analysis of one kernel invocation
ncu --set full -o metrics.ncu-rep python my_benchmark.py

# Key metrics for memory-bound kernels
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed \
    python my_benchmark.py
```

### Key Metrics Per GPU

| Metric | RTX 3090 Target | RTX 5090 / 6000 Pro Target | GB10 Target |
|--------|-----------------|---------------------------|-------------|
| `dram__throughput` | > 35% of 936 GB/s | > 35% of 1.79 TB/s | > 30% of 273 GB/s |
| `sm__warps_active` | > 50% | > 50% | > 60% |
| `lts__throughput` (L2) | Low-moderate (6 MB L2) | Varies (96/128 MB L2) | Moderate (24 MB L2) |
| `sm__throughput` | Low (memory-bound) | Low (memory-bound) | Low (memory-bound) |

For memory-bound kernels, `dram__throughput` is the metric that matters. If it's below target, investigate coalescing (vectorize loads), occupancy (check thread/register count), and bank conflicts (usually not an issue for RMSNorm).

### Build System

Each GPU's kernel uses a `build.toml` file that specifies the CUDA architecture target:

```toml
# Example: rtx-3090/qwen3_rmsnorm/build.toml
[general]
name = "qwen3_kernels"
backends = ["cuda"]

[kernel.rmsnorm]
backend = "cuda"
src = ["kernel_src/rmsnorm.cu"]
cuda-capabilities = ["8.6"]        # ← changes per GPU
```

Build with the `kernel-builder` package:
```bash
pip install kernel-builder
cd gpu-kernels/rtx-3090/qwen3_rmsnorm/
kernel-builder build
```

Or with `setup.py` for direct NVCC compilation:
```bash
python setup.py install
```

The `cuda-capabilities` field maps directly to NVCC's `-arch` flag:

| build.toml | NVCC flag | GPU |
|------------|-----------|-----|
| `"8.6"` | `-arch=sm_86` | RTX 3090 |
| `"10.0"` | `-arch=sm_100` | RTX 5090, RTX 6000 Pro |
| `"12.0"` | `-gencode=arch=compute_120,code=sm_120` | GB10 |

---

## Appendix: File Reference

| File | Description |
|------|-------------|
| `gpu-kernels/GPU-COMPARISON.md` | Hardware specs comparison table for all four GPUs |
| `gpu-kernels/rtx-3090/qwen3_rmsnorm/kernel_src/rmsnorm.cu` | RTX 3090 kernel (MAX_THREADS=512, unroll 4) |
| `gpu-kernels/rtx-5090/qwen3_rmsnorm/kernel_src/rmsnorm.cu` | RTX 5090 kernel (MAX_THREADS=1024, unroll 4) |
| `gpu-kernels/rtx-6000-pro/qwen3_rmsnorm/kernel_src/rmsnorm.cu` | RTX 6000 Pro kernel (MAX_THREADS=1024, unroll 4) |
| `gpu-kernels/GB10/qwen3_rmsnorm/kernel_src/rmsnorm.cu` | GB10 kernel (MAX_THREADS=512, unroll 8) |
| `gpu-kernels/rtx-3090/references/rtx3090-optimization-guide.md` | RTX 3090 deep dive |
| `gpu-kernels/rtx-5090/references/rtx5090-optimization-guide.md` | RTX 5090 deep dive |
| `gpu-kernels/GB10/references/gb10-optimization-guide.md` | GB10 deep dive |
| `gpu-kernels/GB10/references/unified-memory-notes.md` | GB10 unified memory analysis |
