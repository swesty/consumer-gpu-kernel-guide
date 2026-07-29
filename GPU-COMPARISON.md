# GPU Comparison Matrix

Side-by-side comparison of all GPUs in this repository, covering architecture, memory, kernel tuning, and model fit.

## Hardware Specifications

| Spec | GB10 (DGX Spark) | RTX 3090 | RTX 5090 | RTX 6000 / RTX 6000 Pro |
|------|:-----------------:|:--------:|:--------:|:-----------------------:|
| **Architecture** | Blackwell | Ampere | Blackwell | Blackwell |
| **GPU Die** | GB10 SoC | GA102 | GB202 | GB202 |
| **Process** | TSMC 5nm | Samsung 8nm | TSMC 4NP | TSMC 5nm |
| **Compute Capability** | sm_121a | sm_86 | sm_120 | sm_120 |
| **SMs** | 48 | 82 | 170 | 188 |
| **CUDA Cores** | 6,144 | 10,496 | 21,760 | 24,064 |
| **Tensor Cores** | 192 (5th gen) | 328 (3rd gen) | 680 (5th gen) | 752 (5th gen) |
| **Base / Boost Clock** | — | — | 2,017 / 2,407 MHz | 1,590 / 2,617 MHz |
| **TDP** | 140W (SoC) | 350W | 575W | 600W |

## Memory Subsystem

| Spec | GB10 (DGX Spark) | RTX 3090 | RTX 5090 | RTX 6000 / RTX 6000 Pro |
|------|:-----------------:|:--------:|:--------:|:-----------------------:|
| **Memory Type** | LPDDR5X (unified) | GDDR6X | GDDR7 | GDDR7 |
| **Memory Size** | 128 GB | 24 GB | 32 GB | 96 GB |
| **Memory Interface** | — (C2C) | 384-bit | 512-bit | 512-bit |
| **Memory Bandwidth** | 273 GB/s (shared) | 936 GB/s | 1.79 TB/s | 1.79 TB/s |
| **L2 Cache** | 24 MB | 6 MB | 96 MB | 128 MB |
| **Shared Memory / SM** | 128 KB | 100 KB | 128 KB | 128 KB |
| **Registers / SM** | 64K (32-bit) | 64K (32-bit) | 64K (32-bit) | 64K (32-bit) |
| **ECC** | No | No | No | Yes |
| **CPU-GPU Link** | NVLink-C2C (zero-copy) | PCIe Gen 4 x16 | PCIe Gen 5 x16 | PCIe Gen 5 x16 |
| **NVLink (GPU-GPU)** | No | 2-way (112.5 GB/s) | No | No |

## Thread / Occupancy Limits

| Spec | GB10 (DGX Spark) | RTX 3090 | RTX 5090 | RTX 6000 / RTX 6000 Pro |
|------|:-----------------:|:--------:|:--------:|:-----------------------:|
| **Max Threads / SM** | 2,048 | **1,536** | 2,048 | 2,048 |
| **Max Threads / Block** | 1,024 | 1,024 | 1,024 | 1,024 |
| **Max Warps / SM** | 64 | **48** | 64 | 64 |
| **Max Blocks / SM** | 32 | 16 | 32 | 32 |

## Precision Support

| Format | GB10 (DGX Spark) | RTX 3090 | RTX 5090 | RTX 6000 / RTX 6000 Pro |
|--------|:-----------------:|:--------:|:--------:|:-----------------------:|
| **FP32** | Yes | Yes | Yes | Yes |
| **TF32** | Yes | Yes | Yes | Yes |
| **BF16** | Yes | Yes | Yes | Yes |
| **FP16** | Yes | Yes | Yes | Yes |
| **FP8 (E4M3/E5M2)** | Yes | **No** | Yes | Yes |
| **FP4 / NVFP4** | Yes | **No** | Yes | Yes |
| **INT8** | Yes | Yes | Yes | Yes |
| **INT4** | Yes | Yes | Yes | Yes |

## Kernel Tuning Parameters

Recommended settings for RMSNorm-class reduction kernels (hidden_size=4096, BF16):

| Parameter | GB10 (DGX Spark) | RTX 3090 | RTX 5090 | RTX 6000 / RTX 6000 Pro |
|-----------|:-----------------:|:--------:|:--------:|:-----------------------:|
| **Threads / Block** | 512 | 512 | 1,024 | 1,024 |
| **Blocks / SM** | 4 | 3 | 2 | 2 |
| **Occupancy** | 100% | 100% | 100% | 100% |
| **`#pragma unroll`** | 8 | 4 | 4 | 4 |
| **Grid size target** | >= 48 | >= 82 | >= 170 | >= 188 |
| **BW target (% peak)** | > 30% | > 35% | > 35% | > 35% |
| **`cuda-capabilities`** | `"12.1"` | `"8.6"` | `"12.0"` | `"12.0"` |
| **Min CUDA Toolkit** | 12.9+ | 11.1+ | 12.8+ | 12.8+ |

**Why the differences:**

- **GB10 uses 512 threads, unroll 8** — LPDDR5X has higher latency (~150-200 ns) than GDDR, so more blocks per SM and deeper unrolling are needed to keep enough memory requests in flight.
- **RTX 3090 uses 512 threads, unroll 4** — sm_86 limits threads/SM to 1,536; using 1024 threads/block would give only 1 block/SM (66.7% occupancy). GDDR6X latency (~100 ns) is low enough for unroll 4.
- **RTX 5090 / RTX 6000 Pro use 1024 threads, unroll 4** — sm_120 supports 2,048 threads/SM, so 2 blocks of 1024 achieves 100% occupancy with less scheduling overhead. GDDR7 latency is low.

### Build target compatibility (Blackwell)

The Blackwell parts split across two compute capabilities (12.0 for the GDDR7 cards, 12.1 for
GB10), so a single build target does not always cover them. Measured by launching SASS-only
binaries — no embedded PTX, so no JIT rescue:

| Built for | RTX 5090 / RTX 6000 Pro (CC 12.0) | GB10 (CC 12.1) | Min CUDA |
|-----------|:---------------------------------:|:--------------:|:--------:|
| `sm_120` | runs | runs | 12.8+ |
| `sm_120f` | runs | runs | 12.9+ |
| `sm_121` | fails | runs | 12.9+ |
| `sm_121a` | fails | runs | 12.9+ |

`sm_120` covers all three (cubins are forward-compatible across minor revisions). `sm_120f` adds
family-specific features at the same coverage. The `a` suffix is architecture-specific and loads
on CC 12.1 only — failures show up at launch as `no kernel image is available for execution on
the device`. See [guides/tuning-guide.md](guides/tuning-guide.md) for flag recipes.

## Model Fit (Qwen3 Family, BF16)

| Model | Size (BF16) | GB10 (128 GB) | RTX 3090 (24 GB) | RTX 5090 (32 GB) | RTX 6000 Pro (96 GB) |
|-------|:-----------:|:-------------:|:-----------------:|:-----------------:|:--------------------:|
| Qwen3-8B | ~16 GB | 112 GB free | 8 GB free | 16 GB free | 80 GB free |
| Qwen3-14B | ~28 GB | 100 GB free | Needs INT4 (~7 GB) | Needs FP8 (~14 GB) | 68 GB free |
| Qwen3-30B | ~60 GB | 68 GB free | Needs INT4 (~15 GB) | Needs INT4 (~15 GB) | 36 GB free |
| Qwen3-72B | ~144 GB | Needs FP8 (~72 GB) | Does not fit | Does not fit | Does not fit |

## Expected RMSNorm Performance

Estimated kernel latency for `[batch, 2048, 4096]` BF16 RMSNorm:

| Shape | GB10 | RTX 3090 | RTX 5090 | RTX 6000 Pro |
|-------|:----:|:--------:|:--------:|:------------:|
| [1, 2048, 4096] | ~0.12 ms | ~0.08 ms | ~0.04 ms | ~0.04 ms |
| [4, 2048, 4096] | ~0.47 ms | ~0.32 ms | ~0.17 ms | ~0.17 ms |

Performance scales roughly with memory bandwidth (273 → 936 → 1,790 GB/s).

## Roofline Crossover (FLOP/byte)

The arithmetic intensity where a kernel transitions from memory-bound to compute-bound:

| GPU | Peak FP32 (TFLOPS) | Peak BW (TB/s) | Crossover (FLOP/byte) |
|-----|:-------------------:|:--------------:|:---------------------:|
| GB10 | ~10 (est.) | 0.273 | ~37 |
| RTX 3090 | ~35.6 | 0.936 | ~38 |
| RTX 5090 | ~90 | 1.79 | ~50 |
| RTX 6000 Pro | ~100 | 1.79 | ~56 |

RMSNorm (~6 FLOP/byte) is **memory-bound on all GPUs**. GEMM (~100+ FLOP/byte) is compute-bound on all.

## L2 Cache Working Set Analysis

Whether common activation tensors fit in L2 (enabling inter-kernel reuse):

| Working Set | Size | GB10 (24 MB) | RTX 3090 (6 MB) | RTX 5090 (96 MB) | RTX 6000 Pro (128 MB) |
|-------------|:----:|:------------:|:----------------:|:-----------------:|:---------------------:|
| All RMSNorm weights | ~1.2 MB | Yes | Yes | Yes | Yes |
| Activations [1, 512, 4096] BF16 | 4 MB | Yes | Barely | Yes | Yes |
| Activations [1, 2048, 4096] BF16 | 16 MB | Yes | **No** | Yes | Yes |
| Activations [4, 2048, 4096] BF16 | 64 MB | **No** | **No** | Yes | Yes |
| Activations [4, 4096, 4096] BF16 | 128 MB | **No** | **No** | **No** | Yes |

## Key Differentiators

| Feature | GB10 (DGX Spark) | RTX 3090 | RTX 5090 | RTX 6000 / RTX 6000 Pro |
|---------|:-----------------:|:--------:|:--------:|:-----------------------:|
| **Best for** | Large models, low power | Budget, legacy | Consumer flagship | Workstation, large models |
| **Unique advantage** | 128 GB unified memory | NVLink 2-way (48 GB) | Best perf/$ | 96 GB + ECC + 128 MB L2 |
| **Primary constraint** | 273 GB/s shared BW | 24 GB VRAM, 6 MB L2 | 32 GB VRAM | 600W power |
| **Cooling** | Fanless possible | Air/AIO | Air/AIO (850W+ PSU) | Active (workstation) |
| **Multi-GPU** | Single only | NVLink or PCIe | PCIe only | PCIe only |
| **Quantization strategy** | BF16 (fits most models) | INT4/INT8 (no FP8) | FP8/FP4 preferred | BF16 up to ~30B |

## Directory Structure

```
gpu-kernels/
├── GPU-COMPARISON.md              ← this file
├── GB10/                          ← DGX Spark (sm_121a, 128 GB LPDDR5X)
│   ├── references/
│   └── qwen3_rmsnorm/
├── rtx-3090/                      ← Ampere (sm_86, 24 GB GDDR6X)
│   ├── references/
│   └── qwen3_rmsnorm/
├── rtx-5090/                      ← Blackwell (sm_120, 32 GB GDDR7)
│   ├── references/
│   └── qwen3_rmsnorm/
├── rtx-6000-pro/                  ← RTX 6000 Pro (sm_120, 96 GB GDDR7) — structured layout
    ├── references/
    └── qwen3_rmsnorm/
```

