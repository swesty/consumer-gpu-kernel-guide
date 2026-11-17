# Attention kernels on RTX PRO 6000 Blackwell (SM120) and DGX Spark (SM121)

**A source-level technical report: instruction set, kernel internals, engine integration, open work, published numbers, and a research path**

Snapshot date: 2026-11-17. Companion diagrams: [`visualizer.html`](visualizer.html).

---

## Contents

0. Scope, method, and how to read the labels
1. Summary of findings
2. The hardware target: instruction set and resource limits
3. The common anatomy of an SM12x attention kernel
4. Kernel-by-kernel internals
5. How vLLM drives these kernels
6. How SGLang drives these kernels
7. MLA and sparse attention
8. Open work, pull request by pull request
9. Published performance numbers
10. Where SM120 and SM121 diverge
11. Research path
12. Operator reference: flags, log lines, verification commands
13. Corrections, disagreements between sources, and unknowns
14. Sources

---

## 0. Scope, method, and how to read the labels

**Scope.** Every attention kernel that vLLM, SGLang or FlashInfer can run on compute capability 12.0 (`sm_120`: RTX PRO 6000 Blackwell, RTX 5090) and 12.1 (`sm_121`: DGX Spark GB10), what each kernel does internally, what is unfinished, and what is worth researching.

**Method.** Six repositories were cloned and read at the commits below. NVIDIA's PTX manual, Blackwell tuning guide and CUDA programming guide were read from docs.nvidia.com. Pull requests and issues were read through the GitHub API. **No code was compiled or executed and no GPU was used.** Every performance number in this report is a claim made by a pull-request or issue author.

| Component | Commit / version | Read as |
|---|---|---|
| vLLM | `main` @ [`f39d833`](https://github.com/vllm-project/vllm/tree/f39d83306ea62d4ad8919995f378d8ca4aed8b57) (2026-11-17) | full checkout |
| SGLang | `main` @ [`b2cc023f`](https://github.com/sgl-project/sglang/tree/b2cc023f23e14e6482490238968e018573847d0c) (2026-11-17) | shallow clone |
| FlashInfer | `main` @ [`778f57e`](https://github.com/flashinfer-ai/flashinfer/tree/778f57e4d73a73a439ad9cb88daabb5e14cbf816) (2026-11-17); release notes through v0.7.4 | shallow clone |
| FlashAttention (upstream) | Dao-AILab `main` @ [`9f9742c`](https://github.com/Dao-AILab/flash-attention/tree/9f9742c80e69b3ca28a231c26ef931e98c2c3826) (2026-11-17) | shallow clone |
| FlashAttention (vLLM fork) | `vllm-project/flash-attention` @ [`8dd9eaea`](https://github.com/vllm-project/flash-attention/tree/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2) (the commit vLLM pins) | checkout |
| CUTLASS | NVIDIA `main` @ [`b2f249a`](https://github.com/NVIDIA/cutlass/tree/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd) (2026-11-06) | shallow clone |
| PTX ISA manual | version 9.5 (CUDA 13.5) | web |
| Blackwell tuning guide | version 13.5, updated 2026-11-02 | web |
| CUDA programming guide | version 13.5, section 5.1 | web |

Both engines pin `flashinfer-python==0.7.4` (vLLM [`requirements/cuda.txt:13`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/requirements/cuda.txt#L13); SGLang [`python/pyproject.toml:41`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/pyproject.toml#L41)). The FlashInfer source read here is `main`, which is ahead of the 0.7.4 wheel; where that matters it is said.

**Labels.**

| Label | Meaning |
|---|---|
| **[SRC]** | Read directly from source at the cited `path:line` |
| **[DOC]** | Read from NVIDIA or project documentation |
| **[PR]** / **[ISSUE]** | A claim by a pull-request or issue author; not reproduced |
| **[CALC]** | Arithmetic performed for this report from a quoted formula |
| **[INF]** | Inference from the cited facts; not observed |
| **[VAULT]** | From the author's own notes, with the date last checked |

File citations are relative to the repository named in the surrounding section. Bracketed codes such as [S7] refer to section 14.

**Terms.** "SM12x" means compute-capability major 12, both machines. "SM120" means exactly 12.0. "SM121" means exactly 12.1. A "CTA" is a thread block. "Prefill" is processing a prompt (many query rows). "Decode" is generating one token (one query row against the whole KV cache).

---

## 1. Summary of findings

1. **Both engines default to FlashAttention-2-class kernels on SM12x.** vLLM selects `FLASH_ATTN` running FA2; SGLang selects FlashInfer's `fa2`. Neither default uses a kernel written for these GPUs.
2. **vLLM's default changes with the model configuration.** FA2 fails validation for an FP8 or NVFP4 KV cache, attention sinks, or a multimodal prefix. vLLM then falls through to `FLASHINFER`, which on SM12x pairs FlashInfer's native prefill with the XQA decode kernel.
3. **Three kernel families were written specifically for SM12x.** FlashInfer's b12x paged attention (reachable from vLLM), SGLang's `fa4_sm120` kernels (reachable from SGLang), and an unmerged set of upstream FlashAttention pull requests. All three use warp-level `mma.sync`. All three use TMA for at least one path.
4. **FA2 on SM12x runs with tiles tuned for A100 and H100.** No file in FA2 tests for capability 12. Head dimension 128 gets a 128×64 tile with 4 warps, which is 64 KB of shared memory.
5. **Upstream FlashAttention-4 on SM120 is a 61-line subclass of the Ampere kernel.** It uses `cp.async`, one pipeline stage and no TMA. It rejects paged KV, split-KV, FP8 and learnable sinks. vLLM's bundled fork additionally reports FA4 as unsupported unless the capability major is exactly 10.
6. **b12x paged decode already does page-aligned TMA.** The page id is a tensor coordinate, the page size must be a multiple of the KV tile, and there is one block-table lookup per tile. b12x paged prefill does not: it uses per-thread 16-byte `cp.async` with a per-row page lookup, except for one shape.
7. **Native FP8 tensor-core multiply exists on SM12x and is mostly unused.** b12x has it behind an environment variable. Upstream and SGLang have it in open pull requests that report 1.7× to 2.0× over BF16.
8. **Public benchmark numbers exist, and they are modest for BF16.** On an RTX PRO 6000, the new BF16 kernels are reported at roughly 1.1× to 1.2× over FA2. The large reported gains are for FP8, for decode against the unoptimized upstream FA4, and against Triton.
9. **The DGX Spark is admitted but not tuned.** b12x's tile specializations are keyed on exactly (12, 0). SGLang's split-KV latency model was fitted on 110-SM and 188-SM discrete cards. One FA2 tile fix explicitly excludes 12.1.
10. **NVIDIA's documents disagree on shared memory per SM for capability 12.** The tuning guide says 128 KB. The programming guide says 100 KB, with 128 KB as the unified data cache. Both agree on 99 KB per thread block. The answer decides how many thread blocks fit per SM for every kernel in this report.

---

## 2. The hardware target: instruction set and resource limits

### 2.1 The tensor-core instruction: `mma.sync` only

**Conclusion.** SM12x has no warpgroup instruction and no tensor-memory instruction. The only tensor-core instruction is the warp-level, register-operand `mma.sync`.

- **`wgmma` is Hopper-only.** PTX: `wgmma.mma_async` "Requires sm_90a" [DOC] [S1 §9.7.17.5.2].
- **`tcgen05` is data-center Blackwell only.** PTX lists `sm_100a`, `sm_101a` and the `sm_100f` / `sm_110f` families; no SM12x target [DOC] [S1 §9.7.18.10.10.1].
- **CUTLASS gates both by equality.** `CUTLASS_ARCH_MMA_SM90_ENABLED` requires `__CUDA_ARCH__ == 900`; `CUTLASS_ARCH_MMA_SM100_ENABLED` requires `== 1000` [SRC] [`cutlass/include/cutlass/arch/config.h:45, 89`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/arch/config.h#L45).
- **CUTLASS's README states the consequence:** "kernels compiled for Blackwell SM100 architecture with arch conditional features (using `sm100a`) are not compatible with RTX 50 series GPUs" [SRC] [`cutlass/README.md:331-334`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/README.md#L331-L334).

What `mma.sync` offers on SM12x:

| Variant | Shape | Operands | Target needed | Source |
|---|---|---|---|---|
| BF16 / FP16 | `m16n8k16` | 16-bit | baseline `sm_120` ("requires sm_80 or higher") | [S1 §9.7.16.5.14]; [`cute/arch/mma_sm80.hpp:39, 239`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm80.hpp#L39) |
| FP8, legacy form | `m16n8k32` | `.e4m3`, `.e5m2` | baseline `sm_120` ("requires sm_89 or higher") | [S1]; [`cute/arch/mma_sm89.hpp:50, 79`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm89.hpp#L50) |
| `.kind::f8f6f4` | `m16n8k32` | FP8, FP6, FP4 mixed | `sm_120a` or `sm_120f` | [S1]; [`cute/arch/mma_sm120.hpp:68`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm120.hpp#L68) |
| `.kind::mxf8f6f4` (block-scaled) | `m16n8k32` | with `.ue8m0` scales | `sm_120a` or `sm_120f` | [S1]; [`mma_sm120.hpp:1809`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm120.hpp#L1809) |
| `.kind::mxf4nvf4` (block-scaled FP4) | `m16n8k64` | `.e2m1` with scales | dense: `sm_120a` / `sm_120f`; sparse: `sm_120a` / `sm_121a` only | [S1 §9.7.16.6.3]; [`mma_sm120.hpp:3159`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm120.hpp#L3159) |

Two consequences for attention:

- **A BF16 attention kernel needs no special target.** The Ampere atom works on plain `sm_120`.
- **A native FP8 attention kernel is possible on plain `sm_120`** through the legacy `m16n8k32` form. The `.kind` forms need the `a` or `f` target [INF from the PTX target notes].

**CUTLASS ships no BF16 reference kernel for SM120.** Its SM120 builder asserts "SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA" [SRC] [`cutlass/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:80`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl#L80). A BF16 attention kernel therefore combines the Ampere atom with the SM120 TMA scaffolding by hand.

### 2.2 Operand path: shared memory → `ldmatrix` → registers

On Hopper, `wgmma` reads operands directly from shared memory through a descriptor. On SM12x every operand must first be copied into registers.

- CUTLASS's SM120 mainloop asserts the MMA fragments are not descriptor iterators, then copies: "copy smem->rmem for A/B operand" before `cute::gemm` [SRC] [`cutlass/include/cutlass/gemm/collective/sm120_mma_tma.hpp:153, 511-523`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/collective/sm120_mma_tma.hpp#L153).
- Each `mma.sync` atom holds 4 A registers, 2 B registers and 4 accumulator registers per thread [SRC] [`cute/arch/mma_sm120.hpp:57-60`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm120.hpp#L57-L60).
- The limit is 255 registers per thread [DOC] [S2 §1.4.1.1].

**Consequence [INF].** Operand fragments compete with accumulators for the register budget. This is the register pressure that Hopper and data-center Blackwell kernels avoid, and it bounds how many MMA atoms a warp can hold live.

`ldmatrix` variants: the classic `m8n8` 16-bit form is baseline. The `m16n16` transposing 8-bit form and the sub-byte FP4/FP6 forms need `sm_120a` or `sm_120f` [DOC] [S1 §9.7.16.5.15].

### 2.3 TMA: available, without multicast

**Conclusion.** TMA and mbarrier are baseline features on SM12x. Cluster multicast is not available.

- `cp.async.bulk.tensor`: "Requires sm_90 or higher." The multicast qualifier lists `sm_90a`, `sm_100*`, `sm_103*`, `sm_110*` and no SM12x target [DOC] [S1 §9.7.10.28.5.3].
- CUTLASS emits a CTA-local destination on SM120 (`.shared::cta.global`) instead of the cluster form [SRC] [`cute/arch/copy_sm90_tma.hpp:59-61`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/copy_sm90_tma.hpp#L59-L61).
- CUTLASS asserts a 1×1×1 cluster: "no programmatic multicast on this arch" [SRC] [`sm120_mma_builder.inl:84`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl#L84).
- CUTLASS's SM120 GEMM uses `PipelineTmaAsync`, 8 MMA warps and one mainloop load warp, 384 threads in total [SRC] [`cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp:115-125`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L115-L125).
- `setmaxnreg` (moving registers from the loader warp to the math warps) is available with `sm_120a` or `sm_120f` [DOC] [S1 §9.7.21.5].

### 2.4 Memory proxies and `fence.proxy.async`

Ordinary loads and stores use the "generic proxy". TMA uses the "async proxy". PTX: "Accessing the same memory location across multiple proxies needs a cross-proxy fence" [DOC] [S1 §9.7.10.28.2].

Two directions matter:

- **TMA write, then ordinary read.** No explicit fence is needed: "The completion of a `cp{.reduce}.async.bulk` operation is followed by an implicit generic-async proxy fence" [DOC] [S1 §9.7.10.28.2].
- **Ordinary access, then TMA access to the same location.** A fence is required. CUTLASS's epilogue does this before a TMA store: `fence_view_async_shared(); // ensure smem writes are visible to TMA` [SRC] [`cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp:736-737`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp#L736-L737). It also fences before releasing a consumed stage back to the producer [SRC] same file `:817`.

The second case is the one FlashInfer issue #7675 reports for the b12x GEMM (section 8.3).

### 2.5 Shared memory, occupancy and other limits

| Limit | Capability 10.0 | Capability 12.0 | Source |
|---|---|---|---|
| Shared memory per thread block | 227 KB | **99 KB** | [S2 §1.4.1.1]; [S3 Table 34] |
| Shared memory per SM | 228 KB | **128 KB** (tuning guide) or **100 KB** (programming guide) | [S2]; [S3] |
| Unified data cache | 256 KB | 128 KB (programming guide) | [S3] |
| Concurrent warps per SM | 64 | 48 | [S2]; [S3] |
| Thread blocks per SM | 32 | 32 (tuning guide) or 24 (programming guide) | [S2]; [S3] |
| Registers per thread | 255 | 255 | [S2] |
| Register file per SM | 64K 32-bit | 64K 32-bit | [S2] |

CUTLASS encodes the per-block limit as `sm120_smem_capacity_bytes = 101376` (99 × 1024), used for both SM120 and SM121 [SRC] [`cutlass/include/cutlass/arch/arch.h:46-48, 118-121`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/arch/arch.h#L46-L48).

**The per-SM disagreement is unresolved and it matters.** Under the 128 KB reading, two 64 KB blocks fit on one SM. Under the 100 KB reading, only one does. Kernels that read the limit at run time (b12x, FlashInfer's fa2) adapt. Kernels that assume a count (FA2's split heuristic hard-codes two blocks per SM) do not. Neither guide mentions capability 12.1.

**The 48-warp limit is a second ceiling.** A 128-thread block is 4 warps, so at most 12 blocks fit by warp count. A 384-thread block is 12 warps, so at most 4.

### 2.6 Compilation targets

| Target | First toolkit | Runs on | Adds |
|---|---|---|---|
| `sm_120` | CUDA 12.8 | 12.0, 12.1 | baseline: BF16/FP16 and legacy FP8 `mma.sync`, TMA, mbarrier, `fence.proxy.async` |
| `sm_120a` | CUDA 12.8 | 12.0 only | `.kind::*` MMA, FP6/FP4, sparse block-scaled FP4, new `ldmatrix` shapes, `setmaxnreg` |
| `sm_120f` | CUDA 12.9 | 12.0 and 12.1 | the same, except sparse `mxf4` / `mxf4nvf4` |
| `sm_121a` | CUDA 12.9 | 12.1 only | adds sparse `mxf4` / `mxf4nvf4` |

Source: PTX §11.1.2 and the programming guide's family table [DOC] [S1], [S3 Table 31].

**Practical rule [INF].** `sm_120f` is the target that covers both machines with every dense feature. FlashInfer follows it: capability 12.0 compiles as `compute_120f` on CUDA ≥ 12.9, and 12.1 as `compute_121a` [SRC] [`flashinfer/compilation_context.py:96-107`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/compilation_context.py#L96-L107). SGLang compiles 12.0 as `sm_120f` and 12.1 with the `a` suffix, with the comment "SM120/SM121 need separate cubins to avoid cudaErrorIllegalInstruction" [SRC] [`sglang/python/sglang/kernels/jit/utils/arch.py:83-102`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/jit/utils/arch.py#L83-L102).

### 2.7 The two machines

| | RTX PRO 6000 Blackwell | DGX Spark (GB10) | Source |
|---|---|---|---|
| Compute capability | 12.0 | 12.1 | [S4] |
| CUDA cores | 24,064 | 6,144 | third-party datasheet [S6]; NVIDIA [S5] |
| Streaming multiprocessors | 188 (derived: cores ÷ 128) | 48 (derived: cores ÷ 128) | [CALC]; pull requests also state 188 and 48 [S13], [S27] |
| Memory | 96 GB GDDR7 with ECC | 128 GB unified LPDDR5x, 256-bit interface | [S7]; [S5] |
| Memory bandwidth | 1,792 GB/s | 273 GB/s | [S7]; [S5] |
| L2 cache | 128 MB | 24 MB | [VAULT] `consumer-gpu-kernel-guide`; [VAULT] deviceQuery note |
| Host | x86-64, discrete | aarch64, integrated | [VAULT] |

The bandwidth ratio is 6.6×. The SM ratio is 3.9×. Section 11 builds on the difference between those two ratios.

---

## 3. The common anatomy of an SM12x attention kernel

Every kernel in section 4 is a variation on one loop. Knowing the loop makes the differences legible.

For each tile of query rows, and for each tile of KV columns:

1. **Stage K.** Move a K tile from GPU memory into shared memory, by TMA or by `cp.async`. For a paged cache, resolve the page first.
2. **Load fragments.** `ldmatrix` the Q and K tiles from shared memory into registers.
3. **QK multiply.** `mma.sync` produces the score tile S = QKᵀ in accumulator registers.
4. **Mask and online softmax.** Apply causal, window and sink terms. Update each row's running maximum and running sum, and rescale the output accumulated so far.
5. **Stage V** (often overlapped with step 3).
6. **PV multiply.** Convert the probabilities P to the operand type, then `mma.sync` accumulates O += PV.

After the last KV tile, normalize O and write it out. For split-KV, write an unnormalized partial O and a log-sum-exp per row, and let a second kernel merge.

**The design choices that distinguish kernels:**

| Choice | Options seen on SM12x |
|---|---|
| Staging | `cp.async` per thread; TMA issued by a math warp; TMA issued by a dedicated loader warp |
| Pipeline depth | 1 stage is the norm at 99 KB; 2 stages for small head dimensions |
| Warp roles | every warp does QK then PV; or separate QK warps and PV warps exchanging P through shared memory |
| Tile shape | from 16×64 for decode up to 128×128 for small-head-dimension prefill |
| Where P lives | registers (the usual choice), or shared memory when QK and PV warps are separate |
| Paged cache | per-row page lookup; per-thread lookup with contiguous tile; page id as a TMA coordinate |
| Decode parallelism | split along KV with an LSE merge; or one block per request |
| FP8 KV | convert to 16-bit after load; or multiply FP8 operands directly |

**Shared-memory arithmetic [CALC].** With 16-bit elements, one stage, and Q, K and V all in shared memory, the bytes are 2 × (tile_m × d + tile_n × d + tile_n × d_v):

| Head dim | Tile (m×n) | Bytes | Under 101,376? | Two per SM at 128 KB? |
|---|---|---|---|---|
| 64 | 128×128 | 49,152 | yes | yes |
| 128 | 128×64 | 65,536 | yes | no (131,072 needed) |
| 128 | 64×64 | 49,152 | yes | yes |
| 128 | 128×128 | 98,304 | yes | no |
| 256 | 64×64 | 98,304 | yes | no |
| 256 | 128×64 | 131,072 | **no** | — |
| 512 | 64×64 | 196,608 | **no** | — |

This table explains most of what follows: why head dimension 128 forces a choice between a wide tile and two blocks per SM, why head dimension 256 fills a block, and why head dimension 512 has no general solution in this layout.

---

## 4. Kernel-by-kernel internals

### 4.1 FlashAttention-2 (CUDA C++)

**Role.** vLLM's automatic choice on SM12x, through its bundled `vllm_flash_attn`.

- **MMA.** `SM80_16x8x16_F32F16F16F32_TN` or the BF16 variant, for any architecture ≥ 800 [SRC] [`csrc/flash_attn/src/kernel_traits.h:29-34`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/kernel_traits.h#L29-L34).
- **Staging.** `SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>`, sequenced with `cp_async_fence()` and `cp_async_wait<0>()`. No TMA anywhere [SRC] [`kernel_traits.h:125-129`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/kernel_traits.h#L125-L129); [`flash_fwd_kernel.h:261-409`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/flash_fwd_kernel.h#L261-L409).
- **Tile selection has no SM12x branch.** The only capability test is `bool is_sm8x = cc_major == 8 && cc_minor > 0` [SRC] [`csrc/flash_attn/src/flash_fwd_launch_template.h:251-265`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/flash_fwd_launch_template.h#L251-L265). Capability 12 fails it and takes the branch commented "good for H100, A100" [SRC] `:272`.

Tiles a capability-12 device receives [SRC] [`flash_fwd_launch_template.h:200-321`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/flash_fwd_launch_template.h#L200-L321):

| Head dim | Tile, warps | Shared memory [CALC] |
|---|---|---|
| 32, 64 | 128×128, 4 | — |
| 96, 128 | 128×64, 4 | 65,536 B at head dim 128 |
| 192 | 128×64, 8 | — |
| 256 | 64×64, 4 (chosen by a run-time shared-memory test) | 98,304 B |

For comparison, the `is_sm8x` branch (Ada, RTX 30/40 series) uses 128×32 for non-causal, with the comment "128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM", and 64×64 for causal [SRC] `:251-265`.

- **Split-KV.** `num_splits_heuristic(batch × heads × m_blocks, num_sm * 2, num_n_blocks, 128)`, with the comment "We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block" [SRC] [`csrc/flash_attn/flash_api.cpp:282-336`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/flash_api.cpp#L282-L336). It returns 1 if the work already fills 80% of that budget, otherwise the smallest split count within 85% of the best wave-fill efficiency. Split-KV tiles are 64×128 for head dimension 128, which is 81,920 B [CALC].
- **Paged KV.** Only in the split-KV kernel. Upstream requires `page_block_size % 256 == 0` [SRC] [`flash_api.cpp:614`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/flash_api.cpp#L614). The vLLM fork relaxes this to `% 16` with an added copy atom that gives each thread a contiguous tile, so "each thread need only fetch 1 page index per mainloop iteration" [SRC] fork [`csrc/flash_attn/flash_api.cpp:638`](https://github.com/vllm-project/flash-attention/blob/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2/csrc/flash_attn/flash_api.cpp#L638); fork [`kernel_traits.h:135-144`](https://github.com/vllm-project/flash-attention/blob/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2/csrc/flash_attn/src/kernel_traits.h#L135-L144).

**What vLLM actually passes [SRC]** [`vllm/v1/attention/backends/flash_attn.py:1378-1403`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L1378-L1403): one `flash_attn_varlen_func` call serves the whole batch, with the block table, `fa_version=2`, and `num_splits=0`. The fork's wrapper raises `NotImplementedError("FA2 does not support num_splits > 1")` [SRC] fork [`vllm_flash_attn/flash_attn_interface.py:251-252`](https://github.com/vllm-project/flash-attention/blob/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2/vllm_flash_attn/flash_attn_interface.py#L251-L252). With 0, the C++ heuristic runs only in the single-query GQA branch [INF] fork [`flash_api.cpp:757-762`](https://github.com/vllm-project/flash-attention/blob/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2/csrc/flash_attn/flash_api.cpp#L757-L762).

**Assessment [INF].** FA2 is correct and complete on SM12x, with split-KV decode and paged KV. Its tiles are inherited, not tuned, and its split heuristic assumes two blocks per SM, which holds only under the 128 KB reading and only for tiles of 64 KB or less.

### 4.2 FlashInfer `fa2` templates

**Role.** SGLang's default on SM12x, and the prefill half of vLLM's `FLASHINFER`.

- **Dispatch.** `determine_attention_backend` returns `"fa3"` only for capability major 9. Everything else, including SM12x, gets `"fa2"` [SRC] [`flashinfer/utils.py:717-730`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/utils.py#L717-L730).
- **MMA.** `mma_sync_m16n16k16_row_col_f16f16f32` [SRC] [`include/flashinfer/attention/prefill.cuh:1706-1717`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/prefill.cuh#L1706-L1717).
- **Staging.** `cp.async` only: `load_128b_async` per thread, predicated on `kv_idx < kv_len` [SRC] [`prefill.cuh:691-739`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/prefill.cuh#L691-L739).
- **Tile selection reads the device.** The dispatcher reads `cudaDevAttrMaxSharedMemoryPerMultiprocessor` and `cudaDevAttrMaxSharedMemoryPerBlockOptin`, sets two blocks per SM if two minimum tiles fit, and picks the largest `NUM_MMA_KV` that fits both shared memory and the register bound [SRC] [`prefill.cuh:5005-5107`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/prefill.cuh#L5005-L5107). A comment names SM120 as a tight-shared-memory part [SRC] [`prefill.cuh:5046-5050`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/prefill.cuh#L5046-L5050).
- **Query tile.** 128 rows if the packed query length exceeds 64 and head dimension is under 256; else 64; else 16 [SRC] [`include/flashinfer/utils.cuh:411-450`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/utils.cuh#L411-L450).
- **Decode split-KV.** `max_grid_size = blocks_per_sm × num_sm` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. If the batch already fills it, no split. Otherwise a binary search finds the minimum pages per chunk. Under CUDA graphs split is always on [SRC] [`include/flashinfer/attention/scheduler.cuh:74-99, 175-207`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/scheduler.cuh#L74-L99).
- **Prefill split-KV.** Two blocks per SM is hard-coded here [SRC] [`scheduler.cuh:795-799`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/scheduler.cuh#L795-L799).
- **Automatic upgrade to FMHAv2.** In ragged prefill planning, `"fa2"` is rewritten to `"fmha_v2"` on SM120/SM121 for multi-head attention (query heads equal KV heads), head dimension 64/128/256, sequence length ≥ 256, no window, no soft cap [SRC] [`flashinfer/prefill.py:7016-7044`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/prefill.py#L7016-L7044). This applies to the ragged wrapper, not the paged wrapper vLLM uses.

**Assessment [INF].** Unlike FA2, this kernel adapts its tile and its decode split to the device it finds. It is still `cp.async` with per-thread loads.

### 4.3 XQA decode (TensorRT-LLM port)

**Role.** The decode half of vLLM's `FLASHINFER` on SM12x, and of SGLang's `trtllm_mha`.

- **Gate.** JIT `supported_major_versions=[9, 10, 12]` [SRC] [`flashinfer/jit/xqa.py:140-142`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/xqa.py#L140-L142). The C++ lists both: `__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210` with `kMAX_SMEM_SIZE = (99u << 10)` [SRC] [`csrc/xqa/utils.cuh:49-51`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/utils.cuh#L49-L51).
- **Layout.** 8 warps, 256 threads, warp-specialized into two groups. The CTA tile is 256 KV tokens [SRC] [`csrc/xqa/mha.cu:76-102`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mha.cu#L76-L102). On 99 KB architectures the V tile is 32 tokens [SRC] [`mha.cu:114-117`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mha.cu#L114-L117).
- **MMA.** `mma.sync.aligned.m16n8k16` in FP16 or BF16 [SRC] [`csrc/xqa/mma.cuh:35-50`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mma.cuh#L35-L50).
- **Staging.** `cp.async` with a per-token page lookup, `pageIndices[absoluteTokenIdx / tokensPerPage]` [SRC] [`csrc/xqa/ldgsts.cuh:29-41`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/ldgsts.cuh#L29-L41); [`csrc/xqa/mhaUtils.cuh:42-43`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mhaUtils.cuh#L42-L43).
- **Split-KV.** `nbSubSeqPerSeq = min(max(1, multiProcessorCount / (batchSize * nbKHeads)), divUp(maxSeqLen, ctaTile.x))` [SRC] [`mha.cu:3123-3129`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mha.cu#L3123-L3129).
- **Head dimensions.** Multiples of 16 from 16 to 512 [SRC] [`flashinfer/jit/xqa.py:88-103`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/xqa.py#L88-L103). vLLM restricts XQA to 16–256 [SRC] [`vllm/v1/attention/backends/flashinfer.py:840-841`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L840-L841).
- **KV dtypes.** FP16/BF16, int8, FP8 e4m3, and packed NVFP4. NVFP4 is asserted SM12x-only [SRC] [`flashinfer/jit/xqa.py:73-80`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/xqa.py#L73-L80); [`flashinfer/xqa.py:396-399`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/xqa.py#L396-L399).
- **FP8 handling.** Cache words are converted with `cvt.rn.f16x2.e4m3x2`, then multiplied as 16-bit. The Hopper FP8 tensor-core kernel is selected only for major 9 [SRC] [`csrc/xqa/utils.cuh:747-762`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/utils.cuh#L747-L762); [`flashinfer/xqa.py:387-394`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/xqa.py#L387-L394).
- **Features.** Sliding window, sinks, and ragged multi-token queries for speculative verification [SRC] [`flashinfer/jit/xqa.py:108-131`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/xqa.py#L108-L131); [`flashinfer/xqa.py:259-262`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/xqa.py#L259-L262). Sinks are applied in-kernel only when the request is not split [SRC] [`mha.cu:2618`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mha.cu#L2618).
- **Limits.** No LSE output, so no decode context parallelism [SRC] [`flashinfer/decode.py:4901-4916`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/decode.py#L4901-L4916).

### 4.4 FMHAv2 prefill (TensorRT-LLM port)

**Role.** SGLang's `trtllm_mha` prefill on SM12x. Not wired into vLLM.

- **Gate.** `supported_major_versions=[9, 12]`; a dedicated SM120 module uses `[12]`. `fmha_v2_prefill_sm120` requires capability (12, 0) or (12, 1) and says "validated for SM120 and SM121" [SRC] [`flashinfer/jit/attention/modules.py:2606-2669`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/attention/modules.py#L2606-L2669); [`flashinfer/prefill.py:9915-9942`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/prefill.py#L9915-L9942).
- **Head dimensions.** 64, 128, 256, 512. Head dimension 512 uses 64-row steps and a "tiled noloop" kernel [SRC] [`flashinfer/jit/attention/fmha_v2/fmha_library.py:351-355, 1294`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/attention/fmha_v2/fmha_library.py#L351-L355).
- **FP8.** The ragged entry raises "FP8 (e4m3) is not yet supported for FMHAv2 on SM120 (Blackwell)" [SRC] [`flashinfer/prefill.py:10443-10448`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/prefill.py#L10443-L10448). A separate fixed-shape entry requires FP8 Q/K/V with BF16 output, contiguous layout, and equal sequence lengths [SRC] [`prefill.py:9905-9975`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/prefill.py#L9905-L9975).
- **Masks.** Sliding window is supported. Chunked-causal is rejected because "the non-warp-specialized 'tiled' kernels used on SM120 honor `sliding_window_size` only" [SRC] [`prefill.py:10410-10421`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/prefill.py#L10410-L10421).
- **Staging mechanism inside the generated kernels:** not read.

FMHAv2 is the only general prefill kernel on SM12x that handles head dimension 512.

### 4.5 b12x paged attention

**Role.** vLLM's opt-in `--attention-backend b12x`. Ported into FlashInfer from `lukealonso/b12x` at commit `6627d342` [SRC] [`flashinfer/experimental/b12x/attention/paged/__init__.py:69-79`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/__init__.py#L69-L79). All paths below are under `flashinfer/experimental/b12x/attention/paged/`.

**Structure.** 16 files; the decode kernel is 11,267 lines, the prefill kernel 7,621 lines, the planner 2,601 lines. Three modes: `decode` (every request has one query token), `verify` (multi-token speculative verification), `extend` (everything else) [SRC] [`planner.py:72, 788-792`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L72). A plan is a flat list of work items, each a (request, query tile, KV chunk). The decode grid is one block per (work item, KV head) [SRC] [`planner.py:1834-1842`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L1834-L1842); [`forward_paged.py:4178`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L4178).

**Support gate.** Capability major 12 with minor 0 or 1, `nvidia-cutlass-dsl >= 4.6.0`, and Triton [SRC] [`flashinfer/experimental/b12x/_lib/gating.py:14-31, 89-95`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/_lib/gating.py#L14-L31).

#### Decode and verify

- **Tile.** The query tile is forced to 16 rows, giving 1 query warp × 4 KV warps, 128 threads, and a 16×64 tile. Some specializations use 16×128 [SRC] [`planner.py:826-832`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L826-L832); [`traits.py:272-276, 464, 483`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L272-L276).
- **MMA.** Inline `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, issued twice to form a 16×16×16 step. Operands come through `ldmatrix` [SRC] [`_lib/intrinsics.py:3920-3985`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/_lib/intrinsics.py#L3920-L3985); [`forward_paged.py:1657-1682`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L1657-L1682).
- **Staging is TMA only.** The constructor raises otherwise: "PagedForwardKernel now only supports exact-plane paged K/V TMA decode" [SRC] [`forward_paged.py:3376-3382`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L3376-L3382).

How the paged cache becomes a TMA source [SRC]:

1. The cache `[num_pages, page_size, kv_heads, head_dim]` is re-viewed as `[page_size, head_dim, kv_heads, num_pages]`, so the page id is the last coordinate ([`forward_paged.py:4077-4083`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L4077-L4083)).
2. A tiled TMA atom covers `(stage_tile_rows, plane_head_dim)`. A plane is 128 bytes per row: 64 BF16 elements or 128 FP8 bytes. A head dimension is covered by 1 to 4 planes ([`forward_paged.py:3427-3444, 4107-4147`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L3427-L3444)).
3. One thread looks up the page and issues the copy:

```python
page_id = mPageTable[request_idx, page_idx]
pipeline_tma.producer_acquire(producer_state)
src_idx = self._tile_src_idx_from_page_id(page_id, sub_tile)
load_tma0(src_idx=src_idx, producer_state=producer_state)
```
   ([`forward_paged.py:3782-3789`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L3782-L3789))

4. A tile never crosses a page: the constructor requires `page_size % cta_tile_kv == 0` ([`forward_paged.py:3159-3163`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L3159-L3163)).

- **Who issues TMA.** By default no dedicated loader: warp 0 issues the next tile after releasing the consumed stage. A "role-specialized" variant adds a fifth warp that runs the loader loop, calls `setmaxregister_decrease(80)`, and lets the math warps raise theirs to 168 or 248 [SRC] [`forward_paged.py:5236-5335, 5598, 6350-6366`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L5236-L5335).
- **Stages.** One, whenever there are multiple KV warps or the cache is FP8 [SRC] [`forward_paged.py:3271-3285`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L3271-L3285).
- **Shared memory is read at run time.** `max_smem_per_sm` and `max_smem_per_block` come from the device. A candidate gets two blocks per SM if `2 × launch_bytes <= max_smem_per_sm`. Selection maximizes `(residency, num_mma_kv)`: occupancy first, tile width second [SRC] [`traits.py:204-211, 308, 329`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L204-L211).
- **Worked sizes [CALC].** BF16 decode at head dimension 128: 39,424 B. At head dimension 256: 73,728 B, which cannot be doubled on a 100 KB SM and can on a 128 KB SM only if the launch overhead is under about 9 KB.
- **Split-KV.** On under CUDA graphs with BF16 queries; always on for verify; never for extend [SRC] [`planner.py:2024-2039`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L2024-L2039). Capacity is `num_sms × graph_ctas_per_sm // num_kv_heads`, where `graph_ctas_per_sm` comes from a shape-keyed table of 1, 2, 4 or 6 [SRC] [`planner.py:227-228, 493-548`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L227-L228). Chunk budgets are table-driven by head dimension, GQA group, KV dtype and batch [SRC] [`planner.py:697-749`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L697-L749).
- **Merge.** A persistent kernel reduces normalized partial outputs with base-2 log-sum-exp scores. It supports only output head dimensions 128 and 256 [SRC] [`merge.py:1-14, 198-223`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/merge.py#L1-L14); [`planner.py:75-84`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L75-L84).
- **Query staging.** Predicated 16-byte `cp.async` [SRC] [`forward_paged.py:890-914`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L890-L914).

#### Extend (prefill)

- **Query tile.** 16 if the packed query length is ≤ 32, otherwise FlashInfer's FA2 rule (128, 64 or 16) [SRC] [`planner.py:795-801, 833-835`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L795-L801).
- **Staging is mostly `cp.async`.** TMA is enabled for one shape only: BF16, head dimension 256, 16-row query tile, page size 64. The comment: "Page-128 uses the byte-addressed copy path until extend TMA gets the decode kernel's page-entry flattening" [SRC] [`forward_extend_generic.py:3272-3290`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_extend_generic.py#L3272-L3290). Otherwise every thread does a per-row page-table lookup and a 16-byte `cp.async` [SRC] [`forward_extend_generic.py:3836-3870`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_extend_generic.py#L3836-L3870).
- **No split-KV** [SRC] [`planner.py:2038-2039`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L2038-L2039).

#### FP8, sinks, window, GQA

- **FP8 KV is converted, by default.** Bytes are TMA-staged raw, loaded with `ldmatrix`, converted in registers, then multiplied as BF16 [SRC] [`forward_paged.py:1680-1709`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L1680-L1709).
- **Native FP8 multiply is opt-in** through `B12X_TURBO_ATTN=1`, and for decode is further limited by batch and chunk thresholds [SRC] [`_forward.py:78-79, 164-192`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/_forward.py#L78-L79). The instruction is `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32` [SRC] [`_lib/intrinsics.py:4019, 4083`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/_lib/intrinsics.py#L4019).
- **Sinks.** `attention_sink_bias[q_head]` is folded into the running softmax state as an extra logit, by one warp, and under split-KV only by the chunk that owns the causal limit [SRC] [`forward_paged.py:977-1011`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L977-L1011).
- **Sliding window.** Converted to a per-request window start page by the planner [SRC] [`planner.py:198-213, 2070-2086`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L198-L213).
- **GQA packing.** Query rows for heads sharing one KV head are packed together. One block serves one KV head and all its query heads [SRC] [`forward_paged.py:3909-3920`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py#L3909-L3920).

#### Why head dimension 64 fails

vLLM excludes head dimension 64 "until flashinfer-ai/flashinfer#6977 is fixed" [SRC] [`vllm/v1/attention/backends/b12x.py:203-206`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/b12x.py#L203-L206). The issue title: "experimental/b12x paged attention cannot plan head_dim 64 decode: trait rule and decode kernel contradic[t]" [ISSUE] [S20].

The mechanism [INF from SRC]:

1. Head dimension 64 means `num_mma_d_vo = 4` ([`traits.py:199`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L199)).
2. Decode forces a 16-row query tile, and for that tile the candidate sweep tries only `num_mma_kv = 1` ([`traits.py:272-276`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L272-L276)).
3. A trait rule rejects `num_mma_d_vo == 4 and num_mma_kv % 2 == 1` ([`traits.py:88-105`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L88-L105)).
4. The sole candidate is rejected, raising "no valid NUM_MMA_KV typed shared-memory allocation fits the device" ([`traits.py:319-322`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L319-L322)).

Independently, the split-KV merge requires output head dimension 128 or 256.

#### SM121 and fences

- **Tile specializations are keyed on exactly (12, 0).** Five trait specializations and one planner budget test `device_capability == (12, 0)`; two use `>= 12` [SRC] [`traits.py:383, 411, 440, 459, 468, 486, 533`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L383); [`planner.py:272`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L272). On a Spark the generic sweep is used [INF].
- **No `fence_proxy` call appears in the paged attention files** [SRC, by grep]. Fences are used elsewhere in b12x. Section 8.3 discusses whether that is a risk.

#### How vLLM drives b12x

[SRC] [`vllm/v1/attention/backends/b12x.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/b12x.py), [`vllm/utils/b12x.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/utils/b12x.py):

- vLLM imports a top-level `b12x.attention.paged` module. There is no version comparison; the "requires flashinfer-python>=0.7.2" message is returned when the import fails ([`b12x.py:245-250`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/b12x.py#L245-L250)).
- Page sizes 64 and 128 only; plans are built for both because the final page size is not known at construction (`:51-52, 402-405`).
- Decode plans are built for each CUDA-graph capture size. Verify plans for every batch size when speculative decoding is on. Extend plans for every batch × capacity bucket, the buckets being 128, 512, 1024, 2048, 4096 and the maximum (`:426-437, 500-504, 687-704`).
- Only the extend entry is registered for CuTe-DSL warm-up compilation (`:819-839`).
- Raised restrictions, verbatim: "b12x does not support ALiBi.", "b12x does not support logits soft cap.", "b12x currently supports decoder self-attention only.", "b12x currently requires canonical softmax scale head_dim**-0.5", "b12x does not yet support decode/prefill context parallelism.", "b12x currently requires bfloat16 queries.", "b12x supports causal attention only." (`:349-411, 978-1075`).

### 4.6 Upstream FlashAttention-4 on SM120 (main)

**Role.** Not reachable from vLLM. Documented here because it is the base that open pull requests extend.

- **It is a subclass of the Ampere kernel.** `FlashAttentionForwardSm120(FlashAttentionForwardSm80)` overrides only `can_implement()` and forces `self.arch = Arch.sm_80` [SRC] [`flash_attn/cute/flash_fwd_sm120.py:4-19`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd_sm120.py#L4-L19). The header: "SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has a smaller shared memory capacity (99 KB vs 163 KB on SM80)".
- **Why the arch override exists.** Without it, a DGX Spark reported `Arch.sm_121a` and the kernel crashed with `tma_atom_O=None` [PR #2336].
- **Fixed launch configuration.** `num_stages=1, num_threads=128, Q_in_regs=False` are hard-coded in the dispatch [SRC] [`flash_attn/cute/interface.py:1484-1502`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/interface.py#L1484-L1502).
- **Tiles.** 128×128 for head dimension ≤ 64, and 128×64 above, with the comment "128×64 avoids the occupancy loss from a 96 KB tile for larger head dimensions" [SRC] [`flash_attn/cute/config.py:644-666`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/config.py#L644-L666).
- **Staging.** `cp.async` with a 128-bit copy atom. With one stage, the V load overlaps the QK multiply and the next K load overlaps softmax and the PV multiply [SRC] [`flash_attn/cute/flash_fwd.py:246-253, 1125-1205`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd.py#L246-L253).
- **Head dimension 256 does not fit.** The kernel's own formula gives 131,072 B for a 128×64 tile [CALC from [`flash_fwd_sm120.py:47-58`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd_sm120.py#L47-L58)]. `can_implement` is not called for the forward path; the dispatch carries `# TODO: check @can_implement` [SRC] [`interface.py:1507`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/interface.py#L1507). So the call should pass host validation and fail at launch [INF]. Three pull requests state that main cannot run head dimension 256 on SM120 [PR #2980, #2998, #3109].

Restrictions when the capability major is 12 [SRC]:

| Feature | Restriction | Citation |
|---|---|---|
| `num_splits` | `ValueError("SM120 forward only supports num_splits=1")` | [`config.py:1542-1543`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/config.py#L1542-L1543) |
| Paged KV, block sparsity | `ValueError("SM120 forward does not support paged KV or block sparsity")` | [`config.py:1887-1888`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/config.py#L1887-L1888) |
| FP8 | `assert arch // 10 == 10, "FP8 is only supported on SM100..."` | [`interface.py:825`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/interface.py#L825) |
| Learnable sink | `assert learnable_sink is None` | [`flash_fwd.py:651`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd.py#L651) |
| MLA (`qv`) | `assert arch // 10 in [10, 11]` | [`interface.py:866`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/interface.py#L866) |

Working on SM120: dense, variable-length, causal, local window, and packed GQA forward; and a backward kernel with fixed 64×64 tiles [SRC] [`interface.py:1492-1493`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/interface.py#L1492-L1493); [`config.py:3049-3064`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/config.py#L3049-L3064).

**The paged exclusion is "not implemented", not a hardware limit [INF].** Upstream's paged manager is `cp.async`-based and SM120 supports `cp.async`. Pull request #3060 makes that argument.

**For contrast, the kernels SM12x cannot use [SRC]:**

- **Hopper ([`flash_fwd_sm90.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd_sm90.py)).** `wgmma` with a 64-row atom; one producer warp group plus one to three MMA warp groups; TMA through `PipelineTmaAsync`; two K/V stages; 128×128 at head dimension 128; 227 KB budget.
- **Data-center Blackwell ([`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd_sm100.py)).** `tcgen05` with accumulators in TMEM; 512 threads in role-specialized warps (softmax, correction, one MMA warp, one load warp); K/V stages chosen to fill 224 KB; an optional two-CTA mode; and a separate "SwapAB" single-token decode kernel that puts KV tokens on the 128-row MMA dimension.

### 4.7 SGLang's `fa4_sm120` kernels

**Role.** SGLang's opt-in `--attention-backend fa4` on SM12x. Added by pull request #32991, merged 2026-08-13 [S27]. All paths under `python/sglang/kernels/ops/attention/fa4_sm120/`.

- **Language.** CuTe DSL. The forward kernel is 4,554 lines; a separate transposed decode kernel and a 56 KB run-time policy file sit beside it [SRC] directory listing.
- **Dispatch.** `arch // 10 == 12`, so SM121 takes the same family [SRC] [`dispatch.py:13-22`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/dispatch.py#L13-L22).
- **MMA.** `warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16))` for both multiplies [SRC] [`flash_fwd.py:970-991`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L970-L991). Class docstring: "SM120 warp-MMA forward kernel with TMA fused into the QK warps" [SRC] [`flash_fwd.py:72`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L72).
- **Stages.** Always one K stage [SRC] [`flash_fwd.py:691-696`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L691-L696).
- **Two warp organizations.** For head dimension 256, dedicated QK warps and dedicated PV warps exchange P through shared memory, with 2 P stages. For every other head dimension, each warp does QK then PV [SRC] [`flash_fwd.py:736-757, 782-786`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L736-L757).
- **Threads.** 128 by default; 64 to 256 for head dimension 256 depending on the tile; plus one 32-thread loader warp when paged [SRC] [`flash_fwd.py:698-734, 797-802`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L698-L734).

**Staging [SRC]:**

- **Contiguous K/V: TMA** ([`flash_fwd.py:1193-1223, 1456-1472`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L1193-L1223)).
- **Paged K/V: `cp.async` gather** by a dedicated 32-thread loader group. Each loader thread does a fast divmod by the page size, reads the page table, and builds a pointer. Any page size works, including 1 ([`flash_fwd.py:1473-1504`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L1473-L1504); [`paged_kv.py:58-147`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/paged_kv.py#L58-L147)).
- **Paged TMA exists in one specialization**, the transposed decode kernel, selected only when the page size equals the 64-row tile, head dimension is 256, the request is single-token, and the packed GQA group is 2 to 8 rows ([`flash_fwd_decode.py:41`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd_decode.py#L41); [`runtime.py:919-932`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/runtime.py#L919-L932)).

**Tile candidates [SRC]** [`flash_fwd.py:399-689`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L399-L689):

| Head dim | Candidate order (m×n) |
|---|---|
| 64 or 128, paged single-token decode | 16×64 or 32×64 |
| 256, ≤ 16 packed query rows | 16×64, 16×80, 32×64, 64×64 |
| 256, otherwise | 64×64, 64×48, 48×64, 32×64 |
| 32, 64, 96, 128, 192/128 | 64×128 or 64×64 |
| other | 128×128 (head dim ≤ 64), 128×64, 64×64 |

For batch 1 with packed GQA, the KV tile width is chosen analytically from the SM count. A "compact SM array" branch is `num_sms <= 64`, which a 48-SM Spark takes [SRC] [`flash_fwd.py:282-360`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L282-L360).

**Shared memory [CALC from [`flash_fwd.py:362-397`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L362-L397)].** Head dimension 256 at 64×64 comes to exactly 101,376 B, the whole block. Head dimension 512 has no fitting candidate and raises "exceeds SM120 shared-memory capacity" [SRC] [`flash_fwd.py:686-689`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L686-L689).

**Split-KV [SRC]** [`dispatch.py:25-38`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/dispatch.py#L25-L38); [`runtime.py:157-368, 670-749`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/runtime.py#L157-L368):

- Prefill: always one split.
- Decode: automatic, unless deterministic inference is requested.
- For head dimension 256, a fitted latency and bandwidth model picks the split count. Its constants are a "Joint fit of interleaved NCU duration curves from the 110-SM/384-bit and 188-SM/512-bit SM120 SKUs" ([`runtime.py:45-48`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/runtime.py#L45-L48)).
- The hardware probe reads `is_integrated`, memory bus width and clock rate, and has integrated-GPU branches. One returns a single split with the comment "Three unsplit CTAs per LPDDR channel already saturate the LPDDR path" ([`runtime.py:125-146, 245-248`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/runtime.py#L125-L146)).

**Page size is not forced on SM12x.** SGLang's forcing of page size 128 for FA4 is gated on `get_platform().is_sm100` [SRC] [`python/sglang/srt/arg_groups/overrides.py:1486-1510`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/overrides.py#L1486-L1510). On SM12x the page size stays at the default of 1, which runs the gather kernel. `--page-size 64` unlocks the page-TMA decode specialization [INF].

**Feature support on SM12x [SRC]:**

| Feature | Behaviour | Citation |
|---|---|---|
| Attention sinks | supported (`learnable_sink`, BF16) | [`flash_attention_v4_sm120.py:118-119`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attention_v4_sm120.py#L118-L119) |
| Sliding window | supported | [`flash_fwd.py:1082-1083`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L1082-L1083) |
| GQA packing | default on when query heads exceed KV heads | [`interface.py:686-688`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py#L686-L688) |
| FP8 KV | rejected: `assert arch // 10 == 10` | [`interface.py:787-790`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py#L787-L790) |
| NVFP4 KV | FA4 excluded; resolver forces FlashInfer prefill and XQA decode | [`kv_cache_hook.py:28-93`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/kv_cache_hook.py#L28-L93) |
| MLA | rejected | [`interface.py:1084-1085`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py#L1084-L1085) |
| Hybrid GDN models | FA4 not in the allowed set | [`attention_registry.py:536-550`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/attention_registry.py#L536-L550) |
| Speculative decoding top-k > 1 with page size > 1 | rejected | [`speculative_hook.py:1464-1475`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/speculative_hook.py#L1464-L1475) |
| Head dimension 512 | raises | [`flash_fwd.py:686-689`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py#L686-L689) |

### 4.8 Triton attention

Both engines offer it on any capability. vLLM's `TRITON_ATTN` supports sinks, non-causal attention, multimodal prefixes and FP8 KV, and reports CUDA-graph support `ALWAYS` [SRC] [`vllm/v1/attention/backends/triton_attn.py:97, 383-406`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_attn.py#L97). Two SM12x-specific details:

- vLLM uses 64 parallel softmax segments for small batches, keyed on **exactly** (12, 0); not applied on a Spark [SRC] [`triton_attn.py:152-162`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_attn.py#L152-L162).
- SGLang uses smaller Triton tiles on SM12x: 64×128 for head dimension ≤ 128, 64×64 for ≤ 256, else 32×32 [SRC] [`sglang/python/sglang/kernels/ops/attention/extend_attention.py:101-108`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/extend_attention.py#L101-L108).

### 4.9 Side-by-side

| Kernel | MMA | K/V staging | Stages | Threads | Decode split-KV | Paged | FP8 KV | Reads device limits |
|---|---|---|---|---|---|---|---|---|
| FA2 | `m16n8k16` | `cp.async` | 1 | 128 (256 at head dim 192) | yes; assumes 2 blocks per SM | yes | no | only at head dim 256 |
| FlashInfer fa2 | `m16n16k16` | `cp.async` | — | 128 | yes; from occupancy query | yes | yes, 16-bit query | yes |
| XQA | `m16n8k16` | `cp.async`, per-token page lookup | — | 256 | yes; from SM count | yes | yes, converted | 99 KB constant |
| FMHAv2 | "HMMA" per comments | not read | — | — | not read | optional | ragged: no | — |
| b12x decode | `m16n8k16`; opt-in native FP8 | **TMA**, page id as coordinate | 1 | 128 or 160 | yes, under CUDA graphs | yes, 64/128 | yes, converted by default | yes |
| b12x extend | same | `cp.async`; TMA for one shape | 1–2 | 128 | no | yes | yes | yes |
| Upstream FA4 SM120 | `m16n8k16` | `cp.async` | 1 | 128 | no | no | no | 99 KB constant |
| SGLang `fa4_sm120` | `m16n8k16` | **TMA** contiguous; `cp.async` gather paged; TMA for one paged decode shape | 1 | 64–256, +32 loader | yes, fitted model | yes, any page size | no | 99 KB constant; SM count; integrated flag |

---

## 5. How vLLM drives these kernels

All citations in this section are to vLLM `main` @ `f39d833`.

### 5.1 Selection

- **Flags.** `--attention-backend X`, or `--attention-config.backend X` / `-ac.backend X`. Setting both raises "attention_backend and attention_config.backend are mutually exclusive" [SRC] [`vllm/engine/arg_utils.py:1121, 1918, 2764-2774`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/engine/arg_utils.py#L1121). Names are case-insensitive [SRC] [`vllm/config/attention.py:273-285`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/config/attention.py#L273-L285).
- **A named backend** is validated alone; failure raises `"Selected backend {x} is not valid for this configuration. Reason: {...}"` [SRC] [`vllm/platforms/cuda.py:471-490`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/cuda.py#L471-L490).
- **Automatic selection** walks a priority list and takes the first backend with no validation objection [SRC] [`cuda.py:417-458, 520-523`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/cuda.py#L417-L458).
- **Priority for capability major 12 [SRC]** [`cuda.py:131-135, 166-178`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/cuda.py#L131-L135):
  - Standard attention: `FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, `FLEX_ATTENTION`, `TURBOQUANT`.
  - MLA: `TRITON_MLA`, `FLASHINFER_MLA_SPARSE_SM120`.
- **`B12X` is in no list** [SRC] [`cuda.py:81-178`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/cuda.py#L81-L178).

Validation runs 22 checks in order: head size, model dtype, KV dtype, block size, multimodal prefix, MLA, sinks, sparse, per-head scales, compute capability, attention type, sliding window, and so on, ending with a backend-specific `supports_combination` [SRC] [`vllm/v1/attention/backend.py:296-389`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backend.py#L296-L389).

### 5.2 `FLASH_ATTN` on capability 12

- **Version.** FA3 if major 9; FA4 if major 10; otherwise FA2 [SRC] [`vllm/v1/attention/backends/fa_utils.py:91-100`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/fa_utils.py#L91-L100).
- **What FA2 cannot serve, and the reason string [SRC]:**
  - **Sinks.** `supports_sink()` is true only for FA3 or FA4 ([`flash_attn.py:422-426`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L422-L426); [`fa_utils.py:244-247`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/fa_utils.py#L244-L247)).
  - **FP8 KV.** "FP8 KV requires FA3 (SM90), equal-d512 FA4 (SM90), or FA4 (SM100)." ([`flash_attn.py:464-474`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L464-L474)).
  - **Multimodal prefix.** Requires FA4 ([`flash_attn.py:418-420`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L418-L420)).
- **What it serves:** sliding window, non-causal attention, batch invariance, decode context parallelism, head sizes up to 256 that are multiples of 8, block sizes that are multiples of 16 [SRC] [`flash_attn.py:332-406, 1038`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L332-L406).
- **CUDA graphs.** FA2 reports `UNIFORM_BATCH`, not `ALWAYS`, because FA2 has special packed-GQA handling for single-token queries [SRC] [`flash_attn.py:578-597`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L578-L597).

**Consequence [INF].** On SM12x, automatic selection yields FA2 for a plain BF16 model and `FLASHINFER` as soon as the model needs FP8 or NVFP4 KV, sinks, or a multimodal prefix.

**Forcing FA4.** The gate is in vLLM's tree: "FA4 is only supported on devices with compute capability 9.x, 10.x, or 11.x" [SRC] [`vllm/vllm_flash_attn/flash_attn_interface.py:72-86`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/vllm_flash_attn/flash_attn_interface.py#L72-L86). With `--attention-config.flash_attn_version=4` on capability 12, `get_flash_attn_version` logs "FlashAttention 4 is not supported (...); it will not be used.", fails an assertion that is caught, and returns `None` [SRC] [`fa_utils.py:189-199`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/fa_utils.py#L189-L199). Validation does not require a version, so the backend is still selected, and the first forward pass fails on `"FlashAttention version not detected."` [INF from SRC] [`flash_attn.py:1201-1203`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L1201-L1203). Expected outcome: a startup failure, not a silent fallback. Not run.

### 5.3 `FLASHINFER` on SM12x

**Capability window.** `>= (8, 0)` and `<= (12, 1)` [SRC] [`vllm/v1/attention/backends/flashinfer.py:779-781`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L779-L781).

**Prefill.** `supports_trtllm_attention(is_prefill=True)` is false on the SM12x family, so prefill always uses `BatchPrefillWithPagedKVCacheWrapper(..., backend="auto")` [SRC] [`vllm/utils/flashinfer.py:884-887`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/utils/flashinfer.py#L884-L887); [`flashinfer.py:1573-1581`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1573-L1581). In FlashInfer, `"auto"` resolves to `"fa2"` for any capability major other than 9 [SRC] [`flashinfer/utils.py:717-730`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/utils.py#L717-L730). vLLM's own comments call SM12x prefill "fa2-only" [SRC] [`flashinfer.py:1276-1277`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1276-L1277).

**Decode uses XQA only if all of these hold [SRC]:**

1. `--attention-config.use_trtllm_attention` is not false.
2. `VLLM_BATCH_INVARIANT` is unset.
3. **`has_nvidia_artifactory()` is true.** That is: the `flashinfer_cubin` package is importable, or `VLLM_HAS_FLASHINFER_CUBIN=1`, or an HTTP request to the cubin repository returns 200 within 5 seconds ([`vllm/utils/flashinfer.py:61-64, 85-92, 840-864`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/utils/flashinfer.py#L61-L64)).
4. Query heads are a multiple of KV heads.
5. Head dimension is 16 to 256 in steps of 16 ([`flashinfer.py:840-841`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L840-L841)).
6. Decode context parallelism is off ([`flashinfer.py:1122-1132`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1122-L1132)).

Otherwise decode uses `BatchDecodeWithPagedKVCacheWrapper` with tensor cores and `backend="auto"` [SRC] [`flashinfer.py:1626-1645`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1626-L1645).

**Consequence [INF].** On an air-gapped Spark without the cubin package, XQA is silently off. Sinks then become unsupported, because sink support on SM12x is defined as XQA being available [SRC] [`flashinfer.py:783-796`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L783-L796), and CUDA-graph support drops from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE` [SRC] [`flashinfer.py:1309-1352`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1309-L1352).

**Query dtype on SM12x [INF from SRC]** [`flashinfer.py:1264-1305`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1264-L1305): the model dtype in every case. With an FP8 cache, "Architectures with only fa2 (e.g. SM89, SM120) cannot consume FP8 queries". With XQA, "requires BF16/FP16-Q even with FP8 KV cache".

**Other SM12x specifics [SRC]:**

- Block sizes 16, 32, 64. Sizes of 128 and above are advertised only on the SM100 family ([`flashinfer.py:595-612`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L595-L612)).
- NVFP4 KV: only the literal `nvfp4` dtype; head size 64 and sinks are refused; decode context parallelism and non-causal attention raise ([`flashinfer.py:727-759, 1048-1051, 1517-1521`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L727-L759)).
- Multimodal prefix: served natively by `BatchPrefillWithCausalBidirectionalRangesWrapper` ([`flashinfer.py:1585-1603`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1585-L1603)).
- Speculative verification: ragged multi-token queries stay on the XQA decode path with a packed draft mask ([`flashinfer.py:1465-1506, 1731-1743`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1465-L1506)).
- Batch invariance: `FLASHINFER` fails validation ([`backend.py:221-223`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backend.py#L221-L223)).

### 5.4 Multimodal-prefix models on capability 12

Both composite backends are invalid: `TRITON_FLASH_ATTN` requires major 9 and `TRITON_FLASHINFER` requires major 10 [SRC] [`triton_flash_attn.py:55-61`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_flash_attn.py#L55-L61); [`triton_flashinfer.py:12-22`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_flashinfer.py#L12-L22). The resulting chain [INF from SRC]:

1. `FLASH_ATTN` fails (needs FA4).
2. `FLASHINFER` serves it, unless the model uses soft-capped logits, an unsupported head size, or sinks.
3. Otherwise `TRITON_ATTN`.

### 5.5 Log lines to grep [SRC]

| Situation | Format string | Location |
|---|---|---|
| Named backend | `"Using %s backend."` | [`cuda.py:489`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/cuda.py#L489) |
| Automatic | `"Using %s attention backend out of potential backends: %s."` | [`cuda.py:549-557`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/cuda.py#L549-L557) |
| FlashAttention version | `"Using FlashAttention version %s"` | [`flash_attn.py:1083-1086`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py#L1083-L1086) |
| FlashInfer kernels | `"FlashInfer resolved query dtypes: prefill=%s, decode=%s, decode_backend=%s, kv_cache_dtype=%s, arch=%s"` | [`flashinfer.py:1216-1224`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py#L1216-L1224) |
| b12x | `"Using b12x with q_heads=%d kv_heads=%d head_dim_qk=%d ... planned_page_sizes=%s ... scratch=%d bytes."` | [`b12x.py:718-731`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/b12x.py#L718-L731) |

In the FlashInfer line, `decode_backend` is `xqa`, `trtllm-gen` or `flashinfer-native`, and `arch` prints `sm120` or `sm121`.

---

## 6. How SGLang drives these kernels

All citations in this section are to SGLang `main` @ `b2cc023f`.

### 6.1 Defaults

- **Standard attention on SM12x:** `flashinfer`, or `triton` if the model has attention sinks. The source comment: "FlashInfer does not support attention sinks." [SRC] [`python/sglang/srt/arg_groups/model_override_base.py:367-371`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/model_override_base.py#L367-L371).
- **MLA on SM12x:** `triton` [SRC] [`model_override_base.py:372-388`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/model_override_base.py#L372-L388).
- For comparison: SM90 gets `fa3`; SM100 gets `trtllm_mha`, or `fa4` for asymmetric KV.

### 6.2 The `flashinfer` backend

Prefill and decode both use `backend="fa2"` on the paged wrappers. Ragged prefill uses `"auto"`. The only architecture branch is for SM100 [SRC] [`python/sglang/srt/layers/attention/flashinfer_backend.py:373-374, 550-616`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/flashinfer_backend.py#L373-L374).

### 6.3 `trtllm_mha`

- **Kernels.** FMHAv2 for prefill, XQA for decode, both gated on `get_platform().is_sm90 or get_platform().is_sm120` [SRC] [`python/sglang/srt/layers/attention/trtllm_mha_backend.py:6-12, 309, 417`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/trtllm_mha_backend.py#L6-L12).
- **Prefill restriction,** verbatim: "TRTLLM FMHAv2 prefill on SM120 does not support fp8_e4m3 KV cache or skip-softmax." The check is on prefill only; XQA decode with FP8 KV is not rejected [SRC] [`python/sglang/srt/arg_groups/attention_hook.py:216-230`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/attention_hook.py#L216-L230).
- **Page size.** 16, 32, 64 or 128; anything else snaps to 64. The default of 1 is therefore changed to 64 with a warning [SRC] [`python/sglang/srt/arg_groups/overrides.py:1387-1400`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/overrides.py#L1387-L1400).
- **Speculative decoding.** Top-k must be 1 [SRC] [`python/sglang/srt/arg_groups/speculative_hook.py:1369-1373`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/speculative_hook.py#L1369-L1373).
- **Prefill copies pages.** FMHAv2 derives strides from shapes, so SGLang first copies the batch's pages into a `[pages, 2, page_size, heads, head_dim]` tensor [SRC] [`trtllm_mha_backend.py:1565-1582`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/trtllm_mha_backend.py#L1565-L1582).

### 6.4 Per-phase flags

`--prefill-attention-backend` and `--decode-attention-backend` override `--attention-backend` for their phase. Differing phases get a hybrid wrapper [SRC] [`python/sglang/srt/arg_groups/fields/exec_.py:237-275`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/fields/exec_.py#L237-L275); [`model_override_base.py:154-168`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/model_override_base.py#L154-L168).

Combinations rejected on SM12x [SRC]:

| Combination | Result | Citation |
|---|---|---|
| `fa3`, either phase | AssertionError | [`attention_registry.py:311-318`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/attention_registry.py#L311-L318) |
| `trtllm_mha` prefill + FP8 KV or skip-softmax | ValueError | [`attention_hook.py:216-230`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/attention_hook.py#L216-L230) |
| `trtllm_mha` + speculative top-k > 1 | ValueError | [`speculative_hook.py:1369-1373`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/speculative_hook.py#L1369-L1373) |
| `fa4` + FP8 KV | AssertionError at first call | [`flash_attn/cute/interface.py:787-790`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py#L787-L790) |
| `fa4` + top-k > 1 + page size > 1 | ValueError | [`speculative_hook.py:1464-1475`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/speculative_hook.py#L1464-L1475) |
| `--kv-cache-dtype nvfp4` + explicit prefill other than `flashinfer` | ValueError | [`kv_cache_hook.py:61-79`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/kv_cache_hook.py#L61-L79) |
| `--kv-cache-dtype nvfp4` + explicit decode other than `trtllm_mha` | ValueError | [`kv_cache_hook.py:81-87`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/kv_cache_hook.py#L81-L87) |
| Hybrid GDN model + `fa4` | AssertionError | [`attention_registry.py:536-550`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/attention_registry.py#L536-L550) |
| Deterministic inference + `trtllm_mha` | ValueError | [`overrides.py:1288-1306`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/overrides.py#L1288-L1306) |

`fa4` prefill with `trtllm_mha` decode passes every argument-level check for a plain model with a BF16 cache. Run-time compatibility of the shared cache layout was not verified [INF].

### 6.5 Two helpers with the same name and different meanings

| Call | True on SM120 | True on SM121 | Citation |
|---|---|---|---|
| `get_platform().is_sm120` (maps to `is_sm120_supported`) | yes | **yes** | [`runtime_context.py:1855`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/runtime_context.py#L1855); [`utils/device.py:311-315`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/utils/device.py#L311-L315) |
| `device.is_sm120()` | yes | **no** | [`utils/device.py:355-358`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/utils/device.py#L355-L358) |
| `device.is_sm121()` | no | yes | [`utils/device.py:361-365`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/utils/device.py#L361-L365) |
| `is_blackwell_supported()` | yes | yes | [`utils/device.py:301-308`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/utils/device.py#L301-L308) |

Section 10 lists the call sites where this changes behaviour on a Spark.

---

## 7. MLA and sparse attention

MLA (multi-head latent attention) stores a compressed latent per token. Sparse variants attend only to a top-k subset of the context chosen by an indexer. Neither uses the kernels in section 4.

### 7.1 vLLM

| Backend | Gate | KV dtypes | Block sizes | FlashInfer call | Citation |
|---|---|---|---|---|---|
| `TRITON_MLA` | any | `auto`, FP16, BF16, FP8 | multiple of 16 | none | [`mla/triton_mla.py:124-165`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/triton_mla.py#L124-L165) |
| `FLASHINFER_MLA_SPARSE_SM120` | major 12 | `auto`, `fp8`, `fp8_e4m3`, `fp8_ds_mla` | multiple of 64 | `trtllm_batch_decode_with_kv_cache_mla` | [`mla/flashinfer_mla_sparse.py:195-225`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py#L195-L225) |
| `FLASHINFER_MLA_SPARSE_DSV4` | major 10 or 12 | on major 12: FP8 variants only | 256 | `trtllm_batch_decode_sparse_mla_dsv4` | [`models/deepseek_v4/nvidia/flashinfer_sparse.py:118-166`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py#L118-L166) |
| `FLASHINFER_MLA_SPARSE_DSV41` | major 10 or 12 | on major 12: FP8 variants only | 64 on SM12x, else 128 | same | [`models/deepseek_v41/nvidia/flashinfer_sparse.py:66-116`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py#L66-L116) |

Details [SRC]:

- **Dense MLA resolves to `TRITON_MLA`; sparse MLA to `FLASHINFER_MLA_SPARSE_SM120`** [INF from the sparse-flag check].
- **The SM120 sparse backend always stores an FP8-packed cache.** `auto`, `fp8` and `fp8_e4m3` are canonicalized to `fp8_ds_mla` ([`model_executor/layers/attention/mla_attention.py:383-388`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/model_executor/layers/attention/mla_attention.py#L383-L388)).
- **It requires `index_topk == 2048`** ([`mla/flashinfer_mla_sparse.py:239-272`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py#L239-L272)) and 64-row pages: "The SM120 kernels are built for 64-row pages only." ([`mla/flashinfer_mla_sparse_sm120.py:210-211`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py#L210-L211)).
- **DeepSeek V4 and V4.1 are model-driven.** With no explicit backend, major 12 returns the FlashInfer SM120 class ([`models/deepseek_v4/nvidia/model.py:556-579`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/models/deepseek_v4/nvidia/model.py#L556-L579)).
- **MLA prefill off SM100 is FlashAttention only**, which on SM12x means FA2 with V zero-padded to the QK head dimension ([`platforms/interface.py:472-492`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/interface.py#L472-L492); [`mla/prefill/flash_attn.py:537-546`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/prefill/flash_attn.py#L537-L546)).
- **`backend="cake"` and `kv_cache_format="nvfp4"` are not used on vLLM's SM12x MLA paths.** The NVFP4 MLA cache dtype belongs to a backend gated on major 10 ([`mla/flashinfer_mla_sparse.py:99-122`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py#L99-L122)). FlashInfer ships these SM12x kernels; vLLM `main` does not call them.

### 7.2 SGLang

- **`trtllm_mla`**: the argument layer admits SM12x ("only supported on Blackwell GPUs (SM100/SM12x)") [SRC] [`overrides.py:1417-1431`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/overrides.py#L1417-L1431). Issue #40286 reports the underlying call failing on SM121 with `TllmGenFmhaRunner ... Unsupported architecture` [ISSUE]. Dense `trtllm_mla` likely fails the same way [INF]; not verified.
- **DeepSeek V4**: the model override forces `attention_backend = "dsv4"` and page size 256. On SM12x the sparse call goes to `flash_mla_with_kvcache_sm120` [SRC] [`deepseek_v4_backend.py:4578-4582`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L4578-L4582).
- **`SGLANG_SM120_FLASHMLA_BACKEND`**: `"flashinfer"` (default), `"triton"`, or anything else, which falls through to a pure-PyTorch path. The variable is not validated, so a typo silently selects PyTorch [SRC] [`environ.py:1447-1448`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/environ.py#L1447-L1448); [`flash_mla_sm120.py:298-374`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_mla_sm120.py#L298-L374).
- **GLM sparse attention with FP8 KV** selects `flashinfer_sparse_mla`, valid only for two `GlmMoeDsa*` model classes [SRC] [`overrides.py:688-722`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/overrides.py#L688-L722); [`flash_mla_sm120.py:29-32`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_mla_sm120.py#L29-L32).

### 7.3 FlashInfer's SM12x sparse kernels

| Kernel | MMA | Staging | SM121 | Citation |
|---|---|---|---|---|
| Cake DeepSeek-V4 NVFP4 sparse MLA | BF16 k16, FP8 k32, `mxf4nvf4` k64 | `cp.async` ring; TMA-free | README says SM120 / SM121 | [`csrc/cake_dsv4/README.md:107-211`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/cake_dsv4/README.md#L107-L211) |
| Hand-written sparse MLA SM120 | same three | 1-D `cp.async.bulk` plus `cp.async`; split grid and LSE merge | not determined | `include/flashinfer/attention/sparse_mla_sm120/` |
| XQA MLA SM120 | not determined | TMA, page id as the last coordinate; 384 threads | not determined | [`csrc/xqa/mla_sm120.cu:132`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mla_sm120.cu#L132) |
| `cake_vsa_sm120` (block-sparse) | `m16n8k16` | TMA 4-D with `fence.proxy.async` | yes, `sm_120a` and `sm_121a` | [`flashinfer/jit/cake_vsa_sm120.py:15, 42`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/cake_vsa_sm120.py#L15) |
| SM120 DSA indexer | FP8 k32; `mxf4nvf4` k64 | TMA 2-D/3-D; `kSmemLimit = 101376` | "on SM12x GPUs" | `flashinfer/experimental/sm120_dsa_indexer/` |
| DeepGEMM SM120 paged MQA logits | FP8 k32 | TMA; 384-thread persistent program | README: SM120a only | [`flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/README.md:10-20`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/README.md#L10-L20) |
| `nvfp4_attention_sm120` (dense, non-causal) | `mxf4nvf4` k64 | TMA header present | yes | [`flashinfer/nvfp4_attention_sm120.py:171`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/nvfp4_attention_sm120.py#L171) |

The DeepGEMM README states the constraint in one line: "no tcgen05/TMEM/WGMMA, which `ptxas` rejects on this architecture".

**Experimental policy.** `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` gates only *automatic* selection of experimental backends; an explicit `backend="<name>"` always works [SRC] [`flashinfer/api_logging.py:2600`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/api_logging.py#L2600); [`flashinfer/experimental/__init__.py:26-30`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/__init__.py#L26-L30). Experimental features are "JIT-only and must not ship in pre-built packages" [SRC] [`flashinfer/experimental/AGENTS.md:24`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/AGENTS.md#L24). The variable does not appear in vLLM's tree.

---

## 8. Open work, pull request by pull request

States were read from the GitHub API on 2026-11-17.

### 8.1 Upstream FlashAttention: two competing lines

**Line A extends the Ampere subclass.**

| PR | State | Size | What it adds | Notes |
|---|---|---|---|---|
| #2329 | merged 2026-03-12 | +102/−7 | The SM120 subclass; fixes an SM80 causal bug that reset the running maximum | Correctness only, on GB10: 17/17 configurations |
| #2336 | open, last update 2026-08-14 | +262/−178 | Split-KV, writing FP32 partials from accumulator registers | Maintainer review: "The paged KV stuff doesnt' look right, we should not be indexing in pytorch... For split we should be writing in fp32, not bf16." Author reports both addressed |
| #2348 | open, last update 2026-11-01 | +561/−224 | In-kernel paged KV; moved to 2 stages | A commenter first measured 10× dense; maintainer: "In the past we've see 0-10% perf diff between dense and paged KV" |
| #2980 | open | +1885/−641 | FP8 e4m3 `m16n8k32`; split-KV; FP8-only paged KV | BF16 paged calls rejected explicitly |
| #2998 | open | +17/−0 | Tile selection: falls back to 64×64, fixing head dim 256 | No comments |
| #3060 | open | +310/−40 | Paged KV for SM80 and SM120 | Reviewer found two correctness bugs on an RTX 5070 Ti; author acknowledged |

**Line B replaces the subclass with SM12x-owned kernels.**

| PR | State | Size | What it adds |
|---|---|---|---|
| #3108 | open | +342/−18 | Groundwork: SM count from the stream context, per-batch split-KV pieces |
| #3109 | open | +16,216/−290 | New kernels under `flash_attn/cute/sm120/`: warp-MMA forward **with TMA K/V**, head dims to 256, paged KV, block sparsity, split-KV, a transposed paged-decode kernel, and a new backward. The author states the forward is a port of the SGLang SM12x implementation |
| #3132 | open, draft | +17,412/−320 | FP8 (e4m3 and e5m2) forward for the #3109 kernels |

Line B's FP8 pull request states it is not stacked on #2980 and that #3109 replaces the file #2980 edits. The two lines conflict.

**All open pull requests show `REVIEW_REQUIRED`.** The only maintainer comments found are on #2336 and #2348.

A separate FA2 pull request, #3026 (+3/−2, open), adds capability 12.0 to FA2's `is_sm8x` predicate so SM120 gets the Ada tiles. Its rationale: "64KB per block means only 1 CTA per SM on `sm_120`, 4 of 48 warps". **It explicitly does not cover capability 12.1** [PR].

### 8.2 SGLang

- **#32991** (merged 2026-08-13, +10,449/−198): the `fa4_sm120` kernels of section 4.7. A later comment reports a bug fixed in #49044, which was not read.
- **#45871** (open, +649/−68, zero comments, CI red at reading): native FP8 KV for those kernels. QK and PV on FP8 `m16n8k32`; V kept as FP8 in shared memory; P pre-scaled by 256 before the FP8 cast; new tile candidates for head dimensions above 256.
- **#45262** (merged 2026-11-10): b12x for MXFP8 linear layers on SM12x. GEMM, not attention.
- **#35899** (closed, not merged): a b12x compressed-MLA attention branch for DeepSeek-V4-Flash.

### 8.3 FlashInfer

- **#7675** (open, filed 2026-10-28). In the b12x MXFP8 **GEMM** on SM120, "the TMA refill of a pipeline slot is not ordered after the MMA warps' ldmatrix". Under CUDA-graph replay with concurrent work on a second stream, wrong tiles are returned; eager results match CUTLASS. The reporter's sweep: 1,537 bad replays stock, 0 with a one-line `fence_proxy("async.shared", space="cta")`. Reproduced on an RTX 5090 [ISSUE] [S18].
- **#7676** (open): the fence fix [S19].
- **#6977** (open): b12x paged attention cannot plan head dimension 64 decode [S20].
- **#3170** (open, updated 2026-11-13): the DGX Spark support audit, the canonical gap list [S17].

**Does the #7675 hazard apply to b12x paged attention? [INF, unresolved]**

- *Observation:* the paged attention files contain no `fence_proxy` call.
- *Mechanism that would cause the bug:* a math warp reads a slot with `ldmatrix` (generic proxy), the stage is released, and the next TMA write (async proxy) to the same slot is issued without a fence.
- *Why it might not apply:* in the default decode kernel the same warp that reads also issues the next TMA, after releasing the stage, and the pipeline has one stage. In the GEMM, the producer and consumers are different warps.
- *Why it might:* the role-specialized decode variant has a separate loader warp, which is the GEMM's structure.
- *Discriminating test:* the #7675 reproduction pattern (CUDA-graph replay with a copy and matmul on a second stream) applied to b12x paged decode, with and without the role-specialized variant.

### 8.4 SGLang issues on these GPUs

- **#40286** (open): GLM-5.3-Flash on 2× DGX Spark. Every sparse-attention backend fails: the automatic one with `Unsupported architecture`; TileLang with `Failed to set the allowed dynamic shared memory size to 169984`; the FlashInfer sparse backend because the model class is not in an allow-list. The allow-list is unchanged at the commit read [ISSUE]; [SRC] [`flash_mla_sm120.py:29-32`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_mla_sm120.py#L29-L32).
- **#42364** (open): DeepSeek-V4.1 on 4× DGX Spark fails at attention initialization because a `major >= 10` test sent SM12x down a path requiring an SM100-only DeepGEMM feature. The commit read contains the fix shape (`if not is_sm100() ...: return dense_blocks`), so the failure appears fixed on `main` although the issue is open [INF from SRC] [`dsv4/v41_indexer/__init__.py:139-145`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py#L139-L145).
- **#50749** (open, one day old): a TileLang kernel on an RTX 5090 requests 170,048 B of shared memory against 101,376 B. The two KV buffers alone need 2 × 64 × (512 + 64) × 2 = 147,456 B [ISSUE].

---

## 9. Published performance numbers

**Every number here is an author's claim from a pull request or issue.** None was reproduced. Hardware is as the author named it.

### 9.1 BF16 kernels against FA2 and against each other

| Source | Hardware | Comparison | Result |
|---|---|---|---|
| FA #3109 | RTX PRO 6000 (188 SMs) | new SM12x kernels vs upstream FA4 main, forward, 111 cases | 1.34× geomean |
| FA #3109 | RTX PRO 6000 | vs FA2 2.8.4, forward | 1.13× |
| FA #3109 | RTX PRO 6000 | decode on a dense cache vs upstream FA4 main | 2.54× (up to 12.3×) |
| FA #3109 | RTX PRO 6000 | decode vs FA2 dense / FA2 paged | 1.15× / 1.12× |
| FA #3109 | GB10 (48 SMs) | head dim 128 GQA, 8,192 tokens, causal forward vs main | 1.16× |
| FA #3109 | GB10 | decode, one 32K-context request vs main | 1.7–2.0× |
| FA #2998 | RTX PRO 6000 | tile tuning vs default FA4, 49 shapes | 1.19× (range 0.997–1.87×) |
| FA #2998 | RTX PRO 6000 | tile tuning vs FA2 | 1.11× |
| FA #3026 | RTX PRO 6000 Server Edition | FA2 with Ada tiles vs FA2 as shipped | 1.02–1.09× non-causal; ≈1.00× causal |
| FA #2980 | RTX PRO 6000 | BF16 decode vs FA2, 108 shapes | 1.17× (61/108 faster; worst 0.43×) |
| FA #2348 | GB10 | paged vs dense decode, page size 64 | 1.09–1.18× slower |
| SGLang #32991 | RTX PRO 6000 | `fa4_sm120` prefill vs Triton, 228 cases | 3.03× |
| SGLang #32991 | GB10 | `fa4_sm120` prefill vs Triton | 3.61× |
| SGLang #32991 | RTX PRO 6000 / GB10 | prefill vs the old SM80-derived path | 2.38× / 1.73× |
| SGLang #32991 | RTX PRO 6000 / GB10 | paged decode vs Triton, page size 64 | 1.31× / 1.21× |

One absolute data point from SGLang #32991, head dimension 256, 6,144 tokens: on GB10 the new kernel took 2,108 µs against 1,824 µs for the old fallback, so the new kernel **lost** in that case. On the RTX PRO 6000 the same case was 456 µs against 572 µs.

### 9.2 FP8 against BF16

| Source | Hardware | Comparison | Result |
|---|---|---|---|
| FA #2980 | RTX PRO 6000 | FP8 prefill vs FA2 BF16, 49 rows | 1.31× |
| FA #2980 | RTX PRO 6000 | FP8 decode vs FA2, 108 shapes | 1.70× (103/108 faster; worst 0.85×) |
| FA #2980 | RTX PRO 6000 | FP8 paged vs FP8 contiguous | 0.883× |
| FA #3132 | RTX PRO 6000 | FP8 vs BF16 prefill, 40 cases | 1.69× |
| FA #3132 | RTX PRO 6000 | FP8 vs BF16 paged decode, page 256 | 1.84× |
| SGLang #45871 | RTX PRO 6000 | FP8 decode, head dim 256, 9,216 KV tokens | 2.03× |
| SGLang #45871 | RTX PRO 6000 | FP8 prefill, head dim 512 | 2.19× |
| SGLang #45871 | RTX PRO 6000 | serving `gemma-4-31B-it`, 8k in / 1k out | 2.25× throughput (383 → 864 tok/s) |

SGLang #45871 also reports GSM8K-Platinum accuracy of 0.730 for BF16 and 0.742 for FP8.

### 9.3 What the numbers suggest [INF]

- **Against FA2, BF16 gains on the RTX PRO 6000 are 1.1× to 1.2×.** FA2 is not far behind on this card, despite running inherited tiles.
- **The largest BF16 gains are against weak baselines:** Triton, or upstream FA4's untuned single-split SM120 path.
- **FP8 is the consistent large gain,** 1.7× to 2.0× across three independent pull requests.
- **There are no FP8 numbers for the DGX Spark.** Every FP8 result is from an RTX PRO 6000.
- **GB10 gains are smaller than RTX PRO 6000 gains for prefill** (1.16× against 1.34× in #3109; 1.73× against 2.38× in SGLang #32991), with one outright loss.

---

## 10. Where SM120 and SM121 diverge

Most code treats the two machines alike. These are the places it does not.

### 10.1 Tuned for 12.0, generic on 12.1

| Where | What happens on a Spark | Citation |
|---|---|---|
| b12x paged trait specializations (5 of them) and one planner budget | generic tile sweep instead of the tuned tiles | [`flashinfer/experimental/b12x/attention/paged/traits.py:383, 411, 459, 468, 486`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py#L383); [`planner.py:272`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py#L272) |
| vLLM Triton attention, 64 parallel softmax segments | not applied | [`vllm/v1/attention/backends/triton_attn.py:152-162`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_attn.py#L152-L162) |
| FA2 tile fix (FA #3026) | explicitly not covered | [PR] |
| SGLang split-KV latency model | constants fitted on 110-SM and 188-SM discrete cards | [`sglang/.../fa4_sm120/runtime.py:45-48`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/runtime.py#L45-L48) |

### 10.2 Excluded on 12.1 in SGLang (`device.is_sm120()` is false)

| Where | Effect | Citation |
|---|---|---|
| Qwen sparse attention decode via the TRT-LLM kernel | disabled; comment: "it silently corrupts long-context decode on SM121/GB10" | [`srt/layers/attention/qwen_sparse_attn_backend.py:108`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L108) |
| FP4 MoE fused-finalize fast path | SM120 only | [`srt/layers/moe/moe_runner/flashinfer_cutlass.py:284, 323`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py#L284) |
| SM120 BF16 GEMM, FP8 and NVFP4 GEMM kernels | not used | [`kernels/ops/gemm/sm120_bf16.py:59-60`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/gemm/sm120_bf16.py#L59-L60); [`kernels/kda_kernels/sm120_fp8.py:129`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/kda_kernels/sm120_fp8.py#L129) |
| Dense block-FP8 GEMM routing | SM121 keeps the DeepGEMM route | [`srt/model_loader/utils.py:285`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/model_loader/utils.py#L285) |

### 10.3 Specific to 12.1

| Where | Effect | Citation |
|---|---|---|
| SGLang Qwen sparse variable-length attention | uses a dedicated `qwen38_qsa_sm121_varlen` kernel | [`qwen_sparse_attn_backend.py:121-124`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L121-L124) |
| SGLang MoE kernel | 3 stages instead of 4: "sm_121 ... caps shared memory at 99 KB per block and num_stages=4 needs 108 KB" | [`kernels/ops/moe/inkling_moe.py:973`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/moe/inkling_moe.py#L973) |
| SGLang FA4 split heuristic | integrated-GPU branches; "Three unsplit CTAs per LPDDR channel already saturate the LPDDR path" | [`fa4_sm120/runtime.py:125-146, 245-248`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/runtime.py#L125-L146) |
| vLLM default batch tier | GB10 kept on the small `max_num_batched_tokens` tier | [`vllm/engine/arg_utils.py:3004-3006`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/engine/arg_utils.py#L3004-L3006) |
| FlashInfer wheels | CUDA-13 aarch64 wheels carry no native `sm_121a` binaries; the Spark runs SM120-family binaries | [VAULT, from v0.6.18 release notes] |
| FlashInfer b12x FP4 **GEMM** auto-selection | SM121 excluded by policy; comment says cutlass and cuDNN are faster on GB10 in most cases | [VAULT, source inspection 2026-09-10] |

### 10.4 What the divergence means [INF]

The Spark gets correct kernels almost everywhere. It gets tuned kernels almost nowhere. The single strongest piece of evidence about what limits it is SGLang's own comment that three unsplit blocks per LPDDR channel "already saturate the LPDDR path": the authors of the most Spark-aware heuristic found memory, not compute, to be the ceiling for decode.

---

## 11. Research path

### 11.1 Framing

The earlier version of this plan proposed inventing page-aligned TMA staging and SM12x split-KV. Both exist. The questions that remain open are narrower and better defined:

1. Which existing kernel wins for which workload on each machine, measured on one harness.
2. Whether the DGX Spark's limit is memory bandwidth, and what follows if so.
3. Where the unfinished pieces are: prefill staging on a paged cache, native FP8 in the paths the engines actually call, Spark-specific tuning, and dispatch.

### 11.2 The bandwidth argument for the Spark

**Arithmetic [CALC].** A hypothetical model with 32 layers, 8 KV heads and head dimension 128 stores 8 × 128 × 2 × 2 = 4,096 bytes per token per layer in BF16. That is 128 KiB per token, and 4 GiB at a 32,768-token context. One decode step reads all of it.

| Cache | Bytes read per step | Floor at 273 GB/s (Spark) | Floor at 1,792 GB/s (RTX PRO 6000) |
|---|---|---|---|
| BF16 | 4.29 GB | 15.7 ms | 2.4 ms |
| FP8 | 2.15 GB | 7.9 ms | 1.2 ms |
| NVFP4 (≈4.5 bits) | 1.21 GB | 4.4 ms | 0.7 ms |

These are floors for reading the cache once. They ignore compute.

**Hypothesis A.** On the Spark, long-context decode is bandwidth-bound for every kernel. Kernel choice matters little; cache precision matters almost linearly.

- *Supporting evidence:* SGLang's integrated-GPU comment (section 10.3); the smaller GB10 gains in section 9; the bandwidth ratio (6.6×) exceeding the SM ratio (3.9×).
- *Falsified if:* two BF16 kernels differ by more than about 20% at batch 1 and a 32K context on the Spark.

**Hypothesis B.** On the RTX PRO 6000, batch-1 decode is occupancy-bound, so split-KV and tile choice matter.

- *Supporting evidence:* #3109's 2.54× decode gain over an unsplit baseline; #2980's worst case of 0.43× at batch 32, where splitting should not help.
- *Falsified if:* achieved bandwidth is already near 1,792 GB/s without split-KV.

### 11.3 Phase 0 — Settle the device facts

Four device attributes resolve the documentation conflict in section 2.5 and the derived SM counts. On each machine:

```
cudaDevAttrMultiProcessorCount
cudaDevAttrMaxSharedMemoryPerMultiprocessor
cudaDevAttrMaxSharedMemoryPerBlockOptin
cudaDevAttrMaxBlocksPerMultiprocessor
```

- **Why first:** every residency claim in this report depends on the per-SM value. FA2's split heuristic assumes two blocks per SM. b12x and FlashInfer's fa2 read the value at run time, so their behaviour differs between a 100 KB and a 128 KB device.
- **Done when:** a four-row table exists for each machine.

### 11.4 Phase 1 — Harness

- **Record per run:** capability; engine and library commits; the log lines of section 5.5; head dimension; query-to-KV head ratio; page size; KV dtype; query length, KV length, batch.
- **Correctness oracle:** FP32 `scaled_dot_product_attention` on a de-paged copy. Per-row maximum absolute and relative error with a per-dtype tolerance. For scale: SGLang #32991 reports a maximum absolute difference of 0.0078 on RTX cards and 0.0156 on GB10 against Triton.
- **Edge cases:** lengths at page size ± 1; contexts long enough to recycle every pipeline slot; sinks under split-KV (b12x applies the sink only in the chunk owning the causal limit; XQA applies sinks in-kernel only when not split).
- **Stress gate:** every case under CUDA-graph replay with a copy and matmul on a second stream, the pattern from FlashInfer #7675.
- **Two levels:** a kernel microbenchmark and a served benchmark with time to first token and inter-token latency reported separately.

### 11.5 Phase 2 — Baseline the existing kernels

| Engine | Paths to run |
|---|---|
| vLLM | no flag; `FLASH_ATTN`; `FLASHINFER` with XQA on; `FLASHINFER` with `use_trtllm_attention=0`; `b12x`; `b12x` with `B12X_TURBO_ATTN=1` and FP8 KV; `TRITON_ATTN` |
| SGLang | no flag; `fa4` at page size 1 and 64; `trtllm_mha`; `fa4` prefill with `trtllm_mha` decode; `triton` |

- **Model:** one BF16 GQA model with head dimension 128, then one with head dimension 256.
- **Shapes:** prefill at 512, 4K and 32K tokens; decode at batch 1, 8 and 64 with 1K, 8K, 32K and 128K context; KV dtype BF16 and FP8.
- **Tools:** Nsight Compute per kernel: memory throughput, tensor-core utilization, achieved occupancy, shared memory, registers per thread.
- **Output:** a table for FlashInfer #3170. Maintainers mostly lack a GB10.

### 11.6 Phase 3 — FA2 tiles on SM12x (small, concrete)

- **Finding:** FA2 gives capability 12 the A100/H100 tile. FA #3026 fixes 12.0 only and reports 1.02–1.09×.
- **Question:** does the Ada tile (128×32 non-causal, 64×64 causal) help on a Spark, where #3026 does not apply?
- **Why it matters:** FA2 is vLLM's default. The change is three lines in upstream and must also land in vLLM's fork.
- **Dependency:** the phase-0 per-SM value decides whether 48 KB tiles double occupancy.

### 11.7 Phase 4 — Spark tuning for b12x and `fa4_sm120`

- **Finding:** b12x's specializations are keyed on (12, 0). SGLang's split model was fitted on discrete cards.
- **Experiments:**
  - Sweep b12x `num_mma_kv`, `graph_ctas_per_sm` and chunk budgets on 48 SMs.
  - Sweep `fa4_sm120` split counts on GB10 and compare against the fitted model's choice.
  - Test whether the "three unsplit CTAs per LPDDR channel" rule holds across head dimensions.
- **Contribution:** a (12, 1) specialization table for b12x, or evidence that the generic sweep is already near-optimal.

### 11.8 Phase 5 — Prefill staging on a paged cache

- **Finding:** decode has page-aligned TMA in b12x and in one SGLang specialization. Prefill does not. b12x extend uses per-thread `cp.async` with a per-row lookup; its own comment names the missing piece ("until extend TMA gets the decode kernel's page-entry flattening"). SGLang uses a `cp.async` gather with a loader warp.
- **Question:** for prefill, does TMA with the page id as a coordinate beat the `cp.async` gather at 99 KB with `mma.sync` consumers?
- **Design constraint:** the page size must be a multiple of the KV tile. Page 64 with a 64-row tile, or page 128 with sub-tile addressing as b12x decode does.
- **Reference for cost:** FA #2348 reports paged at 1.09–1.18× the dense time on GB10 with `cp.async`; FA #2980 reports paged FP8 at 0.883× of contiguous.
- **Gate:** phase-1 stress test, with attention to the fence question of section 8.3.

### 11.9 Phase 6 — Native FP8 in the paths the engines call

- **Finding:** native FP8 multiply exists in b12x (opt-in), FA #2980 and #3132 (open), SGLang #45871 (open). vLLM's `FLASHINFER` path converts FP8 to 16-bit. Reported gains are 1.7–2.0×, all on RTX PRO 6000.
- **Experiments:**
  - BF16 KV, FP8 converted, FP8 native, on both machines.
  - `B12X_TURBO_ATTN=1` against the default, with the batch and chunk thresholds that limit it for decode.
  - Accuracy against the FP32 oracle and a task metric, reported separately from speed.
- **Why the Spark matters here:** if hypothesis A holds, converted FP8 already halves the bytes read, and native FP8 adds little for decode. If native FP8 still helps on the Spark, decode there is not purely bandwidth-bound.
- **Technique to note:** both #3132 and #45871 scale P by 2⁸ before the FP8 cast so that probabilities use the format's range.

### 11.10 Phase 7 — Coverage gaps

| Gap | Where | Possible work |
|---|---|---|
| Head dimension 64 in b12x decode | trait rule plus merge restriction (section 4.5) | allow `num_mma_kv = 2` for 16-row tiles; extend the merge |
| Head dimension 512 | SGLang `fa4_sm120` raises; b12x supports 128/192/256; FMHAv2 handles it | a 512 tile within 101,376 B needs FP8 operands or a narrower tile |
| BF16 paged KV in upstream FA4 | #2348 and #3060 open | review and test on both machines |
| Multimodal prefix | composites are major 9 or 10 only | measure FlashInfer's native bidirectional wrapper against Triton |
| Decode context parallelism | b12x and XQA lack LSE output | relevant for 2× RTX PRO 6000 or 2× Spark |

### 11.11 Phase 8 — Dispatch

- **Finding:** vLLM picks by validity. Its SM12x default is FA2, and b12x is unreachable without a flag.
- **Deliverable:** a rule per machine, phase and KV dtype, backed by phase-2 tables. SGLang's per-phase flags can run such a rule today. vLLM needs it in `_get_backend_priorities`.
- **A caution from the data:** #2980's BF16 decode was slower than FA2 in 47 of 108 shapes. A blanket "prefer the new kernel" rule would regress nearly half of them.

### 11.12 Order

Phases 0–2 are prerequisites. After that the order depends on which hypothesis survives:

- **A holds (Spark is bandwidth-bound):** phase 6 first on the Spark, then phase 4.
- **B holds (RTX PRO 6000 is occupancy-bound):** phases 3 and 5 first on the RTX PRO 6000.

---

## 12. Operator reference

### 12.1 vLLM on SM12x

| Goal | Flags | Expected log | Preconditions |
|---|---|---|---|
| FA2 | `--attention-backend FLASH_ATTN` | `Using FlashAttention version 2` | no sinks, no FP8/NVFP4 KV, no multimodal prefix |
| FlashInfer with XQA decode | `--attention-backend FLASHINFER` | `decode_backend=xqa ... arch=sm120` or `sm121` | cubins reachable; head dim 16–256 |
| FlashInfer, XQA off | add `--attention-config.use_trtllm_attention=0` | `decode_backend=flashinfer-native` | sinks become unsupported |
| b12x | `--attention-backend b12x` | `Using b12x with q_heads=...` | BF16 model; head dim 128/192/256; block size unset, 64 or 128 |
| Triton | `--attention-backend TRITON_ATTN` | `Using AttentionBackendEnum.TRITON_ATTN backend.` | none |
| Sparse MLA | `--attention-backend FLASHINFER_MLA_SPARSE_SM120` | same pattern | `index_topk == 2048`, BF16 |

Environment variables:

| Variable | Effect on SM12x | Citation |
|---|---|---|
| `VLLM_HAS_FLASHINFER_CUBIN=1` | skips the network probe that gates XQA | [`vllm/utils/flashinfer.py:85-92`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/utils/flashinfer.py#L85-L92) |
| `VLLM_BATCH_INVARIANT=1` | disables XQA; `FLASHINFER` and `B12X` fail validation | [`utils/flashinfer.py:875-877`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/utils/flashinfer.py#L875-L877); [`backend.py:221-223`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backend.py#L221-L223) |
| `VLLM_KV_CACHE_LAYOUT` | overrides the cache layout | [`vllm/envs.py:270-273`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/envs.py#L270-L273) |
| `B12X_TURBO_ATTN=1` | native FP8 multiply in b12x (read by FlashInfer) | [`flashinfer/.../paged/_forward.py:78-79`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/_forward.py#L78-L79) |

### 12.2 SGLang on SM12x

```bash
# FA4, both phases. BF16/FP16 KV only; head dim <= 256; not for MLA or hybrid GDN.
# Page size is not forced on SM12x; 64 enables the page-TMA head-dim-256 decode kernel.
--attention-backend fa4 --page-size 64

# FA4 prefill + XQA decode
--prefill-attention-backend fa4 --decode-attention-backend trtllm_mha --page-size 64

# FMHAv2 prefill + XQA decode. Not with --kv-cache-dtype fp8_e4m3.
--attention-backend trtllm_mha --page-size 64

# FP8 KV with XQA decode
--prefill-attention-backend flashinfer --decode-attention-backend trtllm_mha \
  --kv-cache-dtype fp8_e4m3 --page-size 64

# NVFP4 KV: pass only the dtype; the resolver picks flashinfer prefill and XQA decode
--kv-cache-dtype nvfp4
```

Debug aid: `FLASH_ATTENTION_ARCH=sm_120` overrides FA4's architecture detection [SRC] [`flash_attn/cute/interface.py:86-101`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py#L86-L101).

### 12.3 Verification commands

Each falsifies one claim in this report if its output differs.

| Claim | Command | Expected |
|---|---|---|
| vLLM default is FA2 | `vllm serve <bf16 model>`, grep the log | `Using FLASH_ATTN attention backend out of potential backends` and `Using FlashAttention version 2` |
| FP8 KV moves vLLM to FlashInfer + XQA | add `--kv-cache-dtype fp8` | backend `FLASHINFER`; `decode_backend=xqa` |
| b12x is installed | `python -c "import importlib.util as u; print(u.find_spec('b12x'))"` | not `None` |
| Fork gates FA4 off | `python -c "from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported as f; print(f(4), f(2))"` | `False True` |
| Shared-memory capacity constant | `python -c "import cutlass.utils as u; print(u.get_smem_capacity_in_bytes('sm_120'))"` | `101376` |
| SGLang helper semantics on a Spark | `python -c "from sglang.srt.utils.device import is_sm120, is_sm120_supported, is_sm121; print(is_sm120(), is_sm120_supported(), is_sm121())"` | `False True True` |
| b12x head dimension 64 mechanism | `select_paged_forward_traits(cta_tile_q=16, head_dim_qk=64, head_dim_vo=64, ...)` | `ValueError: no valid NUM_MMA_KV...`; success with `exact_num_mma_kv=2` |
| Upstream FA4 head dimension 256 | call `flash_attn.cute.interface.flash_attn_func` with BF16, head dim 256 | a launch-time shared-memory error |

---

## 13. Corrections, disagreements between sources, and unknowns

### 13.1 Corrections to the previous version of this report

| Previous statement | Correct statement |
|---|---|
| "No public measurement was found comparing any two of these kernels" | Pull requests report measurements on both machines (section 9). None is independent. |
| The research path proposed page-aligned TMA tiles as a new idea | b12x decode and one SGLang specialization already do it (sections 4.5, 4.7) |
| The research path proposed split-KV for SM12x as missing | It exists in FA2, FlashInfer fa2, XQA, b12x decode and SGLang `fa4_sm120`; it is missing only in upstream FA4 |
| vLLM's default on SM12x is FA2 | True for plain BF16 models; `FLASHINFER` for FP8/NVFP4 KV, sinks, or a multimodal prefix |
| Only the 99 KB shared-memory limit was given | 99 KB is per block; per SM is 128 KB or 100 KB depending on the NVIDIA document |
| Forcing FA4 in vLLM "depends on the fork, which was not inspected" | The gate is in vLLM's tree and excludes SM12x; the expected result is a startup failure |
| vLLM PR #50862 was cited as an FA4 gate fix | It is closed without merging and concerns GDN prefill |

### 13.2 Documentation that lags the code

- **vLLM docs** call XQA "the SM90 decode path" and say the FlashAttention default is "FA2 otherwise"; the code enables XQA on the SM12x family [SRC] [`docs/design/attention_backends.md:176-178`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/docs/design/attention_backends.md#L176-L178).
- **vLLM docs** list FlashInfer capability as 8.x–9.x, 9.0 and 10.x; the source accepts up to 12.1.
- **vLLM docs** list b12x KV dtypes as `auto` and capability "Any"; the source accepts BF16 and FP8 and exactly (12, 0) and (12, 1).
- **SGLang docs** list FA4's native page size as 128 with no qualifier; the forcing is SM100-only.
- **SGLang docs** say Blackwell defaults to `trtllm_mha`; that is SM100 only.
- **SGLang docs** say `trtllm_mha` is decode-only for hybrid GDN models on SM120; the assertion allows prefill.
- **SGLang docstring** in `get_default_attn_backend` says "trtllm_mha does not support SM120"; the validation code accepts it.
- **The author's vault note** "Writing FlashInfer Kernels for DGX Spark GB10" says "FA4 is tcgen05-only → SM100 only". Upstream has had an `mma.sync` SM120 path since March 2026.

### 13.3 Disagreements between primary sources

- **Shared memory per SM, capability 12:** 128 KB (tuning guide) against 100 KB (programming guide).
- **Thread blocks per SM, capability 12:** 32 against 24, same two documents.
- **Minimum toolkit for DGX Spark:** CUTLASS's README says CUDA 13.0; its [`config.h`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/arch/config.h) enables SM121 from 12.9.
- **FA #2998's body** says `can_implement` rejects head dimension 256; no call to `can_implement` was found on the forward path at `9f9742c`.
- **FA #3109's body** describes main's SM120 path as "fixed 64 × 64 tiles"; that matches the backward kernel, not the forward.

### 13.4 Not determined

- SM counts read from a device (188 and 48 are derived).
- Any limit specific to capability 12.1; neither NVIDIA guide mentions it.
- Whether the `flashinfer-python==0.7.4` wheel contains `b12x.attention.paged`. FlashInfer's own guidance says experimental features are JIT-only and must not ship in pre-built packages.
- FMHAv2's staging mechanism.
- Whether b12x paged attention has the #7675 ordering hazard.
- Whether SGLang's dense `trtllm_mla` runs on SM12x.
- Whether `fa4` prefill with `trtllm_mha` decode is numerically correct end to end.
- The contents of SGLang #49044.
- The fleet's current FlashInfer versions (0.6.16.post3 and 0.7.0 as of 2026-10-10 [VAULT]).
- Maintainer positions on any open FlashAttention pull request except #2336 and #2348.

### 13.5 The smallest experiment that would change the plan

On each machine, serve one BF16 head-dimension-128 model twice: with no flag (FA2), and with `--attention-backend b12x`. Confirm each backend from the log lines in section 5.5. Measure batch-1 inter-token latency at a 32K context.

- **Near-equal on the Spark, clearly different on the RTX PRO 6000:** hypotheses A and B both stand. Do phase 6 first on the Spark and phases 3 and 5 first on the RTX PRO 6000.
- **b12x clearly faster on the Spark too:** decode there is not purely bandwidth-bound. Phase 4 (Spark tuning) moves ahead of phase 6.
- **FA2 faster on the RTX PRO 6000:** consistent with #2980's finding that new BF16 kernels lose to FA2 on many decode shapes, and a reason to treat phase 8 (dispatch) as the main deliverable.

---

## 14. Sources

All accessed 2026-11-17.

**NVIDIA documentation**
- [S1] PTX ISA manual, version 9.5 — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- [S2] Blackwell tuning guide, version 13.5 — <https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>
- [S3] CUDA programming guide, compute capabilities — <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- [S4] CUDA GPU compute capability list — <https://developer.nvidia.com/cuda/gpus>
- [S5] DGX Spark hardware overview — <https://docs.nvidia.com/dgx/dgx-spark/hardware.html>
- [S6] RTX PRO 6000 Blackwell datasheet (hosted by PNY; seen as a search snippet, not opened) — <https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20RTX/English/NVIDIA-RTX-PRO-6000-Blackwell-Workstation-Edition.pdf>
- [S7] RTX PRO 6000 Blackwell product page — <https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/>

**CUTLASS** (`b2f249a`)
- [S8] [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/tree/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd) — files cited (11), linked at the pinned commit:
  - [`README.md`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/README.md)
  - [`include/cute/arch/copy_sm90_tma.hpp`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/copy_sm90_tma.hpp)
  - [`include/cute/arch/mma_sm120.hpp`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm120.hpp)
  - [`include/cute/arch/mma_sm80.hpp`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm80.hpp)
  - [`include/cute/arch/mma_sm89.hpp`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cute/arch/mma_sm89.hpp)
  - [`include/cutlass/arch/arch.h`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/arch/arch.h)
  - [`include/cutlass/arch/config.h`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/arch/config.h)
  - [`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp)
  - [`include/cutlass/gemm/collective/builders/sm120_mma_builder.inl`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl)
  - [`include/cutlass/gemm/collective/sm120_mma_tma.hpp`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/collective/sm120_mma_tma.hpp)
  - [`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp)

**FlashAttention**
- [S9] Upstream repository (`9f9742c`) — [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/9f9742c80e69b3ca28a231c26ef931e98c2c3826) — files cited (10), linked at the pinned commit:
  - [`csrc/flash_attn/flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/flash_api.cpp)
  - [`csrc/flash_attn/src/flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/flash_fwd_kernel.h)
  - [`csrc/flash_attn/src/flash_fwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/flash_fwd_launch_template.h)
  - [`csrc/flash_attn/src/kernel_traits.h`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/csrc/flash_attn/src/kernel_traits.h)
  - [`flash_attn/cute/config.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/config.py)
  - [`flash_attn/cute/flash_fwd.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd.py)
  - [`flash_attn/cute/flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd_sm100.py)
  - [`flash_attn/cute/flash_fwd_sm120.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd_sm120.py)
  - [`flash_attn/cute/flash_fwd_sm90.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/flash_fwd_sm90.py)
  - [`flash_attn/cute/interface.py`](https://github.com/Dao-AILab/flash-attention/blob/9f9742c80e69b3ca28a231c26ef931e98c2c3826/flash_attn/cute/interface.py)
- [S10] Issue #2307 "FA4 SM120 support?" (open) — <https://github.com/Dao-AILab/flash-attention/issues/2307>
- [S11] Issue #2996 (closed 2026-10-23) — <https://github.com/Dao-AILab/flash-attention/issues/2996>
- [S12] PRs [#2329](https://github.com/Dao-AILab/flash-attention/pull/2329) (merged 2026-03-12), [#2336](https://github.com/Dao-AILab/flash-attention/pull/2336), [#2348](https://github.com/Dao-AILab/flash-attention/pull/2348), [#2980](https://github.com/Dao-AILab/flash-attention/pull/2980) (open)
- [S13] PRs [#2998](https://github.com/Dao-AILab/flash-attention/pull/2998), [#3026](https://github.com/Dao-AILab/flash-attention/pull/3026), [#3060](https://github.com/Dao-AILab/flash-attention/pull/3060), [#3108](https://github.com/Dao-AILab/flash-attention/pull/3108), [#3109](https://github.com/Dao-AILab/flash-attention/pull/3109), [#3132](https://github.com/Dao-AILab/flash-attention/pull/3132) (open)
- [S14] vLLM fork (`8dd9eaea`) — [vllm-project/flash-attention](https://github.com/vllm-project/flash-attention/tree/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2) — files cited (3), linked at the pinned commit:
  - [`csrc/flash_attn/flash_api.cpp`](https://github.com/vllm-project/flash-attention/blob/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2/csrc/flash_attn/flash_api.cpp)
  - [`csrc/flash_attn/src/kernel_traits.h`](https://github.com/vllm-project/flash-attention/blob/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2/csrc/flash_attn/src/kernel_traits.h)
  - [`vllm_flash_attn/flash_attn_interface.py`](https://github.com/vllm-project/flash-attention/blob/8dd9eaea16a6c71d2cff4d57c925a84d333e2de2/vllm_flash_attn/flash_attn_interface.py)

**FlashInfer** (`778f57e`)
- [S15] Repository — [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer/tree/778f57e4d73a73a439ad9cb88daabb5e14cbf816) — files cited (33), linked at the pinned commit:
  - [`csrc/cake_dsv4/README.md`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/cake_dsv4/README.md)
  - [`csrc/xqa/ldgsts.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/ldgsts.cuh)
  - [`csrc/xqa/mha.cu`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mha.cu)
  - [`csrc/xqa/mhaUtils.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mhaUtils.cuh)
  - [`csrc/xqa/mla_sm120.cu`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mla_sm120.cu)
  - [`csrc/xqa/mma.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/mma.cuh)
  - [`csrc/xqa/utils.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/csrc/xqa/utils.cuh)
  - [`flashinfer/api_logging.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/api_logging.py)
  - [`flashinfer/compilation_context.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/compilation_context.py)
  - [`flashinfer/decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/decode.py)
  - [`flashinfer/experimental/AGENTS.md`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/AGENTS.md)
  - [`flashinfer/experimental/__init__.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/__init__.py)
  - [`flashinfer/experimental/b12x/_lib/gating.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/_lib/gating.py)
  - [`flashinfer/experimental/b12x/_lib/intrinsics.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/_lib/intrinsics.py)
  - [`flashinfer/experimental/b12x/attention/paged/__init__.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/__init__.py)
  - [`flashinfer/experimental/b12x/attention/paged/_forward.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/_forward.py)
  - [`flashinfer/experimental/b12x/attention/paged/forward_extend_generic.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_extend_generic.py)
  - [`flashinfer/experimental/b12x/attention/paged/forward_paged.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/forward_paged.py)
  - [`flashinfer/experimental/b12x/attention/paged/merge.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/merge.py)
  - [`flashinfer/experimental/b12x/attention/paged/planner.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/planner.py)
  - [`flashinfer/experimental/b12x/attention/paged/traits.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/b12x/attention/paged/traits.py)
  - [`flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/README.md`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/README.md)
  - [`flashinfer/jit/attention/fmha_v2/fmha_library.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/attention/fmha_v2/fmha_library.py)
  - [`flashinfer/jit/attention/modules.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/attention/modules.py)
  - [`flashinfer/jit/cake_vsa_sm120.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/cake_vsa_sm120.py)
  - [`flashinfer/jit/xqa.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/jit/xqa.py)
  - [`flashinfer/nvfp4_attention_sm120.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/nvfp4_attention_sm120.py)
  - [`flashinfer/prefill.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/prefill.py)
  - [`flashinfer/utils.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/utils.py)
  - [`flashinfer/xqa.py`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/flashinfer/xqa.py)
  - [`include/flashinfer/attention/prefill.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/prefill.cuh)
  - [`include/flashinfer/attention/scheduler.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/attention/scheduler.cuh)
  - [`include/flashinfer/utils.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/778f57e4d73a73a439ad9cb88daabb5e14cbf816/include/flashinfer/utils.cuh)
- [S16] Release notes — <https://flashinfer.ai/releases/>
- [S17] Issue #3170, DGX Spark support audit (open) — <https://github.com/flashinfer-ai/flashinfer/issues/3170>
- [S18] Issue #7675, b12x GEMM TMA ordering (open) — <https://github.com/flashinfer-ai/flashinfer/issues/7675>
- [S19] PR #7676, fence fix (open) — <https://github.com/flashinfer-ai/flashinfer/pull/7676>
- [S20] Issue #6977, b12x head dimension 64 (open) — <https://github.com/flashinfer-ai/flashinfer/issues/6977>
- [S21] b12x repository (archived 2026-10-06) — <https://github.com/local-inference-lab/b12x>
- [S22] PR #5767, b12x migration (merged 2026-10-06) — <https://github.com/flashinfer-ai/flashinfer/pull/5767>

**vLLM** (`f39d833`)
- [S23] Repository — [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/f39d83306ea62d4ad8919995f378d8ca4aed8b57) — files cited (26), linked at the pinned commit:
  - [`docs/design/attention_backends.md`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/docs/design/attention_backends.md)
  - [`requirements/cuda.txt`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/requirements/cuda.txt)
  - [`vllm/config/attention.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/config/attention.py)
  - [`vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/engine/arg_utils.py)
  - [`vllm/envs.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/envs.py)
  - [`vllm/model_executor/layers/attention/mla_attention.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/model_executor/layers/attention/mla_attention.py)
  - [`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py)
  - [`vllm/models/deepseek_v4/nvidia/model.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/models/deepseek_v4/nvidia/model.py)
  - [`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py)
  - [`vllm/platforms/cuda.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/cuda.py)
  - [`vllm/platforms/interface.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/platforms/interface.py)
  - [`vllm/utils/b12x.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/utils/b12x.py)
  - [`vllm/utils/flashinfer.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/utils/flashinfer.py)
  - [`vllm/v1/attention/backend.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backend.py)
  - [`vllm/v1/attention/backends/b12x.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/b12x.py)
  - [`vllm/v1/attention/backends/fa_utils.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/fa_utils.py)
  - [`vllm/v1/attention/backends/flash_attn.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flash_attn.py)
  - [`vllm/v1/attention/backends/flashinfer.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/flashinfer.py)
  - [`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py)
  - [`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py)
  - [`vllm/v1/attention/backends/mla/prefill/flash_attn.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/prefill/flash_attn.py)
  - [`vllm/v1/attention/backends/mla/triton_mla.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/mla/triton_mla.py)
  - [`vllm/v1/attention/backends/triton_attn.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_attn.py)
  - [`vllm/v1/attention/backends/triton_flash_attn.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_flash_attn.py)
  - [`vllm/v1/attention/backends/triton_flashinfer.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/v1/attention/backends/triton_flashinfer.py)
  - [`vllm/vllm_flash_attn/flash_attn_interface.py`](https://github.com/vllm-project/vllm/blob/f39d83306ea62d4ad8919995f378d8ca4aed8b57/vllm/vllm_flash_attn/flash_attn_interface.py)
- [S24] Attention backends documentation — <https://docs.vllm.ai/en/latest/design/attention_backends/>

**SGLang** (`b2cc023f`)
- [S25] Repository — [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/b2cc023f23e14e6482490238968e018573847d0c) — files cited (31), linked at the pinned commit:
  - [`python/pyproject.toml`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/pyproject.toml)
  - [`python/sglang/kernels/jit/utils/arch.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/jit/utils/arch.py)
  - [`python/sglang/kernels/kda_kernels/sm120_fp8.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/kda_kernels/sm120_fp8.py)
  - [`python/sglang/kernels/ops/attention/extend_attention.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/extend_attention.py)
  - [`python/sglang/kernels/ops/attention/fa4_sm120/dispatch.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/dispatch.py)
  - [`python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py)
  - [`python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd_decode.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd_decode.py)
  - [`python/sglang/kernels/ops/attention/fa4_sm120/paged_kv.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/paged_kv.py)
  - [`python/sglang/kernels/ops/attention/fa4_sm120/runtime.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/fa4_sm120/runtime.py)
  - [`python/sglang/kernels/ops/attention/flash_attention_v4_sm120.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attention_v4_sm120.py)
  - [`python/sglang/kernels/ops/attention/flash_attn/cute/interface.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py)
  - [`python/sglang/kernels/ops/attention/flash_mla_sm120.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/attention/flash_mla_sm120.py)
  - [`python/sglang/kernels/ops/gemm/sm120_bf16.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/gemm/sm120_bf16.py)
  - [`python/sglang/kernels/ops/moe/inkling_moe.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/kernels/ops/moe/inkling_moe.py)
  - [`python/sglang/srt/arg_groups/attention_hook.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/attention_hook.py)
  - [`python/sglang/srt/arg_groups/fields/exec_.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/fields/exec_.py)
  - [`python/sglang/srt/arg_groups/kv_cache_hook.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/kv_cache_hook.py)
  - [`python/sglang/srt/arg_groups/model_override_base.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/model_override_base.py)
  - [`python/sglang/srt/arg_groups/overrides.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/overrides.py)
  - [`python/sglang/srt/arg_groups/speculative_hook.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/arg_groups/speculative_hook.py)
  - [`python/sglang/srt/environ.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/environ.py)
  - [`python/sglang/srt/layers/attention/attention_registry.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/attention_registry.py)
  - [`python/sglang/srt/layers/attention/deepseek_v4_backend.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/deepseek_v4_backend.py)
  - [`python/sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py)
  - [`python/sglang/srt/layers/attention/flashinfer_backend.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/flashinfer_backend.py)
  - [`python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py)
  - [`python/sglang/srt/layers/attention/trtllm_mha_backend.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/attention/trtllm_mha_backend.py)
  - [`python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py)
  - [`python/sglang/srt/model_loader/utils.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/model_loader/utils.py)
  - [`python/sglang/srt/runtime_context.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/runtime_context.py)
  - [`python/sglang/srt/utils/device.py`](https://github.com/sgl-project/sglang/blob/b2cc023f23e14e6482490238968e018573847d0c/python/sglang/srt/utils/device.py)
- [S26] Attention backend documentation — <https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx>
- [S27] PR #32991, SM12x FA4 kernels (merged 2026-08-13) — <https://github.com/sgl-project/sglang/pull/32991>
- [S28] PR #45871, native FP8 KV (open) — <https://github.com/sgl-project/sglang/pull/45871>
- [S29] PR #45262 (merged 2026-11-10) — <https://github.com/sgl-project/sglang/pull/45262>
- [S30] PR #35899 (closed, not merged) — <https://github.com/sgl-project/sglang/pull/35899>
- [S31] Issue #50749 (open) — <https://github.com/sgl-project/sglang/issues/50749>
- [S32] Issue #40286 (open) — <https://github.com/sgl-project/sglang/issues/40286>
- [S33] Issue #42364 (open) — <https://github.com/sgl-project/sglang/issues/42364>
- [S34] Issue #33422 (open) — <https://github.com/sgl-project/sglang/issues/33422>

**Author's notes**
- "FlashInfer SM12x Release Tracker" (reviewed 2026-10-10); "Writing FlashInfer Kernels for DGX Spark GB10" (status dated 2026-09); "FlashInfer Issue 3013 - CUTLASS Slow on SM120-121" (reviewed 2026-10-11); `consumer-gpu-kernel-guide` tuning notes.
