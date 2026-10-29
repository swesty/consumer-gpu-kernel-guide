# qwen3_rmsnorm audit: HF `cuda-kernels` skill, GB10 + RTX PRO 6000 (2026-10-29)

Scope: `GB10/qwen3_rmsnorm` and `rtx-6000-pro/qwen3_rmsnorm`. The two trees are the same kernel; they differ only in the block-size cap (512 vs 1024), `#pragma unroll` (8 vs 4) and nvcc arch flags. Audit only: no kernel source was changed.

Skill: `huggingface/kernels` → `kernel-builder/skills/cuda-kernels` at commit `18c47a9`, installed in `.claude/skills/cuda-kernels/` (the same files `kernel-builder skills add --claude` would fetch). The skill has tuning guides for H100, A100 and T4 only. Its tuning advice was not applied to sm_120/121; only its compliance checklist and benchmarking rules were.

## Verdict

1. **The kernel's arithmetic is correct.** Every dtype and size tested is within one output ulp of an exact fp32 reference, on both GPUs.
2. **The binding is unsafe under misuse.** Seven edge cases produced a crash, NaN output, an out-of-bounds write or silently wrong data.
3. **It is not Hub-compliant.** It fails every item in the skill's pre-flight checklist.
4. **Its performance case does not hold against `torch.compile`.** On the RTX PRO 6000, inductor beats it at every shape measured. On GB10 it wins only at large, DRAM-bound shapes, by about 15%.
5. **On the RTX PRO 6000, the "RTX-tuned" variant is the slower of the two.**

## How it was measured

- **Hosts:**
  - spark-5: NVIDIA GB10, cc 12.1, image `vllm/vllm-openai:v0.32.0`.
  - jarvis: RTX PRO 6000 Blackwell Workstation, cc 12.0, image `vllm/vllm-openai:v0.30.0`.
  - Both images ship torch 2.13.0+cu130 and nvcc 13.0.
- **Containers:** each run was a side container. The live engines (`vllm_qwen38_next` on spark-5, and the 27B on jarvis, at about 6% utilization) were not restarted. jarvis numbers include whatever load that live engine added during the run.
- **Build:** each variant was built with its own `setup.py` (`build_ext --inplace`). Nix and kernel-builder were unavailable, so `nix run .#build-and-copy` and `kernel-builder check-abi` were **not run**. The vLLM images lack `cusparse.h` on the default include path, so `CPATH` pointed at `site-packages/nvidia/cu13/include`.
- **GPU time:** 100 back-to-back calls were captured in one CUDA graph and timed with CUDA events. The figure is the median of 9 replays divided by 100.
- **Host-loop timing:** there is also an event-timed Python loop in `raw/*/perf-*.json`. Every shape below about 512 rows sits at a flat 4.10 µs in that loop, which is the host launch rate, not GPU time. That is why the graph numbers below are the ones to use.
- **Baselines:**
  - `compile`: `torch.compile(mode="max-autotune-no-cudagraphs")` of the HF `Qwen3RMSNorm` math.
  - `copy`: `out.copy_(x)`, a read+write floor for the same bytes.

## GPU time per call, bf16 (µs, CUDA-graph replay)

| shape (rows × h) | GB10: GB10 var | GB10: RTX var | GB10: compile | RTX: GB10 var | RTX: RTX var | RTX: compile | RTX: copy |
|---|---|---|---|---|---|---|---|
| 1 × 4096 | 2.29 | 1.96 | **1.67** | 2.08 | 1.78 | **1.15** | 0.69 |
| 64 × 4096 | 3.07 | 3.76 | **2.65** | 2.44 | 2.00 | **1.72** | 0.77 |
| 512 × 4096 | 9.72 | 15.86 | **7.18** | 3.20 | 4.87 | **2.33** | 1.59 |
| 4096 × 4096 | **241.8** | 268.6 | 284.3 | 15.57 | 27.87 | **13.33** | 8.63 |
| 2048 × 128 (QK) | 3.06 | 3.06 | **1.61** | 1.88 | 1.98 | **1.16** | 0.75 |
| 131072 × 128 (QK) | 251.7 | **244.3** | 289.9 | 59.13 | 59.21 | **9.40** | 8.64 |

Raw data: `raw/*/run2.log`.

## Findings, most severe first

Each finding is labelled by how it is known:
- **[obs]**: measured in this audit.
- **[src]**: read from source.
- **[inf]**: an inference from the evidence.

### P1: The RTX tuning is backwards
- **What:** on the RTX PRO 6000, the 1024-thread-cap "RTX" build is 1.79× slower than the 512-cap "GB10" build at 4096×4096 (27.87 vs 15.57 µs), and 1.52× slower at 512×4096. **[obs]**
- **Mechanism:** both GPUs report `max_threads_per_multi_processor = 1536`. **[obs]**
  - A 1024-thread block therefore fits only once per SM, which caps occupancy at 67%. The register count of 40 per thread allows only one block as well: 40 × 1024 = 40,960 of 65,536 registers.
  - A 512-thread block fits three times per SM, for 100%. **[obs → inf]**
- **Location:** `rtx-6000-pro/qwen3_rmsnorm/kernel_src/rmsnorm.cu:35` (`MAX_THREADS = 1024`).

### P1: `torch.compile` matches or beats the kernel almost everywhere [obs]
- **RTX PRO 6000:** inductor wins at every shape measured.
  - The QK-norm shape (131072×128) runs at **59.1 µs against inductor's 9.4 µs**, 6.3× slower.
  - Both kernel variants are equally slow there (59.13 / 59.21 µs) because at hidden size 128 the block cap never comes into play.
- **GB10:** the kernel wins only at DRAM-bound sizes, by 15% at 4096×4096 and 13% at 131072×128. At decode-size shapes (1–512 rows) it is 1.15–1.4× slower than inductor.
- **Why QK-norm is slow [inf]:** at h=128 the kernel launches one 64-thread block per row, and each block does a shared-memory reduction with two `__syncthreads` for only 256 bytes of data. On a 1.8 TB/s part, block scheduling and synchronization dominate. On GB10 that overhead is hidden because the kernel is DRAM-bound at 273 GB/s. Inductor packs several rows into each block. A warp-per-row path for h ≤ 1024 is the usual fix.
- **Location:** `rmsnorm.cu:347-353` (thread count = h/2 → 64 threads), `:389`.

### P1: An undersized `out` is written out of bounds [obs]
- **What:** `rmsnorm(x[64,4096], w, out=pool[0:1])` overwrote the neighbouring row of `pool` on both GPUs.
- **Cause:** `torch_binding.cpp:31-76` never checks the shape, numel, dtype, contiguity or device of `out`. **[src]**

### P2: Silent wrong results from unchecked dtype and layout [obs]
| Case | Result |
|---|---|
| fp32 weight with bf16 input | NaN output |
| fp32 `out` with bf16 input | max error 5.0–5.9 |
| Non-contiguous `out` | max error 5.6–7.0 |

- **Cause:**
  - `torch_binding.cpp:41-47` checks only the weight's numel and contiguity. **[src]**
  - `__init__.py:41` builds `out` with `empty_like(input)`, which copies the input's strides, before the binding makes the input contiguous. A transposed input therefore gives a strided output buffer. **[src]**

### P2: Crashes on valid-looking inputs [obs]
- **Misaligned input:** a contiguous bf16 or fp32 view at a 2- or 4-byte offset gives `CUDA error: misaligned address`. This is a sticky error that poisons the CUDA context. The `__nv_bfloat162` loads need 4-byte alignment and the `float4` loads need 16-byte alignment, and neither is checked. **[src]** Locations: `rmsnorm.cu:171,193,296,320`.
- **Zero rows:** a `[0, 4096]` input gives `CUDA error: invalid argument`, because the kernel is launched with a grid of 0. There is no `C10_CUDA_KERNEL_LAUNCH_CHECK`, so the error surfaces at the next synchronization rather than at the call. **[obs + src]**
- **Zero hidden size:** `hidden_size == 0` divides by zero on the host at `torch_binding.cpp:44`, which raises SIGFPE. **[src, not run]**
- **Overflow:** `row * hidden_size` is a 32-bit int, and `num_rows` truncates numel to int, so any tensor of 2³¹ or more elements overflows. A bf16 `[524288, 4096]` tensor is 4 GiB, which fits on both GPUs. **[src, not run]** Locations: `rmsnorm.cu:119,167,230,292`.

### P2: The project's own benchmark fails a correct kernel [obs]
- **What:** `benchmark_rmsnorm.py --dtype bfloat16` prints **FAIL** for all three correctness shapes on both GPUs (`max_diff=0.0625` against a threshold of `< 0.01`).
- **Mechanism:**
  - The kernel rounds once: `round(x·r·w)`.
  - HF `Qwen3RMSNorm` rounds twice: `w * round(x·r)`, with that multiply in bf16.
  - One bf16 ulp at |y| ≥ 8 is 0.0625.
- **Who is closer to exact:** the kernel. Measured against an exact fp32 reference, the kernel is within 0.0078 (one ulp), while HF itself is 0.031 away. **[obs]**
- **Fix:** the test should use a ulp- or relative tolerance. If HF bit-identity matters, for example for greedy-decode A/B runs, the kernel would have to reproduce the double rounding.

### P3: The GB10 inject script crashes before loading the model [obs]
- `GB10/qwen3_rmsnorm/inject_qwen3.py:122` reads `get_device_properties(0).total_mem`. On torch 2.13 that attribute is absent (`hasattr` returns False); the attribute is `total_memory`. The RTX copy does not have this line.
- The patch's sanity check (`:150-156`) compares shapes only, never values. **[src]**

### P3: Wrong hardware claims in comments and docs [obs]
| Claim | Location | Measured |
|---|---|---|
| "4 blocks/SM … 2048 threads/SM" | GB10 `rmsnorm.cu:24-26,49-53` | 1536 threads/SM, so 3 blocks of 512 |
| "Max threads/SM: 2048" | RTX header | 1536 |
| "Shared memory/SM: 128 KB (configurable up to 228 KB)" | RTX header | 100 KB (102,400 B) on both. 228 KB is a Hopper figure. |
| "unroll 8 … loop body runs 4 times — fully unrolled" | GB10 `:148-151` | The loop bound depends on runtime `blockDim`, so the pragma does not produce a fixed 4-iteration body. **[inf]** |

On GB10, 4096×4096 runs 9% faster with the 512 cap than with the 1024 cap (241.8 vs 268.6 µs), so the 512 choice is right, but the stated reason (4 blocks/SM) is not.

### P3: The project benchmark's timing measures launch overhead [obs]
- **What:** `benchmark_fn` times a Python loop with `perf_counter` around a sync. Below about 512 rows that measures how many kernels each side launches, not GPU time.
- **Evidence:**
  - The custom kernel issues one launch and eager issues about six. In the host loop that shows as 4.1 µs vs 27–31 µs.
  - In GPU time (graph replay), both sides are 1–3 µs.
- **The 4096×4096 bandwidth figure is also inflated on RTX.** The host loop reports 4,094 GB/s, which is above the RTX's 1.79 TB/s DRAM peak. The 64 MB working set sits in its 128 MB L2. **[obs]**

### Compliance: skill pre-flight checklist (both trees, identical) [obs]
| # | Check | Result |
|---|---|---|
| 1 | No pybind11 / `torch/extension.h` | ✗ `torch_binding.cpp:6,79-82` |
| 2 | `TORCH_LIBRARY_EXPAND` + `REGISTER_EXTENSION` only | ✗ none; no `torch_binding.h`, no `registration.h` |
| 3 | No hand-written `setup.py` | ✗ `CUDAExtension` / `BuildExtension` |
| 4 | `build.toml` dash name, `license`, `[torch]` section | ✗ `name = "qwen3_kernels"`; no license, no `[torch]`, no version |
| 5 | Kernel dir in git, committed | ✓ |
| 6 | `nix run .#build-and-copy` | not run (no nix); no `flake.nix` |
| 7 | `kernel-builder check-abi` | not run |
| – | Relative imports (`from ._ops import ops`) | ✗ `__init__.py:19` absolute `from qwen3_kernels._C` |
| – | `torch.compile` support (fake op) | ✗ 1 graph break, Dynamo cannot trace the pybind builtin |
| – | `flake.nix`, `CARD.md`, `tests/`, `benchmarks/` | ✗ all missing |

## Assumptions and open questions
- jarvis was serving a live engine during the run. The large ratios (1.79×, 6.3×) far exceed plausible contention noise, but the small-shape differences of 10–20% on jarvis are not robust.
- The comparison is against inductor with `max-autotune-no-cudagraphs`. Inside vLLM or SGLang the relevant baseline is their own fused RMSNorm (often fused with the residual add), which was not measured.
- No kernel-builder build was done, so ABI3 compliance is asserted only from source.

## Smallest next experiments
1. **Separate cause from fix for P1 (occupancy):**
   - Rebuild the RTX tree with `MAX_THREADS = 512` and time 4096×4096 on jarvis. Expect about 15.6 µs if occupancy is the cause.
   - Then try 384 (4 blocks/SM).
2. **QK-norm:** prototype a warp-per-row path for h ≤ 1024 and time 131072×128 on jarvis against inductor's 9.4 µs.

---

## Follow-up experiments (same day)

Scratch copies only; the user's kernel trees were not changed. GPU time per call (µs) uses CUDA-graph replay, the same method as above. Scripts and logs are in `followup-experiments/`.

### Experiment 1: thread cap sweep of the RTX tree (only `MAX_THREADS` changed; two runs each)

| rows × 4096 | cap 1024 | cap 512 | cap 384 | cap 256 |
|---|---|---|---|---|
| **RTX** 512 | 4.76 / 4.74 | 2.73 / 2.74 | 3.04 / 3.04 | 3.04 / 3.05 |
| **RTX** 4096 | 27.81 / 27.96 | 13.20 / 13.24 | 13.15 / 13.15 | **11.75 / 11.66** |
| **RTX** 16384 | 195.3 / 195.6 | 182.0 / 182.1 | 182.7 / 182.8 | 183.4 / 183.6 |
| **GB10** 512 | 15.70 / 15.70 | 8.02 / 8.00 | 8.27 / 8.29 | 7.77 / 7.79 |
| **GB10** 4096 | 306.5 / 307.5 | 302.6 / 301.5 | 301.7 / 302.9 | 304.9 / 307.1 |
| **GB10** 16384 | 1205 / 1217 | 1179 / 1174 | 1183 / 1183 | 1189 / 1188 |

**Verdict: occupancy is the cause. Confirmed.** [obs]
- **RTX:** dropping the cap from 1024 to 512 makes 4096×4096 2.1× faster, from 27.9 to 13.2 µs. That beats the predicted 15.6 µs, which was taken from the GB10 tree, whose `unroll 8` also differs.
- **Cap 256 is fastest on RTX.** 11.7 µs at 4096×4096 is below the 13.3 µs `torch.compile` figure from the first run. That comparison is across runs, so treat it as indicative.
- **Bandwidth-bound shapes:** at 16384 rows every cap except 1024 sits at the copy floor of about 182 µs.
- **GB10:** the cap matters only while the data is cache-resident. At 512 rows, cap 1024 is about 2× slower. At the DRAM-bound sizes all caps are within about 3% of each other.

### Experiment 2: warp-per-row prototype (`followup-experiments/warprow.cu`)

Design: one warp per row, 16-byte loads, the row held in registers so the input is read once, shuffle-only reduction (no shared memory, no `__syncthreads`), bf16, h ≤ 1024.

**Correctness:** maximum relative error against an exact fp32 reference is ≤ 0.0078, which is one bf16 ulp, on every shape on both GPUs. [obs]

| shape | RTX orig | RTX proto (best) | RTX compile | RTX copy | GB10 orig | GB10 proto (best) | GB10 compile | GB10 copy |
|---|---|---|---|---|---|---|---|---|
| 32 × 128 | 1.20 | 1.04 | **0.83** | 0.70 | 1.25 | **1.08** | 1.24 | 0.82 |
| 2048 × 128 | 1.98 | **1.36** | 1.45 | 0.76 | 3.04 | 1.63 | **1.55** | 1.38 |
| 131072 × 128 | 59.09 | 12.07 | **9.31** | 8.63 | 289.9 | 293.9 | 289.9 | 301.3 |
| 4096 × 512 | 4.50 | **2.30** | 2.59 | 1.64 | 15.63 | **6.40** | 7.16 | 5.77 |
| 4096 × 1024 | 8.62 | 3.89 | **3.59** | 2.65 | 35.19 | **19.21** | 24.75 | 20.43 |
| 65536 × 1024 | 181.8 | 182.2 | 181.9 | 183.0 | 1129 | 1183 | **1121** | 1172 |

**Verdict: the prototype fixes most of the QK-norm gap but does not beat inductor at 131072×128 on RTX.** [obs]
- **QK-norm on RTX:** the prototype is 4.9× faster than the original (59.1 → 12.1 µs) but still 30% slower than inductor's 9.3 µs. The prediction that it would beat inductor is **refuted**.
- **Where it wins:** it beats inductor at mid-size, cache-resident shapes on both GPUs, for example GB10 4096×1024 (19.2 vs 24.8 µs) and 4096×512 (6.4 vs 7.2 µs).
- **Where it ties:** at DRAM-bound shapes everything (original, prototype, inductor, copy) sits at the copy floor.
- **Likely cause of the remaining gap [inf]:** at h=128 only 16 of 32 lanes have data, since 128 / 8 bf16 per 16-byte load = 16. A half-warp-per-row layout (two rows per warp) is the obvious next test.

### Retraction: "GB10: kernel wins by about 15% at DRAM-bound sizes"

That claim does not reproduce.
- **First run:** the original kernel was 251.7 µs at 131072×128 against inductor's 289.9 µs.
- **This run:** it was 289.9 µs, and inductor was again 289.9 µs.
- **At 65536×1024 in this run:** the original kernel took 1129 µs and inductor 1121 µs, which is parity.

GB10 absolute times at DRAM-bound sizes drifted 15–25% between runs. The live engine on spark-5 logged no throughput lines in that window, so it was idle; the cause of the drift is **unknown**. Treat only comparisons made within one run as valid on GB10. Revised statement: **on GB10, at DRAM-bound sizes, the original kernel is at parity with inductor.** It loses at decode and cache-resident sizes.

---

## Follow-up 2: comparisons within a single run

Every candidate for a shape was captured as its own CUDA graph in **one process**, then replayed round-robin for 7 rounds. Numbers are medians (µs per call). Scripts and raw JSON are in `followup-samerun/`.

### Test A: thread cap vs `torch.compile`, same process

| shape (bf16) | cap 1024 | cap 512 | cap 256 | compile | copy |
|---|---|---|---|---|---|
| **RTX** 512×4096 | 4.74 | 2.74 | 3.03 | **2.35** | 1.62 |
| **RTX** 4096×4096 | 27.80 | 13.20 | **11.74** | 13.89 | 8.68 |
| **RTX** 16384×4096 | 180.6 | **165.7** | 171.4 | 180.9 | 165.8 |
| **GB10** 512×4096 | 15.74 | 8.20 | 7.92 | **7.48** | 5.78 |
| **GB10** 4096×4096 | 290.0 | 283.3 | 280.8 | 286.7 | 288.0 |
| **GB10** 16384×4096 | 1199 | 1140 | 1175 | **1137** | 1177 |

**Verdict [obs]:**
- **RTX, 4096×4096:** cap 256 beats `torch.compile` by 15% in the same run (11.74 vs 13.89 µs). It also beats the best `torch.compile` time from any run (13.33 µs). **Confirmed.**
- **RTX, 16384×4096:** cap 512 runs at the copy floor, and `torch.compile` is 9% slower.
- **RTX, 512×4096:** `torch.compile` still wins at this cache-resident decode size.
- **GB10, 4096×4096:** all candidates fall within one another's min–max spread, so this is parity.
- **GB10 overall:** cap 1024 is the only clear loser.

**Recommended cap [inf]:** 512 when you don't know the shape, since it is never worse than 4% from the best cap. Use 256 for mid-size RTX shapes.

### Test B: rows per warp at h=128 (L = threads per row; t = threads per block)

| shape | L32 | **L16** | L8 | orig | compile (this run) | compile (best any run) | copy |
|---|---|---|---|---|---|---|---|
| **RTX** 131072×128 | 12.30 | **9.44** | 9.90 | 59.19 | 22.12 | 9.31 | 8.80 |
| **RTX** 2048×128 | 1.28 | **1.21** | 1.42 | 1.87 | 1.20 | 1.16 | 0.77 |
| **RTX** 32×128 | 1.02 | **0.99** | 1.32 | 1.19 | 0.81 | 0.81 | 0.69 |
| **RTX** 4096×512 | **2.26** | 2.77 | 4.29 | 4.51 | 2.16 | 2.16 | 1.60 |
| **RTX** 4096×1024 | **3.84** | 4.68 | 6.96 | 8.65 | 3.59 | 3.59 | 2.66 |
| **GB10** 131072×128 | 301.6 | 302.8 | 310.3 | **287.7** | 293.2 | 289.9 | 311.9 |
| **GB10** 4096×512 | **6.69** | 7.04 | 7.21 | 15.94 | 7.83 | 7.16 | 6.28 |
| **GB10** 4096×1024 | **17.14** | 20.05 | 24.47 | 35.42 | 54.11 | 24.75 | 22.09 |

Each L column shows the better of t=128 and t=256. **Correctness:** the maximum relative error against an exact fp32 reference is ≤ 0.0078 (one bf16 ulp) for every L, shape and GPU. [obs]

**Verdict [obs]:**
- **RTX QK-norm:** 16 threads per row closes the gap. At 131072×128 it takes 9.44 µs, against 12.1–12.3 µs with 32 threads per row. That is within 1.5% of the best `torch.compile` time from any run (9.31 µs) and 7% above the copy floor. It is 6.3× faster than the original kernel.
- **Choosing threads per row:** the best value is one 16-byte load per thread, so threads per row = h/8, capped at 32. That gives 16 for h=128 and 32 for h ≥ 256.
- **GB10 at DRAM-bound sizes:** everything is within about 8%, and the original kernel has the lowest median. There is no win to find there.

### New caveat: the `torch.compile` baseline itself varies between runs [obs]

`mode="max-autotune-no-cudagraphs"` compiled the same function to very different speeds in different runs:

| GPU, shape | Run 1 | Run 2 |
|---|---|---|
| RTX, 131072×128 | 9.31 µs | 22.12 µs (spread 22.09–22.15, so not noise within the run) |
| GB10, 4096×1024 | 24.75 µs | 54.11 µs |

**Likely cause [inf]:** autotuning picks kernel configurations by timing them on a GPU shared with the live engine, so a noisy tuning pass can lock in a slow configuration. I did not verify this. Fixing the inductor config, or caching the autotune result, would remove the variance. Every verdict above uses both the same-run and the best-ever `torch.compile` time.
