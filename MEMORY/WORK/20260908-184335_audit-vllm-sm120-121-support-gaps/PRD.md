---
task: Audit method for vLLM sm120/121 kernel support gaps
slug: 20260908-184335_audit-vllm-sm120-121-support-gaps
effort: standard
phase: complete
progress: 15/15
mode: interactive
started: 2026-09-08T22:43:35Z
updated: 2026-09-08T22:52:00Z
---

## Context

Shawn asked for a *method* to audit the vLLM repository to find where sm_120 (RTX 5090 /
RTX 6000 Pro, GB202) and sm_121 (GB10, DGX Spark) support is lacking. The deliverable is
an audit procedure with runnable commands, not the completed audit.

This repo (consumer-gpu-kernel-guide) already establishes the compile-target framing:
`sm_120` cubins are forward-compatible across Blackwell minor revisions but **do not run
on sm_121**; `sm_121`/`sm_121a` require CUDA 12.9+. That asymmetry is the core reason an
sm120-vs-sm121 audit must be two separate audits sharing one procedure.

Constraint discovered in OBSERVE: there is no vLLM source checkout on this Mac
(spotlight-search returned only vault notes and browser cache). Therefore the audit
method must be written so that its commands are self-verifying — they fail loudly if a
path has moved — rather than asserting current vLLM file contents from recall.

### Risks

- Recall-vs-verified confusion: vLLM's tree moves fast; any path I name from memory may
  be stale. Mitigation: label every path as expected-not-verified, and make step 0 of the
  audit a path-existence check.
- Four-gate model may be incomplete — a kernel can also be gated by a runtime env var or
  a model-config predicate, not just build/source/dispatch/dep.
- "Support" is ambiguous: compiles vs. runs vs. runs fast. Audit must grade all three.
- sm_121 hardware access is required to close the loop; a static audit alone cannot
  distinguish "dispatches to Triton fallback" from "dispatches to CUTLASS".

## Criteria

- [x] ISC-1: Method names vLLM's CMake arch-gating mechanism
- [x] ISC-2: Method covers per-target arch intersection filtering
- [x] ISC-3: Method covers __CUDA_ARCH__ source guards under csrc/
- [x] ISC-4: Method covers Python runtime capability dispatch gates
- [x] ISC-5: Method covers third-party dep arch coverage
- [x] ISC-6: Method distinguishes sm_120 from sm_121 binary compatibility
- [x] ISC-7: Provides copy-pasteable grep/rg commands
- [x] ISC-8: Identifies exact-equality capability checks as sm121 bug class
- [x] ISC-9: Defines a support-matrix output artifact schema
- [x] ISC-10: Names a runtime probe confirming actual kernel dispatch
- [x] ISC-11: States which fleet host provides sm_121 hardware
- [x] ISC-12: Labels recalled facts separately from verified facts
- [x] ISC-15: Method includes cuobjdump inspection of built artifacts
- [x] ISC-A1: Does not modify existing repo guide content
- [x] ISC-A2: Does not assert vLLM line numbers without reading source

## Decisions

- Deliver the audit as a five-gate procedure rather than a file-by-file walkthrough,
  because the file list is volatile and the gate model is not.
- Do not clone vLLM in this run — the user asked for the method, and cloning plus a real
  audit exceeds the Standard tier SLA. Offer it as a follow-up instead.

## Verification

Delivered a six-gate audit procedure (build / source / dispatch / dependency /
built-artifact / runtime probe) with copy-pasteable commands.

- ISC-A1 verified by `git status --porcelain` → only `?? MEMORY/` untracked; no guide
  file modified.
- ISC-A2 verified by inspection: no vLLM line numbers asserted; all vLLM paths carry an
  explicit recalled-not-verified label, and every command fails loudly on a moved path.
- ISC-11 verified against `homelab-build/RUNBOOK.md`: GB10/sm_121a on spark-1 (:1244),
  spark-3 (:31, :1299), spark-4 (:32, :1402); sm_120 on jarvis (:33) and arasaka (:2313).
- Known sm_121 defects cited: DeepGEMM `Unknown SF transformation` (RUNBOOK:1377-1380);
  GDN hoist sufficient on sm_120/sm_86 but not GB10 sm_121a (RUNBOOK:336-337).
- Capability invocations: Skill(FirstPrinciples) and Skill(HOMELAB), both real tool calls.

## Audit run (iteration 2) — vLLM @ 60ad959b6f1a5c8f602edbd608c8decbc0788c50

Gates 0-4 executed. Gates 5-6 not run (need CUDA + hardware; user chose local-only).

Findings, ranked:
1. QuTLASS NVFP4 — Python floor `has_device_capability(100)` selects it on sm_120/121
   (compressed_tensors/transform/linear.py:52-54), but qutlass.cmake:71-79 drops SM120
   whenever SM100 is also requested (single TARGET_CUDA_ARCH selector). Affects official
   multi-arch wheels. Workaround: build targeting only 12.0.
2. CUTLASS grouped-GEMM FP8/INT8 MoE — excluded on all consumer Blackwell.
   _custom_ops.py:864 (`>= 110`) + no grouped_mm_c3x_sm120.cu. Error-string asymmetry at
   scaled_mm_entry.cu:303 vs :332/352/378. Degrades to Triton.
3. No dense MLA for major==12 at kernel (CMakeLists.txt:1098), KV-cache
   (cache_kernels.cu:978,1720), or backend-selection layer. FlashMLA has no 12.x at all
   (flashmla.cmake:55-67). Falls back to Triton MLA.
4. cutlass_scaled_mm_azp unbounded `>= 90` (scaled_mm_entry.cu:427) vs bounded sibling
   (:238) — misroutes Blackwell into the Hopper kernel. Real bug.
5. FlashKDA missing 12.1a on CUDA 12.9-<13.0 (flashkda.cmake:32) — THE ONLY genuine
   sm_121-specific gap in the entire audit. Siblings all include 12.1a (deepgemm.cmake:80).
6. NVFP4 MoE b12x withheld from auto-selection (fused_moe/oracle/nvfp4.py:192-194).
7. DeepGEMM on by default on GB10 (cuda.py:711-717) — matches RUNBOOK.md:1377-1380 crash,
   not re-verified at this SHA.
8. family(100) where comment says SM100+/SM90+: mamba_mixer2.py:558, gate_linear.py:49-50.
9. Dead enable_sm120_only template, `__CUDA_ARCH__ == 1200` (common.hpp:118).

Corrections made during the run:
- I claimed sm_120 cubins do not run on sm_121. False; GPU-COMPARISON.md:83-88 measured
  the opposite. vLLM's whole 12.0f strategy depends on that forward compatibility.
- I predicted `== 120` predicates would dominate. Zero exist in Python (~170 sites checked).
- I stated zero sm_121-specific gaps. One exists: FlashKDA (finding 5).
- Context above (line ~20) says sm_121/sm_121a require CUDA 12.9+. Superseded 2026-09-09:
  NVIDIA gives CUDA 13.0 as the DGX Spark toolkit floor (vault
  Wiki-GPU/wiki/concepts/cuda-target-families-and-compute-capability.md:49,77, citing
  docs.nvidia.com/cutlass/latest/overview.html). The 12.9 figure belongs to the `f`
  family-target class (PTX ISA 8.8), not to sm_121. Guide repo corrected in 8 places.

## Patch triage (2026-09-09)

Upstream SM12.x workstream found via gh — changes the plan:
- #41215 MERGED 2026-05-20 "Use enable_sm120_family for per-tensor FP8 CUTLASS kernels on
  SM12.1" — fixed issue #40758, a GB10 FP8 crash (`cutlass_gemm_caller ... Error Internal`)
  caused by `enable_sm120_only`'s `__CUDA_ARCH__ == 1200` guard. So finding 9 is dead code
  BECAUSE this merged; the `== 120` bug class I predicted was real and live on GB10.
- #35568 MERGED "Fix SM121 (DGX Spark) exclusion from Marlin/CUTLASS FP8 paths"
- #38338 OPEN  "Add Triton reference implementation of NVFP4 GEMM" (SM12.x fallback pattern)
- #41028 OPEN  "OAITritonExperts MXFP4: include SM 12.x in supported device range"
- #41062 CLOSED "DeepGEMM MoE: extend device gates to SM 12.x consumer Blackwell"

Fix classes:
  A (one-line, write today): finding 4 azp bound, finding 5 flashkda 12.1a, finding 9 delete
    enable_sm120_only.
  B (small patch, validate on hardware first): finding 1 QuTLASS gate/build, finding 7
    DeepGEMM (see #41062 closed — find out why), finding 8 family(100) widenings.
  C (kernel work, scaffolding already present): finding 2 grouped_mm_c3x_sm120.cu. Gating
    unknown = whether CUTLASS 4.7.1 has a PtrArray collective for arch::Sm120.
  D (not ours): finding 3 MLA on SM12x, finding 6 b12x auto-selection (upstream CUTLASS).
