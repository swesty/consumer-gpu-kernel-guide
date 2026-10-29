# Kernels generated during the 2026-10-29 audit

All are experimental. None is wired into `GB10/` or `rtx-6000-pro/`, and those trees are unchanged. Timings and verdicts are in `../README.md`.

| File | What it is | Measured outcome |
|---|---|---|
| `rmsnorm_rtx_cap512.cu` | `rtx-6000-pro/.../rmsnorm.cu` with `MAX_THREADS = 512` (the only change) | RTX 4096×4096: 13.2 µs vs 27.8 µs at cap 1024; at the copy floor at 16384×4096 |
| `rmsnorm_rtx_cap384.cu` | same, `MAX_THREADS = 384` | ≈ cap 512; no advantage |
| `rmsnorm_rtx_cap256.cu` | same, `MAX_THREADS = 256` | RTX 4096×4096: **11.74 µs vs `torch.compile` 13.89 µs** in the same run |
| `rmsnorm_warp_per_row_bf16.cu` | Prototype 1: one warp per row, 16-byte loads, row kept in registers, shuffle-only reduction, h ≤ 1024 | RTX QK 131072×128: 12.1 µs vs 59.1 µs for the original; wins at mid shapes |
| `rmsnorm_subwarp_bf16.cu` | Prototype 2: L = 32/16/8 threads per row (`lanes` argument) | L = 16 at h=128: **9.44 µs**, level with the best `torch.compile` time (9.31 µs). Rule: threads per row = h/8, capped at 32 |

Build notes:
- The cap variants are drop-in replacements for `kernel_src/rmsnorm.cu`, built with the tree's existing `setup.py`.
- The prototypes are self-contained torch extensions, built with `torch.utils.cpp_extension.load_inline`; see `../followup-samerun/samerun.py`. They use pybind via `load_inline`, so they are **not** kernel-builder / Hub compliant.
- In `vllm/vllm-openai` images, put on `CPATH` a directory holding symlinks to only the cusparse/cublas/cusolver/cufft/curand headers from `site-packages/nvidia/cu13/include`. Pointing `CPATH` at the whole directory breaks `load_inline` (a `crt/host_runtime.h` mismatch).

All variants are correct to within one bf16 ulp of an exact fp32 reference on GB10 (sm_121) and RTX PRO 6000 (sm_120).
