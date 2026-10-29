# Audit log

| Date | Audit | Summary |
|---|---|---|
| 2026-10-29 | [HF `cuda-kernels` skill audit of `qwen3_rmsnorm`](2026-10-29-hf-cuda-kernels-skill/README.md) | Arithmetic correct. Binding unsafe (out-of-bounds write, NaN, crashes). Not Hub-compliant. Both GPUs have 1536 threads/SM, so the 1024-thread RTX tuning is 2.1× slower. Cap 256 beats `torch.compile` at 4096² on RTX; a 16-threads-per-row prototype ties inductor's best on QK-norm. Generated kernels are in `kernels/`. |
