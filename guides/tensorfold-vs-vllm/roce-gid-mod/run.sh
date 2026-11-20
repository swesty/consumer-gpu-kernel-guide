#!/usr/bin/env bash
# b12x-roce-gid-auto — let vLLM's RoCEnante all-reduce come up on DGX Spark pairs.
#
# STATUS: Fix. Without it, VLLM_ENABLE_ROCE_ALLREDUCE=1 silently falls back to PyNCCL.
# WHY: eugr's cluster env sets NCCL_IB_GID_INDEX=-1 (NCCL auto-select). b12x reads
# that as a literal GID index and ibv_query_gid(-1) fails with EINVAL on every rank.
# The right index also differs per node (spark-3: 5, spark-4: 4), so a fixed
# B12X_ROCE_GID_INDEX in the shared recipe cannot work. This patch makes a negative
# or unset index auto-select the IPv4 RoCE v2 GID per node, like NCCL.
# RUNTIME EVIDENCE: "[B12X-ROCE-GID-AUTO] gid_index=N on <hcas>" and then
# "Using RoCEnante (b12x one-shot RoCE collectives)" in the log.
#
# VALIDATED AGAINST: b12x 1.5.0 (roce_oneshot.py sha256 572429e6...),
# vLLM 0.1.dev21553+gab86b7073, image glm53-b12x:ab86b70 / deepseek0731-b12x:20261003.
#
# ROLLBACK: remove this mod from the recipe and recreate the container
# (RoCEnante then fails to initialise again and vLLM keeps PyNCCL).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SITE=/usr/local/lib/python3.12/dist-packages

python3 "$HERE/patch_roce_gid_auto.py" --site-root "$SITE"
python3 -m py_compile "$SITE/b12x/comm/roce/roce_oneshot.py"

echo "[b12x-roce-gid-auto] OK"
