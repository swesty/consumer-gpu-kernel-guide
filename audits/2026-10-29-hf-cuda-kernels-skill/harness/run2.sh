#!/usr/bin/env bash
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
for v in GB10 rtx-6000-pro; do
  python3 /k/graph.py /k/src/$v/qwen3_rmsnorm 2>/dev/null | grep RESULT | sed "s/^/GRAPH $v /"
  (cd /k/src/$v/qwen3_rmsnorm && PYTHONPATH=torch-ext python3 benchmark_rmsnorm.py --dtype bfloat16 2>&1 | grep -A5 "Correctness" | sed "s/^/OWNBENCH $v /")
done
python3 -c "import torch; p=torch.cuda.get_device_properties(0); print('HAS_total_mem', hasattr(p,'total_mem'), 'HAS_total_memory', hasattr(p,'total_memory'))"
