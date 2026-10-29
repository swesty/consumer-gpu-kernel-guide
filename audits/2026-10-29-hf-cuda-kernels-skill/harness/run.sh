#!/usr/bin/env bash
# Runs inside the container. /k = audit dir with copies of both kernel variants.
set -u
cd /k
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include TORCH_CUDA_ARCH_LIST="${ARCH_LIST}" NVCC_APPEND_FLAGS="-Xptxas=-v"
for v in GB10 rtx-6000-pro; do
  echo "##### BUILD $v"
  (cd /k/src/$v/qwen3_rmsnorm && python3 setup.py build_ext --inplace > /k/out/build-$v.log 2>&1; echo "exit=$?")
  grep -E "Used [0-9]+ registers|error" /k/out/build-$v.log | sed 's/^/  /' | head -20
  # build_ext --inplace drops _C*.so next to setup.py's package_dir target
  find /k/src/$v -name "_C*.so" | sed 's/^/  so: /'
done
python3 /k/audit.py /k/src/GB10/qwen3_rmsnorm props
for v in GB10 rtx-6000-pro; do
  echo "##### $v correct"; python3 /k/audit.py /k/src/$v/qwen3_rmsnorm correct > /k/out/correct-$v.json 2>/k/out/correct-$v.err; tail -c 300 /k/out/correct-$v.err
  for c in zero_rows weight_fp32_input_bf16 out_fp32_input_bf16 out_too_small noncontig_out misaligned_bf16 misaligned_fp32 torch_compile; do
    python3 /k/audit.py /k/src/$v/qwen3_rmsnorm edge $c 2>&1 | grep RESULT | sed "s/^/  $v /" || echo "  $v $c: process died"
  done
done
for v in GB10 rtx-6000-pro; do
  echo "##### $v perf"; python3 /k/audit.py /k/src/$v/qwen3_rmsnorm perf > /k/out/perf-$v.json 2>/k/out/perf-$v.err; tail -c 300 /k/out/perf-$v.err
done
echo DONE
