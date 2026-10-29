#!/usr/bin/env bash
# Inside container. /k/src/rtx-6000-pro = pristine copy of the user's RTX tree.
set -u
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
cd /k
for cap in 1024 512 384 256; do
  d=/k/caps/c$cap; rm -rf $d; mkdir -p $d; cp -r /k/src/rtx-6000-pro/qwen3_rmsnorm $d/
  sed -i "s/^constexpr int MAX_THREADS = 1024;/constexpr int MAX_THREADS = $cap;/" $d/qwen3_rmsnorm/kernel_src/rmsnorm.cu
  echo "cap $cap: $(grep -c "MAX_THREADS = $cap;" $d/qwen3_rmsnorm/kernel_src/rmsnorm.cu) line(s) set"
  (cd $d/qwen3_rmsnorm && python3 setup.py build_ext --inplace > /k/out/build-c$cap.log 2>&1 || echo "BUILD FAIL c$cap")
done
for rep in 1 2; do
  for cap in 1024 512 384 256; do python3 /k/exp.py cap /k/caps/c$cap/qwen3_rmsnorm cap$cap 2>/dev/null | grep RESULT | sed "s/^/CAP rep$rep /"; done
done
# baseline build of the untouched RTX tree for exp 2's "orig" column
(cd /k/src/rtx-6000-pro/qwen3_rmsnorm && python3 setup.py build_ext --inplace > /k/out/build-orig.log 2>&1 || echo "BUILD FAIL orig")
python3 /k/exp.py warp 2>/k/out/warp.err | grep RESULT | sed "s/^/WARP /"
grep -h "registers" /k/out/warp.err | sort -u | head -5
tail -c 400 /k/out/warp.err | grep -i error
echo DONE
