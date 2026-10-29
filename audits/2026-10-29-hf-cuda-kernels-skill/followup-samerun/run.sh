#!/usr/bin/env bash
# Inside container. Only library headers from the pip wheel go on CPATH (nvcc's own crt/ stays authoritative).
set -u
mkdir -p /tmp/libinc; for f in /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/{cusparse,cublas,cusolver,cufft,curand}*.h; do ln -sf $f /tmp/libinc/; done
export CPATH=/tmp/libinc TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
SOS=()
for cap in 1024 512 256; do
  d=/k/caps/c$cap; rm -rf $d; mkdir -p $d; cp -r /k/src/rtx-6000-pro/qwen3_rmsnorm $d/
  sed -i "s/^constexpr int MAX_THREADS = 1024;/constexpr int MAX_THREADS = $cap;/" $d/qwen3_rmsnorm/kernel_src/rmsnorm.cu
  echo "cap $cap set: $(grep -c "MAX_THREADS = $cap;" $d/qwen3_rmsnorm/kernel_src/rmsnorm.cu)"
  (cd $d/qwen3_rmsnorm && python3 setup.py build_ext --inplace > /k/out/build-c$cap.log 2>&1 || echo "BUILD FAIL c$cap")
  SOS+=("$(ls $d/qwen3_rmsnorm/torch-ext/qwen3_kernels/_C*.so)")
done
python3 /k/samerun.py "${SOS[@]}" > /k/out/samerun.out 2> /k/out/samerun.err
grep RESULT /k/out/samerun.out || { echo "NO RESULT"; grep -iE "error" /k/out/samerun.err | head -8; }
echo DONE
