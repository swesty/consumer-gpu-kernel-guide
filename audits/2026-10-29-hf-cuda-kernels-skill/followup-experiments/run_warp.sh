#!/usr/bin/env bash
# Only library headers (cusparse/cublas/...) from the pip wheel; nvcc's own crt/ headers stay authoritative.
mkdir -p /tmp/libinc; for f in /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/{cusparse,cublas,cusolver,cufft,curand}*.h; do ln -sf $f /tmp/libinc/; done
export CPATH=/tmp/libinc TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
python3 /k/exp.py warp 2>/k/out/warp.err | grep RESULT | sed "s/^/WARP /"
grep -E "error" /k/out/warp.err | head -5
echo DONE
