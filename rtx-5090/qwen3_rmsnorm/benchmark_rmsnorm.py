#!/usr/bin/env python3
"""
Micro-benchmark: Vectorized RMSNorm kernel vs PyTorch baseline.

Tests Qwen3-8B specific shapes on RTX 5090 Blackwell:
  - hidden_size=4096  (input_layernorm, post_attention_layernorm, final norm)
  - head_dim=128      (QK norm in attention)

Usage:
    pip install -e .
    python benchmark_rmsnorm.py
    python benchmark_rmsnorm.py --dtype float16
    python benchmark_rmsnorm.py --dtype float32
"""

import argparse
import time

import torch
import torch.nn

# ---------------------------------------------------------------------------
# PyTorch reference RMSNorm (matches Qwen3RMSNorm implementation)
# ---------------------------------------------------------------------------

def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation matching Qwen3RMSNorm.forward()."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def benchmark_fn(fn, *args, warmup: int = 50, iterations: int = 200):
    """Time a CUDA function with warmup and sync."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # ms


def compute_bandwidth(shape, dtype, time_ms):
    """Compute effective memory bandwidth in GB/s."""
    numel = 1
    for s in shape:
        numel *= s
    hidden_size = shape[-1]

    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    # Read input + weight, write output
    # input: numel, weight: hidden_size (broadcast), output: numel
    total_bytes = (2 * numel + hidden_size) * bytes_per_elem
    bandwidth_gbs = (total_bytes / 1e9) / (time_ms / 1000)
    return bandwidth_gbs


def main():
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm for Qwen3-8B")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16", help="Data type (default: bfloat16)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 72)
    print(f"RMSNorm Micro-benchmark — Qwen3-8B on {gpu_name}")
    print(f"dtype: {args.dtype}  |  warmup: {args.warmup}  |  iterations: {args.iterations}")
    print("=" * 72)

    # Import custom kernel
    try:
        from qwen3_kernels import rmsnorm as custom_rmsnorm
        has_custom = True
    except ImportError:
        print("\nWARNING: qwen3_kernels not installed. Run: pip install -e .")
        print("         Showing PyTorch baseline only.\n")
        has_custom = False

    # Qwen3-8B shapes: [batch, seq_len, hidden_size]
    # Layer norms: hidden_size=4096
    # QK norms: head_dim=128 (applied per-head)
    shapes = [
        # Layer norm shapes (typical inference/training batches)
        (1, 1, 4096),          # Single token decode
        (1, 128, 4096),        # Short sequence
        (1, 512, 4096),        # Medium sequence
        (1, 2048, 4096),       # Long sequence
        (1, 8192, 4096),       # Very long sequence (Qwen3 supports 32k+)
        (4, 512, 4096),        # Batch inference
        (4, 2048, 4096),       # Batch long sequence
        # QK norm shapes (head_dim=128, applied after reshape)
        (1, 2048, 128),        # QK norm single sequence
        (4, 2048, 128),        # QK norm batched
    ]

    # RTX 5090 Blackwell theoretical bandwidth
    PEAK_BW_GBS = 1792  # 1.792 TB/s

    header = f"{'Shape':>22s} | {'Custom (ms)':>12s} | {'PyTorch (ms)':>12s} | {'Speedup':>8s} | {'BW (GB/s)':>10s} | {'% Peak':>7s}"
    print(f"\n{header}")
    print("-" * len(header))

    eps = 1e-6

    for shape in shapes:
        x = torch.randn(shape, dtype=dtype, device=device)
        w = torch.ones(shape[-1], dtype=dtype, device=device)

        # PyTorch baseline
        pt_ms = benchmark_fn(pytorch_rmsnorm, x, w, eps,
                             warmup=args.warmup, iterations=args.iterations)

        if has_custom:
            # Custom kernel
            custom_ms = benchmark_fn(custom_rmsnorm, x, w, eps,
                                     warmup=args.warmup, iterations=args.iterations)
            speedup = pt_ms / custom_ms
            bw = compute_bandwidth(shape, dtype, custom_ms)
            pct_peak = (bw / PEAK_BW_GBS) * 100

            shape_str = f"{list(shape)}"
            print(f"{shape_str:>22s} | {custom_ms:>12.4f} | {pt_ms:>12.4f} | {speedup:>7.2f}x | {bw:>9.1f} | {pct_peak:>6.1f}%")
        else:
            bw = compute_bandwidth(shape, dtype, pt_ms)
            pct_peak = (bw / PEAK_BW_GBS) * 100
            shape_str = f"{list(shape)}"
            print(f"{shape_str:>22s} | {'N/A':>12s} | {pt_ms:>12.4f} | {'N/A':>8s} | {bw:>9.1f} | {pct_peak:>6.1f}%")

    if has_custom:
        # Correctness verification
        print(f"\n{'Correctness Verification':=^72}")
        for shape in [(1, 512, 4096), (4, 2048, 4096), (1, 2048, 128)]:
            x = torch.randn(shape, dtype=dtype, device=device)
            w = torch.randn(shape[-1], dtype=dtype, device=device).abs() + 0.1

            ref = pytorch_rmsnorm(x, w, eps)
            out = custom_rmsnorm(x, w, eps)

            max_diff = (ref - out).abs().max().item()
            rel_err = ((ref - out).abs() / (ref.abs() + 1e-8)).max().item()

            status = "PASS" if max_diff < 0.01 else "FAIL"
            print(f"  {list(shape)}: max_diff={max_diff:.6f}  rel_err={rel_err:.6f}  [{status}]")

    print()


if __name__ == "__main__":
    main()
