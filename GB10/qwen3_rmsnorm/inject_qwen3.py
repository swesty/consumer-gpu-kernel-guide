#!/usr/bin/env python3
"""
Inject vectorized RMSNorm kernel into Qwen3-8B (HuggingFace transformers).

Target: NVIDIA GB10 (DGX Spark) — 128 GB unified memory fits Qwen3-8B easily.

Patches all Qwen3RMSNorm modules (145 total in Qwen3-8B):
  - 36 x input_layernorm          (hidden_size=4096)
  - 36 x post_attention_layernorm (hidden_size=4096)
  -  1 x model.norm               (hidden_size=4096)
  - 36 x self_attn.q_norm         (head_dim=128)
  - 36 x self_attn.k_norm         (head_dim=128)

Usage:
    pip install -e .
    python inject_qwen3.py
    python inject_qwen3.py --model Qwen/Qwen3-8B --prompt "Explain quantum computing"
    python inject_qwen3.py --benchmark --max-new-tokens 100

    # Compare against baseline:
    python inject_qwen3.py --benchmark --max-new-tokens 100
    python inject_qwen3.py --benchmark --max-new-tokens 100 --no-patch
"""

import argparse
import time

import torch
import torch.nn as nn

from qwen3_kernels import rmsnorm


# ---------------------------------------------------------------------------
# Kernel injection
# ---------------------------------------------------------------------------

def patch_rmsnorm_modules(model: nn.Module, verbose: bool = False) -> int:
    """
    Patch all RMSNorm modules in a Qwen3 model to use the custom CUDA kernel.

    Handles both Qwen3RMSNorm (variance_epsilon) and any future variants.

    Returns:
        Number of modules patched.
    """
    patched = 0

    for name, module in model.named_modules():
        class_name = type(module).__name__

        if "RMSNorm" not in class_name:
            continue

        # Qwen3 uses 'variance_epsilon', fallback to 'eps'
        eps = getattr(module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(module, "eps", 1e-6)

        has_weight = hasattr(module, "weight") and module.weight is not None

        if has_weight:
            def make_forward(mod, epsilon):
                def forward(hidden_states):
                    return rmsnorm(hidden_states, mod.weight, eps=epsilon)
                return forward

            module.forward = make_forward(module, eps)
            patched += 1

            if verbose:
                dim = module.weight.shape[0]
                print(f"  Patched {name} ({class_name}, dim={dim}, eps={eps})")
        else:
            if verbose:
                print(f"  SKIP {name} ({class_name}) — no weight")

    return patched


def inject_optimized_kernels(model, verbose: bool = False) -> dict:
    """
    Inject all custom CUDA kernels into a Qwen3 model.

    Call AFTER loading model to CUDA.

    Returns:
        dict with patching statistics.
    """
    stats = {"rmsnorm_modules": patch_rmsnorm_modules(model, verbose=verbose)}
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inject custom RMSNorm kernel into Qwen3-8B (GB10 DGX Spark)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model ID (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument("--prompt", default="What is the theory of relativity?")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run generation benchmark")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each patched module")
    parser.add_argument("--no-patch", action="store_true",
                        help="Skip kernel injection (baseline run)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 64)
    print(f"Qwen3 RMSNorm Kernel Injection — GB10 (DGX Spark)")
    print(f"Model: {args.model}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"VRAM:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print("=" * 64)

    # Load model — GB10 has 128 GB unified memory, Qwen3-8B fits in BF16 (~16 GB)
    print(f"\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Count RMSNorm modules
    norm_count = sum(
        1 for _, m in model.named_modules() if "RMSNorm" in type(m).__name__
    )
    print(f"   Found {norm_count} RMSNorm modules")

    # Inject kernels
    if not args.no_patch:
        print("\n2. Injecting custom RMSNorm kernel...")
        stats = inject_optimized_kernels(model, verbose=args.verbose)
        print(f"   Patched: {stats['rmsnorm_modules']} modules")
    else:
        print("\n2. Skipping kernel injection (baseline mode)")

    # Verify forward pass
    print("\n3. Verifying forward pass...")
    x = torch.randn(1, 4, model.config.hidden_size,
                     device="cuda", dtype=torch.bfloat16)
    for name, module in model.named_modules():
        if "RMSNorm" in type(module).__name__:
            out = module(x)
            print(f"   {name}: {x.shape} -> {out.shape}  OK")
            break

    # Generate
    print(f"\n4. Generating text...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Timed generation
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = generated_tokens / elapsed

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Prompt:  {args.prompt}")
    print(f"   Output:  {text}")
    print(f"   Tokens:  {generated_tokens} in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

    if args.benchmark:
        print(f"\n{'Benchmark (5 runs)':=^64}")
        times = []
        for i in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            tps = generated_tokens / times[-1]
            print(f"  Run {i+1}: {times[-1]:.3f}s  ({tps:.1f} tok/s)")

        avg = sum(times) / len(times)
        avg_tps = generated_tokens / avg
        print(f"  Average: {avg:.3f}s  ({avg_tps:.1f} tok/s)")

    mode = "CUSTOM KERNEL" if not args.no_patch else "BASELINE"
    print(f"\n[{mode}] Done.")


if __name__ == "__main__":
    main()
