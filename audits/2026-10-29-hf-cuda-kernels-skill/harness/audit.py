"""RMSNorm kernel audit: correctness, edge cases, timing.

Usage (inside a CUDA container, variant already built in place):
  python audit.py <variant_dir> props|correct|perf
  python audit.py <variant_dir> edge <case>     # one edge case per process
Prints one JSON line prefixed with RESULT.
"""
import json
import sys

import torch

vdir, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{vdir}/torch-ext")
from qwen3_kernels import rmsnorm  # noqa: E402

dev = torch.device("cuda")
EPS = 1e-6


def hf_ref(x, w, eps=EPS):
    # transformers Qwen3RMSNorm: fp32 normalize, round to input dtype, then * weight
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(x.dtype)


def exact_ref(x, w, eps=EPS):
    # single rounding at the end: what the kernel intends to compute
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype)


def ulp_err(a, b):
    # max |a-b| measured in units of the output dtype's spacing at |b|
    a, b = a.float(), b.float()
    spacing = torch.finfo(torch.float32).eps if a.dtype == torch.float32 else None
    d = (a - b).abs()
    return d.max().item()


def out(res):
    print("RESULT " + json.dumps(res), flush=True)


if mode == "props":
    p = torch.cuda.get_device_properties(0)
    out({
        "name": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
        "max_threads_per_sm": p.max_threads_per_multi_processor,
        "l2_bytes": p.L2_cache_size,
        "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", None),
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", None),
        "torch": torch.__version__, "cuda": torch.version.cuda,
    })

elif mode == "correct":
    torch.manual_seed(0)
    rows_out = []
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        for h in (4096, 128, 4095, 4098, 64, 32, 2):
            for rows in (1, 7, 513):
                x = (torch.randn(rows, h, device=dev) * 3).to(dt)
                w = (torch.rand(h, device=dev) + 0.5).to(dt)
                y = rmsnorm(x, w, eps=EPS)
                torch.cuda.synchronize()
                ex, hf = exact_ref(x, w), hf_ref(x, w)
                d_ex = (y.float() - ex.float()).abs()
                rel_ex = (d_ex / ex.float().abs().clamp_min(1e-3)).max().item()
                rows_out.append({
                    "dtype": str(dt).split(".")[-1], "h": h, "rows": rows,
                    "max_abs_vs_exact": d_ex.max().item(),
                    "max_rel_vs_exact": rel_ex,
                    "max_abs_vs_hf": (y.float() - hf.float()).abs().max().item(),
                    "hf_vs_exact": (hf.float() - ex.float()).abs().max().item(),
                    "finite": bool(torch.isfinite(y).all().item()),
                })
    out({"cases": rows_out})

elif mode == "edge":
    case = sys.argv[3]
    res = {"case": case}
    try:
        if case == "zero_rows":
            x = torch.empty(0, 4096, device=dev, dtype=torch.bfloat16)
            w = torch.ones(4096, device=dev, dtype=torch.bfloat16)
            y = rmsnorm(x, w)
            torch.cuda.synchronize()
            # did a launch error get swallowed and surface on the next op?
            z = torch.ones(4, device=dev) * 2
            torch.cuda.synchronize()
            res["returned_shape"] = list(y.shape)
            res["next_op_ok"] = True
        elif case == "weight_fp32_input_bf16":
            x = torch.randn(4, 4096, device=dev).to(torch.bfloat16)
            w = (torch.rand(4096, device=dev) + 0.5)  # fp32
            y = rmsnorm(x, w)
            torch.cuda.synchronize()
            res["max_abs_vs_exact"] = (y.float() - exact_ref(x, w.to(torch.bfloat16)).float()).abs().max().item()
            res["finite"] = bool(torch.isfinite(y).all().item())
        elif case == "out_fp32_input_bf16":
            x = torch.randn(4, 4096, device=dev).to(torch.bfloat16)
            w = torch.ones(4096, device=dev, dtype=torch.bfloat16)
            o = torch.zeros(4, 4096, device=dev, dtype=torch.float32)
            y = rmsnorm(x, w, out=o)
            torch.cuda.synchronize()
            res["max_abs_vs_exact"] = (y.float() - exact_ref(x, w).float()).abs().max().item()
            res["out_dtype"] = str(y.dtype)
        elif case == "out_too_small":
            x = torch.randn(64, 4096, device=dev).to(torch.bfloat16)
            w = torch.ones(4096, device=dev, dtype=torch.bfloat16)
            pool = torch.full((2, 4096), 7.0, device=dev, dtype=torch.bfloat16)
            o = pool[0:1]  # 1 row, kernel writes 64
            rmsnorm(x, w, out=o)
            torch.cuda.synchronize()
            res["neighbour_row_clobbered"] = bool((pool[1] != 7.0).any().item())
        elif case == "noncontig_out":
            x = torch.randn(8, 4096, device=dev).to(torch.bfloat16)
            w = torch.ones(4096, device=dev, dtype=torch.bfloat16)
            o = torch.zeros(4096, 8, device=dev, dtype=torch.bfloat16).t()
            y = rmsnorm(x, w, out=o)
            torch.cuda.synchronize()
            res["max_abs_vs_exact"] = (y.float() - exact_ref(x, w).float()).abs().max().item()
        elif case == "misaligned_bf16":
            base = torch.randn(4 * 4096 + 1, device=dev).to(torch.bfloat16)
            x = base[1:].view(4, 4096)  # contiguous, 2-byte offset
            w = torch.ones(4096, device=dev, dtype=torch.bfloat16)
            res["is_contiguous"] = x.is_contiguous()
            y = rmsnorm(x, w)
            torch.cuda.synchronize()
            res["max_abs_vs_exact"] = (y.float() - exact_ref(x, w).float()).abs().max().item()
        elif case == "misaligned_fp32":
            base = torch.randn(4 * 4096 + 1, device=dev)
            x = base[1:].view(4, 4096)  # contiguous, 4-byte offset (float4 needs 16)
            w = torch.ones(4096, device=dev)
            res["is_contiguous"] = x.is_contiguous()
            y = rmsnorm(x, w)
            torch.cuda.synchronize()
            res["max_abs_vs_exact"] = (y.float() - exact_ref(x, w).float()).abs().max().item()
        elif case == "torch_compile":
            x = torch.randn(8, 4096, device=dev).to(torch.bfloat16)
            w = torch.ones(4096, device=dev, dtype=torch.bfloat16)
            import torch._dynamo as dynamo
            expl = dynamo.explain(lambda a, b: rmsnorm(a, b) * 2)(x, w)
            res["graph_breaks"] = expl.graph_break_count
            res["break_reasons"] = [str(r.reason)[:160] for r in expl.break_reasons][:3]
        else:
            res["error"] = "unknown case"
        res["raised"] = None
    except Exception as e:  # noqa: BLE001
        res["raised"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    out(res)

elif mode == "perf":
    torch.manual_seed(0)
    compiled = torch.compile(hf_ref, dynamic=False)

    def bench(fn, iters=200, reps=7):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000 / iters)  # us per call
        ts.sort()
        return ts[len(ts) // 2]

    shapes = [("layer", 1, 4096), ("layer", 8, 4096), ("layer", 64, 4096),
              ("layer", 512, 4096), ("layer", 4096, 4096), ("layer", 16384, 4096),
              ("qk", 32, 128), ("qk", 2048, 128), ("qk", 131072, 128)]
    res = []
    for kind, rows, h in shapes:
        dt = torch.bfloat16
        x = torch.randn(rows, h, device=dev).to(dt)
        w = (torch.rand(h, device=dev) + 0.5).to(dt)
        o = torch.empty_like(x)
        nbytes = 2 * x.numel() * x.element_size() + w.numel() * w.element_size()
        t_k = bench(lambda: rmsnorm(x, w, out=o))
        t_eager = bench(lambda: hf_ref(x, w))
        t_comp = bench(lambda: compiled(x, w))
        t_copy = bench(lambda: o.copy_(x))  # read+write same bytes: empirical ceiling
        res.append({
            "kind": kind, "rows": rows, "h": h,
            "kernel_us": round(t_k, 3), "eager_us": round(t_eager, 3),
            "compile_us": round(t_comp, 3), "copy_us": round(t_copy, 3),
            "kernel_GBps": round(nbytes / t_k / 1e3, 1),
            "copy_GBps": round(nbytes / t_copy / 1e3, 1),
        })
    out({"perf": res})
