"""Exp 1: thread-cap sweep of the user's kernel. Exp 2: warp-per-row prototype vs inductor.

Usage: python exp.py cap <variant_dir> <label>
       python exp.py warp
GPU time = CUDA-graph replay of 100 calls, median of 9 replays.
"""
import json, os, sys, torch

dev = torch.device("cuda"); N = 100


def gtime(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(N): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(9):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / N)
    ts.sort(); return round(ts[4], 3)


def hf_ref(x, w, eps=1e-6):
    x32 = x.float(); x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(x.dtype)


def exact(x, w, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def data(rows, h):
    torch.manual_seed(0)
    x = (torch.randn(rows, h, device=dev) * 3).bfloat16(); w = (torch.rand(h, device=dev) + .5).bfloat16()
    return x, w, torch.empty_like(x)


mode = sys.argv[1]
if mode == "cap":
    sys.path.insert(0, f"{sys.argv[2]}/torch-ext")
    from qwen3_kernels._C import rmsnorm_forward
    res = {"label": sys.argv[3], "t": {}}
    for rows, h in [(64, 4096), (512, 4096), (4096, 4096), (16384, 4096)]:
        x, w, o = data(rows, h)
        res["t"][f"{rows}x{h}"] = gtime(lambda: rmsnorm_forward(o, x, w, 1e-6))
    print("RESULT " + json.dumps(res), flush=True)

elif mode == "warp":
    from torch.utils.cpp_extension import load_inline
    src = open(os.path.join(os.path.dirname(__file__), "warprow.cu")).read()
    ext = load_inline("warprow", cpp_sources="void warp_row(torch::Tensor out, torch::Tensor x, torch::Tensor w, double eps, int64_t warps_per_block);",
                      cuda_sources=src, functions=["warp_row"], extra_cuda_cflags=["-O3", "-Xptxas=-v"], verbose=False)
    sys.path.insert(0, "/k/src/rtx-6000-pro/qwen3_rmsnorm/torch-ext")
    from qwen3_kernels._C import rmsnorm_forward
    comp = torch.compile(hf_ref, dynamic=False, mode="max-autotune-no-cudagraphs")
    out = []
    for rows, h in [(32, 128), (2048, 128), (131072, 128), (4096, 1024), (65536, 1024), (4096, 4096 // 8)]:
        torch._dynamo.reset()
        x, w, o = data(rows, h)
        r = {"shape": f"{rows}x{h}"}
        # correctness of the prototype
        ext.warp_row(o, x, w, 1e-6, 8); torch.cuda.synchronize()
        e = exact(x, w)
        r["proto_max_abs_vs_exact"] = (o.float() - e.float()).abs().max().item()
        r["proto_max_rel_vs_exact"] = ((o.float() - e.float()).abs() / e.float().abs().clamp_min(1e-3)).max().item()
        r["orig_us"] = gtime(lambda: rmsnorm_forward(o, x, w, 1e-6))
        for wpb in (4, 8, 16):
            r[f"proto_wpb{wpb}_us"] = gtime(lambda: ext.warp_row(o, x, w, 1e-6, wpb))
        r["compile_us"] = gtime(lambda: comp(x, w))
        r["copy_us"] = gtime(lambda: o.copy_(x))
        out.append(r)
    print("RESULT " + json.dumps(out), flush=True)
