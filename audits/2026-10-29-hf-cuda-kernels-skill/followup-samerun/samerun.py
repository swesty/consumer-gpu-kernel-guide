"""Same-process, interleaved comparison (GB10 drifts 15-25% between runs, so nothing is compared across runs).

Test A: user's kernel at caps 1024/512/256 vs torch.compile vs copy.
Test B: L-lanes-per-row prototype (L=32/16/8) vs torch.compile vs copy.
Each shape: build every candidate's CUDA graph once, then replay them round-robin for REPS rounds;
report the median per candidate. GPU time = graph replay of N calls / N.
"""
import importlib.util, json, os, sys, torch

dev = torch.device("cuda"); N = 100; REPS = 7


def load_so(name, so):
    spec = importlib.util.spec_from_file_location(f"{name}._C", so)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def graph_of(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(N): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); return g


def interleaved(cands):
    graphs = {k: graph_of(f) for k, f in cands.items()}
    ts = {k: [] for k in cands}
    for _ in range(REPS):
        for k, g in graphs.items():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts[k].append(a.elapsed_time(b) * 1000 / N)
    out = {}
    for k, v in ts.items():
        v.sort(); out[k] = {"med": round(v[REPS // 2], 3), "min": round(v[0], 3), "max": round(v[-1], 3)}
    return out


def hf_ref(x, w, eps=1e-6):
    x32 = x.float(); x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(x.dtype)


def exact(x, w, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def data(rows, h):
    torch.manual_seed(0)
    x = (torch.randn(rows, h, device=dev) * 3).bfloat16(); w = (torch.rand(h, device=dev) + .5).bfloat16()
    return x, w


caps = {c: load_so(f"cap{c}", sys.argv[i + 1]) for i, c in enumerate((1024, 512, 256))}
from torch.utils.cpp_extension import load_inline
ext = load_inline("subwarp", cpp_sources="void subwarp(torch::Tensor out, torch::Tensor x, torch::Tensor w, double eps, int64_t lanes, int64_t threads);",
                  cuda_sources=open(os.path.join(os.path.dirname(__file__), "subwarp.cu")).read(),
                  functions=["subwarp"], extra_cuda_cflags=["-O3"], verbose=False)

result = {"A": [], "B": [], "correct": []}

# Test A
for rows, h in [(512, 4096), (4096, 4096), (16384, 4096)]:
    torch._dynamo.reset(); comp = torch.compile(hf_ref, dynamic=False, mode="max-autotune-no-cudagraphs")
    x, w = data(rows, h); outs = {c: torch.empty_like(x) for c in caps}; oc = torch.empty_like(x)
    cands = {f"cap{c}": (lambda c=c: caps[c].rmsnorm_forward(outs[c], x, w, 1e-6)) for c in caps}
    cands["compile"] = lambda: comp(x, w)
    cands["copy"] = lambda: oc.copy_(x)
    result["A"].append({"shape": f"{rows}x{h}", **interleaved(cands)})

# Test B (+ correctness of every lane config)
for rows, h in [(32, 128), (2048, 128), (131072, 128), (4096, 512), (4096, 1024)]:
    torch._dynamo.reset(); comp = torch.compile(hf_ref, dynamic=False, mode="max-autotune-no-cudagraphs")
    x, w = data(rows, h); e = exact(x, w)
    outs = {}
    for L in (32, 16, 8):
        o = torch.empty_like(x); ext.subwarp(o, x, w, 1e-6, L, 256); torch.cuda.synchronize()
        rel = ((o.float() - e.float()).abs() / e.float().abs().clamp_min(1e-3)).max().item()
        result["correct"].append({"shape": f"{rows}x{h}", "L": L, "max_rel_vs_exact": rel})
        outs[L] = o
    oo, oc = torch.empty_like(x), torch.empty_like(x)
    cands = {f"L{L}_t{t}": (lambda L=L, t=t: ext.subwarp(outs[L], x, w, 1e-6, L, t)) for L in (32, 16, 8) for t in (128, 256)}
    cands["orig_cap1024"] = lambda: caps[1024].rmsnorm_forward(oo, x, w, 1e-6)
    cands["compile"] = lambda: comp(x, w)
    cands["copy"] = lambda: oc.copy_(x)
    result["B"].append({"shape": f"{rows}x{h}", **interleaved(cands)})

print("RESULT " + json.dumps(result), flush=True)
