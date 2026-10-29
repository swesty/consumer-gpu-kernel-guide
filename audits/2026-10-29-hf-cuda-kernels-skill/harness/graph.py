"""GPU-only time per RMSNorm call via CUDA-graph replay (removes host launch cost).
Usage: python graph.py <variant_dir>"""
import json, sys, torch
sys.path.insert(0, f"{sys.argv[1]}/torch-ext")
from qwen3_kernels._C import rmsnorm_forward
dev = torch.device("cuda"); N = 100
def hf_ref(x, w, eps=1e-6):
    x32 = x.float(); x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps); return w * x32.to(x.dtype)
comp = torch.compile(hf_ref, dynamic=False, mode="max-autotune-no-cudagraphs")
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
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000 / N)
    ts.sort(); return round(ts[4], 3)
res = []
for kind, rows, h in [("layer",1,4096),("layer",8,4096),("layer",64,4096),("layer",512,4096),("layer",4096,4096),("qk",32,128),("qk",2048,128),("qk",131072,128)]:
    torch._dynamo.reset()
    x = torch.randn(rows, h, device=dev).bfloat16(); w = (torch.rand(h, device=dev) + .5).bfloat16(); o = torch.empty_like(x)
    res.append({"kind": kind, "rows": rows, "h": h,
        "kernel_us": gtime(lambda: rmsnorm_forward(o, x, w, 1e-6)),
        "compile_us": gtime(lambda: comp(x, w)),
        "copy_us": gtime(lambda: o.copy_(x))})
print("RESULT " + json.dumps(res), flush=True)
