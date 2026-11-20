#!/usr/bin/env python3
"""Small decode benchmark against an OpenAI-compatible endpoint (stdlib only).

Measures per-stream decode tok/s = completion_tokens / (t_last - t_first_token),
plus TTFT, for 1 stream (N reps per prompt) and 4 concurrent streams.
Temperature 0, thinking off, fixed prompts, so before/after runs are comparable.
"""
import json, sys, time, threading, statistics, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://10.10.10.46:8888/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3-flash"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "run"
REPS = 3
MAX_TOKENS = 512

PROMPTS = {
    "prose": "Write a detailed, original essay of about 400 words on why lighthouses were built where they were, covering geography, trade routes and engineering.",
    "code": "Write a complete Python module implementing an LRU cache class with get, put, delete, resize and a __repr__, with docstrings and type hints. Output only code.",
}


def one(prompt, out):
    body = {
        "model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; t_last = None; ntok = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                ntok = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
    assert t_first is not None and t_last is not None and ntok, "no tokens streamed"
    out.append({"ttft": t_first - t0, "tok": ntok, "tps": (ntok - 1) / (t_last - t_first)})


def main():
    res = {"label": LABEL, "single": {}, "concurrent4": {}}
    one(PROMPTS["prose"], [])  # warm-up
    for name, p in PROMPTS.items():
        runs = []
        for _ in range(REPS):
            one(p, runs)
        res["single"][name] = {"tps_median": statistics.median(r["tps"] for r in runs),
                               "tps_all": [round(r["tps"], 2) for r in runs],
                               "ttft_median": statistics.median(r["ttft"] for r in runs),
                               "tokens": [r["tok"] for r in runs]}
    for name, p in PROMPTS.items():
        runs = []
        ths = [threading.Thread(target=one, args=(p, runs)) for _ in range(4)]
        t0 = time.perf_counter()
        [t.start() for t in ths]; [t.join() for t in ths]
        wall = time.perf_counter() - t0
        res["concurrent4"][name] = {"per_stream_tps": [round(r["tps"], 2) for r in runs],
                                    "aggregate_tps": sum(r["tok"] for r in runs) / wall}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
