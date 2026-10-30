"""Prefix-cache eviction with offload counters: does an evicted ~600K session come back from the CPU tier?

Like bench_evict.py (session A, then B/C/D push it out of the GPU cache, then A returns), but /metrics is scraped
around every request, so the return can be classified from counters rather than only from timing:
  - kv_offload_load_bytes / load_size delta: bytes actually copied CPU->GPU for the request
  - external_prefix_cache_hits delta: tokens the connector (CPU tier) supplied
  - prefix_cache_hits delta: tokens served from the GPU prefix cache
  - usage.prompt_tokens_details.cached_tokens: what the API reports as cached
A cold return with zero load bytes = never restored; load bytes > 0 with few cached tokens = restored but not credited.

Usage: python bench_evict2.py PORT MODEL [TARGET_TOKENS]
"""
import glob, json, os, random, re, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 600_000
URL = f"http://localhost:{port}"
SRC = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../.venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
KEYS = ["vllm:kv_offload_load_bytes", "vllm:kv_offload_load_size_sum", "vllm:kv_offload_load_size_count",
        "vllm:kv_offload_load_time_sum", "vllm:kv_offload_store_bytes", "vllm:kv_offload_store_size_sum",
        "vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_queries",
        "vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:kv_offload_stores_skipped",
        "vllm:kv_offload_allocation_failure"]


def metrics():
    text = urllib.request.urlopen(URL + "/metrics", timeout=60).read().decode()
    out = {}
    for k in KEYS:
        vals = re.findall(rf"^{re.escape(k)}(?:_total)?(?:\{{[^}}]*\}})? ([0-9.e+-]+)$", text, re.M)
        out[k.replace("vllm:", "")] = sum(float(v) for v in vals)
    return out


def corpus(seed):
    files = SRC[:]; random.Random(seed).shuffle(files)
    out, size = [], 0
    for f in files:
        body = open(f, errors="ignore").read(); out.append(f"=== {f.rsplit('vllm/', 1)[-1]} ===\n{body}"); size += len(body)
        if size >= target * 3.9:
            return "\n\n".join(out)


def ask(content):
    before = metrics()
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=3600)); dt = time.time() - t
    time.sleep(12)   # let async offload stores/loads and the 10 s stats logger settle
    after = metrics()
    u = r["usage"]
    return {"s": round(dt, 2), "prompt": u["prompt_tokens"], "api_cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "delta": {k: round(after[k] - before[k], 3) for k in after if after[k] != before[k]}}


docs = {k: corpus(i) for i, k in enumerate("ABCD")}
res = {"A_cold": ask(docs["A"] + "\n\nSummarize this code.")}
for k in "BCD":
    res[f"{k}_cold"] = ask(docs[k] + "\n\nSummarize this code.")
res["A_return"] = ask(docs["A"] + "\n\nSummarize this code.\n\nNow list the main classes.")
print(json.dumps(res))
