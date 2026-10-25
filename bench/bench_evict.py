"""Prefix-cache eviction benchmark: several ~600K-token sessions, then return to the first.

Session A is prefilled, then sessions B, C, D (different code) push it out of the ~2M-token GPU
cache. Returning to A shows whether its prefix comes back from somewhere (CPU offload) or has
to be re-prefilled from scratch.

Usage: python bench_evict.py PORT MODEL [TARGET_TOKENS]
"""
import glob, json, os, random, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 600_000
URL = f"http://localhost:{port}"
SRC = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../.venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))


def corpus(seed):
    files = SRC[:]; random.Random(seed).shuffle(files)
    out, size = [], 0
    for f in files:
        body = open(f, errors="ignore").read(); out.append(f"=== {f.rsplit('vllm/', 1)[-1]} ===\n{body}"); size += len(body)
        if size >= target * 3.9:
            return "\n\n".join(out)


def ask(content):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=3600))
    u = r["usage"]
    return {"s": round(time.time() - t, 2), "prompt": u["prompt_tokens"],
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


docs = {k: corpus(i) for i, k in enumerate("ABCD")}
res = {"A_cold": ask(docs["A"] + "\n\nSummarize this code.")}
for k in "BCD":
    res[f"{k}_cold"] = ask(docs[k] + "\n\nSummarize this code.")
res["A_return"] = ask(docs["A"] + "\n\nSummarize this code.\n\nNow list the main classes.")
print(json.dumps(res))
