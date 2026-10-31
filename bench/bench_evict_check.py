"""Does an evicted ~600K session restored from the CPU tier give the same answer as when it was cold?

Like bench_evict2.py, but session A has seven passcodes spread through it, and every request asks for them with
greedy decoding: A cold, B/C/D (each also checked against its own passcodes, evicting A), then A again.
Reports whether A's restored answer is identical to its cold answer, passcodes found, and the offload counters.

Usage: python bench_evict_check.py PORT MODEL [TARGET_TOKENS]
"""
import json, random, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_evict2 import SRC, URL, metrics, model, target   # noqa: E402  (bench_evict2 parses argv at import)
import urllib.request


def doc(seed):
    rng = random.Random(seed); files = SRC[:]; rng.shuffle(files)
    parts, size = [], 0
    for f in files:
        parts.append(f"=== {f.rsplit('vllm/', 1)[-1]} ===\n{open(f, errors='ignore').read()}"); size += len(parts[-1])
        if size >= target * 3.9: break
    codes = [f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10**5, 10**6)}" for _ in range(7)]
    for i, c in enumerate(codes):   # spread evenly, including near the start and the end
        parts.insert(int(i * len(parts) / 6.5) + i, f"# PASSCODE {i + 1} of 7: {c}")
    return "\n\n".join(parts) + "\n\nList passcodes 1 to 7 from this text, one per line as 'N: CODE'. Nothing else.", codes


def ask(content):
    before = metrics()
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 120, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=3600)); dt = time.time() - t
    time.sleep(12); after = metrics()
    return {"s": round(dt, 2), "cached": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": r["choices"][0]["message"]["content"], "delta": {k: round(after[k] - before[k]) for k in after if after[k] != before[k]}}


docs = {k: doc(i) for i, k in enumerate("ABCD")}
res = {}
for k in "ABCD":
    res[f"{k}_cold"] = ask(docs[k][0]); res[f"{k}_cold"]["found"] = sum(c in res[f"{k}_cold"]["text"] for c in docs[k][1])
res["A_return"] = ask(docs["A"][0]); res["A_return"]["found"] = sum(c in res["A_return"]["text"] for c in docs["A"][1])
res["identical"] = res["A_return"]["text"] == res["A_cold"]["text"]
print(json.dumps(res))
