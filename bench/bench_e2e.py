"""End-to-end time-to-answer benchmark (single stream, real EOS, model's recommended sampling).

Measures what a user waits for: wall time and tokens per finished answer, plus correctness on
problems with known answers, so a model that thinks less can be compared fairly.

Usage: python bench_e2e.py PORT MODEL [SAMPLES]
"""
import json, re, statistics, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 2
URL = f"http://localhost:{port}"

PROBLEMS = [  # (prompt, expected answer or None for open-ended)
    ("How many positive integers n <= 1000 make n^2 + 1 divisible by 5? End with 'ANSWER: <number>'.", "400"),
    ("What is the sum of all primes less than 100? End with 'ANSWER: <number>'.", "1060"),
    ("How many trailing zeros does 1000! have? End with 'ANSWER: <number>'.", "249"),
    ("In how many ways can a 2x10 board be tiled with 1x2 dominoes? End with 'ANSWER: <number>'.", "89"),
    ("A bag has 5 red and 3 blue balls. Two are drawn without replacement. What is the probability both are red, "
     "as a reduced fraction a/b? End with 'ANSWER: a/b'.", "5/14"),
    ("Write a Python function `merge_intervals(intervals)` that merges overlapping intervals, with a few tests.", None),
    ("This Python code is meant to return the k most frequent words but is wrong: "
     "`def topk(ws,k): c={}; [c.__setitem__(w,c.get(w,0)+1) for w in ws]; return sorted(c)[:k]`. Explain the bug and fix it.", None),
    ("Design a rate limiter for an HTTP API serving 10k req/s across 20 servers. Compare token bucket vs sliding window "
     "and recommend one.", None),
]


def ask(prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 30000,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=3600)); dt = time.time() - t
    u = r["usage"]; content = r["choices"][0]["message"]["content"] or ""
    return dt, u["completion_tokens"], (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), content


rows = []
for prompt, expected in PROBLEMS:
    for _ in range(samples):
        dt, n, nr, content = ask(prompt)
        ok = None
        if expected:
            m = re.findall(r"ANSWER:\s*\**\s*([0-9/]+)", content)
            ok = bool(m) and m[-1] == expected
        rows.append({"dt": dt, "tokens": n, "reasoning": nr, "ok": ok})

graded = [r["ok"] for r in rows if r["ok"] is not None]
print(json.dumps({
    "answers": len(rows),
    "total_s": round(sum(r["dt"] for r in rows), 1),
    "median_s": round(statistics.median(r["dt"] for r in rows), 1),
    "total_tokens": sum(r["tokens"] for r in rows),
    "reasoning_tokens": sum(r["reasoning"] or 0 for r in rows),
    "tok_per_s": round(sum(r["tokens"] for r in rows) / sum(r["dt"] for r in rows), 1),
    "correct": f"{sum(graded)}/{len(graded)}",
}))
