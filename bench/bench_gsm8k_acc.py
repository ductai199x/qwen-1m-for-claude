"""GSM8K answer accuracy (greedy, thinking on): does a drafter change answer quality?

Speculative decoding is meant to be lossless, so two drafters on the same target should score the same up to
the small numeric drift that different verify widths cause. Takes the first N test questions, extracts the last
number in each reply, and compares it with the reference. Also saves the replies for side-by-side comparison.

Usage: python bench_gsm8k_acc.py PORT MODEL [N] [OUT_JSON]
"""
import json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT, MODEL = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 and int(sys.argv[3]) < 10000 else 100
OUT = sys.argv[4] if len(sys.argv) > 4 else None
qa = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_test_qa.json")))[:N]


def ask(item):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4096,
            "messages": [{"role": "user", "content": item["q"] + "\nSolve step by step. End with 'Answer: <number>'."}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    text = r["choices"][0]["message"]["content"] or ""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    pred = nums[-1] if nums else None
    ok = pred is not None and abs(float(pred) - float(item["a"])) < 1e-6
    return {"ok": ok, "pred": pred, "ref": item["a"], "text": text}


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(ask, qa))
if OUT: json.dump(res, open(OUT, "w"))
print(json.dumps({"n": len(res), "accuracy": round(sum(r["ok"] for r in res) / len(res), 3)}))
