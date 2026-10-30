"""Is a GGUF->HF conversion faithful? Teacher-forced agreement between llama.cpp (original GGUF) and vLLM (converted).

For each prompt: llama.cpp generates N greedy tokens; then the same token ids (prompt + llama.cpp's continuation)
are scored by a vLLM server with prompt_logprobs, and at every continuation position we check whether vLLM's own
top-1 token equals llama.cpp's token, and how much probability vLLM gives it. A faithful conversion agrees almost
everywhere (differences only at near-ties, from different kernels and quantization); a different model does not.

Usage: python compare.py LLAMA_PORT VLLM_PORT VLLM_MODEL_NAME TOKENIZER_DIR OUT_JSON [N_TOKENS]
"""
import json, math, sys, urllib.request

from transformers import AutoTokenizer

llama_port, vllm_port, vllm_model, tok_dir, out_path = sys.argv[1:6]
n_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 300
tok = AutoTokenizer.from_pretrained(tok_dir)


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku about debugging at 3am, then explain its imagery.",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "Translate to French: 'The meeting has been moved to Thursday afternoon because the room is booked.'",
    "Refactor this function to be more readable:\ndef f(l):\n  r=[]\n  for i in l:\n    if i%2==0: r.append(i*i)\n  return r",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes?",
    "Give three arguments for and three against remote work.",
]

rows = []
for p in PROMPTS:
    text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    prompt_ids = tok.encode(text, add_special_tokens=False)
    g = post(llama_port, "/completion", {"prompt": prompt_ids, "n_predict": n_tokens, "temperature": 0, "top_k": 1,
                                          "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"]})
    cont = g["tokens"]
    # vLLM: score prompt + llama.cpp's continuation; prompt_logprobs gives, per position, the logprob of the
    # actual token plus vLLM's top-k (k=1 -> its argmax)
    r = post(vllm_port, "/v1/completions", {"model": vllm_model, "prompt": prompt_ids + cont, "max_tokens": 1,
                                             "temperature": 0, "prompt_logprobs": 1})
    plp = r["choices"][0]["prompt_logprobs"][len(prompt_ids):]
    agree, lp_actual, first_dis = 0, [], None
    for i, (t, d) in enumerate(zip(cont, plp)):
        d = {int(k): v for k, v in d.items()}
        best = max(d, key=lambda k: d[k]["logprob"])
        lp_actual.append(d[t]["logprob"] if t in d else float("nan"))
        if best == t: agree += 1
        elif first_dis is None: first_dis = i
    n = len(cont)
    rows.append({"prompt": p[:40], "n": n, "top1_agree": round(agree / n, 4), "first_disagree": first_dis,
                 "mean_p_of_llama_token": round(sum(math.exp(x) for x in lp_actual if x == x) / n, 4),
                 "llama_text": tok.decode(cont)[:120]})
    print(json.dumps(rows[-1]), flush=True)

summary = {"model": vllm_model, "positions": sum(r["n"] for r in rows),
           "top1_agree": round(sum(r["top1_agree"] * r["n"] for r in rows) / sum(r["n"] for r in rows), 4),
           "mean_p_of_llama_token": round(sum(r["mean_p_of_llama_token"] * r["n"] for r in rows) / sum(r["n"] for r in rows), 4)}
json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1)
print("SUMMARY", json.dumps(summary))
