import json, sys
for l in open(sys.argv[1] if len(sys.argv) > 1 else "bench/results.jsonl"):
    r = json.loads(l); x = r.get("result") or {}
    if "error" in r or not x: print(f"{r['name']:18} ERROR {r.get('error')}"); continue
    c, w, s = x["cold"], x["warm"], x["short"]
    print(f"{r['name']:18} cold {c['ttft']:6.1f}s  warm {w['ttft']:5.2f}s ({w['cached']}/{w['prompt']})  "
          f"decode@600K {c['decode_tps']:5.1f}/{w['decode_tps']:5.1f}  short {s['decode_tps']:5.1f}  kv {r['kv_tokens']}  startup {r['startup_s']}s")
