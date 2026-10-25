import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l); x = r.get("result") or {}
    if "error" in r or not x: print(f"{r['name']:18} ERROR {r.get('error')}"); continue
    c, w, s = x["cold"], x["warm"], x["short"]
    print(f"{r['name']:18} cold {c['ttft']:6.1f}s  warm ttft(med) {w['ttft_med']:5.2f}s  cached {w['cached_frac']:.1%}  "
          f"decode@600K(med of 5) {w['decode_med']:6.1f}  short(med of 3) {s['decode_med']:6.1f}  kv {r['kv_tokens']}")
