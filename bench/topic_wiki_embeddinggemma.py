"""Topic retrieval at scale: 100 long Wikipedia articles (wikimedia/wikipedia 20231101.en), embeddinggemma-2.

Query = the article's first sentence with its title blanked ("___ was an American outlaw who ..."); document = the rest
of the article (lead paragraph removed). Per length L and trial, one L-token excerpt per article at a random offset, plus
each full article as one input. Each query must rank its own article first among the 100. Whole-document embedding vs
512-token chunks (best chunk).

Usage: python topic_wiki_embeddinggemma.py PORT ARTICLES_JSON [LENGTHS...]
"""
import json, random, re, sys, time, urllib.request
import numpy as np
from transformers import AutoTokenizer

PORT, ARTS = sys.argv[1], sys.argv[2]
LENGTHS = [int(x) for x in sys.argv[3:]] or [512, 2048, 8192, 16384]
N, TRIALS, CHUNK = 100, 3, 512
tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
arts = json.load(open(ARTS))[:N]


def split(a):
    title = re.sub(r"\s*\(.*\)$", "", a["title"])
    lead, _, rest = a["text"].partition("\n")
    first = re.split(r"(?<=[.!?])\s", lead, maxsplit=1)[0]
    q = re.sub(re.escape(title), "___", re.sub(r"\s*\([^)]*\)", "", first, count=1), flags=re.I)
    return q, tok(rest, add_special_tokens=False)["input_ids"]


def embed(texts):
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/embeddings", json.dumps({"model": "embeddinggemma-2", "input": texts}).encode(),
                                 {"Content-Type": "application/json"})
    return np.array([d["embedding"] for d in json.load(urllib.request.urlopen(req, timeout=3600))["data"]])


def scores(sim):
    ranks = [(sim[q] > sim[q, q]).sum() + 1 for q in range(len(sim))]
    return np.mean([r == 1 for r in ranks]), np.mean([1 / r for r in ranks])


def run(excerpts):
    t = time.time(); D = np.vstack([embed(["title: none | text: " + tok.decode(e)]) for e in excerpts]); dt = (time.time() - t) / len(excerpts)
    csim = np.zeros((N, N))
    for j, e in enumerate(excerpts):
        ch = ["title: none | text: " + tok.decode(e[k:k + CHUNK]) for k in range(0, len(e), CHUNK)]
        csim[:, j] = (Q @ np.vstack([embed(ch[k:k + 64]) for k in range(0, len(ch), 64)]).T).max(1)
    return scores(Q @ D.T), scores(csim), dt


pairs = [split(a) for a in arts]; bodies = [b for _, b in pairs]
print("example queries:", [p[0][:110] for p in pairs[:3]])
print(f"article body tokens: min {min(map(len, bodies))}, median {int(np.median(list(map(len, bodies))))}, max {max(map(len, bodies))}", flush=True)
Q = embed([f"task: search result | query: {q}" for q, _ in pairs])
rng = random.Random(0)
print(f"{'length':>12} | {'whole: top1':>11} {'MRR':>5} {'s/doc':>6} | {'chunked: top1':>13} {'MRR':>5}", flush=True)
for L in LENGTHS:
    res = [run([b[(s := rng.randrange(0, len(b) - L)):s + L] for b in bodies]) for _ in range(TRIALS)]
    w = np.mean([r[0] for r in res], 0); c = np.mean([r[1] for r in res], 0); dt = np.mean([r[2] for r in res])
    print(f"{L:>12} | {w[0]:>11.2f} {w[1]:>5.2f} {dt:>6.2f} | {c[0]:>13.2f} {c[1]:>5.2f}", flush=True)
w, c, dt = run(bodies)
print(f"{'full article':>12} | {w[0]:>11.2f} {w[1]:>5.2f} {dt:>6.2f} | {c[0]:>13.2f} {c[1]:>5.2f}", flush=True)
