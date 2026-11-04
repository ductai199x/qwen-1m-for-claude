"""Does embeddinggemma-2 still find one fact in a very long input? Needle retrieval at increasing input lengths.

For each length L, N documents of L tokens are cut from a long text (War and Peace) at different offsets, and each gets
its own invented fact (the needle) inserted at a random depth. Each fact's question must rank its own document first
among the N. Two ways of embedding the documents:
  whole    one embedding of the entire document (what pushing max_model_len to 256K would mean)
  chunked  CHUNK-token chunks (default 512), document score = best chunk (the usual way to index long content)
A first row embeds each needle sentence alone (sanity check: the questions should find their own facts).
Reports top-1 accuracy and MRR per length, the similarity between each question and its document, and embedding time.

Usage: python needle_embeddinggemma.py PORT TEXT_FILE [LENGTHS...]   (needs transformers for the tokenizer)
"""
import json, os, random, sys, time, urllib.request
import numpy as np
from transformers import AutoTokenizer

PORT, TEXT = sys.argv[1], sys.argv[2]
LENGTHS = [int(x) for x in sys.argv[3:]] or [256, 512, 1024, 2048, 8192, 32768, 131072, 250000]
N, CHUNK = 20, int(os.environ.get("CHUNK", 512))
tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
book = tok(open(TEXT, encoding="utf-8").read()[60_000:], add_special_tokens=False)["input_ids"]   # skip Gutenberg header
people = ["Ilse Brandt", "Tomasz Wierzba", "Ama Owusu", "Kenji Morioka", "Lucia Ferrante", "Ragnar Holm", "Priya Natarajan",
          "Mateo Quispe", "Aino Lehtonen", "Dmitri Vasko", "Chiara Bellini", "Oluwaseun Adeyemi", "Hana Kobayashi",
          "Felix Aumann", "Saoirse Kehoe", "Rafael Duarte", "Yusuf Demir", "Elin Nyström", "Marek Novotny", "Leila Haddad"]
things = ["smoked paprika from Szeged", "a teaspoon of black cardamom", "candied lemon peel", "roasted fennel seeds",
          "a splash of plum brandy", "dried porcini dust", "toasted sesame oil", "pickled green walnuts", "sumac and orange zest",
          "a pinch of saffron", "fermented black garlic", "juniper berries", "grated horseradish", "tamarind paste",
          "pine pollen", "bergamot syrup", "smoked sea salt", "crushed pink peppercorns", "sea buckthorn jam", "sorrel leaves"]
needle = lambda i: f" The secret ingredient in {people[i]}'s famous winter soup is {things[i]}. "
question = lambda i: f"task: question answering | query: What is the secret ingredient in {people[i]}'s winter soup?"


def embed(texts):
    body = {"model": "embeddinggemma-2", "input": texts}
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/embeddings", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return np.array([d["embedding"] for d in r["data"]]), r["usage"]["prompt_tokens"]


def scores(sim):   # sim[q, d]; correct document for query q is d = q
    ranks = [(sim[q] > sim[q, q]).sum() + 1 for q in range(len(sim))]
    return np.mean([r == 1 for r in ranks]), np.mean([1 / r for r in ranks])


Q, _ = embed([question(i) for i in range(N)])
rng = random.Random(0)
S, _ = embed(["title: none | text: " + needle(i).strip() for i in range(N)]); top1, mrr = scores(Q @ S.T)
print(f"needle sentence alone: top1 {top1:.2f} MRR {mrr:.2f} sim(q,own) {np.mean(np.diag(Q @ S.T)):.3f}")
print(f"{'length':>8} | {'whole: top1':>11} {'MRR':>5} {'sim(q,own)':>10} {'s/doc':>6} | {'chunked: top1':>13} {'MRR':>5}", flush=True)
for L in LENGTHS:
    docs, depths = [], []
    for i in range(N):
        start = rng.randrange(0, len(book) - L); body = book[start:start + L]
        nt = tok(needle(i), add_special_tokens=False)["input_ids"]; d = rng.random(); depths.append(d)
        cut = int(d * (L - len(nt)))
        docs.append(body[:cut] + nt + body[cut:L - len(nt)])
    texts = ["title: none | text: " + tok.decode(d) for d in docs]
    t = time.time(); D = np.vstack([embed([x])[0] for x in texts]); dt = (time.time() - t) / N
    sim = Q @ D.T; top1, mrr = scores(sim)
    # chunked: same documents, 4K-token chunks, best chunk per document
    csim = np.full((N, N), -1.0)
    for j, d in enumerate(docs):
        chunks = ["title: none | text: " + tok.decode(d[k:k + CHUNK]) for k in range(0, len(d), CHUNK)]
        E = np.vstack([embed(chunks[k:k + 32])[0] for k in range(0, len(chunks), 32)])
        csim[:, j] = (Q @ E.T).max(1)
    ctop1, cmrr = scores(csim)
    print(f"{L:>8} | {top1:>11.2f} {mrr:>5.2f} {np.mean(np.diag(sim)):>10.3f} {dt:>6.2f} | {ctop1:>13.2f} {cmrr:>5.2f}", flush=True)
