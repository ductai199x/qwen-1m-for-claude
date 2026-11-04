"""Companion to needle_embeddinggemma.py: can embeddinggemma-2 match a long excerpt by its overall subject?

20 Gutenberg books with distinct subjects; each query describes a book's subject without its title or character names.
Per length L and trial, one L-token excerpt is cut from each book at a random offset; each query must rank its own
book's excerpt first among the 20. Whole-document embedding vs 512-token chunks (best chunk). Mean over trials.

Usage: python topic_embeddinggemma.py PORT BOOKS_DIR [LENGTHS...]   (needs transformers for the tokenizer)
"""
import json, random, sys, time, urllib.request
import numpy as np
from transformers import AutoTokenizer

PORT, BOOKS = sys.argv[1], sys.argv[2]
LENGTHS = [int(x) for x in sys.argv[3:]] or [512, 2048, 8192, 32768, 100000]
TRIALS, CHUNK = 3, 512
TOPICS = {
    2701: "a sailor's account of life aboard a whaling ship and the obsessive hunt for a whale",
    1342: "a story of courtship, marriage prospects and social class among English country families",
    345: "a horror story told through letters and diaries about a vampire who preys on people in England",
    1661: "detective stories in which a brilliant investigator solves crimes through deduction",
    1228: "a scientific argument that species change over time through natural selection",
    3300: "an economic treatise on the division of labour, markets, prices and the wealth of nations",
    76: "a boy's adventures drifting down a great American river on a raft, told in dialect",
    98: "a novel set in London and Paris during the French Revolution",
    135: "an ex-convict seeking redemption amid poverty and uprisings in nineteenth-century France",
    996: "a comic tale of a deluded would-be knight and his squire wandering through Spain",
    1184: "a wrongfully imprisoned man escapes, becomes rich and takes elaborate revenge on those who betrayed him",
    28054: "a philosophical novel about faith, doubt, guilt and a murder within a Russian family",
    6130: "an epic poem about gods and warriors fighting in the war at Troy",
    3207: "a work of political philosophy on the social contract and the absolute authority of the sovereign",
    203: "a novel depicting the cruelty of slavery in the American South",
    1260: "an orphaned governess's hardships and her love for her brooding employer",
    1400: "an English orphan boy's rise in fortune thanks to an unknown benefactor",
    4280: "a philosophical investigation of the limits of reason and a priori knowledge",
    205: "reflections on simple, self-reliant living in a cabin by a pond in the woods",
    164: "an undersea voyage aboard a submarine commanded by a mysterious captain",
}
tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")


def body(i):
    t = open(f"{BOOKS}/{i}.txt", encoding="utf-8").read()
    a, b = t.find("*** START OF"), t.find("*** END OF")
    return tok(t[t.find("\n", a) if a >= 0 else 0: b if b > 0 else None], add_special_tokens=False)["input_ids"]


def embed(texts):
    body_ = {"model": "embeddinggemma-2", "input": texts}
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/embeddings", json.dumps(body_).encode(), {"Content-Type": "application/json"})
    return np.array([d["embedding"] for d in json.load(urllib.request.urlopen(req, timeout=3600))["data"]])


def scores(sim):
    ranks = [(sim[q] > sim[q, q]).sum() + 1 for q in range(len(sim))]
    return np.mean([r == 1 for r in ranks]), np.mean([1 / r for r in ranks])


ids = list(TOPICS); books = {i: body(i) for i in ids}
print("book lengths (K tokens):", {i: len(b) // 1000 for i, b in books.items()}, flush=True)
Q = embed([f"task: search result | query: {TOPICS[i]}" for i in ids])
rng = random.Random(0)
print(f"{'length':>8} | {'whole: top1':>11} {'MRR':>5} {'s/doc':>6} | {'chunked: top1':>13} {'MRR':>5}", flush=True)
for L in LENGTHS:
    w, c, dts = [], [], []
    for _ in range(TRIALS):
        ex = []
        for i in ids:
            s = rng.randrange(0, len(books[i]) - L); ex.append(books[i][s:s + L])
        t = time.time(); D = np.vstack([embed(["title: none | text: " + tok.decode(e)]) for e in ex]); dts.append((time.time() - t) / len(ids))
        w.append(scores(Q @ D.T))
        csim = np.zeros((len(ids), len(ids)))
        for j, e in enumerate(ex):
            chunks = ["title: none | text: " + tok.decode(e[k:k + CHUNK]) for k in range(0, len(e), CHUNK)]
            E = np.vstack([embed(chunks[k:k + 64]) for k in range(0, len(chunks), 64)])
            csim[:, j] = (Q @ E.T).max(1)
        c.append(scores(csim))
    w, c = np.mean(w, 0), np.mean(c, 0)
    print(f"{L:>8} | {w[0]:>11.2f} {w[1]:>5.2f} {np.mean(dts):>6.2f} | {c[0]:>13.2f} {c[1]:>5.2f}", flush=True)
