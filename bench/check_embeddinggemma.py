"""Check the vLLM embeddinggemma-2 server against sentence-transformers (the model card's reference implementation).

Text: queries and documents with the model's task prefixes, embedded by both; per-item cosine between the two
implementations, and whether the query->document ranking is the same. Matryoshka: vLLM "dimensions": 256 vs the
reference truncated and renormalised. Image: one image embedded via vLLM's chat-style embeddings request vs the
reference, and its similarity to matching / non-matching captions.

Usage: python check_embeddinggemma.py PORT IMAGE_PATH   (needs sentence-transformers + transformers>=5.19, a GPU)
"""
import base64, json, sys, urllib.request
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

PORT, IMG = sys.argv[1], sys.argv[2]
URL = f"http://localhost:{PORT}/v1/embeddings"
queries = ["task: search result | query: how do I reverse a list in python",
           "task: search result | query: capital of france",
           "task: search result | query: symptoms of dehydration"]
docs = ["title: none | text: Use list.reverse() to reverse in place, or reversed(seq) / seq[::-1] for a reversed copy.",
        "title: none | text: Paris is the capital and most populous city of France.",
        "title: none | text: Signs of dehydration include thirst, dark urine, fatigue, dizziness and dry mouth.",
        "title: none | text: The Great Barrier Reef is the world's largest coral reef system."]
captions = ["task: search result | query: two cats lying on a pink couch",
            "task: search result | query: a bowl of ramen on a wooden table"]


def vllm(inp, **kw):
    body = {"model": "embeddinggemma-2", **({"input": inp} if inp is not None else {}), **kw}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})))
    return np.array([d["embedding"] for d in r["data"]])


st = SentenceTransformer("google/embeddinggemma-2", device="cuda")
cos = lambda a, b: (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)

texts = queries + docs + captions
v, r = vllm(texts), st.encode(texts, normalize_embeddings=True)
print(f"text: dim {v.shape[1]}, vLLM vs reference cosine min {cos(v, r).min():.5f} mean {cos(v, r).mean():.5f}")
rank = lambda e: np.argsort(-(e[:3] @ e[3:7].T), axis=1)[:, 0].tolist()
print(f"query->doc top-1: vLLM {rank(v)} reference {rank(r)} (expected [0, 1, 2])")

v256 = vllm(texts, dimensions=256); r256 = r[:, :256] / np.linalg.norm(r[:, :256], axis=1, keepdims=True)
print(f"MRL 256-d: dim {v256.shape[1]}, cosine vs reference min {cos(v256, r256).min():.5f}, top-1 {rank(v256)}")

img64 = base64.b64encode(open(IMG, "rb").read()).decode()
msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}}]}]
vi0 = vllm(None, messages=msg)[0]   # vLLM default for chat requests: no BOS/EOS (assumes the template adds them)
vi = vllm(None, messages=msg, add_special_tokens=True)[0]   # this template adds none, so ask vLLM to
print(f"image without BOS/EOS: cosine vs reference {cos(vi0, st.encode(Image.open(IMG).convert('RGB'), normalize_embeddings=True)):.5f}")
ri = st.encode(Image.open(IMG).convert("RGB"), normalize_embeddings=True)
vc, rc = v[-2:], r[-2:]
print(f"image: vLLM vs reference cosine {cos(vi, ri):.5f}; similarity to [cats caption, ramen caption]: "
      f"vLLM {np.round(vc @ vi / np.linalg.norm(vi), 3).tolist()} reference {np.round(rc @ ri, 3).tolist()}")
