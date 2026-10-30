"""Sanity-check a converted fine-tune against its base model, tensor by tensor (cosine similarity).

A fine-tune stays close to its base, so a correctly converted tensor has cosine ~0.9+ against the base while a
mis-ordered one (wrong head layout) drops far lower. For the reordered linear-attention tensors the script also
scores the *alternative* ordering (the V-head reorder left in place), so it shows which layout actually matches.

Usage: python check_against_base.py CONVERTED_HF_DIR BASE_HF_DIR (block-FP8 base weights are dequantized)
"""
import json, os, sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_to_hf_qwen35 import reorder_v  # noqa: E402

conv_dir, base_dir = sys.argv[1:3]
ci = json.load(open(os.path.join(conv_dir, "model.safetensors.index.json")))["weight_map"]
bi = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]


def load(d, idx, k):
    with safe_open(os.path.join(d, idx[k]), "pt") as h:
        t = h.get_tensor(k).float()
        if t.dtype == torch.float8_e4m3fn or h.get_slice(k).get_dtype() == "F8_E4M3":
            pass
    s = k[:-len("weight")] + "weight_scale_inv"
    if s in idx:   # block-FP8 (128x128) -> float
        with safe_open(os.path.join(d, idx[s]), "pt") as h:
            sc = h.get_tensor(s).float()
        t = t * sc.repeat_interleave(128, 0)[: t.shape[0]].repeat_interleave(128, 1)[:, : t.shape[1]]
    return t


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()
N_K, VPK, HD = 16, 3, 128
alt = {  # what the tensor would be if the V-head reorder had NOT been inverted
    "in_proj_qkv": lambda t: torch.cat([t[:4096], reorder_v(t[4096:], 0, N_K, VPK, HD, inverse=False)]),
    "in_proj_z": lambda t: reorder_v(t, 0, N_K, VPK, HD, inverse=False),
    "out_proj": lambda t: reorder_v(t, 1, N_K, VPK, HD, inverse=False),
    "in_proj_a": lambda t: reorder_v(t, 0, N_K, VPK, 1, inverse=False),
    "conv1d": lambda t: torch.cat([t[:4096], reorder_v(t[4096:], 0, N_K, VPK, HD, inverse=False)]),
}
rows, worst = [], 1.0
keys = [k for k in ci if "visual" not in k and k in bi and any(f"layers.{i}." in k for i in (0, 3, 63)) or k.startswith("mtp.") or "embed" in k or k == "lm_head.weight"]
for k in sorted(set(keys)):
    if k not in bi: continue
    a, b = load(conv_dir, ci, k), load(base_dir, bi, k)
    if a.shape != b.shape:
        rows.append((k, "SHAPE", list(a.shape), list(b.shape))); worst = 0; continue
    c = cos(a, b); worst = min(worst, c)
    name = next((n for n in alt if n in k), None)
    rows.append((k, round(c, 4), round(cos(alt[name](a), b), 4) if name else ""))
for r in rows: print(*r)
print(json.dumps({"tensors_checked": len(rows), "min_cosine": round(worst, 4)}))
