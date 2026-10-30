"""Convert a llama.cpp `qwen35` GGUF (Qwen3.5 / Qwen3.8 dense, e.g. Qwen3.8-27B fine-tunes published only as GGUF)
back to a Hugging Face checkpoint that vLLM can serve.

Weights are dequantized to BF16 and every transform llama.cpp's converter applied (conversion/qwen.py:
Qwen3_5TextModel -> _LinearAttentionVReorderBase -> Qwen3NextModel) is inverted:
  - linear-attention V heads were reordered grouped -> tiled (in_proj_qkv V rows, in_proj_z, in_proj_a/b, A_log,
    dt_bias, conv1d V channels, out_proj columns): reordered back
  - A_log was stored as -exp(A_log): A_log = log(-ssm_a)
  - every *norm.weight except linear_attn.norm got +1: subtract 1
  - conv1d was squeezed to [C, K]: reshaped to [C, 1, K]
  - mtp.* was renamed to the extra block's nextn.* / layer tensors: renamed back
The rest of the checkpoint (config, tokenizer, vision tower) comes from a reference HF checkpoint of the same
architecture; the GGUF's own chat template is kept. `--verify` re-applies llama.cpp's forward transforms to the
output and checks it against the dequantized GGUF tensor by tensor.

Usage: python gguf_to_hf_qwen35.py GGUF REFERENCE_HF_DIR OUT_DIR [--verify]
"""
import json, math, os, shutil, sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

SHARD_BYTES = 5 * 1024**3


def field(reader, key):
    f = reader.fields[key]
    v = f.parts[f.data[0]]
    return bytes(v).decode() if f.types[0].name == "STRING" else v.tolist()[0]


def to_torch(t):
    return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32))


def reorder_v(x, dim, n_k, n_v_per_k, head_dim, inverse):
    """llama.cpp's _reorder_v_heads (grouped [K, Vper, hd] -> tiled [Vper, K, hd]) and its inverse."""
    shape = list(x.shape); dim %= len(shape)
    a, b = (n_v_per_k, n_k) if inverse else (n_k, n_v_per_k)
    x = x.reshape(*shape[:dim], a, b, head_dim, *shape[dim + 1:])
    perm = list(range(x.ndim)); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return x.permute(*perm).contiguous().reshape(*shape)


class Qwen35Map:
    def __init__(self, reader):
        a = lambda k: field(reader, f"qwen35.{k}")
        self.n_layer = a("block_count") - a("nextn_predict_layers")
        self.n_k = a("ssm.group_count")                      # linear_num_key_heads
        self.n_v = a("ssm.time_step_rank")                   # linear_num_value_heads
        self.hd_k = a("ssm.state_size")                      # linear_key_head_dim
        self.hd_v = a("ssm.inner_size") // self.n_v          # linear_value_head_dim
        self.vpk = self.n_v // self.n_k

    def hf_name(self, g):
        """GGUF tensor name -> (HF name, kind) where kind selects the value transform."""
        if g == "token_embd.weight": return "model.language_model.embed_tokens.weight", None
        if g == "output.weight": return "lm_head.weight", None
        if g == "output_norm.weight": return "model.language_model.norm.weight", "norm"
        _, bid, rest = g.split(".", 2)
        bid = int(bid)
        mtp = bid >= self.n_layer
        nextn = {"nextn.eh_proj.weight": ("mtp.fc.weight", None), "nextn.enorm.weight": ("mtp.pre_fc_norm_embedding.weight", "norm"),
                 "nextn.hnorm.weight": ("mtp.pre_fc_norm_hidden.weight", "norm"), "nextn.shared_head_norm.weight": ("mtp.norm.weight", "norm")}
        if rest in nextn: return nextn[rest]
        p = f"mtp.layers.{bid - self.n_layer}." if mtp else f"model.language_model.layers.{bid}."
        table = {
            "attn_norm.weight": ("input_layernorm.weight", "norm"), "post_attention_norm.weight": ("post_attention_layernorm.weight", "norm"),
            "ffn_gate.weight": ("mlp.gate_proj.weight", None), "ffn_up.weight": ("mlp.up_proj.weight", None), "ffn_down.weight": ("mlp.down_proj.weight", None),
            "attn_q.weight": ("self_attn.q_proj.weight", None), "attn_k.weight": ("self_attn.k_proj.weight", None),
            "attn_v.weight": ("self_attn.v_proj.weight", None), "attn_output.weight": ("self_attn.o_proj.weight", None),
            "attn_q_norm.weight": ("self_attn.q_norm.weight", "norm"), "attn_k_norm.weight": ("self_attn.k_norm.weight", "norm"),
            "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "qkv"), "attn_gate.weight": ("linear_attn.in_proj_z.weight", "z"),
            "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "ab"), "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "ab"),
            "ssm_out.weight": ("linear_attn.out_proj.weight", "out"), "ssm_a": ("linear_attn.A_log", "A"),
            "ssm_dt.bias": ("linear_attn.dt_bias", "dt"), "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv"),
            "ssm_norm.weight": ("linear_attn.norm.weight", None),
        }
        hf, kind = table[rest]
        return p + hf, kind

    def to_hf(self, x, kind):
        r = lambda t, dim, hd: reorder_v(t, dim, self.n_k, self.vpk, hd, inverse=True)
        qk = self.hd_k * self.n_k
        if kind == "norm": return x - 1
        if kind == "qkv": return torch.cat([x[:2 * qk], r(x[2 * qk:], 0, self.hd_v)])
        if kind == "z": return r(x, 0, self.hd_v)
        if kind == "ab": return r(x, 0, 1)
        if kind == "out": return r(x, 1, self.hd_v)
        if kind == "A": return r(torch.log(-x).unsqueeze(-1), 0, 1).squeeze(-1)
        if kind == "dt": return r(x.unsqueeze(-1), 0, 1).squeeze(-1)
        if kind == "conv": return torch.cat([x[:2 * qk], r(x[2 * qk:], 0, self.hd_v)]).unsqueeze(1)
        return x

    def to_gguf(self, x, kind):
        """llama.cpp's forward transforms, for --verify."""
        f = lambda t, dim, hd: reorder_v(t, dim, self.n_k, self.vpk, hd, inverse=False)
        qk = self.hd_k * self.n_k
        if kind == "norm": return x + 1
        if kind == "qkv": return torch.cat([x[:2 * qk], f(x[2 * qk:], 0, self.hd_v)])
        if kind == "z": return f(x, 0, self.hd_v)
        if kind == "ab": return f(x, 0, 1)
        if kind == "out": return f(x, 1, self.hd_v)
        if kind == "A": return -torch.exp(f(x.unsqueeze(-1), 0, 1).squeeze(-1))
        if kind == "dt": return f(x.unsqueeze(-1), 0, 1).squeeze(-1)
        if kind == "conv": x = x.squeeze(1); return torch.cat([x[:2 * qk], f(x[2 * qk:], 0, self.hd_v)])
        return x


def main():
    gguf_path, ref_dir, out_dir = sys.argv[1:4]
    verify = "--verify" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    reader = GGUFReader(gguf_path)
    m = Qwen35Map(reader)
    ref_idx = json.load(open(os.path.join(ref_dir, "model.safetensors.index.json")))["weight_map"]

    weight_map, shard, shard_bytes, n_shard, seen = {}, {}, 0, 0, set()
    def flush():
        nonlocal shard, shard_bytes, n_shard
        if not shard: return
        n_shard += 1; fname = f"model-{n_shard:05d}.safetensors"
        save_file(shard, os.path.join(out_dir, fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in shard}); shard, shard_bytes = {}, 0

    worst = 0.0
    for t in reader.tensors:
        hf, kind = m.hf_name(t.name)
        src = to_torch(t)
        out = m.to_hf(src, kind).to(torch.bfloat16)
        if verify:
            back = m.to_gguf(out.float(), kind)
            err = (back - src).abs().max().item() / max(src.abs().max().item(), 1e-12)
            worst = max(worst, err)
            exact = kind not in ("A", "norm") and t.tensor_type.name in ("Q8_0", "BF16")
            if exact and err > 0.004:   # Q8_0 values are 8-bit ints times an fp16 scale: BF16 holds them to ~2^-8
                raise SystemExit(f"VERIFY FAILED {t.name} -> {hf}: rel err {err:.3g}")
        seen.add(hf); shard[hf] = out; shard_bytes += out.numel() * 2
        if shard_bytes >= SHARD_BYTES: flush()

    # vision tower (not in the text GGUF): from the reference checkpoint
    n_vis = 0
    by_file = {}
    for k, f in ref_idx.items():
        if "visual" in k: by_file.setdefault(f, []).append(k)
    for f, ks in by_file.items():
        with safe_open(os.path.join(ref_dir, f), "pt") as h:
            for k in ks:
                shard[k] = h.get_tensor(k).to(torch.bfloat16); shard_bytes += shard[k].numel() * 2; n_vis += 1
                if shard_bytes >= SHARD_BYTES: flush()
    flush()

    missing = sorted(k for k in ref_idx if k not in weight_map)
    extra = sorted(k for k in weight_map if k not in ref_idx)
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)

    # config / tokenizer from the reference; chat template from the GGUF
    for fn in os.listdir(ref_dir):
        if fn.endswith((".json", ".txt", ".jinja", ".model")) and not fn.startswith("model.safetensors"):
            shutil.copy(os.path.join(ref_dir, fn), out_dir)
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    cfg.pop("quantization_config", None); cfg.get("text_config", {}).pop("quantization_config", None)
    cfg["torch_dtype"] = cfg.get("text_config", {})["dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    if "tokenizer.chat_template" in reader.fields:
        open(os.path.join(out_dir, "chat_template.jinja"), "w").write(field(reader, "tokenizer.chat_template"))

    print(json.dumps({"gguf_tensors": len(reader.tensors), "vision_tensors": n_vis, "shards": n_shard,
                      "total_gb": round(total / 1e9, 2), "missing_vs_reference": missing, "extra_vs_reference": extra,
                      "verify_worst_rel_err": worst if verify else None}))


if __name__ == "__main__":
    main()
