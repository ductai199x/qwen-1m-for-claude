<!-- DRAFT for review -- not posted. Evidence for every number: bench/verify/NOTES.md and the raw files it lists. -->

# Field report: H200 (sm90), plus feedback from running Qwen3.8-27B on longer contexts

Thanks for HyperQwen. The patch notes and docs made it easy to understand what each piece does, and the
DFlash2 lookup drafting in particular turned out to be a big win for us. Here's our H200 data point (the sm90 one
you asked for), plus a few things we learned by stretching the setup past what you document.

## TL;DR

- **Setup B works on a single H200** and runs about **2× the 3090 numbers**: 258 tok/s single-stream, 1,370 tok/s at
  8 concurrent, GSM8K 0.955 (same as your reference).
- **Lookup drafting works great on H200.** We ported it to vLLM 0.32. For an agent that rewrites files already in its
  context, decode got **24% faster**, with output token-identical to plain DFlash2.
- **On H200, `VLLM_SPEC_DECODE_ATTN=1` hurts at long context.** It's neutral on short prompts, but decode drops 36% at
  100K and 68% at 600K compared with vLLM's stock attention. You might want to gate it by GPU or context length.
- **The DFlash2 drafter stops working past 262,144 tokens.** Your profiles stay under that, so you're fine today. If
  anyone raises `MAX_LEN` past 262K, the server crashes. A one-line config change fixes it.
- **We went to 1M context, which you don't target.** That needed vLLM 0.32, because 0.31 has an upstream overflow bug at
  that length. It's not your patches, and it's included here only for context.

## Why we went past your documented range

We serve Qwen3.8-27B as a local backend for a coding agent, and its sessions run to 600K tokens and more. So we
extended the context to 1M with YaRN ×4 on 2× H200 (TP=2), and moved to vLLM 0.32. Your docs top out at 262K, so
everything beyond that is our own territory. We're sharing it as notes, not as bug reports against your setup.

## 1. Lookup drafting: a real win on H200, and it ports cleanly to vLLM 0.32

Our workload is mostly an agent that reads files and then rewrites them, so the lookup is exactly what it needs.
We moved it to vLLM 0.32 along with the pieces around it (`dflash2-ngram-chains`, the hybrid sliding-window KV fixes,
the offload and GDN fixes). Porting took three small changes:

- one hunk rebased onto 0.32's draft-token slicing
- one `spec-decode-attn` hunk that fuzzy patching had put in `do_kv_cache_update`, moved back to `forward()`
- `DFlash2Speculator.propose()` updated to 0.32's `dp_sync_state` / `num_speculative_tokens` signature

The test was rewriting a file about 100K tokens back in the prompt (Swift-1.5 FP8 + DFlash2, TP=2):

- **Stock vLLM 0.32:** 550.5, 554.5 and 553.3 tok/s over three repeats.
- **With your lookup:** 691.5, 688.1 and 683.3 tok/s, so **+24.4%**.
- **Port with the lookup off:** 536 tok/s, so the gain comes from the lookup itself.
- **Long-context decode** isn't affected: about 257 tok/s at 600K either way.
- **KV capacity:** the hybrid sliding-window fix also grew our KV pool from 1.54M to 1.68M tokens.

We checked it's lossless too. Under greedy decoding (6 prompts, 1,500 tokens each), the lookup, the port with lookup
off, and plain DFlash2 produced token-identical output. Each was also identical run to run. Happy to share the 0.32
port patch if it's useful when you bump the pin.

## 2. On H200, the split-KV verify attention slows down long contexts

At short context it's a wash. On your stack (setup B, your harness), `SPEC_ATTN=0` moved the eight rows by −7% to
+9%, with no consistent direction. As context grows, though, it falls well behind vLLM's stock FlashAttention path on
H200. This is decode speed on one H200, with the lookup off so only these two settings change (median of 5 turns):

| decode tok/s | 100K context | 600K context |
|---|---|---|
| both off (stock) | 237.8 | 176.6 |
| `VLLM_SPEC_DECODE_ATTN=1` only | 152.6 (−36%) | 56.0 (−68%) |
| `VLLM_USE_FLASHINFER_SAMPLER=0` only | 240.6 | 175.9 |
| both on (your default) | 169.0 | 53.7 |

So the slowdown comes entirely from the verify attention. The torch top-k sampler fallback costs nothing, and prefill
and time-to-first-token aren't affected. A gate by GPU architecture or context length could help, or defaulting it
off on sm90. One caveat: we measured this on our 0.32 port, because your 0.31 stack can't reach 600K (see 4), and
each config was a single boot.

## 3. The DFlash2 drafter has a hard limit at 262,144 tokens

Both `incoai/Qwen3.8-27B-DFlash2` and your `Qwen3.8-27B-DFlash2-W4A16` have `max_position_embeddings: 262144`.
The drafter's RoPE table is sized from that number, so any prompt past it indexes off the end, and the server dies
with `index out of bounds: 0 <= ... < 262144`.

We reproduced it on stock vLLM 0.32 with the official FP8 model on one GPU: a 199K-token prompt worked and a
264,550-token one crashed. Raising `max_position_embeddings` to 1048576 in the drafter's config fixes it (weights
unchanged). After that, 264,550, 299,947 and 391,843-token prompts all worked, and we've since run it to 875K.

The change is safe because every drafter layer uses a 2,048-token sliding window with plain RoPE, which only cares
about relative distance, so the drafter never sees anything it wasn't trained on. Your profiles stop at 240K, so
none of this bites today. It might be worth a note, or doing it in `fetch_dflash2.py`, in case someone raises
`MAX_LEN`.

## 4. vLLM 0.31 itself can't reach 1M context (outside your scope, for reference)

To make sure this wasn't your patches, we ran a **pristine vLLM 0.31.0**: official FP8 model, no speculative
decoding, one GPU. It boots at 262K and crashes during warmup at 1M with an illegal memory access. With blocking
launches, the fault is in `StagedWriteTensor.apply_write` (via `mm/rope.py`), the int32 row-offset overflow from
[vllm#57030](https://github.com/vllm-project/vllm/issues/57030). Patching just that spot moves the crash to the next
kernel, as [vllm#60535](https://github.com/vllm-project/vllm/pull/60535) describes. vLLM 0.32 switched to int64
index mappings and boots fine at 1M. We didn't narrow down the exact length where 0.31 starts failing.

## 5. One thing we couldn't get working (not root-caused)

On vLLM 0.32, the CPU KV offload tier (`--kv-offloading-size`) never restores a prefix when DFlash2 is the drafter:
returning to an evicted 600K session got 0 cached tokens. It works fine with MTP. Porting `offload-dflash-eagle-groups`
didn't change that, and vLLM 0.32 logs "no KV-cache group is annotated as a drafter group". We haven't dug further.

---

<details>
<summary>Field-report details for setup B (the form fields)</summary>

- **Card:** 1× NVIDIA H200 141 GB SXM (sm90). Findings 1, 2 and 5 also used a second H200 (NVLink, TP=2).
- **Power limit:** 700 W, the card's default, with no cap set. Not directly comparable to the 250 W 3090 rows.
- **Stack:** driver 580.105.08, CUDA 13.0, Ubuntu 22.04.5 (kernel 5.15). Native venv per `docs/install.md`: vLLM 0.31.0
  plus the full `patches/series` applied with `--fuzz 0`, torch 2.13.0+cu130. Model prepared as documented
  (`dbirks/Qwen3.8-27B-W4A16-AutoRound` + `quant_lm_head` / `quant_embed` / `quant_mtp` / `build_draft_vocab` +
  `fetch_fast_variant` + `fetch_dflash2`).
- **Commit:** `ce9615c`
- **Setup:** B (`.env.example` defaults: `SPEC=dflash2`, `CTX=fast`, `PREFIX_CACHE=1`), `single-user/start_qwen.sh`, one card.
- **Quality:** `quality_battery.py h200-setupB --gsm-only`: GSM8K n=200, acc=0.955, mean 382 tokens.
- **Harness output, second run** (the discarded first run was within a few percent, e.g. C1 T=default 254.5):

```
ROW cohort C1 real prompts T=default | e2e=252.03 tok/s | decode(C/meanTPOT)=258.4 | tok/step=3.24 | meanTTFT=99.07 ms
ROW cohort C2 real prompts T=default | e2e=429.41 tok/s | decode(C/meanTPOT)=501.3 | tok/step=3.19 | meanTTFT=117.19 ms
ROW cohort C4 real prompts T=default | e2e=668.20 tok/s | decode(C/meanTPOT)=856.5 | tok/step=3.01 | meanTTFT=141.10 ms
ROW cohort C8 real prompts T=default | e2e=927.78 tok/s | decode(C/meanTPOT)=1369.9 | tok/step=3.19 | meanTTFT=909.89 ms
ROW cohort C1 real prompts T=0 | e2e=253.27 tok/s | decode(C/meanTPOT)=263.2 | tok/step=3.19 | meanTTFT=88.99 ms
ROW cohort C2 real prompts T=0 | e2e=444.62 tok/s | decode(C/meanTPOT)=515.5 | tok/step=3.21 | meanTTFT=115.63 ms
ROW cohort C4 real prompts T=0 | e2e=721.85 tok/s | decode(C/meanTPOT)=954.7 | tok/step=3.30 | meanTTFT=138.64 ms
ROW cohort C8 real prompts T=0 | e2e=829.51 tok/s | decode(C/meanTPOT)=1307.2 | tok/step=3.25 | meanTTFT=866.88 ms
```
</details>

<details>
<summary><code>SPEC_ATTN=0</code> on your stack (setup B, second run)</summary>

| row | default (`SPEC_ATTN=1`) | `SPEC_ATTN=0` |
|---|---|---|
| C1 T=default | 258.4 | 239.2 |
| C2 T=default | 501.3 | 523.6 |
| C4 T=default | 856.5 | 930.2 |
| C8 T=default | 1369.9 | 1423.5 |
| C1 T=0 | 263.2 | 269.5 |
| C2 T=0 | 515.5 | 531.9 |
| C4 T=0 | 954.7 | 945.6 |
| C8 T=0 | 1307.2 | 1406.0 |
</details>
