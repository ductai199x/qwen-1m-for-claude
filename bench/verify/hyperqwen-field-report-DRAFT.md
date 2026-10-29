<!-- DRAFT for review -- not posted. Field report for https://github.com/syv-ai/HyperQwen (sm90 / H200).
     Sections follow .github/ISSUE_TEMPLATE/field-report.yml. Evidence: bench/verify/NOTES.md + raw files listed there. -->

# Field report: sm90 (H200), setup B — plus four things we hit going to 1M context

## Card, and how many
1× NVIDIA H200 141 GB SXM (sm90) for the protocol run. The extra findings below also use 2× H200 (NVLink, TP=2).

## Power limit during the run
700 W (the card's default; no cap set). Not comparable to the 250 W 3090 rows — reported as-is.

## Driver, CUDA, OS, and container or venv
Driver 580.105.08, CUDA 13.0, Ubuntu 22.04.5 (kernel 5.15). Native venv per `docs/install.md`
(vLLM 0.31.0 + the full `patches/series`, all applied with `--fuzz 0`; torch 2.13.0+cu130). Model prepared exactly as
documented: `dbirks/Qwen3.8-27B-W4A16-AutoRound` + `quant_lm_head` / `quant_embed` / `quant_mtp` /
`build_draft_vocab` + `fetch_fast_variant` + `fetch_dflash2`.

## Commit you ran
`ce9615c`

## Which setup
B (`.env.example` defaults: `SPEC=dflash2`, `CTX=fast`, `PREFIX_CACHE=1`), `single-user/start_qwen.sh`, one card.

## Harness output — the second run
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
(The first run, discarded per the README, read within a few percent of these — e.g. C1 T=default 254.5.)

## Quality check
`quality_battery.py h200-setupB --gsm-only`: GSM8K n=200 **acc=0.955**, mean 382 tokens.

## Anything that did not work
Everything in setup B worked. The rest came from running Qwen3.8-27B at **1M context** (YaRN ×4) on 2× H200 for a
long-context coding-agent workload. Each item was re-checked against a clean control; details and raw output on request.

**1. vLLM 0.31.0 cannot serve this model at 1M context — upstream, not your patches.**
Pristine vLLM 0.31.0 (no HyperQwen patches), official `Qwen/Qwen3.8-27B-FP8`, no speculative decoding, TP=1:
`max_model_len=262144` boots; `max_model_len=1000000` dies in warmup with a CUDA illegal address. With
`CUDA_LAUNCH_BLOCKING=1` the fault is `mm/rope.py:76` → `buffer_utils.py:215 StagedWriteTensor.apply_write`, the int32
row-offset overflow in vllm-project/vllm#57030. Backporting only that cast (vllm#57524) moves the fault to the next
kernel (GDN), as vllm#60535 predicts. vLLM 0.32.0 (int64 index mappings, vllm#60736) boots at 1M. We did not bisect
the threshold between 262K and 1M. Your profiles top out at 240K, so this only matters if the pin is used past that.

**2. The DFlash2 drafter's RoPE table ends at 262,144 positions — a latent crash past that.**
`incoai/Qwen3.8-27B-DFlash2` (and your `Qwen3.8-27B-DFlash2-W4A16`, same config) has `max_position_embeddings: 262144`.
On stock vLLM 0.32, `Qwen/Qwen3.8-27B-FP8` target, TP=1: prompts of 199K tokens work; 264,550 tokens kills the
server with `index out of bounds: 0 <= ... < 262144` in the drafter's cos/sin lookup. Setting
`max_position_embeddings` to 1048576 (weights untouched) fixes it — 264,550 / 299,947 / 391,843-token prompts then run with no faults, and
we have since used it to 875K. This is safe because every drafter layer is `sliding_attention` with a 2048 window and
unscaled RoPE, so relative distances never exceed 2048. Not reachable with your shipped profiles (≤240K); worth a guard
or a config tweak in `fetch_dflash2.py` if `MAX_LEN` is ever raised past 262,144.

**3. On sm90, `VLLM_SPEC_DECODE_ATTN=1` costs throughput at long context (fine at short context).**
At short context it is neutral on H200: on your stack, setup B, your harness, `SPEC_ATTN=0` vs the default moved the
eight rows by −7% … +9% with no consistent sign (tracks tok/step noise). At long context it is not: measured on vLLM
0.32 with the `spec-decode-attn` patch ported (your 0.31 stack cannot reach 600K, see 1), TP=1, `Qwen/Qwen3.8-27B-FP8`
+ DFlash2, lookup off so only the two knobs vary, median of 5 decode turns:

| decode tok/s | 100K ctx | 600K ctx |
|---|---|---|
| both knobs off | 237.8 | 176.6 |
| `VLLM_SPEC_DECODE_ATTN=1` | 152.6 (−36%) | 56.0 (−68%) |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | 240.6 | 175.9 |
| both on | 169.0 | 53.7 |

So on sm90 the split-KV Triton verify attention loses to the stock FlashAttention path as context grows; the sampler
fallback is free. Prefill and time-to-first-token are unaffected. A context- or arch-based gate (or default off on sm90)
might be worth considering. Single card, single boot per config, so treat the exact percentages as approximate.

**4. Lookup drafting ports to vLLM 0.32 cleanly and is a real win on H200 — and greedy-identical to stock DFlash2.**
We ported `dflash2-lookup-drafting` + `dflash2-ngram-chains` + the hybrid-KV/offload/GDN fixes around them to 0.32.0
(two hunks rebased, one fuzzy-applied `spec-decode-attn` hunk relocated, `DFlash2Speculator.propose()` moved to
0.32's `dp_sync_state` / `num_speculative_tokens` signature). Swift-1.5 (FP8) + DFlash2, TP=2, 1M context, rewriting a
file ~100K tokens back in the prompt: stock 0.32 550.5 / 554.5 / 553.3 tok/s vs port with lookup 691.5 / 688.1 / 683.3 (three
repeats, +24.4%); port with lookup off 536.0, so the gain is the lookup itself. Decode at 600K unchanged (257 vs 258),
and the hybrid-SW block fix raises the KV pool from 1.54M to 1.68M tokens. Greedy (T=0, thinking off, 6 prompts ×
1500 tokens, TP=1): port+lookup, port lookup-off and stock DFlash2 are token-identical to each other on all six, and
each is reproducible run to run. (All three leave the no-spec reference at the same early positions on free-form
prompts — stock vLLM DFlash2 behaviour, cf. your gotcha 14.) Happy to share the 0.32 port patch if useful.

**Also observed, not root-caused:** on vLLM 0.32, the CPU KV offload tier (`--kv-offloading-size`) never restores a
prefix when DFlash2 is the drafter (0 cached tokens returning to an evicted 600K session; restores fine with MTP),
with or without `offload-dflash-eagle-groups` ported — 0.32 logs "no KV-cache group is annotated as a drafter group".
