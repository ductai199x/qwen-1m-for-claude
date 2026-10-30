<!-- DRAFT reply to @EmilienIT in syv-ai/HyperQwen#405 -- for review, not posted. Raw: bench/verify/offload.jsonl,
     bench/logs/R{1,2,3,4}-*.log, script bench/bench_evict2.py. -->

Thanks @EmilienIT, and thanks @mhenrichsen for landing all of this so quickly. Your question was the right one, so we
re-ran the H200 case with the offload counters scraped around every request, on your current main as well as ours.

**Short answer: it's a miss, not a slow restore, and it reproduces on main (`f382342`).** With DFlash2 the evicted
session *is* stored to the CPU tier and still fits there, but on return the connector finds 0 tokens and loads 0
bytes. With MTP on the same builds and the same scenario it restores fully in about 3 s.

| run (2x H200, TP=2) | drafter | return to evicted 600K session | `external_prefix_cache_hits` | `kv_offload_load_bytes` |
|---|---|---|---|---|
| HyperQwen main `f382342` | DFlash2 | 77.4 s (full re-prefill) | **0** | **0** |
| HyperQwen main `f382342` | MTP | 2.9 s | 600,000 | 41.9 GB |
| our 0.32 port | DFlash2 | 77.5 s (full re-prefill) | **0** | **0** |
| our 0.32 port | MTP | 3.2 s | 600,000 | 41.9 GB |

**Retention isn't the issue.** In the DFlash2 runs, session A's first request stored 39.6 GB
(`kv_offload_store_bytes` delta), and B, C and D stored 41.3, 40.6 and 40.4 GB. That's 161.9 GB in total, inside the
200 GiB tier (214.7 GB mmap). On A's return, `external_prefix_cache_queries` went up by all 601,046 prompt tokens,
while `external_prefix_cache_hits` and every `kv_offload_load_*` counter stayed at 0. So the lookup ran, found the tier
holding enough data, and still matched nothing.

**Setup, as you asked:**
- `Qwen/Qwen3.8-27B-FP8` target, TP=2, `max_model_len` 1,000,000 (YaRN x4 via `--hf-overrides`), DFlash2 drafter with
  `max_position_embeddings` raised to 1,048,576 (your #409 guard accepts it), 7 drafts, lookup on.
- GPU KV pool: 1,664,703 tokens on main (`--gpu-memory-utilization 0.62`), BF16 KV (`kv_cache_dtype=auto`),
  FLASH_ATTN backend, `--max-num-batched-tokens 32768`, prefix caching on (`mamba_cache_mode` align).
- CPU tier: `--kv-offloading-size 200 --kv-offloading-backend native`.
- Cache groups on main: `kv cache group sizes (832, 832, 832, 832, 832, 832, 832, 832, 832)`, with "Sliding-window
  bucket(s) are the smallest; using group_size 8" and "Add 3 padding layers, may waste at most 60.00% KV cache memory".
  Both "no KV cache group could be identified as the draft model's" (kv_cache_utils) and "no KV-cache group is
  annotated as a drafter group" (offloading scheduler) are logged. Agreed that the second one alone proves nothing;
  the MTP runs log no such group either and restore fine.
- Workload: four distinct ~600K-token prompts (A, B, C, D) to evict A from the GPU pool, then A again with a short
  suffix; `max_tokens=1`; counters read 12 s after each request.

**What differs from your A4000 run** is mostly scale: 600K-token sessions instead of 8K, a ~1.66M-token GPU pool, and
BF16/FLASH_ATTN rather than Triton int8. Our guess (untested) is that matching gets cut short somewhere in how the
drafter's sliding-window group is chunked when only a full-length hit counts. A ~32K-200K sweep would probably show
where it starts. Happy to run that, or a branch, on the H200s if it helps.
