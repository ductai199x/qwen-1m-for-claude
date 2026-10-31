<!-- DRAFT reply to @EmilienIT in syv-ai/HyperQwen#405 (re #424) -- for review, not posted. Raw: bench/verify/offload.jsonl
     (P1, P3, P4, P5), bench/logs/P*-*424*.log, scripts bench/verify/offload-pr424.sh, bench/bench_evict2.py,
     bench/bench_evict_check.py. -->

Great find @EmilienIT, the sparse-retention explanation fits our counters exactly. **#424 fixes it on H200 at 600K.**
We ran the same workload as before (four ~600K sessions through a 200 GiB tier, then A again) on main `f382342` +
#424 (`8e4238f`), applying the patch on top of the installed vLLM 0.32.0 package:

| 2x H200, TP=2, 600K | drafter | A returns in | `external_prefix_cache_hits` | `kv_offload_load_bytes` |
|---|---|---|---|---|
| main `f382342` (from before) | DFlash2 | 77.4 s | 0 | 0 |
| **main + #424** | DFlash2 | **5.1 s** | **599,872** / 601,046 | **39.5 GB** |
| main + #424 | MTP | 2.6 s | 600,000 / 601,046 | 41.9 GB |
| our 0.32 port + #424 | DFlash2 | 2.6 s | 599,872 / 601,046 | 39.5 GB |

- **Stores didn't change.** With DFlash2, sessions A/B/C/D stored 39.6/41.3/40.6/40.4 GB, the same as before the patch,
  so retention cost isn't measurably higher at this scale.
- **The hit stops one block short (599,872 = 601,046 minus 1,174)**, as you'd expect from Mamba resuming at the
  earlier replay boundary. That leaves about 1.2K tokens to recompute. MTP gets the full 600,000.
- **MTP isn't affected.** 2.6 s with #424 against 2.9 s without, same tokens and bytes.
- The 5.1 s vs 2.6 s gap between the two DFlash2 rows is a single run each. We don't read anything into it.

**Answers survive the restore.** We also ran a passcode check at this scale (our port + #424, DFlash2, greedy, thinking
off): seven passcodes spread through A's 600K tokens, asked cold, then after B/C/D (each answering its own seven), then
again from the CPU tier. All seven codes came back in every request. A's restored answer was **byte-identical** to its
cold answer: 597,376 tokens restored, 39.4 GB loaded, 3.05 s against 77.0 s cold.

Setup as in our previous comment: `Qwen/Qwen3.8-27B-FP8`, 1M YaRN x4, extended DFlash2 drafter (k=7, lookup on), BF16
KV, FLASH_ATTN, `--kv-offloading-size 200 --kv-offloading-backend native`, default sparse retention, `max_tokens=1` for
the counter runs. Your unit test also fails on main and our port without the patch and passes with it.

Thanks again for chasing this down. It lets us run the CPU tier with DFlash2 for long multi-agent sessions.
