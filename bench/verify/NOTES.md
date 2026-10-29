# Verification notes for a HyperQwen field report (2x H200, sm90)

Status per claim: VERIFIED (reproduced with a clean control), PENDING, or DROPPED (could not stand behind it).

## Environment
- 8x NVIDIA H200 (141 GB, sm90, NVLink), driver 580.105.08, Ubuntu 22.04 (kernel 5.15), CUDA 13.0
- HyperQwen commit ce9615c (2026-10-22), its venv: vLLM 0.31.0 + patch series (all 49 applied with --fuzz 0)
- Pristine control: vLLM 0.31.0 unpatched (hyperqwen/venv-pristine031); vLLM 0.32.0 (this repo's uv.lock)

## Claim 1 -- vLLM 0.31 cannot serve Qwen3.8-27B at 1M context (int32 row-offset overflow)
- Observed (HyperQwen venv, Swift 1.5 FP8, TP=2, YaRN x4, max_model_len 1M): CUDA illegal address in warmup.
  With CUDA_LAUNCH_BLOCKING=1 the fault is `StagedWriteTensor.apply_write` (v1/worker/gpu/buffer_utils.py:313,
  via mm/rope.py M-RoPE staged writes). Backporting the int64 cast of row_idx moves the fault to the GDN layer
  (qwen_gdn_linear_attn _forward_core_fused_norm_packed), consistent with upstream's note that a single-site fix
  only moves the fault to the next kernel.
- Upstream: issue vllm#57030 (apply_write int32 row offsets), fix vllm#57524 (staged-write cast), broader
  vllm#60535 (11 MRV2 kernels) and vllm#60736 (int64 index mappings).
- 0.31.0 vs 0.32.0: idx_mapping dtype int32 -> int64; `.to(tl.int64)` casts in v1/worker/gpu 49 -> 63.
- VERIFIED (clean repro, GPU 1, TP=1, GPU_UTIL=0.9, official Qwen/Qwen3.8-27B-FP8, NO speculative decoding, YaRN
  x4 via --hf-overrides, bench/boot_test.sh):
    pristine vLLM 0.31.0 (no HyperQwen patches), max_model_len 262144  -> READY
    pristine vLLM 0.31.0, max_model_len 1000000                       -> CUDA illegal memory access in warmup
    same + CUDA_LAUNCH_BLOCKING=1 -> fault at v1/worker/gpu/mm/rope.py:76 apply_staged_writes ->
      buffer_utils.py:215 StagedWriteTensor.apply_write (the vllm#57030 site)
    vLLM 0.32.0, max_model_len 1000000                                 -> READY
  So: a vLLM 0.31 bug, independent of HyperQwen's patches, Swift, spec decoding and TP. Raw: bench/verify/laneB-boot.txt,
  bench/logs/boot-B1-*.log.

## Claim 2 -- spec-decode-attn / torch top-k sampler fallback slow at long context on H200
- Observed on the 0.32 port, TP=2, 600K context, Swift 1.5 + DFlash2: VLLM_SPEC_DECODE_ATTN=1 +
  VLLM_USE_FLASHINFER_SAMPLER=0 -> 68 tok/s decode (lookup off); both off -> 276 tok/s (lookup on, 64K window).
- On HyperQwen's own stack, setup B, single H200, their run_benchmarks.sh single (~1K-token real prompts), run 2:
  SPEC_ATTN=0 vs default 1: C1 T=default -7.4%, C2 +4.4%, C4 +8.6%, C8 +3.9%; T=0: C1 +2.4%, C2 +3.2%, C4 -1.0%,
  C8 +7.6% -- mixed sign, tracks tok/step -> NO meaningful effect at short context. The claim must be scoped to
  LONG context only (the 4x was at 600K). Raw: bench/verify/laneA-specattn0-run{1,2}.txt.
- VERIFIED (phase 3a; 0.32 port, TP=1, official Qwen/Qwen3.8-27B-FP8 + DFlash2 with the RoPE-fixed drafter, lookup
  OFF in every config so only the two knobs vary; bench_long, 5 warm turns -> median decode tok/s; same-card
  comparisons on GPU 0, repeats on GPU 1):
                         100K ctx            600K ctx            short ctx
    neither              237.8               176.6               204
    SPEC_DECODE_ATTN=1   152.6 (-36%)         56.0 (-68%)         202
    FLASHINFER_SAMPLER=0 240.6 (+1%)         175.9 (-0%)          204
      GPU 1 repeat       207.3               170.0                202
    both                 169.0 / 168.8(G1)   53.7 / 53.8(G1)      210
  -> the long-context slowdown is the split-KV verify attention alone; it grows with context; the torch top-k
  sampler fallback has no measurable cost. Cold prefill and warm-turn latency are unchanged by either knob
  (prefill does not go through the verify path). Raw: bench/verify/phase3-attn.jsonl, bench/logs/P3*-*.log.
  CAVEAT for the report: measured on the 0.32 port of the patch (HyperQwen's 0.31 stack cannot reach 600K); on
  their own stack at <=64K (setup B) the knob made no consistent difference.

## Claim 3 -- stock DFlash2 drafter crashes past 262K context; extending max_position_embeddings fixes it
- Observed (stock vLLM 0.32, Swift FP8, TP=2): prompts <=199K ok, ~265K -> index out of bounds `< 262144` in the
  drafter's RoPE cos/sin cache (inductor kernel). With max_position_embeddings=1048576: ok to 875K.
- VERIFIED (clean repro, GPU 1, TP=1, stock vLLM 0.32.0, target Qwen/Qwen3.8-27B-FP8, max_model_len 400K, no
  other changes; bench/bench_lengths.py, one server per drafter):
    stock incoai/Qwen3.8-27B-DFlash2: 29,923 / 119,617 / 199,097-token prompts OK; 264,550 -> HTTP 500, log:
      `Assertion index out of bounds: 0 <= ... < 262144 failed` (drafter RoPE cos/sin cache, max_position_embeddings)
    same drafter with config max_position_embeddings=1048576 (weights unchanged): 264,550 / 299,947 / 391,843 OK,
      0 faults (600K -> HTTP 400, above this server's 400K limit, expected).
  All 5 drafter layers are sliding_attention (window 2048) with unscaled RoPE, so relative distances stay <=2048.
  Raw: bench/verify/laneB.jsonl, bench/logs/B3-*.log.

## Claim 4 -- lookup drafting gain on H200, lossless
- Observed (TP=2, 1M, Swift): copy-heavy 548 -> 688 tok/s; 600K decode 258 -> 257.
- VERIFIED (speed, phase1 V1 vs V2, 3 repeats x 3 prompts each, same boot): stock DFlash2 550.5 / 554.5 / 553.3 tok/s,
  port+lookup 691.5 / 688.1 / 683.3 tok/s (median of 3 prompts per repeat) -> +24.4% (ratio of means), spread <1% on each side.
- Greedy identity (6 prompts, 1500 tokens, enable_thinking=False): the 3 copy prompts are token-identical across
  no-spec / stock DFlash2 / port+lookup. The 3 free-form prompts differ from no-spec for BOTH stock DFlash2 and
  port+lookup (and from each other) -- stock spec decoding already is not bitwise-identical to no-spec under
  greedy (batched-verify numerics, cf. HyperQwen gotcha 14), so this does not by itself implicate the lookup.
- VERIFIED (attribution): port with lookup OFF (V3) copy-heavy 536 tok/s ~ stock 551-555 -> the +25% is the lookup
  itself; the other ported patches cost ~3% on this test.
- Greedy: V3 (port, lookup off) also differs from no-spec / stock / port+lookup on the 3 free-form prompts while all
  four agree on the 3 copy prompts -> divergence is spec-verify numerics in general, not the lookup.
- VERIFIED (lossless vs stock spec decoding; phase 3b, GPU 1, TP=1, official Qwen/Qwen3.8-27B-FP8, temperature 0,
  enable_thinking=False, 6 prompts x up to 1500 tokens, token ids saved: bench/verify/tokens-G-*.json):
    no-spec run a == run b (all 6); stock DFlash2 run a == run b (all 6) -> greedy is reproducible per config.
    port+lookup == stock DFlash2 == port lookup-off on ALL 6 prompts (token-identical).
    All three spec configs leave the no-spec reference at the same positions on the free-form prompts (tokens 23 /
    54 / 21) and match it on the 3 copy prompts -> that departure is stock vLLM DFlash2 behaviour, not the lookup.
- OBSERVATION (not a claim): in the TP=2 Swift phase-1 runs the three spec configs also differed from each other on
  the free-form prompts. At TP=1 they are identical, so likely TP=2 run-to-run numerics (separate boots); untested.

## Claim 5 -- sm90 data point on HyperQwen's own protocol
- VERIFIED (2026-10-28, GPU 0 alone, HyperQwen ce9615c native venv, vLLM 0.31.0 + series, setup B = SPEC=dflash2
  CTX=fast PREFIX_CACHE=1 via single-user/start_qwen.sh, launcher defaults otherwise -- no power cap set on the H200
  (700 W default); model prepared exactly per docs/install.md (dbirks W4A16-AutoRound + requant + fast variant +
  W4A16 DFlash2). Run 2 kept (run 1 = warm-up, close anyway):
    T=default  C1 258.4 decode / 252.0 e2e tok/s, 3.24 tok/step, TTFT 99 ms | C2 501.3 | C4 856.5 | C8 1369.9
    T=0        C1 263.2 decode / 253.3 e2e tok/s, 3.19 tok/step, TTFT 89 ms | C2 515.5 | C4 954.7 | C8 1307.2
  GSM8K (quality_battery --gsm-only): n=200 acc=0.955, mean 382 tokens (= the README's shipped-arm 0.955).
  Server log ends in EngineDeadError at 22:41:24 -- verified to be our SIGTERM after the run ([shutdown]
  EngineCore: start mode=abort precedes it), not a fault. Raw: bench/verify/laneA-setupB-run{1,2}.txt, -gsm8k.txt.
  -> sm90 C1 single-stream ~2.0x the 3090 reference (127 tok/s).
