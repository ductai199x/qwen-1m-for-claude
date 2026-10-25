# qwen-1m-for-claude

Local vLLM setup for serving Qwen3.8-27B-family models at 1M context on 2 H200s, tuned for long
Claude Code sessions (one or a few streams, ~600K-token conversations).

## Setup

```bash
./upgrade.sh --no-upgrade   # build the venv from uv.lock (flash-attn etc. built from source)
./setup-dflash2.sh          # build the long-context-safe DFlash2 drafter in models/
```

## Running

```bash
PORT=8001 ./serve-swift.sh   # current setup: Swift 1.5 + DFlash2, served as "swift-27b-1m"
PORT=8001 ./serve.sh         # official Qwen/Qwen3.8-27B-FP8 + MTP, served as "qwen-27b-1m"
```

Both default to GPUs 6,7 (`GPUS=...` to change). For a server that survives the shell:

```bash
PORT=8001 setsid nohup ./serve-swift.sh > logs/vllm-serve.log 2>&1 < /dev/null & disown
```

Startup takes ~5-8 minutes (the RAM cache tier is allocated up front). `serve.sh` lists every
env override at the top (model, spec decoding method, VRAM fraction, RAM tier size, ...).

## What the setup does

| Setting | Why |
|---|---|
| YaRN x4 (`--hf-overrides`), `--max-model-len 1000000` | 262K native context -> 1M (from the model card) |
| `--gpu-memory-utilization 0.62` | ~60-65% VRAM budget per GPU |
| `chat_template.jinja` (patched copy of the model's template) | Claude Code compatibility, see below |
| Prefix caching + 200 GiB host-RAM tier (`KV_OFFLOAD_GIB`) | turns stay warm; evicted sessions return in ~3 s instead of ~75 s. **MTP only**: off by default with DFlash2, see below |
| `--max-num-batched-tokens 32768` | ~4% faster cold prefill |
| DFlash2 drafter, 7 tokens (`SPEC_METHOD=dflash`) | ~40% faster decode than MTP at 600K context |
| `--quantization fp8` (Swift only) | Swift ships BF16; quantized at load to fit the VRAM budget |

### Chat template patches

The model's template rejected or mishandled requests from Claude Code (`/v1/messages`):

1. **Reasoning effort**: Claude Code sends `high`; the template only accepts `xhigh|medium|low` and
   returned HTTP 400 (Claude Code then retries with backoff). `high`/`max` now map to `xhigh`,
   `minimal` to `low`.
2. **Unknown content blocks** are skipped instead of failing the request.
3. **Mid-conversation system messages** (Claude Code's per-turn reminders) are rendered in place.
   Without this, vLLM merges them into the leading system prompt, which changes the prompt prefix
   and makes the whole conversation miss the prefix cache (full ~75 s re-prefill at 600K).

### DFlash2 long-context fix

`models/Qwen3.8-27B-DFlash2-1M` is `incoai/Qwen3.8-27B-DFlash2` with `max_position_embeddings`
raised from 262144 to 1048576 (weights symlinked). The stock drafter's RoPE table ends at 262K, so
prompts past that crash the server (`index out of bounds ... < 262144`). All drafter layers use
2048-token sliding windows and RoPE depends only on relative distance, so the longer table keeps
the drafter in its trained regime. Verified up to 875K tokens.

**Known limitation:** vLLM 0.32's KV offloading never restores a prefix when DFlash2 is the
drafter (its sliding-window KV group breaks the hit; 0 cached tokens on return), so
`serve-swift.sh` disables the RAM tier with DFlash2. `SPEC_METHOD=mtp KV_OFFLOAD_GIB=200
./serve-swift.sh` trades ~30% long-context decode speed for the RAM tier, which only matters
with 3+ concurrent long sessions (the GPU cache holds ~1.5M tokens).

## Benchmarks (2x H200, TP=2)

600K-token code context (`bench/bench_long.py`), single stream:

| | Cold prefill | Warm turn (+~6K tokens) | Decode @600K | Decode, short |
|---|---|---|---|---|
| Qwen FP8 + MTP | 74 s | 3.2 s | ~160 tok/s | ~220 tok/s |
| Swift 1.5 + DFlash2 | 78 s | 3.4 s | ~233 tok/s | ~256 tok/s |

End-to-end, 16 answers to reasoning/coding prompts (`bench/bench_e2e.py`): Qwen FP8 + MTP 155 s,
Swift + MTP 82 s, Swift + DFlash2 52-68 s; all 10/10 on the graded subset. Swift thinks ~50% less.

Tried and rejected: fp8 KV cache (~40% slower at long context), FlashInfer attention backend,
`-O3`, `--performance-mode interactivity`, Triton GDN prefill, MTP with 2/4 draft tokens.
llama.cpp (`serve-llama.sh`, GGUF variants) tops out at ~105-125 tok/s single stream.

Raw results are in `bench/results*.jsonl`; `bench/run_config.sh` reruns any config on GPUs 0,1.

## Other files

- `upgrade.sh`: upgrade the venv (handles the flash-attn / causal-conv1d source builds and the
  `oss-harmony` reinstall).
- `serve-llama.sh`, `llama.cpp/`, `models/`: llama.cpp experiment (huihui and DavidAU GGUFs).
