# qwen-1m-for-claude

Local vLLM setup for serving Qwen3.8-27B-family models at 1M context on 2 H200s, tuned for long
Claude Code sessions (one or a few streams, ~600K-token conversations).

## Setup

```bash
./upgrade.sh --no-upgrade   # build the venv from uv.lock (flash-attn etc. built from source)
./setup-dflash2.sh          # build the long-context-safe DFlash2 drafter in models/
./setup-port.sh             # build .venv-port: vLLM 0.32 + DFlash2 context-lookup drafting (see below)
```

## Running

```bash
PORT=8001 ./serve-swift.sh   # current setup: Swift 1.5 + DFlash2 + context lookup, served as "swift-27b-1m"
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

### DFlash2 context-lookup drafting (`.venv-port`)

`patches/vllm-0.32.0-dflash2-lookup.patch` ports [HyperQwen](https://github.com/syv-ai/HyperQwen)'s DFlash2
lookup drafting (and its DFlash2 sliding-window KV-group fixes) from vLLM 0.31 to 0.32. While the model copies
text from anywhere in the conversation (a file it is editing, quoted code), the lookup drafts the copy straight
from the context instead of from the drafter's 2048-token window. `serve-swift.sh` uses it when `.venv-port`
exists (`LOOKUP=0` to opt out). Measured on this box (Swift 1.5 + DFlash2, TP=2, 1M context):

| | stock 0.32 | 0.32 + lookup |
|---|---|---|
| Copy-heavy decode (rewrite a file 100K tokens back) | 553 tok/s | **688 tok/s** (+24%) |
| Decode @600K / warm turn @600K | 258 tok/s / 3.57 s | 257 tok/s / 3.31 s |
| KV capacity | 1.54M tokens | **1.68M tokens** |

HyperQwen's split-KV verify attention (`VLLM_SPEC_DECODE_ATTN=1`) is left off: on H200 it slows decode as context
grows (-36% at 100K, -68% at 600K; neutral at short context). Their torch top-k sampler fallback
(`VLLM_USE_FLASHINFER_SAMPLER=0`) costs nothing measurable and stays at vLLM's default. HyperQwen's own stack (vLLM
0.31) cannot run this model at 1M context (int32 index overflows, vllm#57030, fixed in 0.32). Lookup output is
token-identical to stock DFlash2 under greedy decoding. Evidence for all of this: `bench/verify/NOTES.md`.

### DFlash2 long-context fix

`models/Qwen3.8-27B-DFlash2-1M` is `incoai/Qwen3.8-27B-DFlash2` with `max_position_embeddings`
raised from 262144 to 1048576 (weights symlinked). The stock drafter's RoPE table ends at 262K, so
prompts past that crash the server (`index out of bounds ... < 262144`). All drafter layers use
2048-token sliding windows and RoPE depends only on relative distance, so the longer table keeps
the drafter in its trained regime. Verified up to 875K tokens.

**Known limitation:** vLLM 0.32's KV offloading never restores a prefix when DFlash2 is the
drafter (0 cached tokens on return; not root-caused), so
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
