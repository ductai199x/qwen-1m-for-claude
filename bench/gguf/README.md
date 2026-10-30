# Serving a GGUF-only Qwen3.8 fine-tune in vLLM (DavidAU Turbo-Brilliance)

vLLM 0.32 has no GGUF loader, so `tools/gguf_to_hf_qwen35.py` converts the GGUF back to a Hugging Face checkpoint:
dequantize to BF16 and invert every transform llama.cpp's converter applied to the `qwen35` architecture (V-head
reorder, `A_log`, norm +1, conv1d, MTP renames). Config, tokenizer and vision tower come from the base release; the
GGUF's own chat template is kept.

```bash
hf download DavidAU/Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF \
    Qwen3.8-27B-Turbo-Brilliance-50X-RI-NEO-MAX-MTP-Q8_0.gguf --local-dir models/davidau
python tools/gguf_to_hf_qwen35.py models/davidau/Qwen3.8-27B-Turbo-Brilliance-50X-RI-NEO-MAX-MTP-Q8_0.gguf \
    <Qwen/Qwen3.8-27B-FP8 snapshot dir> models/davidau-hf --verify     # ~4 min on CPU, 55.6 GB
PORT=8001 MODEL=models/davidau-hf SERVED_NAME=davidau-27b-1m SPEC_METHOD=dflash KV_OFFLOAD_GIB=0 \
    VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic ./serve.sh --quantization fp8
```

It is served with this repo's patched `chat_template.jinja` (Claude Code compatibility): DavidAU's own template keeps
the "System message must be at the beginning" guard that breaks the prefix cache under Claude Code.

## Verification (2026-10-29)

- **Round trip:** re-applying llama.cpp's forward transforms to the output reproduces the dequantized GGUF to BF16
  rounding (worst relative error 0.27%); all 866 GGUF tensors mapped, nothing missing or extra.
- **Layout vs base Qwen3.8** (`tools/check_against_base.py`, 56 tensors): every tensor cosine >= 0.9996; the
  alternative V-head layout scores 0.04-0.73 on the reordered tensors, so the inverse reorder is the right one.
  (This cannot measure how far the fine-tune moved from the base: the base checkpoint's FP8 noise, ~2.7%, is larger
  than fine-tune deltas -- Swift 1.5 also scores 0.9996.)
- **Behavioural equivalence** (`bench/gguf/compare.py`, 8 prompts, 1,981 greedy positions, one H200 each, BF16, no
  spec decode): llama.cpp generates from the original GGUF, vLLM scores the same token ids with prompt_logprobs.
  Converted DavidAU agrees with llama.cpp's token at 99.85% of positions (3 disagreements; 6 of 8 prompts identical
  throughout). Controls on the same sequences: Swift 1.5 98.84% (23), base Qwen3.8 FP8 98.28% (34).
  Tokenizers produce identical ids.
- **Serving config** (TP=2, FP8 at load, DFlash2 + lookup, 1M YaRN, port venv): tool calls, image input (base vision
  tower), prefix cache 98% with Claude Code-style inline reminders, `/v1/messages` effort=high all pass;
  e2e 10/10 graded at 355.6 tok/s; copy-heavy 685 tok/s; 600K cold 78.5 s / warm 3.2 s / decode 212.5 tok/s;
  needle found at 960,630 tokens. Raw: `bench/gguf/*.json`, `bench/gguf/davidau-serving.txt`.

## DavidAU's reasoning modes (`{REASON:...}`)

Most of what makes DavidAU's model answer "like DavidAU" lives in its chat template, not its weights: typing
`{REASON:deeptree}` (or ~35 other modes: `socrates`, `einstein`, `coder`, `brainstorm`, `teacher`, ...; `i` prefix for
instruct variants like `{REASON:isocrates}`; `{REASON:help}` for the menu) swaps in a large mode-specific system prompt.
With no tag the template renders exactly like the official one. Weights alone were verified equivalent above.

To get the modes behind Claude Code, serve with DavidAU's template patched by `tools/patch_davidau_template.py` (same
three fixes as `chat_template.jinja`; DavidAU's template is copyright DavidAU, so the patched copy stays in
`models/davidau-hf/chat_template.claude.jinja` and is not committed):

```bash
python tools/patch_davidau_template.py models/davidau-hf/chat_template.jinja models/davidau-hf/chat_template.claude.jinja
PORT=8001 MODEL=models/davidau-hf SERVED_NAME=davidau-27b-1m SPEC_METHOD=dflash KV_OFFLOAD_GIB=0 \
    VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic \
    ./serve.sh --quantization fp8 --chat-template models/davidau-hf/chat_template.claude.jinja
```

Template checks (CPU): vLLM no longer merges inline system messages; 48 normal renders (modes by tag and by effort,
instruct modes, help, thinking off, tools, system prompts) are byte-identical to the original template; effort
`high`/`max`/`minimal`/None/unknown -- each of which made the original raise -- map to the intended mode; a tag still
overrides the request's effort; unknown content blocks are skipped. Note the original raised on a *missing* effort, so
it could not serve plain requests on vLLM at all.

Instruct modes needed a vLLM fix: they turn thinking off *inside* the template, which vLLM's qwen3 reasoning parser
could not see (it decides from the request's `enable_thinking`), so the whole answer was filed as thinking and the
visible answer was empty. The port patch now makes the parser honour a think block the prompt already closed, on both
the streaming and non-streaming paths.

Live (2026-10-30, TP=2, `/v1/messages` effort=high): `{REASON:deeptree}` produces DavidAU's full "DEEP TREE-OF-THOUGHT
HYPER-MATRIX v4.0" answer (Path Alpha/Beta/Gamma, phases, scoring, pruning) while the same question without the tag
gets a normal concise answer; `{REASON:help}` shows the mode menu; instruct modes (`isocrates`, `icoder`) answer with
no thinking and a visible answer on Anthropic streaming, Anthropic non-streaming and OpenAI non-streaming; reasoning
modes and plain prompts split thinking/answer as before. Regression: tool calls, images, prefix cache 98% with inline
reminders, e2e 10/10, copy-heavy 693 tok/s. One caveat: switching mode mid-conversation changes the system prompt, so
that turn misses the prefix cache (inherent to how the template works).
