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
