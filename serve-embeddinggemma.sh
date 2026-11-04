#!/usr/bin/env bash
# Serve google/embeddinggemma-2 (multimodal embeddings: text, image, audio, video -> 768-d, MRL 128/256/512) on /v1/embeddings.
# Supported natively since vLLM 0.32.0. Queries/documents need the model's task prefixes, e.g.
# "task: search result | query: ..." and "title: none | text: ..." (vLLM doesn't add them). The config doesn't declare
# MRL, so --hf-overrides does; then "dimensions": 128/256/512 in a request truncates and renormalises.
# Images/audio/video go through chat-style requests ("messages": [...]); send "add_special_tokens": true with them, since
# this template adds no BOS/EOS and vLLM assumes it does (without it, image embeddings drift: cosine 0.84 vs reference).
# Verified against sentence-transformers with bench/check_embeddinggemma.py (text, 256-d MRL, image: cosine >= 0.9999).
# Max length 262144 (the backbone's position limit; trained to 8K). Measured (bench/verify/*embeddinggemma*.out): matching
# by overall topic holds or improves past 8K (tested to 100K), but single details get lost past ~8K -- chunk for those.
# Usage: GPU=1 PORT=8003 ./serve-embeddinggemma.sh [extra vllm args]
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID
exec .venv/bin/vllm serve google/embeddinggemma-2 --served-model-name embeddinggemma-2 --port "${PORT:-8003}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.3}" --max-model-len "${MAX_LEN:-262144}" \
    --hf-overrides '{"is_matryoshka": true, "matryoshka_dimensions": [128, 256, 512, 768]}' "$@"
