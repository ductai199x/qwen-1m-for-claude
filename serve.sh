#!/usr/bin/env bash
# Serve a Qwen3.8-27B model with 1M context (YaRN x4 over the native 262K) on 2 GPUs.
#
# Usage: ./serve.sh [extra vllm serve args...]
#
# Env overrides:
#   GPUS        CUDA devices to use (default: 6,7)
#   PORT        API port (default: 8000)
#   GPU_UTIL    fraction of each GPU's VRAM vLLM may use (default: 0.62)
#   MAX_LEN     max context length (default: 1000000)
#   SPEC_METHOD mtp (model's built-in head) or dflash (DFlash2 drafter) (default: mtp)
#   DFLASH_MODEL DFlash2 drafter path (default: models/Qwen3.8-27B-DFlash2-1M, incoai/Qwen3.8-27B-DFlash2 with its
#               RoPE table extended past 262K so it doesn't crash at long context)
#   SPEC_TOKENS speculative tokens per step; 0 disables speculative decoding (default: 3 for mtp, 7 for dflash)
#   MODEL       Hugging Face model to serve (default: Qwen/Qwen3.8-27B-FP8)
#   SERVED_NAME model name clients use (default: qwen-27b-1m)
#   KV_OFFLOAD_GIB host-RAM tier for the prefix cache, in GiB; 0 disables (default: 200, ~3.3M tokens).
#               Evicted sessions come back from RAM in seconds instead of a full re-prefill.
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${GPUS:-6,7}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  # allow max-model-len beyond the config's 262144

# YaRN rope scaling from the model card; factor 4.0 * 262144 ≈ 1M.
# Static YaRN applies to all inputs and may slightly hurt short-prompt quality.
ROPE='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'

# Speculative decoding: the model's built-in multi-token-prediction (MTP) head, or the DFlash2
# block-diffusion drafter, which proposes a block of tokens per pass.
SPEC_METHOD="${SPEC_METHOD:-mtp}"
spec=()
case "$SPEC_METHOD" in
    mtp)    SPEC_TOKENS="${SPEC_TOKENS:-3}"
            spec_json="{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC_TOKENS}" ;;
    dflash) SPEC_TOKENS="${SPEC_TOKENS:-7}"
            spec_json="{\"method\": \"dflash\", \"model\": \"${DFLASH_MODEL:-models/Qwen3.8-27B-DFlash2-1M}\", \"num_speculative_tokens\": $SPEC_TOKENS}" ;;
    *) echo "unknown SPEC_METHOD: $SPEC_METHOD" >&2; exit 1 ;;
esac
if (( SPEC_TOKENS > 0 )); then
    spec=(--speculative-config "$spec_json")
fi

KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-200}"
offload=()
if (( KV_OFFLOAD_GIB > 0 )); then
    offload=(--kv-offloading-size "$KV_OFFLOAD_GIB" --kv-offloading-backend native)
fi

exec .venv/bin/vllm serve "${MODEL:-Qwen/Qwen3.8-27B-FP8}" \
    --served-model-name "${SERVED_NAME:-qwen-27b-1m}" \
    --port "${PORT:-8000}" \
    --tensor-parallel-size 2 \
    --max-model-len "${MAX_LEN:-1000000}" \
    --hf-overrides "$ROPE" \
    --gpu-memory-utilization "${GPU_UTIL:-0.62}" \
    --enable-prefix-caching \
    --max-num-batched-tokens 32768 \
    --chat-template chat_template.jinja \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    "${spec[@]}" \
    "${offload[@]}" \
    "$@"
