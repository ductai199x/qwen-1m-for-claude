#!/usr/bin/env bash
# Serve a Qwen3.8-27B GGUF with llama.cpp at 1M context (YaRN x4 over the native 262K).
#
# Usage: ./serve-llama.sh {huihui|davidau} [extra llama-server args...]
#
# Env overrides:
#   GPUS        CUDA devices (default: 0). One GPU: 1 slot, q8_0 KV cache.
#               Two GPUs (e.g. 0,1): 2 slots x 1M each, f16 KV cache.
#   SPLIT       multi-GPU split mode: tensor (faster decode) or layer (faster long prefill) (default: tensor)
#   PORT        API port (default: 8002)
#   SPEC_TOKENS MTP draft tokens per step; 0 disables speculative decoding (default: 3)
#   QUANT       q8 or q4 (default: q8)
set -euo pipefail
cd "$(dirname "$0")"

QUANT="${QUANT:-q8}"
case "${1:-}" in
    huihui)  [[ $QUANT == q4 ]] && q=Q4_K || q=Q8_0
             MODEL=models/huihui/Huihui-Qwen3.8-27B-abliterated-$q.gguf
             MMPROJ=models/huihui/mmproj-model-bf16.gguf
             ALIAS=huihui-27b-1m ;;
    davidau) [[ $QUANT == q4 ]] && q=Q4_K_M || q=Q8_0
             MODEL=models/davidau/Qwen3.8-27B-Turbo-Brilliance-50X-RI-NEO-MAX-MTP-$q.gguf
             MMPROJ=models/davidau/mmproj-BF16.gguf
             ALIAS=davidau-27b-1m ;;
    *) echo "usage: $0 {huihui|davidau} [llama-server args...]" >&2; exit 1 ;;
esac
shift

export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
IFS=',' read -ra gpus <<<"$CUDA_VISIBLE_DEVICES"
CTX=1000000

# Sized to land around 60-65% VRAM per GPU.
if (( ${#gpus[@]} > 1 )); then
    layout=(-np 2 --kv-unified-per-slot "$CTX" -sm "${SPLIT:-tensor}")
else
    # f16 KV for 1M tokens would put a single GPU at ~72%.
    layout=(-np 1 -c "$CTX" -ctk q8_0 -ctv q8_0)
fi

# Speculative decoding with the model's built-in multi-token-prediction (MTP) head.
SPEC_TOKENS="${SPEC_TOKENS:-3}"
spec=()
if (( SPEC_TOKENS > 0 )); then
    spec=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_TOKENS")
fi

exec llama.cpp/build-nccl/bin/llama-server \
    -m "$MODEL" --mmproj "$MMPROJ" --alias "$ALIAS" \
    --host 0.0.0.0 --port "${PORT:-8002}" \
    -ngl 99 -fa on -ub 2048 -b 2048 \
    --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 \
    "${layout[@]}" \
    --jinja --reasoning-format deepseek \
    --metrics \
    "${spec[@]}" \
    "$@"
