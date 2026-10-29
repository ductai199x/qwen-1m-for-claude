#!/usr/bin/env bash
# Serve Swift 1.5 (ukisai/Swift-1.5-Qwen3.8-27b, a think-less Qwen3.8-27B fine-tune) through serve.sh:
# BF16 weights quantized to FP8 at load time, 1M context, DFlash2 drafter for speculative decoding.
#
# Usage: ./serve-swift.sh [extra vllm serve args...]   (serve.sh env overrides apply, e.g. GPUS, PORT)
#
# LOOKUP (default 1 when .venv-port exists, see setup-port.sh): run on the patched vLLM with DFlash2
# context-lookup drafting -- tokens the model copies from anywhere in the conversation (file edits, quoted
# code) are drafted straight from the context. +24% decode on copy-heavy output, same speed otherwise.
# HyperQwen's split-KV verify attention and torch top-k sampler fallback stay off: on H200 they made
# decode at 600K context ~3x slower (verified: 56 vs 177 tok/s, TP=1). LOOKUP=0 serves on the stock .venv.
#
# The host-RAM prefix-cache tier is off by default here: with the DFlash2 drafter vLLM's offloading
# never restores a prefix (0 cached tokens on return; not root-caused -- vLLM logs that no KV-cache
# group is annotated as a drafter group). For many concurrent long sessions, SPEC_METHOD=mtp KV_OFFLOAD_GIB=200 gets the RAM tier
# back at the cost of ~30% slower decode at long context.
cd "$(dirname "$0")"
export MODEL=ukisai/Swift-1.5-Qwen3.8-27b SERVED_NAME="${SERVED_NAME:-swift-27b-1m}" SPEC_METHOD="${SPEC_METHOD:-dflash}"
if [[ $SPEC_METHOD == dflash ]]; then
    export KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-0}"
    [[ -x .venv-port/bin/vllm ]] && LOOKUP="${LOOKUP:-1}" || LOOKUP=0
    if [[ $LOOKUP == 1 ]]; then
        export VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE="${DRAFT_SAMPLE:-probabilistic}"
    fi
fi
exec ./serve.sh --quantization fp8 "$@"
