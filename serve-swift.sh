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
# The host-RAM prefix-cache tier (KV_OFFLOAD_GIB, default 200) is on only with LOOKUP=1: .venv-port carries
# HyperQwen #424, without which vLLM's offloading never restores a prefix when DFlash2 is on (0 cached tokens on
# return). Verified at 600K on H200: an evicted session restores in 2.6 s vs 77 s, answers byte-identical.
# On the stock .venv (LOOKUP=0) the tier stays off.
cd "$(dirname "$0")"
export MODEL=ukisai/Swift-1.5-Qwen3.8-27b SERVED_NAME="${SERVED_NAME:-swift-27b-1m}" SPEC_METHOD="${SPEC_METHOD:-dflash}"
# 0.78 of 140 GB = about 120 GB actually used per GPU (real usage settles ~10 GB above the fraction)
export GPU_UTIL="${GPU_UTIL:-0.78}"
if [[ $SPEC_METHOD == dflash ]]; then
    [[ -x .venv-port/bin/vllm ]] && LOOKUP="${LOOKUP:-1}" || LOOKUP=0
    if [[ $LOOKUP == 1 ]]; then
        export VLLM_BIN=.venv-port/bin/vllm VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE="${DRAFT_SAMPLE:-probabilistic}"
    else
        export KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-0}"
    fi
fi
exec ./serve.sh --quantization fp8 "$@"
