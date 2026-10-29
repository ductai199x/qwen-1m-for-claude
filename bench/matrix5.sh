#!/usr/bin/env bash
# Ours vs HyperQwen (vLLM 0.31.0 + patch series), same model/drafter/template/context, GPUs 0,1.
cd "$(dirname "$0")/.."
export RESULTS=bench/results8.jsonl BENCH="bench/bench_copy.py bench/bench_long.py bench/bench_e2e.py"
SWIFT=(MODEL=ukisai/Swift-1.5-Qwen3.8-27b SPEC_METHOD=dflash KV_OFFLOAD_GIB=0)
HQ=(VLLM_BIN=/home/tai/1-workdir/hyperqwen/venv/bin/vllm FLASHINFER_DISABLE_VERSION_CHECK=1
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SPEC_DECODE_ATTN=1 VLLM_DFLASH2_LOOKUP=1 DRAFT_SAMPLE=probabilistic)
bench/run_config.sh A-ours                  "${SWIFT[@]}"                                  -- --quantization fp8
bench/run_config.sh B-ours-probabilistic    "${SWIFT[@]}" DRAFT_SAMPLE=probabilistic       -- --quantization fp8
bench/run_config.sh C-hyperqwen-k7          "${SWIFT[@]}" "${HQ[@]}" VLLM_SPEC_DECODE_ATTN_QMAX=8 \
                                                                                           -- --quantization fp8
bench/run_config.sh D-hyperqwen-k15         "${SWIFT[@]}" "${HQ[@]}" SPEC_TOKENS=15 VLLM_SPEC_DECODE_ATTN_QMAX=16 \
                                                                                           -- --quantization fp8 --no-async-scheduling
