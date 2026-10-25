#!/usr/bin/env bash
# Finalists with the 5-warm-turn benchmark (medians).
cd "$(dirname "$0")/.."
export RESULTS=bench/results2.jsonl
bench/run_config.sh base
bench/run_config.sh spec2 SPEC_TOKENS=2
bench/run_config.sh spec4 SPEC_TOKENS=4
bench/run_config.sh chunk32k           -- --max-num-batched-tokens 32768
bench/run_config.sh chunk32k-spec2 SPEC_TOKENS=2 -- --max-num-batched-tokens 32768
