#!/usr/bin/env bash
# One change at a time vs. the serve.sh baseline.
cd "$(dirname "$0")/.."
bench/run_config.sh kvfp8              -- --kv-cache-dtype fp8
bench/run_config.sh chunk32k           -- --max-num-batched-tokens 32768
bench/run_config.sh chunk64k           -- --max-num-batched-tokens 65536
bench/run_config.sh interactivity      -- --performance-mode interactivity
bench/run_config.sh O3                 -- -O3
bench/run_config.sh spec2 SPEC_TOKENS=2
bench/run_config.sh spec4 SPEC_TOKENS=4
bench/run_config.sh attn-flashinfer    -- --attention-backend FLASHINFER
bench/run_config.sh gdn-triton         -- --additional-config '{"gdn_prefill_backend": "triton"}'
