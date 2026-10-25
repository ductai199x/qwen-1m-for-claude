#!/usr/bin/env bash
# Build models/Qwen3.8-27B-DFlash2-1M: incoai/Qwen3.8-27B-DFlash2 with its RoPE table extended past 262K.
#
# The stock drafter sizes its RoPE table by max_position_embeddings (262144), so prompts past 262K
# tokens index out of bounds and crash the server. All drafter layers use 2048-token sliding windows
# and RoPE depends only on relative distance, so a longer table with the same (unscaled) RoPE keeps
# the drafter in its trained regime. Weights are symlinked from the Hugging Face cache.
set -euo pipefail
cd "$(dirname "$0")"

SRC=$(.venv/bin/hf download incoai/Qwen3.8-27B-DFlash2 | tail -1 | sed "s/^path=//")
DST=models/Qwen3.8-27B-DFlash2-1M
mkdir -p "$DST"
ln -sf "$(readlink -f "$SRC/model.safetensors")" "$DST/model.safetensors"
cp "$SRC/README.md" "$DST/"
.venv/bin/python -I - "$SRC/config.json" "$DST/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c["max_position_embeddings"] = 1048576
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
echo "built $DST (max_position_embeddings=1048576)"
