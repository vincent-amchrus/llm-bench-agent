#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────
#  Configuration - change these values as needed
# ────────────────────────────────────────────────
# CEREBRAS API
BASE_URL=
API_KEY=
MODEL=


REASONING="no-thinking"           # or "thinking", "cot", etc.
CCU=5                             # concurrent users
RAMP_UP_RATE=1                    # users spawned per second

TEST_FILE="data/groundtruth/global/_partial_1k5_vi_global_labeled_2502.json"


TOOLS_FILE="data/tools/vivi_global_tools2.json"
RESULT_DIR="results/_partial_1k_vi_smart_0903_given_tools/cerebras-qwen-3-235b-a22b-instruct-2507_no-thinking_ccu_5_20260312_114533"
echo "Normalized data generated"
python norm_predictions_file.py --input "$RESULT_DIR/raw_predictions.ndjson" --output "${RESULT_DIR}/predictions.ndjson"

echo "Run evaluation script on the generated predictions:"
python eval_tool_calls.py --pred_path "${RESULT_DIR}/predictions.ndjson"

python eval_args.py --pred_path "${RESULT_DIR}/predictions.ndjson" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"

python eval_summary_args.py --pred_path "${RESULT_DIR}/predictions.ndjson" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"
