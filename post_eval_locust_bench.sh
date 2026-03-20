#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────
#  Configuration - change these values as needed
# ────────────────────────────────────────────────

MODEL="Qwen/Qwen3.5-4B"
BASE_URL="http://localhost:8268"
BASE_URL="http://localhost:8269"
API_KEY="EMPTY"          # Set your API key if needed

MODEL="qwen3-4b-it-1102"
MODEL="_1903_only_response_r32_alpha64_batch_2x8_lr1e-5_61k_multiturn_unsloth-Qwen3-4B-Instruct-2507"
MESSAGE_COL="messages_with_ground_truth_tools_history"
#MESSAGE_COL="user_message"
MESSAGE_COL="messages"


REASONING="no-thinking"           # or "thinking", "cot", etc.
CCU=10                             # concurrent users
RAMP_UP_RATE=1                    # users spawned per second

TEST_FILE="data/groundtruth/vivi_smart/_partial_1k_vi_smart_0903_given_tools.json"
TOOLS_FILE="data/tools/vivi_smart_tools_0903.json"

TEST_FILE="data/groundtruth/vivi_smart/_partial_1k_vi_smart_labeled_0302.json"
TOOLS_FILE="data/tools/vivi_smart_tools.json"



TEST_FILE="data/groundtruth/vivi_smart/multiturn/_partial_1028_vi_smart_0903_given_tools.json"
#TEST_FILE="data/groundtruth/vivi_smart/multiturn/_partial_20_vi_smart_0903_given_tools.json"
TOOLS_FILE="data/tools/vivi_smart_tools_0903.json"

# ────────────────────────────────────────────────
#  Derived values (usually no need to change)
# ────────────────────────────────────────────────


NOTE="_RTX_3090_vllm_0.17.1"
NOTE="_L40s"
NOTE="_H100"

SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')_${REASONING}_${MESSAGE_COL}${NOTE}
DATA_NAME=$(basename "$TEST_FILE" .json)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/_partial_1028_vi_smart_0903_given_tools/_1903_only_response_r32_alpha64_batch_2x8_lr1e-5_61k_multiturn_unsloth-Qwen3-4B-Instruct-2507_no-thinking_messages_H100_ccu_10_20260320_025324"


RESULT_DIR="results/_partial_1028_vi_smart_0903_given_tools/_1903_only_response_r32_alpha64_batch_2x8_lr1e-5_61k_multiturn_unsloth-Qwen3-4B-Instruct-2507_no-thinking_messages_with_ground_truth_tools_history_H100_ccu_10_20260320_030139"


RESULT_DIR="results/_partial_1028_vi_smart_0903_given_tools/_1903_only_response_r32_alpha64_batch_2x8_lr1e-5_61k_multiturn_unsloth-Qwen3-4B-Instruct-2507_no-thinking_user_message_H100_ccu_10_20260320_031458"

echo "Normalized data generated"
python norm_predictions_file.py --input "$RESULT_DIR/raw_predictions.ndjson" --output "${RESULT_DIR}/predictions.ndjson"

echo "Run evaluation script on the generated predictions:"
python eval_tool_calls.py --pred_path "${RESULT_DIR}/predictions.ndjson"

python eval_args.py --pred_path "${RESULT_DIR}/predictions.ndjson" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"

python eval_summary_args.py --pred_path "${RESULT_DIR}/predictions.ndjson" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"