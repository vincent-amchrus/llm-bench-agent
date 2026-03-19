#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────
#  Configuration - change these values as needed
# ────────────────────────────────────────────────

MODEL="Qwen/Qwen3.5-4B"
BASE_URL="http://localhost:8268"
API_KEY="EMPTY"          # Set your API key if needed

MODEL="qwen3-4b-it-1102"
MESSAGE_COL="messages_with_ground_truth_tools_history"
MESSAGE_COL="user_message"
MESSAGE_COL="messages"


REASONING="no-thinking"           # or "thinking", "cot", etc.
CCU=1                             # concurrent users
RAMP_UP_RATE=1                    # users spawned per second

TEST_FILE="data/groundtruth/vivi_smart/_partial_1k_vi_smart_0903_given_tools.json"
TOOLS_FILE="data/tools/vivi_smart_tools_0903.json"

TEST_FILE="data/groundtruth/vivi_smart/_partial_1k_vi_smart_labeled_0302.json"
TOOLS_FILE="data/tools/vivi_smart_tools.json"



TEST_FILE="data/groundtruth/vivi_smart/multiturn/_partial_1028_vi_smart_0903_given_tools.json"
TEST_FILE="data/groundtruth/vivi_smart/multiturn/_partial_20_vi_smart_0903_given_tools.json"
TOOLS_FILE="data/tools/vivi_smart_tools_0903.json"

# ────────────────────────────────────────────────
#  Derived values (usually no need to change)
# ────────────────────────────────────────────────


NOTE="_RTX_3090_vllm_0.17.1"
NOTE="_L40s"

SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')_${REASONING}${NOTE}
DATA_NAME=$(basename "$TEST_FILE" .json)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/${DATA_NAME}/${SAFE_MODEL}_ccu_${CCU}_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

LOCUST_HTML="${RESULT_DIR}/locust_report.html"
LOCUST_CSV_BASE="${RESULT_DIR}/locust_stats"   # will get _requests.csv, _failures.csv, etc.

echo "┌───────────────────────────────────────────────┐"
echo "│             Locust benchmark                  │"
echo "├───────────────────────────────────────────────┤"
echo "│ Model       : ${MODEL}"
echo "│ Reasoning   : ${REASONING}"
echo "│ Users       : ${CCU}"
echo "│ Test file   : ${TEST_FILE}"
echo "│ Output dir  : ${RESULT_DIR}"
echo "└───────────────────────────────────────────────┘"

# ────────────────────────────────────────────────
#  Run Locust
# ────────────────────────────────────────────────

if [ ! -f "$LOCUST_HTML" ]; then
    locust -f locust_bench.py \
        --headless \
        -u "$CCU" \
        -r "$RAMP_UP_RATE" \
        --html="$LOCUST_HTML" \
        --result-dir="$RESULT_DIR" \
        --csv="$LOCUST_CSV_BASE" \
        --csv-full-history \
        --loglevel INFO \
        --test-file "$TEST_FILE" \
        --tools-file "$TOOLS_FILE" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --reasoning "$REASONING" \
        --message_col "$MESSAGE_COL"
else
    echo "HTML report already exists: $LOCUST_HTML"
fi

# ────────────────────────────────────────────────
#  Final message
# ────────────────────────────────────────────────

if [[ -f "$LOCUST_HTML" ]]; then
    echo ""
    echo "Benchmark finished."
    echo "HTML report  →  $LOCUST_HTML"
    echo "CSV stats    →  ${LOCUST_CSV_BASE}*.csv"
    echo ""
    echo "You can open the report in browser:"
    echo "brave-browser $LOCUST_HTML"
else
    echo "Error: HTML report was not created." >&2
    exit 1
fi






echo "Normalized data generated"
python norm_predictions_file.py --input "$RESULT_DIR/raw_predictions.ndjson" --output "${RESULT_DIR}/predictions.ndjson"

echo "Run evaluation script on the generated predictions:"
python eval_tool_calls.py --pred_path "${RESULT_DIR}/predictions.ndjson"

python eval_args.py --pred_path "${RESULT_DIR}/predictions.ndjson" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"

python eval_summary_args.py --pred_path "${RESULT_DIR}/predictions.ndjson" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"