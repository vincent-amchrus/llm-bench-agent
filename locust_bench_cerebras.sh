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

TOOLS_FILE="data/tools/vivi_smart_tools_0903.json"
TEST_FILE="data/groundtruth/vivi_smart/_partial_1k_vi_smart_0903_given_tools.json"

# ────────────────────────────────────────────────
#  Derived values (usually no need to change)
# ────────────────────────────────────────────────

SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')_${REASONING}
DATA_NAME=$(basename "$TEST_FILE" .json)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/${DATA_NAME}/cerebras-${SAFE_MODEL}_ccu_${CCU}_${TIMESTAMP}"
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