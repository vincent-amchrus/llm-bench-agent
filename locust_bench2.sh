cat locust_bench2.sh
#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────
#  Configuration - change these values as needed
# ────────────────────────────────────────────────

MODEL="Qwen/Qwen3.5-4B"
MODEL="qwen3-4b-it-1102"
MODEL="unsloth/Qwen3-4B-Instruct-2507"

REASONING="no-thinking"           # or "thinking", "cot", etc.
CCU=5                             # concurrent users
RAMP_UP_RATE=1                    # users spawned per second

TEST_FILE="data/groundtruth/vivi_smart/_partial_9_vi_smart_labeled_0302.json"
TEST_FILE="data/vivi_smart/_partial_1k_vi_smart_labeled_0302.json"
TOOLS_FILE="data/tools/vivi_smart_tools.json"
# ────────────────────────────────────────────────
#  Derived values (usually no need to change)
# ────────────────────────────────────────────────

SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')_${REASONING}
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
        --csv="$LOCUST_CSV_BASE" \
        --csv-full-history \
        --loglevel INFO \
        --test-file "$TEST_FILE" \
        --tools-file "$TOOLS_FILE" \
        --base-url "http://localhost:8288" \
        --model "$MODEL" \
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