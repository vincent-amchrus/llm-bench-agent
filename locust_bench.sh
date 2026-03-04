
MODEL="Qwen/Qwen3.5-4B"
REASONING="no-thinking"
CCU=1
TEST_FILE="data/vivi_smart/_partial_90_vi_smart_labeled_0302.json"
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')_${REASONING}_ccu_${CCU}
DATA_NAME=$(basename "$TEST_FILE" .json)

LOCUST_PATH=results/${DATA_NAME}/${SAFE_MODEL}/locust_report.html

locust -f locust_bench.py --headless -u $CCU -r 1   --html=$LOCUST_PATH

echo "Stored result at $LOCUST_PATH"