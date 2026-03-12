#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="data/vivi_global/_partial_1k_hindi_global_mapped_domain.json"
TEST_FILE="data/vivi_global/_partial_1k3_bahasa_global_mapped_domain.json"
TEST_FILE="data/groundtruth/vivi_smart/_partial_1k_vi_smart_0903_given_tools.json"
MODEL="gpt-4.1"

REASONING="no-thinking"
CCU=4


# 🗂️ Predictions path (matches infer.py & evaluate.py logic)
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

# 1️⃣ Inference (resumable)
# python infer.py --test_file "$TEST_FILE" --skip_on_error
# python async_infer.py --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32
python async_infer_gpt.py --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32

# 2️⃣ Quick exact-match metrics (exact name + args, multi-call safe)
# python eval_exact_match.py "$PRED_PATH"

# # 3️⃣ Full evaluation (semantic/schema-aware) (optional)
python eval_tool_calls.py --pred_path "$PRED_PATH" 

python eval_args.py --pred_path "$PRED_PATH" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"

python eval_summary_args.py --pred_path "$PRED_PATH" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"