#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="/media/4TB/haict/function-calling/data/vivi_smart_test/_partial_14k5_8tools_include_autogen_vivi_smart_autogen.json"

# Fill model name here
MODEL="Sen-32B-GRPO"

# 🗂️ Predictions path (matches infer.py & evaluate.py logic)
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

# 1️⃣ Inference (resumable)
python infer.py --test_file "$TEST_FILE" --skip_on_error

# 2️⃣ Quick exact-match metrics (exact name + args, multi-call safe)
python eval_exact_match.py "$PRED_PATH"

# # 3️⃣ Full evaluation (semantic/schema-aware) (optional)
# python evaluate.py --test_file "$TEST_FILE"