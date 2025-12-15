#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="data/vivi_smart_test/gpt_label_vietnamese_11k_vivi_smart.json"

# Fill model name here
MODEL="Qwen3-1.7B-FC"

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