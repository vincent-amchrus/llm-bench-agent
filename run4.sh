#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="data/_partial_vi_full_translate_vivi_global_12k4.json"
TEST_FILE="data/_partial_normalized_autogen_full_vivi_en_global_13k5_test.json"
# TEST_FILE="data/vivi_smart/_partial_vivi_smart.json"
TEST_FILE="data/vivi_smart/_partial_new_14k5_8tools_include_autogen_vivi_smart_autogen.json"
# Fill model name here
MODEL="mix_100k"
MODEL="unsloth/Qwen3-4B-Instruct-2507"
MODEL="data_6k5"

# 🗂️ Predictions path (matches infer.py & evaluate.py logic)
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

# 1️⃣ Inference (resumable)
# python infer.py --test_file "$TEST_FILE" --skip_on_error
python async_infer.py --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32

# 2️⃣ Quick exact-match metrics (exact name + args, multi-call safe)
# python eval_exact_match.py "$PRED_PATH"

# # 3️⃣ Full evaluation (semantic/schema-aware) (optional)
python eval_tool_calls.py --pred_path "$PRED_PATH" 