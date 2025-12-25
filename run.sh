#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="data/_partial_normalized_autogen_full_vivi_en_global_13k5_test.json"
TEST_FILE="data/_partial_vi_full_translate_vivi_global_12k4.json"
TEST_FILE="data/vivi_smart/_partial_vivi_smart.json"

# Fill model name here
MODEL="fc_neg_ckp_4674" 
MODEL="tool_ace_ckp_5993"
MODEL="unsloth/Qwen3-4B-Instruct-2507"
MODEL="fc_global_39k_neg"

# 🗂️ Predictions path (matches infer.py & evaluate.py logic)
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

# 1️⃣ Inference (resumable)
# python infer.py --test_file "$TEST_FILE" --skip_on_error
python async_infer.py --test_file "$TEST_FILE" --skip_on_error

# 2️⃣ Quick exact-match metrics (exact name + args, multi-call safe)
# python eval_exact_match.py "$PRED_PATH"

# # 3️⃣ Full evaluation (semantic/schema-aware) (optional)
python eval_tool_calls.py --pred_path "$PRED_PATH"