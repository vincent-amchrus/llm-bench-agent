#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)

TEST_FILE="data/vivi_global/_partial_1k6_en_global_labeled.json"
#TEST_FILE="data/vivi_global/_13k5_en_global_labeled.json"
TEST_FILE="data/vivi_smart/_partial_6k4_vi_smart_labeled_0302.json"
#TEST_FILE="data/vivi_global/_2k7_hindi_global_args_with_non_call.json"
#TEST_FILE="data/vivi_global/_3k_bahasa_global_args_with_non_call.json"
TEST_FILE="data/groundtruth/global/_partial_12_en_global_labeled.json"

MODEL="Qwen/Qwen3.5-4B"
REASONING="-thinking"
CCU=4

# 🗂️ Predictions path (matches infer.py & evaluate.py logic)
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')${REASONING}_ccu_${CCU}

PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

#  1️⃣ Inference (resumable)
# python infer.py --test_file "$TEST_FILE" --skip_on_error
python async_infer.py \
    --model "$MODEL" \
    --safe_model "$SAFE_MODEL" \
    --test_file "$TEST_FILE" \
    --skip_on_error \
    --max_concurrent "$CCU" \
    --enable_thinking
    # --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32 --use_toon_format
# python async_infer_gpt.py --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32

#  2️⃣ Quick exact-match metrics (exact name + args, multi-call safe)
# python eval_exact_match.py "$PRED_PATH"

# #  3️⃣ Full evaluation (semantic/schema-aware) (optional)
python eval_tool_calls.py --pred_path "$PRED_PATH" 

python eval_args.py --pred_path "$PRED_PATH"

python eval_summary_args.py --pred_path "$PRED_PATH"