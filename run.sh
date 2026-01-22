#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="data/_partial_vi_full_translate_vivi_global_12k4.json"
TEST_FILE="data/_partial_normalized_autogen_full_vivi_en_global_13k5_test.json"
# TEST_FILE="data/vivi_smart/_partial_vivi_smart.json"
TEST_FILE='data/vivi_global/_5k7_bahasa_global_mapped_domain.json'
TEST_FILE="data/vivi_global/_13k5_en_global_labeled.json"
TEST_FILE='data/vivi_global/_5k2_hindi_global_mapped_domain.json'
TEST_FILE="data/vivi_smart/_partial_new_14k5_8tools_include_autogen_vivi_smart_autogen.json"
TEST_FILE="data/vivi_smart/_partial_6k4_labeled_1411_vi_smart.json"

MODEL="ckp300_1201_only_response_r16_alpha32_4epoch_batch_2x8_packing_lr1e-5_6k_nonfc_22k_fc_unsloth-Qwen3-4B-Instruct-2507"

MODEL="_1601_toon_only_response_r16_alpha32_batch_2x8_lr1e-5_6k_nonfc_22k_fc_unsloth-Qwen3-4B-Instruct-2507"
MODEL="_1601_toon_only_response_r16_alpha32_batch_2x8_lr1e-5_full100k_fc_unsloth-Qwen3-4B-Instruct-2507"
MODEL="_1901_toon_only_response_r16_alpha32_batch_2x8_lr2e-5_10k8_balance_fc_4k_non_fc_unsloth-Qwen3-4B-Instruct-2507"
MODEL="unsloth/Qwen3-4B-Instruct-2507"



# 🗂️ Predictions path (matches infer.py & evaluate.py logic)
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

# 1️⃣ Inference (resumable)
# python infer.py --test_file "$TEST_FILE" --skip_on_error
python async_infer.py \
    --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32 \
    --use_toon_format 
# python async_infer_gpt.py --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32

# 2️⃣ Quick exact-match metrics (exact name + args, multi-call safe)
# python eval_exact_match.py "$PRED_PATH"

# # 3️⃣ Full evaluation (semantic/schema-aware) (optional)
python eval_tool_calls.py --pred_path "$PRED_PATH" 

python eval_args.py --pred_path "$PRED_PATH"