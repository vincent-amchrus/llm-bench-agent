#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="data/groundtruth/global/_partial_1k6_en_global_labeled.json"
TEST_FILE="data/groundtruth/vivi_smart/_partial_6k4_vi_smart_labeled_0302.json"
#TEST_FILE="data/groundtruth/global/_13k5_en_global_labeled.json"

# MODEL="fp8_1102_only_response_r32_alpha64_batch_2x8_lr1e-5_30k5_unsloth-Qwen3-4B-Instruct-2507"
# MODEL="senlm-4b-fc-vivi"
# MODEL="models-merged_model-qwen3-4b-it-1102"
# MODEL="dtype_fp16_qwen3-4b-it-1102-quantize_fp8"

# MODEL="_2502_only_response_r256_alpha512_batch_2x8_lr1e-5_30k5_unsloth-Qwen3-0.6B"
# MODEL="_2502_only_response_r128_alpha256_batch_2x8_lr1e-5_30k5_unsloth-Qwen3-1.7B"
# MODEL="senlm-4b-fc-vivi"

# # 🗂️ Predictions path (matches infer.py & evaluate.py logic)
# DATA_NAME=$(basename "$TEST_FILE" .json)
# SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
# PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

# echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

# # 1️⃣ Inference (resumable)
# # python infer.py --test_file "$TEST_FILE" --skip_on_error
# python async_infer.py \
#     --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32
#     # --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32 --use_toon_format 
# # python async_infer_gpt.py --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32

# # 2️⃣ Quick exact-match metrics (exact name + args, multi-call safe)
# # python eval_exact_match.py "$PRED_PATH"

# # # 3️⃣ Full evaluation (semantic/schema-aware) (optional)
# python eval_tool_calls.py --pred_path "$PRED_PATH" 

# python eval_args.py --pred_path "$PRED_PATH"

# python eval_summary_args.py --pred_path "$PRED_PATH"










MODEL="Qwen/Qwen3.5-4B"

#   Predictions path (matches infer.py & evaluate.py logic)
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

#    nference (resumable)
# python infer.py --test_file "$TEST_FILE" --skip_on_error
python async_infer.py \
    --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 256
    # --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32 --use_toon_format
# python async_infer_gpt.py --model "$MODEL" --test_file "$TEST_FILE" --skip_on_error --max_concurrent 32

#    uick exact-match metrics (exact name + args, multi-call safe)
# python eval_exact_match.py "$PRED_PATH"

# #    ull evaluation (semantic/schema-aware) (optional)
python eval_tool_calls.py --pred_path "$PRED_PATH" 

python eval_args.py --pred_path "$PRED_PATH"

python eval_summary_args.py --pred_path "$PRED_PATH"