MODEL="Qwen/Qwen3.5-4B"
REASONING="no-thinking"
CCU=10
PRED_PATH="results/_partial_1k_vi_smart_0903_given_tools/Qwen-Qwen3.5-4B_no-thinking_ccu_10/predictions.ndjson"


MODEL="senlm-4b-fc-vivi"
REASONING="no-thinking"
CCU=10
PRED_PATH="results/_partial_1k_vi_smart_0903_given_tools/senlm-4b-fc-vivi_no-thinking_ccu_10/predictions.ndjson"


python eval_tool_calls.py --pred_path "$PRED_PATH" 

python eval_args.py --pred_path "$PRED_PATH" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"

python eval_summary_args.py --pred_path "$PRED_PATH" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"