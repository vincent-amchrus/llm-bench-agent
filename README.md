Ensure you served model using vLLM on the same node. For example at port 8268, key "EMPTY" (default none key)

### Prepare `.env`

BASE_URL="http://localhost:8268/v1"
API_KEY="EMPTY"

TOOL_PATH = "data/tools/vivi_smart_tools.json"
TOOL_PATH = "data/tools/vivi_global_tools2.json"
TOOL_PATH = "data/tools/vivi_smart_tools_0903.json"


### Edit `eval_local.sh` as below: 


```
    TEST_FILE="data/vivi_global/_partial_1k6_en_global_labeled.json"
    #TEST_FILE="data/vivi_global/_13k5_en_global_labeled.json"
    TEST_FILE="data/vivi_smart/_partial_6k4_vi_smart_labeled_0302.json"
    #TEST_FILE="data/vivi_global/_2k7_hindi_global_args_with_non_call.json"
    #TEST_FILE="data/vivi_global/_3k_bahasa_global_args_with_non_call.json"
    # TEST_FILE="data/groundtruth/global/_partial_12_en_global_labeled.json"
    TEST_FILE="data/vivi_smart/_partial_90_vi_smart_labeled_0302.json"
    TEST_FILE="data/groundtruth/vivi_smart/_partial_1k_vi_smart_labeled_0302.json"
    TEST_FILE="data/groundtruth/vivi_smart/_11k_vi_smart_0903.json"
    TEST_FILE="data/groundtruth/vivi_smart/_11k_vi_smart_0903_given_tools.json"

    TEST_FILE="data/groundtruth/vivi_smart/multiturn/_partial_20_vi_smart_0903_given_tools.json"
    TEST_FILE="data/groundtruth/vivi_smart/multiturn/_partial_1028_vi_smart_0903_given_tools.json"

    MESSAGE_COL="user_message"
    #MESSAGE_COL="messages"
    MESSAGE_COL="messages_with_ground_truth_tools_history"


    MODEL="Qwen/Qwen3.5-4B"
    MODEL="senlm-4b-fc-vivi"

    REASONING="no-thinking"
    CCU=10

    TEMPERATURE=0
    TOP_P=0.8
    PRESENCE_PENALTY=1.5


    DATA_NAME=$(basename "$TEST_FILE" .json)
    SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')_${REASONING}_ccu_${CCU}_${MESSAGE_COLS}
    echo "🚀 Running: SAFE_MODEL=${SAFE_MODEL}"
    PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

    echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

    python async_infer.py \
        --model "$MODEL" \
        --safe_model "$SAFE_MODEL" \
        --test_file "$TEST_FILE" \
        --skip_on_error \
        --max_concurrent "$CCU" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --presence_penalty "$PRESENCE_PENALTY" \
        --message_column "$MESSAGE_COL"
        # --enable_thinking \
        # --system_prompt "Respond in the same language as the user.\n\nCall a tool ONLY when it is clearly necessary to answer correctly using one of the available tools.\n\nNormal chat, greetings, personal questions, jokes → no tool calls. Just reply normally.\n\nNever call a tool unnecessarily."

    python eval_tool_calls.py --pred_path "$PRED_PATH" 

    python eval_args.py --pred_path "$PRED_PATH" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"

    python eval_summary_args.py --pred_path "$PRED_PATH" --model "$MODEL" --reasoning "$REASONING" --ccu "$CCU"

```


### Run `bash eval_local.sh`