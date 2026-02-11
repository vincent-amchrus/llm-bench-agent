Here's a revised and professional **`README.md`** section that clearly explains how to set up and run evaluation using your provided `.env` and `run.sh` configurations:

---

## 🧪 Step-by-Step Evaluation Guide

To evaluate your fine-tuned LLM on tool-calling tasks, follow these steps:

### 1. Configure Environment Variables

Create a `.env` file in your project root with the following settings:

```env
BASE_URL="http://localhost:8268/v1"
API_KEY="EMPTY"
TOOLS_PATH="config/tools.json"  # or your actual tools JSON path
```

> 💡 **Note**:  
> - `BASE_URL` should point to your vLLM OpenAI-compatible API endpoint.  
> - `TOOLS_PATH` must reference a valid JSON file containing your function/tool definitions (e.g., `vf_global_tools2.json` or `vivi_smart_tools.json`).  
> - The evaluation scripts automatically load tools from this path via `config/tools.py`.

---

### 2. Prepare Your Test Data

Place your labeled test cases in the `data/` directory. Each sample must include:
- `user_message`: list of chat messages (typically `[{"role": "user", "content": "..."}]`)
- `tool_calls`: ground-truth tool call(s) in OpenAI format (`[{"name": "...", "arguments": {...}}]`)

Example structure (`data/groundtruth/vivi_smart/_partial_6k4_vi_smart_labeled_0302.json`):
```json
[
    {
        "_source_sheet":"25466_03_LLM Movie Search",
        "tool_calls":[
            {
                "name":"movie_tool",
                "arguments":{
                    "rewrite_message":"lịch chiếu phim cuối tuần này",
                    "movie_book_tickets":false,
                    "movie_information":false
                }
            }
        ],
        "user_message":"lịch chiếu phim cuối tuần này",
        "_source_file":"TestReport_SystemTest_VFVA_Vivi 2.0_NewRewrite_20250918_Retest.xlsx",
        "function":"movie_tool"
    }
]
```

---

### 3. Run Inference + Evaluation

Use the provided `run.sh` script (or adapt it) to perform end-to-end evaluation:

```bash
#!/bin/bash
set -e

# 🔧 Test file and model
TEST_FILE="data/groundtruth/vivi_smart/_partial_6k4_vi_smart_labeled_0302.json"
MODEL="_1002_only_response_r32_alpha64_batch_2x8_lr1e-5_31k_unsloth-Qwen3-4B-Instruct-2507"

# 🗂️ Auto-generate output path
DATA_NAME=$(basename "$TEST_FILE" .json)
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
PRED_PATH="results/${DATA_NAME}/${SAFE_MODEL}/predictions.ndjson"

echo "🚀 Running: MODEL=${MODEL}, TEST_FILE=${TEST_FILE}"

# 1️⃣ Run inference (resumable, skips completed cases)
python async_infer_gpt.py \
  --model "$MODEL" \
  --test_file "$TEST_FILE" \
  --skip_on_error \
  --max_concurrent 32

# 2️⃣ Evaluate tool selection & arguments
python eval_tool_calls.py --pred_path "$PRED_PATH"    # Tool name accuracy + confusion matrix
python eval_args.py --pred_path "$PRED_PATH"          # Per-tool argument correctness (with examples)
python eval_summary_args.py --pred_path "$PRED_PATH"  # Concise summary report
```

> ✅ **Outputs** will be saved under `results/<dataset>/<model>/`:
> - `predictions.ndjson`: raw predictions (resumable checkpoint)
> - `evaluation_summary.md` / `.pdf`: human-readable report
> - `confusion_matrix.png`: visual tool selection performance
> - `metrics_args.json`: structured argument-level metrics

---

### 4. (Optional) Serve Your Model First

If you're evaluating a local vLLM model (not GPT), start the server first:

```bash
# Example: serving.sh
MODEL_PATH=/path/to/your/model
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --port 8268 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Then ensure `BASE_URL="http://localhost:8268/v1"` in `.env`.

---

This setup supports:
- **Resumable inference** (crash-safe via input hashing)
- **Multi-tool, multi-argument evaluation**
- **Detailed error analysis** (missing keys, wrong values, extra calls)
- **Both GPT and self-hosted models**

Happy evaluating! 🚀