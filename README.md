Here's a professional **`README.md`** for your tool-calling evaluation project, tailored to match your codebase and workflow preferences (e.g., resumable inference, field-level evaluation, configurable matching, GPU/server-aware setup):

---

# 🛠️ Tool-Calling Evaluation Framework

A robust, modular framework for **evaluating function/tool-calling accuracy** of LLMs — supporting **exact, semantic, time-aware, and fuzzy argument matching**, resumable inference, and fine-grained metrics beyond simple full-match.

Designed for reproducible, crash-safe evaluation with support for multilingual test sets, LoRA fine-tuned models (e.g., via vLLM), and field-level debugging.

---

## ✨ Features

- ✅ **Field-level argument matching**: per-parameter strategies (`exact`, `normalized`, `semantic`, `time`, `fuzzy`)
- 🧪 **Resumable & crash-safe inference**: auto-skips completed cases using SHA-256 input hashing + `.ndjson` checkpointing
- 📊 **Rich evaluation metrics**:
  - Exact-match accuracy
  - Tool name precision/recall/F1
  - Argument accuracy (call-level & field-level)
  - Strict tool-call-level TP/FP/FN (for F1)
- 🧠 **Semantic matching** via `sentence-transformers`, configurable thresholds & models
- ⚙️ **Environment-driven config** (`.env`), tool schema-defined match modes
- 🌐 **vLLM + LoRA-ready**: pre-built `serving.sh` for Qwen3 with LoRA adapters
- 🌍 **Multilingual test set sampling** (`en`/`vi`) with balanced function distribution

---

## 📁 Project Structure

```
.
├── serving.sh                 # vLLM launch script (Qwen3 + LoRA support)
├── run.sh                     # End-to-end: infer → evaluate
├── infer.py                   # Resumable inference (NDJSON output)
├── evaluate.py                # Offline evaluation (uses precomputed preds)
├── generate_sample_test.py    # Sample test sets per function (balanced)
│
├── core/
│   ├── chat_client.py         # OpenAI-compatible client for local LLMs
│   ├── evaluator.py           # Main evaluation logic
│   └── argument_matcher.py    # Per-field matching with multiple modes
│
├── config/
│   └── tools.py              # Load tools from JSON (env: TOOL_PATH)
│
├── utils/
│   ├── io.py                  # Save JSON/CSV reports
│   └── misc.py                # Hashing, safe model name, etc.
│
├── data/                      # Sample test cases (JSON)
├── results/                   # Auto-generated: predictions, evals, summaries
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Install deps (ensure torch & CUDA-compatible)
pip install -r requirements.txt

# Copy and edit .env (see below)
cp .env.example .env
```

> 🔑 **Key `.env` variables**:
> ```env
> MODEL=Qwen/Qwen3-1.7B
> BASE_URL=http://localhost:8268/v1
> API_KEY=EMPTY
> TOOL_PATH=config/tools.json
> MATCH_MODE=semantic
> SIMILARITY_THRESHOLD=0.85
> EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
> DEVICE=cuda
> ```

### 2. Serve Model (vLLM + LoRA)

```bash
bash serving.sh   # starts on :8268, enables LoRA + auto tool choice
```

### 3. Run Inference & Evaluation

```bash

#!/bin/bash
set -e

# 🔧 Auto-resolve (from env or fallback)
TEST_FILE="data/vi_test_each_max_1002.json"

# Fill model name here
MODEL="checkpoint_0912_fc_chat_990"

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
```

Output:
```
✅ Saved full report to: results/vi_test_each_max_10/checkpoint-180/toolcall_eval_20251211_143022.json
   Per-case CSV:         results/.../toolcall_eval_20251211_143022.csv
   Summary text:         results/.../summary.txt   <-- 👈 human-readable!
```

---

## 📈 Evaluation Output (`summary.txt`)

```
🏆 TOOL-CALLING EVALUATION SUMMARY
============================================================
Generated: 2025-12-11 14:30:22
Model:     checkpoint-180
Test File: vi_test_each_max_10

Total Cases:              48
Exact Match Accuracy:     85.42%

📊 Tool Name Accuracy:
   Precision:             96.77%
   Recall:                93.75%
   F1:                    95.24%

📊 Argument Accuracy:
   Call-level (if name✓): 91.30%
   Field-level:           88.64% (78/88 fields)

📊 Strict Tool-call Level (exact match):
   TP=44 FP=1 FN=2
   Precision:             97.78%
   Recall:                95.65%
   F1:                    96.70%
```

---

## 🛠️ Advanced Usage

### Custom Tool Schema Matching

In your `tools.json`, annotate fields with `match_mode`:

```json
{
  "type": "function",
  "function": {
    "name": "weather_tool",
    "parameters": {
      "properties": {
        "location": {
          "type": "string",
          "match_mode": "semantic"
        },
        "time": {
          "type": "string",
          "match_mode": "time"
        },
        "unit": {
          "type": "string",
          "enum": ["C", "F"],
          "match_mode": "exact"
        }
      }
    }
  }
}
```

### Resume Crashed Inference

`infer.py` auto-detects completed cases via `input_hash` and skips them — safe to re-run.


---

## 🧪 Requirements

- Python ≥ 3.9
- GPU (CUDA) recommended for inference & embedding (optional but faster)
- `vLLM` ≥ 0.6.0 (separately installed; not in `requirements.txt`)
- `sentence-transformers` for semantic/fuzzy matching (optional fallback: text normalization)

---

## 📜 License

MIT — feel free to adapt for internal/tool-calling research use.

---


Happy evaluating! 🚀