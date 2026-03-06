#!/bin/bash
#
# TTFT Benchmark Configuration & Runner
# Configure parameters here and execute the benchmark
#

set -e  # Exit on error

# ===========================
# ⚙️  CONFIGURATION SECTION
# ===========================

# API Endpoint
BASE_URL="${BASE_URL:-http://localhost:8268/v1}"

# API Key (use "EMPTY" for local servers without auth)
API_KEY="${API_KEY:-EMPTY}"

# Model name to test
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"

# Input JSON file containing prompts
# Format: [{"user_message": "Hello"}, {"user_message": "How are you?"}, ...]
INPUT_FILE="${INPUT_FILE:-data/groundtruth/vivi_smart/_partial_9_vi_smart_labeled_0302.json}"
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
# Output file for results
OUTPUT_FILE="${OUTPUT_FILE:-results/ttft/$SAFE_MODEL/ttft_results_$(date +%Y%m%d_%H%M%S).json}"

# Number of concurrent requests (CCU)
CONCURRENCY="${CONCURRENCY:-4}"

# Enable thinking mode? (pass --enable-thinking flag if needed)
ENABLE_THINKING="${ENABLE_THINKING:-false}"

# Per-request timeout in seconds
TIMEOUT="${TIMEOUT:-120}"

# ===========================
# 🚀 EXECUTION
# ===========================

echo "🔧 TTFT Benchmark Configuration"
echo "──────────────────────────────"
echo "Base URL:     $BASE_URL"
echo "Model:        $MODEL"
echo "API Key:      ${API_KEY:0:3}***"
echo "Input File:   $INPUT_FILE"
echo "Output File:  $OUTPUT_FILE"
echo "Concurrency:  $CONCURRENCY"
echo "Thinking:     $ENABLE_THINKING"
echo "Timeout:      ${TIMEOUT}s"
echo "──────────────────────────────"

# Validate input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    echo "💡 Create a test file first, e.g.:"
    echo '   echo "[{\"user_message\": \"Hello\"}, {\"user_message\": \"Test\"}]" > test_prompts.json'
    exit 1
fi

# Build command arguments
CMD_ARGS=(
    "--base-url" "$BASE_URL"
    "--api-key" "$API_KEY"
    "--model" "$MODEL"
    "--input-file" "$INPUT_FILE"
    "--output-file" "$OUTPUT_FILE"
    "--concurrency" "$CONCURRENCY"
    "--timeout" "$TIMEOUT"
)

# Add optional flags
if [[ "$ENABLE_THINKING" == "true" ]]; then
    CMD_ARGS+=("--enable-thinking")
fi

# Run the benchmark
echo "🚀 Starting benchmark..."
python3 measure_ttft.py "${CMD_ARGS[@]}"

echo "✅ Benchmark completed!"