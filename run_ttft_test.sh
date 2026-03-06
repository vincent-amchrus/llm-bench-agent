#!/bin/bash
#
# 🚀 TTFT Benchmark Runner - Configuration File
# All settings defined here, passed to Python script via argparse
#

set -e  # Exit on error
set -o pipefail

# ═══════════════════════════════════════════════════════
# 🌐 API Configuration
# ═══════════════════════════════════════════════════════

# API Endpoint (vLLM, TGI, or OpenAI-compatible server)
BASE_URL="http://localhost:8268/v1"

# API Key (use "EMPTY" for local servers without auth)
API_KEY="EMPTY"

# Model name to benchmark
MODEL="Qwen/Qwen3.5-4B"


# ═══════════════════════════════════════════════════════
# ⚡ Load & Concurrency Settings
# ═══════════════════════════════════════════════════════

# Number of concurrent requests (CCU) - simulate load
CONCURRENCY="1"

# Per-request timeout in seconds
TIMEOUT="120"


# ═══════════════════════════════════════════════════════
# 📁 Input/Output Paths
# ═══════════════════════════════════════════════════════

# Input JSON: [{"user_message": "hello"}, {"user_message": "test"}, ...]
TEST_FILE="data/groundtruth/vivi_smart/_partial_18_vi_smart_labeled_0302.json"

# Optional: Tools JSON for function-calling benchmarks
# Leave empty or comment out if not using tools
TOOLS_FILE=""
TOOLS_FILE="data/tools/vivi_smart_tools.json"

# Output directory structure: results/{data_name}/{model}_ccu_{n}/ttft/
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/-/g')
DATA_NAME=$(basename "$TEST_FILE" .json)
RESULT_DIR="results/${DATA_NAME}/${SAFE_MODEL}_ccu_${CONCURRENCY}/ttft"

# Final output JSON file
OUTPUT_FILE="$RESULT_DIR/ttft_results.json"


# ═══════════════════════════════════════════════════════
# 🎛️ Feature Flags
# ═══════════════════════════════════════════════════════

# Enable thinking/reasoning mode in chat template
ENABLE_THINKING="false"  # "true" or "false"

# Show tqdm progress bar (set to "false" for CI/logs)
SHOW_PROGRESS="true"

# Python interpreter path (override if needed)
PYTHON_BIN="python3"


# ═══════════════════════════════════════════════════════
# 🚀 Execution Logic (No edits needed below)
# ═══════════════════════════════════════════════════════

echo "╔════════════════════════════════════════════╗"
echo "║  🚀 TTFT Benchmark - Starting              ║"
echo "╚════════════════════════════════════════════╝"
echo

# Validate input file
if [[ ! -f "$TEST_FILE" ]]; then
    echo "❌ Error: Test file not found: $TEST_FILE" >&2
    echo "💡 Hint: Create sample data with:" >&2
    echo '   echo "[{\"user_message\": \"Hello\"}]" > '"$TEST_FILE" >&2
    exit 1
fi

# Validate tools file if specified
if [[ -n "$TOOLS_FILE" && ! -f "$TOOLS_FILE" ]]; then
    echo "⚠️  Warning: Tools file not found: $TOOLS_FILE (ignoring)" >&2
    TOOLS_FILE=""
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Build argparse-compatible arguments
PYTHON_ARGS=(
    "--base-url" "$BASE_URL"
    "--api-key" "$API_KEY"
    "--model" "$MODEL"
    "--input-file" "$TEST_FILE"
    "--output-file" "$OUTPUT_FILE"
    "--concurrency" "$CONCURRENCY"
    "--timeout" "$TIMEOUT"
)

# Add optional flags
if [[ "$ENABLE_THINKING" == "true" ]]; then
    PYTHON_ARGS+=("--enable-thinking")
fi

if [[ "$SHOW_PROGRESS" != "true" ]]; then
    PYTHON_ARGS+=("--no-progress")
fi

if [[ -n "$TOOLS_FILE" && -f "$TOOLS_FILE" ]]; then
    PYTHON_ARGS+=("--tools-file" "$TOOLS_FILE")
fi

# Print config summary
echo "📋 Configuration:"
echo "   • Model:        $MODEL"
echo "   • Base URL:     $BASE_URL"
echo "   • API Key:      ${API_KEY:0:3}***"
echo "   • Concurrency:  $CONCURRENCY CCU"
echo "   • Input:        $TEST_FILE"
echo "   • Tools:        ${TOOLS_FILE:-none}"
echo "   • Output:       $OUTPUT_FILE"
echo "   • Thinking:     $ENABLE_THINKING"
echo "   • Progress:     $SHOW_PROGRESS"
echo "   • Timeout:      ${TIMEOUT}s"
echo

# Run benchmark
echo "⏱️  Running TTFT benchmark..."
echo "─────────────────────────────────────────────"

START_TIME=$(date +%s)

if $PYTHON_BIN measure_ttft.py "${PYTHON_ARGS[@]}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "─────────────────────────────────────────────"
    echo "✅ Benchmark completed in ${DURATION}s"
    echo "📁 Results: $OUTPUT_FILE"
    
    # Quick summary from JSON if jq available
    if command -v jq &> /dev/null && [[ -f "$OUTPUT_FILE" ]]; then
        echo
        echo "📊 Quick Summary:"
        jq -r '
            "   • Total: \(.summary.total_requests) | " +
            "✓ Success: \(.summary.successful_requests) | " +
            "✗ Failed: \(.summary.failed_requests)" +
            (if .summary.ttft_stats then 
                " | Mean TTFT: \(.summary.ttft_stats.mean)s" 
            else "" end)
        ' "$OUTPUT_FILE" 2>/dev/null || true
    fi
else
    echo "❌ Benchmark failed with exit code $?" >&2
    exit 1
fi