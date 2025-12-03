# run.sh (final version)

#!/bin/bash
set -e

# Setup
mkdir -p data results

# Optional: generate test cases if missing
if [ ! -f "data/test_cases.json" ]; then
  echo '[]' > data/test_cases.json
  echo "⚠️  Created empty data/test_cases.json — please populate!"
fi

# Step 1: Inference (resumable, safe)
echo "🚀 Running inference..."
python infer.py \
  --test_file data/tiny_test.json \
  --output results/tiny_test/predictions.ndjson \
  --skip_on_error

# Step 2: Evaluation (separate, reproducible)
echo "🔍 Running evaluation..."
python evaluate.py \
  --test_file data/tiny_test.json \
  --predictions results/tiny_test/predictions.ndjson \
  --verbose


echo "✅ Done!"