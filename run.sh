
# 0. Optional
# Sample balanced test set (optional)
# Full control
python generate_sample_test.py \
  --input data/full_test.json \
  --output data/custom_balanced_10.json \
  --max_per_function 10 \
  --random_seed 123

# 1. Run inference
TEST_FILE="data/custom_balanced_10.json"

python infer.py \
  --test_file $TEST_FILE \
  --skip_on_error

# 2. Run evaluation (auto finds predictions & writes to same folder)
python evaluate.py \
  --test_file $TEST_FILE \
  --verbose