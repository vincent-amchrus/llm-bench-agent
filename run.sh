# 1. Run inference

TEST_FILE="data/vi_test_each_max_1002.json"

python infer.py \
  --test_file $TEST_FILE \
  --skip_on_error

# 2. Run evaluation (auto finds predictions & writes to same folder)
python evaluate.py \
  --test_file $TEST_FILE \
  --verbose