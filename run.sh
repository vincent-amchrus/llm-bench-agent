# 1. Run inference
python infer.py \
  --test_file data/test_each_10.json \
  --skip_on_error

# 2. Run evaluation (auto finds predictions & writes to same folder)
python evaluate.py \
  --test_file data/test_each_10.json \
  --verbose