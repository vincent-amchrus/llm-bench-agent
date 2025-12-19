import json
import random
from collections import defaultdict
from math import ceil

RANDOM_SEED = 42
SAMPLE_RATIO = 0.25

random.seed(RANDOM_SEED)

def sample_by_function(data, ratio=0.25):
    """
    Lấy sample theo tỉ lệ cho mỗi loại function
    """
    # 1. Group theo function
    function_groups = defaultdict(list)
    for item in data:
        func = item.get("function", "__NO_FUNCTION__")
        function_groups[func].append(item)

    # 2. Sample mỗi group
    sampled_data = []

    for func, items in function_groups.items():
        n_total = len(items)
        n_sample = max(1, ceil(n_total * ratio))  # ít nhất 1 sample

        sampled_items = random.sample(items, k=min(n_sample, n_total))
        sampled_data.extend(sampled_items)

        print(f"[{func}] total={n_total}, sampled={len(sampled_items)}")

    return sampled_data


# =========================
# Ví dụ sử dụng
# =========================

if __name__ == "__main__":
    # Load data từ file json nếu cần
    with open("data/vivi_smart_test/_full_14k5_8tools_include_autogen_vivi_smart_autogen.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled = sample_by_function(data, ratio=SAMPLE_RATIO)

    # Lưu ra file nếu cần
    with open("data/vivi_smart_test/_partial_14k5_8tools_include_autogen_vivi_smart_autogen.json", "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\nTổng sample lấy được: {len(sampled)}")
