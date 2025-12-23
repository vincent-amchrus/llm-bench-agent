import json
import random
from collections import defaultdict

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Quota mong muốn cho từng function
TARGET_COUNTS = {
    "VEHICLE_CONTROL": 1000,
    "VEHICLE_INFO": 1500,
}

def sample_by_function_with_quota(data, target_counts):
    """
    Sample data theo quota cho từng function.
    - Function có trong target_counts: lấy đúng số lượng (nếu đủ)
    - Function không có trong target_counts: lấy toàn bộ
    """
    # 1. Group theo function
    function_groups = defaultdict(list)
    for item in data:
        func = item.get("function", "__NO_FUNCTION__")
        function_groups[func].append(item)

    sampled_data = []

    print("=== Sampling summary ===")
    for func, items in function_groups.items():
        n_total = len(items)

        if func in target_counts:
            n_target = target_counts[func]
            n_sample = min(n_target, n_total)

            sampled_items = random.sample(items, k=n_sample)
            print(f"[{func}] total={n_total}, sampled={n_sample} (quota={n_target})")
        else:
            sampled_items = items
            print(f"[{func}] total={n_total}, sampled={n_total} (take all)")

        sampled_data.extend(sampled_items)

    print(f"\nTổng sample lấy được: {len(sampled_data)}")
    return sampled_data


# =========================
# Ví dụ sử dụng
# =========================
if __name__ == "__main__":
    input_path = "data/vivi_global_test/vi_full_translate_vivi_global_12k4.json"
    output_path = "data/vivi_global_test/_partial_vi_full_translate_vivi_global_12k4.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled = sample_by_function_with_quota(data, TARGET_COUNTS)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
