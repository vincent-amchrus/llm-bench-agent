import json
from collections import defaultdict

def count_by_function(data):
    """
    Thống kê số lượng sample cho mỗi loại function
    """
    function_counts = defaultdict(int)
    for item in data:
        func = item.get("function") or "__NO_FUNCTION__"
        function_counts[func] += 1
    
    return function_counts


if __name__ == "__main__":
    with open(
        "/media/4TB/haict/function-calling/data/vivi_global_test/normalized_autogen_full_vivi_en_global_13k5_test.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    stats = count_by_function(data)

    print("\n=== Thống kê chi tiết ===")
    total = 0
    for func, count in sorted(stats.items()):
        print(f"{func}: {count}")
        total += count
    
    print(f"\nTổng cộng: {total}")
