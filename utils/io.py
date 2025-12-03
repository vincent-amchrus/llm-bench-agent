# utils/io.py
import json
import csv
import os
from datetime import datetime
from typing import Dict, Any

def save_json(data: Dict[str, Any], output_dir: str = "results", prefix: str = "eval") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{prefix}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def save_csv(results: list, output_dir: str = "results", prefix: str = "eval") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{prefix}_{ts}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        if not results:
            return path
        fieldnames = [
            "index", "user_message", "match", "expected", "predicted", "content"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "index": r["index"],
                "user_message": r["user_message"],
                "match": r["match"],
                "expected": json.dumps(r["expected"], ensure_ascii=False),
                "predicted": json.dumps(r["predicted"], ensure_ascii=False),
                "content": (r["content"] or "").replace("\n", " ").strip()[:200]  # truncate
            })
    return path