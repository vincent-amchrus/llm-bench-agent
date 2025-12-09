# infer.py — crash-safe, resumable, NDJSON output
import os
import json
import hashlib
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from core.chat_client import chat_completion
from config.tools import ALL_TOOLS
from utils.misc import hash_input
from utils.misc import get_model_safe_name
load_dotenv()

def load_test_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_inference_func(base_url, model, api_key, tools, system_prompt=None):
    def infer(user_message):
        return chat_completion(
            messages=user_message,
            base_url=base_url,
            model=model,
            api_key=api_key,
            tools=tools,
            system_prompt=system_prompt,
            temperature=0.0
        )
    return infer

def hash_input(user_message) -> str:
    # Normalize + hash to handle dict/list order variance
    normalized = json.dumps(user_message, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

def load_completed_hashes(output_path: str) -> set:
    completed = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if "input_hash" in item:
                                completed.add(item["input_hash"])
                        except Exception:
                            pass  # skip corrupted lines
        except Exception as e:
            print(f"⚠️ Failed to read completed hashes: {e}")
    return completed

def atomic_append_line(output_path: str, data: dict):
    # Write to tmp, then append atomically
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
    # Append atomically (POSIX guarantees append is atomic if < 4KB)
    with open(tmp_path, "rb") as src, open(output_path, "ab") as dst:
        dst.write(src.read())
    os.remove(tmp_path)

def main():
    parser = argparse.ArgumentParser(description="Crash-safe, resumable inference")
    parser.add_argument("--test_file", default="data/test_cases.json", help="Test cases JSON")
    parser.add_argument("--output", default=None, help="NDJSON output path (default: results/<MODEL>/predictions.ndjson)")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--skip_on_error", action="store_true", help="Continue on inference error")
    args = parser.parse_args()

    data_name = args.test_file.split("/")[-1].split('.json')[0]
    # Auto-determine output path if not provided
    if args.output is None:
        model_name = get_model_safe_name()
        args.output = f"results/{data_name}/{model_name}/predictions.ndjson"

    # Ensure output dir exists (note: dirname of file path)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load config
    base_url = os.getenv("BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("API_KEY", "EMPTY")
    model = os.getenv("MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

    test_cases = load_test_cases(args.test_file)
    tools = ALL_TOOLS
    inference_func = make_inference_func(base_url, model, api_key, tools, args.system_prompt)

    # Load completed
    done_hashes = load_completed_hashes(args.output)

    # Filter
    to_run = []
    for i, case in enumerate(test_cases):
        h = hash_input(case["user_message"])
        if h not in done_hashes:
            to_run.append((i, case, h))

    print(f"🚀 Model: {model}")
    print(f"📊 Total: {len(test_cases)} | Done: {len(done_hashes)} | Remaining: {len(to_run)}")

    if not to_run:
        print("✅ All cases done.")
        return

    # Run with progress
    for i, case, h in tqdm(to_run, desc="Inferring"):
        try:
            pred_raw = inference_func(case["user_message"])
        except Exception as e:
            if args.skip_on_error:
                pred_raw = {
                    "error": str(e),
                    "content": None,
                    "tool_calls": [],
                    "usage": {}
                }
            else:
                raise

        # Build safe record
        record = {
            "index": case.get("index", i),
            "input_hash": h,
            "user_message": case["user_message"],
            "expected": case.get("tool_calls", []),
            "predicted": {
                "content": pred_raw.get("content", ""),
                "tool_calls": pred_raw.get("tool_calls", []),
                "error": pred_raw.get("error"),
                "usage": pred_raw.get("usage", {})
            },
            "timestamp": datetime.now().isoformat()
        }

        # Append immediately (crash-safe)
        atomic_append_line(args.output, record)

    print(f"✅ All done. Output: {args.output}")
    print(f"   Total lines: {sum(1 for _ in open(args.output)) if os.path.exists(args.output) else 0}")

if __name__ == "__main__":
    from datetime import datetime
    main()