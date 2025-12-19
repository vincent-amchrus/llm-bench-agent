import os
import json
import hashlib
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from core.chat_client import chat_completion
from config.tools import ALL_TOOLS
from utils.misc import get_model_safe_name

load_dotenv()

thread_local = threading.local()

def load_test_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def hash_input(user_message) -> str:
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
                                completed.add(item["input_hash"])
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️ Failed to read completed hashes: {e}")
    return completed

def atomic_append_line(output_path: str, data: dict, lock: threading.Lock):
    # Thread-safe atomic append
    tmp_path = output_path + f".tmp.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
    
    with lock: 
        with open(tmp_path, "rb") as src, open(output_path, "ab") as dst:
            dst.write(src.read())
    
    try:
        os.remove(tmp_path)
    except:
        pass

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

def process_case(case_info, inference_func, output_path, file_lock, skip_on_error):
    i, case, h = case_info
    try:
        pred_raw = inference_func(case["user_message"])
        error = None
    except Exception as e:
        if skip_on_error:
            pred_raw = {
                "content": None,
                "tool_calls": [],
                "usage": {}
            }
            error = str(e)
        else:
            raise

    record = {
        "index": case.get("index", i),
        "input_hash": h,
        "user_message": case["user_message"],
        "_source_sheet": case.get("_source_sheet", ""),
        "_source_file": case.get("_source_file", ""),
        "expected": case.get("tool_calls", []),
        "predicted": {
            "content": pred_raw.get("content", ""),
            "tool_calls": pred_raw.get("tool_calls", []),
            "error": error,
            "usage": pred_raw.get("usage", {})
        },
        "timestamp": datetime.now().isoformat()
    }

    atomic_append_line(output_path, record, file_lock)
    return True

def main():
    parser = argparse.ArgumentParser(description="Crash-safe, resumable inference (multithreaded)")
    parser.add_argument("--test_file", default="data/test_cases.json", help="Test cases JSON")
    parser.add_argument("--output", default=None, help="NDJSON output path")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--skip_on_error", action="store_true", help="Continue on inference error")
    parser.add_argument("--max_workers", type=int, default=32, help="Số luồng tối đa (default: 32)")
    args = parser.parse_args()

    data_name = args.test_file.split("/")[-1].split('.json')[0]

    if args.output is None:
        model_name = get_model_safe_name()
        args.output = f"results/{data_name}/{model_name}/predictions.ndjson"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load config
    base_url = os.getenv("BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("API_KEY", "EMPTY")
    model = os.getenv("MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

    test_cases = load_test_cases(args.test_file)
    tools = ALL_TOOLS

    done_hashes = load_completed_hashes(args.output)

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

    inference_func = make_inference_func(base_url, model, api_key, tools, args.system_prompt)

    # Lock để ghi file an toàn
    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_case,
                case_info,
                inference_func,
                args.output,
                file_lock,
                args.skip_on_error
            )
            for case_info in to_run
        ]

        # Progress bar
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Inferring"):
            pass

    print(f"✅ All done. Output: {args.output}")
    if os.path.exists(args.output):
        line_count = sum(1 for _ in open(args.output, "r", encoding="utf-8"))
        print(f" Total lines: {line_count}")

if __name__ == "__main__":
    main()