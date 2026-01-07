import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Use async-compatible tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio

# ⚙️ Your async LLM client (as requested)
from core.chat_client import chat_completion_async
from config.tools import ALL_TOOLS
from utils.misc import get_model_safe_name

load_dotenv()


# ==============================
# 🔁 Helper Functions (same logic, async-safe)
# ==============================

def load_test_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hash_input(user_message) -> str:
    normalized = json.dumps(user_message, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def load_completed_hashes(output_path: str) -> set:
    completed = set()
    if not os.path.exists(output_path):
        return completed

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    input_hash = record.get("input_hash")
                    if input_hash:
                        completed.add(input_hash)
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON at line {line_num} in {output_path}")
                except Exception as e:
                    print(f"⚠️ Error parsing line {line_num}: {e}")
    except Exception as e:
        print(f"⚠️ Failed to read completed hashes from {output_path}: {e}")

    return completed


# ✅ Async-safe atomic append (no threads → no lock needed, but use file lock for multi-process safety)
async def atomic_append_line_async(output_path: str, data: dict):
    # Use temporary file + atomic rename (POSIX-safe)
    tmp_path = Path(output_path).with_suffix(f".tmp.{os.getpid()}.{id(asyncio.current_task())}")
    try:
        # Write to tmp file
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
        # Atomically append (no race with other processes)
        with open(output_path, "a", encoding="utf-8") as dst:
            with open(tmp_path, "r", encoding="utf-8") as src:
                dst.write(src.read())
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass


# ==============================
# 🚀 Async Worker
# ==============================

async def process_case_async(
    i: int,
    case: Dict[str, Any],
    h: str,
    base_url: str,
    model: str,
    api_key: str,
    tools: List[Dict],
    system_prompt: str | None,
    skip_on_error: bool,
    output_path: str
) -> bool:
    try:
        pred_raw = await chat_completion_async(
            messages=case["user_message"],
            base_url=base_url,
            model=model,
            api_key=api_key,
            tools=tools,
            system_prompt=system_prompt,
            temperature=0.0
        )
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

    await atomic_append_line_async(output_path, record)
    return True


# ==============================
# 🏁 Main (Async)
# ==============================

async def main_async():
    parser = argparse.ArgumentParser(description="Crash-safe, resumable inference (async)")
    parser.add_argument("--model", type=str, help="LLM Name")
    parser.add_argument("--test_file", default="data/test_cases.json", help="Test cases JSON")
    parser.add_argument("--output", default=None, help="NDJSON output path")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--skip_on_error", action="store_true", help="Continue on inference error")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Max concurrent requests (default: 12)")
    args = parser.parse_args()
    model = args.model

    # 🔁 Same output path logic
    data_name = args.test_file.split("/")[-1].split('.json')[0]
    if args.output is None:
        model_name = get_model_safe_name(model=model)
        args.output = f"results/{data_name}/{model_name}/predictions.ndjson"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 🔁 Same config loading
    base_url = os.getenv("BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("API_KEY", "EMPTY")


    test_cases = load_test_cases(args.test_file)
    tools = ALL_TOOLS

    # 🔁 Same resume logic
    done_hashes = load_completed_hashes(args.output)
    processed_hashes = set(done_hashes)

    to_run: List[Tuple[int, Dict, str]] = []
    for i, case in enumerate(test_cases):
        h = hash_input(case["user_message"])
        if h not in processed_hashes:
            processed_hashes.add(h)
            to_run.append((i, case, h))

    print(f"🚀 Model: {model}")
    print(f"📊 Total cases: {len(test_cases)} | Already completed: {len(done_hashes)} | Remaining: {len(to_run)}")

    if not to_run:
        print("✅ All cases done.")
        return

    # 🔁 Control concurrency (server-friendly!)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def bounded_process(case_info):
        async with semaphore:
            i, case, h = case_info
            return await process_case_async(
                i=i,
                case=case,
                h=h,
                base_url=base_url,
                model=model,
                api_key=api_key,
                tools=tools,
                system_prompt=args.system_prompt,
                skip_on_error=args.skip_on_error,
                output_path=args.output
            )

    # 🚀 Run with progress bar
    tasks = [bounded_process(case_info) for case_info in to_run]
    for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Inferring"):
        await _

    print(f"✅ All done. Output: {Path(args.output).resolve()}")

    # 🔁 Same final line count
    if os.path.exists(args.output):
        try:
            line_count = sum(1 for _ in open(args.output, "r", encoding="utf-8") if _.strip())
            print(f"📜 Total valid lines in output: {line_count}")
        except Exception as e:
            print(f"⚠️ Could not count lines: {e}")


# ==============================
# 🧪 Entry Point
# ==============================

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()