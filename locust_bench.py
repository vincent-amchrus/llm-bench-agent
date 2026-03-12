# temp_locust_bench.py
from datetime import datetime

from locust import HttpUser, task, constant
from locust.exception import StopUser
import json
import threading
import argparse
import sys
import os
from pathlib import Path

# ================== Argument parsing ==================
parser = argparse.ArgumentParser()
parser.add_argument("--test-file",    type=str, required=True)
parser.add_argument("--tools-file",   type=str, required=True)
parser.add_argument("--result-dir",   type=str, required=True)
parser.add_argument("--base-url",     type=str, required=True)
parser.add_argument("--model",        type=str, required=True)
parser.add_argument("--api_key",        type=str, required=True)
parser.add_argument("--reasoning",    type=str, required=True)

args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

# ================== CONFIG ==================
API_URL = "/v1/chat/completions"
MODEL_NAME   = args.model
BASE_URL     = args.base_url
API_KEY      = args.api_key
TEST_FILE    = args.test_file
RESULT_DIR   = args.result_dir
TOOLS_FILE   = args.tools_file
REASONING    = args.reasoning

# Output predictions to same folder as test file
PREDICTIONS_FILE = os.path.join(RESULT_DIR, "raw_predictions.ndjson")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Load data once
def load_json(p): 
    with open(p, encoding="utf-8") as f: 
        return json.load(f)

TEST_SAMPLES = load_json(TEST_FILE)
TOOLS        = load_json(TOOLS_FILE)

TOTAL_SAMPLES = len(TEST_SAMPLES)
if TOTAL_SAMPLES == 0:
    print("No samples!", file=sys.stderr)
    sys.exit(1)

print(f"Loaded {TOTAL_SAMPLES} samples from {TEST_FILE}")
print(f"Predictions will be saved to: {PREDICTIONS_FILE}")

# ================== Thread-safe state ==================
# Sample assignment
_sample_index = 0
_sample_lock = threading.Lock()

# Completion tracking
_completed_count = 0
_completion_lock = threading.Lock()
_all_done = threading.Event()  # Signals when all requests are fully completed

# File writing lock
_predictions_lock = threading.Lock()

def get_next_sample():
    """Atomically assign the next sample index. Returns None if exhausted."""
    global _sample_index
    with _sample_lock:
        if _sample_index >= TOTAL_SAMPLES:
            return None
        idx = _sample_index
        _sample_index += 1
    return TEST_SAMPLES[idx]

def mark_completed(user_instance):
    """Mark a request as completed. Trigger shutdown when all are done."""
    global _completed_count
    with _completion_lock:
        _completed_count += 1
        just_finished_all = (_completed_count >= TOTAL_SAMPLES)
    
    if just_finished_all:
        print(f"✅ All {TOTAL_SAMPLES} requests COMPLETED. Stopping runner...")
        _all_done.set()
        # Safely trigger shutdown via the user's environment
        if user_instance.environment and user_instance.environment.runner:
            user_instance.environment.runner.quit()

# ✅ Thread-safe atomic append (for multi-threaded Locust)
def atomic_append_line(output_path: str, data: dict):
    """Atomically append a JSON line to the output file."""
    tmp_path = Path(output_path).with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        # Write to tmp file
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
        # Atomically append with lock
        with _predictions_lock:
            with open(output_path, "a", encoding="utf-8") as dst:
                with open(tmp_path, "r", encoding="utf-8") as src:
                    dst.write(src.read())
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass

# ================== Locust User ==================
class ChatCompletionUser(HttpUser):
    wait_time = constant(0.15)  # Small delay to avoid hammering the server
    host = BASE_URL
    stop_timeout = 60  # Wait up to 60s for in-flight requests to finish on shutdown

    @task
    def chat(self):
        # Early exit if all requests are already completed
        if _all_done.is_set():
            raise StopUser()
        
        # Get next sample to process
        sample = get_next_sample()
        if sample is None:
            # No more samples to assign → exit this user
            raise StopUser()

        # Build payload
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": sample["user_message"]}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
        }

        if REASONING == "no-thinking" and "cerebras" not in BASE_URL:
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

        # Execute request
        response_json = None
        with self.client.post(
            API_URL,
            json=payload,
            headers=HEADERS,
            name="chat/completions",
            catch_response=True
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                response_data = {"error": f"HTTP {r.status_code}", "response_text": r.text}
            elif "choices" not in r.json():
                r.failure("No choices in response")
                response_data = {"error": "No choices in response", "response_text": r.text}
            else:
                response_json = r.json()
                response_data = {
                    "predicted": response_json['choices'][0]['message'],
                    "usage": response_json.get("usage", {}),
                }
        
        # ✅ Save prediction to file
        prediction_record = {
            "sample_index": _sample_index - 1,  # The index we just processed
            "user_message": sample["user_message"],
            "_source_sheet": sample.get("_source_sheet", ""),
            "_source_file": sample.get("_source_file", ""),
            "expected": sample.get("tool_calls"),
            "model": MODEL_NAME,
            "reasoning": REASONING,
            "timestamp": datetime.now().isoformat(),
            
        }
        prediction_record.update(response_data)  # Add either the prediction or the error info
        
        # Add metadata if available
        if response_json and "choices" in response_json:
            prediction_record["usage"] = response_json.get("usage", {})
            prediction_record["finish_reason"] = response_json["choices"][0].get("finish_reason") if response_json["choices"] else None
        
        atomic_append_line(PREDICTIONS_FILE, prediction_record)
        
        # ✅ Mark this request as fully completed
        mark_completed(self)
        
        # Final check: if we just completed the last one, exit cleanly
        if _all_done.is_set():
            raise StopUser()