# temp_locust_bench.py
from locust import HttpUser, task, constant
from locust.exception import StopUser
import json
import threading
import argparse
import sys

# ================== Argument parsing ==================
parser = argparse.ArgumentParser()
parser.add_argument("--test-file",    type=str, required=True)
parser.add_argument("--tools-file",   type=str, required=True)
parser.add_argument("--base-url",     type=str, required=True)
parser.add_argument("--model",        type=str, required=True)
parser.add_argument("--reasoning",    type=str, required=True)

args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

# ================== CONFIG ==================
API_URL = "/v1/chat/completions"
MODEL_NAME   = args.model
BASE_URL     = args.base_url
TEST_FILE    = args.test_file
TOOLS_FILE   = args.tools_file
REASONING    = args.reasoning

HEADERS = {"Content-Type": "application/json"}

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

# ================== Thread-safe state ==================
# Sample assignment
_sample_index = 0
_sample_lock = threading.Lock()

# Completion tracking
_completed_count = 0
_completion_lock = threading.Lock()
_all_done = threading.Event()  # Signals when all requests are fully completed

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

        if REASONING == "no-thinking":
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

        # Execute request
        with self.client.post(
            API_URL,
            json=payload,
            headers=HEADERS,
            name="chat/completions",
            catch_response=True
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif "choices" not in r.json():
                r.failure("No choices in response")
            # else: success (no explicit success() needed with catch_response=True)
        
        # ✅ Mark this request as fully completed
        mark_completed(self)
        
        # Final check: if we just completed the last one, exit cleanly
        if _all_done.is_set():
            raise StopUser()