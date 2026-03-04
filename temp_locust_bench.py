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
    with open(p, encoding="utf-8") as f: return json.load(f)

TEST_SAMPLES = load_json(TEST_FILE)
TOOLS        = load_json(TOOLS_FILE)

TOTAL_SAMPLES = len(TEST_SAMPLES)
if TOTAL_SAMPLES == 0:
    print("No samples!", file=sys.stderr)
    sys.exit(1)

print(f"Loaded {TOTAL_SAMPLES} samples from {TEST_FILE}")

# Thread-safe sequential index
_sample_index = 0
_lock = threading.Lock()

def get_next_sample():
    """Returns (sample, is_last) or (None, False) if exhausted"""
    global _sample_index
    with _lock:
        if _sample_index >= TOTAL_SAMPLES:
            return None, False
        idx = _sample_index
        _sample_index += 1
        is_last = (_sample_index >= TOTAL_SAMPLES)
    return TEST_SAMPLES[idx], is_last


class ChatCompletionUser(HttpUser):
    wait_time = constant(0.15)
    host = BASE_URL

    @task
    def chat(self):
        sample, is_last = get_next_sample()
        
        # 🛑 No more samples → stop the entire test run
        if sample is None:
            if self.environment and self.environment.runner:
                print(f"✅ All {TOTAL_SAMPLES} samples processed. Stopping Locust runner...")
                self.environment.runner.quit()
            raise StopUser()

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
        
        # 🛑 If this was the last sample, explicitly quit (extra safety)
        if is_last and self.environment and self.environment.runner:
            print(f"✅ Last sample processed. Stopping Locust runner...")
            self.environment.runner.quit()