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
    global _sample_index
    with _lock:
        if _sample_index >= TOTAL_SAMPLES:
            raise StopUser()
        idx = _sample_index
        _sample_index += 1
    return TEST_SAMPLES[idx]


class ChatCompletionUser(HttpUser):
    wait_time = constant(0.15)          # ← very important: prevents endless stats printing
    host = BASE_URL

    @task
    def chat(self):
        sample = get_next_sample()

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