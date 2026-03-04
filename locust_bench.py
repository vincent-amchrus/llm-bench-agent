from locust import HttpUser, task, constant, events
import json
import random
import copy
import string
from pathlib import Path
import os
import threading  # FIXED: Added missing import

# ================== CONFIG ==================

API_URL = "/v1/chat/completions"

# FIXED: Removed duplicates, keeping the latest values you specified
MODEL_NAME = "Qwen/Qwen3.5-4B"
BASE_URL = "http://localhost:8268"

TEST_FILE = "data/vivi_smart/_partial_90_vi_smart_labeled_0302.json"
TOOLS_FILE = "data/tools/vivi_smart_tools.json"

HEADERS = {
    "Content-Type": "application/json",
}

# ============================================

def load_json(path: str): # Changed type hint to str for flexibility
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load data once at startup
print(f"Loading test samples from: {TEST_FILE}")
TEST_SAMPLES = load_json(TEST_FILE)

print(f"Loading tools from: {TOOLS_FILE}")
TOOLS = load_json(TOOLS_FILE)

# Global state for sequential looping
_sample_index = 0
_index_lock = threading.Lock()
_total_samples = len(TEST_SAMPLES)

if _total_samples == 0:
    raise ValueError("TEST_SAMPLES list is empty. Cannot run sequential test.")

print(f"Loaded {_total_samples} test samples. Ready to start.")


def randomize_first_tool_name(tools: list) -> list:
    """
    Deep copy tools and random 2 characters into the first function name
    to break vLLM prefix caching.
    """
    tools = copy.deepcopy(tools)

    if not tools:
        return tools

    first_tool = tools[0]
    if first_tool.get("type") == "function":
        fn = first_tool.get("function", {})
        if "name" in fn:
            rand_char = random.choice(string.ascii_lowercase) + random.choice(string.ascii_uppercase)
            fn["name"] = f"{fn['name']}_{rand_char}"

    return tools


# def get_next_sequential_sample():
#     """
#     Thread-safe function to get the next sample in order.
#     Wraps around to the start when the end of the list is reached.
#     """
#     global _sample_index
    
#     with _index_lock:
#         current_idx = _sample_index
#         _sample_index += 1
        
#         if _sample_index >= _total_samples:
#             # _sample_index = 0
#             raise StopUser("All samples processed")  # Stop this virtual user
            
#     return TEST_SAMPLES[current_idx]

import sys
from locust import events

_all_samples_processed = False

def get_next_sequential_sample():
    global _sample_index, _all_samples_processed
    
    with _index_lock:
        if _all_samples_processed:
            events.quitting.fire(environment=None)  # Trigger shutdown
            sys.exit(0)
            
        current_idx = _sample_index
        _sample_index += 1
        
        if _sample_index >= _total_samples:
            _all_samples_processed = True  # Mark completion
            
    return TEST_SAMPLES[current_idx]

class ChatCompletionUser(HttpUser):
    wait_time = constant(0)
    host = BASE_URL

    @task
    def chat_completion_latency(self):
        sample = get_next_sequential_sample()
        
        # FIXED: Properly calling the randomization function. 
        # If you strictly want NO randomization, uncomment the line below and comment out the randomized_tools line.
#        randomized_tools = randomize_first_tool_name(TOOLS)
        randomized_tools = TOOLS

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": sample["user_message"]}
            ],
            "tools": randomized_tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
        }

        # Only add thinking kwargs if not using OpenAI official URL
        if "openai.com" not in BASE_URL:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        with self.client.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            name="chat_completions",
            catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text}")
                return

            try:
                data = resp.json()
                if "choices" not in data:
                    resp.failure("No choices in response")
            except Exception as e:
                resp.failure(f"Invalid JSON: {e}")