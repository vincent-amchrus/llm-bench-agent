# utils/misc.py
import json
import hashlib

def hash_input(user_message) -> str:
    normalized = json.dumps(user_message, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    # utils/misc.py

import os

def get_model_safe_name(model) -> str:
    # Replace problematic chars: /, :, \, |, *, ?, ", <, >, space → hyphen or underscore
    safe = model.replace("/", "-") \
                .replace(":", "-") \
                .replace("\\", "-") \
                .replace("|", "-") \
                .replace("*", "_") \
                .replace("?", "_") \
                .replace("\"", "_") \
                .replace("<", "_") \
                .replace(">", "_") \
                .replace(" ", "_")
    return safe