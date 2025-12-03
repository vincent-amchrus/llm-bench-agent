# utils/misc.py
import json
import hashlib

def hash_input(user_message) -> str:
    normalized = json.dumps(user_message, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]