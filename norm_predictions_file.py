

import json
import hashlib
import time
import os
from datetime import datetime

import argparse
def generate_input_hash(user_message: str) -> str:
    """Generate a short 16-char hex hash from user_message (like your example)"""
    # You can adjust the method if you have the exact hashing logic used in your system
    m = hashlib.md5()
    m.update(user_message.encode('utf-8'))
    return m.hexdigest()[:16]


def convert_entry(entry: dict, new_index: int) -> dict:
    """
    Convert one item from your original format to the target format
    """
    user_msg = entry.get("user_message", "")

    # Extract predicted tool call if exists
    predicted = entry.get("predicted", {})
    tool_calls = predicted.get("tool_calls", [])

    converted_tool_calls = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name")
        args_str = func.get("arguments")

        if name and args_str:
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
                print(f"Warning: failed to parse arguments for {name}")

            converted_tool_calls.append({
                "name": name,
                "arguments": args
            })

    # Build predicted structure similar to your target example
    predicted_converted = {
        "content": None,
        "tool_calls": converted_tool_calls if converted_tool_calls else None,
        "error": None,
        "usage": entry.get("usage", {})
    }

    # Clean up None tool_calls
    if predicted_converted["tool_calls"] is None:
        predicted_converted["tool_calls"] = []

    result = {
        "index": new_index,
        "input_hash": generate_input_hash(user_msg),
        "user_message": user_msg,
        "_source_sheet": entry.get("_source_sheet"),
        "_source_file": entry.get("_source_file"),
        "expected": [],               # ← you didn't provide expected → left empty
        "predicted": predicted_converted,
        "timestamp": entry.get("timestamp"),
        "error": entry.get("error"),
    }

    return result

def convert_file(input_path: str, output_path: str, start_index: int = 1):
    """
    Read entries from a JSONL file (one JSON object per line),
    convert them, and write to another JSONL file.
    """
    # Create missing directories for output path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    converted_count = 0
    current_index = start_index

    # We open both files at once — read line by line, write line by line
    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout
    ):
        for line_number, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {line_number}: {e}")
                continue

            # Convert single entry
            new_entry = convert_entry(item, current_index)

            # Write as one JSON line
            json.dump(new_entry, fout, ensure_ascii=False)
            fout.write("\n")

            converted_count += 1
            current_index += 1

    print(f"Converted {converted_count} entries → saved to {output_path}")
# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert VinFast test report JSONL format to target evaluation format"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "-s", "--start",
        type=int,
        default=1,
        help="Starting index for the 'index' field (default: 1)"
    )

    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Start index: {args.start}\n")

    convert_file(
        input_path=args.input,
        output_path=args.output,
        start_index=args.start
    )


if __name__ == "__main__":
    main()