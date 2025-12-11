#!/usr/bin/env python3
# exact_metrics.py — Strict exact-match: all tool calls (name + full args), order-sensitive
import argparse
import pandas as pd
import json
import os

def normalize_value(v):
    """Match evaluator.py's normalize_value"""
    if isinstance(v, float):
        return round(v, 5)
    elif isinstance(v, dict):
        return {k: normalize_value(val) for k, val in sorted(v.items())}
    elif isinstance(v, list):
        return [normalize_value(x) for x in v]
    else:
        return v

def normalize_tool_call(tc):
    """Same as in evaluator.py"""
    return {
        "name": tc["name"],
        "arguments": normalize_value(tc.get("arguments", {}))
    }

def is_exact_match(expected_raw, predicted_raw):
    """Strict exact match: same number of calls, same name & args (order matters)"""
    expected = [normalize_tool_call(tc) for tc in expected_raw] if expected_raw else []
    predicted_tool_calls = (predicted_raw or {}).get("tool_calls", [])
    predicted = [normalize_tool_call(tc) for tc in predicted_tool_calls] if predicted_tool_calls else []

    if len(expected) != len(predicted):
        return False

    for exp, pred in zip(expected, predicted):
        if exp["name"] != pred["name"]:
            return False
        if exp["arguments"] != pred["arguments"]:
            return False
    return True

def compute_function_name_match(expected_raw, predicted_raw):
    """Function name match (per call, order-sensitive — first call only if mismatched count)"""
    expected = [tc.get("name") for tc in expected_raw] if expected_raw else []
    predicted_tool_calls = (predicted_raw or {}).get("tool_calls", [])
    predicted = [tc.get("name") for tc in predicted_tool_calls] if predicted_tool_calls else []

    if not expected and not predicted:
        return 1  # vacuous true
    if not expected or not predicted:
        return 0

    # Compare up to min length
    min_len = min(len(expected), len(predicted))
    matches = sum(1 for i in range(min_len) if expected[i] == predicted[i])
    # Total possible = len(expected)
    return matches / len(expected) if expected else 1.0

def main():
    parser = argparse.ArgumentParser(description="Compute exact-match metrics (name + args, multi-call safe)")
    parser.add_argument("predictions_ndjson", help="Path to predictions.ndjson")
    parser.add_argument("--output", "-o", help="Save per-case CSV (default: <input>_exact_metrics.csv)")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_ndjson):
        print(f"❌ File not found: {args.predictions_ndjson}")
        exit(1)

    df = pd.read_json(args.predictions_ndjson, lines=True)
    print(f"📊 Loaded {len(df)} predictions.")

    # Compute metrics
    df["exact_match"] = df.apply(
        lambda r: is_exact_match(r.get("expected", []), r.get("predicted", {})),
        axis=1
    )
    df["fn_match_ratio"] = df.apply(
        lambda r: compute_function_name_match(r.get("expected", []), r.get("predicted", {})),
        axis=1
    )

    exact_acc = df["exact_match"].mean()
    fn_acc = df["fn_match_ratio"].mean()

    print("\n" + "=" * 50)
    print("✅ STRICT EXACT-MATCH METRICS")
    print("=" * 50)
    print(f"Function Name (per-call ratio): {fn_acc:.2%}")
    print(f"Exact Match (name + all args):   {exact_acc:.2%}")
    print(f"Samples: {len(df)}")
    print("=" * 50)

    # Save
    out_path = args.output or args.predictions_ndjson.replace(".ndjson", "_exact_metrics.csv")
    df[["index", "fn_match_ratio", "exact_match"]].to_csv(out_path, index=False)
    print(f"💾 Saved per-case metrics to: {out_path}")

if __name__ == "__main__":
    main()