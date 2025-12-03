# core/evaluator.py
import os
import json
from typing import List, Dict, Any
from collections import defaultdict
from utils.misc import hash_input
from core.argument_matcher import ArgumentMatcher
import logging

logger = logging.getLogger(__name__)


def normalize_value(v):
    if isinstance(v, float):
        return round(v, 5)
    elif isinstance(v, dict):
        return {k: normalize_value(val) for k, val in sorted(v.items())}
    elif isinstance(v, list):
        return [normalize_value(x) for x in v]
    else:
        return v


def normalize_tool_call(tc: Dict) -> Dict:
    return {
        "name": tc["name"],
        "arguments": normalize_value(tc.get("arguments", {}))
    }


def evaluate_tool_calling_from_predictions(
    test_cases: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    tools: List[Dict],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate tool-calling using precomputed predictions — no live inference.
    """
    # Build lookup: index → prediction
    pred_by_index = {}
    pred_by_hash = {}
    for pred in predictions:
        if "index" in pred:
            pred_by_index[pred["index"]] = pred
        if "input_hash" in pred:
            pred_by_hash[pred["input_hash"]] = pred

    results = []
    correct_names = 0
    total_expected_calls = 0
    total_predicted_calls = 0
    correct_args_when_name_correct = 0
    total_name_correct_calls = 0
    tp = fp = fn = 0

    # Field-level counters
    total_expected_fields = 0
    total_matched_fields = 0

    matcher = ArgumentMatcher(
        tools_schema=tools,
        global_match_mode=os.getenv("MATCH_MODE", "semantic"),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
        device=os.getenv("DEVICE")
    )
    logger.info(f"Argument matcher config: {matcher.get_config()}")

    for i, case in enumerate(test_cases):
        user_msg = case["user_message"]
        expected_raw = case.get("tool_calls", case.get("expected", []))
        expected = [normalize_tool_call(tc) for tc in expected_raw]

        # Find matching prediction
        pred_raw = None
        if i in pred_by_index:
            pred_raw = pred_by_index[i]
        else:
            h = hash_input(user_msg)
            pred_raw = pred_by_hash.get(h)

        if pred_raw is None:
            logger.warning(f"No prediction found for case index={i}, msg='{user_msg[:50]}...'")
            predicted = []
            pred_raw = {"predicted": {"tool_calls": []}, "content": "", "error": "No prediction found"}
        else:
            pred_tool_calls = pred_raw.get("predicted", {}).get("tool_calls", [])
            predicted = [normalize_tool_call(tc) for tc in pred_tool_calls]

        # --- Name-level analysis ---
        exp_names = [tc["name"] for tc in expected]
        pred_names = [tc["name"] for tc in predicted]
        total_expected_calls += len(exp_names)
        total_predicted_calls += len(pred_names)

        exp_name_counter = defaultdict(int)
        pred_name_counter = defaultdict(int)
        for n in exp_names:
            exp_name_counter[n] += 1
        for n in pred_names:
            pred_name_counter[n] += 1

        name_match_count = 0
        all_names = set(exp_name_counter.keys()) | set(pred_name_counter.keys())
        for name in all_names:
            e, p = exp_name_counter[name], pred_name_counter[name]
            name_match_count += min(e, p)
        correct_names += name_match_count

        # --- Argument-level (only where name matches) ---
        exp_by_name = defaultdict(list)
        pred_by_name = defaultdict(list)

        for tc in expected:
            exp_by_name[tc["name"]].append(tc["arguments"])
        for tc in predicted:
            pred_by_name[tc["name"]].append(tc["arguments"])

        # --- Field-level argument matching ---
        case_field_sim_details = []  # Collect all field similarities for this case
        matched_correct_args = 0

        for name in set(exp_by_name.keys()) & set(pred_by_name.keys()):  # Only matching names
            exp_args_list = exp_by_name[name]
            pred_args_list = pred_by_name[name]
            min_len = min(len(exp_args_list), len(pred_args_list))

            for j in range(min_len):
                exp_args = exp_args_list[j]
                pred_args = pred_args_list[j]

                # Get detailed field match info
                detailed_match = matcher.match_arguments(exp_args, pred_args, name, return_scores=True)
                case_field_sim_details.append({
                    "tool_name": name,
                    "call_index": j,
                    "expected_args": exp_args,
                    "predicted_args": pred_args,
                    "overall_match": detailed_match["overall_match"],
                    "field_scores": detailed_match["field_scores"]
                })

                # Count field-level matches
                for field, field_info in detailed_match["field_scores"].items():
                    # Only count expected fields (not extra fields in prediction)
                    if field in exp_args:
                        total_expected_fields += 1
                        if field_info.get("match", False):
                            total_matched_fields += 1

                # Call-level match (all-or-nothing)
                total_name_correct_calls += 1
                if detailed_match["overall_match"]:
                    matched_correct_args += 1

        # Also count expected fields from tools that weren't predicted at all
        for name in set(exp_by_name.keys()) - set(pred_by_name.keys()):
            for exp_args in exp_by_name[name]:
                total_expected_fields += len(exp_args)
                # These are all misses (0 matched)
                case_field_sim_details.append({
                    "tool_name": name,
                    "call_index": 0,
                    "expected_args": exp_args,
                    "predicted_args": None,
                    "overall_match": False,
                    "field_scores": {
                        field: {"match": False, "reason": "tool not predicted"}
                        for field in exp_args
                    }
                })

        correct_args_when_name_correct += matched_correct_args

        # --- Strict full-match (for TP/FP/FN) ---
        pred_counter = defaultdict(int)
        exp_counter = defaultdict(int)
        for tc in predicted:
            key = (tc["name"], json.dumps(tc["arguments"], sort_keys=True))
            pred_counter[key] += 1
        for tc in expected:
            key = (tc["name"], json.dumps(tc["arguments"], sort_keys=True))
            exp_counter[key] += 1

        all_keys = set(pred_counter.keys()) | set(exp_counter.keys())
        for key in all_keys:
            p, e = pred_counter[key], exp_counter[key]
            tp += min(p, e)
            fp += max(0, p - e)
            fn += max(0, e - p)

        # Compute case-level match using ArgumentMatcher
        case_match = True
        if len(expected) != len(predicted):
            case_match = False
        else:
            for exp_tc, pred_tc in zip(expected, predicted):
                if exp_tc["name"] != pred_tc["name"]:
                    case_match = False
                    break
                args_match = matcher.match_arguments(
                    exp_tc["arguments"],
                    pred_tc["arguments"],
                    tool_name=exp_tc["name"],
                    return_scores=False
                )
                if not args_match:
                    case_match = False
                    break

        res = {
            "index": i,
            "user_message": user_msg,
            "expected": expected,
            "predicted": predicted,
            "match": case_match,
            "content": pred_raw.get("content", pred_raw.get("predicted", {}).get("content", "")),
            "error": pred_raw.get("error", pred_raw.get("predicted", {}).get("error")),
            "field_similarities": case_field_sim_details
        }
        results.append(res)

        if verbose:
            status = "✅" if case_match else "❌"
            print(f"[{status}] Case {i}: '{user_msg[:50]}...'")
            if not case_match:
                print(f"  Expected: {expected}")
                print(f"  Got:      {predicted}")
            # Print sim scores if available
            for fs in case_field_sim_details:
                for field, info in fs.get("field_scores", {}).items():
                    sim = info.get("similarity")
                    match_status = '✓' if info.get('match', False) else '✗'
                    mode = info.get('mode', 'unknown')
                    if sim is not None:
                        print(f"    [{field}] sim={sim:.4f} (mode={mode}) → {match_status}")
                    else:
                        reason = info.get('reason', '')
                        print(f"    [{field}] (mode={mode}) → {match_status} {reason}")

    # ==== METRICS ====
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    exact_match_acc = sum(1 for r in results if r["match"]) / len(results) if results else 0.0

    name_precision = correct_names / total_predicted_calls if total_predicted_calls > 0 else 0.0
    name_recall = correct_names / total_expected_calls if total_expected_calls > 0 else 0.0
    name_f1 = 2 * name_precision * name_recall / (name_precision + name_recall) if (name_precision + name_recall) > 0 else 0.0

    arg_accuracy = correct_args_when_name_correct / total_name_correct_calls if total_name_correct_calls > 0 else 0.0
    field_arg_accuracy = total_matched_fields / total_expected_fields if total_expected_fields > 0 else 0.0

    return {
        "metrics": {
            "total_cases": len(test_cases),
            "exact_match_accuracy": round(exact_match_acc, 4),
            "tool_call_level": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            },
            "tool_name_accuracy": {
                "correct_name_calls": correct_names,
                "total_expected_calls": total_expected_calls,
                "total_predicted_calls": total_predicted_calls,
                "precision": round(name_precision, 4),
                "recall": round(name_recall, 4),
                "f1_score": round(name_f1, 4)
            },
            "argument_accuracy_given_correct_name": round(arg_accuracy, 4),
            "argument_field_level": {
                "total_expected_fields": total_expected_fields,
                "total_matched_fields": total_matched_fields,
                "accuracy": round(field_arg_accuracy, 4)
            }
        },
        "results": results
    }