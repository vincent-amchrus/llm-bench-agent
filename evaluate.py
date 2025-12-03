# evaluate.py
import os
import json
import argparse
import logging
from dotenv import load_dotenv

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import project modules
from core.evaluator import evaluate_tool_calling_from_predictions
from utils.io import save_json, save_csv
from config.tools import ALL_TOOLS


def load_test_cases(path: str) -> list:
    """Load test cases from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: str) -> list:
    """Load predictions from .json or .ndjson"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    
    if path.endswith(".ndjson"):
        preds = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        preds.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skip invalid JSON on line {line_num} in {path}: {e}")
        return preds
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("results", data.get("predictions", []))
            else:
                raise ValueError(f"Unexpected data format in {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate precomputed tool-calling predictions")
    parser.add_argument("--test_file", default="data/test_cases.json", help="Test cases JSON")
    parser.add_argument("--predictions", default="results/predictions.ndjson", help="Precomputed predictions (NDJSON/JSON)")
    parser.add_argument("--output_dir", default="results", help="Directory to save eval results")
    parser.add_argument("--verbose", action="store_true", help="Print per-case results")
    args = parser.parse_args()

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    try:
        test_cases = load_test_cases(args.test_file)
        predictions = load_predictions(args.predictions)
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        exit(1)

    print(f"🧪 Loaded {len(test_cases)} test cases and {len(predictions)} predictions.")

    # Run evaluation (NO INFERENCE)
    try:
        eval_result = evaluate_tool_calling_from_predictions(
            test_cases=test_cases,
            predictions=predictions,
            tools=ALL_TOOLS,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

    # Save results
    try:
        json_path = save_json(eval_result, args.output_dir, "toolcall_eval")
        csv_path = save_csv(eval_result["results"], args.output_dir, "toolcall_eval")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        exit(1)

    # Print summary
    m = eval_result["metrics"]
    print("\n" + "="*60)
    print("🏆 EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Cases:              {m['total_cases']}")
    print(f"Exact Match Accuracy:     {m['exact_match_accuracy']:.2%}")
    print()
    print("📊 Tool Name Accuracy:")
    print(f"   Precision:             {m['tool_name_accuracy']['precision']:.2%}")
    print(f"   Recall:                {m['tool_name_accuracy']['recall']:.2%}")
    print(f"   F1:                    {m['tool_name_accuracy']['f1_score']:.2%}")
    print()
    print("📊 Argument Accuracy:")
    print(f"   Call-level (if name✓): {m['argument_accuracy_given_correct_name']:.2%}")
    field_info = m.get('argument_field_level', {})
    if field_info:
        print(f"   Field-level:           {field_info['accuracy']:.2%} ({field_info['total_matched_fields']}/{field_info['total_expected_fields']} fields)")
    print()
    print("📊 Strict Tool-call Level:")
    tc = m['tool_call_level']
    print(f"   TP={tc['true_positives']} FP={tc['false_positives']} FN={tc['false_negatives']}")
    print(f"   Precision:             {tc['precision']:.2%}")
    print(f"   Recall:                {tc['recall']:.2%}")
    print(f"   F1:                    {tc['f1_score']:.2%}")
    print()
    print(f"✅ Saved full report to: {os.path.relpath(json_path)}")
    print(f"   Per-case CSV:         {os.path.relpath(csv_path)}")


if __name__ == "__main__":
    main()