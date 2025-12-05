from datetime import datetime
import os
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
from utils.misc import get_model_safe_name

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
def format_summary(metrics: dict) -> str:
    m = metrics
    tc = m['tool_call_level']
    name_acc = m['tool_name_accuracy']
    field_info = m.get('argument_field_level', {})

    lines = []
    lines.append("=" * 60)
    lines.append("🏆 TOOL-CALLING EVALUATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model:     {os.getenv('MODEL', 'unknown')}")
    lines.append(f"Test File: {os.getenv('TEST_FILE', 'unknown')}")
    lines.append("-" * 60)
    lines.append(f"Total Cases:              {m['total_cases']}")
    lines.append(f"Exact Match Accuracy:     {m['exact_match_accuracy']:.2%}")
    lines.append("")
    lines.append("📊 Tool Name Accuracy:")
    lines.append(f"   Precision:             {name_acc['precision']:.2%}")
    lines.append(f"   Recall:                {name_acc['recall']:.2%}")
    lines.append(f"   F1:                    {name_acc['f1_score']:.2%}")
    lines.append("")
    lines.append("📊 Argument Accuracy:")
    lines.append(f"   Call-level (if name✓): {m['argument_accuracy_given_correct_name']:.2%}")
    if field_info:
        lines.append(f"   Field-level:           {field_info['accuracy']:.2%} "
                     f"({field_info['total_matched_fields']}/{field_info['total_expected_fields']} fields)")
    lines.append("")
    lines.append("📊 Strict Tool-call Level (exact match):")
    lines.append(f"   TP={tc['true_positives']} FP={tc['false_positives']} FN={tc['false_negatives']}")
    lines.append(f"   Precision:             {tc['precision']:.2%}")
    lines.append(f"   Recall:                {tc['recall']:.2%}")
    lines.append(f"   F1:                    {tc['f1_score']:.2%}")
    lines.append("")
    lines.append("✅ Full results saved as JSON & CSV in this directory.")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Evaluate precomputed tool-calling predictions")
    parser.add_argument("--test_file", default="data/test_cases.json", help="Test cases JSON")
    parser.add_argument("--predictions", default=None, help="Precomputed predictions (default: results/<MODEL>/predictions.ndjson)")
    parser.add_argument("--output_dir", default=None, help="Dir to save eval (default: results/<MODEL>/)")
    parser.add_argument("--verbose", action="store_true", help="Print per-case results")
    args = parser.parse_args()

    # Resolve model-based defaults
    model_name = get_model_safe_name()
    default_result_dir = f"results/{model_name}"

    # Set predictions path
    if args.predictions is None:
        args.predictions = f"{default_result_dir}/predictions.ndjson"

    # Set output_dir
    if args.output_dir is None:
        args.output_dir = default_result_dir

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

    # Print summary
    m = eval_result["metrics"]
    # Save results
    try:
        json_path = save_json(eval_result, args.output_dir, "toolcall_eval")
        csv_path = save_csv(eval_result["results"], args.output_dir, "toolcall_eval")
        # 👇 Add this:
        summary_text = format_summary(m)
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        exit(1)

    # Print summary (unchanged)
    # ... [your existing print block]
    print(f"✅ Saved full report to: {os.path.relpath(json_path)}")
    print(f"   Per-case CSV:         {os.path.relpath(csv_path)}")
    print(f"   Summary text:         {os.path.relpath(summary_path)}")  # 👈 add this line

if __name__ == "__main__":
    main()