import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter

tqdm.pandas()

# --- ARGUMENT EVALUATION LOGIC ---

def compare_predictions(ground_truth, predictions, exist_only_args=None):
    if not isinstance(ground_truth, list): ground_truth = []
    if not isinstance(predictions, list): predictions = []
    
    exist_only_args = set(exist_only_args) if exist_only_args else set()
    gt_dict = {item["name"]: item for item in ground_truth}
    pred_dict = {item["name"]: item for item in predictions}

    total_gt = len(ground_truth)
    correct_count = 0
    details = []
    all_names = set(gt_dict.keys()) | set(pred_dict.keys())

    for name in all_names:
        if name not in gt_dict:
            details.append({"name": name, "status": "extra", "errors": ["unexpected name"]})
            continue
        if name not in pred_dict:
            details.append({"name": name, "status": "missing", "errors": ["prediction missing"]})
            continue

        gt_args = gt_dict[name].get("arguments", {})
        pred_args = pred_dict[name].get("arguments", {})
        errors = []

        for key in gt_args:
            if key not in pred_args:
                errors.append(f"lack key: {key}")
            elif key not in exist_only_args:
                if gt_args[key] != pred_args[key]:
                    errors.append(f"wrong value: {key}")

        for key in pred_args:
            if key not in gt_args:
                errors.append(f"redundant key: {key}")

        status = "mismatch" if errors else "correct"
        details.append({"name": name, "status": status, "errors": errors})
        if status == "correct":
            correct_count += 1
            
    has_extra_name = any(d["status"] == "extra" for d in details)
    
    if total_gt == 0:
        accuracy = 1.0 if len(predictions) == 0 else 0.0
    else:
        accuracy = 0.0 if has_extra_name else correct_count / total_gt

    return {"accuracy": accuracy, "details": details}

def compute_tool_statistics(all_details_series):
    tool_stats = defaultdict(lambda: {'total': 0, 'status_counts': Counter(), 'error_counts': Counter()})
    for details in all_details_series:
        for item in details:
            name, status = item['name'], item['status']
            tool_stats[name]['total'] += 1
            tool_stats[name]['status_counts'][status] += 1
            for error in item.get('errors', []):
                tool_stats[name]['error_counts'][error] += 1
    return tool_stats

# --- HELPER FUNCTIONS ---

def parse_tool_call_name(tool_calls):
    if isinstance(tool_calls, dict):
        tool_calls = tool_calls.get('tool_calls', [])
    if not isinstance(tool_calls, list) or len(tool_calls) == 0:
        return "__NO_CALL__"
    return tool_calls[0].get('name', "__NO_CALL__")

def parse_tool_call_list(tool_calls):
    if isinstance(tool_calls, dict):
        return tool_calls.get('tool_calls', [])
    return tool_calls if isinstance(tool_calls, list) else []

# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()

    infer_path = Path(args.pred_path)
    output_dir = infer_path.parent
    df = pd.read_json(infer_path, lines=True)

    # 1. Basic Tool Classification
    df['pred_tool_name'] = df['predicted'].apply(parse_tool_call_name)
    df['gt_tool_name'] = df['expected'].apply(parse_tool_call_name)
    df = df[df['gt_tool_name'] != '__NO_CALL__'].reset_index(drop=True)

    # 2. Argument Accuracy
    df['pred_list'] = df['predicted'].apply(parse_tool_call_list)
    df['gt_list'] = df['expected'].apply(parse_tool_call_list)
    
    arg_evals = df.apply(lambda row: compare_predictions(
        row['gt_list'], row['pred_list'], 
        exist_only_args=["key_word", "rewrite_message", "queries"]
        # exist_only_args=[]
    ), axis=1)
    
    df['arg_accuracy'] = arg_evals.apply(lambda x: x['accuracy'])
    df['arg_details'] = arg_evals.apply(lambda x: x['details'])
    
    avg_arg_acc = df['arg_accuracy'].mean() * 100
    tool_stats = compute_tool_statistics(df['arg_details'])

    # 3. Visualization (Confusion Matrix)
    labels = sorted(set(df['gt_tool_name']) | set(df['pred_tool_name']))
    cm_norm = confusion_matrix(df['gt_tool_name'], df['pred_tool_name'], labels=labels, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(cm_norm * 100, index=labels, columns=labels), annot=True, fmt='.1f', cmap='Blues')
    plt.title('Tool Selection Accuracy (%)')
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # 4. Build Markdown Content
    md_lines = [
        "# Tool-Call & Argument Evaluation Report",
        f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total Samples**: {len(df)}",
        "",
        "## 📊 Executive Summary",
        f"| Metric | Score |",
        f"| :--- | :--- |",
        f"| **Overall Tool Selection Accuracy** | `{accuracy_score(df['gt_tool_name'], df['pred_tool_name'])*100:.2f}%` |",
        f"| **Overall Argument Accuracy** | `{avg_arg_acc:.2f}%` |",
        "",
        "## 🛠️ Per-Tool Argument Statistics"
    ]

    for tool, data in tool_stats.items():
        total = data['total']
        md_lines.append(f"### `{tool}`")
        md_lines.append(f"- **Occurrences**: {total}\n")
        
        # Status Table
        md_lines.append("| Status | Count | Percentage |")
        md_lines.append("| :--- | :--- | :--- |")
        for status, count in data['status_counts'].items():
            md_lines.append(f"| {status} | {count} | {(count/total)*100:.1f}% |")
        
        # Top Errors
        if data['error_counts']:
            md_lines.append("\n**Top Argument Errors:**")
            for err, count in data['error_counts'].most_common(3):
                md_lines.append(f"\n- {err} ({count} hits)")
        md_lines.append("\n---")

    md_lines.append("## 📈 Tool Selection Heatmap")
    md_lines.append("![Confusion Matrix](confusion_matrix.png)")

    # 5. Export to PDF
    try:
        import markdown as md_lib
        from weasyprint import HTML
        
        html_body = md_lib.markdown("\n".join(md_lines), extensions=['tables', 'fenced_code'])
        styled_html = f"""
        <html>
        <head><style>
            body {{ font-family: sans-serif; margin: 40px; line-height: 1.5; color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; }}
            h3 {{ color: #2980b9; margin-top: 20px; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 20px 0; }}
        </style></head>
        <body>{html_body}</body>
        </html>
        """
        
        pdf_path = output_dir / "evaluation_report.pdf"
        HTML(string=styled_html, base_url=str(output_dir)).write_pdf(str(pdf_path))
        print(f"✅ PDF Report generated: {pdf_path}")
    except ImportError:
        print("⚠️ Install 'weasyprint' and 'markdown' for PDF support.")

    # Save JSON for programmatic access
    df[['gt_tool_name', 'pred_tool_name', 'arg_accuracy']].to_json(output_dir / "metrics_raw.json", orient='records')

if __name__ == "__main__":
    main()