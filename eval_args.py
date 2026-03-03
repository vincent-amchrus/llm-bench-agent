import statistics
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter

tqdm.pandas()

# --- ARGUMENT EVALUATION LOGIC ---

def compare_predictions(ground_truth, predictions, exist_only_args=None, user_message=None):
    if not isinstance(ground_truth, list): ground_truth = []
    if not isinstance(predictions, list): predictions = []
    
    exist_only_args = set(exist_only_args) if exist_only_args else set()
    gt_dict = {item.get("name"): item for item in ground_truth}
    pred_dict = {item.get("name"): item for item in predictions}

    total_gt = len(ground_truth)
    correct_count = 0
    details = []
    all_names = set(gt_dict.keys()) | set(pred_dict.keys())

    for name in all_names:
        # Capture raw dicts for reporting examples
        gt_raw = gt_dict.get(name, {}).get("arguments", {})
        pred_raw = pred_dict.get(name, {}).get("arguments", {})

        if name not in gt_dict:
            details.append({
                "name": name, 
                "status": "extra", 
                "errors": ["unexpected name"],
                "example": {"user_message": user_message, "gt": None, "pred": pred_raw}
            })
            continue
        if name not in pred_dict:
            details.append({
                "name": name, 
                "status": "missing", 
                "errors": ["prediction missing"],
                "example": {"user_message": user_message, "gt": gt_raw, "pred": None}
            })
            continue

        gt_args = gt_dict[name].get("arguments", {})
        pred_args = pred_dict[name].get("arguments", {})
        errors = []

        for key in gt_args:
            if key not in pred_args:
                errors.append(f"lack key: {key}")
            elif key not in exist_only_args:
                # Convert to string to compare values safely
                if str(gt_args[key]) != str(pred_args[key]):
                    errors.append(f"wrong value: {key}")

        for key in pred_args:
            if key not in gt_args:
                errors.append(f"redundant key: {key}")

        status = "mismatch" if errors else "correct"
        details.append({
            "name": name, 
            "status": status, 
            "errors": errors,
            "example": {"user_message": user_message, "gt": gt_args, "pred": pred_args} # Capture example for report
        })
        if status == "correct":
            correct_count += 1
            
    has_extra_name = any(d["status"] == "extra" for d in details)
    
    if total_gt == 0:
        accuracy = 1.0 if len(predictions) == 0 else 0.0
    else:
        accuracy = 0.0 if has_extra_name else correct_count / total_gt

    return {"accuracy": accuracy, "details": details}

def compute_tool_statistics(all_details_series):
    # Stats container: { tool_name: { total, status_counts, error_counts, examples } }
    tool_stats = defaultdict(lambda: {
        'total': 0, 
        'status_counts': Counter(), 
        'error_counts': Counter(),
        'error_examples': defaultdict(list)
    })

    for details in all_details_series:
        for item in details:
            name, status = item['name'], item['status']
            tool_stats[name]['total'] += 1
            tool_stats[name]['status_counts'][status] += 1
            
            if 'errors' in item and item['errors']:
                for error in item['errors']:
                    tool_stats[name]['error_counts'][error] += 1
                    # Store up to 5 examples per error type per tool
                    if len(tool_stats[name]['error_examples'][error]) < 5:
                        tool_stats[name]['error_examples'][error].append(item['example'])
                        
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
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--reasoning", type=str, default="")
    parser.add_argument("--ccu", type=int, default=0)
    args = parser.parse_args()

    infer_path = Path(args.pred_path)
    output_dir = infer_path.parent
    df = pd.read_json(infer_path, lines=True)

    # ==========================================
    # 1. TOOL SELECTION EVALUATION (Full Dataset)
    # ==========================================
    df['pred_tool_name'] = df['predicted'].apply(parse_tool_call_name)
    df['gt_tool_name'] = df['expected'].apply(parse_tool_call_name)

    # A. Overall Tool Selection (All samples)
    overall_tool_acc = accuracy_score(df['gt_tool_name'], df['pred_tool_name']) * 100

    # B. Tool-Only Accuracy (Subset where GT is NOT No-Call)
    tool_mask = df['gt_tool_name'] != '__NO_CALL__'
    if tool_mask.sum() > 0:
        tool_only_acc = accuracy_score(
            df[tool_mask]['gt_tool_name'], 
            df[tool_mask]['pred_tool_name']
        ) * 100
    else:
        tool_only_acc = 0.0

    # C. No-Call Accuracy (Recall: Subset where GT IS No-Call)
    no_call_mask = df['gt_tool_name'] == '__NO_CALL__'
    if no_call_mask.sum() > 0:
        # Accuracy on this subset is simply how many were predicted as NO_CALL
        no_call_acc = (df[no_call_mask]['pred_tool_name'] == '__NO_CALL__').mean() * 100
    else:
        no_call_acc = 0.0

    # ==========================================
    # 2. ARGUMENT EVALUATION (Filtered Dataset)
    # ==========================================
    
    # Filter: Only evaluate arguments for rows where a tool was actually expected
    df_args = df[df['gt_tool_name'] != '__NO_CALL__'].copy().reset_index(drop=True)

    df_args['pred_list'] = df_args['predicted'].apply(parse_tool_call_list)
    df_args['gt_list'] = df_args['expected'].apply(parse_tool_call_list)
    
    # Run comparison
    arg_evals = df_args.apply(lambda row: compare_predictions(
        row['gt_list'], row['pred_list'], 
        exist_only_args=["key_word", "rewrite_message", "queries"],
        user_message=row['user_message']
    ), axis=1)
    
    df_args['arg_accuracy'] = arg_evals.apply(lambda x: x['accuracy'])
    df_args['arg_details'] = arg_evals.apply(lambda x: x['details'])
    
    # Metrics based on filtered data
    avg_arg_acc = df_args['arg_accuracy'].mean() * 100
    tool_stats = compute_tool_statistics(
        all_details_series=df_args['arg_details']
    )

    # ==========================================
    # 3. REPORTING
    # ==========================================

    # Visualization (Full Dataset Confusion Matrix)
    labels = sorted(set(df['gt_tool_name']) | set(df['pred_tool_name']))
    cm_norm = confusion_matrix(df['gt_tool_name'], df['pred_tool_name'], labels=labels, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(cm_norm * 100, index=labels, columns=labels), annot=True, fmt='.1f', cmap='Blues')
    plt.title('Tool Selection Accuracy (%)')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # Markdown Content
    md_lines = [
        "# Tool-Call & Argument Evaluation Report",
        f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"**Total Samples**: {len(df)}\n",
        f"**Tool Samples (for Args)**: {len(df_args)}\n",
        f"**Model**: {args.model}\n",
        f"**Reasoning**: {args.reasoning}\n",
        f"**CCU**: {args.ccu}\n",
        "",
        "## 📊 Executive Summary",
        "| Metric | Score | Scope |",
        "| :--- | :--- | :--- |",
        f"| **Overall Tool Selection** | `{overall_tool_acc:.2f}%` | All samples |",
        f"| **Tool-Only Accuracy** | `{tool_only_acc:.2f}%` | Only where GT is a tool |",
        f"| **No-Call Accuracy** | `{no_call_acc:.2f}%` | Only where GT is No-Call |",
        f"| **Argument Accuracy** | `{avg_arg_acc:.2f}%` | Average on Tool samples |",
        "",
        "## Throughput Metrics",
    ]

    throughput_metrics = [
        ('exe_time', 'Execution Time (s)'),
        ('output_token_per_seconds', 'Output Tokens/Sec'),
        ('total_token_per_second', 'Total Tokens/Sec')
    ]
    # Extract values from df (assuming df has 'predicted' column with throughput dict)
    vals = {}
    for k, _ in throughput_metrics:
        v = []
        for _, row in df.iterrows():
            try:
                t = row['predicted']['throughput']
                if k in t and t[k]: v.append(t[k])
            except: pass
        vals[k] = v

    stats = [
        ('Count', lambda x: len(x)),
        ('Mean', statistics.mean),
        ('Median', statistics.median),
        ('Min', min),
        ('Max', max),
        ('Std Dev', lambda x: statistics.stdev(x) if len(x)>1 else 0),
        ('Q25', lambda x: sorted(x)[len(x)//4]),
        ('Q75', lambda x: sorted(x)[3*len(x)//4])
    ]

    md_lines.extend([
        "",
        "| Stat | Execution Time (s) | Output Tokens/Sec | Total Tokens/Sec |",
        "| :--- | :--- | :--- | :--- |"
    ])
    for name, func in stats:
        row = [f"{func(vals[k]):.2f}" if name!='Count' else str(func(vals[k])) for k,_ in throughput_metrics]
        md_lines.append(f"| **{name}** | {row[0]} | {row[1]} | {row[2]} |")


    md_lines.append("")
    md_lines.append("## 🛠️ Per-Tool Argument Statistics")
    for tool, data in tool_stats.items():
        total = data['total']
        md_lines.append(f"### `{tool}`")
        md_lines.append(f"- **Total Occurrences**: {total}\n")
        
        # Status Table
        md_lines.append("| Status | Count | Percentage |")
        md_lines.append("| :--- | :--- | :--- |")
        for status, count in data['status_counts'].items():
            md_lines.append(f"| {status} | {count} | {(count/total)*100:.1f}% |")
        
        n_top_errors = 10
        n_examples = 5
        # Top Errors with Examples
        if data['error_counts']:
            md_lines.append("\n**Top Argument Errors & Examples:**")
            
            # Get top n errors
            for err, count in data['error_counts'].most_common(n_top_errors):
                md_lines.append(f"\n#### ❌ Error: `{err}` ({count} hits)")
                
                examples = data['error_examples'].get(err, [])
                # Show up to n examples
                for i, ex in enumerate(examples[:n_examples], 1):
                    user_msg = json.dumps(ex['user_message'], indent=2, ensure_ascii=False)
                    gt_dump = json.dumps(ex['gt'], indent=2, ensure_ascii=False)
                    pred_dump = json.dumps(ex['pred'], indent=2, ensure_ascii=False)
                    
                    md_lines.append(f"**Example {i}:**")
                    md_lines.append("```json")
                    md_lines.append(f"// User message\n{user_msg}\n\n// Ground Truth\n{gt_dump}\n\n// Predicted\n{pred_dump}")
                    md_lines.append("```")

        md_lines.append("\n---")

    md_lines.append("## 📈 Tool Selection Heatmap")
    md_lines.append("![Confusion Matrix](confusion_matrix.png)")

    # Export
    try:
        import markdown as md_lib
        from weasyprint import HTML
        
        # Add basic CSS for better readability
        html_body = md_lib.markdown("\n".join(md_lines), extensions=['tables', 'fenced_code'])
        styled_html = f"""
        <html>
        <head><style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: 600; }}
            h1 {{ border-bottom: 3px solid #007bff; padding-bottom: 10px; color: #2c3e50; }}
            h2 {{ border-bottom: 1px solid #eee; margin-top: 30px; color: #34495e; }}
            h3 {{ color: #007bff; margin-top: 25px; }}
            h4 {{ color: #dc3545; margin-top: 15px; font-size: 1rem; }}
            pre {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef; overflow-x: auto; }}
            code {{ font-family: 'Consolas', monospace; color: #d63384; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        </style></head>
        <body>{html_body}</body>
        </html>
        """
        
        pdf_path = output_dir / "evaluation_report.pdf"
        HTML(string=styled_html, base_url=str(output_dir)).write_pdf(str(pdf_path))
        print(f"✅ PDF Report generated: {pdf_path}")
    except ImportError:
        print("⚠️ Install 'weasyprint' and 'markdown' for PDF support. Saving MD instead.")
        with open(output_dir / "report.md", "w", encoding='utf-8') as f:
            f.write("\n".join(md_lines))

    # Save metrics
    df_args[['gt_tool_name', 'pred_tool_name', 'arg_accuracy']].to_json(output_dir / "metrics_args.json", orient='records')

if __name__ == "__main__":
    main()