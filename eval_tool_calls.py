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

tqdm.pandas()


def parse_tool_call(tool_calls):
    if isinstance(tool_calls, dict):
        tool_calls = tool_calls.get('tool_calls', [])
    if not isinstance(tool_calls, list) or len(tool_calls) == 0:
        return "__NO_CALL__"
    try:
        return tool_calls[0].get('name', "__NO_CALL__")
    except (IndexError, AttributeError, TypeError):
        return "__NO_CALL__"


def compute_tool_only_accuracy(y_true, y_pred):
    mask = y_true != "__NO_CALL__"
    if mask.sum() == 0:
        return np.nan
    return accuracy_score(y_true[mask], y_pred[mask])


def compute_no_call_accuracy(y_true, y_pred):
    mask = y_true == "__NO_CALL__"
    if mask.sum() == 0:
        return np.nan
    return accuracy_score(y_true[mask], y_pred[mask])


def classification_report_to_markdown(y_true, y_pred, digits=2):
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    lines = []

    lines.append("| Label | Precision (%) | Recall (%) | F1-Score (%) | Support |")
    lines.append("|-------|---------------|------------|--------------|---------|")

    for label in sorted(set(y_true) | set(y_pred)):
        if label not in report_dict:
            continue
        m = report_dict[label]
        prec = f"{m['precision'] * 100:.{digits}f}"
        rec = f"{m['recall'] * 100:.{digits}f}"
        f1 = f"{m['f1-score'] * 100:.{digits}f}"
        supp = int(m['support'])
        lines.append(f"| `{label}` | {prec} | {rec} | {f1} | {supp} |")

    for avg_type in ["macro avg", "weighted avg"]:
        if avg_type in report_dict:
            m = report_dict[avg_type]
            prec = f"{m['precision'] * 100:.{digits}f}"
            rec = f"{m['recall'] * 100:.{digits}f}"
            f1 = f"{m['f1-score'] * 100:.{digits}f}"
            supp = int(m['support'])
            lines.append(f"| **{avg_type}** | {prec} | {rec} | {f1} | {supp} |")

    lines.append("")
    if "accuracy" in report_dict:
        acc_pct = report_dict["accuracy"] * 100
        lines.append(f"- **Overall Accuracy**: {acc_pct:.{digits}f}%")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool-call prediction results.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predictions.ndjson")
    args = parser.parse_args()

    infer_path = Path(args.pred_path)
    if not infer_path.exists():
        raise FileNotFoundError(f"Input file not found: {infer_path}")

    output_dir = infer_path.parent

    # Load and parse
    df = pd.read_json(infer_path, lines=True)
    df['pred_tool'] = df['predicted'].progress_apply(parse_tool_call)
    df['gt_tool'] = df['expected'].progress_apply(parse_tool_call)

    y_true = df['gt_tool']
    y_pred = df['pred_tool']

    # Metrics (as %, rounded to 2 decimals)
    overall_acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    tool_acc = compute_tool_only_accuracy(y_true, y_pred)
    no_call_acc = compute_no_call_accuracy(y_true, y_pred)

    tool_acc_pct = round(tool_acc * 100, 2) if not np.isnan(tool_acc) else None
    no_call_acc_pct = round(no_call_acc * 100, 2) if not np.isnan(no_call_acc) else None

    # Confusion matrix → image
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix (Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_counts_path = output_dir / "confusion_matrix_counts.png"
    plt.savefig(cm_counts_path, dpi=200, bbox_inches='tight')
    plt.close()

    # --- Confusion Matrix: Percentages (row-normalized: % of true label)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')  # row-wise
    cm_norm_df = pd.DataFrame(cm_norm * 100, index=labels, columns=labels)  # convert to %

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm_df,
        annot=True,
        fmt='.1f',  # show 1 decimal: e.g., 92.3
        cmap='Blues',
        cbar_kws={'label': 'Percentage (%)'},
        vmin=0, vmax=100
    )
    plt.title('Confusion Matrix (Row-Normalized %)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_percent_path = output_dir / "confusion_matrix_percent.png"
    plt.savefig(cm_percent_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Classification report dict (for JSON)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Build JSON summary
    json_summary = {
        "metadata": {
            "input_file": str(infer_path),
            "total_samples": len(df),
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": {
            "overall_accuracy_pct": overall_acc,
            "tool_call_accuracy_pct": tool_acc_pct,    # None if no tool-call GT
            "no_call_accuracy_pct": no_call_acc_pct,   # None if no no-call GT
        },
        "data_stats": {
            "tool_call_gt_count": int((y_true != "__NO_CALL__").sum()),
            "no_call_gt_count": int((y_true == "__NO_CALL__").sum()),
        },
        "per_class_metrics": {
            label: {
                "precision_pct": round(report_dict[label]["precision"] * 100, 2),
                "recall_pct": round(report_dict[label]["recall"] * 100, 2),
                "f1_score_pct": round(report_dict[label]["f1-score"] * 100, 2),
                "support": int(report_dict[label]["support"]),
            }
            for label in sorted(set(y_true) | set(y_pred))
            if label in report_dict and isinstance(report_dict[label], dict)
        },
        "averages": {
            avg_type: {
                "precision_pct": round(report_dict[avg_type]["precision"] * 100, 2),
                "recall_pct": round(report_dict[avg_type]["recall"] * 100, 2),
                "f1_score_pct": round(report_dict[avg_type]["f1-score"] * 100, 2),
                "support": int(report_dict[avg_type]["support"]),
            }
            for avg_type in ["macro avg", "weighted avg"]
            if avg_type in report_dict
        }
    }

    # Save JSON
    json_path = output_dir / "evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"✅ Metrics JSON saved to: {json_path.name}")

    # Classification report (for MD)
    report_md = classification_report_to_markdown(y_true, y_pred, digits=2)

    # Markdown
    md_lines = []
    md_lines.append("# Tool-Call Evaluation Report")
    md_lines.append(f"**Input file**: `{infer_path}`")
    md_lines.append(f"**Total samples**: {len(df)}")
    md_lines.append("")

    # 1. Metrics
    md_lines.append("## 📊 Metrics")
    md_lines.append("| Metric | Value | Note |")
    md_lines.append("|--------|-------|------|")
    md_lines.append(f"| Overall Accuracy | `{overall_acc:.2f}%` | All samples |")
    md_lines.append(f"| Tool-call Accuracy | `{tool_acc_pct:.2f}%` if tool_acc_pct is not None else `N/A` | GT ≠ `__NO_CALL__` |")
    md_lines.append(f"| No-call Accuracy | `{no_call_acc_pct:.2f}%` if no_call_acc_pct is not None else `N/A` | GT = `__NO_CALL__` |")
    md_lines.append("")

    # 2. Classification Report
    md_lines.append("## 📋 Classification Report")
    md_lines.append(report_md)
    md_lines.append("")

    # 3. Data Stats
    tool_samples = (y_true != '__NO_CALL__').sum()
    no_call_samples = (y_true == '__NO_CALL__').sum()
    md_lines.append("## 📦 Data Stats")
    md_lines.append("| Statistic | Count |")
    md_lines.append("|-----------|-------|")
    md_lines.append(f"| Tool-call (GT) | {tool_samples} |")
    md_lines.append(f"| No-call (GT) | {no_call_samples} |")
    md_lines.append("")

    # 4. Confusion Matrix
        # 4. Confusion Matrix
    md_lines.append("## 📈 Confusion Matrix")

    md_lines.append("### Row-Normalized (%) — % of *True* Label")
    md_lines.append("Each row sums to 100%. Shows error distribution *per ground-truth class*.")
    md_lines.append("![Percent](confusion_matrix_percent.png)")
    md_lines.append("")

    md_lines.append("### Counts")
    md_lines.append("![Counts](confusion_matrix_counts.png)")
    md_lines.append("")
    
    # Save Markdown
    md_path = output_dir / "evaluation_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ Report saved to: {md_path.name}")
        # --- PDF Export (NEW) ---
    try:
        import markdown as md_lib
        from weasyprint import HTML, CSS
        from tempfile import NamedTemporaryFile

        # Convert Markdown → HTML (with fenced code styling and image support)
        html_content = md_lib.markdown(
            "\n".join(md_lines),
            extensions=[
                'tables',
                'fenced_code',
                'codehilite',
                'nl2br',
                'sane_lists'
            ]
        )

        # Wrap in basic styled template for better PDF appearance
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Tool-Call Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; margin: 1em 0; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 0.5em 1em; text-align: left; }}
                th {{ background-color: #ecf0f1; }}
                img {{ max-width: 100%; height: auto; margin: 1em 0; }}
                code {{ background-color: #f9f9f9; padding: 2px 4px; border-radius: 3px; }}
                pre {{ background-color: #f5f5f5; padding: 1em; overflow: auto; border-radius: 4px; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """

        # Write HTML to temp file (required for relative image path resolution)
        with NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tf:
            tf.write(styled_html)
            temp_html_path = Path(tf.name)

        # Generate PDF (resolve image path relative to output_dir)
        pdf_path = output_dir / "evaluation_summary.pdf"
        HTML(filename=str(temp_html_path), base_url=str(output_dir)).write_pdf(
            str(pdf_path),
            # Optional: improve font rendering
            # stylesheets=[CSS(string='body { font-family: sans-serif; }')]
        )

        # Cleanup temp HTML
        temp_html_path.unlink(missing_ok=True)

        print(f"✅ PDF report saved to: {pdf_path.name}")

    except ImportError as e:
        print("⚠️  `markdown` or `weasyprint` not installed — skipping PDF generation.")
        print("   To enable PDF: pip install markdown weasyprint")
        pdf_path = None
    except Exception as e:
        print(f"⚠️  PDF generation failed: {e}")
        pdf_path = None
    print("\n📁 Output files:")
    print(f"  - {md_path.name}")
    print(f"  - {cm_percent_path.name}")
    print(f"  - {cm_counts_path.name}")
    print(f"  - {json_path.name}")
    if 'pdf_path' in locals() and pdf_path and pdf_path.exists():
        print(f"  - {pdf_path.name}")

if __name__ == "__main__":
    main()