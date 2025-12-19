#!/usr/bin/env python3
# exact_metrics.py
# NAME-ONLY accuracy & F1 (arguments ignored)
# WITH explicit NO_CALL class

import argparse
import pandas as pd
import os
from collections import defaultdict

NO_CALL = "__NO_CALL__"

# =========================================================
# Helper
# =========================================================

def get_names_from_expected(expected):
    if not expected:
        return [NO_CALL]
    return [e.get("name") for e in expected]


def get_names_from_predicted(predicted):
    calls = (predicted or {}).get("tool_calls", [])
    if not calls:
        return [NO_CALL]
    return [c.get("name") for c in calls]


# =========================================================
# Accuracy (NAME-ONLY)
# =========================================================

def is_name_correct(expected, predicted):
    """
    Correct if:
    - both NO_CALL
    - OR at least one function name matches
    """
    gt_names = set(get_names_from_expected(expected))
    pred_names = set(get_names_from_predicted(predicted))

    # both no call
    if gt_names == {NO_CALL} and pred_names == {NO_CALL}:
        return True

    # one correct function name is enough
    return len(gt_names & pred_names) > 0


# =========================================================
# Macro F1 (NAME-ONLY + NO_CALL)
# =========================================================

def compute_macro_f1(df):
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for _, row in df.iterrows():
        gt_names = set(get_names_from_expected(row.get("expected", [])))
        pred_names = set(get_names_from_predicted(row.get("predicted", {})))

        for name in gt_names:
            if name in pred_names:
                stats[name]["TP"] += 1
            else:
                stats[name]["FN"] += 1

        for name in pred_names:
            if name not in gt_names:
                stats[name]["FP"] += 1

    f1s = []
    for s in stats.values():
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Name-only accuracy & F1 (NO_CALL supported)"
    )
    parser.add_argument("predictions_ndjson")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_ndjson):
        raise FileNotFoundError(args.predictions_ndjson)

    df = pd.read_json(args.predictions_ndjson, lines=True)
    print(f"📊 Loaded {len(df)} samples")

    # -----------------------------------------------------
    # Per-sample accuracy
    # -----------------------------------------------------
    df["accuracy"] = df.apply(
        lambda r: is_name_correct(r.get("expected", []), r.get("predicted", {})),
        axis=1
    )

    print("\n✅ GLOBAL METRICS")
    print(f"Accuracy (name-only): {df['accuracy'].mean():.2%}")
    print(f"Macro-F1 (name-only): {compute_macro_f1(df):.2%}")

    # -----------------------------------------------------
    # Per-function F1 (GLOBAL)
    # -----------------------------------------------------
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for _, row in df.iterrows():
        gt = set(get_names_from_expected(row.get("expected", [])))
        pred = set(get_names_from_predicted(row.get("predicted", {})))

        for n in gt:
            if n in pred:
                stats[n]["TP"] += 1
            else:
                stats[n]["FN"] += 1

        for n in pred:
            if n not in gt:
                stats[n]["FP"] += 1

    print("\n📈 PER-FUNCTION METRICS (NAME-ONLY)")
    print("=" * 90)
    print(f"{'Function':<30}{'Precision':<15}{'Recall':<15}{'F1':<10}")
    print("-" * 90)

    for name in sorted(stats.keys()):
        s = stats[name]
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{name:<30}{p:<15.2%}{r:<15.2%}{f1:<10.2%}")

    print("=" * 90)

    # -----------------------------------------------------
    # Per (sheet, file)
    # -----------------------------------------------------
    if "_source_sheet" in df.columns and "_source_file" in df.columns:
        rows = []

        for (sheet, file), gdf in df.groupby(["_source_sheet", "_source_file"]):
            rows.append({
                "source_sheet": sheet,
                "source_file": file,
                "num_samples": len(gdf),
                "accuracy": gdf["accuracy"].mean(),
                "macro_f1": compute_macro_f1(gdf)
            })

        group_df = pd.DataFrame(rows)

        print("\n📊 PER (SHEET, FILE)")
        print("=" * 110)
        print(
            f"{'Sheet':<25}{'File':<35}"
            f"{'#Samples':<12}{'Accuracy':<15}{'Macro-F1':<10}"
        )
        print("-" * 110)

        for _, r in group_df.iterrows():
            print(
                f"{str(r['source_sheet']):<25}"
                f"{str(r['source_file']):<35}"
                f"{int(r['num_samples']):<12}"
                f"{r['accuracy']:<15.2%}"
                f"{r['macro_f1']:<10.2%}"
            )

        print("=" * 110)

        out = args.output or args.predictions_ndjson.replace(".ndjson", "")
        group_df.to_csv(f"{out}_by_sheet_file.csv", index=False)
        print(f"💾 Saved → {out}_by_sheet_file.csv")


if __name__ == "__main__":
    main()
