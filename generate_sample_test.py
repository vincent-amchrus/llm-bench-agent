#!/usr/bin/env python3
# generate_sample_test.py
import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(
        description="Sample balanced test set: at most N samples per function/tool."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file (e.g., data/processed/vi_test_2k1.json)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON path. Auto-generates if omitted: data/<lang>_test_each_max_<N>.json"
    )
    parser.add_argument(
        "--max_per_function", "-n",
        type=int,
        default=10,
        help="Max number of samples per function (default: 10)"
    )
    parser.add_argument(
        "--random_seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--function_col", "-c",
        default="function",
        help="Column name for function/tool name (default: 'function')"
    )
    parser.add_argument(
        "--message_col", default="user_message", help="User message column (for validation)")

    args = parser.parse_args()

    # Load data
    print(f"🔍 Loading {args.input}...")
    df = pd.read_json(args.input)
    print(f"📊 Loaded {len(df)} samples. Functions: {df[args.function_col].nunique()}")

    if args.function_col not in df.columns:
        raise ValueError(f"Column '{args.function_col}' not found. Available: {list(df.columns)}")

    # Separate null and non-null rows
    df_with_values = df[df[args.function_col].notna()].copy()
    df_null_values = df[df[args.function_col].isna()].copy()

    # Sample from non-null rows
    max_each_fn = args.max_per_function
    sampled_non_null = (
        df_with_values.groupby(args.function_col, group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), max_each_fn), random_state=args.random_seed))
          .reset_index(drop=True)
    )

    # Combine sampled non-null rows with all null rows
    sampled = pd.concat([sampled_non_null, df_null_values], ignore_index=True)

    print(f"🎯 Sampled {len(sampled)} samples ({max_each_fn} max per function)")
    if len(df_null_values) > 0:
        print(f"   Included {len(df_null_values)} samples with null values in '{args.function_col}' column")

    # Auto-generate output path if not given
    if args.output is None:
        basename = os.path.basename(args.input)
        stem = basename.rsplit('.', 1)[0]  # e.g., vi_test_2k1
        lang = "en" if "en" in stem.lower() else "vi" if "vi" in stem.lower() else "test"
        args.output = f"data/{lang}_test_each_max_{max_each_fn}.json"

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Save
    sampled.to_json(
        args.output,
        orient="records",
        force_ascii=False,
        indent=2
    )
    print(f"✅ Saved to: {args.output}")

    # Optional: show per-function counts
    counts = sampled[args.function_col].value_counts()
    print("\n📌 Sample counts per function (top 10):")
    print(counts.head(10).to_string())

if __name__ == "__main__":
    main()