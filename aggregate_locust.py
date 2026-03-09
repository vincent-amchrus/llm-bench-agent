import pandas as pd
import glob
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
BASE_DIR = "results/_partial_1k_vi_smart_labeled_0302"

OUTPUT_DIR = "results/_partial_1k_vi_smart_labeled_0302/locust_aggregate"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILENAME = "locust_stats_stats.csv"  # adjust if your file is named differently

PERCENTILES_TO_SHOW = ["50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%", "100%"]

IMPORTANT_COLUMNS = [
    "Model", "CCU", "Requests", "Failures",
    "Avg Latency (ms)", "Min (ms)", "Max (ms)",
    "RPS", "Avg Content Size (bytes)"
] + [f"p{p}" for p in PERCENTILES_TO_SHOW]

# ────────────────────────────────────────────────
def parse_model_and_ccu(folder_name: str) -> tuple[str, Optional[int]]:
    """
    Extract model name and CCU from folder name.
    Example: unsloth-Qwen3-4B-Instruct-2507_no-thinking_ccu_5_20260304_110833
    """
    # Find CCU
    ccu_match = re.search(r'_ccu_(\d+)_', folder_name, re.IGNORECASE)
    ccu = int(ccu_match.group(1)) if ccu_match else None

    # Clean model name: remove CCU part and timestamp suffix
    model = re.sub(r'_ccu_\d+_', '', folder_name, flags=re.IGNORECASE)
    model = re.sub(r'_\d{8}_\d{6}$', '', model)
    model = model.strip().replace('_', ' ')

    return model, ccu


def extract_aggregated_stats(csv_path: str) -> Optional[Dict[str, Any]]:
    """Read Locust stats CSV and extract key values from Aggregated row"""
    try:
        df = pd.read_csv(csv_path)
        # Prefer row where Name contains "Aggregated"
        agg_rows = df[df['Name'].str.contains('Aggregated', na=False, case=False)]
        if agg_rows.empty:
            agg_rows = df[df['Name'] == 'Aggregated']
        if agg_rows.empty and len(df) > 0:
            agg_rows = df.iloc[[-1]]  # fallback to last row

        row = agg_rows.iloc[0]

        stats = {
            "Requests": int(row.get("Request Count", 0)),
            "Failures": int(row.get("Failure Count", 0)),
            "Avg Latency (ms)": round(float(row.get("Average Response Time", 0)), 1),
            "Min (ms)": round(float(row.get("Min Response Time", 0)), 1),
            "Max (ms)": round(float(row.get("Max Response Time", 0)), 1),
            "RPS": round(float(row.get("Requests/s", 0)), 2),
            "Avg Content Size (bytes)": round(float(row.get("Average Content Size", 0)), 1),
        }

        # Percentiles
        for p in PERCENTILES_TO_SHOW:
            if p in row.index and pd.notna(row[p]):
                stats[f"p{p}"] = int(round(float(row[p]), 0))

        return stats
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


# ────────────────────────────────────────────────
# COLLECT DATA
# ────────────────────────────────────────────────
print("Scanning for Locust stats files...\n")

results = []

for csv_path in sorted(glob.glob(os.path.join(BASE_DIR, "**", CSV_FILENAME), recursive=True)):
    folder_name = Path(csv_path).parent.name
    model_name, ccu = parse_model_and_ccu(folder_name)

    if ccu is None:
        print(f"  Skipping (no CCU found): {folder_name}")
        continue

    stats = extract_aggregated_stats(csv_path)
    if stats is None:
        continue

    stats["Model"] = model_name
    stats["CCU"] = ccu
    results.append(stats)

    print(f"  {model_name:<60}  CCU {ccu:>3}   {csv_path}")

if not results:
    print("\nNo valid Locust stats found.")
    exit(0)

# ────────────────────────────────────────────────
# BUILD DATAFRAME & GROUP BY CCU
# ────────────────────────────────────────────────
df = pd.DataFrame(results)
df = df[ [c for c in IMPORTANT_COLUMNS if c in df.columns] ]
df = df.sort_values(["CCU", "Avg Latency (ms)"])

# ────────────────────────────────────────────────
# PRINT MARKDOWN TABLES — one per CCU
# ────────────────────────────────────────────────
print("\n" + "=" * 110)
print("              Latency Comparison Tables (by Concurrency / CCU)              ")
print("=" * 110 + "\n")

for ccu, group in df.groupby("CCU", sort=True):
    print(f"### CCU = {ccu}  ({len(group)} models)\n")

    # Drop CCU column for display
    display_df = group.drop(columns=["CCU"]).reset_index(drop=True)

    # Convert to markdown
    md_table = display_df.to_markdown(index=False, numalign="right", stralign="left")

    print(md_table)
    print("\n")

# Optional: save full comparison as CSV and Markdown

csv_output_path = os.path.join(OUTPUT_DIR, "latency_comparison_all_ccu.csv")
md_output_path = os.path.join(OUTPUT_DIR, "latency_comparison_all_ccu.md")
df.to_csv(csv_output_path, index=False)

with open(md_output_path, "w", encoding="utf-8") as f:
    f.write("# Locust Latency Comparison – Different Models @ Same CCU\n\n")
    for ccu, group in df.groupby("CCU", sort=True):
        f.write(f"## CCU = {ccu}\n\n")
        f.write(group.drop(columns=["CCU"]).to_markdown(index=False))
        f.write("\n\n")

print("\nSaved:")
print(f"  • {csv_output_path}")
print(f"  • {md_output_path}")