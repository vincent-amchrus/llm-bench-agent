import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_ndjson(filepath: str) -> List[Dict[str, Any]]:
    """Load NDJSON file into a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_throughput_metrics(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract throughput metrics from the data."""
    records = []
    for item in data:
        throughput = item.get('predicted', {}).get('throughput', {})
        if throughput:
            record = {
                'index': item.get('index'),
                'input_hash': item.get('input_hash'),
                'user_message': item.get('user_message', '')[:50],  # Truncate for display
                'exe_time': throughput.get('exe_time'),
                'output_token_per_seconds': throughput.get('output_token_per_seconds'),
                'total_token_per_second': throughput.get('total_token_per_second'),
                'prompt_tokens': item.get('predicted', {}).get('usage', {}).get('prompt_tokens'),
                'completion_tokens': item.get('predicted', {}).get('usage', {}).get('completion_tokens'),
                'total_tokens': item.get('predicted', {}).get('usage', {}).get('total_tokens'),
            }
            records.append(record)
    
    return pd.DataFrame(records)

def calculate_statistics(series: pd.Series, metric_name: str) -> Dict[str, float]:
    """Calculate comprehensive statistics for a metric."""
    stats = {
        'metric': metric_name,
        'count': series.count(),
        'mean': series.mean(),
        'median': series.median(),
        'min': series.min(),
        'max': series.max(),
        'std': series.std(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
    }
    return stats

def print_statistics(df: pd.DataFrame):
    """Print throughput statistics in a formatted way."""
    metrics = [
        ('exe_time', 'Execution Time (seconds)'),
        ('output_token_per_seconds', 'Output Tokens/Second'),
        ('total_token_per_second', 'Total Tokens/Second'),
    ]
    
    print("=" * 80)
    print("THROUGHPUT STATISTICS AFTER INFERENCE")
    print("=" * 80)
    print(f"\nTotal Records Analyzed: {len(df)}\n")
    
    for col, name in metrics:
        stats = calculate_statistics(df[col], name)
        print(f"\n{name}")
        print("-" * 60)
        print(f"  Count:  {stats['count']:>10}")
        print(f"  Mean:   {stats['mean']:>10.2f}")
        print(f"  Median: {stats['median']:>10.2f}")
        print(f"  Min:    {stats['min']:>10.2f}")
        print(f"  Max:    {stats['max']:>10.2f}")
        print(f"  Std Dev:{stats['std']:>10.2f}")
        print(f"  Q25:    {stats['q25']:>10.2f}")
        print(f"  Q75:    {stats['q75']:>10.2f}")
    
    print("\n" + "=" * 80)

def print_token_statistics(df: pd.DataFrame):
    """Print token usage statistics."""
    print("\nTOKEN USAGE STATISTICS")
    print("=" * 80)
    
    token_metrics = [
        ('prompt_tokens', 'Prompt Tokens'),
        ('completion_tokens', 'Completion Tokens'),
        ('total_tokens', 'Total Tokens'),
    ]
    
    for col, name in token_metrics:
        if col in df.columns:
            stats = calculate_statistics(df[col], name)
            print(f"\n{name}")
            print("-" * 60)
            print(f"  Mean:   {stats['mean']:>10.0f}")
            print(f"  Median: {stats['median']:>10.0f}")
            print(f"  Min:    {stats['min']:>10.0f}")
            print(f"  Max:    {stats['max']:>10.0f}")

def create_visualizations(df: pd.DataFrame, output_dir: str = 'throughput_plots'):
    """Create visualization plots for throughput metrics."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Execution Time Distribution
    axes[0, 0].hist(df['exe_time'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(df['exe_time'].mean(), color='red', linestyle='--', label=f"Mean: {df['exe_time'].mean():.2f}s")
    axes[0, 0].set_xlabel('Execution Time (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Execution Time Distribution')
    axes[0, 0].legend()
    
    # 2. Output Tokens/Second Distribution
    axes[0, 1].hist(df['output_token_per_seconds'], bins=10, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(df['output_token_per_seconds'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {df['output_token_per_seconds'].mean():.2f}")
    axes[0, 1].set_xlabel('Output Tokens/Second')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Output Token Throughput Distribution')
    axes[0, 1].legend()
    
    # 3. Total Tokens/Second Distribution
    axes[1, 0].hist(df['total_token_per_second'], bins=10, edgecolor='black', alpha=0.7, color='salmon')
    axes[1, 0].axvline(df['total_token_per_second'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['total_token_per_second'].mean():.2f}")
    axes[1, 0].set_xlabel('Total Tokens/Second')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Total Token Throughput Distribution')
    axes[1, 0].legend()
    
    # 4. Execution Time vs Output Throughput Scatter
    axes[1, 1].scatter(df['exe_time'], df['output_token_per_seconds'], alpha=0.6, s=100, color='purple')
    axes[1, 1].set_xlabel('Execution Time (seconds)')
    axes[1, 1].set_ylabel('Output Tokens/Second')
    axes[1, 1].set_title('Execution Time vs Output Throughput')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Box plot for all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df[['exe_time', 'output_token_per_seconds', 'total_token_per_second']].copy()
    # Normalize for better visualization
    df_plot['exe_time_norm'] = df_plot['exe_time'] / df_plot['exe_time'].max()
    df_plot['output_norm'] = df_plot['output_token_per_seconds'] / df_plot['output_token_per_seconds'].max()
    df_plot['total_norm'] = df_plot['total_token_per_second'] / df_plot['total_token_per_second'].max()
    
    df_plot[['exe_time_norm', 'output_norm', 'total_norm']].boxplot(ax=ax)
    ax.set_xticklabels(['Exec Time\n(normalized)', 'Output TPS\n(normalized)', 'Total TPS\n(normalized)'])
    ax.set_ylabel('Normalized Value')
    ax.set_title('Throughput Metrics Comparison (Normalized)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualizations saved to '{output_dir}/' directory")

def export_summary(df: pd.DataFrame, output_file: str = 'throughput_summary.csv'):
    """Export detailed summary to CSV."""
    df.to_csv(output_file, index=False)
    print(f"✓ Detailed data exported to '{output_file}'")

def main():
    # Load data
    filepath = 'results/_partial_12_en_global_labeled/Qwen-Qwen3.5-4B-thinking_ccu_4/predictions.ndjson'
    print(f"Loading data from: {filepath}")
    
    try:
        data = load_ndjson(filepath)
        print(f"✓ Loaded {len(data)} records\n")
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Extract metrics
    df = extract_throughput_metrics(data)
    
    if df.empty:
        print("No throughput data found in the file!")
        return
    
    # Print statistics
    print_statistics(df)
    print_token_statistics(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Export summary
    export_summary(df)
    
    # Print sample records
    print("\n" + "=" * 80)
    print("SAMPLE RECORDS (First 5)")
    print("=" * 80)
    print(df[['index', 'exe_time', 'output_token_per_seconds', 'total_token_per_second']].head().to_string(index=False))

if __name__ == '__main__':
    main()