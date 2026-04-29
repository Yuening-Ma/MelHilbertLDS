import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import warnings
warnings.filterwarnings('ignore')

# Set font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_run_name(run_name):
    """Parse run directory name to extract type and seed"""
    # New format: css_{dataset}_{type}_{hop_length}_{seed} or css_{dataset}_{type}_{seed}
    # e.g., css_CoughDataset_MelHilbertLDS_64_123, css_CoughDataset_Mel_123
    parts = run_name.replace('css_', '').split('_')

    # Remove 'mobile', 'pann', 'naive' from parts if present
    while True:
        if 'mobile' in parts:
            parts.remove('mobile')
        elif 'pann' in parts:
            parts.remove('pann')
        elif 'naive' in parts:
            parts.remove('naive')
        else:
            break
    
    # Check if hop_length exists (LDS types have hop_length, non-LDS types do not)
    # The last part is seed (numeric), the second-to-last may be hop_length or the last part of type
    seed = int(parts[-1])
    
    # Check for hop_length (second-to-last part is purely numeric)
    if parts[-2].isdigit():
        # With hop_length: css_{dataset}_{type}_{hop_length}_{seed}
        # Type part is between dataset and hop_length
        type_name = '_'.join(parts[1:-2])  # Skip dataset, remove hop_length and seed
    else:
        # Without hop_length: css_{dataset}_{type}_{seed}
        type_name = '_'.join(parts[1:-1])  # Skip dataset, remove seed
    
    return type_name, seed


def load_scalar_from_dir(scalar_dir):
    """Load scalar data from a scalar directory"""
    event_files = glob.glob(os.path.join(scalar_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None, None, None
    
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    if not tags:
        return None, None, None
    
    tag = tags[0]  # Typically only one scalar
    events = ea.Scalars(tag)
    
    steps = [e.step for e in events]
    values = [e.value for e in events]
    timestamps = [e.wall_time for e in events]
    
    return steps, values, timestamps


def extract_best_metrics(run_dir):
    """Extract best validation metrics from a run directory"""
    metrics = {}
    
    # Load scalar data
    data = {}
    scalar_names = ['Accuracy_val', 'F1_val', 'Precision_val', 'Recall_val', 'Loss_train']
    
    for scalar_name in scalar_names:
        scalar_dir = os.path.join(run_dir, scalar_name)
        if os.path.exists(scalar_dir):
            steps, values, timestamps = load_scalar_from_dir(scalar_dir)
            if values:
                data[scalar_name] = {
                    'steps': steps,
                    'values': values,
                    'timestamps': timestamps
                }
    
    # Get validation F1 (maximum value)
    if 'F1_val' in data:
        f1_values = data['F1_val']['values']
        metrics['best_f1'] = max(f1_values) if f1_values else 0
        metrics['best_f1_epoch'] = f1_values.index(max(f1_values)) + 1 if f1_values else 0
    else:
        metrics['best_f1'] = 0
        metrics['best_f1_epoch'] = 0
    
    # Get validation Accuracy (maximum value)
    if 'Accuracy_val' in data:
        acc_values = data['Accuracy_val']['values']
        metrics['best_acc'] = max(acc_values) if acc_values else 0
        metrics['best_acc_epoch'] = acc_values.index(max(acc_values)) + 1 if acc_values else 0
    else:
        metrics['best_acc'] = 0
        metrics['best_acc_epoch'] = 0
    
    # Get Precision and Recall at the best epoch
    best_epoch = metrics['best_f1_epoch'] - 1 if metrics['best_f1_epoch'] > 0 else -1
    
    if 'Precision_val' in data and best_epoch >= 0:
        prec_values = data['Precision_val']['values']
        metrics['precision_at_best_f1'] = prec_values[best_epoch] if best_epoch < len(prec_values) else 0
    else:
        metrics['precision_at_best_f1'] = 0
    
    if 'Recall_val' in data and best_epoch >= 0:
        recall_values = data['Recall_val']['values']
        metrics['recall_at_best_f1'] = recall_values[best_epoch] if best_epoch < len(recall_values) else 0
    else:
        metrics['recall_at_best_f1'] = 0
    
    # Compute average epoch time
    if 'Loss_train' in data and len(data['Loss_train']['timestamps']) > 1:
        timestamps = data['Loss_train']['timestamps']
        total_time = timestamps[-1] - timestamps[0]
        num_epochs = len(timestamps)
        metrics['avg_epoch_time'] = total_time / num_epochs if num_epochs > 0 else 0
    else:
        metrics['avg_epoch_time'] = 0
    
    return metrics


def analyze_experiments(base_dir, exp_id):
    """Analyze all experiment results"""
    runs_dir = os.path.join(base_dir, f'runs_{exp_id}')
    
    # Collect data from all runs
    all_results = []
    
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('css_')]
    
    print(f"Found {len(run_dirs)} experiment runs...")
    
    for run_dir in sorted(run_dirs):
        run_path = os.path.join(runs_dir, run_dir)
        
        # Find event files
        event_files = glob.glob(os.path.join(run_path, 'events.out.tfevents.*'))
        
        if not event_files:
            print(f"Warning: No event files found in {run_dir}")
            continue
        
        try:
            # Extract metrics
            metrics = extract_best_metrics(run_path)
            
            # Parse run name
            type_name, seed = parse_run_name(run_dir)
            
            all_results.append({
                'type': type_name,
                'seed': seed,
                **metrics
            })
            
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            continue
    
    return pd.DataFrame(all_results)


def remap_type_name(type_name):
    """Remap type names"""
    # HilbertTime -> HilbertB
    # Hilbert (without Time) -> HilbertA
    # Signal -> Signal + copy as SignalHilbertB
    # SignalLDS -> SignalLDS + copy as SignalHilbertBLDS
    
    if 'HilbertTime' in type_name:
        return type_name.replace('HilbertTime', 'HilbertB')
    elif 'Hilbert' in type_name and 'Time' not in type_name:
        return type_name.replace('Hilbert', 'HilbertA')
    else:
        return type_name


def expand_signal_rows(df):
    """Expand Signal rows into SignalHilbertB and SignalHilbertBLDS"""
    new_rows = []
    
    for _, row in df.iterrows():
        type_name = row['type']
        
        # Add original row
        new_rows.append(row.copy())
        
        # Signal -> additionally add SignalHilbertB
        if type_name == 'Signal':
            new_row = row.copy()
            new_row['type'] = 'SignalHilbertB'
            new_rows.append(new_row)
        
        # SignalLDS -> additionally add SignalHilbertBLDS
        if type_name == 'SignalLDS':
            new_row = row.copy()
            new_row['type'] = 'SignalHilbertBLDS'
            new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)


def compute_statistics(df):
    """Compute statistical metrics for each type"""
    # Remap type names
    df = df.copy()
    df['type'] = df['type'].apply(remap_type_name)
    
    # Expand Signal rows
    df = expand_signal_rows(df)
    
    # Group by type and compute statistics
    stats = df.groupby('type').agg({
        'best_acc': ['mean', 'std'],
        'best_f1': ['mean', 'std'],
        'precision_at_best_f1': 'mean',
        'recall_at_best_f1': 'mean',
        'avg_epoch_time': 'mean'
    })
    
    # Flatten column names
    stats.columns = [
        'acc_mean', 'acc_std',
        'f1_mean', 'f1_std',
        'precision_mean',
        'recall_mean',
        'avg_epoch_time_mean'
    ]
    
    # Reset index
    stats = stats.reset_index()
    
    return stats, df  # Return processed df for generating raw data tables


def create_raw_data_tables(df):
    """Create raw data tables for ACC and F1 (one row per type, one column per seed)"""
    # Remap type names
    df = df.copy()
    df['type'] = df['type'].apply(remap_type_name)
    
    # Expand Signal rows
    df = expand_signal_rows(df)
    
    # Create ACC pivot table
    acc_pivot = df.pivot_table(
        index='type',
        columns='seed',
        values='best_acc',
        aggfunc='first'
    )
    
    # Create F1 pivot table
    f1_pivot = df.pivot_table(
        index='type',
        columns='seed',
        values='best_f1',
        aggfunc='first'
    )
    
    # Sort types in specified order
    type_order = [
        'Mel', 'MelLDS',
        'MelHilbertA', 'MelHilbertALDS',
        'MelHilbertB', 'MelHilbertBLDS',
        'Signal', 'SignalLDS',
        'SignalHilbertA', 'SignalHilbertALDS',
        'SignalHilbertB', 'SignalHilbertBLDS'
    ]
    
    # Filter existing types and sort in order
    acc_pivot = acc_pivot.reindex([t for t in type_order if t in acc_pivot.index])
    f1_pivot = f1_pivot.reindex([t for t in type_order if t in f1_pivot.index])
    
    return acc_pivot, f1_pivot


def create_summary_table(stats_df):
    """Create summary table"""
    # Select and rename columns
    summary = stats_df.copy()
    
    # Format numerical values
    for col in summary.columns:
        if col != 'type':
            summary[col] = summary[col].apply(lambda x: f"{x:.4f}")
    
    return summary


def plot_boxplots(df, output_dir, exp_id):
    """Plot boxplots for F1 and Accuracy"""
    # Remap type names
    df = df.copy()
    df['type'] = df['type'].apply(remap_type_name)
    
    # Filter out copied types (SignalHilbertB, SignalHilbertBLDS)
    excluded_types = ['SignalHilbertB', 'SignalHilbertBLDS']
    df = df[~df['type'].isin(excluded_types)]
    
    # Sort in specified order
    type_order = [
        'Mel', 'MelLDS',
        'MelHilbertA', 'MelHilbertALDS',
        'MelHilbertB', 'MelHilbertBLDS',
        'Signal', 'SignalLDS',
        'SignalHilbertA', 'SignalHilbertALDS'
    ]
    
    types = [t for t in type_order if t in df['type'].unique()]
    
    if len(types) == 0:
        print("Warning: No type data available, skipping boxplot generation")
        return
    
    # Prepare data
    acc_data = [df[df['type'] == t]['best_acc'].values for t in types]
    f1_data = [df[df['type'] == t]['best_f1'].values for t in types]
    
    # Filter out empty data
    valid_indices = [i for i, data in enumerate(acc_data) if len(data) > 0]
    types = [types[i] for i in valid_indices]
    acc_data = [acc_data[i] for i in valid_indices]
    f1_data = [f1_data[i] for i in valid_indices]
    
    if len(types) == 0:
        print("Warning: All type data is empty, skipping boxplot generation")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy boxplot
    bp1 = ax1.boxplot(acc_data, labels=types, patch_artist=True)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Validation Accuracy Distribution by Type', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Set box colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    # F1 boxplot
    bp2 = ax2.boxplot(f1_data, labels=types, patch_artist=True)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Validation F1 Score Distribution by Type', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Set box colors
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplots_{exp_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot saved to: {os.path.join(output_dir, f'boxplots_{exp_id}.png')}")


def main():
    # Specify experiment ID
    exp_id = '260414_pann_css'
    
    # Base directory
    base_dir = f'./results/{exp_id}'
    output_dir = os.path.join(base_dir, 'output')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Analyzing experiment results (exp_id: {exp_id})...")
    print("=" * 60)
    
    # Analyze all experiments
    df = analyze_experiments(base_dir, exp_id)
    
    if df.empty:
        print("No valid experiment data found!")
        return
    
    print(f"\nSuccessfully loaded {len(df)} experiment records")
    print(f"Experiment types: {df['type'].nunique()}")
    print(f"Number of seeds: {df['seed'].nunique()}")
    
    # Compute statistics (also get processed df)
    stats_df, processed_df = compute_statistics(df)
    
    # Create raw data tables
    acc_raw, f1_raw = create_raw_data_tables(df)
    
    # Save raw data tables
    acc_raw_path = os.path.join(output_dir, f'acc_raw_data_{exp_id}.csv')
    f1_raw_path = os.path.join(output_dir, f'f1_raw_data_{exp_id}.csv')
    acc_raw.to_csv(acc_raw_path)
    f1_raw.to_csv(f1_raw_path)
    print(f"ACC raw data table saved to: {acc_raw_path}")
    print(f"F1 raw data table saved to: {f1_raw_path}")
    
    # Create summary table
    summary_table = create_summary_table(stats_df)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f'results_summary_{exp_id}.csv')
    summary_table.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to: {csv_path}")
    
    # Save as Markdown table (transposed: rows are types, columns are metrics)
    md_path = os.path.join(output_dir, f'results_summary_{exp_id}.md')
    
    # Create transposed table
    transposed_data = {}
    for _, row in stats_df.iterrows():
        type_name = row['type']
        transposed_data[type_name] = {
            'Accuracy Mean (%)': f"{row['acc_mean']:.4f}",
            'Accuracy Std (%)': f"{row['acc_std']:.4f}",
            'F1 Mean': f"{row['f1_mean']:.4f}",
            'F1 Std': f"{row['f1_std']:.4f}",
            'Precision Mean': f"{row['precision_mean']:.4f}",
            'Recall Mean': f"{row['recall_mean']:.4f}",
            'Avg Epoch Time (s)': f"{row['avg_epoch_time_mean']:.4f}"
        }
    
    transposed_df = pd.DataFrame(transposed_data).T
    transposed_df.index.name = 'Type'
    
    with open(md_path, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        f.write("## Performance Metrics by Type\n\n")
        f.write(transposed_df.to_markdown())
        f.write("\n\n")
        f.write("## Notes\n\n")
        f.write("- Each type includes 6 independent experiments (different random seeds)\n")
        f.write("- Metrics are the best values on the validation set\n")
        f.write("- Precision and Recall are taken at the best F1 epoch\n")
    
    print(f"Markdown report saved to: {md_path}")
    
    # Plot boxplots
    plot_boxplots(df, output_dir, exp_id)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Experiment Results Summary")
    print("=" * 60)
    print(transposed_df.to_string())
    print("=" * 60)
    
    # Find best types
    best_by_acc = stats_df.loc[stats_df['acc_mean'].idxmax(), 'type']
    best_by_f1 = stats_df.loc[stats_df['f1_mean'].idxmax(), 'type']
    
    print(f"\nBest Accuracy type: {best_by_acc} (avg: {stats_df['acc_mean'].max():.4f}%)")
    print(f"Best F1 type: {best_by_f1} (avg: {stats_df['f1_mean'].max():.4f})")


if __name__ == '__main__':
    main()
