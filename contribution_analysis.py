import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


def run_factor_analysis(data_path, output_path, metric_name):
    """
    Run factorial analysis (Approach 2: Remove redundant data and invalid interaction terms)

    Args:
        data_path: Path to the raw data CSV file
        output_path: Path to the results output file
        metric_name: Metric name ('ACC' or 'F1')
    """
    # Read data
    df = pd.read_csv(data_path, index_col=0)
    
    initial_count = len(df)
    # Filter out rows containing both Signal and HilbertB
    df = df[~((df.index.str.contains('Signal')) & (df.index.str.contains('HilbertB')))]
    final_count = len(df)
    print(f"[{metric_name}] Removed redundant configurations: {initial_count} -> {final_count} (removed SignalHilbertB related duplicates)")
    
    # Get filtered configurations and random seeds
    configs = df.index.tolist()
    seed_columns = ['7', '42', '123', '1309', '5287', '31415']
    
    # Build feature matrix and response variable
    X = []
    y = []
    
    for config in configs:
        # Base features
        is_signal = 1 if 'Signal' in config else 0
        is_lds = 1 if 'LDS' in config else 0
        is_hilbert_a = 1 if 'HilbertA' in config else 0
        is_hilbert_b = 1 if 'HilbertB' in config else 0
        
        features = [
            is_signal,
            is_lds,
            is_hilbert_a,
            is_hilbert_b,
            is_signal * is_hilbert_a,  # Interaction between signal and time-axis folding (valid)
            is_signal * is_lds,        # Interaction between signal and LDS sampling (valid)
            is_hilbert_a * is_lds,     # Interaction between time-axis folding and LDS (valid, for analyzing mechanism conflict)
            is_hilbert_b * is_lds      # Interaction between frequency-axis folding and LDS (valid)
            # Removed is_signal * is_hilbert_b because it does not physically exist
        ]
        
        for seed in seed_columns:
            X.append(features)
            y.append(df.loc[config, seed])
    
    X = np.array(X)
    y = np.array(y)
    
    # Perform regression analysis using statsmodels
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Extract coefficients and p-values
    coefficients = results.params
    p_values = results.pvalues
    
    # Update feature name list (corresponding to features)
    feature_names = [
        'Constant',
        'IsSignal',
        'IsLDS',
        'IsHilbertA',
        'IsHilbertB',
        'IsSignal × IsHilbertA',
        'IsSignal × IsLDS',
        'IsHilbertA × IsLDS',
        'IsHilbertB × IsLDS'
    ]
    
    # Build result string
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append(f"Factorial Analysis Results (Simplified) - {metric_name}")
    output_lines.append("=" * 70)
    output_lines.append("Note: SignalHilbertB redundant data has been removed, and invalid interaction terms have been excluded.")
    output_lines.append("")
    output_lines.append(results.summary().as_text())
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append(f"Coefficient Interpretation - {metric_name}")
    output_lines.append("=" * 70)
    output_lines.append("")
    
    for i, (coef, p_val, name) in enumerate(zip(coefficients, p_values, feature_names)):
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        output_lines.append(f"{name}: {coef:.6f} {significance} (p={p_val:.6f})")
    
    output_lines.append("")
    output_lines.append(f"R-squared: {results.rsquared:.4f}")
    output_lines.append(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
    output_lines.append(f"Total samples: {len(y)} (60 independent observations)")
    output_lines.append("")
    
    # Save results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    # Print results
    print('\n'.join(output_lines))
    
    return results


def main():
    # Base path (modify according to your actual environment)
    exp_id = '260414_mobile_coughvid'
    base_dir = f'results/{exp_id}/output'
    
    os.makedirs(base_dir, exist_ok=True)
    
    acc_data_path = os.path.join(base_dir, f'acc_raw_data_{exp_id}.csv')
    f1_data_path = os.path.join(base_dir, f'f1_raw_data_{exp_id}.csv')
    
    acc_output_path = os.path.join(base_dir, f'contribution_analysis_acc_{exp_id}_v2.txt')
    f1_output_path = os.path.join(base_dir, f'contribution_analysis_f1_{exp_id}_v2.txt')
    
    print("Starting simplified factorial analysis...")
    
    if os.path.exists(acc_data_path):
        run_factor_analysis(acc_data_path, acc_output_path, 'ACC')
    
    if os.path.exists(f1_data_path):
        run_factor_analysis(f1_data_path, f1_output_path, 'F1')

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()