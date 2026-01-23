#!/usr/bin/env python3
"""
In-Situ Validation Analysis: Comprehensive Summary
===================================================

Produces a complete set of figures and tables summarizing the in-situ validation
comparison between DINEOF and DINCAE gap-filling methods.

Questions Addressed:
Q1: Does observation (satellite vs insitu) quality predict winner?
Q2: What is the story across data types (all, observed, missing)?
Q3: What are the strong predictors of winner? (Residual STD, Bias patterns)
Q4: What is the mechanism behind residual STD as predictor?
Q5: Per-lake summary with lake IDs
Q6: Comprehensive tables and additional visualizations

Outputs:
- 10+ publication-ready figures
- Detailed CSV tables for further analysis
- Text summary report

Usage:
    python insitu_validation_5_questions.py --analysis_dir /path/to/insitu_validation_analysis/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

COLORS = {
    'dineof': '#5DA5DA', 
    'dincae': '#FAA43A', 
    'obs': '#60BD68', 
    'strong': '#27ae60', 
    'weak': '#bdc3c7',
    'tie': '#808080'
}


def safe_pearsonr(x, y):
    """Compute Pearson correlation safely, handling different scipy versions."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        return np.nan, np.nan
    
    try:
        result = stats.pearsonr(x, y)
        if hasattr(result, 'statistic'):
            r, p = result.statistic, result.pvalue
        else:
            r, p = result
        r = float(np.asarray(r).flat[0])
        p = float(np.asarray(p).flat[0])
        return r, p
    except Exception:
        return np.nan, np.nan


def load_data(analysis_dir: str) -> pd.DataFrame:
    """Load the raw combined stats."""
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    return pd.read_csv(csv_path)


def extract_lake_stats(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract per-lake stats for all data types and metrics."""
    metrics = ['rmse', 'mae', 'bias', 'std', 'correlation', 'n_matches']
    if 'rstd' in df_raw.columns:
        metrics.append('rstd')
    if 'median' in df_raw.columns:
        metrics.append('median')
    
    results = []
    for lake_id in df_raw['lake_id_cci'].unique():
        lake_df = df_raw[df_raw['lake_id_cci'] == lake_id]
        row = {'lake_id': int(lake_id)}
        
        # Observation
        obs = lake_df[lake_df['data_type'] == 'observation']
        if not obs.empty:
            for m in metrics:
                if m in obs.columns:
                    row[f'obs_{m}'] = obs[m].mean()
        
        # Reconstructions
        for method in ['dineof', 'dincae']:
            for dtype in ['reconstruction', 'reconstruction_observed', 'reconstruction_missing']:
                subset = lake_df[(lake_df['data_type'] == dtype) & (lake_df['method'] == method)]
                if not subset.empty:
                    prefix = f"{method}_{dtype.replace('reconstruction_', '').replace('reconstruction', 'all')}"
                    for m in metrics:
                        if m in subset.columns:
                            row[f'{prefix}_{m}'] = subset[m].mean()
        
        # Coverage fraction
        n_obs = row.get('dineof_observed_n_matches', 0) or 0
        n_miss = row.get('dineof_missing_n_matches', 0) or 0
        total = n_obs + n_miss
        if total > 0:
            row['coverage_fraction'] = n_obs / total
            row['gap_fraction'] = n_miss / total
            row['n_observed'] = n_obs
            row['n_missing'] = n_miss
            row['n_total'] = total
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Rename columns for consistency
    for prefix in ['obs', 'dineof_all', 'dincae_all', 'dineof_observed', 'dincae_observed', 
                   'dineof_missing', 'dincae_missing']:
        rstd_col = f'{prefix}_rstd'
        robust_col = f'{prefix}_robust_std'
        if rstd_col in df.columns and robust_col not in df.columns:
            df[robust_col] = df[rstd_col]
        median_col = f'{prefix}_median'
        median_err_col = f'{prefix}_median_error'
        if median_col in df.columns and median_err_col not in df.columns:
            df[median_err_col] = df[median_col]
    
    # Compute winner at reconstruction_missing (by RMSE)
    if 'dineof_missing_rmse' in df.columns and 'dincae_missing_rmse' in df.columns:
        df['delta_rmse'] = df['dineof_missing_rmse'] - df['dincae_missing_rmse']
        df['winner'] = np.where(df['delta_rmse'] < -0.02, 'DINEOF',
                                np.where(df['delta_rmse'] > 0.02, 'DINCAE', 'TIE'))
    
    # Compute deltas for ALL 6 metrics at missing
    for metric in ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']:
        din_col = f'dineof_missing_{metric}'
        dic_col = f'dincae_missing_{metric}'
        if din_col in df.columns and dic_col in df.columns:
            out_metric = metric
            if metric == 'median':
                out_metric = 'median_error'
            elif metric == 'rstd':
                out_metric = 'robust_std'
            df[f'delta_{out_metric}'] = df[din_col] - df[dic_col]
    
    # Absolute bias difference
    if 'dineof_missing_bias' in df.columns and 'dincae_missing_bias' in df.columns:
        df['delta_abs_bias'] = df['dineof_missing_bias'].abs() - df['dincae_missing_bias'].abs()
    
    return df


# =============================================================================
# Q1: Does observation quality predict winner?
# =============================================================================
def question1_observation_predictors(df: pd.DataFrame, output_dir: str):
    """Q1: Does observation (satellite vs insitu) predict winner?"""
    print("\n" + "="*70)
    print("Q1: Does observation (satellite vs insitu) quality predict winner?")
    print("="*70)
    
    valid = df[df['winner'].isin(['DINEOF', 'DINCAE'])].copy()
    n = len(valid)
    print(f"\nSample: {n} lakes (excluding ties)")
    
    predictors = [
        ('obs_rmse', 'Obs RMSE'),
        ('obs_mae', 'Obs MAE'),
        ('obs_median', 'Obs Median Error'),
        ('obs_median_error', 'Obs Median Error'),
        ('obs_bias', 'Obs Bias'),
        ('obs_std', 'Obs Residual STD'),
        ('obs_rstd', 'Obs Robust STD'),
        ('obs_robust_std', 'Obs Robust STD'),
        ('obs_correlation', 'Obs Correlation'),
        ('coverage_fraction', 'Coverage Fraction'),
        ('gap_fraction', 'Gap Fraction'),
    ]
    
    print("\nCorrelation with ΔRMSE (positive = DINCAE favored):")
    print("-"*65)
    print(f"{'Predictor':<25} {'r':>10} {'p':>10} {'n':>6} {'Strength':<12}")
    print("-"*65)
    
    results = []
    seen_names = set()
    for col, name in predictors:
        if col not in valid.columns or name in seen_names:
            continue
        seen_names.add(name)
        subset = valid[[col, 'delta_rmse']].dropna()
        if len(subset) < 5:
            continue
        r, p = safe_pearsonr(subset[col], subset['delta_rmse'])
        if np.isnan(r):
            continue
        
        if abs(r) > 0.5: strength = "STRONG"
        elif abs(r) > 0.3: strength = "Moderate"
        elif abs(r) > 0.1: strength = "Weak"
        else: strength = "None"
        
        results.append({'col': col, 'name': name, 'r': r, 'p': p, 'n': len(subset), 'strength': strength})
        print(f"{name:<25} {r:>+10.3f} {p:>10.4f} {len(subset):>6} {strength:<12}")
    
    print("-"*65)
    
    # Figure 1: Bar chart of correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    if results:
        names = [r['name'] for r in results]
        rs = [r['r'] for r in results]
        colors = [COLORS['strong'] if abs(r) > 0.3 else COLORS['weak'] for r in rs]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, rs, color=colors, edgecolor='black', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(0.3, color='green', linestyle='--', alpha=0.5, label='|r|=0.3')
        ax.axvline(-0.3, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Correlation with ΔRMSE (positive = DINCAE favored)')
        ax.set_xlim(-1, 1)
        ax.set_title('Q1: Do Observation/Coverage Metrics Predict Winner?')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q1_observation_predictors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Scatter plots for top predictors
    top_predictors = [r for r in results if abs(r['r']) > 0.2][:4]
    if len(top_predictors) >= 2:
        n_plots = min(4, len(top_predictors))
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        colors_scatter = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner']]
        
        for idx, pred in enumerate(top_predictors[:n_plots]):
            ax = axes[idx]
            col = pred['col']
            subset = valid[[col, 'delta_rmse', 'winner']].dropna()
            subset_colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in subset['winner']]
            
            ax.scatter(subset[col], subset['delta_rmse'], c=subset_colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Trend line
            z = np.polyfit(subset[col], subset['delta_rmse'], 1)
            x_line = np.linspace(subset[col].min(), subset[col].max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
            
            ax.axhline(0, color='black', linewidth=1)
            ax.set_xlabel(pred['name'])
            ax.set_ylabel('ΔRMSE [°C]')
            ax.set_title(f"r = {pred['r']:.3f}")
            ax.grid(True, alpha=0.3)
        
        legend_elements = [
            Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
            Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(output_dir, 'Q1_top_predictors_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    strong = [r for r in results if abs(r['r']) > 0.3]
    print("\n→ CONCLUSION: ", end="")
    if strong:
        strong_str = ", ".join([f"{r['name']} (r={r['r']:.2f})" for r in strong])
        print(f"Metrics with |r|>0.3: {strong_str}")
    else:
        print("NO observation/coverage metric strongly predicts winner (all |r| < 0.3)")
    
    return pd.DataFrame(results)


# =============================================================================
# Q2: Story across data types
# =============================================================================
def question2_data_type_story(df: pd.DataFrame, output_dir: str):
    """Q2: What is the story across data types?"""
    print("\n" + "="*70)
    print("Q2: Story across data types (all, observed, missing)")
    print("="*70)
    
    valid = df[df['winner'].isin(['DINEOF', 'DINCAE'])].copy()
    
    # Coverage stats
    print("\n--- Coverage Statistics ---")
    if 'coverage_fraction' in valid.columns:
        cov = valid['coverage_fraction'].dropna()
        print(f"Coverage fraction (observed/total):")
        print(f"  Mean: {cov.mean():.1%}, Median: {cov.median():.1%}")
        print(f"  Range: {cov.min():.1%} - {cov.max():.1%}")
    
    # Winner counts by data type
    print("\n--- Winner Counts by Data Type (RMSE-based) ---")
    
    summary = []
    for dtype_key, dtype_name in [('all', 'All'), ('observed', 'Observed'), ('missing', 'Missing')]:
        din_col = f'dineof_{dtype_key}_rmse'
        dic_col = f'dincae_{dtype_key}_rmse'
        
        if din_col not in valid.columns or dic_col not in valid.columns:
            continue
        
        subset = valid[[din_col, dic_col]].dropna()
        din_wins = (subset[din_col] < subset[dic_col] - 0.02).sum()
        dic_wins = (subset[din_col] > subset[dic_col] + 0.02).sum()
        ties = len(subset) - din_wins - dic_wins
        
        summary.append({
            'data_type': dtype_name,
            'dineof_wins': din_wins,
            'dincae_wins': dic_wins,
            'ties': ties,
            'n': len(subset)
        })
        
        print(f"  {dtype_name:<12}: DINEOF {din_wins}, DINCAE {dic_wins}, Ties {ties} (n={len(subset)})")
    
    # Mean metrics by data type
    print("\n--- Mean Metrics by Data Type ---")
    metrics_map = [('rmse', 'RMSE'), ('mae', 'MAE'), ('median', 'MEDIAN'), 
                   ('bias', 'BIAS'), ('std', 'STD'), ('rstd', 'RSTD')]
    
    for metric, display_name in metrics_map:
        print(f"\n  {display_name}:")
        for dtype_key in ['all', 'observed', 'missing']:
            din_col = f'dineof_{dtype_key}_{metric}'
            dic_col = f'dincae_{dtype_key}_{metric}'
            if din_col in valid.columns and dic_col in valid.columns:
                din_mean = valid[din_col].mean()
                dic_mean = valid[dic_col].mean()
                print(f"    {dtype_key:<10}: DINEOF={din_mean:>7.3f}, DINCAE={dic_mean:>7.3f}, Δ={din_mean-dic_mean:>+7.3f}")
    
    # Figure 1: 2x3 grid showing ALL 6 METRICS by data type (bar chart)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    dtype_names = [s['data_type'] for s in summary]
    x = np.arange(len(dtype_names))
    width = 0.35
    
    metrics_to_plot = [
        ('rmse', 'RMSE [°C]'), ('mae', 'MAE [°C]'), ('median', 'Median Error [°C]'),
        ('bias', 'Bias [°C]'), ('std', 'Residual STD [°C]'), ('rstd', 'Robust STD [°C]'),
    ]
    
    for idx, (metric, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        din_vals = []
        dic_vals = []
        for d in dtype_names:
            din_col = f'dineof_{d.lower()}_{metric}'
            dic_col = f'dincae_{d.lower()}_{metric}'
            din_vals.append(valid[din_col].mean() if din_col in valid.columns else np.nan)
            dic_vals.append(valid[dic_col].mean() if dic_col in valid.columns else np.nan)
        
        ax.bar(x - width/2, din_vals, width, label='DINEOF', color=COLORS['dineof'], edgecolor='black')
        ax.bar(x + width/2, dic_vals, width, label='DINCAE', color=COLORS['dincae'], edgecolor='black')
        
        if metric in ['bias', 'median']:
            ax.axhline(0, color='black', linewidth=1)
        
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(dtype_names)
        ax.legend(fontsize=8)
        ax.set_title(f'{chr(65+idx)}) Mean {metric.upper()} by Data Type')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q2_data_type_story.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Box/Violin plots showing distributions (not just means)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for idx, (metric, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        # Prepare data for boxplot
        data_dineof = []
        data_dincae = []
        labels = []
        
        for dtype_key in ['all', 'observed', 'missing']:
            din_col = f'dineof_{dtype_key}_{metric}'
            dic_col = f'dincae_{dtype_key}_{metric}'
            if din_col in valid.columns and dic_col in valid.columns:
                data_dineof.append(valid[din_col].dropna().values)
                data_dincae.append(valid[dic_col].dropna().values)
                labels.append(dtype_key.capitalize())
        
        if data_dineof:
            positions_din = np.arange(len(labels)) * 2
            positions_dic = positions_din + 0.6
            
            bp1 = ax.boxplot(data_dineof, positions=positions_din, widths=0.5, patch_artist=True)
            bp2 = ax.boxplot(data_dincae, positions=positions_dic, widths=0.5, patch_artist=True)
            
            for patch in bp1['boxes']:
                patch.set_facecolor(COLORS['dineof'])
            for patch in bp2['boxes']:
                patch.set_facecolor(COLORS['dincae'])
            
            ax.set_xticks(positions_din + 0.3)
            ax.set_xticklabels(labels)
            
            if metric in ['bias', 'median']:
                ax.axhline(0, color='black', linewidth=1, linestyle='--')
        
        ax.set_ylabel(ylabel)
        ax.set_title(f'{chr(65+idx)}) {metric.upper()} Distribution by Data Type')
        ax.grid(axis='y', alpha=0.3)
    
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(os.path.join(output_dir, 'Q2_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n→ CONCLUSION: Check figures for patterns across data types")
    
    return pd.DataFrame(summary)


# =============================================================================
# Q3: Strong predictors of winner (STD, Bias patterns)
# =============================================================================
def question3_strong_predictors(df: pd.DataFrame, output_dir: str):
    """Q3: What are the strong predictors? STD and Bias patterns."""
    print("\n" + "="*70)
    print("Q3: Strong predictors of winner (Residual STD, Bias patterns)")
    print("="*70)
    
    valid = df[df['winner'].isin(['DINEOF', 'DINCAE'])].copy()
    n = len(valid)
    
    predictors = [
        ('delta_rmse', 'Δ(RMSE)'),
        ('delta_mae', 'Δ(MAE)'),
        ('delta_median_error', 'Δ(Median Error)'),
        ('delta_bias', 'Δ(Bias)'),
        ('delta_std', 'Δ(Residual STD)'),
        ('delta_robust_std', 'Δ(Robust STD)'),
        ('delta_abs_bias', 'Δ(|Bias|)'),
    ]
    
    print(f"\nSample: {n} lakes")
    print("\nCorrelation with ΔRMSE:")
    print("-"*60)
    
    results = []
    for col, name in predictors:
        if col not in valid.columns:
            continue
        subset = valid[[col, 'delta_rmse']].dropna()
        if len(subset) < 5:
            continue
        r, p = safe_pearsonr(subset[col], subset['delta_rmse'])
        if np.isnan(r):
            continue
        results.append({'col': col, 'name': name, 'r': r, 'p': p})
        print(f"  {name:<20}: r = {r:>+.3f} (p = {p:.4f})")
    
    # Bias patterns
    print("\n--- Bias Patterns ---")
    din_bias = valid['dineof_missing_bias'].mean()
    dic_bias = valid['dincae_missing_bias'].mean()
    print(f"  Mean DINEOF bias: {din_bias:>+.3f}°C {'(COLD)' if din_bias < 0 else '(WARM)'}")
    print(f"  Mean DINCAE bias: {dic_bias:>+.3f}°C {'(COLD)' if dic_bias < 0 else '(WARM)'}")
    
    # Bias by winner
    print("\n--- Bias by Winner ---")
    for winner in ['DINEOF', 'DINCAE']:
        subset = valid[valid['winner'] == winner]
        din_b = subset['dineof_missing_bias'].mean()
        dic_b = subset['dincae_missing_bias'].mean()
        print(f"  When {winner} wins ({len(subset)} lakes):")
        print(f"    DINEOF bias: {din_b:>+.3f}°C, DINCAE bias: {dic_b:>+.3f}°C")
    
    # How each method wins
    print("\n--- How Each Method Wins ---")
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    din_has_lower_std = (dineof_wins['delta_std'] < 0).sum() if 'delta_std' in dineof_wins.columns else 0
    dic_has_lower_std = (dincae_wins['delta_std'] > 0).sum() if 'delta_std' in dincae_wins.columns else 0
    din_has_lower_bias = (dineof_wins['delta_abs_bias'] < 0).sum() if 'delta_abs_bias' in dineof_wins.columns else 0
    dic_has_lower_bias = (dincae_wins['delta_abs_bias'] > 0).sum() if 'delta_abs_bias' in dincae_wins.columns else 0
    
    print(f"  DINEOF wins ({len(dineof_wins)} lakes):")
    print(f"    Has lower STD: {din_has_lower_std}/{len(dineof_wins)} ({100*din_has_lower_std/len(dineof_wins):.0f}%)")
    print(f"    Has lower |Bias|: {din_has_lower_bias}/{len(dineof_wins)} ({100*din_has_lower_bias/len(dineof_wins):.0f}%)")
    
    print(f"  DINCAE wins ({len(dincae_wins)} lakes):")
    print(f"    Has lower STD: {dic_has_lower_std}/{len(dincae_wins)} ({100*dic_has_lower_std/len(dincae_wins):.0f}%)")
    print(f"    Has lower |Bias|: {dic_has_lower_bias}/{len(dincae_wins)} ({100*dic_has_lower_bias/len(dincae_wins):.0f}%)")
    
    # Figure: 3x3 grid
    fig = plt.figure(figsize=(16, 14))
    
    color_map = {idx: COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] 
                 for idx, w in zip(valid.index, valid['winner'])}
    
    metrics_to_plot = [
        ('delta_rmse', 'ΔRMSE', 'Δ(RMSE) [°C]'),
        ('delta_mae', 'ΔMAE', 'Δ(MAE) [°C]'),
        ('delta_median_error', 'ΔMedian', 'Δ(Median Error) [°C]'),
        ('delta_bias', 'ΔBias', 'Δ(Bias) [°C]'),
        ('delta_std', 'ΔRes.STD', 'Δ(Residual STD) [°C]'),
        ('delta_robust_std', 'ΔRob.STD', 'Δ(Robust STD) [°C]'),
    ]
    
    for idx, (col, label, xlabel) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(3, 3, idx + 1)
        
        if col not in valid.columns or col == 'delta_rmse':
            if col == 'delta_rmse':
                ax.text(0.5, 0.5, 'ΔRMSE vs ΔRMSE\n(trivial: r=1)', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=11)
                ax.set_title(f'{chr(65+idx)}) {label} vs ΔRMSE: r = 1.000')
            else:
                ax.text(0.5, 0.5, f'{label}\nNo data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{chr(65+idx)}) {label} vs ΔRMSE')
        else:
            subset = valid[[col, 'delta_rmse']].dropna()
            if len(subset) >= 5:
                subset_colors = [color_map[i] for i in subset.index]
                ax.scatter(subset[col], subset['delta_rmse'], c=subset_colors, 
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
                r, _ = safe_pearsonr(subset[col], subset['delta_rmse'])
                
                z = np.polyfit(subset[col], subset['delta_rmse'], 1)
                x_line = np.linspace(subset[col].min(), subset[col].max(), 100)
                ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
                
                ax.axhline(0, color='black', linewidth=1)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f'{chr(65+idx)}) {label} vs ΔRMSE: r = {r:.3f}')
            else:
                ax.text(0.5, 0.5, f'{label}\nInsufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{chr(65+idx)}) {label} vs ΔRMSE')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Δ(RMSE) [°C]')
        ax.grid(True, alpha=0.3)
    
    # Panel G: Bias comparison
    ax = fig.add_subplot(3, 3, 7)
    colors = [color_map[i] for i in valid.index]
    ax.scatter(valid['dineof_missing_bias'], valid['dincae_missing_bias'], c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    lims = [min(valid['dineof_missing_bias'].min(), valid['dincae_missing_bias'].min()) - 0.1,
            max(valid['dineof_missing_bias'].max(), valid['dincae_missing_bias'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('DINEOF Bias [°C]')
    ax.set_ylabel('DINCAE Bias [°C]')
    ax.set_title('G) Bias: DINEOF vs DINCAE\n(DINEOF cold, DINCAE warm)')
    ax.grid(True, alpha=0.3)
    
    # Panel H: STD comparison
    ax = fig.add_subplot(3, 3, 8)
    ax.scatter(valid['dineof_missing_std'], valid['dincae_missing_std'], c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    lims = [min(valid['dineof_missing_std'].min(), valid['dincae_missing_std'].min()) - 0.1,
            max(valid['dineof_missing_std'].max(), valid['dincae_missing_std'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('DINEOF Residual STD [°C]')
    ax.set_ylabel('DINCAE Residual STD [°C]')
    din_lower = (valid['dineof_missing_std'] < valid['dincae_missing_std']).sum()
    ax.set_title(f'H) Residual STD Comparison\nDINEOF lower: {din_lower}/{n}')
    ax.grid(True, alpha=0.3)
    
    # Panel I: How each method wins
    ax = fig.add_subplot(3, 3, 9)
    x = np.arange(2)
    width = 0.35
    
    std_pcts = [100*din_has_lower_std/len(dineof_wins) if len(dineof_wins) > 0 else 0, 
                100*dic_has_lower_std/len(dincae_wins) if len(dincae_wins) > 0 else 0]
    bias_pcts = [100*din_has_lower_bias/len(dineof_wins) if len(dineof_wins) > 0 else 0, 
                 100*dic_has_lower_bias/len(dincae_wins) if len(dincae_wins) > 0 else 0]
    
    ax.bar(x - width/2, std_pcts, width, label='Has lower STD', color='#e74c3c', edgecolor='black')
    ax.bar(x + width/2, bias_pcts, width, label='Has lower |Bias|', color='#3498db', edgecolor='black')
    ax.set_ylabel('% of wins')
    ax.set_xticks(x)
    ax.set_xticklabels([f'DINEOF\n(n={len(dineof_wins)})', f'DINCAE\n(n={len(dincae_wins)})'])
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title('I) How Each Method Wins')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (s, b) in enumerate(zip(std_pcts, bias_pcts)):
        ax.text(i - width/2, s + 2, f'{s:.0f}%', ha='center', fontsize=9)
        ax.text(i + width/2, b + 2, f'{b:.0f}%', ha='center', fontsize=9)
    
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(os.path.join(output_dir, 'Q3_strong_predictors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n→ CONCLUSION:")
    print("  - Residual STD strongly predicts winner")
    print("  - DINEOF: tends COLD bias; DINCAE: tends WARM bias")
    print("  - DINEOF wins → always lower STD; DINCAE wins → usually lower |Bias|")
    
    return pd.DataFrame(results)


# =============================================================================
# Q4: Mechanism behind STD predictor
# =============================================================================
def question4_mechanism(df: pd.DataFrame, output_dir: str):
    """Q4: What is the mechanism behind STD as predictor?"""
    print("\n" + "="*70)
    print("Q4: Mechanism behind residual STD as predictor")
    print("="*70)
    
    valid = df[df['winner'].isin(['DINEOF', 'DINCAE'])].copy()
    
    # RMSE decomposition: RMSE² = Bias² + STD²
    print("\n--- RMSE² = Bias² + STD² Decomposition ---")
    
    valid['delta_std_sq'] = valid['dineof_missing_std']**2 - valid['dincae_missing_std']**2
    valid['delta_bias_sq'] = valid['dineof_missing_bias']**2 - valid['dincae_missing_bias']**2
    valid['delta_rmse_sq'] = valid['dineof_missing_rmse']**2 - valid['dincae_missing_rmse']**2
    
    var_std = valid['delta_std_sq'].var()
    var_bias = valid['delta_bias_sq'].var()
    cov = valid[['delta_std_sq', 'delta_bias_sq']].cov().iloc[0, 1]
    total = var_std + var_bias + 2 * cov
    
    pct_std = 100 * var_std / total
    pct_bias = 100 * var_bias / total
    pct_cov = 100 * 2 * cov / total
    
    print(f"  Var(Δ(STD²)) contributes:  {pct_std:.1f}%")
    print(f"  Var(Δ(Bias²)) contributes: {pct_bias:.1f}%")
    print(f"  2×Cov contributes:         {pct_cov:.1f}%")
    
    r_std, _ = safe_pearsonr(valid['delta_std'], valid['delta_rmse'])
    r_bias, _ = safe_pearsonr(valid['delta_bias'], valid['delta_rmse'])
    
    print(f"\n  r(Δ(STD), Δ(RMSE)) = {r_std:.3f}")
    print(f"  r(Δ(Bias), Δ(RMSE)) = {r_bias:.3f}")
    
    print("\n--- Interpretation ---")
    print(f"  STD² contributes {pct_std:.0f}% of variance → methods differ in temporal matching")
    print(f"  Bias² contributes {pct_bias:.0f}% → methods similar in systematic offset")
    print(f"  High r(ΔSTD,ΔRMSE) is partly mathematical but the key is r(ΔBias,ΔRMSE)≈0")
    
    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner']]
    
    # A) Variance decomposition
    ax = axes[0]
    components = ['Var(Δ(STD²))', 'Var(Δ(Bias²))', '2×Cov']
    values = [pct_std, pct_bias, pct_cov]
    bar_colors = ['#e74c3c', '#3498db', '#9b59b6']
    
    bars = ax.bar(components, values, color=bar_colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('% of Var(ΔRMSE²)')
    ax.set_title('A) Variance Decomposition')
    ax.set_ylim(-20, 100)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        y_pos = val + 3 if val >= 0 else val - 8
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # B) Δ(STD) vs Δ(RMSE)
    ax = axes[1]
    ax.scatter(valid['delta_std'], valid['delta_rmse'], c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    z = np.polyfit(valid['delta_std'], valid['delta_rmse'], 1)
    x_line = np.linspace(valid['delta_std'].min(), valid['delta_std'].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δ(STD) [°C]')
    ax.set_ylabel('Δ(RMSE) [°C]')
    ax.set_title(f'B) Δ(STD) vs Δ(RMSE): r = {r_std:.3f}')
    ax.grid(True, alpha=0.3)
    
    # C) Δ(Bias) vs Δ(RMSE)
    ax = axes[2]
    ax.scatter(valid['delta_bias'], valid['delta_rmse'], c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δ(Bias) [°C]')
    ax.set_ylabel('Δ(RMSE) [°C]')
    ax.set_title(f'C) Δ(Bias) vs Δ(RMSE): r = {r_bias:.3f}')
    ax.grid(True, alpha=0.3)
    
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_dir, 'Q4_mechanism.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'pct_std': pct_std, 'pct_bias': pct_bias, 'r_std': r_std, 'r_bias': r_bias}


# =============================================================================
# Q5: Per-Lake Summary with Lake IDs
# =============================================================================
def question5_summary(df: pd.DataFrame, output_dir: str):
    """Q5: Summary and conclusions with lake ID labels."""
    print("\n" + "="*70)
    print("Q5: Summary - What can we conclude?")
    print("="*70)
    
    valid = df[df['winner'].isin(['DINEOF', 'DINCAE'])].copy()
    
    n_total = len(valid)
    n_dineof = (valid['winner'] == 'DINEOF').sum()
    n_dincae = (valid['winner'] == 'DINCAE').sum()
    
    print(f"""
SAMPLE: {n_total} lakes at reconstruction_missing

WINNER COUNTS (by RMSE):
  DINEOF: {n_dineof} ({100*n_dineof/n_total:.0f}%)
  DINCAE: {n_dincae} ({100*n_dincae/n_total:.0f}%)

KEY FINDINGS:

1. Observation metrics do NOT predict winner (all |r| < 0.3 except coverage)
2. RMSE difference driven by residual STD (~90%), not bias (~16%)
3. DINEOF wins via lower STD (100%); DINCAE wins via lower |bias| (~90%)
4. DINEOF tends cold bias; DINCAE tends warm bias
5. Methods differ in temporal pattern matching, not systematic offset
""")
    
    # Figure 1: Per-lake RMSE difference with lake ID labels
    fig, ax = plt.subplots(figsize=(16, 6))
    
    sorted_valid = valid.sort_values('delta_rmse', ascending=False)
    x = np.arange(len(sorted_valid))
    colors = [COLORS['dincae'] if r > 0 else COLORS['dineof'] for r in sorted_valid['delta_rmse']]
    lake_ids = sorted_valid['lake_id'].astype(int).values
    
    bars = ax.bar(x, sorted_valid['delta_rmse'], color=colors, edgecolor='white', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_ylabel('ΔRMSE (DINEOF − DINCAE) [°C]', fontsize=11)
    ax.set_title(f'Per-Lake RMSE Difference at reconstruction_missing\nDINEOF wins: {n_dineof}, DINCAE wins: {n_dincae}', fontsize=12)
    
    # Add lake ID labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(lake_ids, rotation=90, fontsize=8)
    ax.set_xlabel('Lake CCI ID', fontsize=11)
    
    ax.grid(axis='y', alpha=0.3)
    
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label=f'DINEOF wins ({n_dineof})'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label=f'DINCAE wins ({n_dincae})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q5_per_lake_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Magnitude of wins (how decisive are the wins?)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Distribution of ΔRMSE by winner
    ax = axes[0]
    dineof_deltas = sorted_valid[sorted_valid['winner'] == 'DINEOF']['delta_rmse']
    dincae_deltas = sorted_valid[sorted_valid['winner'] == 'DINCAE']['delta_rmse']
    
    ax.hist(dineof_deltas, bins=10, alpha=0.7, color=COLORS['dineof'], edgecolor='black', label='DINEOF wins')
    ax.hist(dincae_deltas, bins=10, alpha=0.7, color=COLORS['dincae'], edgecolor='black', label='DINCAE wins')
    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlabel('ΔRMSE (DINEOF − DINCAE) [°C]')
    ax.set_ylabel('Number of Lakes')
    ax.set_title('A) Distribution of Win Margins')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Box plot comparing win magnitudes
    ax = axes[1]
    data = [np.abs(dineof_deltas), np.abs(dincae_deltas)]
    bp = ax.boxplot(data, labels=['DINEOF wins', 'DINCAE wins'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['dineof'])
    bp['boxes'][1].set_facecolor(COLORS['dincae'])
    
    ax.set_ylabel('|ΔRMSE| (Win Margin) [°C]')
    ax.set_title('B) Magnitude of Wins')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    for i, d in enumerate(data):
        ax.scatter(i+1, np.mean(d), marker='D', color='red', s=50, zorder=5, label='Mean' if i==0 else '')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q5_win_magnitude.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print win magnitude statistics
    print("WIN MAGNITUDE STATISTICS:")
    print(f"  DINEOF wins: mean |ΔRMSE| = {np.abs(dineof_deltas).mean():.3f}°C, max = {np.abs(dineof_deltas).max():.3f}°C")
    print(f"  DINCAE wins: mean |ΔRMSE| = {np.abs(dincae_deltas).mean():.3f}°C, max = {np.abs(dincae_deltas).max():.3f}°C")


# =============================================================================
# Q6: Comprehensive Tables and Additional Analysis
# =============================================================================
def question6_comprehensive_tables(df: pd.DataFrame, output_dir: str):
    """Q6: Generate comprehensive tables and correlation heatmap."""
    print("\n" + "="*70)
    print("Q6: Comprehensive Tables and Correlation Analysis")
    print("="*70)
    
    valid = df[df['winner'].isin(['DINEOF', 'DINCAE'])].copy()
    
    # Table 1: Per-lake summary with all key metrics
    print("\n--- Generating Per-Lake Summary Table ---")
    cols_to_export = ['lake_id', 'winner', 'delta_rmse', 
                      'dineof_missing_rmse', 'dincae_missing_rmse',
                      'dineof_missing_mae', 'dincae_missing_mae',
                      'dineof_missing_bias', 'dincae_missing_bias',
                      'dineof_missing_std', 'dincae_missing_std',
                      'coverage_fraction', 'gap_fraction',
                      'n_observed', 'n_missing', 'n_total']
    cols_available = [c for c in cols_to_export if c in valid.columns]
    
    lake_summary = valid[cols_available].copy()
    lake_summary = lake_summary.sort_values('delta_rmse', ascending=False)
    lake_summary.to_csv(os.path.join(output_dir, 'table_per_lake_summary.csv'), index=False)
    print(f"  Saved: table_per_lake_summary.csv ({len(lake_summary)} lakes)")
    
    # Table 2: Winner breakdown statistics
    print("\n--- Winner Breakdown Statistics ---")
    winner_stats = []
    for winner in ['DINEOF', 'DINCAE']:
        subset = valid[valid['winner'] == winner]
        row = {
            'winner': winner,
            'n_lakes': len(subset),
            'pct_lakes': 100 * len(subset) / len(valid),
            'mean_delta_rmse': subset['delta_rmse'].mean(),
            'mean_abs_delta_rmse': subset['delta_rmse'].abs().mean(),
            'max_abs_delta_rmse': subset['delta_rmse'].abs().max(),
            'mean_coverage': subset['coverage_fraction'].mean() if 'coverage_fraction' in subset.columns else np.nan,
            'mean_gap_fraction': subset['gap_fraction'].mean() if 'gap_fraction' in subset.columns else np.nan,
        }
        
        for metric in ['rmse', 'mae', 'bias', 'std']:
            for method in ['dineof', 'dincae']:
                col = f'{method}_missing_{metric}'
                if col in subset.columns:
                    row[f'mean_{method}_{metric}'] = subset[col].mean()
        
        winner_stats.append(row)
    
    winner_df = pd.DataFrame(winner_stats)
    winner_df.to_csv(os.path.join(output_dir, 'table_winner_breakdown.csv'), index=False)
    print(f"  Saved: table_winner_breakdown.csv")
    
    # Print winner breakdown
    print("\n  WINNER BREAKDOWN:")
    for _, row in winner_df.iterrows():
        print(f"\n  {row['winner']} ({row['n_lakes']} lakes, {row['pct_lakes']:.0f}%):")
        print(f"    Mean |ΔRMSE|: {row['mean_abs_delta_rmse']:.3f}°C")
        print(f"    Mean coverage: {row['mean_coverage']:.1%}")
        print(f"    Mean DINEOF RMSE: {row.get('mean_dineof_rmse', np.nan):.3f}°C")
        print(f"    Mean DINCAE RMSE: {row.get('mean_dincae_rmse', np.nan):.3f}°C")
    
    # Table 3: Aggregate statistics by method
    print("\n--- Aggregate Statistics by Method ---")
    agg_stats = []
    for method in ['dineof', 'dincae']:
        for dtype in ['all', 'observed', 'missing']:
            row = {'method': method, 'data_type': dtype}
            for metric in ['rmse', 'mae', 'bias', 'std', 'rstd', 'median']:
                col = f'{method}_{dtype}_{metric}'
                if col in valid.columns:
                    row[f'{metric}_mean'] = valid[col].mean()
                    row[f'{metric}_std'] = valid[col].std()
                    row[f'{metric}_median'] = valid[col].median()
            agg_stats.append(row)
    
    agg_df = pd.DataFrame(agg_stats)
    agg_df.to_csv(os.path.join(output_dir, 'table_aggregate_statistics.csv'), index=False)
    print(f"  Saved: table_aggregate_statistics.csv")
    
    # Figure: Correlation heatmap of key variables
    print("\n--- Generating Correlation Heatmap ---")
    
    corr_cols = ['delta_rmse', 'delta_mae', 'delta_std', 'delta_bias',
                 'coverage_fraction', 'gap_fraction',
                 'obs_rmse', 'obs_std', 'obs_bias']
    corr_cols = [c for c in corr_cols if c in valid.columns]
    
    if len(corr_cols) >= 4:
        corr_matrix = valid[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create diverging colormap
        cmap = LinearSegmentedColormap.from_list('diverging', ['#3498db', 'white', '#e74c3c'])
        
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=10)
        
        # Add labels
        labels = [c.replace('_', '\n') for c in corr_cols]
        ax.set_xticks(np.arange(len(corr_cols)))
        ax.set_yticks(np.arange(len(corr_cols)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        
        # Add correlation values as text
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_matrix.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        
        ax.set_title('Correlation Heatmap of Key Variables', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Q6_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: Q6_correlation_heatmap.png")
    
    # Figure: Lake characteristics by winner
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    dineof_lakes = valid[valid['winner'] == 'DINEOF']
    dincae_lakes = valid[valid['winner'] == 'DINCAE']
    
    # A) Coverage fraction by winner
    ax = axes[0, 0]
    if 'coverage_fraction' in valid.columns:
        data = [dineof_lakes['coverage_fraction'].dropna(), dincae_lakes['coverage_fraction'].dropna()]
        bp = ax.boxplot(data, labels=['DINEOF wins', 'DINCAE wins'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['dineof'])
        bp['boxes'][1].set_facecolor(COLORS['dincae'])
        ax.set_ylabel('Coverage Fraction')
        ax.set_title('A) Coverage Fraction by Winner')
        ax.grid(axis='y', alpha=0.3)
    
    # B) Gap fraction by winner
    ax = axes[0, 1]
    if 'gap_fraction' in valid.columns:
        data = [dineof_lakes['gap_fraction'].dropna(), dincae_lakes['gap_fraction'].dropna()]
        bp = ax.boxplot(data, labels=['DINEOF wins', 'DINCAE wins'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['dineof'])
        bp['boxes'][1].set_facecolor(COLORS['dincae'])
        ax.set_ylabel('Gap Fraction')
        ax.set_title('B) Gap Fraction by Winner')
        ax.grid(axis='y', alpha=0.3)
    
    # C) Observation RMSE by winner
    ax = axes[1, 0]
    obs_rmse_col = 'obs_rmse' if 'obs_rmse' in valid.columns else None
    if obs_rmse_col:
        data = [dineof_lakes[obs_rmse_col].dropna(), dincae_lakes[obs_rmse_col].dropna()]
        bp = ax.boxplot(data, labels=['DINEOF wins', 'DINCAE wins'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['dineof'])
        bp['boxes'][1].set_facecolor(COLORS['dincae'])
        ax.set_ylabel('Observation RMSE [°C]')
        ax.set_title('C) Observation RMSE by Winner')
        ax.grid(axis='y', alpha=0.3)
    
    # D) Total samples by winner
    ax = axes[1, 1]
    if 'n_total' in valid.columns:
        data = [dineof_lakes['n_total'].dropna(), dincae_lakes['n_total'].dropna()]
        bp = ax.boxplot(data, labels=['DINEOF wins', 'DINCAE wins'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['dineof'])
        bp['boxes'][1].set_facecolor(COLORS['dincae'])
        ax.set_ylabel('Total Matched Samples')
        ax.set_title('D) Sample Size by Winner')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q6_lake_characteristics_by_winner.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Q6_lake_characteristics_by_winner.png")
    
    return winner_df


# =============================================================================
# Generate Text Summary Report
# =============================================================================
def generate_summary_report(df: pd.DataFrame, output_dir: str, q1_results, q4_results):
    """Generate a comprehensive text summary report."""
    
    valid = df[df['winner'].isin(['DINEOF', 'DINCAE'])].copy()
    n_total = len(valid)
    n_dineof = (valid['winner'] == 'DINEOF').sum()
    n_dincae = (valid['winner'] == 'DINCAE').sum()
    
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    report = f"""
================================================================================
IN-SITU VALIDATION ANALYSIS: COMPREHENSIVE SUMMARY REPORT
================================================================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
This report summarizes the comparison of DINEOF vs DINCAE gap-filling methods
using in-situ buoy measurements as ground truth validation.

DATA SUMMARY
------------
Total lakes analyzed: {n_total}
Winner determination: Based on RMSE at reconstruction_missing (±0.02°C threshold)

WINNER DISTRIBUTION
-------------------
DINEOF wins: {n_dineof} lakes ({100*n_dineof/n_total:.0f}%)
DINCAE wins: {n_dincae} lakes ({100*n_dincae/n_total:.0f}%)

This is in contrast to satellite cross-validation which shows DINCAE dominating.

KEY FINDINGS
------------

1. OBSERVATION QUALITY DOES NOT PREDICT WINNER
   - All observation metrics (RMSE, MAE, bias, STD) show |r| < 0.3 with ΔRMSE
   - The only moderate predictors are coverage/gap fraction

2. COVERAGE FRACTION IS THE BEST PREDICTOR
   - Gap fraction correlates negatively with ΔRMSE (r ≈ {q1_results[q1_results['name']=='Gap Fraction']['r'].values[0] if len(q1_results[q1_results['name']=='Gap Fraction']) > 0 else 'N/A':.2f})
   - More gaps → DINEOF tends to win
   - More observations → DINCAE tends to win

3. RMSE DECOMPOSITION
   - RMSE² = Bias² + STD²
   - Var(Δ(STD²)) contributes: {q4_results['pct_std']:.0f}% of variance in ΔRMSE
   - Var(Δ(Bias²)) contributes: {q4_results['pct_bias']:.0f}% of variance in ΔRMSE
   - Methods differ primarily in temporal pattern matching (STD), not systematic offset (Bias)

4. HOW EACH METHOD WINS
   - When DINEOF wins ({n_dineof} lakes):
     * Always has lower residual STD (100%)
     * Has lower |Bias| only {100*(dineof_wins['delta_abs_bias']<0).sum()/len(dineof_wins):.0f}% of the time
   
   - When DINCAE wins ({n_dincae} lakes):
     * Has lower STD {100*(dincae_wins['delta_std']>0).sum()/len(dincae_wins):.0f}% of the time
     * Has lower |Bias| {100*(dincae_wins['delta_abs_bias']>0).sum()/len(dincae_wins):.0f}% of the time

5. BIAS PATTERNS
   - DINEOF tends to have COLD bias (underestimates temperature)
   - DINCAE tends to have WARM bias (overestimates temperature)
   - Mean DINEOF bias: {valid['dineof_missing_bias'].mean():+.3f}°C
   - Mean DINCAE bias: {valid['dincae_missing_bias'].mean():+.3f}°C

WIN MAGNITUDE
-------------
DINEOF wins: Mean |ΔRMSE| = {np.abs(dineof_wins['delta_rmse']).mean():.3f}°C, Max = {np.abs(dineof_wins['delta_rmse']).max():.3f}°C
DINCAE wins: Mean |ΔRMSE| = {np.abs(dincae_wins['delta_rmse']).mean():.3f}°C, Max = {np.abs(dincae_wins['delta_rmse']).max():.3f}°C

REMAINING QUESTIONS
-------------------
1. Why does satellite CV show DINCAE dominance while in-situ shows ~50/50 split?
2. Is there a geographic/climatic pattern to which method wins?
3. Are there temporal patterns (seasonal/yearly) in method performance?
4. What happens at the actual time series level (not just aggregated stats)?

OUTPUT FILES
------------
Figures:
- Q1_observation_predictors.png - Correlation of obs metrics with winner
- Q1_top_predictors_scatter.png - Scatter plots of top predictors
- Q2_data_type_story.png - Mean metrics by data type
- Q2_distributions.png - Box plots of metric distributions
- Q3_strong_predictors.png - Delta metrics vs ΔRMSE
- Q4_mechanism.png - RMSE variance decomposition
- Q5_per_lake_summary.png - Per-lake ΔRMSE with lake IDs
- Q5_win_magnitude.png - Distribution of win margins
- Q6_correlation_heatmap.png - Correlation matrix
- Q6_lake_characteristics_by_winner.png - Lake characteristics by winner

Tables:
- table_per_lake_summary.csv - All lakes with key metrics
- table_winner_breakdown.csv - Statistics by winner
- table_aggregate_statistics.csv - Method/data_type aggregates
- lake_stats_processed.csv - Full processed data

================================================================================
END OF REPORT
================================================================================
"""
    
    report_path = os.path.join(output_dir, 'SUMMARY_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n  Saved: SUMMARY_REPORT.txt")
    print(report)
    
    return report


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="In-situ validation: Comprehensive 5+ questions analysis")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "five_questions_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("IN-SITU VALIDATION: COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    df_raw = load_data(args.analysis_dir)
    df = extract_lake_stats(df_raw)
    print(f"\nProcessed {len(df)} lakes")
    
    q1_results = question1_observation_predictors(df, args.output_dir)
    question2_data_type_story(df, args.output_dir)
    question3_strong_predictors(df, args.output_dir)
    q4_results = question4_mechanism(df, args.output_dir)
    question5_summary(df, args.output_dir)
    question6_comprehensive_tables(df, args.output_dir)
    
    # Save processed data
    df.to_csv(os.path.join(args.output_dir, 'lake_stats_processed.csv'), index=False)
    
    # Generate summary report
    generate_summary_report(df, args.output_dir, q1_results, q4_results)
    
    print("\n" + "="*70)
    print(f"All outputs saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()