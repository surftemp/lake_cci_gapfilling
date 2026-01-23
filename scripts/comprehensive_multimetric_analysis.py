#!/usr/bin/env python3
"""
Comprehensive Multi-Metric Analysis for In-Situ Validation

This script examines ALL available metrics, not just RMSE:
- RMSE, MAE, Bias, Residual STD, Correlation, Median Error, Robust STD (MAD)
- Each metric as an OUTCOME: who wins on each metric?
- Each metric as a PREDICTOR: what predicts the winner?
- Cross-metric consistency: do winners agree across metrics?

The goal is to tell the FULL story, not just the RMSE story.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.dpi'] = 150

COLORS = {
    'dineof': '#5DA5DA',
    'dincae': '#FAA43A',
    'tie': '#95a5a6',
    'agree': '#27ae60',
    'disagree': '#e74c3c',
}

# Metrics where LOWER is better
LOWER_IS_BETTER = ['rmse', 'mae', 'std', 'mad', 'median_abs_error']
# Metrics where interpretation is different
SIGNED_METRICS = ['bias', 'median_error']  # Can be positive or negative


def load_combined_stats(analysis_dir: str) -> pd.DataFrame:
    """Load the combined in-situ stats CSV."""
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    return pd.read_csv(csv_path)


def compute_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw CSV may not have all metrics. 
    We work with what's available: rmse, bias, std, mae, correlation, n_matches
    """
    # MAD (Median Absolute Deviation) as robust STD proxy
    # Note: We may not have raw residuals, so we estimate from available stats
    # If we have std, we can estimate MAD ≈ 0.6745 * std (for normal distribution)
    # But this is approximate - ideally we'd compute from raw data
    
    return df  # Return as-is for now, we'll work with available metrics


def extract_all_lake_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-lake statistics for all data types, methods, and metrics.
    """
    results = []
    
    # Available metrics in the data
    metrics = ['rmse', 'bias', 'std', 'mae', 'correlation', 'n_matches']
    
    for lake_id in df['lake_id_cci'].unique():
        lake_df = df[df['lake_id_cci'] == lake_id]
        row = {'lake_id': int(lake_id)}
        
        # OBSERVATION stats (satellite vs buoy baseline)
        obs = lake_df[lake_df['data_type'] == 'observation']
        if not obs.empty:
            for metric in metrics:
                if metric in obs.columns:
                    row[f'obs_{metric}'] = obs[metric].mean()
        
        # RECONSTRUCTION stats for each method and data type
        for method in ['dineof', 'dincae']:
            for dtype in ['reconstruction_observed', 'reconstruction_missing']:
                subset = lake_df[(lake_df['data_type'] == dtype) & (lake_df['method'] == method)]
                if not subset.empty:
                    prefix = f'{method}_{dtype.replace("reconstruction_", "")}'
                    for metric in metrics:
                        if metric in subset.columns:
                            row[f'{prefix}_{metric}'] = subset[metric].mean()
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Compute differences and winners for EACH metric
    metrics_to_compare = ['rmse', 'mae', 'std', 'correlation']
    
    for metric in metrics_to_compare:
        din_col = f'dineof_missing_{metric}'
        dic_col = f'dincae_missing_{metric}'
        
        if din_col in result_df.columns and dic_col in result_df.columns:
            result_df[f'delta_{metric}'] = result_df[din_col] - result_df[dic_col]
            
            # Determine winner for this metric
            if metric in LOWER_IS_BETTER:
                # Lower is better: DINEOF wins if delta < 0 (DINEOF lower)
                result_df[f'winner_{metric}'] = np.where(
                    result_df[f'delta_{metric}'] < -0.02, 'DINEOF',
                    np.where(result_df[f'delta_{metric}'] > 0.02, 'DINCAE', 'TIE'))
            elif metric == 'correlation':
                # Higher correlation is better: DINEOF wins if delta > 0
                result_df[f'winner_{metric}'] = np.where(
                    result_df[f'delta_{metric}'] > 0.02, 'DINEOF',
                    np.where(result_df[f'delta_{metric}'] < -0.02, 'DINCAE', 'TIE'))
    
    # Special handling for bias: winner has smaller |bias|
    din_bias = f'dineof_missing_bias'
    dic_bias = f'dincae_missing_bias'
    if din_bias in result_df.columns and dic_bias in result_df.columns:
        result_df['delta_abs_bias'] = result_df[din_bias].abs() - result_df[dic_bias].abs()
        result_df['winner_abs_bias'] = np.where(
            result_df['delta_abs_bias'] < -0.02, 'DINEOF',
            np.where(result_df['delta_abs_bias'] > 0.02, 'DINCAE', 'TIE'))
    
    return result_df


def analyze_winner_by_metric(df: pd.DataFrame):
    """
    Part 1: Who wins on each metric?
    """
    print("\n" + "="*80)
    print("PART 1: WINNER ANALYSIS BY METRIC")
    print("="*80)
    print("\nFor each metric, who wins more often at reconstruction_missing?\n")
    
    metrics = ['rmse', 'mae', 'std', 'abs_bias', 'correlation']
    
    results = []
    for metric in metrics:
        winner_col = f'winner_{metric}'
        if winner_col not in df.columns:
            continue
        
        valid = df[df[winner_col].notna()]
        dineof_wins = (valid[winner_col] == 'DINEOF').sum()
        dincae_wins = (valid[winner_col] == 'DINCAE').sum()
        ties = (valid[winner_col] == 'TIE').sum()
        total = len(valid)
        
        results.append({
            'metric': metric.upper(),
            'dineof_wins': dineof_wins,
            'dincae_wins': dincae_wins,
            'ties': ties,
            'total': total,
            'dineof_pct': 100 * dineof_wins / (dineof_wins + dincae_wins) if (dineof_wins + dincae_wins) > 0 else 0,
        })
    
    print(f"{'Metric':<15} {'DINEOF':>10} {'DINCAE':>10} {'Ties':>8} {'DINEOF %':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['metric']:<15} {r['dineof_wins']:>10} {r['dincae_wins']:>10} {r['ties']:>8} {r['dineof_pct']:>11.1f}%")
    print("-"*60)
    
    return pd.DataFrame(results)


def analyze_cross_metric_consistency(df: pd.DataFrame):
    """
    Part 2: Do winners agree across metrics?
    """
    print("\n" + "="*80)
    print("PART 2: CROSS-METRIC CONSISTENCY")
    print("="*80)
    print("\nDo the same lakes favor the same method across different metrics?\n")
    
    metrics = ['rmse', 'mae', 'std', 'abs_bias']
    winner_cols = [f'winner_{m}' for m in metrics if f'winner_{m}' in df.columns]
    
    if len(winner_cols) < 2:
        print("Not enough metrics to compare")
        return
    
    # Pairwise agreement
    print("Pairwise agreement (% of lakes where both metrics agree on winner):\n")
    
    agreement_matrix = []
    for i, col1 in enumerate(winner_cols):
        row = []
        for j, col2 in enumerate(winner_cols):
            if i == j:
                row.append(100.0)
            else:
                valid = df[(df[col1] != 'TIE') & (df[col2] != 'TIE')].dropna(subset=[col1, col2])
                agree = (valid[col1] == valid[col2]).sum()
                total = len(valid)
                pct = 100 * agree / total if total > 0 else 0
                row.append(pct)
        agreement_matrix.append(row)
    
    # Print matrix
    metric_names = [m.upper() for m in metrics if f'winner_{m}' in df.columns]
    print(f"{'':>12}", end='')
    for name in metric_names:
        print(f"{name:>12}", end='')
    print()
    
    for i, name in enumerate(metric_names):
        print(f"{name:>12}", end='')
        for j, val in enumerate(agreement_matrix[i]):
            print(f"{val:>11.0f}%", end='')
        print()
    
    # Lakes where RMSE winner differs from other metrics
    print("\n\nLakes where RMSE winner differs from other metrics:")
    print("-"*60)
    
    rmse_col = 'winner_rmse'
    if rmse_col in df.columns:
        for metric in ['mae', 'std', 'abs_bias']:
            other_col = f'winner_{metric}'
            if other_col not in df.columns:
                continue
            
            valid = df[(df[rmse_col] != 'TIE') & (df[other_col] != 'TIE')].dropna(subset=[rmse_col, other_col])
            disagree = valid[valid[rmse_col] != valid[other_col]]
            
            if len(disagree) > 0:
                print(f"\nRMSE vs {metric.upper()}: {len(disagree)} lakes disagree")
                for _, row in disagree.iterrows():
                    print(f"  Lake {int(row['lake_id'])}: RMSE says {row[rmse_col]}, {metric.upper()} says {row[other_col]}")


def analyze_observation_predictors_all_metrics(df: pd.DataFrame):
    """
    Part 3: Do observation stats predict winner for EACH outcome metric?
    """
    print("\n" + "="*80)
    print("PART 3: OBSERVATION-BASED PREDICTORS FOR ALL OUTCOME METRICS")
    print("="*80)
    print("\nCorrelation of observation stats with Δ(metric) for each outcome metric\n")
    
    obs_predictors = ['obs_rmse', 'obs_bias', 'obs_std', 'obs_mae', 'obs_correlation']
    outcome_metrics = ['delta_rmse', 'delta_mae', 'delta_std', 'delta_abs_bias']
    
    results = []
    
    print(f"{'Predictor':<20}", end='')
    for outcome in outcome_metrics:
        print(f"{outcome.replace('delta_', 'Δ'):>12}", end='')
    print()
    print("-"*70)
    
    for pred in obs_predictors:
        if pred not in df.columns:
            continue
        
        print(f"{pred.replace('obs_', ''):<20}", end='')
        
        for outcome in outcome_metrics:
            if outcome not in df.columns:
                print(f"{'N/A':>12}", end='')
                continue
            
            valid = df[[pred, outcome]].dropna()
            if len(valid) < 5:
                print(f"{'N/A':>12}", end='')
                continue
            
            r, p = stats.pearsonr(valid[pred], valid[outcome])
            sig = '*' if p < 0.05 else ''
            print(f"{r:>+11.2f}{sig}", end='')
            
            results.append({
                'predictor': pred,
                'outcome': outcome,
                'r': r,
                'p': p,
            })
        
        print()
    
    print("-"*70)
    print("* p < 0.05")
    
    return pd.DataFrame(results)


def analyze_method_characteristics(df: pd.DataFrame):
    """
    Part 4: Method characteristics across all metrics
    """
    print("\n" + "="*80)
    print("PART 4: METHOD CHARACTERISTICS (reconstruction_missing)")
    print("="*80)
    
    metrics = ['rmse', 'mae', 'std', 'bias', 'correlation']
    
    print("\nMean values across all lakes:\n")
    print(f"{'Metric':<15} {'DINEOF':>12} {'DINCAE':>12} {'Difference':>12}")
    print("-"*55)
    
    for metric in metrics:
        din_col = f'dineof_missing_{metric}'
        dic_col = f'dincae_missing_{metric}'
        
        if din_col not in df.columns or dic_col not in df.columns:
            continue
        
        valid = df[[din_col, dic_col]].dropna()
        din_mean = valid[din_col].mean()
        dic_mean = valid[dic_col].mean()
        diff = din_mean - dic_mean
        
        print(f"{metric.upper():<15} {din_mean:>12.3f} {dic_mean:>12.3f} {diff:>+12.3f}")
    
    # Breakdown by RMSE winner
    print("\n\nBreakdown by RMSE winner:\n")
    
    valid = df[df['winner_rmse'].isin(['DINEOF', 'DINCAE'])]
    
    for winner in ['DINEOF', 'DINCAE']:
        subset = valid[valid['winner_rmse'] == winner]
        print(f"\nWhen {winner} wins on RMSE ({len(subset)} lakes):")
        print(f"  {'Metric':<12} {'DINEOF':>10} {'DINCAE':>10} {'Diff':>10}")
        print("  " + "-"*45)
        
        for metric in metrics:
            din_col = f'dineof_missing_{metric}'
            dic_col = f'dincae_missing_{metric}'
            
            if din_col not in subset.columns or dic_col not in subset.columns:
                continue
            
            din_mean = subset[din_col].mean()
            dic_mean = subset[dic_col].mean()
            diff = din_mean - dic_mean
            
            print(f"  {metric.upper():<12} {din_mean:>10.3f} {dic_mean:>10.3f} {diff:>+10.3f}")


def analyze_component_contributions(df: pd.DataFrame):
    """
    Part 5: How do different components contribute to each metric's winner?
    """
    print("\n" + "="*80)
    print("PART 5: COMPONENT CONTRIBUTIONS")
    print("="*80)
    
    valid = df[df['winner_rmse'].isin(['DINEOF', 'DINCAE'])].copy()
    
    # For RMSE: RMSE² = Bias² + STD²
    print("\n--- RMSE Decomposition ---")
    print("RMSE² = Bias² + STD²\n")
    
    if 'delta_std' in valid.columns and 'dineof_missing_bias' in valid.columns:
        valid['delta_std_sq'] = valid['dineof_missing_std']**2 - valid['dincae_missing_std']**2
        valid['delta_bias_sq'] = valid['dineof_missing_bias']**2 - valid['dincae_missing_bias']**2
        
        var_std = valid['delta_std_sq'].var()
        var_bias = valid['delta_bias_sq'].var()
        cov = valid[['delta_std_sq', 'delta_bias_sq']].cov().iloc[0, 1]
        total = var_std + var_bias + 2 * cov
        
        print(f"Variance decomposition of ΔRMSE²:")
        print(f"  Var(Δ(STD²)):  {100*var_std/total:.1f}%")
        print(f"  Var(Δ(Bias²)): {100*var_bias/total:.1f}%")
        print(f"  2×Cov:         {100*2*cov/total:.1f}%")
    
    # For MAE: How does it relate to RMSE?
    print("\n--- MAE vs RMSE ---")
    if 'delta_rmse' in valid.columns and 'delta_mae' in valid.columns:
        r, p = stats.pearsonr(valid['delta_rmse'], valid['delta_mae'])
        print(f"Correlation between Δ(RMSE) and Δ(MAE): r = {r:.3f}")
        
        # Do they give same winner?
        agree = (valid['winner_rmse'] == valid['winner_mae']).sum()
        total = len(valid)
        print(f"Same winner: {agree}/{total} ({100*agree/total:.0f}%)")
    
    # For STD alone
    print("\n--- STD as standalone metric ---")
    if 'winner_std' in valid.columns:
        std_rmse_agree = (valid['winner_std'] == valid['winner_rmse']).sum()
        print(f"STD winner matches RMSE winner: {std_rmse_agree}/{len(valid)} ({100*std_rmse_agree/len(valid):.0f}%)")


def create_comprehensive_figure(df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive multi-metric figure.
    """
    print("\n" + "="*80)
    print("CREATING: Comprehensive Multi-Metric Figure")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 14))
    
    valid = df.dropna(subset=['winner_rmse'])
    valid = valid[valid['winner_rmse'] != 'TIE']
    
    # Panel A: Winner counts by metric
    ax = fig.add_subplot(2, 3, 1)
    
    metrics = ['rmse', 'mae', 'std', 'abs_bias']
    metric_labels = ['RMSE', 'MAE', 'Residual STD', '|Bias|']
    
    dineof_counts = []
    dincae_counts = []
    
    for metric in metrics:
        winner_col = f'winner_{metric}'
        if winner_col in valid.columns:
            dineof_counts.append((valid[winner_col] == 'DINEOF').sum())
            dincae_counts.append((valid[winner_col] == 'DINCAE').sum())
        else:
            dineof_counts.append(0)
            dincae_counts.append(0)
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dineof_counts, width, label='DINEOF', color=COLORS['dineof'], edgecolor='black')
    bars2 = ax.bar(x + width/2, dincae_counts, width, label='DINCAE', color=COLORS['dincae'], edgecolor='black')
    
    ax.set_ylabel('Number of Lakes')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_title('A) Winner Counts by Metric', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Observation predictors heatmap
    ax = fig.add_subplot(2, 3, 2)
    
    obs_predictors = ['obs_rmse', 'obs_bias', 'obs_std', 'obs_mae', 'obs_correlation']
    obs_labels = ['RMSE', 'Bias', 'STD', 'MAE', 'Corr']
    outcome_metrics = ['delta_rmse', 'delta_mae', 'delta_std', 'delta_abs_bias']
    outcome_labels = ['ΔRMSE', 'ΔMAE', 'ΔSTD', 'Δ|Bias|']
    
    corr_matrix = np.zeros((len(obs_predictors), len(outcome_metrics)))
    
    for i, pred in enumerate(obs_predictors):
        for j, outcome in enumerate(outcome_metrics):
            if pred in df.columns and outcome in df.columns:
                subset = df[[pred, outcome]].dropna()
                if len(subset) >= 5:
                    r, _ = stats.pearsonr(subset[pred], subset[outcome])
                    corr_matrix[i, j] = r
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(outcome_labels)))
    ax.set_yticks(np.arange(len(obs_labels)))
    ax.set_xticklabels(outcome_labels)
    ax.set_yticklabels(obs_labels)
    ax.set_title('B) Observation Predictors vs Outcomes', fontweight='bold')
    
    # Add correlation values
    for i in range(len(obs_labels)):
        for j in range(len(outcome_labels)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation (r)')
    
    # Panel C: Δ(STD) vs Δ(RMSE)
    ax = fig.add_subplot(2, 3, 3)
    
    if 'delta_std' in valid.columns and 'delta_rmse' in valid.columns:
        colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner_rmse']]
        ax.scatter(valid['delta_std'], valid['delta_rmse'], c=colors, s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
        r, _ = stats.pearsonr(valid['delta_std'], valid['delta_rmse'])
        z = np.polyfit(valid['delta_std'], valid['delta_rmse'], 1)
        x_line = np.linspace(valid['delta_std'].min(), valid['delta_std'].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Δ(STD) [°C]')
        ax.set_ylabel('Δ(RMSE) [°C]')
        ax.set_title(f'C) Δ(STD) vs Δ(RMSE): r = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Panel D: Δ(Bias) vs Δ(RMSE)
    ax = fig.add_subplot(2, 3, 4)
    
    if 'delta_bias' in valid.columns and 'delta_rmse' in valid.columns:
        colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner_rmse']]
        ax.scatter(valid['delta_bias'], valid['delta_rmse'], c=colors, s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
        r, _ = stats.pearsonr(valid['delta_bias'], valid['delta_rmse'])
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Δ(Bias) [°C]')
        ax.set_ylabel('Δ(RMSE) [°C]')
        ax.set_title(f'D) Δ(Bias) vs Δ(RMSE): r = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Panel E: Δ(MAE) vs Δ(RMSE)
    ax = fig.add_subplot(2, 3, 5)
    
    if 'delta_mae' in valid.columns and 'delta_rmse' in valid.columns:
        colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner_rmse']]
        ax.scatter(valid['delta_mae'], valid['delta_rmse'], c=colors, s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
        r, _ = stats.pearsonr(valid['delta_mae'], valid['delta_rmse'])
        z = np.polyfit(valid['delta_mae'], valid['delta_rmse'], 1)
        x_line = np.linspace(valid['delta_mae'].min(), valid['delta_mae'].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Δ(MAE) [°C]')
        ax.set_ylabel('Δ(RMSE) [°C]')
        ax.set_title(f'E) Δ(MAE) vs Δ(RMSE): r = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Panel F: Cross-metric winner agreement
    ax = fig.add_subplot(2, 3, 6)
    
    # For each lake, count how many metrics agree on winner
    metrics_to_check = ['winner_rmse', 'winner_mae', 'winner_std', 'winner_abs_bias']
    available = [m for m in metrics_to_check if m in valid.columns]
    
    if len(available) >= 2:
        def count_agreement(row):
            winners = [row[m] for m in available if pd.notna(row[m]) and row[m] != 'TIE']
            if len(winners) == 0:
                return 0
            most_common = max(set(winners), key=winners.count)
            return winners.count(most_common)
        
        valid['agreement_count'] = valid.apply(count_agreement, axis=1)
        
        agreement_dist = valid['agreement_count'].value_counts().sort_index()
        ax.bar(agreement_dist.index, agreement_dist.values, color=COLORS['agree'], edgecolor='black')
        ax.set_xlabel('Number of metrics agreeing on winner')
        ax.set_ylabel('Number of lakes')
        ax.set_title('F) Cross-Metric Agreement', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    
    save_path = os.path.join(output_dir, 'fig_comprehensive_multimetric.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    save_path_png = os.path.join(output_dir, 'fig_comprehensive_multimetric.png')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_detailed_metric_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create detailed per-metric comparison figure.
    """
    print("Creating: Detailed metric comparison figure...")
    
    valid = df.dropna(subset=['winner_rmse'])
    valid = valid[valid['winner_rmse'] != 'TIE']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metric_pairs = [
        ('dineof_missing_rmse', 'dincae_missing_rmse', 'RMSE'),
        ('dineof_missing_mae', 'dincae_missing_mae', 'MAE'),
        ('dineof_missing_std', 'dincae_missing_std', 'Residual STD'),
        ('dineof_missing_bias', 'dincae_missing_bias', 'Bias'),
    ]
    
    for idx, (din_col, dic_col, name) in enumerate(metric_pairs):
        ax = axes[idx // 2, idx % 2]
        
        if din_col not in valid.columns or dic_col not in valid.columns:
            ax.text(0.5, 0.5, f'{name}\nNo data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner_rmse']]
        ax.scatter(valid[din_col], valid[dic_col], c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        lims = [
            min(valid[din_col].min(), valid[dic_col].min()) - 0.1,
            max(valid[din_col].max(), valid[dic_col].max()) + 0.1
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
        
        ax.set_xlabel(f'DINEOF {name}')
        ax.set_ylabel(f'DINCAE {name}')
        ax.set_title(f'{name} Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Count
        if name != 'Bias':
            din_better = (valid[din_col] < valid[dic_col]).sum()
            dic_better = len(valid) - din_better
            ax.text(0.05, 0.95, f'DINEOF lower: {din_better}\nDINCAE lower: {dic_better}',
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.8))
        else:
            din_better = (valid[din_col].abs() < valid[dic_col].abs()).sum()
            dic_better = len(valid) - din_better
            ax.text(0.05, 0.95, f'DINEOF |Bias| lower: {din_better}\nDINCAE |Bias| lower: {dic_better}',
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins (RMSE)'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins (RMSE)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    save_path = os.path.join(output_dir, 'fig_metric_comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    save_path_png = os.path.join(output_dir, 'fig_metric_comparison.png')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive multi-metric analysis")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "comprehensive_multimetric")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE MULTI-METRIC ANALYSIS")
    print("="*80)
    
    # Load and process data
    df_raw = load_combined_stats(args.analysis_dir)
    df = extract_all_lake_stats(df_raw)
    
    print(f"\nLoaded data for {len(df)} lakes")
    
    # Run all analyses
    winner_summary = analyze_winner_by_metric(df)
    analyze_cross_metric_consistency(df)
    obs_pred_results = analyze_observation_predictors_all_metrics(df)
    analyze_method_characteristics(df)
    analyze_component_contributions(df)
    
    # Create figures
    create_comprehensive_figure(df, args.output_dir)
    create_detailed_metric_comparison(df, args.output_dir)
    
    # Save data
    df.to_csv(os.path.join(args.output_dir, 'all_lake_stats_multimetric.csv'), index=False)
    winner_summary.to_csv(os.path.join(args.output_dir, 'winner_summary_by_metric.csv'), index=False)
    obs_pred_results.to_csv(os.path.join(args.output_dir, 'observation_predictors_all_outcomes.csv'), index=False)
    
    print(f"\n{'='*80}")
    print(f"Outputs saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
