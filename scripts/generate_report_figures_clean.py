#!/usr/bin/env python3
"""
Generate Clean Figures for In-Situ Validation Report

Produces publication-ready figures without warning text or informal language.
Designed for integration with LaTeX/Overleaf report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

COLORS = {
    'dineof': '#5DA5DA',
    'dincae': '#FAA43A',
    'std': '#e74c3c',
    'bias': '#3498db',
    'cov': '#9b59b6',
    'weak': '#95a5a6',
}


def load_data(analysis_dir: str) -> pd.DataFrame:
    """Load and process data."""
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    df_raw = pd.read_csv(csv_path)
    
    results = []
    for lake_id in df_raw['lake_id_cci'].unique():
        lake_df = df_raw[df_raw['lake_id_cci'] == lake_id]
        row = {'lake_id': int(lake_id)}
        
        # Observation stats
        obs = lake_df[lake_df['data_type'] == 'observation']
        if not obs.empty:
            for metric in ['rmse', 'bias', 'std', 'mae', 'correlation', 'n_matches']:
                if metric in obs.columns:
                    row[f'obs_{metric}'] = obs[metric].mean()
        
        # Reconstruction_missing stats
        for method in ['dineof', 'dincae']:
            subset = lake_df[(lake_df['data_type'] == 'reconstruction_missing') & 
                            (lake_df['method'] == method)]
            if not subset.empty:
                for metric in ['rmse', 'bias', 'std']:
                    if metric in subset.columns:
                        row[f'{method}_missing_{metric}'] = subset[metric].mean()
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Compute derived metrics
    if 'dineof_missing_rmse' in result_df.columns and 'dincae_missing_rmse' in result_df.columns:
        result_df['delta_rmse'] = result_df['dineof_missing_rmse'] - result_df['dincae_missing_rmse']
        result_df['winner'] = np.where(result_df['delta_rmse'] > 0.02, 'DINCAE',
                                       np.where(result_df['delta_rmse'] < -0.02, 'DINEOF', 'TIE'))
        
        result_df['delta_std'] = result_df['dineof_missing_std'] - result_df['dincae_missing_std']
        result_df['delta_bias'] = result_df['dineof_missing_bias'] - result_df['dincae_missing_bias']
        
        # Squared terms
        result_df['delta_std_sq'] = result_df['dineof_missing_std']**2 - result_df['dincae_missing_std']**2
        result_df['delta_bias_sq'] = result_df['dineof_missing_bias']**2 - result_df['dincae_missing_bias']**2
        result_df['delta_rmse_sq'] = result_df['dineof_missing_rmse']**2 - result_df['dincae_missing_rmse']**2
    
    return result_df


def figure1_observation_predictors(df: pd.DataFrame, output_dir: str):
    """
    Figure 1: Observation-based predictors (non-circular test).
    """
    print("Creating Figure 1: Observation-based predictors")
    
    valid = df.dropna(subset=['delta_rmse', 'winner'])
    valid = valid[valid['winner'] != 'TIE']
    
    predictors = [
        ('obs_rmse', 'RMSE'),
        ('obs_bias', 'Bias'),
        ('obs_std', 'Residual STD'),
        ('obs_mae', 'MAE'),
        ('obs_correlation', 'Correlation'),
        ('obs_n_matches', 'Sample Size'),
    ]
    
    results = []
    for col, name in predictors:
        if col not in valid.columns:
            continue
        subset = valid.dropna(subset=[col])
        if len(subset) < 5:
            continue
        r, p = stats.pearsonr(subset[col], subset['delta_rmse'])
        results.append({'name': name, 'r': r, 'p': p, 'col': col})
    
    if not results:
        print("  No observation predictors available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = [r['name'] for r in results]
    rs = [r['r'] for r in results]
    
    y_pos = np.arange(len(names))
    colors = [COLORS['std'] if abs(r) > 0.3 else COLORS['weak'] for r in rs]
    
    bars = ax.barh(y_pos, rs, color=colors, edgecolor='black', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.3, color='green', linestyle='--', alpha=0.5, label='|r| = 0.3')
    ax.axvline(-0.3, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Correlation with ΔRMSE')
    ax.set_xlim(-1, 1)
    ax.set_title('Observation-Based Predictors\n(Satellite vs Buoy Characteristics)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig1_observation_predictors.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def figure2_decomposition(df: pd.DataFrame, output_dir: str):
    """
    Figure 2: RMSE decomposition - STD vs Bias contributions.
    """
    print("Creating Figure 2: RMSE decomposition")
    
    valid = df.dropna(subset=['delta_rmse', 'delta_std', 'delta_bias', 'winner'])
    valid = valid[valid['winner'] != 'TIE']
    
    colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner']]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: ΔSTD vs ΔRMSE
    ax = axes[0]
    ax.scatter(valid['delta_std'], valid['delta_rmse'], c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    r_std, _ = stats.pearsonr(valid['delta_std'], valid['delta_rmse'])
    z = np.polyfit(valid['delta_std'], valid['delta_rmse'], 1)
    x_line = np.linspace(valid['delta_std'].min() - 0.05, valid['delta_std'].max() + 0.05, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δ(Residual STD) [°C]')
    ax.set_ylabel('Δ(RMSE) [°C]')
    ax.set_title(f'(A) Δ(STD) vs Δ(RMSE)\nr = {r_std:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Panel B: ΔBias vs ΔRMSE
    ax = axes[1]
    ax.scatter(valid['delta_bias'], valid['delta_rmse'], c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    r_bias, _ = stats.pearsonr(valid['delta_bias'], valid['delta_rmse'])
    z = np.polyfit(valid['delta_bias'], valid['delta_rmse'], 1)
    x_line = np.linspace(valid['delta_bias'].min() - 0.05, valid['delta_bias'].max() + 0.05, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δ(Bias) [°C]')
    ax.set_ylabel('Δ(RMSE) [°C]')
    ax.set_title(f'(B) Δ(Bias) vs Δ(RMSE)\nr = {r_bias:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Panel C: Variance decomposition
    ax = axes[2]
    var_std = valid['delta_std_sq'].var()
    var_bias = valid['delta_bias_sq'].var()
    cov = valid[['delta_std_sq', 'delta_bias_sq']].cov().iloc[0, 1]
    total = var_std + var_bias + 2 * cov
    
    pct_std = 100 * var_std / total
    pct_bias = 100 * var_bias / total
    pct_cov = 100 * 2 * cov / total
    
    components = ['Var(Δ(STD²))', 'Var(Δ(Bias²))', '2×Cov']
    values = [pct_std, pct_bias, pct_cov]
    bar_colors = [COLORS['std'], COLORS['bias'], COLORS['cov']]
    
    bars = ax.bar(components, values, color=bar_colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('% of Var(ΔRMSE²)')
    ax.set_title('(C) Variance Decomposition')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        y_pos = val + 3 if val >= 0 else val - 6
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.0f}%', 
               ha='center', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.35, 0.02), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    save_path = os.path.join(output_dir, 'fig2_rmse_decomposition.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    return {'pct_std': pct_std, 'pct_bias': pct_bias, 'r_std': r_std, 'r_bias': r_bias}


def figure3_winner_mechanisms(df: pd.DataFrame, output_dir: str):
    """
    Figure 3: How each method wins - STD vs Bias pathways.
    """
    print("Creating Figure 3: Winner mechanisms")
    
    valid = df.dropna(subset=['delta_rmse', 'delta_std', 'delta_bias', 'winner'])
    valid = valid[valid['winner'] != 'TIE']
    
    # Compute which component favors which method for each lake
    valid['std_favors'] = np.where(valid['delta_std'] > 0, 'DINCAE', 'DINEOF')
    valid['bias_favors'] = np.where(
        valid['dineof_missing_bias'].abs() > valid['dincae_missing_bias'].abs(), 
        'DINCAE', 'DINEOF')
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Panel A: Prediction accuracy by winner
    ax = axes[0]
    
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    std_correct_dineof = (dineof_wins['std_favors'] == 'DINEOF').sum()
    std_correct_dincae = (dincae_wins['std_favors'] == 'DINCAE').sum()
    bias_correct_dineof = (dineof_wins['bias_favors'] == 'DINEOF').sum()
    bias_correct_dincae = (dincae_wins['bias_favors'] == 'DINCAE').sum()
    
    x = np.arange(2)
    width = 0.35
    
    std_pcts = [100*std_correct_dineof/len(dineof_wins), 100*std_correct_dincae/len(dincae_wins)]
    bias_pcts = [100*bias_correct_dineof/len(dineof_wins), 100*bias_correct_dincae/len(dincae_wins)]
    
    bars1 = ax.bar(x - width/2, std_pcts, width, label='Lower STD', color=COLORS['std'], edgecolor='black')
    bars2 = ax.bar(x + width/2, bias_pcts, width, label='Lower |Bias|', color=COLORS['bias'], edgecolor='black')
    
    ax.set_ylabel('% of Wins with This Advantage')
    ax.set_xticks(x)
    ax.set_xticklabels([f'DINEOF Wins\n(n={len(dineof_wins)})', f'DINCAE Wins\n(n={len(dincae_wins)})'])
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')
    ax.set_title('(A) How Each Method Wins')
    ax.grid(axis='y', alpha=0.3)
    
    for bars, pcts in [(bars1, std_pcts), (bars2, bias_pcts)]:
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                   f'{pct:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Panel B: Winner counts
    ax = axes[1]
    winner_counts = valid['winner'].value_counts()
    bars = ax.bar(['DINEOF', 'DINCAE'], 
                  [winner_counts.get('DINEOF', 0), winner_counts.get('DINCAE', 0)],
                  color=[COLORS['dineof'], COLORS['dincae']], edgecolor='black')
    ax.set_ylabel('Number of Lakes')
    ax.set_title('(B) Winner Counts\n(reconstruction\\_missing)')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{int(bar.get_height())}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'fig3_winner_mechanisms.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def figure4_temporal_interpretation(df: pd.DataFrame, output_dir: str):
    """
    Figure 4: Physical interpretation - temporal pattern matching.
    """
    print("Creating Figure 4: Temporal interpretation")
    
    valid = df.dropna(subset=['delta_rmse', 'delta_std', 'obs_std', 'winner'])
    valid = valid[valid['winner'] != 'TIE']
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner']]
    
    # Panel A: Method STD comparison
    ax = axes[0]
    ax.scatter(valid['dineof_missing_std'], valid['dincae_missing_std'], 
              c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    lims = [min(valid['dineof_missing_std'].min(), valid['dincae_missing_std'].min()) - 0.1,
            max(valid['dineof_missing_std'].max(), valid['dincae_missing_std'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal STD')
    
    ax.set_xlabel('DINEOF Residual STD [°C]')
    ax.set_ylabel('DINCAE Residual STD [°C]')
    ax.set_title('(A) Residual STD Comparison\n(Below diagonal = DINCAE lower)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Count points
    dineof_lower = (valid['dineof_missing_std'] < valid['dincae_missing_std']).sum()
    dincae_lower = len(valid) - dineof_lower
    ax.text(0.95, 0.05, f'DINEOF lower: {dineof_lower}\nDINCAE lower: {dincae_lower}',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Panel B: Observation STD vs method performance
    ax = axes[1]
    
    # Does observation STD predict anything?
    if 'obs_std' in valid.columns:
        ax.scatter(valid['obs_std'], valid['delta_rmse'], c=colors, s=80, alpha=0.7, 
                  edgecolors='black', linewidth=0.5)
        r, p = stats.pearsonr(valid['obs_std'].dropna(), 
                             valid.loc[valid['obs_std'].notna(), 'delta_rmse'])
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Observation Residual STD [°C]\n(Satellite vs Buoy baseline)')
        ax.set_ylabel('Δ(RMSE) [°C]')
        ax.set_title(f'(B) Lake Baseline vs Method Difference\nr = {r:.3f}')
        ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    save_path = os.path.join(output_dir, 'fig4_temporal_interpretation.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def figure5_comprehensive_summary(df: pd.DataFrame, output_dir: str):
    """
    Figure 5: Comprehensive 6-panel summary figure.
    """
    print("Creating Figure 5: Comprehensive summary")
    
    valid = df.dropna(subset=['delta_rmse', 'delta_std', 'delta_bias', 'winner'])
    valid = valid[valid['winner'] != 'TIE']
    
    colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner']]
    
    fig = plt.figure(figsize=(15, 10))
    
    # Panel A: Observation-based predictors
    ax = fig.add_subplot(2, 3, 1)
    predictors = [
        ('obs_rmse', 'RMSE'),
        ('obs_bias', 'Bias'),
        ('obs_std', 'Residual STD'),
        ('obs_mae', 'MAE'),
        ('obs_correlation', 'Correlation'),
    ]
    results = []
    for col, name in predictors:
        if col in valid.columns:
            subset = valid.dropna(subset=[col])
            if len(subset) >= 5:
                r, _ = stats.pearsonr(subset[col], subset['delta_rmse'])
                results.append({'name': name, 'r': r})
    
    if results:
        names = [r['name'] for r in results]
        rs = [r['r'] for r in results]
        y_pos = np.arange(len(names))
        bar_colors = [COLORS['std'] if abs(r) > 0.3 else COLORS['weak'] for r in rs]
        ax.barh(y_pos, rs, color=bar_colors, edgecolor='black', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(0.3, color='green', linestyle='--', alpha=0.5)
        ax.axvline(-0.3, color='green', linestyle='--', alpha=0.5)
        ax.set_xlim(-1, 1)
    ax.set_xlabel('Correlation with ΔRMSE')
    ax.set_title('(A) Observation-Based Predictors')
    ax.grid(axis='x', alpha=0.3)
    
    # Panel B: ΔSTD vs ΔRMSE
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(valid['delta_std'], valid['delta_rmse'], c=colors, s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
    r_std, _ = stats.pearsonr(valid['delta_std'], valid['delta_rmse'])
    z = np.polyfit(valid['delta_std'], valid['delta_rmse'], 1)
    x_line = np.linspace(valid['delta_std'].min(), valid['delta_std'].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δ(STD) [°C]')
    ax.set_ylabel('Δ(RMSE) [°C]')
    ax.set_title(f'(B) Δ(STD) vs Δ(RMSE): r = {r_std:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Panel C: ΔBias vs ΔRMSE
    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(valid['delta_bias'], valid['delta_rmse'], c=colors, s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
    r_bias, _ = stats.pearsonr(valid['delta_bias'], valid['delta_rmse'])
    z = np.polyfit(valid['delta_bias'], valid['delta_rmse'], 1)
    x_line = np.linspace(valid['delta_bias'].min(), valid['delta_bias'].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δ(Bias) [°C]')
    ax.set_ylabel('Δ(RMSE) [°C]')
    ax.set_title(f'(C) Δ(Bias) vs Δ(RMSE): r = {r_bias:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Panel D: Variance decomposition
    ax = fig.add_subplot(2, 3, 4)
    var_std = valid['delta_std_sq'].var()
    var_bias = valid['delta_bias_sq'].var()
    cov = valid[['delta_std_sq', 'delta_bias_sq']].cov().iloc[0, 1]
    total = var_std + var_bias + 2 * cov
    
    pct_std = 100 * var_std / total
    pct_bias = 100 * var_bias / total
    pct_cov = 100 * 2 * cov / total
    
    bars = ax.bar(['Var(Δ(STD²))', 'Var(Δ(Bias²))', '2×Cov'], 
                  [pct_std, pct_bias, pct_cov],
                  color=[COLORS['std'], COLORS['bias'], COLORS['cov']], edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('% of Var(ΔRMSE²)')
    ax.set_title('(D) Variance Decomposition')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, [pct_std, pct_bias, pct_cov]):
        y_pos = val + 3 if val >= 0 else val - 6
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.0f}%', 
               ha='center', fontsize=10, fontweight='bold')
    
    # Panel E: How each method wins
    ax = fig.add_subplot(2, 3, 5)
    
    valid['std_favors'] = np.where(valid['delta_std'] > 0, 'DINCAE', 'DINEOF')
    valid['bias_favors'] = np.where(
        valid['dineof_missing_bias'].abs() > valid['dincae_missing_bias'].abs(), 
        'DINCAE', 'DINEOF')
    
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    std_correct_dineof = (dineof_wins['std_favors'] == 'DINEOF').sum()
    std_correct_dincae = (dincae_wins['std_favors'] == 'DINCAE').sum()
    bias_correct_dineof = (dineof_wins['bias_favors'] == 'DINEOF').sum()
    bias_correct_dincae = (dincae_wins['bias_favors'] == 'DINCAE').sum()
    
    x = np.arange(2)
    width = 0.35
    std_pcts = [100*std_correct_dineof/len(dineof_wins), 100*std_correct_dincae/len(dincae_wins)]
    bias_pcts = [100*bias_correct_dineof/len(dineof_wins), 100*bias_correct_dincae/len(dincae_wins)]
    
    ax.bar(x - width/2, std_pcts, width, label='Lower STD', color=COLORS['std'], edgecolor='black')
    ax.bar(x + width/2, bias_pcts, width, label='Lower |Bias|', color=COLORS['bias'], edgecolor='black')
    ax.set_ylabel('% of Wins')
    ax.set_xticks(x)
    ax.set_xticklabels([f'DINEOF\n(n={len(dineof_wins)})', f'DINCAE\n(n={len(dincae_wins)})'])
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('(E) How Each Method Wins')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel F: Winner counts
    ax = fig.add_subplot(2, 3, 6)
    winner_counts = valid['winner'].value_counts()
    bars = ax.bar(['DINEOF', 'DINCAE'], 
                  [winner_counts.get('DINEOF', 0), winner_counts.get('DINCAE', 0)],
                  color=[COLORS['dineof'], COLORS['dincae']], edgecolor='black')
    ax.set_ylabel('Number of Lakes')
    ax.set_title('(F) Winner Counts')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{int(bar.get_height())}', ha='center', fontsize=11, fontweight='bold')
    
    # Add legend for scatter plots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    save_path = os.path.join(output_dir, 'fig5_comprehensive_summary.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate clean figures for report")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "report_figures")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Generating Clean Figures for Report")
    print("="*60)
    
    df = load_data(args.analysis_dir)
    print(f"Loaded data for {len(df)} lakes")
    
    figure1_observation_predictors(df, args.output_dir)
    decomp = figure2_decomposition(df, args.output_dir)
    figure3_winner_mechanisms(df, args.output_dir)
    figure4_temporal_interpretation(df, args.output_dir)
    figure5_comprehensive_summary(df, args.output_dir)
    
    # Save summary stats
    valid = df.dropna(subset=['winner'])
    valid = valid[valid['winner'] != 'TIE']
    
    summary = {
        'n_lakes': len(valid),
        'dineof_wins': (valid['winner'] == 'DINEOF').sum(),
        'dincae_wins': (valid['winner'] == 'DINCAE').sum(),
        'r_std': decomp['r_std'],
        'r_bias': decomp['r_bias'],
        'pct_var_std': decomp['pct_std'],
        'pct_var_bias': decomp['pct_bias'],
    }
    
    pd.Series(summary).to_csv(os.path.join(args.output_dir, 'summary_stats.csv'))
    
    print(f"\nAll figures saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
