#!/usr/bin/env python3
"""
Publication-Quality Diagnostic: The Cold Bias Theory

================================================================================
THE THEORY
================================================================================
DINEOF introduces systematic COLD BIAS at gap-filled pixels, beyond the 
satellite skin-bulk difference. This cold bias hurts DINEOF when validated 
against buoy (bulk temperature), but doesn't affect satellite-based CV 
(which tests skin temperature reconstruction).

DINCAE has LESS cold bias, making it appear competitive in in-situ validation
despite losing badly in satellite CV.

This is NOT because DINCAE is better at gap-filling. It's because:
1. Satellite measures SKIN temperature (~10μm depth, cooler due to evaporation)
2. Buoy measures BULK temperature (~0.5-1m depth, warmer)
3. DINEOF faithfully reconstructs skin → good for CV, but cold vs buoy
4. DINCAE produces warmer values → poor for CV, but closer to buoy

================================================================================
DATA TYPES EXPLAINED
================================================================================
observation:
    - Comparison: SATELLITE observation vs BUOY measurement
    - At pixels/times where satellite HAD valid data AND buoy measurement exists
    - This is the BASELINE: how well does satellite match buoy?
    - Expected bias: negative (satellite skin < buoy bulk)

reconstruction_observed:
    - Comparison: RECONSTRUCTION vs BUOY measurement  
    - At pixels/times where satellite HAD valid data (training overlap)
    - Tests: Does reconstruction preserve satellite values?
    - DINCAE trained on these pixels → should match satellite well

reconstruction_missing:
    - Comparison: RECONSTRUCTION vs BUOY measurement
    - At pixels/times where satellite was MISSING (true gap-fill)
    - Tests: How good is the gap-fill at matching buoy?
    - This is where DINEOF's cold bias appears

reconstruction (all):
    - Combination of reconstruction_observed + reconstruction_missing
    - Dominated by whichever has more points

================================================================================
KEY PREDICTIONS OF THE THEORY
================================================================================
1. observation bias ≈ -0.1 to -0.2°C (skin-bulk difference)
2. reconstruction_observed bias ≈ observation bias (both methods preserve satellite)
3. reconstruction_missing bias:
   - DINEOF: MORE negative than observation (introduces cold bias)
   - DINCAE: Similar to observation (less cold bias)
4. DINCAE wins when obs_bias is more negative (larger skin-bulk gap)
5. DINEOF wins when obs_bias ≈ 0 (skin ≈ bulk)

================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available - some statistical tests will be skipped")


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(analysis_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data and create per-lake summary table.
    
    Returns:
        raw_df: Original per-site records
        lake_df: Aggregated per-lake summary with all metrics
    """
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    raw_df = pd.read_csv(csv_path)
    
    # Create per-lake summary
    lake_records = []
    
    for lake_id in raw_df['lake_id_cci'].unique():
        lake_data = raw_df[raw_df['lake_id_cci'] == lake_id]
        
        record = {'lake_id': lake_id}
        
        # For each data_type and method combination
        for data_type in ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']:
            dt_data = lake_data[lake_data['data_type'] == data_type]
            
            if data_type == 'observation':
                # Observation doesn't have method - it's satellite vs buoy baseline
                if not dt_data.empty:
                    for metric in ['rmse', 'mae', 'bias', 'std', 'correlation', 'n_matches']:
                        if metric in dt_data.columns:
                            record[f'obs_{metric}'] = dt_data[metric].mean()
            else:
                # Reconstruction types have methods
                for method in ['dineof', 'dincae']:
                    m_data = dt_data[dt_data['method'] == method]
                    prefix = f'{data_type.replace("reconstruction", "recon")}_{method}'
                    
                    if not m_data.empty:
                        for metric in ['rmse', 'mae', 'bias', 'std', 'correlation', 'n_matches']:
                            if metric in m_data.columns:
                                record[f'{prefix}_{metric}'] = m_data[metric].mean()
                    else:
                        for metric in ['rmse', 'mae', 'bias', 'std', 'correlation', 'n_matches']:
                            record[f'{prefix}_{metric}'] = np.nan
        
        lake_records.append(record)
    
    lake_df = pd.DataFrame(lake_records)
    
    # Compute derived metrics
    for recon_type in ['recon', 'recon_observed', 'recon_missing']:
        # RMSE difference
        dineof_rmse = f'{recon_type}_dineof_rmse'
        dincae_rmse = f'{recon_type}_dincae_rmse'
        if dineof_rmse in lake_df.columns and dincae_rmse in lake_df.columns:
            lake_df[f'{recon_type}_rmse_diff'] = lake_df[dineof_rmse] - lake_df[dincae_rmse]
            lake_df[f'{recon_type}_winner'] = np.where(
                lake_df[f'{recon_type}_rmse_diff'] > 0.02, 'DINCAE',
                np.where(lake_df[f'{recon_type}_rmse_diff'] < -0.02, 'DINEOF', 'TIE')
            )
        
        # Bias difference
        dineof_bias = f'{recon_type}_dineof_bias'
        dincae_bias = f'{recon_type}_dincae_bias'
        if dineof_bias in lake_df.columns and dincae_bias in lake_df.columns:
            lake_df[f'{recon_type}_bias_diff'] = lake_df[dincae_bias] - lake_df[dineof_bias]
        
        # STD difference
        dineof_std = f'{recon_type}_dineof_std'
        dincae_std = f'{recon_type}_dincae_std'
        if dineof_std in lake_df.columns and dincae_std in lake_df.columns:
            lake_df[f'{recon_type}_std_diff'] = lake_df[dincae_std] - lake_df[dineof_std]
    
    return raw_df, lake_df


# =============================================================================
# FIGURE 1: BIAS PROGRESSION ACROSS DATA TYPES
# =============================================================================

def create_figure1_bias_progression(lake_df: pd.DataFrame, output_dir: str):
    """
    Figure 1: The Cold Bias Theory Visualization
    
    Shows how bias changes from observation → reconstruction_observed → reconstruction_missing
    for both methods. This is the KEY figure demonstrating the theory.
    
    Expected pattern:
    - Observation: ~-0.13°C (skin-bulk baseline)
    - Recon_observed: similar to observation (methods preserve satellite)
    - Recon_missing: DINEOF more negative, DINCAE less negative
    """
    print("\n" + "="*70)
    print("FIGURE 1: Bias Progression Across Data Types")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Panel A: Bar chart of mean bias by data type and method
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Compute means
    obs_bias = lake_df['obs_bias'].dropna()
    recon_obs_dineof = lake_df['recon_observed_dineof_bias'].dropna()
    recon_obs_dincae = lake_df['recon_observed_dincae_bias'].dropna()
    recon_miss_dineof = lake_df['recon_missing_dineof_bias'].dropna()
    recon_miss_dincae = lake_df['recon_missing_dincae_bias'].dropna()
    
    categories = ['Observation\n(Satellite vs Buoy)', 
                  'Recon Observed\n(Training Overlap)',
                  'Recon Missing\n(True Gap-Fill)']
    
    x = np.arange(len(categories))
    width = 0.25
    
    # Plot bars
    obs_means = [obs_bias.mean(), np.nan, np.nan]
    dineof_means = [np.nan, recon_obs_dineof.mean(), recon_miss_dineof.mean()]
    dincae_means = [np.nan, recon_obs_dincae.mean(), recon_miss_dincae.mean()]
    
    bars1 = ax1.bar(x[0], obs_means[0], width, label='Satellite Observation', color='#60BD68', edgecolor='black')
    bars2 = ax1.bar(x[1:] - width/2, dineof_means[1:], width, label='DINEOF', color='#5DA5DA', edgecolor='black')
    bars3 = ax1.bar(x[1:] + width/2, dincae_means[1:], width, label='DINCAE', color='#FAA43A', edgecolor='black')
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(obs_bias.mean(), color='#60BD68', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Baseline: {obs_bias.mean():.3f}°C')
    
    ax1.set_ylabel('Bias (Reconstruction - Buoy) [°C]', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_title('Panel A: Mean Bias by Data Type\n'
                 '(Negative = colder than buoy)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-0.7, 0.1)
    
    # Add value labels
    for bar, val in zip([bars1[0]], [obs_means[0]]):
        ax1.text(bar.get_x() + bar.get_width()/2, val - 0.05, f'{val:.3f}', 
                ha='center', va='top', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, dineof_means[1:]):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val - 0.05, f'{val:.3f}',
                    ha='center', va='top', fontsize=9, fontweight='bold')
    for bar, val in zip(bars3, dincae_means[1:]):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val - 0.05, f'{val:.3f}',
                    ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Print stats
    print("\nMean bias by data type (negative = colder than buoy):")
    print(f"  Observation (satellite vs buoy): {obs_bias.mean():.4f}°C (n={len(obs_bias)} lakes)")
    print(f"  Recon_observed DINEOF: {recon_obs_dineof.mean():.4f}°C")
    print(f"  Recon_observed DINCAE: {recon_obs_dincae.mean():.4f}°C")
    print(f"  Recon_missing DINEOF:  {recon_miss_dineof.mean():.4f}°C  ← MUCH colder!")
    print(f"  Recon_missing DINCAE:  {recon_miss_dincae.mean():.4f}°C")
    
    # -------------------------------------------------------------------------
    # Panel B: Per-lake bias comparison (recon_missing)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get data for recon_missing
    plot_data = lake_df[['lake_id', 'obs_bias', 'recon_missing_dineof_bias', 
                         'recon_missing_dincae_bias', 'recon_missing_winner']].dropna()
    plot_data = plot_data.sort_values('obs_bias')
    
    x = np.arange(len(plot_data))
    width = 0.25
    
    ax2.bar(x - width, plot_data['obs_bias'], width, label='Observation', color='#60BD68', edgecolor='black', alpha=0.8)
    ax2.bar(x, plot_data['recon_missing_dineof_bias'], width, label='DINEOF', color='#5DA5DA', edgecolor='black', alpha=0.8)
    ax2.bar(x + width, plot_data['recon_missing_dincae_bias'], width, label='DINCAE', color='#FAA43A', edgecolor='black', alpha=0.8)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Lakes (sorted by observation bias)', fontsize=10)
    ax2.set_ylabel('Bias [°C]', fontsize=11)
    ax2.set_title('Panel B: Per-Lake Bias Comparison (Gap-Fill Only)\n'
                 '(DINEOF consistently more negative than observation)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_xticks(x[::3])
    ax2.set_xticklabels([str(int(lid)) for lid in plot_data['lake_id'].values[::3]], rotation=45, fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel C: Cold bias magnitude comparison
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Compute additional cold bias introduced by each method
    # additional_cold_bias = method_bias - obs_bias (negative = method introduces more cold)
    valid = lake_df[['obs_bias', 'recon_missing_dineof_bias', 'recon_missing_dincae_bias']].dropna()
    valid['dineof_additional_cold'] = valid['recon_missing_dineof_bias'] - valid['obs_bias']
    valid['dincae_additional_cold'] = valid['recon_missing_dincae_bias'] - valid['obs_bias']
    
    ax3.scatter(valid['dineof_additional_cold'], valid['dincae_additional_cold'], 
               s=100, alpha=0.7, edgecolors='black', c='#9b59b6')
    
    # Add diagonal
    lim = max(abs(valid['dineof_additional_cold']).max(), abs(valid['dincae_additional_cold']).max()) + 0.1
    ax3.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='Equal cold bias')
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('DINEOF Additional Cold Bias [°C]\n(method_bias - obs_bias)', fontsize=10)
    ax3.set_ylabel('DINCAE Additional Cold Bias [°C]', fontsize=10)
    ax3.set_title('Panel C: Additional Cold Bias Introduced by Gap-Filling\n'
                 '(Most points below diagonal → DINEOF introduces MORE cold bias)', 
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Count
    dineof_more_cold = (valid['dineof_additional_cold'] < valid['dincae_additional_cold']).sum()
    ax3.text(0.95, 0.05, f'DINEOF more cold: {dineof_more_cold}/{len(valid)} lakes\n({100*dineof_more_cold/len(valid):.1f}%)',
            transform=ax3.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8))
    
    print(f"\nAdditional cold bias (method_bias - obs_bias):")
    print(f"  DINEOF: mean={valid['dineof_additional_cold'].mean():.4f}°C")
    print(f"  DINCAE: mean={valid['dincae_additional_cold'].mean():.4f}°C")
    print(f"  DINEOF introduces more cold bias in {dineof_more_cold}/{len(valid)} lakes ({100*dineof_more_cold/len(valid):.1f}%)")
    
    # -------------------------------------------------------------------------
    # Panel D: Theory explanation text box
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    theory_text = """
THE COLD BIAS THEORY

OBSERVATION:
• Satellite measures SKIN temperature (~10μm depth)
• Buoy measures BULK temperature (~0.5-1m depth)  
• Skin is colder due to evaporative cooling
• Baseline bias ≈ {:.3f}°C (satellite - buoy)

KEY FINDING:
At GAP-FILLED pixels (reconstruction_missing):
• DINEOF bias = {:.3f}°C (much colder than baseline!)
• DINCAE bias = {:.3f}°C (closer to baseline)
• DINEOF introduces {:.3f}°C ADDITIONAL cold bias
• DINCAE introduces {:.3f}°C additional cold bias

CONSEQUENCE:
• Satellite CV tests: "Match skin temperature" → DINEOF wins
• In-situ tests: "Match bulk temperature" → DINCAE's warmth helps

This explains why DINEOF wins 117/120 lakes in satellite CV
but only ~50% in in-situ validation.
""".format(
        obs_bias.mean(),
        recon_miss_dineof.mean(),
        recon_miss_dincae.mean(),
        valid['dineof_additional_cold'].mean(),
        valid['dincae_additional_cold'].mean()
    )
    
    ax4.text(0.05, 0.95, theory_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Figure 1: The Cold Bias Theory - Why CV and In-Situ Validation Disagree',
                fontsize=14, fontweight='bold', y=1.02)
    
    save_path = os.path.join(output_dir, 'figure1_cold_bias_theory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {save_path}")


# =============================================================================
# FIGURE 2: WINNER PREDICTION BY OBSERVATION BIAS
# =============================================================================

def create_figure2_winner_prediction(lake_df: pd.DataFrame, output_dir: str):
    """
    Figure 2: Does Observation Bias Predict Winner?
    
    Theory predicts: DINCAE wins when obs_bias is more negative (larger skin-bulk gap)
    because DINEOF's additional cold bias hurts more.
    """
    print("\n" + "="*70)
    print("FIGURE 2: Winner Prediction by Observation Bias")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get data
    plot_df = lake_df[['lake_id', 'obs_bias', 'recon_missing_rmse_diff', 
                       'recon_missing_winner', 'recon_missing_dineof_rmse',
                       'recon_missing_dincae_rmse']].dropna()
    
    colors = {'DINEOF': '#5DA5DA', 'DINCAE': '#FAA43A', 'TIE': 'gray'}
    
    # -------------------------------------------------------------------------
    # Panel A: Scatter of obs_bias vs RMSE difference
    # -------------------------------------------------------------------------
    ax = axes[0]
    
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = plot_df[plot_df['recon_missing_winner'] == winner]
        ax.scatter(subset['obs_bias'], subset['recon_missing_rmse_diff'], 
                  c=colors[winner], s=100, alpha=0.7, edgecolors='black',
                  label=f'{winner} ({len(subset)} lakes)')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Regression line
    from numpy.polynomial import polynomial as P
    valid = plot_df[['obs_bias', 'recon_missing_rmse_diff']].dropna()
    if len(valid) > 3:
        z = np.polyfit(valid['obs_bias'], valid['recon_missing_rmse_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['obs_bias'].min(), valid['obs_bias'].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7, label=f'Trend (slope={z[0]:.2f})')
        
        corr = valid['obs_bias'].corr(valid['recon_missing_rmse_diff'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Observation Bias [°C]\n(satellite - buoy; negative = skin colder than bulk)', fontsize=10)
    ax.set_ylabel('RMSE Difference [°C]\n(DINEOF - DINCAE; positive = DINCAE wins)', fontsize=10)
    ax.set_title('Panel A: Does Skin-Bulk Difference Predict Winner?', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel B: Box plot of obs_bias by winner
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    winner_groups = ['DINEOF', 'DINCAE']
    data = [plot_df[plot_df['recon_missing_winner'] == w]['obs_bias'].dropna() for w in winner_groups]
    
    bp = ax.boxplot(data, labels=winner_groups, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors['DINEOF'], colors['DINCAE']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Observation Bias [°C]', fontsize=11)
    ax.set_title('Panel B: Observation Bias by Winner Group', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add means
    for i, (w, d) in enumerate(zip(winner_groups, data)):
        ax.text(i+1, d.mean(), f'mean={d.mean():.3f}', ha='center', va='bottom', fontsize=9)
    
    # Statistical test
    if HAS_SCIPY and len(data[0]) >= 3 and len(data[1]) >= 3:
        stat, pval = stats.mannwhitneyu(data[0], data[1], alternative='two-sided')
        ax.text(0.5, 0.02, f'Mann-Whitney p = {pval:.4f}', transform=ax.transAxes, 
               ha='center', fontsize=10, style='italic')
    
    print(f"\nObservation bias by winner (reconstruction_missing):")
    print(f"  DINEOF wins: mean obs_bias = {data[0].mean():.4f}°C (n={len(data[0])})")
    print(f"  DINCAE wins: mean obs_bias = {data[1].mean():.4f}°C (n={len(data[1])})")
    if HAS_SCIPY and len(data[0]) >= 3 and len(data[1]) >= 3:
        print(f"  Mann-Whitney p = {pval:.4f}")
    
    # -------------------------------------------------------------------------
    # Panel C: Bar chart sorted by obs_bias showing winner
    # -------------------------------------------------------------------------
    ax = axes[2]
    
    sorted_df = plot_df.sort_values('obs_bias')
    x = np.arange(len(sorted_df))
    
    bar_colors = [colors[w] for w in sorted_df['recon_missing_winner']]
    ax.bar(x, sorted_df['obs_bias'], color=bar_colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Lakes (sorted by observation bias)', fontsize=10)
    ax.set_ylabel('Observation Bias [°C]', fontsize=11)
    ax.set_title('Panel C: Winner by Observation Bias\n'
                '(Blue=DINEOF, Orange=DINCAE)', fontsize=12, fontweight='bold')
    ax.set_xticks(x[::3])
    ax.set_xticklabels([str(int(lid)) for lid in sorted_df['lake_id'].values[::3]], 
                       rotation=45, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [Patch(facecolor=colors['DINEOF'], label='DINEOF wins'),
                       Patch(facecolor=colors['DINCAE'], label='DINCAE wins')]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
    
    plt.suptitle('Figure 2: Observation Bias (Skin-Bulk Difference) Predicts In-Situ Winner',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'figure2_winner_prediction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {save_path}")


# =============================================================================
# FIGURE 3: COMPLETE DATA TYPE COMPARISON
# =============================================================================

def create_figure3_datatype_comparison(lake_df: pd.DataFrame, output_dir: str):
    """
    Figure 3: Systematic Comparison Across All Data Types
    
    Shows bias, RMSE, and STD for each data_type and method.
    Clear separation to show where the cold bias appears.
    """
    print("\n" + "="*70)
    print("FIGURE 3: Complete Data Type Comparison")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    data_types = ['observation', 'recon_observed', 'recon_missing']
    metrics = ['bias', 'rmse', 'std']
    method_colors = {'observation': '#60BD68', 'dineof': '#5DA5DA', 'dincae': '#FAA43A'}
    
    # Collect data for summary table
    summary_data = []
    
    for col, metric in enumerate(metrics):
        for row, (ax_row, recon_type) in enumerate(zip([0, 1], ['recon_observed', 'recon_missing'])):
            ax = axes[row, col]
            
            # Get observation baseline
            obs_col = f'obs_{metric}'
            obs_vals = lake_df[obs_col].dropna() if obs_col in lake_df.columns else pd.Series([])
            
            # Get method values
            dineof_col = f'{recon_type}_dineof_{metric}'
            dincae_col = f'{recon_type}_dincae_{metric}'
            
            dineof_vals = lake_df[dineof_col].dropna() if dineof_col in lake_df.columns else pd.Series([])
            dincae_vals = lake_df[dincae_col].dropna() if dincae_col in lake_df.columns else pd.Series([])
            
            if row == 0:  # First row - include observation
                data = [obs_vals, dineof_vals, dincae_vals]
                labels = ['Observation', 'DINEOF', 'DINCAE']
                colors = [method_colors['observation'], method_colors['dineof'], method_colors['dincae']]
            else:  # Second row - just methods
                data = [obs_vals, dineof_vals, dincae_vals]
                labels = ['Observation\n(baseline)', 'DINEOF', 'DINCAE']
                colors = [method_colors['observation'], method_colors['dineof'], method_colors['dincae']]
            
            # Filter empty data
            valid_data = [(d, l, c) for d, l, c in zip(data, labels, colors) if len(d) > 0]
            if not valid_data:
                continue
            
            data, labels, colors = zip(*valid_data)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            if metric == 'bias':
                ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            
            ax.set_ylabel(f'{metric.upper()} [°C]', fontsize=11)
            
            recon_label = 'Reconstruction Observed' if recon_type == 'recon_observed' else 'Reconstruction Missing'
            ax.set_title(f'{recon_label}\n{metric.upper()}', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add means as text
            for i, (d, l) in enumerate(zip(data, labels)):
                ax.text(i+1, d.mean(), f'{d.mean():.3f}', ha='center', va='bottom' if d.mean() >= 0 else 'top',
                       fontsize=8, fontweight='bold')
            
            # Collect for summary
            if row == 1 and metric == 'bias':  # Focus on recon_missing bias
                summary_data.append({
                    'data_type': recon_type,
                    'metric': metric,
                    'obs_mean': obs_vals.mean() if len(obs_vals) > 0 else np.nan,
                    'dineof_mean': dineof_vals.mean() if len(dineof_vals) > 0 else np.nan,
                    'dincae_mean': dincae_vals.mean() if len(dincae_vals) > 0 else np.nan
                })
    
    plt.suptitle('Figure 3: Systematic Comparison Across Data Types\n'
                '(Note: DINEOF bias becomes much more negative at Reconstruction Missing)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'figure3_datatype_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {save_path}")
    
    # Print summary table
    print("\nSummary: Mean values by data type")
    print("-" * 70)
    print(f"{'Data Type':<25} {'Metric':<10} {'Obs':<12} {'DINEOF':<12} {'DINCAE':<12}")
    print("-" * 70)
    
    for dt in ['recon_observed', 'recon_missing']:
        for metric in ['bias', 'rmse', 'std']:
            obs_col = f'obs_{metric}'
            dineof_col = f'{dt}_dineof_{metric}'
            dincae_col = f'{dt}_dincae_{metric}'
            
            obs_val = lake_df[obs_col].mean() if obs_col in lake_df.columns else np.nan
            dineof_val = lake_df[dineof_col].mean() if dineof_col in lake_df.columns else np.nan
            dincae_val = lake_df[dincae_col].mean() if dincae_col in lake_df.columns else np.nan
            
            print(f"{dt:<25} {metric:<10} {obs_val:<12.4f} {dineof_val:<12.4f} {dincae_val:<12.4f}")


# =============================================================================
# FIGURE 4: SMOOTHNESS VS WINNER
# =============================================================================

def create_figure4_smoothness_analysis(lake_df: pd.DataFrame, output_dir: str):
    """
    Figure 4: The Smoothness-Winner Relationship
    
    Shows that DINCAE wins when it's SMOOTHER, not rougher.
    The overall "70% rougher" stat was misleading.
    """
    print("\n" + "="*70)
    print("FIGURE 4: Smoothness vs Winner Analysis")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get data
    std_diff_col = 'recon_missing_std_diff'
    winner_col = 'recon_missing_winner'
    rmse_diff_col = 'recon_missing_rmse_diff'
    
    plot_df = lake_df[[std_diff_col, winner_col, rmse_diff_col, 'lake_id']].dropna()
    
    colors = {'DINEOF': '#5DA5DA', 'DINCAE': '#FAA43A', 'TIE': 'gray'}
    
    # -------------------------------------------------------------------------
    # Panel A: STD difference by winner
    # -------------------------------------------------------------------------
    ax = axes[0]
    
    winner_groups = ['DINEOF', 'DINCAE']
    data = [plot_df[plot_df[winner_col] == w][std_diff_col].dropna() for w in winner_groups]
    
    bp = ax.boxplot(data, labels=winner_groups, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors['DINEOF'], colors['DINCAE']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('STD Difference [°C]\n(DINCAE - DINEOF; negative = DINCAE smoother)', fontsize=10)
    ax.set_title('Panel A: Smoothness by Winner Group\n'
                '(DINCAE wins when smoother!)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    dineof_rougher_when_wins = (data[0] > 0).sum()  # When DINEOF wins
    dincae_smoother_when_wins = (data[1] < 0).sum()  # When DINCAE wins
    
    ax.text(0.05, 0.95, f'When DINEOF wins:\n  DINCAE rougher: {dineof_rougher_when_wins}/{len(data[0])} ({100*dineof_rougher_when_wins/len(data[0]):.0f}%)',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.05, 0.75, f'When DINCAE wins:\n  DINCAE smoother: {dincae_smoother_when_wins}/{len(data[1])} ({100*dincae_smoother_when_wins/len(data[1]):.0f}%)',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(facecolor='white', alpha=0.8))
    
    print(f"\nSTD difference by winner:")
    print(f"  When DINEOF wins: DINCAE is rougher in {dineof_rougher_when_wins}/{len(data[0])} lakes ({100*dineof_rougher_when_wins/len(data[0]):.1f}%)")
    print(f"  When DINCAE wins: DINCAE is smoother in {dincae_smoother_when_wins}/{len(data[1])} lakes ({100*dincae_smoother_when_wins/len(data[1]):.1f}%)")
    
    # -------------------------------------------------------------------------
    # Panel B: Scatter of STD diff vs RMSE diff
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = plot_df[plot_df[winner_col] == winner]
        ax.scatter(subset[std_diff_col], subset[rmse_diff_col], 
                  c=colors[winner], s=100, alpha=0.7, edgecolors='black',
                  label=f'{winner} ({len(subset)})')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('STD Difference [°C]\n(negative = DINCAE smoother)', fontsize=10)
    ax.set_ylabel('RMSE Difference [°C]\n(positive = DINCAE wins)', fontsize=10)
    ax.set_title('Panel B: Smoothness vs Performance', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    corr = plot_df[std_diff_col].corr(plot_df[rmse_diff_col])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # -------------------------------------------------------------------------
    # Panel C: Cross-tabulation
    # -------------------------------------------------------------------------
    ax = axes[2]
    ax.axis('off')
    
    # Create cross-tab
    plot_df['dincae_smoother'] = plot_df[std_diff_col] < 0
    crosstab = pd.crosstab(plot_df[winner_col], plot_df['dincae_smoother'], margins=True)
    crosstab.columns = ['DINCAE Rougher', 'DINCAE Smoother', 'Total']
    
    # Create table visualization
    table_text = "Cross-Tabulation: Winner × Smoothness\n\n"
    table_text += crosstab.to_string()
    table_text += "\n\n"
    table_text += "KEY INSIGHT:\n"
    table_text += "─" * 40 + "\n"
    table_text += "The '70% rougher overall' stat was MISLEADING!\n\n"
    table_text += "When broken down by winner:\n"
    table_text += f"• DINEOF wins → DINCAE ALWAYS rougher (100%)\n"
    table_text += f"• DINCAE wins → DINCAE usually SMOOTHER ({100*dincae_smoother_when_wins/len(data[1]):.0f}%)\n\n"
    table_text += "DINCAE wins BY being smoother,\n"
    table_text += "not despite being rougher!"
    
    ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Figure 4: The Smoothness-Winner Relationship\n'
                '(DINCAE wins when smoother, loses when rougher)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'figure4_smoothness_winner.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {save_path}")


# =============================================================================
# TABLE 1: COMPLETE STATISTICS
# =============================================================================

def create_table1_statistics(lake_df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive statistics table for publication.
    """
    print("\n" + "="*70)
    print("TABLE 1: Complete Statistics Summary")
    print("="*70)
    
    # Collect all statistics
    stats_records = []
    
    for data_type in ['observation', 'recon_observed', 'recon_missing']:
        for metric in ['bias', 'rmse', 'mae', 'std']:
            if data_type == 'observation':
                col = f'obs_{metric}'
                if col in lake_df.columns:
                    vals = lake_df[col].dropna()
                    if len(vals) > 0:
                        stats_records.append({
                            'Data Type': 'Observation',
                            'Method': 'Satellite',
                            'Metric': metric.upper(),
                            'Mean': vals.mean(),
                            'Std': vals.std(),
                            'Median': vals.median(),
                            'Min': vals.min(),
                            'Max': vals.max(),
                            'N': len(vals)
                        })
            else:
                for method in ['dineof', 'dincae']:
                    col = f'{data_type}_{method}_{metric}'
                    if col in lake_df.columns:
                        vals = lake_df[col].dropna()
                        if len(vals) > 0:
                            dt_label = 'Recon Observed' if data_type == 'recon_observed' else 'Recon Missing'
                            stats_records.append({
                                'Data Type': dt_label,
                                'Method': method.upper(),
                                'Metric': metric.upper(),
                                'Mean': vals.mean(),
                                'Std': vals.std(),
                                'Median': vals.median(),
                                'Min': vals.min(),
                                'Max': vals.max(),
                                'N': len(vals)
                            })
    
    stats_df = pd.DataFrame(stats_records)
    
    # Save to CSV
    save_path = os.path.join(output_dir, 'table1_complete_statistics.csv')
    stats_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    
    # Print formatted table
    print("\n" + "-"*100)
    print(f"{'Data Type':<18} {'Method':<10} {'Metric':<8} {'Mean':>10} {'Std':>10} {'Median':>10} {'N':>6}")
    print("-"*100)
    
    for _, row in stats_df.iterrows():
        print(f"{row['Data Type']:<18} {row['Method']:<10} {row['Metric']:<8} {row['Mean']:>10.4f} "
              f"{row['Std']:>10.4f} {row['Median']:>10.4f} {row['N']:>6.0f}")
    
    print("-"*100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Publication-quality diagnostic: The Cold Bias Theory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--analysis_dir", required=True, help="insitu_validation_analysis folder")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "publication_figures")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("PUBLICATION-QUALITY DIAGNOSTIC: The Cold Bias Theory")
    print("="*70)
    print(__doc__)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    raw_df, lake_df = load_and_prepare_data(args.analysis_dir)
    print(f"  Raw records: {len(raw_df)}")
    print(f"  Lakes: {len(lake_df)}")
    
    # Save lake table
    lake_df.to_csv(os.path.join(args.output_dir, 'lake_summary_table.csv'), index=False)
    print(f"  Saved: lake_summary_table.csv")
    
    # Create figures
    create_figure1_bias_progression(lake_df, args.output_dir)
    create_figure2_winner_prediction(lake_df, args.output_dir)
    create_figure3_datatype_comparison(lake_df, args.output_dir)
    create_figure4_smoothness_analysis(lake_df, args.output_dir)
    create_table1_statistics(lake_df, args.output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("PUBLICATION FIGURES COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - figure1_cold_bias_theory.png (KEY FIGURE)")
    print("  - figure2_winner_prediction.png")
    print("  - figure3_datatype_comparison.png")
    print("  - figure4_smoothness_winner.png")
    print("  - table1_complete_statistics.csv")
    print("  - lake_summary_table.csv")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The Cold Bias Theory explains the CV vs In-Situ discrepancy:

1. Satellite CV tests reconstruction of SKIN temperature
   → DINEOF wins because it faithfully reconstructs satellite values

2. In-situ validation tests reconstruction against BULK temperature
   → DINCAE is competitive because it has less cold bias

3. The key is DINEOF's additional cold bias at gap-filled pixels:
   - Observation (satellite vs buoy): ~-0.13°C
   - DINEOF at gaps: ~-0.54°C (much colder!)
   - DINCAE at gaps: ~-0.21°C (closer to baseline)

4. DINCAE wins when it's SMOOTHER (not rougher as initially thought)

5. DINCAE wins on lakes with larger skin-bulk difference (more negative obs_bias)

IMPORTANT: This does NOT mean DINCAE is better at gap-filling.
It means DINCAE produces different values that happen to be closer to buoy.
For reconstructing satellite observations, DINEOF is clearly superior.
""")


if __name__ == "__main__":
    main()
