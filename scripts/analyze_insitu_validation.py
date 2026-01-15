#!/usr/bin/env python3
"""
In-Situ Validation Analysis Script

Collects all in-situ validation results across all lakes and generates:
1. Aggregated statistics CSV
2. Summary report (text + markdown)
3. Comprehensive visualizations comparing DINEOF vs DINCAE

Usage:
    python analyze_insitu_validation.py --run_root /path/to/experiment --output_dir /path/to/output

    # With specific alpha
    python analyze_insitu_validation.py --run_root /path/to/experiment --alpha a1000

    # Include lake metadata for geographic analysis
    python analyze_insitu_validation.py --run_root /path/to/experiment --lake_metadata /path/to/lake_info.csv
"""

import argparse
import os
import sys
from glob import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats


def find_validation_csvs(run_root: str, alpha: str = None) -> List[str]:
    """Find all in-situ validation CSV files in the run directory."""
    if alpha:
        pattern = os.path.join(run_root, "post", "*", alpha, "insitu_cv_validation", "*_insitu_stats_site*.csv")
    else:
        pattern = os.path.join(run_root, "post", "*", "*", "insitu_cv_validation", "*_insitu_stats_site*.csv")
    
    csv_files = glob(pattern)
    
    # Filter out yearly stats files
    csv_files = [f for f in csv_files if '_yearly_' not in f]
    
    return sorted(csv_files)


def load_all_stats(csv_files: List[str]) -> pd.DataFrame:
    """Load and concatenate all validation stats CSVs."""
    all_dfs = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add source file for debugging
            df['source_file'] = os.path.basename(csv_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def compute_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics across all lakes for each method/data_type combination."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by method and data_type
    grouped = df.groupby(['method', 'data_type'])
    
    agg_stats = []
    for (method, data_type), group in grouped:
        n_lakes = group['lake_id_cci'].nunique()
        n_points = group['n_matches'].sum()
        
        # Weighted average RMSE (weighted by n_matches)
        weights = group['n_matches'].values
        if weights.sum() > 0:
            weighted_rmse = np.average(group['rmse'].values, weights=weights)
            weighted_bias = np.average(group['bias'].values, weights=weights)
            weighted_mae = np.average(group['mae'].values, weights=weights)
        else:
            weighted_rmse = np.nan
            weighted_bias = np.nan
            weighted_mae = np.nan
        
        # Simple statistics
        agg_stats.append({
            'method': method,
            'data_type': data_type,
            'n_lakes': n_lakes,
            'n_total_points': n_points,
            'rmse_weighted': weighted_rmse,
            'rmse_mean': group['rmse'].mean(),
            'rmse_std': group['rmse'].std(),
            'rmse_median': group['rmse'].median(),
            'rmse_min': group['rmse'].min(),
            'rmse_max': group['rmse'].max(),
            'bias_weighted': weighted_bias,
            'bias_mean': group['bias'].mean(),
            'mae_weighted': weighted_mae,
            'mae_mean': group['mae'].mean(),
            'correlation_mean': group['correlation'].mean(),
            'correlation_median': group['correlation'].median(),
        })
    
    return pd.DataFrame(agg_stats)


def generate_summary_report(df: pd.DataFrame, agg_df: pd.DataFrame, output_dir: str) -> str:
    """Generate a text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("IN-SITU VALIDATION SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall summary
    n_lakes = df['lake_id_cci'].nunique()
    n_sites = len(df.groupby(['lake_id_cci', 'site_id']))
    methods = df['method'].unique()
    
    report_lines.append("OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total lakes with in-situ validation: {n_lakes}")
    report_lines.append(f"Total validation sites: {n_sites}")
    report_lines.append(f"Methods analyzed: {', '.join(methods)}")
    report_lines.append("")
    
    # Key comparisons
    report_lines.append("KEY FINDINGS")
    report_lines.append("-" * 40)
    
    # DINEOF vs DINCAE comparison (reconstruction)
    dineof_recon = agg_df[(agg_df['method'] == 'dineof') & (agg_df['data_type'] == 'reconstruction')]
    dincae_recon = agg_df[(agg_df['method'] == 'dincae') & (agg_df['data_type'] == 'reconstruction')]
    
    if not dineof_recon.empty and not dincae_recon.empty:
        dineof_rmse = dineof_recon['rmse_weighted'].values[0]
        dincae_rmse = dincae_recon['rmse_weighted'].values[0]
        better = "DINEOF" if dineof_rmse < dincae_rmse else "DINCAE"
        diff = abs(dineof_rmse - dincae_rmse)
        report_lines.append(f"Overall Reconstruction (all points):")
        report_lines.append(f"  DINEOF RMSE: {dineof_rmse:.3f}°C")
        report_lines.append(f"  DINCAE RMSE: {dincae_rmse:.3f}°C")
        report_lines.append(f"  → {better} is better by {diff:.3f}°C")
        report_lines.append("")
    
    # Observed vs Gap-fill comparison
    dineof_obs = agg_df[(agg_df['method'] == 'dineof') & (agg_df['data_type'] == 'reconstruction_observed')]
    dineof_miss = agg_df[(agg_df['method'] == 'dineof') & (agg_df['data_type'] == 'reconstruction_missing')]
    dincae_obs = agg_df[(agg_df['method'] == 'dincae') & (agg_df['data_type'] == 'reconstruction_observed')]
    dincae_miss = agg_df[(agg_df['method'] == 'dincae') & (agg_df['data_type'] == 'reconstruction_missing')]
    
    if not dineof_obs.empty and not dineof_miss.empty:
        report_lines.append("DINEOF - Observed vs Gap-filled:")
        report_lines.append(f"  Observed RMSE:  {dineof_obs['rmse_weighted'].values[0]:.3f}°C (N={int(dineof_obs['n_total_points'].values[0])})")
        report_lines.append(f"  Gap-fill RMSE:  {dineof_miss['rmse_weighted'].values[0]:.3f}°C (N={int(dineof_miss['n_total_points'].values[0])})")
        gap_penalty = dineof_miss['rmse_weighted'].values[0] - dineof_obs['rmse_weighted'].values[0]
        report_lines.append(f"  → Gap-fill penalty: +{gap_penalty:.3f}°C")
        report_lines.append("")
    
    if not dincae_obs.empty and not dincae_miss.empty:
        report_lines.append("DINCAE - Observed vs Gap-filled:")
        report_lines.append(f"  Observed RMSE:  {dincae_obs['rmse_weighted'].values[0]:.3f}°C (N={int(dincae_obs['n_total_points'].values[0])})")
        report_lines.append(f"  Gap-fill RMSE:  {dincae_miss['rmse_weighted'].values[0]:.3f}°C (N={int(dincae_miss['n_total_points'].values[0])})")
        gap_penalty = dincae_miss['rmse_weighted'].values[0] - dincae_obs['rmse_weighted'].values[0]
        report_lines.append(f"  → Gap-fill penalty: +{gap_penalty:.3f}°C")
        report_lines.append("")
    
    # True gap-filling comparison
    if not dineof_miss.empty and not dincae_miss.empty:
        dineof_gap = dineof_miss['rmse_weighted'].values[0]
        dincae_gap = dincae_miss['rmse_weighted'].values[0]
        better = "DINEOF" if dineof_gap < dincae_gap else "DINCAE"
        diff = abs(dineof_gap - dincae_gap)
        report_lines.append("True Gap-Filling Performance (missing pixels only):")
        report_lines.append(f"  DINEOF: {dineof_gap:.3f}°C")
        report_lines.append(f"  DINCAE: {dincae_gap:.3f}°C")
        report_lines.append(f"  → {better} is better at gap-filling by {diff:.3f}°C")
        report_lines.append("")
    
    # Per-lake winner analysis
    report_lines.append("PER-LAKE WINNER ANALYSIS")
    report_lines.append("-" * 40)
    
    # Aggregate to per-lake level
    lake_stats = df.groupby(['lake_id_cci', 'method', 'data_type']).agg({'rmse': 'mean'}).reset_index()
    
    # All reconstruction
    dineof_all = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction')][['lake_id_cci', 'rmse']]
    dincae_all = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction')][['lake_id_cci', 'rmse']]
    if not dineof_all.empty and not dincae_all.empty:
        merged = dineof_all.merge(dincae_all, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        if not merged.empty:
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            total = len(merged)
            report_lines.append(f"All Reconstruction (N={total} lakes):")
            report_lines.append(f"  DINEOF better: {dineof_wins} lakes ({100*dineof_wins/total:.1f}%)")
            report_lines.append(f"  DINCAE better: {dincae_wins} lakes ({100*dincae_wins/total:.1f}%)")
            report_lines.append("")
    
    # Gap-fill only
    dineof_gap = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    dincae_gap = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    if not dineof_gap.empty and not dincae_gap.empty:
        merged = dineof_gap.merge(dincae_gap, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        if not merged.empty:
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            total = len(merged)
            report_lines.append(f"Gap-Fill Only (N={total} lakes):")
            report_lines.append(f"  DINEOF better: {dineof_wins} lakes ({100*dineof_wins/total:.1f}%)")
            report_lines.append(f"  DINCAE better: {dincae_wins} lakes ({100*dincae_wins/total:.1f}%)")
            report_lines.append("")
    
    # Detailed table
    report_lines.append("")
    report_lines.append("DETAILED STATISTICS BY METHOD AND DATA TYPE")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Method':<20} {'Data Type':<25} {'N Lakes':<8} {'N Points':<10} {'RMSE (weighted)':<15} {'Bias':<10} {'Corr':<8}")
    report_lines.append("-" * 80)
    
    for _, row in agg_df.iterrows():
        report_lines.append(
            f"{row['method']:<20} {row['data_type']:<25} {int(row['n_lakes']):<8} "
            f"{int(row['n_total_points']):<10} {row['rmse_weighted']:.3f}°C{'':<8} "
            f"{row['bias_weighted']:.3f}°C{'':<4} {row['correlation_mean']:.3f}"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_path = os.path.join(output_dir, "insitu_validation_summary_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_path


def create_method_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Create bar chart comparing methods across data types."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Filter for main comparisons
    methods_order = ['dineof', 'dincae', 'eof_filtered', 'interp_full', 'eof_filtered_interp_full']
    data_types = ['reconstruction', 'reconstruction_observed', 'reconstruction_missing']
    colors = {'dineof': '#2ecc71', 'dincae': '#3498db', 'eof_filtered': '#9b59b6', 
              'interp_full': '#e74c3c', 'eof_filtered_interp_full': '#f39c12'}
    
    for ax_idx, data_type in enumerate(data_types):
        ax = axes[ax_idx]
        subset = df[df['data_type'] == data_type]
        
        if subset.empty:
            ax.set_title(f"No data for {data_type}")
            continue
        
        # Aggregate by method
        method_stats = subset.groupby('method').agg({
            'rmse': ['mean', 'std'],
            'n_matches': 'sum'
        }).reset_index()
        method_stats.columns = ['method', 'rmse_mean', 'rmse_std', 'n_total']
        
        # Sort by predefined order
        method_stats['sort_key'] = method_stats['method'].apply(
            lambda x: methods_order.index(x) if x in methods_order else 99
        )
        method_stats = method_stats.sort_values('sort_key')
        
        x = np.arange(len(method_stats))
        bars = ax.bar(x, method_stats['rmse_mean'], 
                     yerr=method_stats['rmse_std'],
                     color=[colors.get(m, 'gray') for m in method_stats['method']],
                     capsize=3, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(method_stats['method'], rotation=45, ha='right')
        ax.set_ylabel('RMSE (°C)')
        ax.set_title(data_type.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, method_stats['rmse_mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Method Comparison: RMSE by Data Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "method_comparison_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_dineof_vs_dincae_scatter(df: pd.DataFrame, output_dir: str):
    """Create scatter plots comparing DINEOF vs DINCAE RMSE per lake."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    data_types = [
        ('reconstruction', 'All Reconstruction'),
        ('reconstruction_observed', 'Observed Pixels Only'),
        ('reconstruction_missing', 'Gap-Filled Pixels Only')
    ]
    
    for ax_idx, (data_type, title) in enumerate(data_types):
        ax = axes[ax_idx]
        
        # Get DINEOF and DINCAE stats per lake
        dineof = df[(df['method'] == 'dineof') & (df['data_type'] == data_type)][['lake_id_cci', 'site_id', 'rmse', 'n_matches']]
        dincae = df[(df['method'] == 'dincae') & (df['data_type'] == data_type)][['lake_id_cci', 'site_id', 'rmse', 'n_matches']]
        
        if dineof.empty or dincae.empty:
            ax.set_title(f"{title}\n(No data)")
            continue
        
        # Merge on lake_id and site_id
        merged = dineof.merge(dincae, on=['lake_id_cci', 'site_id'], suffixes=('_dineof', '_dincae'))
        
        if merged.empty:
            ax.set_title(f"{title}\n(No matching lakes)")
            continue
        
        # Scatter plot
        scatter = ax.scatter(merged['rmse_dineof'], merged['rmse_dincae'], 
                            c=merged['n_matches_dineof'], cmap='viridis',
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Diagonal line (y=x)
        max_val = max(merged['rmse_dineof'].max(), merged['rmse_dincae'].max()) * 1.1
        min_val = min(merged['rmse_dineof'].min(), merged['rmse_dincae'].min()) * 0.9
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal performance')
        
        ax.set_xlabel('DINEOF RMSE (°C)')
        ax.set_ylabel('DINCAE RMSE (°C)')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Count wins
        dineof_better = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
        dincae_better = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
        ax.text(0.05, 0.95, f'DINEOF better: {dineof_better}\nDINCAE better: {dincae_better}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.colorbar(scatter, ax=ax, label='N matches')
    
    plt.suptitle('DINEOF vs DINCAE: Per-Lake RMSE Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "dineof_vs_dincae_scatter.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_observed_vs_gapfill_comparison(df: pd.DataFrame, output_dir: str):
    """Create visualization comparing observed vs gap-fill performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['dineof', 'dincae']
    colors = {'observed': '#2ecc71', 'gap-fill': '#e74c3c'}
    
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        
        obs = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_observed')]['rmse']
        miss = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_missing')]['rmse']
        
        if obs.empty and miss.empty:
            ax.set_title(f"{method.upper()}\n(No data)")
            continue
        
        data_to_plot = []
        labels = []
        plot_colors = []
        
        if not obs.empty:
            data_to_plot.append(obs.values)
            labels.append(f'Observed\n(N={len(obs)})')
            plot_colors.append(colors['observed'])
        
        if not miss.empty:
            data_to_plot.append(miss.values)
            labels.append(f'Gap-fill\n(N={len(miss)})')
            plot_colors.append(colors['gap-fill'])
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('RMSE (°C)')
        ax.set_title(f'{method.upper()}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean values
        for i, data in enumerate(data_to_plot):
            mean_val = np.mean(data)
            ax.axhline(y=mean_val, xmin=(i+0.7)/len(data_to_plot), xmax=(i+1.3)/len(data_to_plot),
                      color='black', linestyle='--', alpha=0.5)
            ax.text(i+1.2, mean_val, f'μ={mean_val:.2f}', fontsize=9, va='center')
    
    plt.suptitle('Observed vs Gap-Fill Performance Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "observed_vs_gapfill_boxplot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_rmse_distribution_plot(df: pd.DataFrame, output_dir: str):
    """Create histogram of RMSE distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    plot_configs = [
        ('dineof', 'reconstruction', 'DINEOF - All Reconstruction'),
        ('dincae', 'reconstruction', 'DINCAE - All Reconstruction'),
        ('dineof', 'reconstruction_missing', 'DINEOF - Gap-Fill Only'),
        ('dincae', 'reconstruction_missing', 'DINCAE - Gap-Fill Only'),
    ]
    
    for ax, (method, data_type, title) in zip(axes.flatten(), plot_configs):
        subset = df[(df['method'] == method) & (df['data_type'] == data_type)]['rmse']
        
        if subset.empty:
            ax.set_title(f"{title}\n(No data)")
            continue
        
        ax.hist(subset, bins=20, edgecolor='black', alpha=0.7,
               color='#3498db' if method == 'dincae' else '#2ecc71')
        
        # Add statistics
        mean_val = subset.mean()
        median_val = subset.median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}°C')
        ax.axvline(median_val, color='orange', linestyle='-', label=f'Median: {median_val:.2f}°C')
        
        ax.set_xlabel('RMSE (°C)')
        ax.set_ylabel('Number of Lakes')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('RMSE Distribution Across Lakes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "rmse_distribution_hist.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_correlation_comparison(df: pd.DataFrame, output_dir: str):
    """Create correlation comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['dineof', 'dincae', 'eof_filtered', 'interp_full']
    data_types = ['reconstruction', 'observation']
    
    x = np.arange(len(methods))
    width = 0.35
    
    for i, data_type in enumerate(data_types):
        correlations = []
        for method in methods:
            subset = df[(df['method'] == method) & (df['data_type'] == data_type)]
            if not subset.empty:
                correlations.append(subset['correlation'].mean())
            else:
                correlations.append(0)
        
        bars = ax.bar(x + i*width, correlations, width, 
                     label=data_type.replace('_', ' ').title(),
                     alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, correlations):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Correlation (R)')
    ax.set_xlabel('Method')
    ax.set_title('Mean Correlation with In-Situ Measurements')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "correlation_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_gap_penalty_analysis(df: pd.DataFrame, output_dir: str):
    """Analyze the 'gap penalty' - how much worse is gap-filling vs observed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['dineof', 'dincae']
    
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        
        # Get observed and missing RMSE per lake
        obs = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_observed')][['lake_id_cci', 'site_id', 'rmse']]
        miss = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'site_id', 'rmse']]
        
        if obs.empty or miss.empty:
            ax.set_title(f"{method.upper()}\n(Insufficient data)")
            continue
        
        merged = obs.merge(miss, on=['lake_id_cci', 'site_id'], suffixes=('_obs', '_miss'))
        merged['gap_penalty'] = merged['rmse_miss'] - merged['rmse_obs']
        
        # Histogram of gap penalty
        ax.hist(merged['gap_penalty'], bins=20, edgecolor='black', alpha=0.7,
               color='#e74c3c')
        
        mean_penalty = merged['gap_penalty'].mean()
        median_penalty = merged['gap_penalty'].median()
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='No penalty')
        ax.axvline(mean_penalty, color='blue', linestyle='--', label=f'Mean: {mean_penalty:.3f}°C')
        ax.axvline(median_penalty, color='green', linestyle='-.', label=f'Median: {median_penalty:.3f}°C')
        
        # Count positive/negative
        n_positive = (merged['gap_penalty'] > 0).sum()
        n_negative = (merged['gap_penalty'] < 0).sum()
        
        ax.set_xlabel('Gap Penalty (Gap-fill RMSE - Observed RMSE) (°C)')
        ax.set_ylabel('Number of Lakes')
        ax.set_title(f'{method.upper()}\nGap-fill worse: {n_positive}, Gap-fill better: {n_negative}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Gap-Filling Penalty Analysis\n(Positive = gap-fill performs worse than observed)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "gap_penalty_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_per_lake_winner_analysis(df: pd.DataFrame, output_dir: str):
    """Analyze which method wins per lake (aggregating across sites if multiple)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Aggregate to per-lake level (average across sites if multiple)
    lake_stats = df.groupby(['lake_id_cci', 'method', 'data_type']).agg({
        'rmse': 'mean',
        'bias': 'mean', 
        'correlation': 'mean',
        'n_matches': 'sum'
    }).reset_index()
    
    # --- Plot 1: DINEOF vs DINCAE on ALL reconstruction (per lake) ---
    ax = axes[0, 0]
    dineof_all = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction')][['lake_id_cci', 'rmse']]
    dincae_all = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction')][['lake_id_cci', 'rmse']]
    
    if not dineof_all.empty and not dincae_all.empty:
        merged = dineof_all.merge(dincae_all, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        
        if not merged.empty:
            ax.scatter(merged['rmse_dineof'], merged['rmse_dincae'], alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
            max_val = max(merged['rmse_dineof'].max(), merged['rmse_dincae'].max()) * 1.1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            ax.set_xlabel('DINEOF RMSE (°C)')
            ax.set_ylabel('DINCAE RMSE (°C)')
            ax.set_title(f'All Reconstruction - Per Lake\n(N={len(merged)} lakes)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            ax.text(0.05, 0.95, f'DINEOF better: {dineof_wins} lakes\nDINCAE better: {dincae_wins} lakes',
                   transform=ax.transAxes, fontsize=11, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Plot 2: DINEOF vs DINCAE on GAP-FILL only (per lake) ---
    ax = axes[0, 1]
    dineof_gap = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    dincae_gap = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    
    if not dineof_gap.empty and not dincae_gap.empty:
        merged = dineof_gap.merge(dincae_gap, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        
        if not merged.empty:
            ax.scatter(merged['rmse_dineof'], merged['rmse_dincae'], alpha=0.6, s=60, 
                      edgecolors='black', linewidth=0.5, c='orange')
            max_val = max(merged['rmse_dineof'].max(), merged['rmse_dincae'].max()) * 1.1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            ax.set_xlabel('DINEOF RMSE (°C)')
            ax.set_ylabel('DINCAE RMSE (°C)')
            ax.set_title(f'Gap-Fill Only - Per Lake\n(N={len(merged)} lakes)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            ax.text(0.05, 0.95, f'DINEOF better: {dineof_wins} lakes\nDINCAE better: {dincae_wins} lakes',
                   transform=ax.transAxes, fontsize=11, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Plot 3: Gap Penalty per lake - DINEOF ---
    ax = axes[1, 0]
    dineof_obs = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction_observed')][['lake_id_cci', 'rmse']]
    dineof_miss = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    
    if not dineof_obs.empty and not dineof_miss.empty:
        merged = dineof_obs.merge(dineof_miss, on='lake_id_cci', suffixes=('_obs', '_miss'))
        
        if not merged.empty:
            merged['gap_penalty'] = merged['rmse_miss'] - merged['rmse_obs']
            
            ax.hist(merged['gap_penalty'], bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='black', linewidth=2, label='No penalty')
            mean_pen = merged['gap_penalty'].mean()
            median_pen = merged['gap_penalty'].median()
            ax.axvline(mean_pen, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_pen:.3f}°C')
            ax.axvline(median_pen, color='red', linestyle=':', linewidth=2, label=f'Median: {median_pen:.3f}°C')
            
            n_worse = (merged['gap_penalty'] > 0).sum()
            n_better = (merged['gap_penalty'] < 0).sum()
            
            ax.set_xlabel('Gap Penalty (Gap-fill RMSE − Observed RMSE) (°C)')
            ax.set_ylabel('Number of Lakes')
            ax.set_title(f'DINEOF Gap Penalty (Per Lake)\nGap-fill worse: {n_worse} lakes, Gap-fill better: {n_better} lakes')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 4: Gap Penalty per lake - DINCAE ---
    ax = axes[1, 1]
    dincae_obs = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction_observed')][['lake_id_cci', 'rmse']]
    dincae_miss = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    
    if not dincae_obs.empty and not dincae_miss.empty:
        merged = dincae_obs.merge(dincae_miss, on='lake_id_cci', suffixes=('_obs', '_miss'))
        
        if not merged.empty:
            merged['gap_penalty'] = merged['rmse_miss'] - merged['rmse_obs']
            
            ax.hist(merged['gap_penalty'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='black', linewidth=2, label='No penalty')
            mean_pen = merged['gap_penalty'].mean()
            median_pen = merged['gap_penalty'].median()
            ax.axvline(mean_pen, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_pen:.3f}°C')
            ax.axvline(median_pen, color='red', linestyle=':', linewidth=2, label=f'Median: {median_pen:.3f}°C')
            
            n_worse = (merged['gap_penalty'] > 0).sum()
            n_better = (merged['gap_penalty'] < 0).sum()
            
            ax.set_xlabel('Gap Penalty (Gap-fill RMSE − Observed RMSE) (°C)')
            ax.set_ylabel('Number of Lakes')
            ax.set_title(f'DINCAE Gap Penalty (Per Lake)\nGap-fill worse: {n_worse} lakes, Gap-fill better: {n_better} lakes')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Per-Lake Analysis (Sites Averaged)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "per_lake_winner_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Also save the per-lake winner summary to CSV
    summary_rows = []
    
    # All reconstruction comparison
    dineof_all = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction')][['lake_id_cci', 'rmse', 'n_matches']]
    dincae_all = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction')][['lake_id_cci', 'rmse', 'n_matches']]
    
    if not dineof_all.empty and not dincae_all.empty:
        merged = dineof_all.merge(dincae_all, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        for _, row in merged.iterrows():
            summary_rows.append({
                'lake_id_cci': row['lake_id_cci'],
                'comparison': 'all_reconstruction',
                'dineof_rmse': row['rmse_dineof'],
                'dincae_rmse': row['rmse_dincae'],
                'winner': 'dineof' if row['rmse_dineof'] < row['rmse_dincae'] else 'dincae',
                'difference': row['rmse_dincae'] - row['rmse_dineof'],  # positive = DINEOF better
            })
    
    # Gap-fill comparison
    dineof_gap = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    dincae_gap = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    
    if not dineof_gap.empty and not dincae_gap.empty:
        merged = dineof_gap.merge(dincae_gap, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        for _, row in merged.iterrows():
            summary_rows.append({
                'lake_id_cci': row['lake_id_cci'],
                'comparison': 'gap_fill_only',
                'dineof_rmse': row['rmse_dineof'],
                'dincae_rmse': row['rmse_dincae'],
                'winner': 'dineof' if row['rmse_dineof'] < row['rmse_dincae'] else 'dincae',
                'difference': row['rmse_dincae'] - row['rmse_dineof'],
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "per_lake_winner_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")


def create_comprehensive_summary_figure(df: pd.DataFrame, agg_df: pd.DataFrame, output_dir: str):
    """Create a single comprehensive summary figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Overall RMSE comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    recon_data = agg_df[agg_df['data_type'] == 'reconstruction'].copy()
    if not recon_data.empty:
        methods = recon_data['method'].tolist()
        rmse_vals = recon_data['rmse_weighted'].tolist()
        colors = ['#2ecc71' if 'dineof' in m else '#3498db' if 'dincae' in m else '#9b59b6' for m in methods]
        bars = ax1.bar(methods, rmse_vals, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('RMSE (°C)')
        ax1.set_title('Overall Reconstruction RMSE')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, rmse_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Observed vs Gap-fill (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    obs_miss_data = agg_df[agg_df['data_type'].isin(['reconstruction_observed', 'reconstruction_missing'])]
    if not obs_miss_data.empty:
        methods = ['dineof', 'dincae']
        x = np.arange(len(methods))
        width = 0.35
        
        obs_vals = []
        miss_vals = []
        for method in methods:
            obs_row = obs_miss_data[(obs_miss_data['method'] == method) & (obs_miss_data['data_type'] == 'reconstruction_observed')]
            miss_row = obs_miss_data[(obs_miss_data['method'] == method) & (obs_miss_data['data_type'] == 'reconstruction_missing')]
            obs_vals.append(obs_row['rmse_weighted'].values[0] if not obs_row.empty else 0)
            miss_vals.append(miss_row['rmse_weighted'].values[0] if not miss_row.empty else 0)
        
        ax2.bar(x - width/2, obs_vals, width, label='Observed', color='#2ecc71', alpha=0.8)
        ax2.bar(x + width/2, miss_vals, width, label='Gap-fill', color='#e74c3c', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.set_ylabel('RMSE (°C)')
        ax2.set_title('Observed vs Gap-Fill RMSE')
        ax2.legend()
    
    # 3. Number of lakes/points (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if not agg_df.empty:
        summary_data = agg_df[agg_df['data_type'] == 'reconstruction'][['method', 'n_lakes', 'n_total_points']]
        if not summary_data.empty:
            x = np.arange(len(summary_data))
            ax3.bar(x, summary_data['n_total_points'], color='#3498db', alpha=0.8)
            ax3.set_xticks(x)
            ax3.set_xticklabels(summary_data['method'], rotation=45)
            ax3.set_ylabel('Total Validation Points')
            ax3.set_title('Data Volume by Method')
            
            # Add lake count as text
            for i, (_, row) in enumerate(summary_data.iterrows()):
                ax3.text(i, row['n_total_points'] + 100, f"({int(row['n_lakes'])} lakes)",
                        ha='center', fontsize=9)
    
    # 4. DINEOF vs DINCAE scatter (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    dineof = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction')][['lake_id_cci', 'site_id', 'rmse']]
    dincae = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction')][['lake_id_cci', 'site_id', 'rmse']]
    if not dineof.empty and not dincae.empty:
        merged = dineof.merge(dincae, on=['lake_id_cci', 'site_id'], suffixes=('_dineof', '_dincae'))
        if not merged.empty:
            ax4.scatter(merged['rmse_dineof'], merged['rmse_dincae'], alpha=0.6, edgecolors='black', linewidth=0.5)
            max_val = max(merged['rmse_dineof'].max(), merged['rmse_dincae'].max()) * 1.1
            ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            ax4.set_xlabel('DINEOF RMSE (°C)')
            ax4.set_ylabel('DINCAE RMSE (°C)')
            ax4.set_title('Per-Lake: DINEOF vs DINCAE')
            ax4.set_aspect('equal', adjustable='box')
            
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            ax4.text(0.05, 0.95, f'DINEOF better: {dineof_wins}\nDINCAE better: {dincae_wins}',
                    transform=ax4.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. RMSE distribution (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    dineof_rmse = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction')]['rmse']
    dincae_rmse = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction')]['rmse']
    if not dineof_rmse.empty or not dincae_rmse.empty:
        data_to_plot = []
        labels = []
        if not dineof_rmse.empty:
            data_to_plot.append(dineof_rmse.values)
            labels.append('DINEOF')
        if not dincae_rmse.empty:
            data_to_plot.append(dincae_rmse.values)
            labels.append('DINCAE')
        
        bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['#2ecc71', '#3498db']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_ylabel('RMSE (°C)')
        ax5.set_title('RMSE Distribution')
    
    # 6. Correlation comparison (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    corr_data = df[df['data_type'] == 'reconstruction'].groupby('method')['correlation'].mean()
    if not corr_data.empty:
        methods = corr_data.index.tolist()
        corr_vals = corr_data.values
        colors = ['#2ecc71' if 'dineof' in m else '#3498db' if 'dincae' in m else '#9b59b6' for m in methods]
        ax6.bar(methods, corr_vals, color=colors, alpha=0.8, edgecolor='black')
        ax6.set_ylabel('Correlation (R)')
        ax6.set_title('Mean Correlation')
        ax6.tick_params(axis='x', rotation=45)
        ax6.set_ylim(0, 1)
    
    # 7. Gap penalty histogram (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    obs_dineof = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction_observed')][['lake_id_cci', 'site_id', 'rmse']]
    miss_dineof = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'site_id', 'rmse']]
    if not obs_dineof.empty and not miss_dineof.empty:
        merged = obs_dineof.merge(miss_dineof, on=['lake_id_cci', 'site_id'], suffixes=('_obs', '_miss'))
        if not merged.empty:
            gap_penalty = merged['rmse_miss'] - merged['rmse_obs']
            ax7.hist(gap_penalty, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax7.axvline(0, color='black', linewidth=2)
            ax7.axvline(gap_penalty.mean(), color='blue', linestyle='--', label=f'Mean: {gap_penalty.mean():.2f}°C')
            ax7.set_xlabel('Gap Penalty (°C)')
            ax7.set_ylabel('Number of Lakes')
            ax7.set_title('DINEOF Gap-Fill Penalty')
            ax7.legend()
    
    # 8. Gap penalty DINCAE (bottom middle)
    ax8 = fig.add_subplot(gs[2, 1])
    obs_dincae = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction_observed')][['lake_id_cci', 'site_id', 'rmse']]
    miss_dincae = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'site_id', 'rmse']]
    if not obs_dincae.empty and not miss_dincae.empty:
        merged = obs_dincae.merge(miss_dincae, on=['lake_id_cci', 'site_id'], suffixes=('_obs', '_miss'))
        if not merged.empty:
            gap_penalty = merged['rmse_miss'] - merged['rmse_obs']
            ax8.hist(gap_penalty, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
            ax8.axvline(0, color='black', linewidth=2)
            ax8.axvline(gap_penalty.mean(), color='red', linestyle='--', label=f'Mean: {gap_penalty.mean():.2f}°C')
            ax8.set_xlabel('Gap Penalty (°C)')
            ax8.set_ylabel('Number of Lakes')
            ax8.set_title('DINCAE Gap-Fill Penalty')
            ax8.legend()
    
    # 9. Bias comparison (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])
    bias_data = agg_df[agg_df['data_type'] == 'reconstruction'][['method', 'bias_weighted']]
    if not bias_data.empty:
        methods = bias_data['method'].tolist()
        bias_vals = bias_data['bias_weighted'].tolist()
        colors = ['#2ecc71' if b < 0 else '#e74c3c' for b in bias_vals]
        ax9.bar(methods, bias_vals, color=colors, alpha=0.8, edgecolor='black')
        ax9.axhline(0, color='black', linestyle='-')
        ax9.set_ylabel('Bias (°C)')
        ax9.set_title('Mean Bias (Satellite - In-situ)')
        ax9.tick_params(axis='x', rotation=45)
    
    plt.suptitle('In-Situ Validation Comprehensive Summary', fontsize=16, fontweight='bold', y=1.02)
    
    save_path = os.path.join(output_dir, "comprehensive_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_per_lake_table(df: pd.DataFrame, output_dir: str):
    """Create detailed per-lake statistics table."""
    # Pivot to get DINEOF vs DINCAE side by side
    pivot_data = []
    
    lakes = df['lake_id_cci'].unique()
    
    for lake in lakes:
        lake_df = df[df['lake_id_cci'] == lake]
        sites = lake_df['site_id'].unique()
        
        for site in sites:
            site_df = lake_df[lake_df['site_id'] == site]
            
            row = {'lake_id_cci': lake, 'site_id': site}
            
            # Get stats for each method/data_type combo
            for method in ['dineof', 'dincae']:
                for data_type in ['reconstruction', 'reconstruction_observed', 'reconstruction_missing']:
                    subset = site_df[(site_df['method'] == method) & (site_df['data_type'] == data_type)]
                    
                    prefix = f"{method}_{data_type.replace('reconstruction', 'recon').replace('_', '')}"
                    
                    if not subset.empty:
                        row[f'{prefix}_rmse'] = subset['rmse'].values[0]
                        row[f'{prefix}_n'] = subset['n_matches'].values[0]
                    else:
                        row[f'{prefix}_rmse'] = np.nan
                        row[f'{prefix}_n'] = 0
            
            pivot_data.append(row)
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Save to CSV
    save_path = os.path.join(output_dir, "per_lake_comparison.csv")
    pivot_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    
    return pivot_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze in-situ validation results across all lakes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_insitu_validation.py --run_root /path/to/experiment
    python analyze_insitu_validation.py --run_root /path/to/experiment --alpha a1000
    python analyze_insitu_validation.py --run_root /path/to/experiment --output_dir /path/to/output
        """
    )
    
    parser.add_argument("--run_root", required=True, help="Root directory of the experiment run")
    parser.add_argument("--alpha", default=None, help="Specific alpha to analyze (e.g., a1000)")
    parser.add_argument("--output_dir", default=None, help="Output directory for results (default: {run_root}/analysis)")
    parser.add_argument("--lake_metadata", default=None, help="Optional CSV with lake metadata (lat, lon, area, etc.)")
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "insitu_validation_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("IN-SITU VALIDATION ANALYSIS")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Alpha: {args.alpha or 'all'}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Find all validation CSVs
    csv_files = find_validation_csvs(args.run_root, args.alpha)
    print(f"\nFound {len(csv_files)} validation CSV files")
    
    if not csv_files:
        print("ERROR: No validation CSV files found!")
        print(f"Expected pattern: {args.run_root}/post/*/{'*' if not args.alpha else args.alpha}/insitu_cv_validation/*_insitu_stats_site*.csv")
        sys.exit(1)
    
    # Load all stats
    print("\nLoading validation statistics...")
    df = load_all_stats(csv_files)
    
    if df.empty:
        print("ERROR: Could not load any statistics!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} records from {df['lake_id_cci'].nunique()} lakes")
    
    # Save aggregated raw data
    raw_path = os.path.join(args.output_dir, "all_insitu_stats_combined.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved combined data: {raw_path}")
    
    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    agg_df = compute_aggregate_stats(df)
    
    agg_path = os.path.join(args.output_dir, "aggregate_statistics.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregate stats: {agg_path}")
    
    # Generate text report
    print("\n" + "=" * 70)
    report_path = generate_summary_report(df, agg_df, args.output_dir)
    print("=" * 70)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    create_method_comparison_plot(df, args.output_dir)
    create_dineof_vs_dincae_scatter(df, args.output_dir)
    create_observed_vs_gapfill_comparison(df, args.output_dir)
    create_rmse_distribution_plot(df, args.output_dir)
    create_correlation_comparison(df, args.output_dir)
    create_gap_penalty_analysis(df, args.output_dir)
    create_per_lake_winner_analysis(df, args.output_dir)
    create_comprehensive_summary_figure(df, agg_df, args.output_dir)
    
    # Create per-lake comparison table
    print("\nGenerating per-lake comparison table...")
    create_per_lake_table(df, args.output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()