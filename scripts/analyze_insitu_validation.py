#!/usr/bin/env python3
"""
In-Situ Validation Analysis Script

Collects all in-situ validation results across all lakes and generates:
1. Aggregated statistics CSV
2. Summary report (text + markdown)
3. Clear, readable visualizations comparing DINEOF vs DINCAE

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
from matplotlib.patches import Patch
from scipy import stats

# Try to import cartopy for global maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: Cartopy not available - global maps will be skipped")


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

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
            df['source_file'] = os.path.basename(csv_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================

def compute_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics across all lakes for each method/data_type combination."""
    if df.empty:
        return pd.DataFrame()
    
    grouped = df.groupby(['method', 'data_type'])
    
    agg_stats = []
    for (method, data_type), group in grouped:
        n_lakes = group['lake_id_cci'].nunique()
        n_points = group['n_matches'].sum()
        
        weights = group['n_matches'].values
        if weights.sum() > 0:
            weighted_rmse = np.average(group['rmse'].values, weights=weights)
            weighted_bias = np.average(group['bias'].values, weights=weights)
            weighted_mae = np.average(group['mae'].values, weights=weights)
        else:
            weighted_rmse = np.nan
            weighted_bias = np.nan
            weighted_mae = np.nan
        
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
            'correlation_mean': group['correlation'].mean() if 'correlation' in group.columns else np.nan,
            'correlation_median': group['correlation'].median() if 'correlation' in group.columns else np.nan,
        })
    
    return pd.DataFrame(agg_stats)


def compute_comprehensive_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global stats (all 6 metrics) for all methods/data_types.
    TASK 2: Includes all 6 metrics with proper weighting by n_matches.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd', 'correlation']
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 
                  'reconstruction_missing', 'observation_cropped']
    
    results = []
    
    for data_type in data_types:
        for method in df['method'].unique():
            subset = df[(df['method'] == method) & (df['data_type'] == data_type)]
            
            if subset.empty:
                continue
            
            row = {
                'method': method,
                'data_type': data_type,
                'n_lakes': subset['lake_id_cci'].nunique(),
                'n_sites': len(subset),
                'n_total_points': subset['n_matches'].sum(),
            }
            
            weights = subset['n_matches'].values
            total_weight = weights.sum()
            
            for metric in metrics:
                if metric in subset.columns and total_weight > 0:
                    valid = subset[~subset[metric].isna()]
                    if len(valid) > 0:
                        w = valid['n_matches'].values
                        if w.sum() > 0:
                            row[f'{metric}_weighted'] = np.average(valid[metric].values, weights=w)
                        else:
                            row[f'{metric}_weighted'] = np.nan
                        row[f'{metric}_mean'] = valid[metric].mean()
                        row[f'{metric}_std'] = valid[metric].std()
                        row[f'{metric}_median'] = valid[metric].median()
                    else:
                        for suffix in ['_weighted', '_mean', '_std', '_median']:
                            row[f'{metric}{suffix}'] = np.nan
                else:
                    for suffix in ['_weighted', '_mean', '_std', '_median']:
                        row[f'{metric}{suffix}'] = np.nan
            
            results.append(row)
    
    return pd.DataFrame(results)


def create_per_lake_stats_table(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Create comprehensive per-lake statistics table with all 6 metrics."""
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd', 'correlation', 'n_matches']
    
    per_lake_data = []
    
    for (lake, site, method, data_type), group in df.groupby(['lake_id_cci', 'site_id', 'method', 'data_type']):
        row = {
            'lake_id_cci': lake,
            'site_id': site,
            'method': method,
            'data_type': data_type,
        }
        
        for metric in metrics:
            if metric in group.columns:
                row[metric] = group[metric].values[0]
            else:
                row[metric] = np.nan
        
        per_lake_data.append(row)
    
    per_lake_df = pd.DataFrame(per_lake_data)
    per_lake_df = per_lake_df.sort_values(['lake_id_cci', 'site_id', 'method', 'data_type'])
    
    save_path = os.path.join(output_dir, 'per_lake_insitu_stats.csv')
    per_lake_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    
    return per_lake_df


# =============================================================================
# GLOBAL MAPS (TASK 2 - Cartopy)
# =============================================================================

def create_global_metric_maps(df: pd.DataFrame, lake_locations: pd.DataFrame, output_dir: str):
    """Create global scatter maps for key metric/data_type combinations."""
    if not HAS_CARTOPY:
        print("Warning: Cartopy not available, skipping global maps")
        return
    
    df_merged = df.merge(lake_locations[['lake_id', 'lat', 'lon']], 
                         left_on='lake_id_cci', right_on='lake_id', how='left')
    df_merged = df_merged[df_merged['lat'].notna()]
    
    if df_merged.empty:
        print("Warning: No lakes matched with location data for global maps")
        return
    
    # Focus on key combinations
    plot_configs = [
        ('reconstruction', 'rmse', 'YlOrRd'),
        ('reconstruction_missing', 'rmse', 'YlOrRd'),
        ('observation', 'rmse', 'YlOrRd'),
        ('reconstruction', 'bias', 'RdBu_r'),
    ]
    
    maps_created = 0
    for data_type, metric, cmap in plot_configs:
        try:
            df_dt = df_merged[df_merged['data_type'] == data_type]
            
            if df_dt.empty or metric not in df_dt.columns:
                continue
            
            # Aggregate to per-lake level
            lake_stats = df_dt.groupby(['lake_id_cci', 'lat', 'lon']).agg({
                metric: 'mean', 
                'n_matches': 'sum'
            }).reset_index()
            
            if len(lake_stats) == 0:
                continue
            
            fig, ax = plt.subplots(figsize=(14, 9),
                                   subplot_kw={'projection': ccrs.Robinson()})
            ax.set_global()
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='lightgray')
            
            values = lake_stats[metric].dropna()
            if len(values) == 0:
                continue
            
            if metric in ['bias', 'median']:
                vmax = max(abs(values.quantile(0.05)), abs(values.quantile(0.95)))
                vmin = -vmax
            else:
                vmin = values.quantile(0.02)
                vmax = values.quantile(0.98)
            
            sizes = np.log1p(lake_stats['n_matches']) * 15 + 30
            
            sc = ax.scatter(lake_stats['lon'], lake_stats['lat'],
                           c=lake_stats[metric], s=sizes,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           alpha=0.85, edgecolors='black', linewidth=0.5,
                           transform=ccrs.PlateCarree(), zorder=5)
            
            cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label(f'{metric.upper()} (°C)', fontsize=11)
            
            data_type_label = data_type.replace('_', ' ').title()
            ax.set_title(f'In-Situ Validation: {metric.upper()} - {data_type_label}\n'
                        f'(N={len(lake_stats)} lakes)', fontsize=13, fontweight='bold')
            
            save_path = os.path.join(output_dir, f'global_map_{data_type}_{metric}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            maps_created += 1
            print(f"  Created: global_map_{data_type}_{metric}.png")
            
        except Exception as e:
            print(f"  Error creating map for {data_type}/{metric}: {e}")
            plt.close('all')
    
    print(f"Created {maps_created} global maps")


# =============================================================================
# READABLE COMPARISON PLOTS (IMPROVED)
# =============================================================================

def create_method_comparison_by_lake(df: pd.DataFrame, output_dir: str,
                                      method1: str = 'dineof', 
                                      method2: str = 'dincae'):
    """
    Create clear, readable method comparison plots (like the original RMSE plot).
    Generates separate plots for each metric and data_type.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']
    
    colors = {method1: '#5DA5DA', method2: '#FAA43A'}  # Blue, Orange
    
    for data_type in data_types:
        df_dt = df[df['data_type'] == data_type]
        
        if df_dt.empty:
            continue
        
        available_methods = df_dt['method'].unique()
        if method1 not in available_methods or method2 not in available_methods:
            continue
        
        for metric in metrics:
            if metric not in df_dt.columns:
                continue
            
            pivot = df_dt.pivot_table(
                index='lake_id_cci',
                columns='method',
                values=metric,
                aggfunc='mean'
            ).reset_index()
            
            if method1 not in pivot.columns or method2 not in pivot.columns:
                continue
            
            pivot = pivot.sort_values('lake_id_cci')
            pivot = pivot.dropna(subset=[method1, method2])
            
            if len(pivot) == 0:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(14, len(pivot) * 0.5), 6))
            
            x = np.arange(len(pivot))
            width = 0.35
            
            ax.bar(x - width/2, pivot[method1], width, 
                  label=method1.upper(), color=colors[method1], 
                  edgecolor='white', linewidth=0.5)
            ax.bar(x + width/2, pivot[method2], width,
                  label=method2.upper(), color=colors[method2],
                  edgecolor='white', linewidth=0.5)
            
            ax.set_xlabel('Lake ID', fontsize=12)
            ax.set_ylabel(f'{metric.upper()} (°C)', fontsize=12)
            
            data_type_label = data_type.replace('_', ' ').title()
            ax.set_title(f'In-Situ Validation: {method1.upper()} vs {method2.upper()} {metric.upper()} by Lake\n'
                        f'Data Type: {data_type_label}', fontsize=14, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(lid)) for lid in pivot['lake_id_cci']], 
                              rotation=45, ha='right', fontsize=9)
            
            if metric in ['bias', 'median']:
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Count winners
            n_lakes = len(pivot)
            if metric in ['rmse', 'mae', 'std', 'rstd']:
                m1_better = (pivot[method1] < pivot[method2]).sum()
            else:
                m1_better = (abs(pivot[method1]) < abs(pivot[method2])).sum()
            m2_better = n_lakes - m1_better
            
            ax.text(0.02, 0.98, f'{method1.upper()} better: {m1_better}/{n_lakes} lakes\n'
                               f'{method2.upper()} better: {m2_better}/{n_lakes} lakes',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            save_path = f'{output_dir}/{method1}_vs_{method2}_{metric}_{data_type}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved: {method1}_vs_{method2}_{metric}_{data_type}.png")


def create_diagnostic_comparison_plot(df: pd.DataFrame, output_dir: str,
                                       method1: str = 'dineof',
                                       method2: str = 'dincae',
                                       recon_data_type: str = 'reconstruction'):
    """
    KEY DIAGNOSTIC PLOT: Method comparison with observation quality below.
    
    Top panel: DINEOF vs DINCAE for reconstruction
    Bottom panel: Observation RMSE per lake (satellite vs in-situ quality)
    
    This answers: Do lakes where DINCAE wins have worse observation quality?
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    
    colors = {method1: '#5DA5DA', method2: '#FAA43A', 'observation': '#60BD68'}
    
    for metric in metrics:
        df_recon = df[df['data_type'] == recon_data_type]
        df_obs = df[df['data_type'] == 'observation']
        
        if df_recon.empty or df_obs.empty:
            continue
        
        pivot_recon = df_recon.pivot_table(
            index='lake_id_cci',
            columns='method',
            values=metric,
            aggfunc='mean'
        ).reset_index()
        
        pivot_obs = df_obs.groupby('lake_id_cci')[metric].mean().reset_index()
        pivot_obs.columns = ['lake_id_cci', f'obs_{metric}']
        
        if method1 not in pivot_recon.columns or method2 not in pivot_recon.columns:
            continue
        
        merged = pivot_recon[['lake_id_cci', method1, method2]].merge(
            pivot_obs, on='lake_id_cci', how='inner'
        )
        merged = merged.dropna()
        merged = merged.sort_values('lake_id_cci')
        
        if len(merged) < 3:
            continue
        
        # Create 2-panel figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(merged) * 0.5), 10),
                                        sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        
        x = np.arange(len(merged))
        width = 0.35
        
        # Top panel: Method comparison
        ax1.bar(x - width/2, merged[method1], width, label=method1.upper(), 
               color=colors[method1], edgecolor='white', linewidth=0.5)
        ax1.bar(x + width/2, merged[method2], width, label=method2.upper(),
               color=colors[method2], edgecolor='white', linewidth=0.5)
        
        ax1.set_ylabel(f'{metric.upper()} (°C)', fontsize=12)
        recon_label = recon_data_type.replace('_', ' ').title()
        ax1.set_title(f'{method1.upper()} vs {method2.upper()} - {recon_label} - {metric.upper()}\n'
                     f'(Compare with observation quality below)', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add zero line for bias/median
        if metric in ['bias', 'median']:
            ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Mark winners (for unsigned metrics, lower is better; for signed, closer to 0 is better)
        for i, (_, row) in enumerate(merged.iterrows()):
            if metric in ['rmse', 'mae', 'std', 'rstd']:
                # Lower is better
                if row[method1] < row[method2]:
                    winner_val = row[method1] * 0.95
                    winner_color = 'blue'
                else:
                    winner_val = row[method2] * 0.95
                    winner_color = 'orange'
            else:
                # Closer to 0 is better (bias, median)
                if abs(row[method1]) < abs(row[method2]):
                    winner_val = row[method1]
                    winner_color = 'blue'
                else:
                    winner_val = row[method2]
                    winner_color = 'orange'
            ax1.plot(i, winner_val, 'v', color=winner_color, markersize=8)
        
        # Bottom panel: Observation quality
        ax2.bar(x, merged[f'obs_{metric}'], width*2, label='Observation vs In-Situ',
               color=colors['observation'], edgecolor='white', linewidth=0.5)
        
        ax2.set_xlabel('Lake ID', fontsize=12)
        ax2.set_ylabel(f'Observation {metric.upper()} (°C)', fontsize=12)
        ax2.set_title(f'Satellite Observation vs In-Situ Quality\n'
                     f'(Higher = worse satellite-buoy match)', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add zero line for bias/median
        if metric in ['bias', 'median']:
            ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(int(lid)) for lid in merged['lake_id_cci']], 
                           rotation=45, ha='right', fontsize=9)
        
        # Count winners
        if metric in ['rmse', 'mae', 'std', 'rstd']:
            m1_wins = (merged[method1] < merged[method2]).sum()
        else:
            m1_wins = (abs(merged[method1]) < abs(merged[method2])).sum()
        m2_wins = len(merged) - m1_wins
        
        # Add winner count to top panel
        ax1.text(0.02, 0.98, f'{method1.upper()} wins: {m1_wins}/{len(merged)}\n'
                            f'{method2.upper()} wins: {m2_wins}/{len(merged)}',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Compute correlation
        merged['method_diff'] = merged[method1] - merged[method2]  # Positive = method2 better
        corr = merged['method_diff'].corr(merged[f'obs_{metric}'])
        
        if corr > 0.3:
            interpretation = f"→ {method2.upper()} tends to win on lakes with WORSE obs quality (r={corr:.2f})"
            color = 'red'
        elif corr < -0.3:
            interpretation = f"→ {method1.upper()} tends to win on lakes with WORSE obs quality (r={corr:.2f})"
            color = 'red'
        else:
            interpretation = f"→ No strong correlation between method advantage and obs quality (r={corr:.2f})"
            color = 'green'
        
        fig.text(0.5, 0.02, interpretation,
                ha='center', fontsize=12, style='italic', color=color,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        save_path = f'{output_dir}/diagnostic_{method1}_vs_{method2}_{metric}_{recon_data_type}_with_obs.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: diagnostic_{method1}_vs_{method2}_{metric}_{recon_data_type}_with_obs.png")


def create_observed_vs_missing_plot(df: pd.DataFrame, output_dir: str, method: str = 'dineof'):
    """
    Compare reconstruction_observed vs reconstruction_missing for a single method.
    Shows the "gap-fill penalty" - how much worse at truly missing pixels.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    
    colors = {'observed': '#5DA5DA', 'missing': '#F17CB0'}
    
    for metric in metrics:
        df_obs = df[(df['data_type'] == 'reconstruction_observed') & (df['method'] == method)]
        df_miss = df[(df['data_type'] == 'reconstruction_missing') & (df['method'] == method)]
        
        if df_obs.empty or df_miss.empty:
            continue
        
        if metric not in df_obs.columns:
            continue
        
        obs_stats = df_obs.groupby('lake_id_cci')[metric].mean().reset_index()
        obs_stats.columns = ['lake_id_cci', 'observed']
        
        miss_stats = df_miss.groupby('lake_id_cci')[metric].mean().reset_index()
        miss_stats.columns = ['lake_id_cci', 'missing']
        
        merged = obs_stats.merge(miss_stats, on='lake_id_cci', how='inner')
        merged = merged.dropna().sort_values('lake_id_cci')
        
        if len(merged) < 3:
            continue
        
        fig, ax = plt.subplots(figsize=(max(14, len(merged) * 0.5), 6))
        
        x = np.arange(len(merged))
        width = 0.35
        
        ax.bar(x - width/2, merged['observed'], width, 
              label='Observed Pixels (training overlap)', color=colors['observed'],
              edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, merged['missing'], width,
              label='Missing Pixels (true gap-fill)', color=colors['missing'],
              edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Lake ID', fontsize=12)
        ax.set_ylabel(f'{metric.upper()} (°C)', fontsize=12)
        ax.set_title(f'{method.upper()}: Observed vs Missing Pixel Performance - {metric.upper()}\n'
                    f'(Comparison between training-overlap pixels vs true gap-filled pixels)',
                    fontsize=14, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(lid)) for lid in merged['lake_id_cci']],
                          rotation=45, ha='right', fontsize=9)
        
        # Add zero line for bias/median
        if metric in ['bias', 'median']:
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        avg_obs = merged['observed'].mean()
        avg_miss = merged['missing'].mean()
        
        # For unsigned metrics, show gap-fill penalty
        if metric in ['rmse', 'mae', 'std', 'rstd']:
            penalty = avg_miss - avg_obs
            ax.text(0.02, 0.98, f'Avg Observed: {avg_obs:.3f}°C\n'
                               f'Avg Missing: {avg_miss:.3f}°C\n'
                               f'Gap-fill penalty: {penalty:+.3f}°C',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # For signed metrics, just show averages
            ax.text(0.02, 0.98, f'Avg Observed: {avg_obs:.3f}°C\n'
                               f'Avg Missing: {avg_miss:.3f}°C',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        save_path = f'{output_dir}/{method}_observed_vs_missing_{metric}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {method}_observed_vs_missing_{metric}.png")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(df: pd.DataFrame, agg_df: pd.DataFrame, output_dir: str) -> str:
    """Generate a text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("IN-SITU VALIDATION SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    n_lakes = df['lake_id_cci'].nunique()
    n_sites = len(df.groupby(['lake_id_cci', 'site_id']))
    methods = df['method'].unique()
    
    report_lines.append("OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total lakes with in-situ validation: {n_lakes}")
    report_lines.append(f"Total validation sites: {n_sites}")
    report_lines.append(f"Methods analyzed: {', '.join(methods)}")
    report_lines.append("")
    
    # DINEOF vs DINCAE comparison
    report_lines.append("KEY FINDINGS - DINEOF vs DINCAE")
    report_lines.append("-" * 40)
    
    for data_type in ['reconstruction', 'reconstruction_missing']:
        dineof = agg_df[(agg_df['method'] == 'dineof') & (agg_df['data_type'] == data_type)]
        dincae = agg_df[(agg_df['method'] == 'dincae') & (agg_df['data_type'] == data_type)]
        
        if not dineof.empty and not dincae.empty:
            dineof_rmse = dineof['rmse_weighted'].values[0]
            dincae_rmse = dincae['rmse_weighted'].values[0]
            better = "DINEOF" if dineof_rmse < dincae_rmse else "DINCAE"
            diff = abs(dineof_rmse - dincae_rmse)
            
            label = "All Points" if data_type == 'reconstruction' else "Gap-Fill Only"
            report_lines.append(f"{label}:")
            report_lines.append(f"  DINEOF RMSE: {dineof_rmse:.3f}°C")
            report_lines.append(f"  DINCAE RMSE: {dincae_rmse:.3f}°C")
            report_lines.append(f"  → {better} is better by {diff:.3f}°C")
            report_lines.append("")
    
    # Per-lake winner counts
    report_lines.append("PER-LAKE WINNER ANALYSIS")
    report_lines.append("-" * 40)
    
    lake_stats = df.groupby(['lake_id_cci', 'method', 'data_type']).agg({'rmse': 'mean'}).reset_index()
    
    for data_type in ['reconstruction', 'reconstruction_missing']:
        dineof_dt = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == data_type)][['lake_id_cci', 'rmse']]
        dincae_dt = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == data_type)][['lake_id_cci', 'rmse']]
        
        if not dineof_dt.empty and not dincae_dt.empty:
            merged = dineof_dt.merge(dincae_dt, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
            if not merged.empty:
                dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
                dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
                total = len(merged)
                
                label = "All Points" if data_type == 'reconstruction' else "Gap-Fill Only"
                report_lines.append(f"{label} (N={total} lakes):")
                report_lines.append(f"  DINEOF better: {dineof_wins} lakes ({100*dineof_wins/total:.1f}%)")
                report_lines.append(f"  DINCAE better: {dincae_wins} lakes ({100*dincae_wins/total:.1f}%)")
                report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    report_path = os.path.join(output_dir, "insitu_validation_summary_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_path


def create_per_lake_table(df: pd.DataFrame, output_dir: str):
    """Create a simple per-lake comparison table CSV."""
    lake_stats = df.groupby(['lake_id_cci', 'method', 'data_type']).agg({
        'rmse': 'mean',
        'mae': 'mean',
        'bias': 'mean',
        'correlation': 'mean',
        'n_matches': 'sum'
    }).reset_index()
    
    save_path = os.path.join(output_dir, 'per_lake_comparison_table.csv')
    lake_stats.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze in-situ validation results across all lakes")
    parser.add_argument("--run_root", required=True, help="Path to run root directory")
    parser.add_argument("--alpha", default=None, help="Specific alpha slug (e.g., 'a1000')")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--lake_metadata", default=None, help="Optional CSV with lake metadata (lat, lon)")
    
    args = parser.parse_args()
    
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
    
    # Find and load data
    csv_files = find_validation_csvs(args.run_root, args.alpha)
    print(f"\nFound {len(csv_files)} validation CSV files")
    
    if not csv_files:
        print("ERROR: No validation CSV files found!")
        sys.exit(1)
    
    print("\nLoading validation statistics...")
    df = load_all_stats(csv_files)
    
    if df.empty:
        print("ERROR: Could not load any statistics!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} records from {df['lake_id_cci'].nunique()} lakes")
    
    # Save combined raw data
    raw_path = os.path.join(args.output_dir, "all_insitu_stats_combined.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved combined data: {raw_path}")
    
    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    agg_df = compute_aggregate_stats(df)
    agg_df.to_csv(os.path.join(args.output_dir, "aggregate_statistics.csv"), index=False)
    
    # Task 2: Comprehensive global stats
    print("\nComputing comprehensive global statistics (Task 2)...")
    global_stats = compute_comprehensive_global_stats(df)
    global_stats.to_csv(os.path.join(args.output_dir, 'global_insitu_stats_comprehensive.csv'), index=False)
    
    # Per-lake stats table
    print("\nGenerating per-lake stats table...")
    create_per_lake_stats_table(df, args.output_dir)
    
    # Load lake locations for global maps
    lake_locations = None
    possible_paths = [
        args.lake_metadata,
        os.path.join(os.path.dirname(__file__), '../src/processors/data/globolakes-static_lake_centre_fv1.csv'),
        os.path.join(os.path.dirname(__file__), 'globolakes-static_lake_centre_fv1.csv'),
    ]
    
    for loc_path in possible_paths:
        if loc_path and os.path.exists(loc_path):
            try:
                with open(loc_path, 'r', encoding='iso-8859-1') as f:
                    first_line = f.readline()
                
                if 'BADC-CSV' in first_line or 'Conventions,G' in first_line:
                    import csv
                    lake_data = []
                    with open(loc_path, 'r', encoding='iso-8859-1') as f:
                        reader = csv.reader(f)
                        row_num = 0
                        for line in reader:
                            row_num += 1
                            if row_num < 43:
                                continue
                            if line[0].strip() == 'end_data':
                                break
                            try:
                                lake_id = int(line[0].strip())
                                lat = float(line[3].strip())
                                lon = float(line[4].strip())
                                lake_data.append({'lake_id': lake_id, 'lat': lat, 'lon': lon})
                            except (ValueError, IndexError):
                                continue
                    lake_locations = pd.DataFrame(lake_data)
                else:
                    lake_locations = pd.read_csv(loc_path)
                    if 'Lake ID' in lake_locations.columns:
                        lake_locations = lake_locations.rename(columns={
                            'Lake ID': 'lake_id', 'Latitude, Centre': 'lat', 'Longitude, Centre': 'lon'
                        })
                    elif 'lake_id_cci' in lake_locations.columns:
                        lake_locations = lake_locations.rename(columns={'lake_id_cci': 'lake_id'})
                
                if lake_locations is not None and len(lake_locations) > 0:
                    print(f"Loaded lake locations: {len(lake_locations)} lakes")
                    break
            except Exception as e:
                print(f"Warning: Could not load {loc_path}: {e}")
                lake_locations = None
    
    # Task 2: Global maps
    if lake_locations is not None:
        print("\nGenerating global metric maps (Task 2)...")
        create_global_metric_maps(df, lake_locations, args.output_dir)
    else:
        print("\nSkipping global maps: No lake location data")
    
    # Generate summary report
    print("\n" + "=" * 70)
    generate_summary_report(df, agg_df, args.output_dir)
    print("=" * 70)
    
    # ==========================================================================
    # READABLE VISUALIZATIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GENERATING READABLE COMPARISON PLOTS")
    print("=" * 70)
    
    # 1. Method comparison plots (DINEOF vs DINCAE)
    print("\n[1/4] DINEOF vs DINCAE comparison plots (6 metrics × 4 data types)...")
    create_method_comparison_by_lake(df, args.output_dir, 'dineof', 'dincae')
    
    # 2. Diagnostic plots with observation quality
    print("\n[2/4] Diagnostic plots (method comparison + observation quality)...")
    create_diagnostic_comparison_plot(df, args.output_dir, 'dineof', 'dincae', 'reconstruction')
    create_diagnostic_comparison_plot(df, args.output_dir, 'dineof', 'dincae', 'reconstruction_missing')
    
    # 3. Observed vs Missing comparison
    print("\n[3/4] Gap-fill penalty plots (observed vs missing pixels)...")
    create_observed_vs_missing_plot(df, args.output_dir, 'dineof')
    create_observed_vs_missing_plot(df, args.output_dir, 'dincae')
    
    # 4. Additional method comparisons if available
    if 'eof_filtered' in df['method'].unique():
        print("\n[4/4] Additional method comparisons (eof_filtered)...")
        create_method_comparison_by_lake(df, args.output_dir, 'dineof', 'eof_filtered')
        create_observed_vs_missing_plot(df, args.output_dir, 'eof_filtered')
    else:
        print("\n[4/4] Skipping eof_filtered (not in data)")
    
    # Per-lake table
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
