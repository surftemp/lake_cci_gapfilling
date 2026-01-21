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

# Try to import cartopy for global maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: Cartopy not available - global maps will be skipped")


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


# ============================================================================
# TASK 2 & 3: Comprehensive Global Stats, Maps, and Multi-Panel Comparison
# ============================================================================

def compute_comprehensive_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global stats (all 6 metrics) for all methods/data_types.
    
    Returns a DataFrame with weighted averages, means, stds, and medians
    for each metric, grouped by method and data_type.
    
    NEW in Task 2: Includes all 6 metrics (rmse, mae, median, bias, std, rstd)
    plus correlation, computed with proper weighting by n_matches.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd', 'correlation']
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 
                  'reconstruction_missing', 'observation_same_dates']
    
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
                        row[f'{metric}_weighted'] = np.nan
                        row[f'{metric}_mean'] = np.nan
                        row[f'{metric}_std'] = np.nan
                        row[f'{metric}_median'] = np.nan
                else:
                    row[f'{metric}_weighted'] = np.nan
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_median'] = np.nan
            
            results.append(row)
    
    return pd.DataFrame(results)


def create_global_metric_maps(df: pd.DataFrame, lake_locations: pd.DataFrame, output_dir: str):
    """
    Create global scatter maps for each metric/data_type combination.
    
    NEW in Task 2: Creates 24 Cartopy maps (4 data_types × 6 metrics)
    showing per-lake performance geographically.
    
    Args:
        df: Combined in-situ validation stats DataFrame
        lake_locations: DataFrame with lake_id, lat, lon columns
        output_dir: Where to save the map images
    """
    if not HAS_CARTOPY:
        print("Warning: Cartopy not available, skipping global maps")
        return
    
    # Merge df with lake locations
    df_merged = df.merge(lake_locations[['lake_id', 'lat', 'lon']], 
                         left_on='lake_id_cci', right_on='lake_id', how='left')
    df_merged = df_merged[df_merged['lat'].notna()]
    
    if df_merged.empty:
        print("Warning: No lakes matched with location data for global maps")
        return
    
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']
    
    # Color schemes for different metric types
    cmap_diverging = 'RdBu_r'  # For bias, median (can be +/-)
    cmap_sequential = 'YlOrRd'  # For rmse, mae, std, rstd (always positive)
    
    maps_created = 0
    for data_type in data_types:
        for metric in metrics:
            try:
                df_dt = df_merged[df_merged['data_type'] == data_type]
                
                if df_dt.empty or metric not in df_dt.columns:
                    continue
                
                # Aggregate to per-lake level (mean across sites if multiple)
                lake_stats = df_dt.groupby(['lake_id_cci', 'lat', 'lon']).agg({
                    metric: 'mean', 
                    'n_matches': 'sum'
                }).reset_index()
                
                if len(lake_stats) == 0:
                    continue
                
                # Create the map
                fig, ax = plt.subplots(figsize=(14, 9),
                                       subplot_kw={'projection': ccrs.Robinson()})
                ax.set_global()
                ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
                ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='lightgray')
                ax.add_feature(cfeature.LAKES, facecolor='#b0d0ff', edgecolor='gray', linewidth=0.3)
                
                # Determine color scale
                values = lake_stats[metric].dropna()
                if len(values) == 0:
                    continue
                
                if metric in ['bias', 'median']:
                    # Diverging colormap centered at 0
                    vmax = max(abs(values.quantile(0.05)), abs(values.quantile(0.95)))
                    vmin = -vmax
                    cmap = cmap_diverging
                else:
                    # Sequential colormap
                    vmin = values.quantile(0.02)
                    vmax = values.quantile(0.98)
                    cmap = cmap_sequential
                
                # Size points by number of matches (log scale)
                sizes = np.log1p(lake_stats['n_matches']) * 15 + 30
                
                sc = ax.scatter(lake_stats['lon'], lake_stats['lat'],
                               c=lake_stats[metric], s=sizes,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               alpha=0.85, edgecolors='black', linewidth=0.5,
                               transform=ccrs.PlateCarree(), zorder=5)
                
                # Colorbar
                cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
                cbar.set_label(f'{metric.upper()} (°C)', fontsize=11)
                
                # Title
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


def create_comprehensive_multi_panel_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create large multi-panel figure: rows=data_types, cols=metrics, bars=methods per lake.
    
    NEW in Task 3: 4×6 panel figure for comprehensive method comparison
    across all data types and metrics.
    
    Purpose: Investigate if DINCAE's advantage correlates with poor in-situ 
    data quality. Visual inspection can reveal patterns.
    """
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    method_colors = {
        'dineof': '#3498db',          # Blue
        'dincae': '#e67e22',           # Orange
        'eof_filtered': '#27ae60',     # Green
        'eof_filtered_interp_full': '#9b59b6',  # Purple
        'interp_full': '#e74c3c'       # Red
    }
    
    # Get list of methods present in data
    available_methods = [m for m in method_colors.keys() if m in df['method'].unique()]
    
    if not available_methods:
        print("Warning: No recognized methods found for multi-panel comparison")
        return
    
    fig, axes = plt.subplots(len(data_types), len(metrics), figsize=(36, 24))
    
    for i, data_type in enumerate(data_types):
        df_dt = df[df['data_type'] == data_type]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            if df_dt.empty or metric not in df_dt.columns:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_visible(True)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Pivot to get methods as columns
            pivot = df_dt.pivot_table(
                index='lake_id_cci', 
                columns='method',
                values=metric, 
                aggfunc='mean'
            ).reset_index()
            
            if len(pivot) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            lakes = pivot['lake_id_cci'].values
            methods = [m for m in available_methods if m in pivot.columns]
            n_lakes, n_methods = len(lakes), len(methods)
            
            if n_methods == 0:
                ax.text(0.5, 0.5, 'No Methods', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            x = np.arange(n_lakes)
            width = 0.8 / n_methods
            
            for k, method in enumerate(methods):
                offset = (k - n_methods/2 + 0.5) * width
                values = pivot[method].values
                ax.bar(x + offset, values, width, label=method, 
                      color=method_colors[method], alpha=0.8, edgecolor='white', linewidth=0.3)
            
            # Formatting
            ax.set_xlabel('Lake ID' if i == len(data_types)-1 else '', fontsize=9)
            ax.set_ylabel(f'{metric.upper()} (°C)' if j == 0 else '', fontsize=9)
            
            data_type_label = data_type.replace('_', '\n').title()
            ax.set_title(f'{data_type_label}\n{metric.upper()}', fontsize=10, fontweight='bold')
            
            # X-tick labels (show subset to avoid crowding)
            tick_step = max(1, n_lakes // 12)
            ax.set_xticks(x[::tick_step])
            ax.set_xticklabels([f"{int(l)}" for l in lakes[::tick_step]], 
                              rotation=45, ha='right', fontsize=6)
            
            # Add horizontal line at 0 for bias/median
            if metric in ['bias', 'median']:
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            # Legend only in first panel
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='upper right')
            
            ax.tick_params(axis='y', labelsize=8)
    
    plt.suptitle('Comprehensive In-Situ Validation: All Methods × Data Types × Metrics\n'
                 '(Grouped bars per lake)', 
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'comprehensive_multi_panel_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def analyze_method_performance_vs_obs_quality(df: pd.DataFrame, output_dir: str):
    """
    KEY DIAGNOSTIC: Analyze if method advantage correlates with poor observation quality.
    
    This analysis answers: "Does DINCAE win on lakes where the satellite observation 
    vs in-situ comparison already has high error metrics?"
    
    If yes (positive correlation), it suggests DINCAE's advantage may be due to 
    noisy validation data rather than true gap-filling superiority.
    
    Creates scatter plots showing:
    - X-axis: Observation quality metrics (obs RMSE, obs MAE, obs bias, etc.)
    - Y-axis: Method RMSE difference (DINEOF - DINCAE, positive = DINCAE wins)
    
    Color coding: Blue = DINEOF wins, Orange = DINCAE wins
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    
    # Get observation quality stats per lake (using observation data_type)
    obs_df = df[df['data_type'] == 'observation'].copy()
    if obs_df.empty:
        print("Warning: No observation data for diagnostic analysis")
        return
    
    # Aggregate observation stats per lake
    obs_agg = obs_df.groupby('lake_id_cci').agg({
        **{m: 'mean' for m in metrics if m in obs_df.columns},
        'n_matches': 'sum'
    }).reset_index()
    
    # Rename to obs_ prefix
    rename_map = {m: f'obs_{m}' for m in metrics if m in obs_agg.columns}
    obs_agg = obs_agg.rename(columns=rename_map)
    
    # Also compute absolute values for signed metrics
    for m in ['bias', 'median']:
        if f'obs_{m}' in obs_agg.columns:
            obs_agg[f'obs_abs_{m}'] = obs_agg[f'obs_{m}'].abs()
    
    # Get reconstruction stats for method comparison
    recon_df = df[df['data_type'] == 'reconstruction'].copy()
    if recon_df.empty:
        print("Warning: No reconstruction data for diagnostic analysis")
        return
    
    # Method pairs to compare
    method_pairs = [('dineof', 'dincae')]
    
    # Also compare on reconstruction_missing (true gap-fill performance)
    data_types_to_analyze = ['reconstruction', 'reconstruction_missing']
    
    for recon_data_type in data_types_to_analyze:
        recon_df_dt = df[df['data_type'] == recon_data_type].copy()
        if recon_df_dt.empty:
            continue
        
        for method1, method2 in method_pairs:
            # Check if both methods exist
            if method1 not in recon_df_dt['method'].unique() or method2 not in recon_df_dt['method'].unique():
                continue
            
            # Get method stats per lake for ALL metrics
            m1_stats = recon_df_dt[recon_df_dt['method'] == method1].groupby('lake_id_cci')[metrics].mean().reset_index()
            m1_stats.columns = ['lake_id_cci'] + [f'{method1}_{m}' for m in metrics]
            
            m2_stats = recon_df_dt[recon_df_dt['method'] == method2].groupby('lake_id_cci')[metrics].mean().reset_index()
            m2_stats.columns = ['lake_id_cci'] + [f'{method2}_{m}' for m in metrics]
            
            # Merge with observation quality
            merged = obs_agg.merge(m1_stats, on='lake_id_cci', how='inner').merge(m2_stats, on='lake_id_cci', how='inner')
            
            if len(merged) < 3:
                print(f"Warning: Not enough lakes for {method1} vs {method2} on {recon_data_type}")
                continue
            
            # Create comprehensive diagnostic figure
            # Rows: different observation quality metrics
            # Columns: different recon metrics being compared
            obs_metrics_to_plot = [f'obs_{m}' for m in ['rmse', 'mae', 'std', 'rstd'] if f'obs_{m}' in merged.columns]
            obs_metrics_to_plot += [f'obs_abs_{m}' for m in ['bias', 'median'] if f'obs_abs_{m}' in merged.columns]
            
            recon_metrics = ['rmse', 'mae', 'std']  # Key metrics to compare
            
            n_obs_metrics = len(obs_metrics_to_plot)
            n_recon_metrics = len(recon_metrics)
            
            if n_obs_metrics == 0 or n_recon_metrics == 0:
                continue
            
            fig, axes = plt.subplots(n_obs_metrics, n_recon_metrics, figsize=(5*n_recon_metrics, 4*n_obs_metrics))
            if n_obs_metrics == 1:
                axes = axes.reshape(1, -1)
            if n_recon_metrics == 1:
                axes = axes.reshape(-1, 1)
            
            for i, obs_metric in enumerate(obs_metrics_to_plot):
                for j, recon_metric in enumerate(recon_metrics):
                    ax = axes[i, j]
                    
                    # Compute difference: positive = method2 (DINCAE) is better
                    diff_col = f'{method1}_{recon_metric}'
                    diff_col2 = f'{method2}_{recon_metric}'
                    
                    if diff_col not in merged.columns or diff_col2 not in merged.columns:
                        ax.set_visible(False)
                        continue
                    
                    merged['diff'] = merged[diff_col] - merged[diff_col2]
                    merged['winner'] = np.where(merged['diff'] > 0, method2.upper(), method1.upper())
                    
                    # Color by winner
                    colors = ['#e67e22' if w == method2.upper() else '#3498db' for w in merged['winner']]
                    
                    ax.scatter(merged[obs_metric], merged['diff'], 
                              c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
                    ax.axhline(0, color='black', linestyle='--', linewidth=1)
                    
                    # Compute correlation
                    corr = merged[obs_metric].corr(merged['diff'])
                    
                    # Add regression line
                    if abs(corr) > 0.1 and len(merged) > 5:
                        z = np.polyfit(merged[obs_metric], merged['diff'], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(merged[obs_metric].min(), merged[obs_metric].max(), 100)
                        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)
                    
                    obs_label = obs_metric.replace('obs_', '').replace('abs_', '|') + ('|' if 'abs_' in obs_metric else '')
                    ax.set_xlabel(f'Obs {obs_label.upper()} (°C)', fontsize=9)
                    ax.set_ylabel(f'{recon_metric.upper()} Diff\n({method1}-{method2}) °C', fontsize=9)
                    ax.set_title(f'r = {corr:.3f}', fontsize=10, 
                                color='green' if abs(corr) < 0.3 else 'red' if abs(corr) > 0.5 else 'orange')
                    ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3498db', edgecolor='black', label=f'{method1.upper()} wins'),
                Patch(facecolor='#e67e22', edgecolor='black', label=f'{method2.upper()} wins')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)
            
            dt_label = recon_data_type.replace('_', ' ').title()
            plt.suptitle(f'{method1.upper()} vs {method2.upper()}: Method Advantage vs Observation Quality\n'
                         f'Data Type: {dt_label} | Positive diff = {method2.upper()} better\n'
                         f'(If high correlation: {method2.upper()} wins mainly on lakes with poor obs quality)', 
                         fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'diagnostic_{method1}_vs_{method2}_{recon_data_type}_obs_quality.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved: diagnostic_{method1}_vs_{method2}_{recon_data_type}_obs_quality.png")
            
            # Also save correlation summary to CSV
            corr_summary = []
            for obs_metric in obs_metrics_to_plot:
                for recon_metric in recon_metrics:
                    diff_col = f'{method1}_{recon_metric}'
                    diff_col2 = f'{method2}_{recon_metric}'
                    if diff_col in merged.columns and diff_col2 in merged.columns:
                        diff = merged[diff_col] - merged[diff_col2]
                        corr = merged[obs_metric].corr(diff)
                        corr_summary.append({
                            'data_type': recon_data_type,
                            'method_comparison': f'{method1}_vs_{method2}',
                            'obs_quality_metric': obs_metric,
                            'recon_metric': recon_metric,
                            'correlation': corr,
                            'interpretation': 'DINCAE wins on poor obs quality' if corr > 0.3 else 
                                             'DINEOF wins on poor obs quality' if corr < -0.3 else
                                             'No clear pattern'
                        })
            
            if corr_summary:
                corr_df = pd.DataFrame(corr_summary)
                corr_path = os.path.join(output_dir, f'diagnostic_correlation_{method1}_vs_{method2}_{recon_data_type}.csv')
                corr_df.to_csv(corr_path, index=False)
                print(f"  Saved: diagnostic_correlation_{method1}_vs_{method2}_{recon_data_type}.csv")


def create_per_lake_stats_table(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Create comprehensive per-lake statistics table with all 6 metrics.
    
    NEW in Task 2: Generates per_lake_insitu_stats.csv with all metrics
    for each lake/site/method/data_type combination.
    """
    # Define all metrics to include
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd', 'correlation', 'n_matches']
    
    # Create a clean per-lake summary
    per_lake_data = []
    
    for (lake, site, method, data_type), group in df.groupby(['lake_id_cci', 'site_id', 'method', 'data_type']):
        row = {
            'lake_id_cci': lake,
            'site_id': site,
            'method': method,
            'data_type': data_type,
        }
        
        # Add all metrics
        for metric in metrics:
            if metric in group.columns:
                row[metric] = group[metric].values[0]
            else:
                row[metric] = np.nan
        
        per_lake_data.append(row)
    
    per_lake_df = pd.DataFrame(per_lake_data)
    
    # Sort for readability
    per_lake_df = per_lake_df.sort_values(['lake_id_cci', 'site_id', 'method', 'data_type'])
    
    # Save
    save_path = os.path.join(output_dir, 'per_lake_insitu_stats.csv')
    per_lake_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    
    return per_lake_df


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
    report_lines.append("PER-SITE WINNER ANALYSIS")
    report_lines.append("-" * 40)
    
    # Per-site: All reconstruction
    dineof_all = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction')][['lake_id_cci', 'site_id', 'rmse']]
    dincae_all = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction')][['lake_id_cci', 'site_id', 'rmse']]
    if not dineof_all.empty and not dincae_all.empty:
        merged = dineof_all.merge(dincae_all, on=['lake_id_cci', 'site_id'], suffixes=('_dineof', '_dincae'))
        if not merged.empty:
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            total = len(merged)
            report_lines.append(f"All Reconstruction (N={total} sites):")
            report_lines.append(f"  DINEOF better: {dineof_wins} sites ({100*dineof_wins/total:.1f}%)")
            report_lines.append(f"  DINCAE better: {dincae_wins} sites ({100*dincae_wins/total:.1f}%)")
            report_lines.append("")
    
    # Per-site: Gap-fill only
    dineof_gap_site = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'site_id', 'rmse']]
    dincae_gap_site = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'site_id', 'rmse']]
    if not dineof_gap_site.empty and not dincae_gap_site.empty:
        merged = dineof_gap_site.merge(dincae_gap_site, on=['lake_id_cci', 'site_id'], suffixes=('_dineof', '_dincae'))
        if not merged.empty:
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            total = len(merged)
            report_lines.append(f"Gap-Fill Only (N={total} sites):")
            report_lines.append(f"  DINEOF better: {dineof_wins} sites ({100*dineof_wins/total:.1f}%)")
            report_lines.append(f"  DINCAE better: {dincae_wins} sites ({100*dincae_wins/total:.1f}%)")
            report_lines.append("")
    
    report_lines.append("PER-LAKE WINNER ANALYSIS (sites averaged)")
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


def create_method_comparison_plots_all_metrics(df: pd.DataFrame, output_dir: str):
    """
    Create bar charts comparing methods across data types FOR ALL 6 METRICS.
    
    Generates one figure per metric, each with panels for all data types.
    This replaces the old single-metric version.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    methods_order = ['dineof', 'dincae', 'eof_filtered', 'interp_full', 'eof_filtered_interp_full']
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing', 'observation_same_dates']
    colors = {'dineof': '#2ecc71', 'dincae': '#3498db', 'eof_filtered': '#9b59b6', 
              'interp_full': '#e74c3c', 'eof_filtered_interp_full': '#f39c12'}
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"  Skipping {metric} - not in data")
            continue
        
        # Count how many data types have data
        n_data_types = sum(1 for dt in data_types if not df[df['data_type'] == dt].empty)
        if n_data_types == 0:
            continue
        
        fig, axes = plt.subplots(1, min(n_data_types, 5), figsize=(min(n_data_types, 5) * 5, 5))
        if n_data_types == 1:
            axes = [axes]
        
        ax_idx = 0
        for data_type in data_types:
            subset = df[df['data_type'] == data_type]
            
            if subset.empty:
                continue
            
            ax = axes[ax_idx]
            ax_idx += 1
            
            # Aggregate by method
            method_stats = subset.groupby('method').agg({
                metric: ['mean', 'std'],
                'n_matches': 'sum'
            }).reset_index()
            method_stats.columns = ['method', f'{metric}_mean', f'{metric}_std', 'n_total']
            
            # Sort by predefined order
            method_stats['sort_key'] = method_stats['method'].apply(
                lambda x: methods_order.index(x) if x in methods_order else 99
            )
            method_stats = method_stats.sort_values('sort_key')
            
            x = np.arange(len(method_stats))
            bars = ax.bar(x, method_stats[f'{metric}_mean'], 
                         yerr=method_stats[f'{metric}_std'],
                         color=[colors.get(m, 'gray') for m in method_stats['method']],
                         capsize=3, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(method_stats['method'], rotation=45, ha='right')
            ax.set_ylabel(f'{metric.upper()} (°C)')
            ax.set_title(data_type.replace('_', '\n').title())
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at 0 for bias/median
            if metric in ['bias', 'median']:
                ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add value labels
            for bar, val in zip(bars, method_stats[f'{metric}_mean']):
                if not np.isnan(val):
                    y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                           f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
        
        plt.suptitle(f'Method Comparison: {metric.upper()} by Data Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"method_comparison_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: method_comparison_{metric}.png")


def create_method_comparison_plot(df: pd.DataFrame, output_dir: str):
    """DEPRECATED - calls new all-metrics version. Kept for backward compatibility."""
    create_method_comparison_plots_all_metrics(df, output_dir)


def create_dineof_vs_dincae_scatter_all_metrics(df: pd.DataFrame, output_dir: str):
    """
    Create scatter plots comparing DINEOF vs DINCAE per lake FOR ALL 6 METRICS.
    
    Generates one figure per metric, each with panels for all data types.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    data_types = [
        ('observation', 'Observation'),
        ('reconstruction', 'All Reconstruction'),
        ('reconstruction_observed', 'Observed Pixels Only'),
        ('reconstruction_missing', 'Gap-Filled Pixels Only'),
        ('observation_same_dates', 'Obs (Same Dates)')
    ]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Count data types with data
        n_valid = sum(1 for dt, _ in data_types 
                      if not df[(df['data_type'] == dt) & (df['method'].isin(['dineof', 'dincae']))].empty)
        if n_valid == 0:
            continue
        
        n_cols = min(n_valid, 3)
        n_rows = (n_valid + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        ax_idx = 0
        for data_type, title in data_types:
            # Get DINEOF and DINCAE stats per lake
            dineof = df[(df['method'] == 'dineof') & (df['data_type'] == data_type)][['lake_id_cci', 'site_id', metric, 'n_matches']]
            dincae = df[(df['method'] == 'dincae') & (df['data_type'] == data_type)][['lake_id_cci', 'site_id', metric, 'n_matches']]
            
            if dineof.empty or dincae.empty:
                continue
            
            ax = axes[ax_idx]
            ax_idx += 1
            
            # Merge on lake_id and site_id
            merged = dineof.merge(dincae, on=['lake_id_cci', 'site_id'], suffixes=('_dineof', '_dincae'))
            
            if merged.empty:
                ax.set_title(f"{title}\n(No matching lakes)")
                continue
            
            # Scatter plot
            scatter = ax.scatter(merged[f'{metric}_dineof'], merged[f'{metric}_dincae'], 
                                c=merged['n_matches_dineof'], cmap='viridis',
                                s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Diagonal line (y=x)
            all_vals = pd.concat([merged[f'{metric}_dineof'], merged[f'{metric}_dincae']]).dropna()
            if len(all_vals) > 0:
                max_val = all_vals.max() * 1.1
                min_val = all_vals.min() * 0.9 if all_vals.min() >= 0 else all_vals.min() * 1.1
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal')
            
            ax.set_xlabel(f'DINEOF {metric.upper()} (°C)')
            ax.set_ylabel(f'DINCAE {metric.upper()} (°C)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Count wins
            dineof_better = (merged[f'{metric}_dineof'].abs() < merged[f'{metric}_dincae'].abs()).sum() if metric in ['bias', 'median'] else (merged[f'{metric}_dineof'] < merged[f'{metric}_dincae']).sum()
            dincae_better = (merged[f'{metric}_dincae'].abs() < merged[f'{metric}_dineof'].abs()).sum() if metric in ['bias', 'median'] else (merged[f'{metric}_dincae'] < merged[f'{metric}_dineof']).sum()
            ax.text(0.05, 0.95, f'DINEOF better: {dineof_better}\nDINCAE better: {dincae_better}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.colorbar(scatter, ax=ax, label='N matches', shrink=0.8)
        
        # Hide unused axes
        for i in range(ax_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'DINEOF vs DINCAE: Per-Site {metric.upper()} Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"dineof_vs_dincae_scatter_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: dineof_vs_dincae_scatter_{metric}.png")


def create_dineof_vs_dincae_scatter(df: pd.DataFrame, output_dir: str):
    """DEPRECATED - calls new all-metrics version. Kept for backward compatibility."""
    create_dineof_vs_dincae_scatter_all_metrics(df, output_dir)


def create_observed_vs_gapfill_comparison_all_metrics(df: pd.DataFrame, output_dir: str):
    """
    Create visualization comparing observed vs gap-fill performance FOR ALL METRICS.
    
    Generates one figure per metric showing boxplot comparison.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    methods = ['dineof', 'dincae']
    colors = {'observed': '#2ecc71', 'gap-fill': '#e74c3c'}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        for ax_idx, method in enumerate(methods):
            ax = axes[ax_idx]
            
            obs = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_observed')][metric]
            miss = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_missing')][metric]
            
            if obs.empty and miss.empty:
                ax.set_title(f"{method.upper()}\n(No data)")
                continue
            
            data_to_plot = []
            labels = []
            plot_colors = []
            
            if not obs.empty:
                data_to_plot.append(obs.dropna().values)
                labels.append(f'Observed\n(N={len(obs.dropna())})')
                plot_colors.append(colors['observed'])
            
            if not miss.empty:
                data_to_plot.append(miss.dropna().values)
                labels.append(f'Gap-fill\n(N={len(miss.dropna())})')
                plot_colors.append(colors['gap-fill'])
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], plot_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_ylabel(f'{metric.upper()} (°C)')
            ax.set_title(f'{method.upper()}')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add zero line for signed metrics
            if metric in ['bias', 'median']:
                ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add mean values
            for i, data in enumerate(data_to_plot):
                if len(data) > 0:
                    mean_val = np.mean(data)
                    ax.text(i+1, ax.get_ylim()[1]*0.95, f'μ={mean_val:.2f}', 
                           fontsize=9, ha='center', va='top')
        
        plt.suptitle(f'Observed vs Gap-Fill: {metric.upper()} Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"observed_vs_gapfill_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: observed_vs_gapfill_{metric}.png")


def create_observed_vs_gapfill_comparison(df: pd.DataFrame, output_dir: str):
    """DEPRECATED - calls new all-metrics version. Kept for backward compatibility."""
    create_observed_vs_gapfill_comparison_all_metrics(df, output_dir)


def create_metric_distribution_plots(df: pd.DataFrame, output_dir: str):
    """
    Create histogram distributions for ALL 6 metrics.
    
    Generates one figure per metric showing distributions across methods and data types.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    methods_colors = {
        'dineof': '#2ecc71', 'dincae': '#3498db', 
        'eof_filtered': '#9b59b6', 'interp_full': '#e74c3c'
    }
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        plot_configs = [
            ('dineof', 'reconstruction', 'DINEOF - All Recon'),
            ('dincae', 'reconstruction', 'DINCAE - All Recon'),
            ('dineof', 'reconstruction_missing', 'DINEOF - Gap-Fill'),
            ('dincae', 'reconstruction_missing', 'DINCAE - Gap-Fill'),
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for ax, (method, data_type, title) in zip(axes.flatten(), plot_configs):
            subset = df[(df['method'] == method) & (df['data_type'] == data_type)][metric]
            
            if subset.empty:
                ax.set_title(f"{title}\n(No data)")
                continue
            
            color = methods_colors.get(method, 'gray')
            ax.hist(subset.dropna(), bins=20, edgecolor='black', alpha=0.7, color=color)
            
            # Add statistics
            mean_val = subset.mean()
            median_val = subset.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}°C')
            ax.axvline(median_val, color='orange', linestyle='-', label=f'Med: {median_val:.2f}°C')
            
            # Add zero line for signed metrics
            if metric in ['bias', 'median']:
                ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
            
            ax.set_xlabel(f'{metric.upper()} (°C)')
            ax.set_ylabel('Number of Lakes')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'{metric.upper()} Distribution Across Lakes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"distribution_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: distribution_{metric}.png")


def create_rmse_distribution_plot(df: pd.DataFrame, output_dir: str):
    """DEPRECATED - calls new all-metrics version. Kept for backward compatibility."""
    create_metric_distribution_plots(df, output_dir)


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
    """Analyze the 'gap penalty' - how much worse is gap-filling vs observed (PER-SITE)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['dineof', 'dincae']
    
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        
        # Get observed and missing RMSE per site (lake_id + site_id)
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
        n_total = len(merged)
        
        ax.set_xlabel('Gap Penalty (Gap-fill RMSE - Observed RMSE) (°C)')
        ax.set_ylabel('Number of Sites')
        ax.set_title(f'{method.upper()} (N={n_total} sites)\nGap-fill worse: {n_positive}, Gap-fill better: {n_negative}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Gap-Filling Penalty Analysis - PER SITE\n(Positive = gap-fill performs worse than observed)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "gap_penalty_analysis_per_site.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_gap_penalty_analysis_per_lake(df: pd.DataFrame, output_dir: str):
    """Analyze the 'gap penalty' - how much worse is gap-filling vs observed (PER-LAKE, averaged across sites)."""
    
    # First aggregate to per-lake level
    lake_stats = df.groupby(['lake_id_cci', 'method', 'data_type']).agg({
        'rmse': 'mean',
        'n_matches': 'sum'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['dineof', 'dincae']
    
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        
        # Get observed and missing RMSE per lake
        obs = lake_stats[(lake_stats['method'] == method) & (lake_stats['data_type'] == 'reconstruction_observed')][['lake_id_cci', 'rmse']]
        miss = lake_stats[(lake_stats['method'] == method) & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
        
        if obs.empty or miss.empty:
            ax.set_title(f"{method.upper()}\n(Insufficient data)")
            continue
        
        merged = obs.merge(miss, on='lake_id_cci', suffixes=('_obs', '_miss'))
        merged['gap_penalty'] = merged['rmse_miss'] - merged['rmse_obs']
        
        # Histogram of gap penalty
        ax.hist(merged['gap_penalty'], bins=15, edgecolor='black', alpha=0.7,
               color='#3498db' if method == 'dincae' else '#2ecc71')
        
        mean_penalty = merged['gap_penalty'].mean()
        median_penalty = merged['gap_penalty'].median()
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='No penalty')
        ax.axvline(mean_penalty, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_penalty:.3f}°C')
        ax.axvline(median_penalty, color='red', linestyle=':', linewidth=2, label=f'Median: {median_penalty:.3f}°C')
        
        # Count positive/negative
        n_positive = (merged['gap_penalty'] > 0).sum()
        n_negative = (merged['gap_penalty'] < 0).sum()
        n_total = len(merged)
        
        ax.set_xlabel('Gap Penalty (Gap-fill RMSE - Observed RMSE) (°C)')
        ax.set_ylabel('Number of Lakes')
        ax.set_title(f'{method.upper()} (N={n_total} lakes)\nGap-fill worse: {n_positive}, Gap-fill better: {n_negative}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Gap-Filling Penalty Analysis - PER LAKE\n(Sites averaged, Positive = gap-fill performs worse than observed)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "gap_penalty_analysis_per_lake.png")
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
    """
    Create detailed per-lake statistics table with ALL 6 metrics.
    
    Updated to include: rmse, mae, median, bias, std, rstd for each method/data_type.
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd', 'correlation', 'n_matches']
    pivot_data = []
    
    lakes = df['lake_id_cci'].unique()
    
    for lake in lakes:
        lake_df = df[df['lake_id_cci'] == lake]
        sites = lake_df['site_id'].unique()
        
        for site in sites:
            site_df = lake_df[lake_df['site_id'] == site]
            
            row = {'lake_id_cci': lake, 'site_id': site}
            
            # Get stats for each method/data_type combo - ALL methods
            for method in ['dineof', 'dincae', 'eof_filtered', 'interp_full', 'eof_filtered_interp_full']:
                for data_type in ['observation', 'reconstruction', 'reconstruction_observed', 
                                  'reconstruction_missing', 'observation_same_dates']:
                    subset = site_df[(site_df['method'] == method) & (site_df['data_type'] == data_type)]
                    
                    # Create short prefix
                    method_short = method[:3] if len(method) > 5 else method
                    dt_short = data_type.replace('reconstruction', 'rec').replace('observation', 'obs').replace('_', '')
                    prefix = f"{method_short}_{dt_short}"
                    
                    if not subset.empty:
                        for metric in metrics:
                            if metric in subset.columns:
                                row[f'{prefix}_{metric}'] = subset[metric].values[0]
                    else:
                        for metric in metrics:
                            row[f'{prefix}_{metric}'] = np.nan
            
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
    
    # =========================================================================
    # TASK 2: Comprehensive Global Stats
    # =========================================================================
    print("\nComputing comprehensive global statistics (Task 2)...")
    global_stats = compute_comprehensive_global_stats(df)
    global_stats_path = os.path.join(args.output_dir, 'global_insitu_stats_comprehensive.csv')
    global_stats.to_csv(global_stats_path, index=False)
    print(f"Saved comprehensive global stats: {global_stats_path}")
    
    # Generate per-lake stats table with all metrics
    print("\nGenerating per-lake stats table with all metrics...")
    create_per_lake_stats_table(df, args.output_dir)
    
    # =========================================================================
    # Load lake locations for global maps
    # =========================================================================
    lake_locations = None
    
    # Try multiple possible locations for the lake locations CSV
    possible_lake_loc_paths = [
        args.lake_metadata,  # Explicitly provided (standard CSV format expected)
        os.path.join(os.path.dirname(__file__), 
                     '../src/processors/data/globolakes-static_lake_centre_fv1.csv'),
        os.path.join(os.path.dirname(__file__), 
                     'globolakes-static_lake_centre_fv1.csv'),
    ]
    
    for loc_path in possible_lake_loc_paths:
        if loc_path and os.path.exists(loc_path):
            try:
                # Check if it's a BADC-CSV format (globolakes file)
                with open(loc_path, 'r', encoding='iso-8859-1') as f:
                    first_line = f.readline()
                
                if 'BADC-CSV' in first_line or 'Conventions,G' in first_line:
                    # BADC-CSV format - use custom parser
                    print(f"Loading lake locations from BADC-CSV: {loc_path}")
                    import csv
                    lake_data = []
                    with open(loc_path, 'r', encoding='iso-8859-1') as f:
                        reader = csv.reader(f)
                        row_num = 0
                        for line in reader:
                            row_num += 1
                            # First 42 rows are comments/metadata
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
                    # Standard CSV format
                    lake_locations = pd.read_csv(loc_path)
                    # Handle different column naming conventions
                    if 'Lake ID' in lake_locations.columns:
                        lake_locations = lake_locations.rename(columns={
                            'Lake ID': 'lake_id',
                            'Latitude, Centre': 'lat', 
                            'Longitude, Centre': 'lon'
                        })
                    elif 'lake_id_cci' in lake_locations.columns:
                        lake_locations = lake_locations.rename(columns={'lake_id_cci': 'lake_id'})
                
                if lake_locations is not None and len(lake_locations) > 0:
                    print(f"Loaded lake locations from: {loc_path}")
                    print(f"  {len(lake_locations)} lakes with location data")
                    break
            except Exception as e:
                print(f"Warning: Could not load {loc_path}: {e}")
                lake_locations = None
    
    # =========================================================================
    # TASK 2: Global Metric Maps (requires Cartopy)
    # =========================================================================
    if lake_locations is not None:
        print("\nGenerating global metric maps (Task 2)...")
        create_global_metric_maps(df, lake_locations, args.output_dir)
    else:
        print("\nSkipping global maps: No lake location data available")
        print("  Provide --lake_metadata or ensure globolakes CSV is accessible")
    
    # =========================================================================
    # TASK 3: Multi-Panel Comparison and Diagnostic Analysis
    # =========================================================================
    print("\nGenerating comprehensive multi-panel comparison (Task 3)...")
    create_comprehensive_multi_panel_comparison(df, args.output_dir)
    
    print("\nAnalyzing method performance vs observation quality (Task 3)...")
    analyze_method_performance_vs_obs_quality(df, args.output_dir)
    
    # Generate text report
    print("\n" + "=" * 70)
    report_path = generate_summary_report(df, agg_df, args.output_dir)
    print("=" * 70)
    
    # =========================================================================
    # Generate visualizations - ALL 6 METRICS for ALL 4 DATA TYPES
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS FOR ALL METRICS")
    print("Metrics: RMSE, MAE, Median, Bias, STD, RSTD")
    print("Data Types: observation, reconstruction, recon_observed, recon_missing")
    print("=" * 70)
    
    print("\n[1/6] Method comparison bar charts (all metrics)...")
    create_method_comparison_plot(df, args.output_dir)
    
    print("\n[2/6] DINEOF vs DINCAE scatter plots (all metrics)...")
    create_dineof_vs_dincae_scatter(df, args.output_dir)
    
    print("\n[3/6] Observed vs Gap-fill boxplots...")
    create_observed_vs_gapfill_comparison(df, args.output_dir)
    
    print("\n[4/6] Metric distribution histograms (all metrics)...")
    create_rmse_distribution_plot(df, args.output_dir)
    create_correlation_comparison(df, args.output_dir)
    create_gap_penalty_analysis(df, args.output_dir)
    create_gap_penalty_analysis_per_lake(df, args.output_dir)
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