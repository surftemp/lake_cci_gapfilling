#!/usr/bin/env python3
"""
In-Situ Validation Analysis Script

Collects all in-situ validation results across all lakes and generates:
1. Aggregated statistics CSV with ALL 6 METRICS
2. Summary report (text + markdown)
3. Clear, readable visualizations comparing DINEOF vs DINCAE

ALL 6 CORE METRICS (from insitu_validation.py):
  - rmse: Root Mean Square Error
  - mae: Mean Absolute Error
  - median: Median of (satellite - insitu) differences
  - bias: Mean of (satellite - insitu) differences
  - std: Standard deviation of differences (residual STD)
  - rstd: Robust STD (1.4826 * MAD)

Usage:
    python analyze_insitu_validation.py --run_root /path/to/experiment --output_dir /path/to/output
"""

import argparse
import os
import sys
from glob import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# Import completion check utilities for fair comparison filtering
try:
    from completion_check import (
        get_fair_comparison_lakes,
        filter_dataframe_to_fair_comparison,
        save_exclusion_log,
        generate_unique_output_dir,
        print_fair_comparison_header,
        CompletionSummary
    )
    HAS_COMPLETION_CHECK = True
except ImportError:
    HAS_COMPLETION_CHECK = False
    print("Note: completion_check module not found - fair comparison filtering disabled")

# Try to import cartopy for global maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: Cartopy not available - global maps will be skipped")


# =============================================================================
# METRIC DEFINITIONS - ALL 6 CORE METRICS
# =============================================================================
# Column names from insitu_validation.py compute_stats():
#   rmse, mae, median, bias, std, rstd, correlation, n_matches

CORE_METRICS = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
ALL_METRICS = CORE_METRICS + ['correlation', 'n_matches']


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def find_validation_csvs(run_root: str, alpha: str = None) -> List[str]:
    """Find all in-situ validation CSV files in the run directory.
    
    FIX: Uses regex to match only integer site IDs (site1.csv, site2.csv)
    and excludes old float format files (site1.0.csv) to avoid duplicates.
    """
    import re
    
    if alpha:
        pattern = os.path.join(run_root, "post", "*", alpha, "insitu_cv_validation", "*_insitu_stats_site*.csv")
    else:
        pattern = os.path.join(run_root, "post", "*", "*", "insitu_cv_validation", "*_insitu_stats_site*.csv")
    
    all_files = glob(pattern)
    
    # Filter to only include files with integer site IDs (not float like site1.0)
    # Pattern: site followed by digits, then .csv (no decimal point)
    valid_files = []
    for f in all_files:
        basename = os.path.basename(f)
        # Match site<digits>.csv but NOT site<digits>.<digits>.csv
        if re.search(r'_site\d+\.csv$', basename):
            valid_files.append(f)
    
    # Also filter out yearly stats files
    valid_files = [f for f in valid_files if '_yearly_' not in f]
    
    n_excluded = len(all_files) - len(valid_files)
    if n_excluded > 0:
        print(f"  (Excluded {n_excluded} old format files like site1.0.csv)")
    
    return sorted(valid_files)


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
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Report which metrics are available
    available = [m for m in ALL_METRICS if m in combined.columns]
    missing = [m for m in ALL_METRICS if m not in combined.columns]
    print(f"  Available metrics: {', '.join(available)}")
    if missing:
        print(f"  Missing metrics: {', '.join(missing)}")
    
    return combined


# =============================================================================
# STATISTICS FUNCTIONS - NOW WITH ALL 6 METRICS
# =============================================================================

def compute_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate statistics across all lakes for each method/data_type combination.
    
    NOW INCLUDES ALL 6 CORE METRICS: rmse, mae, median, bias, std, rstd
    """
    if df.empty:
        return pd.DataFrame()
    
    grouped = df.groupby(['method', 'data_type'])
    
    agg_stats = []
    for (method, data_type), group in grouped:
        n_lakes = group['lake_id_cci'].nunique()
        n_points = group['n_matches'].sum() if 'n_matches' in group.columns else 0
        
        weights = group['n_matches'].values if 'n_matches' in group.columns else np.ones(len(group))
        total_weight = weights.sum()
        
        row = {
            'method': method,
            'data_type': data_type,
            'n_lakes': n_lakes,
            'n_total_points': n_points,
        }
        
        # Compute weighted and unweighted stats for ALL 6 CORE METRICS
        for metric in CORE_METRICS:
            if metric in group.columns:
                valid = group[~group[metric].isna()]
                if len(valid) > 0:
                    w = valid['n_matches'].values if 'n_matches' in valid.columns else np.ones(len(valid))
                    if w.sum() > 0:
                        row[f'{metric}_weighted'] = np.average(valid[metric].values, weights=w)
                    else:
                        row[f'{metric}_weighted'] = np.nan
                    row[f'{metric}_mean'] = valid[metric].mean()
                    row[f'{metric}_std'] = valid[metric].std()
                    row[f'{metric}_median'] = valid[metric].median()
                    row[f'{metric}_min'] = valid[metric].min()
                    row[f'{metric}_max'] = valid[metric].max()
                else:
                    for suffix in ['_weighted', '_mean', '_std', '_median', '_min', '_max']:
                        row[f'{metric}{suffix}'] = np.nan
            else:
                for suffix in ['_weighted', '_mean', '_std', '_median', '_min', '_max']:
                    row[f'{metric}{suffix}'] = np.nan
        
        # Correlation (not in CORE_METRICS but important)
        if 'correlation' in group.columns:
            valid = group[~group['correlation'].isna()]
            if len(valid) > 0:
                w = valid['n_matches'].values if 'n_matches' in valid.columns else np.ones(len(valid))
                if w.sum() > 0:
                    row['correlation_weighted'] = np.average(valid['correlation'].values, weights=w)
                else:
                    row['correlation_weighted'] = np.nan
                row['correlation_mean'] = valid['correlation'].mean()
                row['correlation_median'] = valid['correlation'].median()
            else:
                row['correlation_weighted'] = np.nan
                row['correlation_mean'] = np.nan
                row['correlation_median'] = np.nan
        
        agg_stats.append(row)
    
    return pd.DataFrame(agg_stats)


def compute_comprehensive_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global stats (all 6 metrics) for all methods/data_types.
    INCLUDES: rmse, mae, median, bias, std, rstd, correlation
    """
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
                'n_total_points': subset['n_matches'].sum() if 'n_matches' in subset.columns else 0,
            }
            
            weights = subset['n_matches'].values if 'n_matches' in subset.columns else np.ones(len(subset))
            total_weight = weights.sum()
            
            # ALL 6 METRICS + correlation
            for metric in CORE_METRICS + ['correlation']:
                if metric in subset.columns and total_weight > 0:
                    valid = subset[~subset[metric].isna()]
                    if len(valid) > 0:
                        w = valid['n_matches'].values if 'n_matches' in valid.columns else np.ones(len(valid))
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
    per_lake_data = []
    
    for (lake, site, method, data_type), group in df.groupby(['lake_id_cci', 'site_id', 'method', 'data_type']):
        row = {
            'lake_id_cci': lake,
            'site_id': site,
            'method': method,
            'data_type': data_type,
        }
        
        for metric in ALL_METRICS:
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
# GLOBAL MAPS (Cartopy)
# =============================================================================

def create_global_metric_maps(df: pd.DataFrame, lake_locations: pd.DataFrame, output_dir: str):
    """
    Create global scatter maps comparing DINEOF vs DINCAE at reconstruction_missing.
    
    For each metric, shows:
    1. Winner map: Categorical showing which method wins at each lake
    2. Difference maps (DINEOF - DINCAE): Blue = DINEOF better, Orange = DINCAE better
    3. Side-by-side comparison maps
    
    For error metrics (rmse, mae, std, rstd): negative difference = DINEOF better
    For bias/median: smaller |value| = better
    """
    if not HAS_CARTOPY:
        print("Warning: Cartopy not available, skipping global maps")
        return
    
    # Merge with lake locations
    df_merged = df.merge(lake_locations[['lake_id', 'lat', 'lon']], 
                         left_on='lake_id_cci', right_on='lake_id', how='left')
    df_merged = df_merged[df_merged['lat'].notna()]
    
    if df_merged.empty:
        print("Warning: No lakes matched with location data for global maps")
        return
    
    # Focus on reconstruction_missing only
    df_miss = df_merged[df_merged['data_type'] == 'reconstruction_missing']
    
    if df_miss.empty:
        print("Warning: No reconstruction_missing data for global maps")
        return
    
    maps_created = 0
    
    # ----- 1. CREATE WINNER MAP (categorical) -----
    try:
        # Aggregate RMSE per lake per method
        lake_method_stats = df_miss.groupby(['lake_id_cci', 'lat', 'lon', 'method']).agg({
            'rmse': 'mean',
            'n_matches': 'sum'
        }).reset_index()
        
        # Pivot to wide format
        pivot = lake_method_stats.pivot_table(
            index=['lake_id_cci', 'lat', 'lon'],
            columns='method',
            values='rmse'
        ).reset_index()
        
        if 'dineof' in pivot.columns and 'dincae' in pivot.columns:
            pivot = pivot.dropna(subset=['dineof', 'dincae'])
            pivot['delta_rmse'] = pivot['dineof'] - pivot['dincae']
            pivot['winner'] = pivot['delta_rmse'].apply(
                lambda x: 'DINEOF' if x < -0.02 else ('DINCAE' if x > 0.02 else 'TIE')
            )
            
            # Create winner map
            fig, ax = plt.subplots(figsize=(16, 10),
                                   subplot_kw={'projection': ccrs.Robinson()})
            ax.set_global()
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='lightgray')
            
            # Color mapping
            color_map = {'DINEOF': '#5DA5DA', 'DINCAE': '#FAA43A', 'TIE': '#808080'}
            colors = [color_map[w] for w in pivot['winner']]
            
            # Size by absolute difference (bigger = more decisive)
            sizes = np.abs(pivot['delta_rmse']) * 300 + 50
            sizes = np.clip(sizes, 50, 400)
            
            ax.scatter(pivot['lon'], pivot['lat'],
                      c=colors, s=sizes,
                      alpha=0.85, edgecolors='black', linewidth=0.5,
                      transform=ccrs.PlateCarree(), zorder=5)
            
            # Count winners
            n_dineof = (pivot['winner'] == 'DINEOF').sum()
            n_dincae = (pivot['winner'] == 'DINCAE').sum()
            n_tie = (pivot['winner'] == 'TIE').sum()
            
            # Legend
            legend_elements = [
                Patch(facecolor='#5DA5DA', edgecolor='black', label=f'DINEOF wins ({n_dineof})'),
                Patch(facecolor='#FAA43A', edgecolor='black', label=f'DINCAE wins ({n_dincae})'),
                Patch(facecolor='#808080', edgecolor='black', label=f'Tie ({n_tie})'),
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=11,
                     framealpha=0.9, title='Winner (by RMSE)')
            
            ax.set_title(f'DINEOF vs DINCAE: Per-Lake Winner (reconstruction_missing)\n'
                        f'N={len(pivot)} lakes | Marker size ∝ |ΔRMSE|',
                        fontsize=14, fontweight='bold')
            
            save_path = os.path.join(output_dir, 'global_map_winner_reconstruction_missing.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            maps_created += 1
            print(f"  Created: global_map_winner_reconstruction_missing.png")
    
    except Exception as e:
        print(f"  Error creating winner map: {e}")
        plt.close('all')
    
    # ----- 2. CREATE DIFFERENCE MAPS FOR EACH METRIC -----
    # For each metric: show (DINEOF - DINCAE) with diverging colormap
    # Negative (blue) = DINEOF better for error metrics
    
    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ['#5DA5DA', '#FFFFFF', '#FAA43A']  # Blue -> White -> Orange
    cmap_diverging = LinearSegmentedColormap.from_list('dineof_dincae', colors_cmap)
    
    for metric in CORE_METRICS:
        try:
            if metric not in df_miss.columns:
                continue
            
            # Aggregate per lake per method
            lake_method_stats = df_miss.groupby(['lake_id_cci', 'lat', 'lon', 'method']).agg({
                metric: 'mean',
                'n_matches': 'sum'
            }).reset_index()
            
            # Pivot to wide format
            pivot = lake_method_stats.pivot_table(
                index=['lake_id_cci', 'lat', 'lon'],
                columns='method',
                values=metric
            ).reset_index()
            
            if 'dineof' not in pivot.columns or 'dincae' not in pivot.columns:
                continue
            
            pivot = pivot.dropna(subset=['dineof', 'dincae'])
            
            if len(pivot) < 3:
                continue
            
            # Compute difference
            pivot['delta'] = pivot['dineof'] - pivot['dincae']
            
            fig, ax = plt.subplots(figsize=(16, 10),
                                   subplot_kw={'projection': ccrs.Robinson()})
            ax.set_global()
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='lightgray')
            
            values = pivot['delta'].values
            
            # Symmetric colorbar centered at 0
            vmax = np.percentile(np.abs(values), 95)
            vmin = -vmax
            
            sizes = 80  # Fixed size for difference maps
            
            sc = ax.scatter(pivot['lon'], pivot['lat'],
                           c=pivot['delta'], s=sizes,
                           cmap=cmap_diverging, vmin=vmin, vmax=vmax,
                           alpha=0.85, edgecolors='black', linewidth=0.5,
                           transform=ccrs.PlateCarree(), zorder=5)
            
            cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label(f'Δ{metric.upper()} (DINEOF − DINCAE) [°C]', fontsize=11)
            
            # Count which method is better
            if metric in ['rmse', 'mae', 'std', 'rstd']:
                # For error metrics: negative = DINEOF better
                n_dineof_better = (pivot['delta'] < -0.02).sum()
                n_dincae_better = (pivot['delta'] > 0.02).sum()
                interpretation = "Blue = DINEOF better (lower error)"
            else:
                # For bias/median: compare absolute values
                n_dineof_better = (np.abs(pivot['dineof']) < np.abs(pivot['dincae'])).sum()
                n_dincae_better = (np.abs(pivot['dincae']) < np.abs(pivot['dineof'])).sum()
                interpretation = "Shows signed difference"
            
            ax.set_title(f'DINEOF vs DINCAE: Δ{metric.upper()} (reconstruction_missing)\n'
                        f'{interpretation} | N={len(pivot)} lakes',
                        fontsize=14, fontweight='bold')
            
            # Add text box with counts
            textstr = f'DINEOF better: {n_dineof_better}\nDINCAE better: {n_dincae_better}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
            
            save_path = os.path.join(output_dir, f'global_map_delta_{metric}_reconstruction_missing.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            maps_created += 1
            print(f"  Created: global_map_delta_{metric}_reconstruction_missing.png")
            
        except Exception as e:
            print(f"  Error creating {metric} difference map: {e}")
            plt.close('all')
    
    # ----- 3. CREATE SIDE-BY-SIDE COMPARISON MAPS -----
    # Show DINEOF and DINCAE absolute values side by side for RMSE
    try:
        metric = 'rmse'
        
        lake_method_stats = df_miss.groupby(['lake_id_cci', 'lat', 'lon', 'method']).agg({
            metric: 'mean',
            'n_matches': 'sum'
        }).reset_index()
        
        pivot = lake_method_stats.pivot_table(
            index=['lake_id_cci', 'lat', 'lon'],
            columns='method',
            values=metric
        ).reset_index()
        
        if 'dineof' in pivot.columns and 'dincae' in pivot.columns:
            pivot = pivot.dropna(subset=['dineof', 'dincae'])
            
            if len(pivot) >= 3:
                # Shared color scale
                all_values = np.concatenate([pivot['dineof'].values, pivot['dincae'].values])
                vmin = np.percentile(all_values, 2)
                vmax = np.percentile(all_values, 98)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),
                                               subplot_kw={'projection': ccrs.Robinson()})
                
                for ax, method, title in [(ax1, 'dineof', 'DINEOF'), 
                                          (ax2, 'dincae', 'DINCAE')]:
                    ax.set_global()
                    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
                    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
                    
                    sc = ax.scatter(pivot['lon'], pivot['lat'],
                                   c=pivot[method], s=80,
                                   cmap='YlOrRd', vmin=vmin, vmax=vmax,
                                   alpha=0.85, edgecolors='black', linewidth=0.5,
                                   transform=ccrs.PlateCarree(), zorder=5)
                    
                    mean_val = pivot[method].mean()
                    ax.set_title(f'{title} RMSE\nMean: {mean_val:.3f}°C', fontsize=13, fontweight='bold')
                
                # Shared colorbar
                cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
                cbar = fig.colorbar(sc, cax=cbar_ax)
                cbar.set_label('RMSE (°C)', fontsize=11)
                
                fig.suptitle(f'Side-by-Side: DINEOF vs DINCAE RMSE (reconstruction_missing)\nN={len(pivot)} lakes',
                            fontsize=14, fontweight='bold', y=1.02)
                
                plt.tight_layout()
                save_path = os.path.join(output_dir, 'global_map_sidebyside_rmse_reconstruction_missing.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                maps_created += 1
                print(f"  Created: global_map_sidebyside_rmse_reconstruction_missing.png")
    
    except Exception as e:
        print(f"  Error creating side-by-side map: {e}")
        plt.close('all')
    
    print(f"Created {maps_created} global comparison maps")


# =============================================================================
# READABLE COMPARISON PLOTS
# =============================================================================

def create_method_comparison_by_lake(df: pd.DataFrame, output_dir: str,
                                      method1: str = 'dineof', 
                                      method2: str = 'dincae'):
    """
    Create clear, readable method comparison plots.
    Generates separate plots for each metric and data_type.
    NOW INCLUDES ALL 6 METRICS.
    """
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']
    
    colors = {method1: '#5DA5DA', method2: '#FAA43A'}  # Blue, Orange
    
    for data_type in data_types:
        df_dt = df[df['data_type'] == data_type]
        
        if df_dt.empty:
            continue
        
        available_methods = df_dt['method'].unique()
        if method1 not in available_methods or method2 not in available_methods:
            continue
        
        for metric in CORE_METRICS:
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
    """
    colors = {method1: '#5DA5DA', method2: '#FAA43A', 'observation': '#60BD68'}
    
    for metric in CORE_METRICS:
        df_recon = df[df['data_type'] == recon_data_type]
        df_obs = df[df['data_type'] == 'observation']
        
        if df_recon.empty or df_obs.empty:
            continue
        
        if metric not in df_recon.columns or metric not in df_obs.columns:
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
        ax1.set_title(f'{method1.upper()} vs {method2.upper()} - {recon_label}\n'
                     f'(Compare with observation quality below)', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        if metric in ['bias', 'median']:
            ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
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
        
        if metric in ['bias', 'median']:
            ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(int(lid)) for lid in merged['lake_id_cci']], 
                           rotation=45, ha='right', fontsize=9)
        
        # Compute correlation
        merged['method_diff'] = merged[method1] - merged[method2]
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
    NOW INCLUDES ALL 6 METRICS.
    """
    colors = {'observed': '#5DA5DA', 'missing': '#F17CB0'}
    
    for metric in CORE_METRICS:
        df_obs = df[(df['data_type'] == 'reconstruction_observed') & (df['method'] == method)]
        df_miss = df[(df['data_type'] == 'reconstruction_missing') & (df['method'] == method)]
        
        if df_obs.empty or df_miss.empty:
            continue
        
        if metric not in df_obs.columns or metric not in df_miss.columns:
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
        ax.set_title(f'{method.upper()}: Observed vs Missing Pixel Performance ({metric.upper()})\n'
                    f'(Gap-fill penalty = how much worse at truly missing pixels)',
                    fontsize=14, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(lid)) for lid in merged['lake_id_cci']],
                          rotation=45, ha='right', fontsize=9)
        
        if metric in ['bias', 'median']:
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        avg_obs = merged['observed'].mean()
        avg_miss = merged['missing'].mean()
        penalty = avg_miss - avg_obs
        
        ax.text(0.02, 0.98, f'Avg Observed: {avg_obs:.3f}°C\n'
                           f'Avg Missing: {avg_miss:.3f}°C\n'
                           f'Gap-fill penalty: {penalty:+.3f}°C',
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
    """Generate a text summary report with ALL 6 METRICS."""
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
    
    # Metrics available
    available_metrics = [m for m in CORE_METRICS if m in df.columns]
    report_lines.append(f"Metrics available: {', '.join(available_metrics)}")
    report_lines.append("")
    
    # DINEOF vs DINCAE comparison - ALL METRICS
    report_lines.append("KEY FINDINGS - DINEOF vs DINCAE (reconstruction_missing)")
    report_lines.append("-" * 40)
    
    dineof = agg_df[(agg_df['method'] == 'dineof') & (agg_df['data_type'] == 'reconstruction_missing')]
    dincae = agg_df[(agg_df['method'] == 'dincae') & (agg_df['data_type'] == 'reconstruction_missing')]
    
    if not dineof.empty and not dincae.empty:
        for metric in CORE_METRICS:
            col = f'{metric}_weighted'
            if col in dineof.columns and col in dincae.columns:
                din_val = dineof[col].values[0]
                dic_val = dincae[col].values[0]
                if np.isnan(din_val) or np.isnan(dic_val):
                    continue
                
                if metric in ['rmse', 'mae', 'std', 'rstd']:
                    better = "DINEOF" if din_val < dic_val else "DINCAE"
                    diff = abs(din_val - dic_val)
                else:
                    better = "DINEOF" if abs(din_val) < abs(dic_val) else "DINCAE"
                    diff = abs(abs(din_val) - abs(dic_val))
                
                report_lines.append(f"  {metric.upper():12s}: DINEOF={din_val:+.3f}, DINCAE={dic_val:+.3f} → {better} by {diff:.3f}°C")
    
    report_lines.append("")
    
    # Per-lake winner counts
    report_lines.append("PER-LAKE WINNER ANALYSIS (reconstruction_missing)")
    report_lines.append("-" * 40)
    
    lake_stats = df.groupby(['lake_id_cci', 'method', 'data_type']).agg({'rmse': 'mean'}).reset_index()
    
    dineof_dt = lake_stats[(lake_stats['method'] == 'dineof') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    dincae_dt = lake_stats[(lake_stats['method'] == 'dincae') & (lake_stats['data_type'] == 'reconstruction_missing')][['lake_id_cci', 'rmse']]
    
    if not dineof_dt.empty and not dincae_dt.empty:
        merged = dineof_dt.merge(dincae_dt, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        if not merged.empty:
            dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
            dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
            ties = len(merged) - dineof_wins - dincae_wins
            total = len(merged)
            
            report_lines.append(f"  DINEOF better: {dineof_wins} lakes ({100*dineof_wins/total:.1f}%)")
            report_lines.append(f"  DINCAE better: {dincae_wins} lakes ({100*dincae_wins/total:.1f}%)")
            if ties > 0:
                report_lines.append(f"  Ties: {ties} lakes")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    report_path = os.path.join(output_dir, "insitu_validation_summary_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_path


def create_per_lake_table(df: pd.DataFrame, output_dir: str):
    """Create a simple per-lake comparison table CSV with ALL 6 METRICS."""
    agg_cols = {m: 'mean' for m in CORE_METRICS + ['correlation'] if m in df.columns}
    if 'n_matches' in df.columns:
        agg_cols['n_matches'] = 'sum'
    
    lake_stats = df.groupby(['lake_id_cci', 'method', 'data_type']).agg(agg_cols).reset_index()
    
    save_path = os.path.join(output_dir, 'per_lake_comparison_table.csv')
    lake_stats.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze in-situ validation results across all lakes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FAIR COMPARISON MODE (default):
  Only includes lakes where BOTH DINEOF and DINCAE completed processing.
  This ensures all comparative statistics are computed on the same sample.
  
  Use --no-fair-comparison to include all available data (not recommended
  for method comparisons).

OUTPUT DIRECTORY:
  Default: {run_root}/insitu_validation_analysis/ (overwrites previous)
  
  Use --archive to also save a timestamped copy:
    {run_root}/archive_insitu_validation_analysis/insitu_validation_analysis_{ts}_{hash}/

Examples:
    # Standard analysis with fair comparison (recommended)
    python analyze_insitu_validation.py --run_root /path/to/experiment
    
    # Also preserve a timestamped archive copy
    python analyze_insitu_validation.py --run_root /path/to/experiment --archive
    
    # Include all data (not recommended for comparisons)
    python analyze_insitu_validation.py --run_root /path/to/experiment --no-fair-comparison
        """
    )
    parser.add_argument("--run_root", required=True, help="Path to run root directory")
    parser.add_argument("--alpha", default=None, help="Specific alpha slug (e.g., 'a1000')")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--lake_metadata", default=None, help="Optional CSV with lake metadata (lat, lon)")
    
    # Fair comparison arguments
    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Disable fair comparison filtering (include lakes with incomplete methods)")
    parser.add_argument("--archive", action="store_true",
                        help="Also save a timestamped copy to archive_insitu_validation_analysis/")
    
    args = parser.parse_args()
    
    # Determine output directory - always use fixed path for working copy
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "insitu_validation_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine archive directory if --archive is set
    archive_dir = None
    if args.archive and HAS_COMPLETION_CHECK:
        timestamp_suffix = generate_unique_output_dir("", prefix="insitu_validation_analysis")
        timestamp_suffix = os.path.basename(timestamp_suffix)  # Just the folder name
        archive_dir = os.path.join(
            args.run_root, 
            "archive_insitu_validation_analysis",
            timestamp_suffix
        )
        os.makedirs(archive_dir, exist_ok=True)
    
    print("=" * 70)
    print("IN-SITU VALIDATION ANALYSIS (WITH ALL 6 METRICS)")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Alpha: {args.alpha or 'all'}")
    print(f"Output: {args.output_dir}")
    if archive_dir:
        print(f"Archive: {archive_dir}")
    print(f"Core metrics: {', '.join(CORE_METRICS)}")
    print(f"Fair comparison mode: {'DISABLED' if args.no_fair_comparison else 'ENABLED'}")
    print("=" * 70)
    
    # =========================================================================
    # FAIR COMPARISON FILTERING
    # =========================================================================
    fair_lake_ids = None
    completion_summary = None
    
    if HAS_COMPLETION_CHECK and not args.no_fair_comparison:
        print("\n" + "=" * 70)
        print("STEP 0: Fair Comparison Pre-filtering")
        print("=" * 70)
        
        fair_lake_ids, completion_summary = get_fair_comparison_lakes(
            args.run_root, args.alpha, verbose=True
        )
        
        if not fair_lake_ids:
            print("ERROR: No lakes found with both DINEOF and DINCAE complete!")
            print("Check your experiment directory or use --no-fair-comparison")
            sys.exit(1)
        
        # Save exclusion log
        log_path = save_exclusion_log(completion_summary, args.output_dir)
        print(f"Exclusion log saved: {log_path}")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
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
    
    # =========================================================================
    # APPLY FAIR COMPARISON FILTER
    # =========================================================================
    
    if fair_lake_ids is not None:
        print("\n" + "-" * 70)
        print("Applying fair comparison filter to loaded data...")
        df = filter_dataframe_to_fair_comparison(
            df, fair_lake_ids, lake_id_column='lake_id_cci', verbose=True
        )
        print("-" * 70)
        
        if df.empty:
            print("ERROR: No data remaining after fair comparison filter!")
            sys.exit(1)
    
    # Save combined raw data (after filtering)
    raw_path = os.path.join(args.output_dir, "all_insitu_stats_combined.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved combined data: {raw_path}")
    
    # Add fair comparison metadata to the saved file
    if completion_summary is not None:
        metadata_path = os.path.join(args.output_dir, "analysis_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Analysis Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Run Root: {args.run_root}\n")
            f.write(f"Alpha: {args.alpha or 'all'}\n")
            f.write(f"Fair Comparison Mode: ENABLED\n")
            f.write(f"\n{completion_summary}\n")
            f.write(f"\nIncluded Lakes ({len(fair_lake_ids)}):\n")
            f.write(", ".join(map(str, fair_lake_ids)) + "\n")
            f.write(f"\nExcluded Lakes ({len(completion_summary.excluded_lake_ids)}):\n")
            for lake_id in completion_summary.excluded_lake_ids:
                reason = completion_summary.exclusion_reasons.get(lake_id, "Unknown")
                f.write(f"  {lake_id}: {reason}\n")
        print(f"Saved analysis metadata: {metadata_path}")
    
    # Compute aggregate statistics (ALL 6 METRICS)
    print("\nComputing aggregate statistics (all 6 metrics)...")
    agg_df = compute_aggregate_stats(df)
    agg_df.to_csv(os.path.join(args.output_dir, "aggregate_statistics.csv"), index=False)
    
    # Comprehensive global stats
    print("\nComputing comprehensive global statistics...")
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
    
    # Global maps
    if lake_locations is not None:
        print("\nGenerating global metric maps...")
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
    print("GENERATING READABLE COMPARISON PLOTS (ALL 6 METRICS)")
    print("=" * 70)
    
    # 1. Method comparison plots (DINEOF vs DINCAE) - ALL 6 METRICS
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
    
    # Copy to archive if --archive was specified
    if archive_dir:
        import shutil
        print(f"\nArchiving to: {archive_dir}")
        for item in os.listdir(args.output_dir):
            src = os.path.join(args.output_dir, item)
            dst = os.path.join(archive_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Archive complete: {archive_dir}")


if __name__ == "__main__":
    main()