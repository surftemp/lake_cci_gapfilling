"""
Improved plotting functions for analyze_insitu_validation.py

These functions generate clear, readable plots following the style of 
rmse_comparison_by_lake.png - simple side-by-side bars that immediately 
tell the story.

Replace the corresponding functions in analyze_insitu_validation.py with these.

To use: 
1. Copy this file to scripts/improved_plotting_functions.py
2. In analyze_insitu_validation.py, add at top:
   from improved_plotting_functions import create_all_readable_plots
3. Replace the visualization section in main() with:
   create_all_readable_plots(df, args.output_dir)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional


def create_method_comparison_by_lake(df: pd.DataFrame, output_dir: str,
                                      method1: str = 'dineof', 
                                      method2: str = 'dincae'):
    """
    Create clear, readable DINEOF vs DINCAE comparison plots.
    
    Generates separate plots for:
    - Each metric (rmse, mae, median, bias, std, rstd)
    - Each data_type (observation, reconstruction, reconstruction_observed, reconstruction_missing)
    
    Style: Simple side-by-side bars like rmse_comparison_by_lake.png
    """
    metrics = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    data_types = ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']
    
    # Colors matching the original plot style
    colors = {method1: '#5DA5DA', method2: '#FAA43A'}  # Blue, Orange
    
    for data_type in data_types:
        df_dt = df[df['data_type'] == data_type]
        
        if df_dt.empty:
            print(f"  Skipping {data_type}: no data")
            continue
        
        # Check if both methods exist
        available_methods = df_dt['method'].unique()
        if method1 not in available_methods or method2 not in available_methods:
            print(f"  Skipping {data_type}: missing {method1} or {method2}")
            continue
        
        for metric in metrics:
            if metric not in df_dt.columns:
                continue
            
            # Pivot to get methods as columns, aggregate by lake
            pivot = df_dt.pivot_table(
                index='lake_id_cci',
                columns='method',
                values=metric,
                aggfunc='mean'
            ).reset_index()
            
            if method1 not in pivot.columns or method2 not in pivot.columns:
                continue
            
            # Sort by lake_id for consistent ordering
            pivot = pivot.sort_values('lake_id_cci')
            
            # Drop rows with NaN in either method
            pivot = pivot.dropna(subset=[method1, method2])
            
            if len(pivot) == 0:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 0.4), 6))
            
            x = np.arange(len(pivot))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pivot[method1], width, 
                          label=method1.upper(), color=colors[method1], 
                          edgecolor='white', linewidth=0.5)
            bars2 = ax.bar(x + width/2, pivot[method2], width,
                          label=method2.upper(), color=colors[method2],
                          edgecolor='white', linewidth=0.5)
            
            # Formatting
            ax.set_xlabel('Lake ID', fontsize=12)
            ax.set_ylabel(f'{metric.upper()} (°C)', fontsize=12)
            
            data_type_label = data_type.replace('_', ' ').title()
            ax.set_title(f'In-Situ Validation: {method1.upper()} vs {method2.upper()} {metric.upper()} by Lake\n'
                        f'Data Type: {data_type_label}', fontsize=14, fontweight='bold')
            
            # X-axis labels
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(lid)) for lid in pivot['lake_id_cci']], 
                              rotation=45, ha='right', fontsize=9)
            
            # Add horizontal line at 0 for bias/median
            if metric in ['bias', 'median']:
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Add count annotation
            n_lakes = len(pivot)
            m1_better = (pivot[method1] < pivot[method2]).sum() if metric != 'correlation' else (pivot[method1] > pivot[method2]).sum()
            m2_better = n_lakes - m1_better
            
            # For metrics where lower is better (rmse, mae, std, rstd)
            if metric in ['rmse', 'mae', 'std', 'rstd']:
                ax.text(0.02, 0.98, f'{method1.upper()} better: {m1_better}/{n_lakes} lakes\n'
                                    f'{method2.upper()} better: {m2_better}/{n_lakes} lakes',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save
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
    
    Creates a 2-panel figure:
    - Top: DINEOF vs DINCAE RMSE for reconstruction (or reconstruction_missing)
    - Bottom: Observation RMSE per lake (satellite vs in-situ quality)
    
    This allows visual inspection: Do lakes where DINCAE wins have worse 
    observation quality (higher obs RMSE)?
    """
    metrics = ['rmse', 'mae', 'std']  # Key metrics for diagnosis
    
    colors = {method1: '#5DA5DA', method2: '#FAA43A', 'observation': '#60BD68'}
    
    for metric in metrics:
        # Get reconstruction data for method comparison
        df_recon = df[df['data_type'] == recon_data_type]
        df_obs = df[df['data_type'] == 'observation']
        
        if df_recon.empty or df_obs.empty:
            continue
        
        # Pivot reconstruction data
        pivot_recon = df_recon.pivot_table(
            index='lake_id_cci',
            columns='method',
            values=metric,
            aggfunc='mean'
        ).reset_index()
        
        # Pivot observation data (use first method, they should all be same for obs)
        pivot_obs = df_obs.groupby('lake_id_cci')[metric].mean().reset_index()
        pivot_obs.columns = ['lake_id_cci', f'obs_{metric}']
        
        # Merge
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
        
        # Mark which method wins for each lake
        for i, (_, row) in enumerate(merged.iterrows()):
            if row[method1] < row[method2]:
                ax1.plot(i, row[method1], 'v', color='blue', markersize=6, alpha=0.7)
            else:
                ax1.plot(i, row[method2], 'v', color='orange', markersize=6, alpha=0.7)
        
        # Bottom panel: Observation quality
        ax2.bar(x, merged[f'obs_{metric}'], width*2, label='Observation vs In-Situ',
               color=colors['observation'], edgecolor='white', linewidth=0.5)
        
        ax2.set_xlabel('Lake ID', fontsize=12)
        ax2.set_ylabel(f'Observation {metric.upper()} (°C)', fontsize=12)
        ax2.set_title(f'Satellite Observation vs In-Situ Quality\n'
                     f'(Higher = worse match between satellite and buoy)', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        # X-axis labels
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(int(lid)) for lid in merged['lake_id_cci']], 
                           rotation=45, ha='right', fontsize=9)
        
        # Add correlation annotation
        # Compute: does method2 advantage correlate with obs quality?
        merged['method_diff'] = merged[method1] - merged[method2]  # Positive = method2 better
        corr = merged['method_diff'].corr(merged[f'obs_{metric}'])
        
        # Add text annotation
        interpretation = ""
        if corr > 0.3:
            interpretation = f"→ {method2.upper()} tends to win on lakes with WORSE obs quality"
        elif corr < -0.3:
            interpretation = f"→ {method1.upper()} tends to win on lakes with WORSE obs quality"
        else:
            interpretation = "→ No strong correlation between method advantage and obs quality"
        
        fig.text(0.5, 0.02, f'Correlation: r = {corr:.3f}\n{interpretation}',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        # Save
        save_path = f'{output_dir}/diagnostic_{method1}_vs_{method2}_{metric}_{recon_data_type}_with_obs_quality.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: diagnostic_{method1}_vs_{method2}_{metric}_{recon_data_type}_with_obs_quality.png")


def create_reconstruction_observed_vs_missing_plot(df: pd.DataFrame, output_dir: str,
                                                    method: str = 'dineof'):
    """
    Compare reconstruction_observed vs reconstruction_missing for a single method.
    
    This shows the "gap-fill penalty" - how much worse is the method at 
    truly missing pixels vs pixels where it had original observations?
    """
    metrics = ['rmse', 'mae', 'std']
    
    colors = {'observed': '#5DA5DA', 'missing': '#F17CB0'}  # Blue, Pink
    
    for metric in metrics:
        df_obs = df[(df['data_type'] == 'reconstruction_observed') & (df['method'] == method)]
        df_miss = df[(df['data_type'] == 'reconstruction_missing') & (df['method'] == method)]
        
        if df_obs.empty or df_miss.empty:
            continue
        
        # Aggregate by lake
        obs_stats = df_obs.groupby('lake_id_cci')[metric].mean().reset_index()
        obs_stats.columns = ['lake_id_cci', 'observed']
        
        miss_stats = df_miss.groupby('lake_id_cci')[metric].mean().reset_index()
        miss_stats.columns = ['lake_id_cci', 'missing']
        
        merged = obs_stats.merge(miss_stats, on='lake_id_cci', how='inner')
        merged = merged.dropna().sort_values('lake_id_cci')
        
        if len(merged) < 3:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(merged) * 0.4), 6))
        
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
        ax.set_title(f'{method.upper()}: Observed vs Missing Pixel Performance\n'
                    f'(Gap-fill penalty = how much worse at truly missing pixels)',
                    fontsize=14, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(lid)) for lid in merged['lake_id_cci']],
                          rotation=45, ha='right', fontsize=9)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Compute average gap-fill penalty
        avg_obs = merged['observed'].mean()
        avg_miss = merged['missing'].mean()
        penalty = avg_miss - avg_obs
        
        ax.text(0.02, 0.98, f'Avg Observed: {avg_obs:.3f}°C\n'
                           f'Avg Missing: {avg_miss:.3f}°C\n'
                           f'Gap-fill penalty: +{penalty:.3f}°C',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        save_path = f'{output_dir}/{method}_observed_vs_missing_{metric}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {method}_observed_vs_missing_{metric}.png")


def create_all_readable_plots(df: pd.DataFrame, output_dir: str):
    """
    Main function to generate all readable, informative plots.
    
    Call this from main() in analyze_insitu_validation.py instead of
    the existing plotting functions.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("GENERATING READABLE COMPARISON PLOTS")
    print("=" * 70)
    
    # 1. Method comparison plots (DINEOF vs DINCAE) for all metrics and data types
    print("\n[1/4] DINEOF vs DINCAE comparison plots (all metrics × data types)...")
    create_method_comparison_by_lake(df, output_dir, 'dineof', 'dincae')
    
    # 2. Diagnostic plots with observation quality
    print("\n[2/4] Diagnostic plots (method comparison + observation quality)...")
    create_diagnostic_comparison_plot(df, output_dir, 'dineof', 'dincae', 'reconstruction')
    create_diagnostic_comparison_plot(df, output_dir, 'dineof', 'dincae', 'reconstruction_missing')
    
    # 3. Observed vs Missing comparison (gap-fill penalty)
    print("\n[3/4] Gap-fill penalty plots (observed vs missing pixels)...")
    create_reconstruction_observed_vs_missing_plot(df, output_dir, 'dineof')
    create_reconstruction_observed_vs_missing_plot(df, output_dir, 'dincae')
    
    # 4. If eof_filtered exists, compare it too
    if 'eof_filtered' in df['method'].unique():
        print("\n[4/4] Additional method comparisons (eof_filtered)...")
        create_method_comparison_by_lake(df, output_dir, 'dineof', 'eof_filtered')
        create_reconstruction_observed_vs_missing_plot(df, output_dir, 'eof_filtered')
    else:
        print("\n[4/4] Skipping eof_filtered comparisons (not in data)")
    
    print("\n" + "=" * 70)
    print("READABLE PLOTS COMPLETE")
    print("=" * 70)


# Summary of what this generates:
#
# For DINEOF vs DINCAE:
#   - dineof_vs_dincae_rmse_observation.png
#   - dineof_vs_dincae_mae_observation.png
#   - dineof_vs_dincae_median_observation.png
#   - dineof_vs_dincae_bias_observation.png
#   - dineof_vs_dincae_std_observation.png
#   - dineof_vs_dincae_rstd_observation.png
#   - dineof_vs_dincae_rmse_reconstruction.png
#   - ... (6 metrics × 4 data_types = 24 plots)
#
# Diagnostic plots:
#   - diagnostic_dineof_vs_dincae_rmse_reconstruction_with_obs_quality.png
#   - diagnostic_dineof_vs_dincae_rmse_reconstruction_missing_with_obs_quality.png
#   - ... (3 metrics × 2 data_types = 6 plots)
#
# Gap-fill penalty:
#   - dineof_observed_vs_missing_rmse.png
#   - dineof_observed_vs_missing_mae.png
#   - dineof_observed_vs_missing_std.png
#   - dincae_observed_vs_missing_rmse.png
#   - ... (3 metrics × 2 methods = 6 plots)
