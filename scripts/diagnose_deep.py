#!/usr/bin/env python3
"""
Deep Diagnostic: Why does DINCAE win on some lakes despite being rougher?

Key questions:
1. In DINCAE-winning lakes, is DINCAE also rougher?
2. What is the reconstruction BIAS of each method? (not just obs bias)
3. Is DINCAE systematically warmer/colder than DINEOF?
4. Cross-tabulate ALL characteristics for ALL 29 lakes

Uses ALL lakes - no subsampling.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_data(analysis_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    return pd.read_csv(csv_path)


def build_comprehensive_lake_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comprehensive table with ALL metrics for ALL lakes.
    One row per lake with all relevant comparisons.
    """
    results = []
    
    for lake_id in df['lake_id_cci'].unique():
        lake_df = df[df['lake_id_cci'] == lake_id]
        
        row = {'lake_id': lake_id}
        
        # For each data type, get metrics for both methods
        for data_type in ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']:
            dt_df = lake_df[lake_df['data_type'] == data_type]
            
            for method in ['dineof', 'dincae']:
                m_df = dt_df[dt_df['method'] == method]
                
                if m_df.empty:
                    for metric in ['rmse', 'mae', 'bias', 'std', 'correlation', 'n_matches']:
                        row[f'{data_type}_{method}_{metric}'] = np.nan
                else:
                    for metric in ['rmse', 'mae', 'bias', 'std', 'correlation', 'n_matches']:
                        if metric in m_df.columns:
                            row[f'{data_type}_{method}_{metric}'] = m_df[metric].mean()
                        else:
                            row[f'{data_type}_{method}_{metric}'] = np.nan
            
            # Observation doesn't have method distinction - it's satellite vs buoy
            if data_type == 'observation' and not dt_df.empty:
                for metric in ['rmse', 'mae', 'bias', 'std', 'correlation', 'n_matches']:
                    if metric in dt_df.columns:
                        row[f'obs_{metric}'] = dt_df[metric].mean()
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Compute derived metrics
    for data_type in ['reconstruction', 'reconstruction_missing']:
        # RMSE difference: positive = DINCAE better (lower RMSE)
        dineof_col = f'{data_type}_dineof_rmse'
        dincae_col = f'{data_type}_dincae_rmse'
        if dineof_col in result_df.columns and dincae_col in result_df.columns:
            result_df[f'{data_type}_rmse_diff'] = result_df[dineof_col] - result_df[dincae_col]
            result_df[f'{data_type}_winner'] = np.where(
                result_df[f'{data_type}_rmse_diff'] > 0.02, 'DINCAE',
                np.where(result_df[f'{data_type}_rmse_diff'] < -0.02, 'DINEOF', 'TIE')
            )
        
        # STD difference: positive = DINCAE rougher (higher STD)
        dineof_std = f'{data_type}_dineof_std'
        dincae_std = f'{data_type}_dincae_std'
        if dineof_std in result_df.columns and dincae_std in result_df.columns:
            result_df[f'{data_type}_std_diff'] = result_df[dincae_std] - result_df[dineof_std]
            result_df[f'{data_type}_dincae_rougher'] = result_df[f'{data_type}_std_diff'] > 0
        
        # BIAS difference: How do the biases compare?
        dineof_bias = f'{data_type}_dineof_bias'
        dincae_bias = f'{data_type}_dincae_bias'
        if dineof_bias in result_df.columns and dincae_bias in result_df.columns:
            result_df[f'{data_type}_bias_diff'] = result_df[dincae_bias] - result_df[dineof_bias]
            # Is DINCAE warmer than DINEOF? (positive bias diff)
            result_df[f'{data_type}_dincae_warmer'] = result_df[f'{data_type}_bias_diff'] > 0
    
    return result_df


def analyze_winner_vs_roughness(lake_table: pd.DataFrame, output_dir: str):
    """
    Key question: In lakes where DINCAE wins, is DINCAE also rougher?
    """
    print("\n" + "="*70)
    print("ANALYSIS: Winner vs Roughness (ALL 29 LAKES)")
    print("="*70)
    
    for data_type in ['reconstruction', 'reconstruction_missing']:
        winner_col = f'{data_type}_winner'
        rougher_col = f'{data_type}_dincae_rougher'
        
        if winner_col not in lake_table.columns or rougher_col not in lake_table.columns:
            continue
        
        valid = lake_table[[winner_col, rougher_col]].dropna()
        
        print(f"\n{data_type.upper()}:")
        print("-" * 50)
        
        # Cross-tabulation
        crosstab = pd.crosstab(valid[winner_col], valid[rougher_col], margins=True)
        crosstab.columns = ['DINCAE Smoother', 'DINCAE Rougher', 'Total']
        crosstab.index = [i if i != 'All' else 'TOTAL' for i in crosstab.index]
        print("\nCross-tabulation (Winner × Roughness):")
        print(crosstab)
        
        # Key insight: In DINCAE-winning lakes, how often is DINCAE rougher?
        dincae_wins = valid[valid[winner_col] == 'DINCAE']
        if len(dincae_wins) > 0:
            rougher_when_wins = dincae_wins[rougher_col].sum()
            print(f"\n→ When DINCAE wins ({len(dincae_wins)} lakes):")
            print(f"   DINCAE is ROUGHER in {rougher_when_wins}/{len(dincae_wins)} lakes ({100*rougher_when_wins/len(dincae_wins):.1f}%)")
            print(f"   DINCAE is SMOOTHER in {len(dincae_wins)-rougher_when_wins}/{len(dincae_wins)} lakes")
        
        dineof_wins = valid[valid[winner_col] == 'DINEOF']
        if len(dineof_wins) > 0:
            rougher_when_loses = dineof_wins[rougher_col].sum()
            print(f"\n→ When DINEOF wins ({len(dineof_wins)} lakes):")
            print(f"   DINCAE is ROUGHER in {rougher_when_loses}/{len(dineof_wins)} lakes ({100*rougher_when_loses/len(dineof_wins):.1f}%)")


def analyze_reconstruction_bias(lake_table: pd.DataFrame, output_dir: str):
    """
    Key question: Is DINCAE systematically warmer/colder than DINEOF?
    If DINCAE is warmer and buoy is warmer than satellite, that explains the pattern.
    """
    print("\n" + "="*70)
    print("ANALYSIS: Reconstruction BIAS Comparison (ALL LAKES)")
    print("="*70)
    
    for data_type in ['reconstruction', 'reconstruction_missing']:
        dineof_bias = f'{data_type}_dineof_bias'
        dincae_bias = f'{data_type}_dincae_bias'
        bias_diff = f'{data_type}_bias_diff'
        winner_col = f'{data_type}_winner'
        
        if dineof_bias not in lake_table.columns:
            continue
        
        valid = lake_table[[dineof_bias, dincae_bias, bias_diff, winner_col, 'obs_bias', 'lake_id']].dropna()
        
        print(f"\n{data_type.upper()}:")
        print("-" * 50)
        
        print(f"\nMethod biases (negative = reconstruction colder than buoy):")
        print(f"  DINEOF bias: mean={valid[dineof_bias].mean():.4f}, median={valid[dineof_bias].median():.4f}")
        print(f"  DINCAE bias: mean={valid[dincae_bias].mean():.4f}, median={valid[dincae_bias].median():.4f}")
        print(f"  Observation bias: mean={valid['obs_bias'].mean():.4f} (satellite vs buoy baseline)")
        
        print(f"\nBias difference (DINCAE - DINEOF):")
        print(f"  mean={valid[bias_diff].mean():.4f}, median={valid[bias_diff].median():.4f}")
        dincae_warmer = (valid[bias_diff] > 0).sum()
        print(f"  DINCAE warmer than DINEOF: {dincae_warmer}/{len(valid)} lakes ({100*dincae_warmer/len(valid):.1f}%)")
        
        # Key: Does DINCAE being warmer correlate with winning?
        print(f"\nCorrelation: bias_diff vs rmse_diff:")
        rmse_diff = f'{data_type}_rmse_diff'
        if rmse_diff in valid.columns:
            corr = valid[bias_diff].corr(lake_table.loc[valid.index, rmse_diff])
            print(f"  r = {corr:.3f}")
            if corr > 0.3:
                print("  → DINCAE being warmer correlates with DINCAE winning!")
            elif corr < -0.3:
                print("  → DINCAE being warmer correlates with DINEOF winning!")
        
        # By winner group
        print(f"\nBias comparison by winner:")
        for winner in ['DINEOF', 'DINCAE', 'TIE']:
            subset = valid[valid[winner_col] == winner]
            if len(subset) > 0:
                print(f"  {winner} wins ({len(subset)} lakes):")
                print(f"    DINEOF bias: {subset[dineof_bias].mean():.4f}")
                print(f"    DINCAE bias: {subset[dincae_bias].mean():.4f}")
                print(f"    Obs bias: {subset['obs_bias'].mean():.4f}")
                print(f"    DINCAE warmer: {(subset[bias_diff] > 0).sum()}/{len(subset)}")


def analyze_detailed_per_lake(lake_table: pd.DataFrame, output_dir: str):
    """
    Print detailed per-lake breakdown showing ALL relevant metrics.
    """
    print("\n" + "="*70)
    print("DETAILED PER-LAKE BREAKDOWN (ALL 29 LAKES)")
    print("="*70)
    
    data_type = 'reconstruction_missing'
    
    cols = ['lake_id', 
            f'{data_type}_winner',
            f'{data_type}_rmse_diff',
            f'{data_type}_dineof_rmse', 
            f'{data_type}_dincae_rmse',
            f'{data_type}_dineof_bias',
            f'{data_type}_dincae_bias',
            f'{data_type}_bias_diff',
            f'{data_type}_dineof_std',
            f'{data_type}_dincae_std',
            f'{data_type}_std_diff',
            'obs_bias',
            'obs_rmse']
    
    available_cols = [c for c in cols if c in lake_table.columns]
    
    # Sort by DINCAE advantage (rmse_diff)
    rmse_diff_col = f'{data_type}_rmse_diff'
    if rmse_diff_col in lake_table.columns:
        sorted_table = lake_table[available_cols].sort_values(rmse_diff_col, ascending=False)
    else:
        sorted_table = lake_table[available_cols]
    
    print(f"\nSorted by DINCAE advantage (positive = DINCAE better):")
    print("-" * 150)
    
    # Rename columns for display
    display_names = {
        'lake_id': 'Lake',
        f'{data_type}_winner': 'Winner',
        f'{data_type}_rmse_diff': 'RMSE_diff',
        f'{data_type}_dineof_rmse': 'D_RMSE',
        f'{data_type}_dincae_rmse': 'C_RMSE',
        f'{data_type}_dineof_bias': 'D_Bias',
        f'{data_type}_dincae_bias': 'C_Bias',
        f'{data_type}_bias_diff': 'Bias_diff',
        f'{data_type}_dineof_std': 'D_STD',
        f'{data_type}_dincae_std': 'C_STD',
        f'{data_type}_std_diff': 'STD_diff',
        'obs_bias': 'Obs_Bias',
        'obs_rmse': 'Obs_RMSE'
    }
    
    display_df = sorted_table.rename(columns=display_names)
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print(display_df.to_string(index=False))
    
    # Save to CSV
    sorted_table.to_csv(os.path.join(output_dir, 'detailed_per_lake_breakdown.csv'), index=False)
    print(f"\nSaved: detailed_per_lake_breakdown.csv")


def create_comprehensive_scatter_plots(lake_table: pd.DataFrame, output_dir: str):
    """
    Create scatter plots showing relationships between all key variables.
    All 29 lakes shown with labels.
    """
    print("\n" + "="*70)
    print("CREATING: Comprehensive Scatter Plots (ALL LAKES)")
    print("="*70)
    
    data_type = 'reconstruction_missing'
    
    rmse_diff = f'{data_type}_rmse_diff'
    bias_diff = f'{data_type}_bias_diff'
    std_diff = f'{data_type}_std_diff'
    dineof_bias = f'{data_type}_dineof_bias'
    dincae_bias = f'{data_type}_dincae_bias'
    winner_col = f'{data_type}_winner'
    
    if rmse_diff not in lake_table.columns:
        print("Required columns not found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = {'DINEOF': '#5DA5DA', 'DINCAE': '#FAA43A', 'TIE': 'gray'}
    
    # Plot 1: RMSE diff vs Bias diff
    ax = axes[0, 0]
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = lake_table[lake_table[winner_col] == winner]
        ax.scatter(subset[bias_diff], subset[rmse_diff], c=colors[winner], 
                  label=winner, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bias Diff (DINCAE - DINEOF)\n← DINCAE colder | DINCAE warmer →', fontsize=10)
    ax.set_ylabel('RMSE Diff (DINEOF - DINCAE)\n← DINEOF better | DINCAE better →', fontsize=10)
    ax.set_title('Does DINCAE being warmer help?', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    corr = lake_table[bias_diff].corr(lake_table[rmse_diff])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 2: RMSE diff vs STD diff
    ax = axes[0, 1]
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = lake_table[lake_table[winner_col] == winner]
        ax.scatter(subset[std_diff], subset[rmse_diff], c=colors[winner],
                  label=winner, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('STD Diff (DINCAE - DINEOF)\n← DINCAE smoother | DINCAE rougher →', fontsize=10)
    ax.set_ylabel('RMSE Diff', fontsize=10)
    ax.set_title('Does DINCAE being rougher help?', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    corr = lake_table[std_diff].corr(lake_table[rmse_diff])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 3: RMSE diff vs Observation bias
    ax = axes[0, 2]
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = lake_table[lake_table[winner_col] == winner]
        ax.scatter(subset['obs_bias'], subset[rmse_diff], c=colors[winner],
                  label=winner, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Obs Bias (satellite - buoy)\n← Satellite colder | Satellite warmer →', fontsize=10)
    ax.set_ylabel('RMSE Diff', fontsize=10)
    ax.set_title('Does skin-bulk difference predict winner?', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    corr = lake_table['obs_bias'].corr(lake_table[rmse_diff])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 4: DINEOF bias vs DINCAE bias (colored by winner)
    ax = axes[1, 0]
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = lake_table[lake_table[winner_col] == winner]
        ax.scatter(subset[dineof_bias], subset[dincae_bias], c=colors[winner],
                  label=winner, s=100, alpha=0.7, edgecolors='black')
    # Add diagonal line
    lims = [min(lake_table[dineof_bias].min(), lake_table[dincae_bias].min()),
            max(lake_table[dineof_bias].max(), lake_table[dincae_bias].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('DINEOF Bias (reconstruction - buoy)', fontsize=10)
    ax.set_ylabel('DINCAE Bias (reconstruction - buoy)', fontsize=10)
    ax.set_title('Method Biases\n(points above line = DINCAE warmer)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Obs bias vs DINCAE bias (is DINCAE closer to buoy?)
    ax = axes[1, 1]
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = lake_table[lake_table[winner_col] == winner]
        ax.scatter(subset['obs_bias'], subset[dincae_bias], c=colors[winner],
                  label=winner, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='red', linestyle='-', alpha=0.7, label='Zero bias (perfect)')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Obs Bias (satellite - buoy)', fontsize=10)
    ax.set_ylabel('DINCAE Bias (reconstruction - buoy)', fontsize=10)
    ax.set_title('Does DINCAE correct satellite bias?', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    corr = lake_table['obs_bias'].corr(lake_table[dincae_bias])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 6: Obs bias vs DINEOF bias
    ax = axes[1, 2]
    for winner in ['DINEOF', 'DINCAE', 'TIE']:
        subset = lake_table[lake_table[winner_col] == winner]
        ax.scatter(subset['obs_bias'], subset[dineof_bias], c=colors[winner],
                  label=winner, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='red', linestyle='-', alpha=0.7, label='Zero bias (perfect)')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Obs Bias (satellite - buoy)', fontsize=10)
    ax.set_ylabel('DINEOF Bias (reconstruction - buoy)', fontsize=10)
    ax.set_title('Does DINEOF correct satellite bias?', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    corr = lake_table['obs_bias'].corr(lake_table[dineof_bias])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Comprehensive Analysis: What Predicts DINCAE vs DINEOF Winner?\n({data_type}, N={len(lake_table)} lakes)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'diagnostic_comprehensive_scatter.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def analyze_bias_correction_hypothesis(lake_table: pd.DataFrame, output_dir: str):
    """
    Key hypothesis: Does DINCAE "correct" the satellite bias toward buoy?
    
    If obs_bias is negative (satellite colder than buoy), and DINCAE produces
    warmer values than satellite, it would appear to do better against buoy.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS TEST: Does DINCAE correct satellite bias?")
    print("="*70)
    
    data_type = 'reconstruction_missing'
    
    # Compute: how much does each method deviate from observation?
    # If method produces values closer to buoy than satellite does, 
    # method_bias should be closer to 0 than obs_bias
    
    dineof_bias = f'{data_type}_dineof_bias'
    dincae_bias = f'{data_type}_dincae_bias'
    
    if dineof_bias not in lake_table.columns:
        return
    
    valid = lake_table[['lake_id', dineof_bias, dincae_bias, 'obs_bias']].dropna()
    
    # How much does each method reduce the bias?
    # bias_reduction = |obs_bias| - |method_bias|
    # positive = method reduces bias (gets closer to buoy)
    
    valid['dineof_bias_reduction'] = np.abs(valid['obs_bias']) - np.abs(valid[dineof_bias])
    valid['dincae_bias_reduction'] = np.abs(valid['obs_bias']) - np.abs(valid[dincae_bias])
    
    print("\nBias reduction (positive = method reduces satellite-buoy gap):")
    print(f"  DINEOF: mean={valid['dineof_bias_reduction'].mean():.4f}")
    print(f"  DINCAE: mean={valid['dincae_bias_reduction'].mean():.4f}")
    
    print(f"\nHow many lakes does each method reduce bias?")
    dineof_reduces = (valid['dineof_bias_reduction'] > 0).sum()
    dincae_reduces = (valid['dincae_bias_reduction'] > 0).sum()
    print(f"  DINEOF reduces bias: {dineof_reduces}/{len(valid)} lakes")
    print(f"  DINCAE reduces bias: {dincae_reduces}/{len(valid)} lakes")
    
    # Key: Does DINCAE reduce bias MORE than DINEOF?
    valid['dincae_reduces_more'] = valid['dincae_bias_reduction'] > valid['dineof_bias_reduction']
    dincae_better_at_bias = valid['dincae_reduces_more'].sum()
    print(f"\n  DINCAE reduces bias MORE than DINEOF: {dincae_better_at_bias}/{len(valid)} lakes ({100*dincae_better_at_bias/len(valid):.1f}%)")
    
    # Does this correlate with winning?
    winner_col = f'{data_type}_winner'
    if winner_col in lake_table.columns:
        valid = valid.merge(lake_table[['lake_id', winner_col]], on='lake_id')
        
        print("\nBy winner group:")
        for winner in ['DINEOF', 'DINCAE', 'TIE']:
            subset = valid[valid[winner_col] == winner]
            if len(subset) > 0:
                dincae_reduces_more = subset['dincae_reduces_more'].sum()
                print(f"  {winner} wins ({len(subset)} lakes): DINCAE reduces bias more in {dincae_reduces_more}/{len(subset)} ({100*dincae_reduces_more/len(subset):.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.scatter(valid['dineof_bias_reduction'], valid['dincae_bias_reduction'], 
              c=['#FAA43A' if w == 'DINCAE' else '#5DA5DA' if w == 'DINEOF' else 'gray' 
                 for w in valid[winner_col]], s=100, alpha=0.7, edgecolors='black')
    lims = [min(valid['dineof_bias_reduction'].min(), valid['dincae_bias_reduction'].min()) - 0.05,
            max(valid['dineof_bias_reduction'].max(), valid['dincae_bias_reduction'].max()) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('DINEOF Bias Reduction (°C)', fontsize=11)
    ax.set_ylabel('DINCAE Bias Reduction (°C)', fontsize=11)
    ax.set_title('Which Method Reduces Satellite-Buoy Bias More?\n'
                '(Points above diagonal = DINCAE reduces more)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#5DA5DA', label='DINEOF wins'),
                       Patch(facecolor='#FAA43A', label='DINCAE wins'),
                       Patch(facecolor='gray', label='TIE')]
    ax.legend(handles=legend_elements)
    
    ax = axes[1]
    # Show bias reduction difference vs RMSE difference
    rmse_diff = f'{data_type}_rmse_diff'
    valid_with_rmse = valid.merge(lake_table[['lake_id', rmse_diff]], on='lake_id')
    valid_with_rmse['bias_reduction_diff'] = valid_with_rmse['dincae_bias_reduction'] - valid_with_rmse['dineof_bias_reduction']
    
    ax.scatter(valid_with_rmse['bias_reduction_diff'], valid_with_rmse[rmse_diff],
              c=['#FAA43A' if w == 'DINCAE' else '#5DA5DA' if w == 'DINEOF' else 'gray'
                 for w in valid_with_rmse[winner_col]], s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bias Reduction Diff (DINCAE - DINEOF)\n← DINEOF reduces more | DINCAE reduces more →', fontsize=10)
    ax.set_ylabel('RMSE Diff (DINEOF - DINCAE)\n← DINEOF better | DINCAE better →', fontsize=10)
    ax.set_title('Does Bias Reduction Predict Winner?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    corr = valid_with_rmse['bias_reduction_diff'].corr(valid_with_rmse[rmse_diff])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(facecolor='lightyellow', alpha=0.8))
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'diagnostic_bias_reduction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep diagnostic: Why does DINCAE win on some lakes?")
    parser.add_argument("--analysis_dir", required=True, help="insitu_validation_analysis folder")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "deep_diagnostics")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("DEEP DIAGNOSTIC: Why does DINCAE win despite being rougher?")
    print("="*70)
    
    # Load data
    df = load_data(args.analysis_dir)
    print(f"Loaded {len(df)} records from {df['lake_id_cci'].nunique()} lakes")
    
    # Build comprehensive lake table
    print("\nBuilding comprehensive per-lake table...")
    lake_table = build_comprehensive_lake_table(df)
    print(f"Built table with {len(lake_table)} lakes and {len(lake_table.columns)} columns")
    
    # Run analyses
    analyze_winner_vs_roughness(lake_table, args.output_dir)
    analyze_reconstruction_bias(lake_table, args.output_dir)
    analyze_detailed_per_lake(lake_table, args.output_dir)
    create_comprehensive_scatter_plots(lake_table, args.output_dir)
    analyze_bias_correction_hypothesis(lake_table, args.output_dir)
    
    # Save full table
    lake_table.to_csv(os.path.join(args.output_dir, 'comprehensive_lake_table.csv'), index=False)
    print(f"\nSaved: comprehensive_lake_table.csv")
    
    print("\n" + "="*70)
    print("DEEP DIAGNOSTIC COMPLETE")
    print("="*70)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
