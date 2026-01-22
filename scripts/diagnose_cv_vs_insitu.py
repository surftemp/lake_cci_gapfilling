#!/usr/bin/env python3
"""
Diagnostic script to investigate why CV and In-Situ validation tell different stories.

Key question: DINEOF dominates in CV (117/120 lakes), but in-situ shows mixed results.
Why?

Hypotheses to test:
1. reconstruction_observed (like CV) vs reconstruction_missing (true gap-fill) split
2. Buoy location bias (always-cloudy areas)
3. Noise floor effect (observation already has ~1°C error)
4. Sample size differences

Run after analyze_insitu_validation.py has generated all_insitu_stats_combined.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(analysis_dir: str) -> pd.DataFrame:
    """Load the combined stats CSV."""
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        print("Run analyze_insitu_validation.py first")
        sys.exit(1)
    return pd.read_csv(csv_path)


def analyze_observed_vs_missing_winner(df: pd.DataFrame, output_dir: str):
    """
    HYPOTHESIS 1: Does DINEOF dominate reconstruction_observed (like CV) 
    but lose on reconstruction_missing (true gap-fill)?
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Observed vs Missing Split")
    print("="*70)
    
    results = []
    
    for data_type in ['reconstruction_observed', 'reconstruction_missing', 'reconstruction']:
        dineof = df[(df['method'] == 'dineof') & (df['data_type'] == data_type)]
        dincae = df[(df['method'] == 'dincae') & (df['data_type'] == data_type)]
        
        if dineof.empty or dincae.empty:
            continue
        
        # Aggregate to per-lake
        dineof_lakes = dineof.groupby('lake_id_cci')['rmse'].mean().reset_index()
        dincae_lakes = dincae.groupby('lake_id_cci')['rmse'].mean().reset_index()
        
        merged = dineof_lakes.merge(dincae_lakes, on='lake_id_cci', suffixes=('_dineof', '_dincae'))
        
        dineof_wins = (merged['rmse_dineof'] < merged['rmse_dincae']).sum()
        dincae_wins = (merged['rmse_dincae'] < merged['rmse_dineof']).sum()
        total = len(merged)
        
        results.append({
            'data_type': data_type,
            'dineof_wins': dineof_wins,
            'dincae_wins': dincae_wins,
            'total_lakes': total,
            'dineof_pct': 100 * dineof_wins / total if total > 0 else 0
        })
        
        print(f"\n{data_type}:")
        print(f"  DINEOF wins: {dineof_wins}/{total} lakes ({100*dineof_wins/total:.1f}%)")
        print(f"  DINCAE wins: {dincae_wins}/{total} lakes ({100*dincae_wins/total:.1f}%)")
    
    # Create comparison figure
    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results))
        width = 0.35
        
        dineof_pcts = [r['dineof_pct'] for r in results]
        dincae_pcts = [100 - r['dineof_pct'] for r in results]
        labels = [r['data_type'].replace('reconstruction_', 'recon_') for r in results]
        
        ax.bar(x - width/2, dineof_pcts, width, label='DINEOF wins', color='#5DA5DA')
        ax.bar(x + width/2, dincae_pcts, width, label='DINCAE wins', color='#FAA43A')
        
        ax.set_ylabel('% of Lakes', fontsize=12)
        ax.set_title('Who Wins? Observed Pixels vs Missing Pixels vs All\n'
                    '(If DINEOF dominates "observed" but not "missing", explains CV vs in-situ discrepancy)',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.axhline(50, color='black', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add counts as text
        for i, r in enumerate(results):
            ax.text(i - width/2, dineof_pcts[i] + 2, f"{r['dineof_wins']}", ha='center', fontsize=9)
            ax.text(i + width/2, dincae_pcts[i] + 2, f"{r['dincae_wins']}", ha='center', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'diagnostic_observed_vs_missing_winner.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {save_path}")
    
    return results


def analyze_noise_floor(df: pd.DataFrame, output_dir: str):
    """
    HYPOTHESIS 2: Is the observation error so high that method differences don't matter?
    
    If observation RMSE is ~1°C and method difference is ~0.1°C, noise dominates.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Noise Floor Analysis")
    print("="*70)
    
    # Get observation stats (satellite vs buoy baseline)
    obs_df = df[df['data_type'] == 'observation']
    
    # Get reconstruction stats for both methods
    dineof_recon = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction')]
    dincae_recon = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction')]
    
    if obs_df.empty or dineof_recon.empty or dincae_recon.empty:
        print("Insufficient data for noise floor analysis")
        return
    
    # Per-lake comparison
    obs_lakes = obs_df.groupby('lake_id_cci')['rmse'].mean().reset_index()
    obs_lakes.columns = ['lake_id_cci', 'obs_rmse']
    
    dineof_lakes = dineof_recon.groupby('lake_id_cci')['rmse'].mean().reset_index()
    dineof_lakes.columns = ['lake_id_cci', 'dineof_rmse']
    
    dincae_lakes = dincae_recon.groupby('lake_id_cci')['rmse'].mean().reset_index()
    dincae_lakes.columns = ['lake_id_cci', 'dincae_rmse']
    
    merged = obs_lakes.merge(dineof_lakes, on='lake_id_cci').merge(dincae_lakes, on='lake_id_cci')
    
    # Compute method difference relative to observation baseline
    merged['method_diff'] = abs(merged['dineof_rmse'] - merged['dincae_rmse'])
    merged['signal_to_noise'] = merged['method_diff'] / merged['obs_rmse']
    
    print(f"\nObservation RMSE (satellite vs buoy baseline):")
    print(f"  Mean:   {merged['obs_rmse'].mean():.3f}°C")
    print(f"  Median: {merged['obs_rmse'].median():.3f}°C")
    print(f"  Range:  {merged['obs_rmse'].min():.3f} - {merged['obs_rmse'].max():.3f}°C")
    
    print(f"\nMethod difference (|DINEOF - DINCAE|):")
    print(f"  Mean:   {merged['method_diff'].mean():.3f}°C")
    print(f"  Median: {merged['method_diff'].median():.3f}°C")
    
    print(f"\nSignal-to-noise ratio (method_diff / obs_rmse):")
    print(f"  Mean:   {merged['signal_to_noise'].mean():.3f}")
    print(f"  Median: {merged['signal_to_noise'].median():.3f}")
    
    low_snr = (merged['signal_to_noise'] < 0.1).sum()
    print(f"\n  Lakes where method difference < 10% of obs noise: {low_snr}/{len(merged)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Method difference vs observation RMSE
    ax = axes[0]
    ax.scatter(merged['obs_rmse'], merged['method_diff'], alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Observation RMSE (°C) - Satellite vs Buoy Baseline', fontsize=11)
    ax.set_ylabel('|DINEOF - DINCAE| RMSE (°C)', fontsize=11)
    ax.set_title('Method Difference vs Observation Noise\n'
                '(Points near bottom = method difference lost in noise)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add 10% and 20% lines
    x_line = np.linspace(0, merged['obs_rmse'].max() * 1.1, 100)
    ax.plot(x_line, x_line * 0.1, 'r--', label='10% of obs noise', alpha=0.7)
    ax.plot(x_line, x_line * 0.2, 'orange', linestyle='--', label='20% of obs noise', alpha=0.7)
    ax.legend()
    
    # Panel 2: Histogram of signal-to-noise
    ax = axes[1]
    ax.hist(merged['signal_to_noise'], bins=20, edgecolor='black', alpha=0.7, color='#3498db')
    ax.axvline(0.1, color='red', linestyle='--', linewidth=2, label='10% threshold')
    ax.set_xlabel('Signal-to-Noise Ratio (method_diff / obs_rmse)', fontsize=11)
    ax.set_ylabel('Number of Lakes', fontsize=11)
    ax.set_title(f'Distribution of Signal-to-Noise\n'
                f'({low_snr}/{len(merged)} lakes below 10% threshold)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'diagnostic_noise_floor.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def analyze_sample_size(df: pd.DataFrame, output_dir: str):
    """
    HYPOTHESIS 3: Sample size differences between reconstruction_observed and reconstruction_missing
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Sample Size Analysis")
    print("="*70)
    
    for data_type in ['observation', 'reconstruction', 'reconstruction_observed', 'reconstruction_missing']:
        subset = df[df['data_type'] == data_type]
        if subset.empty:
            continue
        
        total_points = subset['n_matches'].sum()
        n_lakes = subset['lake_id_cci'].nunique()
        avg_per_lake = total_points / n_lakes if n_lakes > 0 else 0
        
        print(f"\n{data_type}:")
        print(f"  Total validation points: {total_points:,}")
        print(f"  Number of lakes: {n_lakes}")
        print(f"  Average points per lake: {avg_per_lake:.0f}")


def analyze_specific_discrepant_lakes(df: pd.DataFrame, output_dir: str):
    """
    Find lakes where in-situ says DINCAE wins but we expect DINEOF to win (from CV).
    Investigate what's special about these lakes.
    """
    print("\n" + "="*70)
    print("ANALYSIS: Lakes Where DINCAE Wins (Unexpected)")
    print("="*70)
    
    # Get reconstruction_observed (most like CV)
    dineof = df[(df['method'] == 'dineof') & (df['data_type'] == 'reconstruction_observed')]
    dincae = df[(df['method'] == 'dincae') & (df['data_type'] == 'reconstruction_observed')]
    obs = df[df['data_type'] == 'observation']
    
    if dineof.empty or dincae.empty:
        print("Insufficient data")
        return
    
    dineof_lakes = dineof.groupby('lake_id_cci').agg({
        'rmse': 'mean', 'n_matches': 'sum'
    }).reset_index()
    dineof_lakes.columns = ['lake_id_cci', 'dineof_rmse', 'dineof_n']
    
    dincae_lakes = dincae.groupby('lake_id_cci').agg({
        'rmse': 'mean', 'n_matches': 'sum'
    }).reset_index()
    dincae_lakes.columns = ['lake_id_cci', 'dincae_rmse', 'dincae_n']
    
    obs_lakes = obs.groupby('lake_id_cci').agg({
        'rmse': 'mean', 'n_matches': 'sum'
    }).reset_index()
    obs_lakes.columns = ['lake_id_cci', 'obs_rmse', 'obs_n']
    
    merged = dineof_lakes.merge(dincae_lakes, on='lake_id_cci').merge(obs_lakes, on='lake_id_cci')
    
    # Lakes where DINCAE wins on reconstruction_observed
    dincae_wins = merged[merged['dincae_rmse'] < merged['dineof_rmse']].copy()
    dincae_wins['advantage'] = dincae_wins['dineof_rmse'] - dincae_wins['dincae_rmse']
    dincae_wins = dincae_wins.sort_values('advantage', ascending=False)
    
    print(f"\nLakes where DINCAE wins on reconstruction_observed ({len(dincae_wins)} lakes):")
    print("-" * 80)
    print(f"{'Lake ID':<12} {'DINEOF RMSE':<12} {'DINCAE RMSE':<12} {'Advantage':<10} {'Obs RMSE':<10} {'N points':<10}")
    print("-" * 80)
    
    for _, row in dincae_wins.head(15).iterrows():
        print(f"{int(row['lake_id_cci']):<12} {row['dineof_rmse']:.3f}°C{'':<5} {row['dincae_rmse']:.3f}°C{'':<5} "
              f"{row['advantage']:.3f}°C{'':<3} {row['obs_rmse']:.3f}°C{'':<3} {int(row['dineof_n']):<10}")
    
    # Check if these lakes have high observation error
    if len(dincae_wins) > 0:
        print(f"\nCharacteristics of lakes where DINCAE wins:")
        print(f"  Mean obs RMSE: {dincae_wins['obs_rmse'].mean():.3f}°C")
        
        dineof_wins_lakes = merged[merged['dineof_rmse'] < merged['dincae_rmse']]
        if len(dineof_wins_lakes) > 0:
            print(f"  (vs lakes where DINEOF wins: {dineof_wins_lakes['obs_rmse'].mean():.3f}°C)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostic analysis: CV vs In-Situ discrepancy")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis folder")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as analysis_dir)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.analysis_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("DIAGNOSTIC: Why does CV and In-Situ validation tell different stories?")
    print("="*70)
    
    df = load_data(args.analysis_dir)
    print(f"Loaded {len(df)} records from {df['lake_id_cci'].nunique()} lakes")
    
    # Run analyses
    analyze_observed_vs_missing_winner(df, args.output_dir)
    analyze_noise_floor(df, args.output_dir)
    analyze_sample_size(df, args.output_dir)
    analyze_specific_discrepant_lakes(df, args.output_dir)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
