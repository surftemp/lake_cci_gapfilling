#!/usr/bin/env python3
"""
Comprehensive Diagnostic: Why does In-Situ validation tell a different story than CV?

CV Result: DINEOF wins 117/120 lakes
In-Situ: More balanced (DINCAE competitive)

This script investigates:
1. Direct comparison: DINEOF vs DINCAE vs Satellite vs Buoy at same points
2. Pattern analysis: What do lakes where DINCAE wins have in common?
   - Geographic patterns (latitude, climate zone)
   - Lake characteristics (size, depth if available)
   - Buoy characteristics (distance to shore, gap frequency at buoy pixel)
   - Data quality patterns (observation counts, seasonal coverage)
   - Method behavior patterns (smoothness, variability)

Usage:
    python diagnose_cv_vs_insitu_comprehensive.py \
        --run_root /path/to/experiment \
        --analysis_dir /path/to/insitu_validation_analysis \
        --output_dir /path/to/output

Requires:
    - In-situ validation results (CSV)
    - Post-processed NetCDF files (DINEOF, DINCAE outputs)
    - Prepared NetCDF files (for gap frequency analysis)
    - Lake metadata (globolakes CSV) if available
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("Warning: xarray not available - some analyses will be skipped")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_insitu_stats(analysis_dir: str) -> pd.DataFrame:
    """Load combined in-situ validation stats."""
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stats file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_lake_metadata(metadata_paths: List[str]) -> Optional[pd.DataFrame]:
    """Load lake metadata (lat, lon, area, etc.) from globolakes or similar."""
    for path in metadata_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='iso-8859-1') as f:
                    first_line = f.readline()
                
                if 'BADC-CSV' in first_line or 'Conventions,G' in first_line:
                    import csv
                    lake_data = []
                    with open(path, 'r', encoding='iso-8859-1') as f:
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
                                lake_name = line[1].strip() if len(line) > 1 else ''
                                country = line[2].strip() if len(line) > 2 else ''
                                lat = float(line[3].strip())
                                lon = float(line[4].strip())
                                area = float(line[5].strip()) if len(line) > 5 and line[5].strip() else np.nan
                                lake_data.append({
                                    'lake_id': lake_id, 
                                    'lake_name': lake_name,
                                    'country': country,
                                    'lat': lat, 
                                    'lon': lon,
                                    'area_km2': area
                                })
                            except (ValueError, IndexError):
                                continue
                    return pd.DataFrame(lake_data)
                else:
                    df = pd.read_csv(path)
                    if 'Lake ID' in df.columns:
                        df = df.rename(columns={'Lake ID': 'lake_id', 'Latitude, Centre': 'lat', 'Longitude, Centre': 'lon'})
                    return df
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
    return None


# =============================================================================
# ANALYSIS 1: WINNER CLASSIFICATION
# =============================================================================

def classify_winners(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each lake as DINEOF-winner, DINCAE-winner, or TIE based on RMSE.
    """
    results = []
    
    for lake_id in df['lake_id_cci'].unique():
        lake_df = df[df['lake_id_cci'] == lake_id]
        
        row = {'lake_id_cci': lake_id}
        
        for data_type in ['reconstruction', 'reconstruction_observed', 'reconstruction_missing', 'observation']:
            dt_df = lake_df[lake_df['data_type'] == data_type]
            
            dineof = dt_df[dt_df['method'] == 'dineof']['rmse'].mean()
            dincae = dt_df[dt_df['method'] == 'dincae']['rmse'].mean()
            
            if pd.isna(dineof) or pd.isna(dincae):
                row[f'{data_type}_winner'] = 'NA'
                row[f'{data_type}_diff'] = np.nan
                row[f'{data_type}_dineof_rmse'] = dineof
                row[f'{data_type}_dincae_rmse'] = dincae
            else:
                diff = dineof - dincae  # Positive = DINCAE better
                if abs(diff) < 0.02:
                    winner = 'TIE'
                elif diff > 0:
                    winner = 'DINCAE'
                else:
                    winner = 'DINEOF'
                
                row[f'{data_type}_winner'] = winner
                row[f'{data_type}_diff'] = diff
                row[f'{data_type}_dineof_rmse'] = dineof
                row[f'{data_type}_dincae_rmse'] = dincae
        
        # Get observation stats
        obs_df = lake_df[lake_df['data_type'] == 'observation']
        for metric in ['rmse', 'mae', 'bias', 'std', 'correlation']:
            if metric in obs_df.columns:
                row[f'obs_{metric}'] = obs_df[metric].mean()
        
        row['n_matches_obs'] = lake_df[lake_df['data_type'] == 'observation']['n_matches'].sum()
        row['n_matches_recon'] = lake_df[lake_df['data_type'] == 'reconstruction']['n_matches'].sum()
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_geographic_patterns(winner_df: pd.DataFrame, lake_meta: pd.DataFrame, output_dir: str):
    """Do DINCAE-winning lakes cluster geographically?"""
    print("\n" + "="*70)
    print("ANALYSIS: Geographic Patterns")
    print("="*70)
    
    if lake_meta is None:
        print("No lake metadata available - skipping geographic analysis")
        return None
    
    merged = winner_df.merge(lake_meta[['lake_id', 'lat', 'lon', 'lake_name', 'country']], 
                              left_on='lake_id_cci', right_on='lake_id', how='left')
    merged = merged[merged['lat'].notna()]
    
    if len(merged) == 0:
        print("No lakes matched with location data")
        return None
    
    data_type = 'reconstruction_missing'
    winner_col = f'{data_type}_winner'
    if winner_col not in merged.columns:
        data_type = 'reconstruction'
        winner_col = f'{data_type}_winner'
    
    dineof_wins = merged[merged[winner_col] == 'DINEOF']
    dincae_wins = merged[merged[winner_col] == 'DINCAE']
    
    print(f"\nUsing data_type: {data_type}")
    print(f"  DINEOF wins: {len(dineof_wins)} lakes")
    print(f"  DINCAE wins: {len(dincae_wins)} lakes")
    
    if len(dineof_wins) > 0 and len(dincae_wins) > 0:
        print(f"\nLatitude comparison:")
        print(f"  DINEOF wins: mean lat={dineof_wins['lat'].mean():.2f}°")
        print(f"  DINCAE wins: mean lat={dincae_wins['lat'].mean():.2f}°")
        
        if HAS_SCIPY and len(dineof_wins) >= 3 and len(dincae_wins) >= 3:
            stat, pval = stats.mannwhitneyu(dineof_wins['lat'], dincae_wins['lat'], alternative='two-sided')
            print(f"  Mann-Whitney p={pval:.4f}")
    
    # Create map
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if len(dineof_wins) > 0:
        ax.scatter(dineof_wins['lon'], dineof_wins['lat'], c='#5DA5DA', s=100, 
                  label=f'DINEOF wins ({len(dineof_wins)})', alpha=0.7, edgecolors='black')
    if len(dincae_wins) > 0:
        ax.scatter(dincae_wins['lon'], dincae_wins['lat'], c='#FAA43A', s=100,
                  label=f'DINCAE wins ({len(dincae_wins)})', alpha=0.7, edgecolors='black')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Geographic Distribution of Winners\nBlue=DINEOF, Orange=DINCAE', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_geographic_winners.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: diagnostic_geographic_winners.png")
    
    return merged


def analyze_observation_quality_patterns(winner_df: pd.DataFrame, output_dir: str):
    """Do DINCAE-winning lakes have worse satellite-buoy baseline?"""
    print("\n" + "="*70)
    print("ANALYSIS: Observation Quality Patterns")
    print("="*70)
    
    data_type = 'reconstruction_missing'
    winner_col = f'{data_type}_winner'
    if winner_col not in winner_df.columns:
        data_type = 'reconstruction'
        winner_col = f'{data_type}_winner'
    
    dineof_wins = winner_df[winner_df[winner_col] == 'DINEOF']
    dincae_wins = winner_df[winner_df[winner_col] == 'DINCAE']
    
    metrics = ['obs_rmse', 'obs_mae', 'obs_bias', 'obs_std']
    available = [m for m in metrics if m in winner_df.columns]
    
    print(f"\nComparing observation quality between winner groups:")
    
    for metric in available:
        dineof_vals = dineof_wins[metric].dropna()
        dincae_vals = dincae_wins[metric].dropna()
        
        if len(dineof_vals) > 0 and len(dincae_vals) > 0:
            print(f"\n  {metric}:")
            print(f"    DINEOF-winning: mean={dineof_vals.mean():.3f}, median={dineof_vals.median():.3f}")
            print(f"    DINCAE-winning: mean={dincae_vals.mean():.3f}, median={dincae_vals.median():.3f}")
            
            if HAS_SCIPY and len(dineof_vals) >= 3 and len(dincae_vals) >= 3:
                stat, pval = stats.mannwhitneyu(dineof_vals, dincae_vals, alternative='two-sided')
                print(f"    Mann-Whitney p={pval:.4f} {'*' if pval < 0.05 else ''}")
    
    # Visualization
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(4*len(available), 5))
        if len(available) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available):
            dineof_vals = dineof_wins[metric].dropna()
            dincae_vals = dincae_wins[metric].dropna()
            
            data = [dineof_vals, dincae_vals] if len(dineof_vals) > 0 and len(dincae_vals) > 0 else []
            labels = [f'DINEOF\nwins ({len(dineof_vals)})', f'DINCAE\nwins ({len(dincae_vals)})']
            
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('#5DA5DA')
                bp['boxes'][1].set_facecolor('#FAA43A')
                for patch in bp['boxes']:
                    patch.set_alpha(0.7)
            
            ax.set_ylabel(f'{metric} (°C)', fontsize=11)
            ax.set_title(f'{metric}', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Observation Quality by Winner Group\n(Higher = worse satellite-buoy match)', 
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diagnostic_obs_quality_by_winner.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: diagnostic_obs_quality_by_winner.png")


def analyze_method_smoothness(df: pd.DataFrame, winner_df: pd.DataFrame, output_dir: str):
    """Is DINCAE systematically smoother?"""
    print("\n" + "="*70)
    print("ANALYSIS: Method Smoothness (STD comparison)")
    print("="*70)
    
    print("\nComparing STD between methods (lower STD = smoother):")
    
    for data_type in ['reconstruction', 'reconstruction_missing']:
        dt_df = df[df['data_type'] == data_type]
        
        if 'std' not in dt_df.columns:
            continue
        
        dineof_std = dt_df[dt_df['method'] == 'dineof'].groupby('lake_id_cci')['std'].mean()
        dincae_std = dt_df[dt_df['method'] == 'dincae'].groupby('lake_id_cci')['std'].mean()
        
        common = dineof_std.index.intersection(dincae_std.index)
        
        if len(common) > 0:
            std_diff = dincae_std[common] - dineof_std[common]
            
            print(f"\n  {data_type}:")
            print(f"    DINCAE STD - DINEOF STD: mean={std_diff.mean():.4f}, median={std_diff.median():.4f}")
            smoother_dincae = (std_diff < 0).sum()
            smoother_dineof = (std_diff > 0).sum()
            print(f"    DINCAE smoother: {smoother_dincae}/{len(common)} lakes ({100*smoother_dincae/len(common):.1f}%)")


def create_comprehensive_comparison(df: pd.DataFrame, winner_df: pd.DataFrame, output_dir: str):
    """Create comprehensive multi-panel comparison figure."""
    print("\n" + "="*70)
    print("CREATING: Comprehensive Comparison Figure")
    print("="*70)
    
    data_type = 'reconstruction_missing'
    winner_col = f'{data_type}_winner'
    diff_col = f'{data_type}_diff'
    
    if winner_col not in winner_df.columns:
        data_type = 'reconstruction'
        winner_col = f'{data_type}_winner'
        diff_col = f'{data_type}_diff'
    
    plot_df = winner_df[['lake_id_cci', winner_col, diff_col, 'obs_rmse', 'n_matches_recon']].copy()
    plot_df = plot_df.dropna(subset=[diff_col])
    plot_df = plot_df.sort_values(diff_col)
    
    if len(plot_df) == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: RMSE difference per lake
    ax = axes[0, 0]
    colors = ['#FAA43A' if d > 0 else '#5DA5DA' for d in plot_df[diff_col]]
    ax.bar(range(len(plot_df)), plot_df[diff_col], color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Lakes (sorted)', fontsize=11)
    ax.set_ylabel('RMSE Diff (DINEOF - DINCAE) °C', fontsize=11)
    ax.set_title(f'Method RMSE Difference ({data_type})\n'
                f'Positive = DINCAE better', fontsize=12, fontweight='bold')
    
    n_dincae = (plot_df[diff_col] > 0).sum()
    n_dineof = (plot_df[diff_col] < 0).sum()
    ax.text(0.02, 0.98, f'DINCAE wins: {n_dincae}\nDINEOF wins: {n_dineof}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 2: RMSE diff vs obs RMSE
    ax = axes[0, 1]
    if 'obs_rmse' in plot_df.columns:
        colors = ['#FAA43A' if d > 0 else '#5DA5DA' for d in plot_df[diff_col]]
        ax.scatter(plot_df['obs_rmse'], plot_df[diff_col], c=colors, alpha=0.7, 
                  edgecolors='black', linewidth=0.5, s=80)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Observation RMSE (satellite vs buoy) °C', fontsize=11)
        ax.set_ylabel('Method RMSE Diff °C', fontsize=11)
        ax.set_title('Method Advantage vs Observation Quality', fontsize=12, fontweight='bold')
        
        valid = plot_df.dropna(subset=['obs_rmse'])
        if len(valid) > 3:
            corr = valid[diff_col].corr(valid['obs_rmse'])
            ax.text(0.98, 0.98, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.3)
    
    # Panel 3: Distribution of differences
    ax = axes[1, 0]
    ax.hist(plot_df[diff_col], bins=20, edgecolor='black', alpha=0.7, color='#9b59b6')
    ax.axvline(0, color='black', linewidth=2)
    ax.axvline(plot_df[diff_col].mean(), color='blue', linestyle='--', linewidth=2, 
              label=f'Mean: {plot_df[diff_col].mean():.3f}')
    ax.axvline(plot_df[diff_col].median(), color='red', linestyle=':', linewidth=2,
              label=f'Median: {plot_df[diff_col].median():.3f}')
    ax.set_xlabel('RMSE Diff (DINEOF - DINCAE) °C', fontsize=11)
    ax.set_ylabel('Number of Lakes', fontsize=11)
    ax.set_title('Distribution of Differences', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 4: RMSE diff vs sample size
    ax = axes[1, 1]
    colors = ['#FAA43A' if d > 0 else '#5DA5DA' for d in plot_df[diff_col]]
    ax.scatter(plot_df['n_matches_recon'], plot_df[diff_col], c=colors, alpha=0.7,
              edgecolors='black', linewidth=0.5, s=80)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Number of Validation Points', fontsize=11)
    ax.set_ylabel('Method RMSE Diff °C', fontsize=11)
    ax.set_title('Method Advantage vs Sample Size', fontsize=12, fontweight='bold')
    
    valid = plot_df.dropna(subset=['n_matches_recon'])
    if len(valid) > 3:
        corr = valid[diff_col].corr(valid['n_matches_recon'])
        ax.text(0.98, 0.98, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Analysis: What Predicts Winner?', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_comprehensive_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: diagnostic_comprehensive_comparison.png")


def list_dincae_winning_lakes(winner_df: pd.DataFrame, merged_geo: pd.DataFrame, output_dir: str):
    """List DINCAE-winning lakes for investigation."""
    print("\n" + "="*70)
    print("DINCAE-WINNING LAKES")
    print("="*70)
    
    data_type = 'reconstruction_missing'
    winner_col = f'{data_type}_winner'
    diff_col = f'{data_type}_diff'
    
    if winner_col not in winner_df.columns:
        data_type = 'reconstruction'
        winner_col = f'{data_type}_winner'
        diff_col = f'{data_type}_diff'
    
    dincae_wins = winner_df[winner_df[winner_col] == 'DINCAE'].copy()
    
    if len(dincae_wins) == 0:
        print("No DINCAE-winning lakes found")
        return
    
    # Add geo info if available
    if merged_geo is not None:
        geo_cols = ['lake_id_cci', 'lake_name', 'country', 'lat', 'lon']
        geo_cols = [c for c in geo_cols if c in merged_geo.columns]
        dincae_wins = dincae_wins.merge(merged_geo[geo_cols], on='lake_id_cci', how='left')
    
    dincae_wins = dincae_wins.sort_values(diff_col, ascending=False)
    
    print(f"\nTotal DINCAE-winning lakes: {len(dincae_wins)}")
    print("\nDetails (sorted by DINCAE advantage):")
    print("-" * 100)
    
    for _, row in dincae_wins.iterrows():
        lake_id = int(row['lake_id_cci'])
        diff = row[diff_col]
        obs_rmse = row.get('obs_rmse', np.nan)
        name = row.get('lake_name', 'Unknown')
        country = row.get('country', '')
        lat = row.get('lat', np.nan)
        
        print(f"  Lake {lake_id}: DINCAE wins by {diff:.3f}°C | obs_rmse={obs_rmse:.3f}°C | "
              f"lat={lat:.1f}° | {name} ({country})")
    
    # Save
    save_path = os.path.join(output_dir, 'dincae_winning_lakes.csv')
    dincae_wins.to_csv(save_path, index=False)
    print(f"\nSaved: {save_path}")


def generate_summary(winner_df: pd.DataFrame, output_dir: str):
    """Generate summary of findings."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    data_type = 'reconstruction_missing'
    winner_col = f'{data_type}_winner'
    if winner_col not in winner_df.columns:
        data_type = 'reconstruction'
        winner_col = f'{data_type}_winner'
    
    dineof_n = (winner_df[winner_col] == 'DINEOF').sum()
    dincae_n = (winner_df[winner_col] == 'DINCAE').sum()
    tie_n = (winner_df[winner_col] == 'TIE').sum()
    total = dineof_n + dincae_n + tie_n
    
    summary = f"""
================================================================================
DIAGNOSTIC SUMMARY
================================================================================

PUZZLE:
  - Satellite-based CV: DINEOF wins ~117/120 lakes
  - In-situ validation: Results below

IN-SITU RESULTS ({data_type}):
  - DINEOF wins: {dineof_n}/{total} ({100*dineof_n/total:.1f}%)
  - DINCAE wins: {dincae_n}/{total} ({100*dincae_n/total:.1f}%)
  - TIE (<0.02°C): {tie_n}/{total} ({100*tie_n/total:.1f}%)

KEY DIFFERENCES BETWEEN CV AND IN-SITU:
  1. Ground truth: CV=satellite, In-situ=buoy
     - Satellite = skin temperature (~10μm depth)
     - Buoy = bulk temperature (~0.5-1m depth)
     - These are PHYSICALLY DIFFERENT quantities!
  
  2. Spatial sampling: CV=many pixels, In-situ=ONE pixel
  
  3. What's tested:
     - CV: "Does reconstruction match satellite?"
     - In-situ: "Does reconstruction match buoy?"

POSSIBLE EXPLANATIONS:
  1. Skin vs bulk: DINCAE smoothing might match bulk better by coincidence
  2. Single pixel: Buoy pixel might be atypical (always cloudy, near shore)
  3. Sample size: CV has more statistical power
  4. Physical mismatch: Neither method "should" match buoy perfectly

RECOMMENDED ACTIONS:
  1. Compare at observation_cropped to see if methods preserve satellite values
  2. Check if DINCAE is systematically smoother
  3. Investigate top DINCAE-winning lakes manually
  4. Check cloud/gap frequency at buoy pixels

================================================================================
"""
    
    print(summary)
    
    with open(os.path.join(output_dir, 'diagnostic_summary.txt'), 'w') as f:
        f.write(summary)
    print(f"Saved: diagnostic_summary.txt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostic: CV vs In-Situ discrepancy")
    parser.add_argument("--run_root", required=True, help="Experiment run root")
    parser.add_argument("--analysis_dir", default=None, help="insitu_validation_analysis folder")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--lake_metadata", default=None, help="Lake metadata CSV")
    args = parser.parse_args()
    
    if args.analysis_dir is None:
        args.analysis_dir = os.path.join(args.run_root, "insitu_validation_analysis")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "diagnostics")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("DIAGNOSTIC: CV vs In-Situ Validation Discrepancy")
    print("="*70)
    
    # Load data
    df = load_insitu_stats(args.analysis_dir)
    print(f"Loaded {len(df)} records from {df['lake_id_cci'].nunique()} lakes")
    
    # Load metadata
    metadata_paths = [
        args.lake_metadata,
        '/home/users/shaerdan/lake_cci_gapfilling/src/processors/data/globolakes-static_lake_centre_fv1.csv'
    ]
    lake_meta = load_lake_metadata(metadata_paths)
    if lake_meta is not None:
        print(f"Loaded metadata for {len(lake_meta)} lakes")
    
    # Classify winners
    winner_df = classify_winners(df)
    
    # Run analyses
    merged_geo = analyze_geographic_patterns(winner_df, lake_meta, args.output_dir)
    analyze_observation_quality_patterns(winner_df, args.output_dir)
    analyze_method_smoothness(df, winner_df, args.output_dir)
    create_comprehensive_comparison(df, winner_df, args.output_dir)
    list_dincae_winning_lakes(winner_df, merged_geo, args.output_dir)
    generate_summary(winner_df, args.output_dir)
    
    # Save winner classification
    winner_df.to_csv(os.path.join(args.output_dir, 'lake_winner_classification.csv'), index=False)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
