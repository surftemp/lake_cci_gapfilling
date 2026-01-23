#!/usr/bin/env python3
"""
Diagnostic script to investigate data inconsistencies in in-situ validation.

Issues to investigate:
1. Why do DINEOF and DINCAE have different lake counts?
2. Why doesn't observation_cropped match reconstruction_observed point counts?

Then: Fix aggregation to only include lakes with BOTH methods.
"""

import os
import sys
import pandas as pd
import numpy as np
from glob import glob

def diagnose_inconsistencies(analysis_dir: str):
    """Diagnose the data inconsistencies."""
    
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print("DIAGNOSTIC: DATA INCONSISTENCIES IN IN-SITU VALIDATION")
    print("="*80)
    
    # ----- Issue 1: Different lake counts per method -----
    print("\n" + "-"*60)
    print("ISSUE 1: Lake counts per method")
    print("-"*60)
    
    lakes_per_method = df.groupby('method')['lake_id_cci'].nunique()
    print(f"\nLakes per method:")
    for method, count in lakes_per_method.items():
        print(f"  {method}: {count} lakes")
    
    # Find which lakes are missing from each method
    all_lakes = set(df['lake_id_cci'].unique())
    
    for method in df['method'].unique():
        method_lakes = set(df[df['method'] == method]['lake_id_cci'].unique())
        missing = all_lakes - method_lakes
        if missing:
            print(f"\n  Lakes MISSING from {method}: {sorted(missing)}")
    
    # Find common lakes
    methods = df['method'].unique()
    if len(methods) >= 2:
        lakes_method1 = set(df[df['method'] == methods[0]]['lake_id_cci'].unique())
        lakes_method2 = set(df[df['method'] == methods[1]]['lake_id_cci'].unique())
        common_lakes = lakes_method1 & lakes_method2
        print(f"\n  Lakes with BOTH methods: {len(common_lakes)}")
        print(f"  Lakes with only {methods[0]}: {sorted(lakes_method1 - common_lakes)}")
        print(f"  Lakes with only {methods[1]}: {sorted(lakes_method2 - common_lakes)}")
    
    # ----- Issue 2: observation_cropped vs reconstruction_observed -----
    print("\n" + "-"*60)
    print("ISSUE 2: observation_cropped vs reconstruction_observed point counts")
    print("-"*60)
    
    # Per-lake comparison
    for method in df['method'].unique():
        print(f"\n{method.upper()}:")
        
        obs_crop = df[(df['method'] == method) & (df['data_type'] == 'observation_cropped')]
        rec_obs = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_observed')]
        
        if obs_crop.empty or rec_obs.empty:
            print("  One of the data types is missing!")
            continue
        
        # Merge on lake_id and site_id
        merged = obs_crop[['lake_id_cci', 'site_id', 'n_matches']].merge(
            rec_obs[['lake_id_cci', 'site_id', 'n_matches']],
            on=['lake_id_cci', 'site_id'],
            suffixes=('_obs_crop', '_rec_obs'),
            how='outer'
        )
        
        # Check for mismatches
        merged['match'] = merged['n_matches_obs_crop'] == merged['n_matches_rec_obs']
        merged['diff'] = merged['n_matches_rec_obs'] - merged['n_matches_obs_crop']
        
        mismatches = merged[~merged['match']]
        if len(mismatches) > 0:
            print(f"  MISMATCHES found in {len(mismatches)} lake-site combinations:")
            for _, row in mismatches.head(10).iterrows():
                print(f"    Lake {int(row['lake_id_cci'])}, Site {row['site_id']}: "
                      f"obs_crop={row['n_matches_obs_crop']}, rec_obs={row['n_matches_rec_obs']}, "
                      f"diff={row['diff']}")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches) - 10} more")
        else:
            print("  All point counts match!")
        
        # Summary stats
        total_obs_crop = merged['n_matches_obs_crop'].sum()
        total_rec_obs = merged['n_matches_rec_obs'].sum()
        print(f"  Total obs_cropped: {total_obs_crop}")
        print(f"  Total rec_observed: {total_rec_obs}")
        print(f"  Ratio rec_obs/obs_crop: {total_rec_obs/total_obs_crop:.2f}")
    
    # ----- Hypothesis: Is rec_observed counting spatial pixels, not temporal matches? -----
    print("\n" + "-"*60)
    print("HYPOTHESIS CHECK: Temporal vs Spatial counting")
    print("-"*60)
    
    # Check if the ratio is consistent (suggesting a systematic issue)
    for method in df['method'].unique():
        obs_crop = df[(df['method'] == method) & (df['data_type'] == 'observation_cropped')]
        rec_obs = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_observed')]
        rec_miss = df[(df['method'] == method) & (df['data_type'] == 'reconstruction_missing')]
        rec_all = df[(df['method'] == method) & (df['data_type'] == 'reconstruction')]
        
        if not obs_crop.empty and not rec_obs.empty:
            total_obs_crop = obs_crop['n_matches'].sum()
            total_rec_obs = rec_obs['n_matches'].sum()
            total_rec_miss = rec_miss['n_matches'].sum() if not rec_miss.empty else 0
            total_rec_all = rec_all['n_matches'].sum() if not rec_all.empty else 0
            
            print(f"\n{method.upper()}:")
            print(f"  observation_cropped:      {total_obs_crop:>8}")
            print(f"  reconstruction_observed:  {total_rec_obs:>8}")
            print(f"  reconstruction_missing:   {total_rec_miss:>8}")
            print(f"  reconstruction (all):     {total_rec_all:>8}")
            print(f"  obs + miss =              {total_rec_obs + total_rec_miss:>8} (should = reconstruction)")
            
            # Check the relationship
            if total_rec_all > 0:
                check = total_rec_obs + total_rec_miss
                if abs(check - total_rec_all) < 10:
                    print(f"  ✓ rec_observed + rec_missing ≈ reconstruction (diff={total_rec_all - check})")
                else:
                    print(f"  ✗ MISMATCH: rec_observed + rec_missing ≠ reconstruction")


def create_filtered_aggregation(analysis_dir: str, output_dir: str = None):
    """
    Create aggregated statistics ONLY for lakes with BOTH methods.
    
    This ensures fair comparison between DINEOF and DINCAE.
    """
    if output_dir is None:
        output_dir = analysis_dir
    
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print("CREATING FILTERED AGGREGATION (BOTH METHODS REQUIRED)")
    print("="*80)
    
    # Find lakes with BOTH dineof AND dincae
    dineof_lakes = set(df[df['method'] == 'dineof']['lake_id_cci'].unique())
    dincae_lakes = set(df[df['method'] == 'dincae']['lake_id_cci'].unique())
    common_lakes = dineof_lakes & dincae_lakes
    
    print(f"\nOriginal counts:")
    print(f"  DINEOF lakes: {len(dineof_lakes)}")
    print(f"  DINCAE lakes: {len(dincae_lakes)}")
    print(f"  Common lakes: {len(common_lakes)}")
    
    excluded_lakes = (dineof_lakes | dincae_lakes) - common_lakes
    if excluded_lakes:
        print(f"\nExcluded lakes (missing one method): {sorted(excluded_lakes)}")
    
    # Filter to common lakes only
    df_filtered = df[df['lake_id_cci'].isin(common_lakes)].copy()
    
    print(f"\nFiltered dataset: {len(df_filtered)} records from {len(common_lakes)} lakes")
    
    # Save filtered combined data
    filtered_path = os.path.join(output_dir, 'all_insitu_stats_combined_filtered.csv')
    df_filtered.to_csv(filtered_path, index=False)
    print(f"Saved: {filtered_path}")
    
    # Compute filtered aggregate statistics
    print("\nFiltered aggregate statistics:")
    print("-"*60)
    
    CORE_METRICS = ['rmse', 'mae', 'median', 'bias', 'std', 'rstd']
    
    agg_stats = []
    for method in ['dineof', 'dincae']:
        for data_type in df_filtered['data_type'].unique():
            subset = df_filtered[(df_filtered['method'] == method) & 
                                  (df_filtered['data_type'] == data_type)]
            if subset.empty:
                continue
            
            row = {
                'method': method,
                'data_type': data_type,
                'n_lakes': subset['lake_id_cci'].nunique(),
                'n_total_points': subset['n_matches'].sum(),
            }
            
            # Weighted averages for all metrics
            weights = subset['n_matches'].values
            for metric in CORE_METRICS:
                if metric in subset.columns:
                    valid = subset[~subset[metric].isna()]
                    if len(valid) > 0:
                        w = valid['n_matches'].values
                        if w.sum() > 0:
                            row[f'{metric}_weighted'] = np.average(valid[metric].values, weights=w)
                        row[f'{metric}_mean'] = valid[metric].mean()
            
            agg_stats.append(row)
    
    agg_df = pd.DataFrame(agg_stats)
    
    # Print summary
    print(f"\n{'method':<10} {'data_type':<25} {'n_lakes':>8} {'n_points':>12} {'rmse':>8} {'mae':>8}")
    print("-"*80)
    for _, row in agg_df.iterrows():
        rmse = row.get('rmse_weighted', np.nan)
        mae = row.get('mae_weighted', np.nan)
        print(f"{row['method']:<10} {row['data_type']:<25} {row['n_lakes']:>8} "
              f"{row['n_total_points']:>12} {rmse:>8.3f} {mae:>8.3f}")
    
    # Save filtered aggregates
    agg_path = os.path.join(output_dir, 'aggregate_statistics_filtered.csv')
    agg_df.to_csv(agg_path, index=False)
    print(f"\nSaved: {agg_path}")
    
    # Now check if filtered data has consistent observation_cropped vs reconstruction_observed
    print("\n" + "-"*60)
    print("CHECKING FILTERED DATA CONSISTENCY")
    print("-"*60)
    
    for method in ['dineof', 'dincae']:
        obs_crop = df_filtered[(df_filtered['method'] == method) & 
                                (df_filtered['data_type'] == 'observation_cropped')]
        rec_obs = df_filtered[(df_filtered['method'] == method) & 
                               (df_filtered['data_type'] == 'reconstruction_observed')]
        
        if not obs_crop.empty and not rec_obs.empty:
            total_obs_crop = obs_crop['n_matches'].sum()
            total_rec_obs = rec_obs['n_matches'].sum()
            print(f"\n{method.upper()} (filtered to {len(common_lakes)} common lakes):")
            print(f"  observation_cropped:      {total_obs_crop:>8}")
            print(f"  reconstruction_observed:  {total_rec_obs:>8}")
            
            if total_obs_crop != total_rec_obs:
                print(f"  ⚠ STILL MISMATCHED! Ratio: {total_rec_obs/total_obs_crop:.2f}")
                print(f"  → This suggests a BUG in insitu_validation.py logic")
    
    return df_filtered, agg_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose in-situ validation inconsistencies")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as analysis_dir)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.analysis_dir
    
    # Run diagnostics
    diagnose_inconsistencies(args.analysis_dir)
    
    # Create filtered aggregation
    df_filtered, agg_df = create_filtered_aggregation(args.analysis_dir, args.output_dir)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
1. Use the FILTERED aggregation files for all comparisons:
   - all_insitu_stats_combined_filtered.csv
   - aggregate_statistics_filtered.csv

2. The observation_cropped vs reconstruction_observed mismatch needs 
   investigation in insitu_validation.py - likely a bug in how temporal
   alignment is done or how n_matches is counted.

3. Possible causes of the mismatch:
   - observation_cropped might be filtering by observation STATUS in prepared.nc
   - reconstruction_observed might be using a different time alignment
   - The "cropped" vs "observed" logic may have different definitions
""")


if __name__ == "__main__":
    main()
