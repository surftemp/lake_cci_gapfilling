#!/usr/bin/env python3
"""
Cross-reference: Does damping predict which method wins?

If hypothesis is correct:
- Lakes where insitu is NOT damped -> DINEOF should still dominate
- Lakes where insitu IS damped -> DINCAE should catch up
"""

import pandas as pd
import numpy as np
import sys
import os

# Paths - adjust as needed
EXP_ROOT = os.environ.get('EXP_ROOT', '/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/archive/anomaly-20251215-8ea02d-exp3')

amplitude_csv = f"{EXP_ROOT}/insitu_vs_satellite_comparison/insitu_vs_satellite_data.csv"
alpha = "a1000"

# Load amplitude data
amp_df = pd.read_csv(amplitude_csv)
print(f"Loaded amplitude data for {len(amp_df)} lakes")

# Compute damping flag
amp_df['insitu_damped'] = amp_df['insitu_std'] < amp_df['satellite_std']

# Load winner data from validation CSVs
from glob import glob

winners = {}
post_dir = os.path.join(EXP_ROOT, "post")

for lake_folder in os.listdir(post_dir):
    lake_path = os.path.join(post_dir, lake_folder, alpha, "insitu_cv_validation")
    if not os.path.isdir(lake_path):
        continue
    
    csv_files = glob(os.path.join(lake_path, "*_insitu_stats_*.csv"))
    if not csv_files:
        continue
    
    try:
        df = pd.read_csv(csv_files[0])
        
        # Find RMSE for dineof and dincae reconstruction
        dineof_row = df[(df['data_type'] == 'reconstruction') & (df['method'].str.contains('dineof', case=False, na=False))]
        dincae_row = df[(df['data_type'] == 'reconstruction') & (df['method'].str.contains('dincae', case=False, na=False))]
        
        if not dineof_row.empty and not dincae_row.empty:
            dineof_rmse = dineof_row['rmse'].values[0]
            dincae_rmse = dincae_row['rmse'].values[0]
            
            lake_id = int(lake_folder.lstrip('0') or '0')
            winners[lake_id] = {
                'winner': 'dineof' if dineof_rmse < dincae_rmse else 'dincae',
                'dineof_rmse': dineof_rmse,
                'dincae_rmse': dincae_rmse,
            }
    except Exception as e:
        continue

print(f"Found winner data for {len(winners)} lakes")

# Merge
amp_df['winner'] = amp_df['lake_id'].map(lambda x: winners.get(x, {}).get('winner'))
amp_df['dineof_rmse'] = amp_df['lake_id'].map(lambda x: winners.get(x, {}).get('dineof_rmse'))
amp_df['dincae_rmse'] = amp_df['lake_id'].map(lambda x: winners.get(x, {}).get('dincae_rmse'))

# Filter to lakes with winner data
merged = amp_df[amp_df['winner'].notna()].copy()
print(f"Lakes with both amplitude and winner data: {len(merged)}")

print("\n" + "="*70)
print("CROSS-REFERENCE: DAMPING vs WINNER")
print("="*70)

# Split by damping
damped = merged[merged['insitu_damped']]
not_damped = merged[~merged['insitu_damped']]

print(f"\nLakes where in-situ IS damped (insitu_std < satellite_std): {len(damped)}")
if len(damped) > 0:
    dineof_wins_damped = (damped['winner'] == 'dineof').sum()
    dincae_wins_damped = (damped['winner'] == 'dincae').sum()
    print(f"  DINEOF wins: {dineof_wins_damped}/{len(damped)} ({100*dineof_wins_damped/len(damped):.0f}%)")
    print(f"  DINCAE wins: {dincae_wins_damped}/{len(damped)} ({100*dincae_wins_damped/len(damped):.0f}%)")

print(f"\nLakes where in-situ is NOT damped (insitu_std >= satellite_std): {len(not_damped)}")
if len(not_damped) > 0:
    dineof_wins_not = (not_damped['winner'] == 'dineof').sum()
    dincae_wins_not = (not_damped['winner'] == 'dincae').sum()
    print(f"  DINEOF wins: {dineof_wins_not}/{len(not_damped)} ({100*dineof_wins_not/len(not_damped):.0f}%)")
    print(f"  DINCAE wins: {dincae_wins_not}/{len(not_damped)} ({100*dincae_wins_not/len(not_damped):.0f}%)")

print("\n" + "-"*70)
print("HYPOTHESIS TEST")
print("-"*70)
print("\nIf hypothesis is correct:")
print("  - Damped lakes: DINCAE should catch up (win more)")
print("  - Not damped lakes: DINEOF should dominate (win most)")

if len(damped) > 0 and len(not_damped) > 0:
    dineof_pct_damped = 100 * dineof_wins_damped / len(damped)
    dineof_pct_not_damped = 100 * dineof_wins_not / len(not_damped)
    
    print(f"\nActual results:")
    print(f"  DINEOF win rate in damped lakes:     {dineof_pct_damped:.0f}%")
    print(f"  DINEOF win rate in NOT damped lakes: {dineof_pct_not_damped:.0f}%")
    
    if dineof_pct_not_damped > dineof_pct_damped:
        print("\n  --> CONSISTENT with hypothesis!")
        print("      DINEOF wins more when in-situ is not damped")
    else:
        print("\n  --> NOT consistent with hypothesis")
        print("      Damping does not predict winner")

# Show individual lakes
print("\n" + "-"*70)
print("INDIVIDUAL LAKES (NOT DAMPED)")
print("-"*70)
if len(not_damped) > 0:
    for _, row in not_damped.iterrows():
        print(f"  Lake {int(row['lake_id'])}: insitu_std={row['insitu_std']:.2f}, sat_std={row['satellite_std']:.2f}, winner={row['winner']}")

print("\n" + "="*70)
