#!/usr/bin/env python3
"""
Direct CSV check: Does observation_cropped == reconstruction_observed per method in each file?

No fancy analysis - just read CSVs and compare counts.
"""

import os
import pandas as pd
from glob import glob

# ============================================================================
# Configuration
# ============================================================================
RUN_ROOT = "/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/archive/anomaly-20251215-8ea02d-exp3"
ALPHA_SLUG = "a1000"


def main():
    # Find all CSV files
    pattern = os.path.join(
        RUN_ROOT, "post", "*", ALPHA_SLUG, "insitu_cv_validation",
        "*_insitu_stats_site*.csv"
    )
    csv_files = sorted(glob(pattern))
    
    print(f"Found {len(csv_files)} CSV files")
    print("="*80)
    
    # Check for duplicate filenames (old vs new naming)
    basenames = [os.path.basename(f) for f in csv_files]
    print(f"Unique basenames: {len(set(basenames))}")
    if len(basenames) != len(set(basenames)):
        print("⚠️  DUPLICATE BASENAMES DETECTED!")
        from collections import Counter
        dupes = [k for k, v in Counter(basenames).items() if v > 1]
        for d in dupes[:10]:
            print(f"   {d}")
    print()
    
    # Track mismatches
    mismatches = []
    matches = []
    errors = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Get unique methods in this file
            methods = df['method'].unique()
            
            for method in methods:
                method_df = df[df['method'] == method]
                
                # Get observation_cropped row
                obs_crop = method_df[method_df['data_type'] == 'observation_cropped']
                rec_obs = method_df[method_df['data_type'] == 'reconstruction_observed']
                
                # Skip if either is missing
                if obs_crop.empty or rec_obs.empty:
                    continue
                
                n_obs_crop = obs_crop['n_matches'].values[0]
                n_rec_obs = rec_obs['n_matches'].values[0]
                
                basename = os.path.basename(csv_file)
                
                if n_obs_crop == n_rec_obs:
                    matches.append({
                        'file': basename,
                        'method': method,
                        'n_obs_cropped': n_obs_crop,
                        'n_recon_observed': n_rec_obs,
                    })
                else:
                    mismatches.append({
                        'file': basename,
                        'method': method,
                        'n_obs_cropped': n_obs_crop,
                        'n_recon_observed': n_rec_obs,
                        'diff': n_rec_obs - n_obs_crop,
                    })
        
        except Exception as e:
            errors.append({'file': os.path.basename(csv_file), 'error': str(e)})
    
    # Report
    print(f"RESULTS:")
    print(f"  Matches (obs_crop == rec_obs): {len(matches)}")
    print(f"  Mismatches: {len(mismatches)}")
    print(f"  Errors: {len(errors)}")
    print()
    
    if mismatches:
        print("="*80)
        print("MISMATCHES:")
        print("="*80)
        for m in mismatches[:20]:  # First 20
            print(f"  {m['file']} | {m['method']}: obs_crop={m['n_obs_cropped']}, rec_obs={m['n_recon_observed']}, diff={m['diff']}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print("✓ All CSV files have matching counts for obs_cropped and recon_observed!")
    
    if errors:
        print()
        print("ERRORS:")
        for e in errors[:10]:
            print(f"  {e['file']}: {e['error']}")


if __name__ == "__main__":
    main()
