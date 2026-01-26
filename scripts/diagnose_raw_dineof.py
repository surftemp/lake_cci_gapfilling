#!/usr/bin/env python3
"""
Diagnose the RAW dineof_results.nc file before any post-processing.

Usage:
    python diagnose_raw_dineof.py --lake-id 310 --run-root /path/to/experiment
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Diagnose raw DINEOF output")
    parser.add_argument("--run-root", required=False, 
                        default="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260125-4d8546-exp0_dineof_noretrain",
                        help="Path to experiment run root")
    parser.add_argument("--lake-id", type=int, required=True, help="Lake ID (e.g., 310)")
    parser.add_argument("--alpha", type=int, default=1000, help="Alpha value")
    args = parser.parse_args()
    
    run_root = Path(args.run_root)
    lake_id9 = f"{args.lake_id:09d}"
    alpha_str = f"a{args.alpha}"
    
    # Paths
    dineof_results = run_root / "dineof" / lake_id9 / alpha_str / "dineof_results.nc"
    prepared = run_root / "prepared" / lake_id9 / "prepared.nc"
    post_output = list((run_root / "post" / lake_id9 / alpha_str).glob("*_dineof.nc"))
    
    print("=" * 70)
    print("RAW DINEOF OUTPUT (dineof_results.nc)")
    print("=" * 70)
    
    if not dineof_results.exists():
        print(f"ERROR: {dineof_results} not found")
        return
    
    ds = xr.open_dataset(dineof_results)
    print(f"\nFile: {dineof_results}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Dimensions: {dict(ds.sizes)}")
    
    if "temp_filled" in ds:
        tf = ds["temp_filled"].values
        print(f"\n>>> temp_filled:")
        print(f"  Shape: {tf.shape}")
        print(f"  Dtype: {tf.dtype}")
        
        n_total = tf.size
        n_valid = int(np.sum(~np.isnan(tf)))
        n_nan = int(np.sum(np.isnan(tf)))
        
        print(f"\n  Total pixels: {n_total:,}")
        print(f"  Valid (not NaN): {n_valid:,} ({100*n_valid/n_total:.2f}%)")
        print(f"  NaN: {n_nan:,} ({100*n_nan/n_total:.2f}%)")
        
        # Per-timestep
        n_time = tf.shape[0]
        valid_per_time = np.sum(~np.isnan(tf), axis=(1, 2))
        unique_counts = np.unique(valid_per_time)
        print(f"\n  Timesteps: {n_time}")
        print(f"  Valid pixels per timestep: min={valid_per_time.min()}, max={valid_per_time.max()}")
        print(f"  Unique counts: {unique_counts[:10]}{'...' if len(unique_counts) > 10 else ''}")
    
    ds.close()
    
    # Compare with prepared.nc
    print("\n" + "=" * 70)
    print("PREPARED.NC (for comparison)")
    print("=" * 70)
    
    if prepared.exists():
        ds_prep = xr.open_dataset(prepared)
        print(f"\nFile: {prepared}")
        print(f"Dimensions: {dict(ds_prep.sizes)}")
        
        lswt_var = None
        for var in ["lake_surface_water_temperature", "lswt", "temp", "sst"]:
            if var in ds_prep:
                lswt_var = var
                break
        
        if lswt_var:
            prep_data = ds_prep[lswt_var].values
            print(f"\n>>> {lswt_var}:")
            print(f"  Shape: {prep_data.shape}")
            
            n_total = prep_data.size
            n_valid = int(np.sum(~np.isnan(prep_data)))
            n_nan = int(np.sum(np.isnan(prep_data)))
            
            print(f"\n  Total pixels: {n_total:,}")
            print(f"  Valid (observations): {n_valid:,}")
            print(f"  NaN (missing): {n_nan:,}")
            
            # Per-timestep
            valid_per_time = np.sum(~np.isnan(prep_data), axis=(1, 2))
            unique_counts = np.unique(valid_per_time)
            print(f"\n  Valid pixels per timestep: min={valid_per_time.min()}, max={valid_per_time.max()}")
            print(f"  Unique counts: {unique_counts[:10]}{'...' if len(unique_counts) > 10 else ''}")
        
        ds_prep.close()
    
    # Compare with post-processed output
    if post_output:
        print("\n" + "=" * 70)
        print("POST-PROCESSED OUTPUT (*_dineof.nc)")
        print("=" * 70)
        
        ds_post = xr.open_dataset(post_output[0])
        print(f"\nFile: {post_output[0]}")
        print(f"Dimensions: {dict(ds_post.sizes)}")
        
        if "temp_filled" in ds_post:
            tf_post = ds_post["temp_filled"].values
            print(f"\n>>> temp_filled:")
            print(f"  Shape: {tf_post.shape}")
            
            n_total = tf_post.size
            n_valid = int(np.sum(~np.isnan(tf_post)))
            n_nan = int(np.sum(np.isnan(tf_post)))
            
            print(f"\n  Total pixels: {n_total:,}")
            print(f"  Valid (not NaN): {n_valid:,}")
            print(f"  NaN: {n_nan:,}")
            
            # Per-timestep
            valid_per_time = np.sum(~np.isnan(tf_post), axis=(1, 2))
            nonzero_times = int(np.sum(valid_per_time > 0))
            unique_counts = np.unique(valid_per_time[valid_per_time > 0])
            print(f"\n  Timesteps with data: {nonzero_times}")
            print(f"  Valid pixels per timestep (non-zero only): min={unique_counts.min()}, max={unique_counts.max()}")
            print(f"  Unique counts: {unique_counts[:10]}{'...' if len(unique_counts) > 10 else ''}")
        
        ds_post.close()
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print("""
    DINEOF reports: 1,563,580 missing out of 3,446,625 (45.37%)
    
    So expected:
    - Total reconstructed pixels: 3,446,625
    - Observations: 3,446,625 - 1,563,580 = 1,883,045
    - Missing (gaps): 1,563,580
    
    Check if the raw dineof_results.nc has 3,446,625 valid pixels.
    If yes, the issue is in post-processing.
    If no, the issue is earlier.
    """)


if __name__ == "__main__":
    main()
