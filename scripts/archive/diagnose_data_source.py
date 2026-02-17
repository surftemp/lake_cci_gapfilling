#!/usr/bin/env python3
"""
Simple diagnostic to check the ground truth in prepared.nc and temp_filled.

Run:
    python diagnose_data_source.py --lake-id 310 --run-root /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260125-4d8546-exp0_dineof_noretrain
    
Or with explicit paths:
    python diagnose_data_source.py --prepared /path/to/prepared.nc --output /path/to/*_dineof.nc
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import glob


def find_files(run_root: str, lake_id: int):
    """Auto-discover file paths from run_root and lake_id."""
    run_root = Path(run_root)
    lake_id9 = f"{lake_id:09d}"
    
    files = {}
    
    # Prepared
    prep_dir = run_root / "prepared" / lake_id9
    files["prepared"] = prep_dir / "prepared.nc"
    files["clouds_index"] = prep_dir / "clouds_index.nc"
    
    # Output - search for *_dineof.nc pattern
    post_dir = run_root / "post" / lake_id9
    dineof_files = list(post_dir.glob("*/*_dineof.nc"))  # includes alpha subdir
    if not dineof_files:
        dineof_files = list(post_dir.glob("*_dineof.nc"))
    if dineof_files:
        files["output"] = dineof_files[0]
    
    return files


def main():
    parser = argparse.ArgumentParser(description="Diagnose data sources")
    parser.add_argument("--run-root", required=False, 
                        default="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260125-4d8546-exp0_dineof_noretrain",
                        help="Path to experiment run root")
    parser.add_argument("--lake-id", type=int, required=False, help="Lake ID (e.g., 310)")
    parser.add_argument("--prepared", required=False, help="Path to prepared.nc (overrides auto-discovery)")
    parser.add_argument("--output", required=False, help="Path to output (e.g., *_dineof.nc)")
    parser.add_argument("--clouds-index", required=False, help="Path to clouds_index.nc")
    args = parser.parse_args()
    
    # Auto-discover paths if lake-id provided
    if args.lake_id:
        files = find_files(args.run_root, args.lake_id)
        if not args.prepared and files.get("prepared") and files["prepared"].exists():
            args.prepared = str(files["prepared"])
        if not args.output and files.get("output") and files["output"].exists():
            args.output = str(files["output"])
        if not args.clouds_index and files.get("clouds_index") and files["clouds_index"].exists():
            args.clouds_index = str(files["clouds_index"])
    
    if not args.prepared:
        print("ERROR: Must provide --prepared or --lake-id")
        return
    
    print("=" * 70)
    print("PREPARED.NC DIAGNOSTICS")
    print("=" * 70)
    
    ds_prep = xr.open_dataset(args.prepared)
    print(f"\nFile: {args.prepared}")
    print(f"Variables: {list(ds_prep.data_vars)}")
    print(f"Coordinates: {list(ds_prep.coords)}")
    print(f"Dimensions: {dict(ds_prep.dims)}")
    
    # Find LSWT variable
    lswt_var = None
    for var in ["lake_surface_water_temperature", "lswt", "temp", "sst"]:
        if var in ds_prep:
            lswt_var = var
            break
    
    if lswt_var:
        data = ds_prep[lswt_var].values
        print(f"\n>>> Using variable: '{lswt_var}'")
        print(f"  Shape: {data.shape}  (time, lat, lon)")
        print(f"  Dtype: {data.dtype}")
        
        n_total = data.size
        n_valid = int(np.sum(~np.isnan(data)))
        n_nan = int(np.sum(np.isnan(data)))
        
        print(f"\n  Total pixels (time x lat x lon): {n_total:,}")
        print(f"  Valid (not NaN) = OBSERVATIONS: {n_valid:,} ({100*n_valid/n_total:.2f}%)")
        print(f"  NaN = MISSING before gapfill: {n_nan:,} ({100*n_nan/n_total:.2f}%)")
        
        # Per-timestep stats
        n_time = data.shape[0]
        valid_per_time = np.sum(~np.isnan(data), axis=(1, 2))
        print(f"\n  Timesteps: {n_time}")
        print(f"  Valid pixels per timestep: min={valid_per_time.min()}, max={valid_per_time.max()}, mean={valid_per_time.mean():.1f}")
    else:
        print("\n  WARNING: Could not find LSWT variable in list: lake_surface_water_temperature, lswt, temp, sst")
        print(f"  Available data variables: {list(ds_prep.data_vars)}")
    
    ds_prep.close()
    
    # Check clouds_index.nc if provided
    if args.clouds_index:
        print("\n" + "=" * 70)
        print("CLOUDS_INDEX.NC DIAGNOSTICS (CV points)")
        print("=" * 70)
        
        ds_cv = xr.open_dataset(args.clouds_index)
        print(f"\nFile: {args.clouds_index}")
        print(f"Variables: {list(ds_cv.data_vars)}")
        print(f"Dimensions: {dict(ds_cv.dims)}")
        
        if "clouds_index" in ds_cv:
            ci = ds_cv["clouds_index"].values
            print(f"\nclouds_index shape: {ci.shape}")
            n_cv = ci.shape[0] if ci.shape[1] == 2 else ci.shape[1]
            print(f"  >>> Number of CV points: {n_cv:,}")
        
        ds_cv.close()
    
    # Check output file if provided
    if args.output:
        print("\n" + "=" * 70)
        print("OUTPUT FILE DIAGNOSTICS (post-processed)")
        print("=" * 70)
        
        ds_out = xr.open_dataset(args.output)
        print(f"\nFile: {args.output}")
        print(f"Variables: {list(ds_out.data_vars)}")
        print(f"Coordinates: {list(ds_out.coords)}")
        print(f"Dimensions: {dict(ds_out.dims)}")
        
        if "temp_filled" in ds_out:
            tf = ds_out["temp_filled"].values
            print(f"\n>>> Variable: 'temp_filled' (reconstruction)")
            print(f"  Shape: {tf.shape}  (time, lat, lon)")
            print(f"  Dtype: {tf.dtype}")
            
            n_total = tf.size
            n_valid = int(np.sum(~np.isnan(tf)))
            n_nan = int(np.sum(np.isnan(tf)))
            
            print(f"\n  Total pixels (time x lat x lon): {n_total:,}")
            print(f"  Valid (not NaN) = RECONSTRUCTED: {n_valid:,} ({100*n_valid/n_total:.2f}%)")
            print(f"  NaN = NOT RECONSTRUCTED: {n_nan:,} ({100*n_nan/n_total:.2f}%)")
            
            # Per-timestep stats
            n_time = tf.shape[0]
            valid_per_time = np.sum(~np.isnan(tf), axis=(1, 2))
            nonzero_times = int(np.sum(valid_per_time > 0))
            print(f"\n  Total timesteps in file: {n_time}")
            print(f"  Timesteps with ANY reconstructed data: {nonzero_times}")
            if nonzero_times > 0:
                # Stats only over timesteps with data
                valid_per_time_nonzero = valid_per_time[valid_per_time > 0]
                print(f"  Valid pixels per reconstructed timestep: min={valid_per_time_nonzero.min()}, max={valid_per_time_nonzero.max()}, mean={valid_per_time_nonzero.mean():.1f}")
        else:
            print("\n  WARNING: 'temp_filled' not found in output file")
            print(f"  Available data variables: {list(ds_out.data_vars)}")
        
        if "data_source" in ds_out:
            ds_flag = ds_out["data_source"].values
            print(f"\n>>> Variable: 'data_source' (flag)")
            print(f"  Shape: {ds_flag.shape}")
            total = ds_flag.size
            for val, meaning in [(0, "true_gap"), (1, "observed_seen"), (2, "cv_withheld"), (255, "not_reconstructed")]:
                count = int(np.sum(ds_flag == val))
                pct = 100 * count / total
                print(f"  Value {val:3d} ({meaning:16s}): {count:>15,} ({pct:6.2f}%)")
        
        ds_out.close()
    
    # Cross-check if both prepared and output provided
    if args.output and lswt_var:
        print("\n" + "=" * 70)
        print("CROSS-CHECK: prepared.nc vs temp_filled")
        print("=" * 70)
        
        ds_prep = xr.open_dataset(args.prepared)
        ds_out = xr.open_dataset(args.output)
        
        prep_data = ds_prep[lswt_var].values
        
        prep_valid = int(np.sum(~np.isnan(prep_data)))
        prep_nan = int(np.sum(np.isnan(prep_data)))
        
        print(f"\n  prepared.nc:")
        print(f"    Shape: {prep_data.shape}")
        print(f"    Valid (observations): {prep_valid:,}")
        print(f"    NaN (missing): {prep_nan:,}")
        
        if "temp_filled" in ds_out:
            temp_filled = ds_out["temp_filled"].values
            tf_valid = int(np.sum(~np.isnan(temp_filled)))
            tf_nan = int(np.sum(np.isnan(temp_filled)))
            
            print(f"\n  temp_filled:")
            print(f"    Shape: {temp_filled.shape}")
            print(f"    Valid (reconstructed): {tf_valid:,}")
            print(f"    NaN (not reconstructed): {tf_nan:,}")
            
            print(f"\n  >>> EXPECTED for data_source flag:")
            print(f"    flag 0+1+2 (reconstructed): should be {tf_valid:,}")
            print(f"    flag 255 (not reconstructed): should be {tf_nan:,}")
            
            # If clouds_index was provided, we can estimate the breakdown
            if args.clouds_index:
                ds_cv = xr.open_dataset(args.clouds_index)
                ci = ds_cv["clouds_index"].values
                n_cv = ci.shape[0] if ci.shape[1] == 2 else ci.shape[1]
                ds_cv.close()
                
                # Within reconstructed pixels:
                # - CV points: n_cv (some may be outside reconstruction, but roughly)
                # - Observed/seen: prep_valid - n_cv (observations minus CV)
                # - True gaps: tf_valid - prep_valid (reconstructed minus observations)
                
                n_observed_seen_approx = prep_valid - n_cv
                n_true_gap_approx = tf_valid - prep_valid
                
                print(f"\n  >>> EXPECTED breakdown (approximate):")
                print(f"    flag 0 (true_gap): ~{n_true_gap_approx:,} = reconstructed - observations")
                print(f"    flag 1 (observed_seen): ~{n_observed_seen_approx:,} = observations - CV")
                print(f"    flag 2 (cv_withheld): ~{n_cv:,} = CV points")
        
        # Check time alignment
        prep_time = ds_prep["time"].values
        out_time = ds_out["time"].values
        
        print(f"\n  Time axis comparison:")
        print(f"    prepared.nc timesteps: {len(prep_time)}")
        print(f"    output timesteps: {len(out_time)}")
        
        ds_prep.close()
        ds_out.close()


if __name__ == "__main__":
    main()

