#!/usr/bin/env python3
"""
Diagnose the difference between DINEOF and DINCAE reconstruction coverage.

DINEOF: 3,446,625 reconstructed pixels
DINCAE: 3,474,905 reconstructed pixels
Difference: 28,280 extra pixels in DINCAE

Where are these extra pixels?

Usage:
    python diagnose_dincae_extra_pixels.py --lake-id 310
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Diagnose DINCAE extra pixels")
    parser.add_argument("--run-root", required=False, 
                        default="/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20260125-4d8546-exp0_dineof_noretrain",
                        help="Path to experiment run root")
    parser.add_argument("--lake-id", type=int, required=True, help="Lake ID (e.g., 310)")
    parser.add_argument("--alpha", type=int, default=1000, help="Alpha value")
    args = parser.parse_args()
    
    run_root = Path(args.run_root)
    lake_id9 = f"{args.lake_id:09d}"
    alpha_str = f"a{args.alpha}"
    
    # Find output files
    post_dir = run_root / "post" / lake_id9 / alpha_str
    dineof_file = list(post_dir.glob("*_dineof.nc"))[0]
    dincae_file = list(post_dir.glob("*_dincae.nc"))[0]
    
    # Also get raw outputs
    dineof_raw = run_root / "dineof" / lake_id9 / alpha_str / "dineof_results.nc"
    dincae_raw = run_root / "dincae" / lake_id9 / alpha_str / "dincae_results.nc"
    
    # Prepared file
    prepared = run_root / "prepared" / lake_id9 / "prepared.nc"
    
    print("=" * 70)
    print("COMPARING DINEOF vs DINCAE RECONSTRUCTION")
    print("=" * 70)
    
    # Load post-processed outputs
    ds_dineof = xr.open_dataset(dineof_file)
    ds_dincae = xr.open_dataset(dincae_file)
    
    tf_dineof = ds_dineof["temp_filled"].values
    tf_dincae = ds_dincae["temp_filled"].values
    
    print(f"\nPost-processed files:")
    print(f"  DINEOF: {dineof_file.name}")
    print(f"    Shape: {tf_dineof.shape}")
    print(f"    Valid pixels: {np.sum(~np.isnan(tf_dineof)):,}")
    print(f"  DINCAE: {dincae_file.name}")
    print(f"    Shape: {tf_dincae.shape}")
    print(f"    Valid pixels: {np.sum(~np.isnan(tf_dincae)):,}")
    
    # Create masks
    dineof_valid = ~np.isnan(tf_dineof)
    dincae_valid = ~np.isnan(tf_dincae)
    
    # Find differences
    both_valid = dineof_valid & dincae_valid
    only_dineof = dineof_valid & ~dincae_valid
    only_dincae = dincae_valid & ~dineof_valid
    
    print(f"\n  Overlap analysis:")
    print(f"    Both have value: {both_valid.sum():,}")
    print(f"    Only DINEOF: {only_dineof.sum():,}")
    print(f"    Only DINCAE: {only_dincae.sum():,}")
    
    ds_dineof.close()
    ds_dincae.close()
    
    # Analyze WHERE the extra DINCAE pixels are
    print("\n" + "=" * 70)
    print("WHERE ARE THE EXTRA DINCAE PIXELS?")
    print("=" * 70)
    
    # Per-timestep analysis
    n_time = tf_dincae.shape[0]
    only_dincae_per_time = only_dincae.sum(axis=(1, 2))
    only_dineof_per_time = only_dineof.sum(axis=(1, 2))
    
    timesteps_with_extra_dincae = np.where(only_dincae_per_time > 0)[0]
    timesteps_with_extra_dineof = np.where(only_dineof_per_time > 0)[0]
    
    print(f"\n  Timesteps with pixels only in DINCAE: {len(timesteps_with_extra_dincae)}")
    print(f"  Timesteps with pixels only in DINEOF: {len(timesteps_with_extra_dineof)}")
    
    if len(timesteps_with_extra_dincae) > 0:
        print(f"\n  Extra DINCAE pixels per timestep (first 20 with extras):")
        for t in timesteps_with_extra_dincae[:20]:
            print(f"    t={t}: {only_dincae_per_time[t]} extra pixels")
    
    # Spatial analysis - where on the grid?
    only_dincae_spatial = only_dincae.sum(axis=0)  # (lat, lon)
    
    print(f"\n  Spatial distribution of extra DINCAE pixels:")
    print(f"    Total extra: {only_dincae_spatial.sum():,}")
    print(f"    Grid positions with extras: {(only_dincae_spatial > 0).sum()}")
    
    # Find the specific (lat, lon) positions
    extra_positions = np.argwhere(only_dincae_spatial > 0)
    print(f"\n  Grid positions (lat_idx, lon_idx) with extra DINCAE pixels (first 20):")
    for pos in extra_positions[:20]:
        lat_idx, lon_idx = pos
        count = only_dincae_spatial[lat_idx, lon_idx]
        print(f"    ({lat_idx}, {lon_idx}): {count} timesteps")
    
    # Load lake mask from DINEOF output
    ds_dineof = xr.open_dataset(dineof_file)
    if "lakeid" in ds_dineof:
        lake_mask = ds_dineof["lakeid"].values
        print(f"\n  Lake mask analysis:")
        print(f"    Lake mask shape: {lake_mask.shape}")
        print(f"    Lake pixels (lakeid=1): {(lake_mask == 1).sum()}")
        
        # Check if extra pixels are inside or outside lake mask
        if lake_mask.ndim == 2:  # (lat, lon)
            extra_in_lake = only_dincae_spatial[lake_mask == 1].sum()
            extra_outside_lake = only_dincae_spatial[lake_mask != 1].sum()
            print(f"    Extra DINCAE pixels inside lake mask: {extra_in_lake:,}")
            print(f"    Extra DINCAE pixels outside lake mask: {extra_outside_lake:,}")
    ds_dineof.close()
    
    # Compare RAW outputs (before post-processing)
    print("\n" + "=" * 70)
    print("RAW OUTPUT COMPARISON (before post-processing)")
    print("=" * 70)
    
    if dineof_raw.exists() and dincae_raw.exists():
        ds_dineof_raw = xr.open_dataset(dineof_raw)
        ds_dincae_raw = xr.open_dataset(dincae_raw)
        
        tf_dineof_raw = ds_dineof_raw["temp_filled"].values
        tf_dincae_raw = ds_dincae_raw["temp_filled"].values
        
        print(f"\n  Raw DINEOF (dineof_results.nc):")
        print(f"    Shape: {tf_dineof_raw.shape}")
        print(f"    Valid pixels: {np.sum(~np.isnan(tf_dineof_raw)):,}")
        print(f"    Valid per timestep: min={np.sum(~np.isnan(tf_dineof_raw), axis=(1,2)).min()}, max={np.sum(~np.isnan(tf_dineof_raw), axis=(1,2)).max()}")
        
        print(f"\n  Raw DINCAE (dincae_results.nc):")
        print(f"    Shape: {tf_dincae_raw.shape}")
        print(f"    Valid pixels: {np.sum(~np.isnan(tf_dincae_raw)):,}")
        print(f"    Valid per timestep: min={np.sum(~np.isnan(tf_dincae_raw), axis=(1,2)).min()}, max={np.sum(~np.isnan(tf_dincae_raw), axis=(1,2)).max()}")
        
        # Check if shapes match
        if tf_dineof_raw.shape != tf_dincae_raw.shape:
            print(f"\n  WARNING: Raw output shapes differ!")
            print(f"    DINEOF: {tf_dineof_raw.shape}")
            print(f"    DINCAE: {tf_dincae_raw.shape}")
        
        ds_dineof_raw.close()
        ds_dincae_raw.close()
    else:
        print(f"\n  Raw files not found:")
        print(f"    DINEOF: {dineof_raw} exists={dineof_raw.exists()}")
        print(f"    DINCAE: {dincae_raw} exists={dincae_raw.exists()}")
    
    # Check DINCAE intermediate files
    print("\n" + "=" * 70)
    print("DINCAE INTERMEDIATE FILES")
    print("=" * 70)
    
    dincae_dir = run_root / "dincae" / lake_id9 / alpha_str
    print(f"\n  Directory: {dincae_dir}")
    
    # List files
    if dincae_dir.exists():
        files = list(dincae_dir.glob("*.nc"))
        print(f"  Files found: {len(files)}")
        for f in sorted(files):
            print(f"    {f.name}")
        
        # Check the cropped prepared files
        clean_file = dincae_dir / "prepared_datetime_cropped_clean.nc"
        cv_file = dincae_dir / "prepared_datetime_cropped_cv.nc"
        
        if clean_file.exists():
            ds_clean = xr.open_dataset(clean_file)
            print(f"\n  prepared_datetime_cropped_clean.nc:")
            print(f"    Dimensions: {dict(ds_clean.sizes)}")
            if "lake_surface_water_temperature" in ds_clean:
                data = ds_clean["lake_surface_water_temperature"].values
                print(f"    Valid pixels: {np.sum(~np.isnan(data)):,}")
            
            # Check crop offsets
            crop_i0 = ds_clean.attrs.get("crop_i0", "N/A")
            crop_j0 = ds_clean.attrs.get("crop_j0", "N/A")
            print(f"    Crop offsets: i0={crop_i0}, j0={crop_j0}")
            ds_clean.close()


if __name__ == "__main__":
    main()
