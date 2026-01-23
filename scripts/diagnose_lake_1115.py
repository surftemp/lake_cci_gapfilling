#!/usr/bin/env python3
"""
Diagnose Lake 1115: Why does obs_cropped have 20 fewer points than recon_observed?

Find the specific dates that are in recon_observed but not in obs_cropped.
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob

# ============================================================================
# Configuration for Lake 1115
# ============================================================================
RUN_ROOT = "/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/archive/anomaly-20251215-8ea02d-exp3"
LAKE_ID = "000001115"
ALPHA_SLUG = "a1000"

# Paths
PREPARED_PATH = os.path.join(RUN_ROOT, "prepared", LAKE_ID, "prepared.nc")
POST_DIR = os.path.join(RUN_ROOT, "post", LAKE_ID, ALPHA_SLUG)


def find_nearest_idx(lat_arr, lon_arr, target_lat, target_lon):
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    return np.unravel_index(np.argmin(distance), distance.shape)


def main():
    print("="*80)
    print(f"DIAGNOSING LAKE {LAKE_ID}")
    print("="*80)
    
    # Find output file
    output_files = glob(os.path.join(POST_DIR, f"LAKE{LAKE_ID}-*_dineof.nc"))
    if not output_files:
        print("ERROR: No dineof output file found")
        return
    output_path = output_files[0]
    
    print(f"Prepared: {PREPARED_PATH}")
    print(f"Output:   {output_path}")
    print()
    
    # Get site coordinates from selection CSV (or hardcode if known)
    # For now, we'll find the buoy site from the insitu validation output
    csv_path = os.path.join(POST_DIR, "insitu_cv_validation", f"LAKE{LAKE_ID}_insitu_stats_site1.csv")
    
    # We need the actual site coordinates - let's get them from the data
    # For now, assume we need to check all lake pixels or use a representative one
    
    # ========================================================================
    # 1. Check prepared.nc
    # ========================================================================
    print("1. PREPARED.NC")
    print("-"*40)
    
    with xr.open_dataset(PREPARED_PATH, decode_times=False) as ds_prep:
        print(f"Time steps: {len(ds_prep['time'])}")
        
        # Decode time
        raw_times = ds_prep['time'].values
        time_units = ds_prep.attrs.get('time_units', None)
        if time_units and 'days since' in time_units:
            base_str = time_units.replace('days since ', '').strip()
            base_time = pd.Timestamp(base_str)
            prep_times = base_time + pd.to_timedelta(raw_times, unit='D')
            prep_dates = set(t.date() for t in prep_times)
            print(f"Date range: {min(prep_dates)} to {max(prep_dates)}")
            print(f"Unique dates: {len(prep_dates)}")
        
        # Check variables
        print(f"Variables: {list(ds_prep.data_vars)}")
        
        if 'quality_level' in ds_prep:
            print("✓ quality_level present in prepared.nc")
        else:
            print("✗ quality_level NOT in prepared.nc")
    
    # ========================================================================
    # 2. Check output file
    # ========================================================================
    print()
    print("2. OUTPUT FILE")
    print("-"*40)
    
    with xr.open_dataset(output_path) as ds_out:
        print(f"Time steps: {len(ds_out['time'])}")
        
        out_times = pd.to_datetime(ds_out['time'].values)
        out_dates = set(t.date() for t in out_times)
        print(f"Date range: {min(out_dates)} to {max(out_dates)}")
        print(f"Unique dates: {len(out_dates)}")
        
        print(f"Variables: {list(ds_out.data_vars)}")
        
        if 'quality_level' in ds_out:
            print("✓ quality_level present in output file")
        else:
            print("✗ quality_level NOT in output file")
        
        if 'lake_surface_water_temperature' in ds_out:
            print("✓ lake_surface_water_temperature present")
        else:
            print("✗ lake_surface_water_temperature NOT in output file")
    
    # ========================================================================
    # 3. Compare time dimensions
    # ========================================================================
    print()
    print("3. TIME COMPARISON")
    print("-"*40)
    
    # Re-open to compare
    with xr.open_dataset(PREPARED_PATH, decode_times=False) as ds_prep:
        raw_times = ds_prep['time'].values
        time_units = ds_prep.attrs.get('time_units', None)
        base_str = time_units.replace('days since ', '').strip()
        base_time = pd.Timestamp(base_str)
        prep_times = base_time + pd.to_timedelta(raw_times, unit='D')
        prep_dates = set(t.date() for t in prep_times)
    
    with xr.open_dataset(output_path) as ds_out:
        out_times = pd.to_datetime(ds_out['time'].values)
        out_dates = set(t.date() for t in out_times)
    
    # Find differences
    in_prep_not_out = prep_dates - out_dates
    in_out_not_prep = out_dates - prep_dates
    
    print(f"Dates in prepared.nc but NOT in output: {len(in_prep_not_out)}")
    print(f"Dates in output but NOT in prepared.nc: {len(in_out_not_prep)}")
    
    if in_prep_not_out:
        print(f"\nFirst 10 dates in prepared.nc but not output:")
        for d in sorted(in_prep_not_out)[:10]:
            print(f"  {d}")
    
    if in_out_not_prep:
        print(f"\nFirst 10 dates in output but not prepared.nc:")
        for d in sorted(in_out_not_prep)[:10]:
            print(f"  {d}")
    
    # ========================================================================
    # 4. Check if this explains the 20-point diff
    # ========================================================================
    print()
    print("4. ANALYSIS")
    print("-"*40)
    
    if len(in_prep_not_out) == 20:
        print(f"✓ FOUND IT: There are exactly 20 dates in prepared.nc that are NOT in output file")
        print(f"  These dates would be counted in recon_observed but not available for obs_cropped")
    elif len(in_prep_not_out) > 0:
        print(f"  There are {len(in_prep_not_out)} dates in prepared.nc but not output")
        print(f"  This could partially explain the discrepancy")
    else:
        print("  Time dimensions match - issue is elsewhere")


if __name__ == "__main__":
    main()
