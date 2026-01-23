#!/usr/bin/env python3
"""
Check if the problematic dates are ice_replaced in prepared.nc
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob

RUN_ROOT = "/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/archive/anomaly-20251215-8ea02d-exp3"
LAKE_ID = "000001115"
ALPHA_SLUG = "a1000"

PREPARED_PATH = os.path.join(RUN_ROOT, "prepared", LAKE_ID, "prepared.nc")
POST_DIR = os.path.join(RUN_ROOT, "post", LAKE_ID, ALPHA_SLUG)

SITE_LAT = 47.84166666666667
SITE_LON = 16.76666666666665

# The 17 dates where output LSWT is NaN but prepared.nc has valid data
PROBLEM_DATES = [
    "2005-02-13", "2005-12-09", "2005-12-26", "2006-12-26", "2006-12-29",
    "2006-12-30", "2008-01-26", "2010-12-13", "2012-01-14", "2014-01-25",
    "2014-12-25", "2021-01-24", "2021-01-30", "2021-12-03", "2021-12-06",
    "2022-01-14", "2022-01-27",
]


def find_nearest_idx(lat_arr, lon_arr, target_lat, target_lon):
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    return np.unravel_index(np.argmin(distance), distance.shape)


def main():
    print("="*80)
    print("CHECKING ICE_REPLACED STATUS FOR PROBLEMATIC DATES")
    print("="*80)
    
    # Load prepared.nc
    with xr.open_dataset(PREPARED_PATH, decode_times=False) as ds_prep:
        prep_idx = find_nearest_idx(ds_prep['lat'].values, ds_prep['lon'].values, SITE_LAT, SITE_LON)
        
        # Decode time
        raw_times = ds_prep['time'].values
        time_units = ds_prep.attrs.get('time_units', '')
        base_str = time_units.replace('days since ', '').strip()
        base_time = pd.Timestamp(base_str)
        prep_times = base_time + pd.to_timedelta(raw_times, unit='D')
        prep_date_to_idx = {t.date(): i for i, t in enumerate(prep_times)}
        
        # Get data at pixel
        prep_lswt = ds_prep['lake_surface_water_temperature'].isel(lat=prep_idx[0], lon=prep_idx[1]).values
        
        prep_ice = None
        if 'ice_replaced' in ds_prep:
            prep_ice = ds_prep['ice_replaced'].isel(lat=prep_idx[0], lon=prep_idx[1]).values
            print("✓ ice_replaced variable found in prepared.nc")
        else:
            print("✗ ice_replaced NOT found")
        
        prep_ql = None
        if 'quality_level' in ds_prep:
            prep_ql = ds_prep['quality_level'].isel(lat=prep_idx[0], lon=prep_idx[1]).values
    
    print(f"\n{'Date':<12} {'LSWT':<10} {'Quality':<10} {'Ice Replaced?'}")
    print("-"*50)
    
    for date_str in PROBLEM_DATES:
        d = pd.Timestamp(date_str).date()
        if d in prep_date_to_idx:
            idx = prep_date_to_idx[d]
            lswt = prep_lswt[idx]
            ql = prep_ql[idx] if prep_ql is not None else "N/A"
            ice = prep_ice[idx] if prep_ice is not None else "N/A"
            
            ice_str = "YES" if ice == 1 else "NO" if ice == 0 else str(ice)
            print(f"{date_str:<12} {lswt:<10.2f} {ql:<10} {ice_str}")
        else:
            print(f"{date_str:<12} NOT IN PREPARED.NC")
    
    # Also check: where does lake_surface_water_temperature in output come from?
    print("\n" + "="*80)
    print("CHECKING OUTPUT FILE SOURCE")
    print("="*80)
    
    output_files = glob(os.path.join(POST_DIR, f"LAKE{LAKE_ID}-*_dineof.nc"))
    output_path = output_files[0]
    
    with xr.open_dataset(output_path) as ds_out:
        # Check attributes
        if 'lake_surface_water_temperature' in ds_out:
            var = ds_out['lake_surface_water_temperature']
            print(f"\nlake_surface_water_temperature attributes:")
            for k, v in var.attrs.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()