#!/usr/bin/env python3
"""
Find the exact dates causing obs_cropped != recon_observed for Lake 1115.

The 20 missing dates are where:
- prepared.nc has valid data at pixel
- output file's lake_surface_water_temperature is NaN at same pixel
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob

# ============================================================================
# Configuration
# ============================================================================
RUN_ROOT = "/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/archive/anomaly-20251215-8ea02d-exp3"
LAKE_ID = "000001115"
ALPHA_SLUG = "a1000"
QUALITY_THRESHOLD = 3

# Site 1 coordinates - need to get from selection CSV or hardcode
# Let's find them from the buoy data location files
SELECTION_CSV_DIR = "/home/users/shaerdan/general_purposes/insitu_cv"

PREPARED_PATH = os.path.join(RUN_ROOT, "prepared", LAKE_ID, "prepared.nc")
POST_DIR = os.path.join(RUN_ROOT, "post", LAKE_ID, ALPHA_SLUG)


def find_nearest_idx(lat_arr, lon_arr, target_lat, target_lon):
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    idx = np.unravel_index(np.argmin(distance), distance.shape)
    return idx, np.min(distance)


def get_site_coords(lake_id_cci):
    """Get site coordinates from selection CSV."""
    selection_files = [
        f"{SELECTION_CSV_DIR}/L3S_QL_MDB_2010_selection.csv",
        f"{SELECTION_CSV_DIR}/L3S_QL_MDB_2007_selection.csv",
        f"{SELECTION_CSV_DIR}/L3S_QL_MDB_2018_selection.csv",
        f"{SELECTION_CSV_DIR}/L3S_QL_MDB_2020_selection.csv",
    ]
    
    for csv_path in selection_files:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        matches = df[df['lake_id_cci'] == lake_id_cci]
        if not matches.empty:
            row = matches.iloc[0]
            return row['latitude'], row['longitude'], row.get('site_id', 1)
    
    return None, None, None


def main():
    print("="*80)
    print(f"FINDING THE 20 MISSING DATES FOR LAKE {LAKE_ID}")
    print("="*80)
    
    # Get site coordinates
    lake_id_cci = int(LAKE_ID)
    site_lat, site_lon, site_id = get_site_coords(lake_id_cci)
    
    if site_lat is None:
        print("ERROR: Could not find site coordinates")
        return
    
    print(f"Site coordinates: ({site_lat}, {site_lon}), site_id={site_id}")
    print()
    
    # Find output file
    output_files = glob(os.path.join(POST_DIR, f"LAKE{LAKE_ID}-*_dineof.nc"))
    output_path = output_files[0]
    
    # ========================================================================
    # Load prepared.nc
    # ========================================================================
    print("Loading prepared.nc...")
    with xr.open_dataset(PREPARED_PATH, decode_times=False) as ds_prep:
        # Find grid point
        prep_idx, prep_dist = find_nearest_idx(
            ds_prep['lat'].values, ds_prep['lon'].values, site_lat, site_lon
        )
        print(f"  Grid index: {prep_idx}, distance: {prep_dist:.6f}°")
        
        # Decode time
        raw_times = ds_prep['time'].values
        time_units = ds_prep.attrs.get('time_units', '')
        base_str = time_units.replace('days since ', '').strip()
        base_time = pd.Timestamp(base_str)
        prep_times = base_time + pd.to_timedelta(raw_times, unit='D')
        
        # Get LSWT at pixel
        prep_lswt = ds_prep['lake_surface_water_temperature'].isel(
            lat=prep_idx[0], lon=prep_idx[1]
        ).values
        
        # Dates where prepared.nc has valid data
        prep_valid_dates = set()
        for i, t in enumerate(prep_times):
            if not np.isnan(prep_lswt[i]):
                prep_valid_dates.add(t.date())
        
        print(f"  Valid dates at pixel: {len(prep_valid_dates)}")
    
    # ========================================================================
    # Load output file
    # ========================================================================
    print("\nLoading output file...")
    with xr.open_dataset(output_path) as ds_out:
        # Find grid point
        out_idx, out_dist = find_nearest_idx(
            ds_out['lat'].values, ds_out['lon'].values, site_lat, site_lon
        )
        print(f"  Grid index: {out_idx}, distance: {out_dist:.6f}°")
        
        out_times = pd.to_datetime(ds_out['time'].values)
        
        # Get LSWT and quality at pixel
        out_lswt = ds_out['lake_surface_water_temperature'].isel(
            lat=out_idx[0], lon=out_idx[1]
        ).values
        
        out_quality = ds_out['quality_level'].isel(
            lat=out_idx[0], lon=out_idx[1]
        ).values
        
        # Dates where output has valid LSWT with quality >= threshold
        out_valid_dates = set()
        out_valid_no_quality = set()  # Valid LSWT regardless of quality
        
        for i, t in enumerate(out_times):
            if not np.isnan(out_lswt[i]):
                out_valid_no_quality.add(t.date())
                ql = out_quality[i]
                if not np.isnan(ql) and ql >= QUALITY_THRESHOLD:
                    out_valid_dates.add(t.date())
        
        print(f"  Valid LSWT dates (any quality): {len(out_valid_no_quality)}")
        print(f"  Valid LSWT dates (quality >= {QUALITY_THRESHOLD}): {len(out_valid_dates)}")
    
    # ========================================================================
    # Compare
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    # Dates in prepared.nc but NOT in output (with quality filter)
    in_prep_not_out = prep_valid_dates - out_valid_dates
    in_out_not_prep = out_valid_dates - prep_valid_dates
    
    print(f"\nDates valid in prepared.nc but NOT valid in output (quality >= {QUALITY_THRESHOLD}):")
    print(f"  Count: {len(in_prep_not_out)}")
    
    if in_prep_not_out:
        print(f"\n  These {len(in_prep_not_out)} dates are the problem:")
        for d in sorted(in_prep_not_out)[:30]:
            print(f"    {d}")
        if len(in_prep_not_out) > 30:
            print(f"    ... and {len(in_prep_not_out) - 30} more")
    
    # Check if it's a quality issue or NaN issue
    in_prep_not_out_noqual = prep_valid_dates - out_valid_no_quality
    print(f"\nDates valid in prepared.nc but LSWT is NaN in output (regardless of quality):")
    print(f"  Count: {len(in_prep_not_out_noqual)}")
    
    quality_rejected = in_prep_not_out - in_prep_not_out_noqual
    print(f"\nDates valid in prepared.nc, LSWT exists in output, but quality < {QUALITY_THRESHOLD}:")
    print(f"  Count: {len(quality_rejected)}")
    
    if len(in_prep_not_out) == 20:
        print("\n" + "="*80)
        print("✓ FOUND THE 20 DATES!")
        print("="*80)
    
    # ========================================================================
    # Detailed analysis of the problematic dates
    # ========================================================================
    if in_prep_not_out:
        print("\n" + "="*80)
        print("DETAILED ANALYSIS OF PROBLEMATIC DATES")
        print("="*80)
        
        # Reload to check specific values
        with xr.open_dataset(output_path) as ds_out:
            out_times = pd.to_datetime(ds_out['time'].values)
            date_to_idx = {t.date(): i for i, t in enumerate(out_times)}
            
            out_lswt = ds_out['lake_surface_water_temperature'].isel(
                lat=out_idx[0], lon=out_idx[1]
            ).values
            out_quality = ds_out['quality_level'].isel(
                lat=out_idx[0], lon=out_idx[1]
            ).values
            
            print(f"\n{'Date':<12} {'In Output?':<12} {'LSWT':<12} {'Quality':<10} {'Issue'}")
            print("-"*60)
            
            for d in sorted(in_prep_not_out)[:20]:
                if d in date_to_idx:
                    idx = date_to_idx[d]
                    lswt_val = out_lswt[idx]
                    ql_val = out_quality[idx]
                    
                    if np.isnan(lswt_val):
                        issue = "LSWT is NaN"
                    elif np.isnan(ql_val):
                        issue = "Quality is NaN"
                    elif ql_val < QUALITY_THRESHOLD:
                        issue = f"Quality={ql_val} < {QUALITY_THRESHOLD}"
                    else:
                        issue = "???"
                    
                    print(f"{d}   Yes          {lswt_val:<12.2f} {ql_val:<10} {issue}")
                else:
                    print(f"{d}   NO - date not in output file!")


if __name__ == "__main__":
    main()