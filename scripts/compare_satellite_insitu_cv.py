#!/usr/bin/env python3
"""
Cross-reference: Satellite CV vs In-situ CV at SAME BUOY PIXEL
==============================================================

For the ~26 lakes with in-situ validation:
1. Get satellite obs vs reconstruction RMSE at buoy pixel (single-pixel satellite CV)
2. Get in-situ vs reconstruction RMSE (in-situ CV)
3. Compare which method wins each CV type
4. Check if DINCAE's in-situ wins correlate with buoy distance from shore

If DINEOF wins ~97-100% in single-pixel satellite CV but only ~60% in in-situ CV,
this confirms the discrepancy is about satellite vs in-situ ground truth,
NOT about spatial averaging artifacts.

Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

import os
import sys
from glob import glob
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

# =============================================================================
# CONFIGURATION
# =============================================================================

_SELECTION_CSV_DIR = "/home/users/shaerdan/general_purposes/insitu_cv"

DEFAULT_SELECTION_CSVS = [
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2010_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2007_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2018_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2020_selection.csv",
]

DEFAULT_BUOY_DIR = "/gws/ssde/j25b/nceo_uor/users/lcarrea01/INSITU/Buoy_Laura/ALL_FILES_QC"


# =============================================================================
# UTILITIES
# =============================================================================

def find_nearest_pixel(lat_array: np.ndarray, lon_array: np.ndarray,
                       target_lat: float, target_lon: float) -> Tuple[int, int]:
    """Find nearest grid point to target coordinates."""
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    return np.unravel_index(np.argmin(distance), distance.shape)


def get_lake_mask(ds: xr.Dataset) -> np.ndarray:
    """Extract binary lake mask from dataset."""
    if 'lakeid' in ds:
        lakeid = ds['lakeid'].values
        if np.nanmax(lakeid) == 1:
            mask = lakeid == 1
        else:
            mask = np.isfinite(lakeid) & (lakeid != 0)
    else:
        if 'temp_filled' in ds:
            mask = np.any(np.isfinite(ds['temp_filled'].values), axis=0)
        else:
            return None
    return mask.astype(bool)


def compute_distance_from_shore(lake_mask: np.ndarray) -> np.ndarray:
    """Compute distance (in pixels) from shore for each lake pixel."""
    if lake_mask is None or not lake_mask.any():
        return None
    return ndimage.distance_transform_edt(lake_mask)


def load_selection_csvs(csv_paths):
    """Load and combine selection CSVs."""
    dfs = []
    for path in csv_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except:
                pass
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['lake_id', 'site_id'], keep='first')
    return combined


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_lake(run_root: str, lake_id: int, alpha: str,
                 selection_df: pd.DataFrame,
                 quality_threshold: int = 3,
                 verbose: bool = False) -> Optional[Dict]:
    """
    For a lake with in-situ validation:
    1. Get satellite CV score at buoy pixel
    2. Get in-situ CV score (from validation CSV)
    3. Get buoy distance from shore
    """
    # Find post directory
    lake_str = f"{lake_id:09d}"
    post_dir = os.path.join(run_root, "post", lake_str, alpha)
    if not os.path.exists(post_dir):
        post_dir = os.path.join(run_root, "post", str(lake_id), alpha)
    
    if not os.path.exists(post_dir):
        return None
    
    # Check for in-situ validation results
    val_dir = os.path.join(post_dir, "insitu_cv_validation")
    if not os.path.exists(val_dir):
        return None
    
    stats_files = glob(os.path.join(val_dir, "*_insitu_stats_*.csv"))
    if not stats_files:
        return None
    
    # Load in-situ validation results
    try:
        stats_df = pd.read_csv(stats_files[0])
    except:
        return None
    
    # Get in-situ CV RMSE for both methods
    dineof_insitu = stats_df[(stats_df['data_type'] == 'reconstruction') & 
                              (stats_df['method'].str.contains('dineof', case=False, na=False))]
    dincae_insitu = stats_df[(stats_df['data_type'] == 'reconstruction') & 
                              (stats_df['method'].str.contains('dincae', case=False, na=False))]
    
    if dineof_insitu.empty or dincae_insitu.empty:
        return None
    
    insitu_dineof_rmse = dineof_insitu['rmse'].values[0]
    insitu_dincae_rmse = dincae_insitu['rmse'].values[0]
    insitu_winner = 'dineof' if insitu_dineof_rmse < insitu_dincae_rmse else 'dincae'
    
    # Get buoy location
    lake_sites = selection_df[selection_df['lake_id'] == lake_id]
    if lake_sites.empty:
        return None
    
    site = lake_sites.iloc[0]
    buoy_lat = site['latitude']
    buoy_lon = site['longitude']
    
    # Find output files
    dineof_files = glob(os.path.join(post_dir, "*_dineof.nc"))
    dincae_files = glob(os.path.join(post_dir, "*_dincae.nc"))
    
    if not dineof_files or not dincae_files:
        return None
    
    # Load datasets and compute satellite CV at buoy pixel
    ds_dineof = xr.open_dataset(dineof_files[0])
    ds_dincae = xr.open_dataset(dincae_files[0])
    
    try:
        # Find buoy pixel
        buoy_idx = find_nearest_pixel(
            ds_dineof['lat'].values, ds_dineof['lon'].values,
            buoy_lat, buoy_lon
        )
        
        # Get lake mask and distance from shore
        lake_mask = get_lake_mask(ds_dineof)
        distance_map = compute_distance_from_shore(lake_mask)
        
        buoy_distance = distance_map[buoy_idx] if distance_map is not None else np.nan
        
        # Extract timeseries at buoy pixel
        dineof_ts = ds_dineof['temp_filled'].isel(lat=buoy_idx[0], lon=buoy_idx[1]).values
        dincae_ts = ds_dincae['temp_filled'].isel(lat=buoy_idx[0], lon=buoy_idx[1]).values
        
        # Convert to Celsius if needed
        if np.nanmean(dineof_ts) > 100:
            dineof_ts = dineof_ts - 273.15
        if np.nanmean(dincae_ts) > 100:
            dincae_ts = dincae_ts - 273.15
        
        # Get satellite observations (quality filtered)
        if 'quality_level' in ds_dineof:
            quality = ds_dineof['quality_level'].isel(lat=buoy_idx[0], lon=buoy_idx[1]).values
            obs_mask = np.isfinite(quality) & (quality >= quality_threshold)
        else:
            obs_mask = np.isfinite(dineof_ts)
        
        # Compute satellite CV RMSE at buoy pixel
        if obs_mask.sum() >= 10:
            obs_temps = dineof_ts.copy()  # Original temps at observed times
            # Note: temp_filled contains the reconstruction, but at observed times 
            # it should match the original obs... Actually we need the original obs
            # Let's use the reconstruction error at observed points
            
            # For satellite CV, we compare reconstruction to original observation
            # The "observation" in the output file at quality >= threshold IS the original
            dineof_at_obs = dineof_ts[obs_mask]
            dincae_at_obs = dincae_ts[obs_mask]
            
            # We need the ORIGINAL satellite observation to compute error
            # In the output file, temp_filled IS the reconstruction
            # The original obs would need to come from elsewhere or we use a proxy
            
            # Actually, for a proper satellite CV, we'd need the original obs
            # But since both reconstructions are evaluated against same obs,
            # we can use the difference between them as a proxy
            # 
            # Better approach: check if there's an 'lswt' or original variable
            # OR use the quality_level mask to identify original obs times
            # and compare reconstruction to some reference
            
            # For now, let's compute the DIFFERENCE between methods
            # and see if they agree with in-situ CV results
            
            # Alternative: load from the original satellite file if available
            # But let's first check what's in the stats CSV for satellite CV
            
            sat_obs_dineof = stats_df[(stats_df['data_type'] == 'observation') & 
                                       (stats_df['method'].str.contains('dineof', case=False, na=False))]
            sat_obs_dincae = stats_df[(stats_df['data_type'] == 'observation') & 
                                       (stats_df['method'].str.contains('dincae', case=False, na=False))]
            
            # The observation stats compare satellite obs to in-situ
            # That's not what we want for satellite CV
            
            # Let's just get from the spatial analysis results we already computed
            sat_dineof_rmse = np.nan
            sat_dincae_rmse = np.nan
            sat_winner = None
            
            # Load from spatial analysis if available
            spatial_csv = os.path.join(run_root, "spatial_analysis", "spatial_analysis_data.csv")
            if os.path.exists(spatial_csv):
                spatial_df = pd.read_csv(spatial_csv)
                lake_row = spatial_df[spatial_df['lake_id'] == lake_id]
                if not lake_row.empty:
                    sat_dineof_rmse = lake_row['buoy_dineof_rmse'].values[0]
                    sat_dincae_rmse = lake_row['buoy_dincae_rmse'].values[0]
                    sat_winner = lake_row['buoy_winner'].values[0] if 'buoy_winner' in lake_row else None
        else:
            sat_dineof_rmse = np.nan
            sat_dincae_rmse = np.nan
            sat_winner = None
        
        result = {
            'lake_id': lake_id,
            'buoy_lat': buoy_lat,
            'buoy_lon': buoy_lon,
            'buoy_distance_from_shore': buoy_distance,
            
            # In-situ CV results
            'insitu_dineof_rmse': insitu_dineof_rmse,
            'insitu_dincae_rmse': insitu_dincae_rmse,
            'insitu_winner': insitu_winner,
            
            # Satellite CV at buoy pixel (from spatial analysis)
            'sat_dineof_rmse': sat_dineof_rmse,
            'sat_dincae_rmse': sat_dincae_rmse,
            'sat_winner': sat_winner,
            
            'n_sat_obs': int(obs_mask.sum()),
        }
        
        if verbose:
            print(f"  Lake {lake_id}: dist={buoy_distance:.1f}px, "
                  f"insitu_winner={insitu_winner}, sat_winner={sat_winner}")
        
        return result
        
    finally:
        ds_dineof.close()
        ds_dincae.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-reference satellite CV vs in-situ CV")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--alpha", default="a1000")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SATELLITE CV vs IN-SITU CV AT BUOY PIXEL")
    print("=" * 70)
    
    # Load selection CSVs
    selection_df = load_selection_csvs(DEFAULT_SELECTION_CSVS)
    print(f"Loaded buoy locations for {selection_df['lake_id'].nunique()} lakes")
    
    # Find lakes with in-situ validation
    post_dir = os.path.join(args.run_root, "post")
    results = []
    
    for folder in os.listdir(post_dir):
        try:
            lake_id = int(folder.lstrip('0') or '0')
        except:
            continue
        
        result = analyze_lake(args.run_root, lake_id, args.alpha, 
                            selection_df, verbose=args.verbose)
        if result is not None:
            results.append(result)
    
    print(f"\nAnalyzed {len(results)} lakes with in-situ validation")
    
    if len(results) < 3:
        print("ERROR: Not enough lakes")
        return 1
    
    df = pd.DataFrame(results)
    
    # Save data
    out_path = os.path.join(args.run_root, "satellite_vs_insitu_cv_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SAME LAKES, DIFFERENT GROUND TRUTH")
    print("=" * 70)
    
    n = len(df)
    
    # In-situ CV results
    insitu_dineof_wins = (df['insitu_winner'] == 'dineof').sum()
    print(f"\nIN-SITU CV (buoy as ground truth):")
    print(f"  DINEOF wins: {insitu_dineof_wins}/{n} ({100*insitu_dineof_wins/n:.0f}%)")
    print(f"  DINCAE wins: {n - insitu_dineof_wins}/{n} ({100*(n-insitu_dineof_wins)/n:.0f}%)")
    
    # Satellite CV at buoy pixel
    sat_valid = df['sat_winner'].notna()
    if sat_valid.sum() > 0:
        sat_df = df[sat_valid]
        sat_dineof_wins = (sat_df['sat_winner'] == 'dineof').sum()
        n_sat = len(sat_df)
        print(f"\nSATELLITE CV at BUOY PIXEL (satellite obs as ground truth):")
        print(f"  DINEOF wins: {sat_dineof_wins}/{n_sat} ({100*sat_dineof_wins/n_sat:.0f}%)")
        print(f"  DINCAE wins: {n_sat - sat_dineof_wins}/{n_sat} ({100*(n_sat-sat_dineof_wins)/n_sat:.0f}%)")
    
    # Cross-reference: same pixel, different ground truth
    print("\n" + "-" * 70)
    print("CROSS-REFERENCE: Same pixel, different winner?")
    print("-" * 70)
    
    both_valid = df['sat_winner'].notna() & df['insitu_winner'].notna()
    if both_valid.sum() > 0:
        both_df = df[both_valid]
        
        # Cases where sat says DINEOF but insitu says DINCAE
        sat_dineof_insitu_dincae = ((both_df['sat_winner'] == 'dineof') & 
                                     (both_df['insitu_winner'] == 'dincae')).sum()
        # Cases where both agree on DINEOF
        both_dineof = ((both_df['sat_winner'] == 'dineof') & 
                       (both_df['insitu_winner'] == 'dineof')).sum()
        # Cases where both agree on DINCAE
        both_dincae = ((both_df['sat_winner'] == 'dincae') & 
                       (both_df['insitu_winner'] == 'dincae')).sum()
        # Cases where sat says DINCAE but insitu says DINEOF
        sat_dincae_insitu_dineof = ((both_df['sat_winner'] == 'dincae') & 
                                     (both_df['insitu_winner'] == 'dineof')).sum()
        
        print(f"\n  Both say DINEOF:     {both_dineof}")
        print(f"  Both say DINCAE:     {both_dincae}")
        print(f"  Sat=DINEOF, In-situ=DINCAE: {sat_dineof_insitu_dincae}")
        print(f"  Sat=DINCAE, In-situ=DINEOF: {sat_dincae_insitu_dineof}")
        
        print(f"\n  Lakes where DINCAE wins in-situ but DINEOF wins satellite CV:")
        flip_lakes = both_df[(both_df['sat_winner'] == 'dineof') & 
                             (both_df['insitu_winner'] == 'dincae')]
        for _, row in flip_lakes.iterrows():
            print(f"    Lake {int(row['lake_id'])}: dist_from_shore={row['buoy_distance_from_shore']:.1f}px")
    
    # Check if DINCAE in-situ wins correlate with shore distance
    print("\n" + "-" * 70)
    print("DINCAE IN-SITU WINS vs SHORE DISTANCE")
    print("-" * 70)
    
    dincae_wins = df[df['insitu_winner'] == 'dincae']
    dineof_wins = df[df['insitu_winner'] == 'dineof']
    
    if len(dincae_wins) > 0 and len(dineof_wins) > 0:
        dincae_mean_dist = dincae_wins['buoy_distance_from_shore'].mean()
        dineof_mean_dist = dineof_wins['buoy_distance_from_shore'].mean()
        
        print(f"\n  Mean buoy distance from shore:")
        print(f"    Lakes where DINCAE wins: {dincae_mean_dist:.1f} pixels")
        print(f"    Lakes where DINEOF wins: {dineof_mean_dist:.1f} pixels")
        
        if dincae_mean_dist < dineof_mean_dist:
            print("\n  --> DINCAE wins tend to be CLOSER to shore")
        else:
            print("\n  --> No clear shore distance pattern")
        
        # Near shore analysis
        near_shore = df['buoy_distance_from_shore'] < 5
        if near_shore.sum() > 0:
            near_df = df[near_shore]
            near_dineof = (near_df['insitu_winner'] == 'dineof').sum()
            print(f"\n  Near shore buoys (<5 pixels): {near_shore.sum()}")
            print(f"    DINEOF wins: {near_dineof}/{near_shore.sum()} ({100*near_dineof/near_shore.sum():.0f}%)")
        
        far_shore = df['buoy_distance_from_shore'] >= 5
        if far_shore.sum() > 0:
            far_df = df[far_shore]
            far_dineof = (far_df['insitu_winner'] == 'dineof').sum()
            print(f"\n  Far from shore buoys (>=5 pixels): {far_shore.sum()}")
            print(f"    DINEOF wins: {far_dineof}/{far_shore.sum()} ({100*far_dineof/far_shore.sum():.0f}%)")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
