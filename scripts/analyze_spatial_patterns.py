#!/usr/bin/env python3
"""
Spatial Analysis: Center vs Shore Performance
==============================================

Hypothesis: DINEOF wins satellite CV (97%) because it's spatially smooth, 
giving safe reconstructions over large areas. But DINCAE might capture 
local patterns better at specific pixels.

Tests:
1. Compare DINEOF vs DINCAE at lake CENTER vs near SHORE
2. Check if DINEOF's dominance holds at individual pixels
3. Calculate how far buoy locations are from shore

If hypothesis is correct:
- DINEOF dominance (97%) should reduce when comparing individual pixels
- Buoys near shore might favor DINCAE

Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

import argparse
import os
import sys
from glob import glob
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
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


# =============================================================================
# SPATIAL UTILITIES
# =============================================================================

def get_lake_mask(ds: xr.Dataset) -> np.ndarray:
    """Extract binary lake mask from dataset."""
    if 'lakeid' in ds:
        lakeid = ds['lakeid'].values
        # Handle different conventions
        if np.nanmax(lakeid) == 1:
            mask = lakeid == 1
        else:
            mask = np.isfinite(lakeid) & (lakeid != 0)
    else:
        # Fallback: use valid data mask
        if 'temp_filled' in ds:
            mask = np.any(np.isfinite(ds['temp_filled'].values), axis=0)
        else:
            return None
    return mask.astype(bool)


def compute_distance_from_shore(lake_mask: np.ndarray) -> np.ndarray:
    """
    Compute distance (in pixels) from shore for each lake pixel.
    
    Uses distance transform on the lake mask.
    Shore pixels have distance 0, interior pixels have positive distance.
    """
    if lake_mask is None or not lake_mask.any():
        return None
    
    # Distance transform gives distance to nearest False (non-lake) pixel
    distance = ndimage.distance_transform_edt(lake_mask)
    return distance


def find_center_pixel(lake_mask: np.ndarray, distance_map: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Find the center pixel (furthest from shore).
    
    Returns (lat_idx, lon_idx).
    """
    if distance_map is None or not lake_mask.any():
        return None
    
    # Find pixel with maximum distance from shore
    masked_distance = np.where(lake_mask, distance_map, -1)
    idx = np.unravel_index(np.argmax(masked_distance), masked_distance.shape)
    return idx


def find_shore_pixel(lake_mask: np.ndarray, distance_map: np.ndarray, 
                     min_distance: int = 3) -> Optional[Tuple[int, int]]:
    """
    Find a pixel near shore (but with safety margin).
    
    Finds pixel with distance between min_distance and min_distance + 2.
    Returns (lat_idx, lon_idx).
    """
    if distance_map is None or not lake_mask.any():
        return None
    
    # Find pixels within the target distance range
    target_mask = lake_mask & (distance_map >= min_distance) & (distance_map <= min_distance + 2)
    
    if not target_mask.any():
        # Relax constraints
        target_mask = lake_mask & (distance_map >= 1) & (distance_map <= 5)
    
    if not target_mask.any():
        return None
    
    # Pick one (e.g., first found)
    indices = np.where(target_mask)
    return (indices[0][0], indices[1][0])


def get_pixel_distance_from_shore(lake_mask: np.ndarray, distance_map: np.ndarray,
                                   lat_idx: int, lon_idx: int) -> float:
    """Get distance from shore for a specific pixel."""
    if distance_map is None:
        return np.nan
    if not lake_mask[lat_idx, lon_idx]:
        return 0  # Not on lake
    return distance_map[lat_idx, lon_idx]


def find_nearest_pixel(lat_array: np.ndarray, lon_array: np.ndarray,
                       target_lat: float, target_lon: float) -> Tuple[int, int]:
    """Find nearest grid point to target coordinates."""
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    return np.unravel_index(np.argmin(distance), distance.shape)


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_pixel_timeseries(ds: xr.Dataset, lat_idx: int, lon_idx: int,
                              var_name: str = 'temp_filled') -> np.ndarray:
    """Extract timeseries at a pixel."""
    if var_name not in ds:
        return np.array([])
    
    temps = ds[var_name].isel(lat=lat_idx, lon=lon_idx).values
    
    # Convert to Celsius if needed
    if len(temps) > 0 and np.nanmean(temps) > 100:
        temps = temps - 273.15
    
    return temps


def extract_observations_at_pixel(ds: xr.Dataset, lat_idx: int, lon_idx: int,
                                   quality_threshold: int = 3) -> np.ndarray:
    """Extract only original observations (not gap-filled) at a pixel."""
    temps = extract_pixel_timeseries(ds, lat_idx, lon_idx, 'temp_filled')
    
    if 'quality_level' in ds:
        quality = ds['quality_level'].isel(lat=lat_idx, lon=lon_idx).values
        valid_mask = np.isfinite(quality) & (quality >= quality_threshold)
        temps = np.where(valid_mask, temps, np.nan)
    
    return temps


def compute_pixel_metrics(temps: np.ndarray) -> Dict[str, float]:
    """Compute metrics for a pixel timeseries."""
    valid = temps[np.isfinite(temps)]
    
    if len(valid) < 10:
        return {
            'n_valid': len(valid),
            'mean': np.nan,
            'std': np.nan,
            'roughness': np.nan,
        }
    
    # Roughness: 1 - lag1 autocorrelation
    x_centered = valid - np.mean(valid)
    autocorr = np.corrcoef(x_centered[:-1], x_centered[1:])[0, 1]
    roughness = 1 - autocorr if not np.isnan(autocorr) else np.nan
    
    return {
        'n_valid': len(valid),
        'mean': np.mean(valid),
        'std': np.std(valid),
        'roughness': roughness,
    }


def compute_rmse_between(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """Compute RMSE between two timeseries (on valid overlapping points)."""
    valid = np.isfinite(ts1) & np.isfinite(ts2)
    if valid.sum() < 10:
        return np.nan
    
    diff = ts1[valid] - ts2[valid]
    return np.sqrt(np.mean(diff**2))


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_lake(run_root: str, lake_id: int, alpha: str,
                 selection_df: Optional[pd.DataFrame] = None,
                 shore_margin: int = 3,
                 quality_threshold: int = 3,
                 verbose: bool = False) -> Optional[Dict]:
    """
    Analyze spatial patterns for a single lake.
    
    Compares DINEOF vs DINCAE at:
    - Lake center (furthest from shore)
    - Near shore (with safety margin)
    - Buoy location (if available)
    """
    # Find output files
    lake_str = f"{lake_id:09d}"
    post_dir = os.path.join(run_root, "post", lake_str, alpha)
    if not os.path.exists(post_dir):
        post_dir = os.path.join(run_root, "post", str(lake_id), alpha)
    
    if not os.path.exists(post_dir):
        if verbose:
            print(f"  Lake {lake_id}: post directory not found")
        return None
    
    dineof_files = glob(os.path.join(post_dir, "*_dineof.nc"))
    dincae_files = glob(os.path.join(post_dir, "*_dincae.nc"))
    
    if not dineof_files or not dincae_files:
        if verbose:
            print(f"  Lake {lake_id}: missing output files")
        return None
    
    # Load datasets
    ds_dineof = xr.open_dataset(dineof_files[0])
    ds_dincae = xr.open_dataset(dincae_files[0])
    
    try:
        # Get lake mask and distance map
        lake_mask = get_lake_mask(ds_dineof)
        if lake_mask is None or not lake_mask.any():
            if verbose:
                print(f"  Lake {lake_id}: could not extract lake mask")
            return None
        
        distance_map = compute_distance_from_shore(lake_mask)
        
        # Find center and shore pixels
        center_idx = find_center_pixel(lake_mask, distance_map)
        shore_idx = find_shore_pixel(lake_mask, distance_map, shore_margin)
        
        if center_idx is None:
            if verbose:
                print(f"  Lake {lake_id}: could not find center pixel")
            return None
        
        # Lake stats
        lake_pixels = lake_mask.sum()
        max_distance = distance_map[lake_mask].max() if lake_mask.any() else 0
        
        result = {
            'lake_id': lake_id,
            'lake_pixels': int(lake_pixels),
            'max_distance_from_shore': float(max_distance),
            'center_lat_idx': center_idx[0],
            'center_lon_idx': center_idx[1],
            'center_distance': float(distance_map[center_idx]),
        }
        
        # Extract timeseries at CENTER
        dineof_center = extract_pixel_timeseries(ds_dineof, center_idx[0], center_idx[1])
        dincae_center = extract_pixel_timeseries(ds_dincae, center_idx[0], center_idx[1])
        obs_center = extract_observations_at_pixel(ds_dineof, center_idx[0], center_idx[1], quality_threshold)
        
        result['center_dineof_std'] = compute_pixel_metrics(dineof_center)['std']
        result['center_dincae_std'] = compute_pixel_metrics(dincae_center)['std']
        result['center_obs_n'] = np.isfinite(obs_center).sum()
        
        # Compare methods at center using observations
        valid_obs = np.isfinite(obs_center)
        if valid_obs.sum() >= 10:
            dineof_at_obs = dineof_center[valid_obs]
            dincae_at_obs = dincae_center[valid_obs]
            obs_vals = obs_center[valid_obs]
            
            result['center_dineof_rmse'] = np.sqrt(np.mean((dineof_at_obs - obs_vals)**2))
            result['center_dincae_rmse'] = np.sqrt(np.mean((dincae_at_obs - obs_vals)**2))
            result['center_winner'] = 'dineof' if result['center_dineof_rmse'] < result['center_dincae_rmse'] else 'dincae'
        else:
            result['center_dineof_rmse'] = np.nan
            result['center_dincae_rmse'] = np.nan
            result['center_winner'] = None
        
        # Extract timeseries at SHORE (if found)
        if shore_idx is not None:
            result['shore_lat_idx'] = shore_idx[0]
            result['shore_lon_idx'] = shore_idx[1]
            result['shore_distance'] = float(distance_map[shore_idx])
            
            dineof_shore = extract_pixel_timeseries(ds_dineof, shore_idx[0], shore_idx[1])
            dincae_shore = extract_pixel_timeseries(ds_dincae, shore_idx[0], shore_idx[1])
            obs_shore = extract_observations_at_pixel(ds_dineof, shore_idx[0], shore_idx[1], quality_threshold)
            
            result['shore_dineof_std'] = compute_pixel_metrics(dineof_shore)['std']
            result['shore_dincae_std'] = compute_pixel_metrics(dincae_shore)['std']
            result['shore_obs_n'] = np.isfinite(obs_shore).sum()
            
            valid_obs_shore = np.isfinite(obs_shore)
            if valid_obs_shore.sum() >= 10:
                dineof_at_obs = dineof_shore[valid_obs_shore]
                dincae_at_obs = dincae_shore[valid_obs_shore]
                obs_vals = obs_shore[valid_obs_shore]
                
                result['shore_dineof_rmse'] = np.sqrt(np.mean((dineof_at_obs - obs_vals)**2))
                result['shore_dincae_rmse'] = np.sqrt(np.mean((dincae_at_obs - obs_vals)**2))
                result['shore_winner'] = 'dineof' if result['shore_dineof_rmse'] < result['shore_dincae_rmse'] else 'dincae'
            else:
                result['shore_dineof_rmse'] = np.nan
                result['shore_dincae_rmse'] = np.nan
                result['shore_winner'] = None
        
        # Check BUOY location if selection_df provided
        if selection_df is not None:
            lake_sites = selection_df[selection_df['lake_id'] == lake_id]
            if not lake_sites.empty:
                site = lake_sites.iloc[0]
                buoy_lat = site['latitude']
                buoy_lon = site['longitude']
                
                buoy_idx = find_nearest_pixel(
                    ds_dineof['lat'].values, ds_dineof['lon'].values,
                    buoy_lat, buoy_lon
                )
                
                result['buoy_lat_idx'] = buoy_idx[0]
                result['buoy_lon_idx'] = buoy_idx[1]
                result['buoy_distance_from_shore'] = float(distance_map[buoy_idx])
                result['buoy_on_lake'] = bool(lake_mask[buoy_idx])
                
                # Extract timeseries at buoy location
                dineof_buoy = extract_pixel_timeseries(ds_dineof, buoy_idx[0], buoy_idx[1])
                dincae_buoy = extract_pixel_timeseries(ds_dincae, buoy_idx[0], buoy_idx[1])
                obs_buoy = extract_observations_at_pixel(ds_dineof, buoy_idx[0], buoy_idx[1], quality_threshold)
                
                result['buoy_obs_n'] = np.isfinite(obs_buoy).sum()
                
                valid_obs_buoy = np.isfinite(obs_buoy)
                if valid_obs_buoy.sum() >= 10:
                    dineof_at_obs = dineof_buoy[valid_obs_buoy]
                    dincae_at_obs = dincae_buoy[valid_obs_buoy]
                    obs_vals = obs_buoy[valid_obs_buoy]
                    
                    result['buoy_dineof_rmse'] = np.sqrt(np.mean((dineof_at_obs - obs_vals)**2))
                    result['buoy_dincae_rmse'] = np.sqrt(np.mean((dincae_at_obs - obs_vals)**2))
                    result['buoy_winner'] = 'dineof' if result['buoy_dineof_rmse'] < result['buoy_dincae_rmse'] else 'dincae'
                else:
                    result['buoy_dineof_rmse'] = np.nan
                    result['buoy_dincae_rmse'] = np.nan
                    result['buoy_winner'] = None
        
        if verbose:
            center_win = result.get('center_winner', '?')
            shore_win = result.get('shore_winner', '?')
            buoy_dist = result.get('buoy_distance_from_shore', np.nan)
            print(f"  Lake {lake_id}: pixels={lake_pixels}, max_dist={max_distance:.1f}, "
                  f"center_winner={center_win}, shore_winner={shore_win}, buoy_dist={buoy_dist:.1f}")
        
        return result
        
    finally:
        ds_dineof.close()
        ds_dincae.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze spatial patterns: center vs shore performance",
    )
    
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--shore-margin", type=int, default=3, help="Min pixels from shore")
    parser.add_argument("--selection-csvs", nargs="+", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "spatial_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    selection_csvs = args.selection_csvs or DEFAULT_SELECTION_CSVS
    
    print("=" * 70)
    print("SPATIAL ANALYSIS: CENTER vs SHORE")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Load selection CSVs for buoy locations
    selection_df = None
    for csv_path in selection_csvs:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if selection_df is None:
                    selection_df = df
                else:
                    selection_df = pd.concat([selection_df, df], ignore_index=True)
            except:
                pass
    
    if selection_df is not None:
        selection_df = selection_df.drop_duplicates(subset=['lake_id', 'site_id'], keep='first')
        print(f"Loaded buoy locations for {selection_df['lake_id'].nunique()} lakes")
    
    # Find all lakes
    post_dir = os.path.join(args.run_root, "post")
    lake_ids = []
    for folder in os.listdir(post_dir):
        try:
            lake_id = int(folder.lstrip('0') or '0')
            if lake_id > 0:
                lake_ids.append(lake_id)
        except:
            continue
    
    lake_ids = sorted(lake_ids)
    print(f"Found {len(lake_ids)} lakes to analyze")
    
    # Analyze each lake
    results = []
    for lake_id in lake_ids:
        result = analyze_lake(
            args.run_root, lake_id, args.alpha,
            selection_df, args.shore_margin,
            verbose=args.verbose
        )
        if result is not None:
            results.append(result)
    
    print(f"\nSuccessfully analyzed {len(results)} lakes")
    
    if len(results) < 3:
        print("ERROR: Not enough lakes")
        return 1
    
    results_df = pd.DataFrame(results)
    
    # Save raw data
    csv_path = os.path.join(args.output_dir, 'spatial_analysis_data.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Center pixel results
    center_valid = results_df['center_winner'].notna()
    if center_valid.sum() > 0:
        center_df = results_df[center_valid]
        dineof_center = (center_df['center_winner'] == 'dineof').sum()
        n_center = len(center_df)
        print(f"\nCENTER PIXEL (furthest from shore):")
        print(f"  DINEOF wins: {dineof_center}/{n_center} ({100*dineof_center/n_center:.0f}%)")
        print(f"  DINCAE wins: {n_center - dineof_center}/{n_center} ({100*(n_center-dineof_center)/n_center:.0f}%)")
    
    # Shore pixel results
    shore_valid = results_df['shore_winner'].notna() if 'shore_winner' in results_df else pd.Series([False]*len(results_df))
    if shore_valid.sum() > 0:
        shore_df = results_df[shore_valid]
        dineof_shore = (shore_df['shore_winner'] == 'dineof').sum()
        n_shore = len(shore_df)
        print(f"\nSHORE PIXEL ({args.shore_margin}-{args.shore_margin+2} pixels from shore):")
        print(f"  DINEOF wins: {dineof_shore}/{n_shore} ({100*dineof_shore/n_shore:.0f}%)")
        print(f"  DINCAE wins: {n_shore - dineof_shore}/{n_shore} ({100*(n_shore-dineof_shore)/n_shore:.0f}%)")
    
    # Buoy location results
    buoy_valid = results_df['buoy_winner'].notna() if 'buoy_winner' in results_df else pd.Series([False]*len(results_df))
    if buoy_valid.sum() > 0:
        buoy_df = results_df[buoy_valid]
        dineof_buoy = (buoy_df['buoy_winner'] == 'dineof').sum()
        n_buoy = len(buoy_df)
        print(f"\nBUOY PIXEL (actual buoy location):")
        print(f"  DINEOF wins: {dineof_buoy}/{n_buoy} ({100*dineof_buoy/n_buoy:.0f}%)")
        print(f"  DINCAE wins: {n_buoy - dineof_buoy}/{n_buoy} ({100*(n_buoy-dineof_buoy)/n_buoy:.0f}%)")
        
        # Buoy distance from shore statistics
        buoy_distances = results_df['buoy_distance_from_shore'].dropna()
        if len(buoy_distances) > 0:
            print(f"\n  Buoy distance from shore (pixels):")
            print(f"    Mean: {buoy_distances.mean():.1f}")
            print(f"    Median: {buoy_distances.median():.1f}")
            print(f"    Min: {buoy_distances.min():.1f}")
            print(f"    Max: {buoy_distances.max():.1f}")
            
            # Near shore buoys (< 5 pixels)
            near_shore = buoy_distances < 5
            if near_shore.sum() > 0:
                print(f"    Near shore (<5 px): {near_shore.sum()}/{len(buoy_distances)} ({100*near_shore.sum()/len(buoy_distances):.0f}%)")
    
    # Compare satellite CV (whole area) vs single pixel
    print("\n" + "-" * 70)
    print("HYPOTHESIS TEST")
    print("-" * 70)
    print("\nSatellite CV uses large cloud-masked areas (30-70% of lake)")
    print("If DINEOF wins via spatial smoothness, single-pixel should differ:")
    
    if center_valid.sum() > 0:
        print(f"\n  Satellite CV (area): ~97% DINEOF wins")
        print(f"  Center pixel:        {100*dineof_center/n_center:.0f}% DINEOF wins")
        
        if dineof_center/n_center < 0.90:
            print("\n  --> Single pixel shows REDUCED DINEOF dominance!")
            print("      Consistent with spatial smoothness hypothesis")
        else:
            print("\n  --> Single pixel still shows DINEOF dominance")
            print("      Spatial smoothness hypothesis NOT supported")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
