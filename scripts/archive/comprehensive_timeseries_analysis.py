#!/usr/bin/env python3
"""
Comprehensive Timeseries Analysis
==================================

Produces two main outputs:

1. INSITU VALIDATION ANALYSIS (insitu_validation_comprehensive.csv)
   - For each lake with in-situ buoy data
   - Buoy distance to shore (degrees and pixels)
   - Full stats comparing reconstruction vs in-situ ground truth
   - Full stats comparing reconstruction vs satellite ground truth (same pixel)
   - Winners for each metric under both ground truths

2. THREE-PIXEL ROUGHNESS ANALYSIS (pixel_roughness_analysis.csv)
   - For all lakes at center, shore, buoy pixels
   - Roughness metrics on reconstruction (STD, lag-1 autocorrelation)
   - Comparison metrics vs satellite ground truth (RMSE, MAE, Bias, etc.)

Uses existing outputs from insitu_validation.py in insitu_cv_validation/ folders.

Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Tuple, Optional
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


# =============================================================================
# STATISTICAL METRICS
# =============================================================================

def compute_full_stats(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics between prediction and observation.
    
    Returns dict with: rmse, mae, bias, median_error, std, rstd, n_samples
    """
    valid = np.isfinite(pred) & np.isfinite(obs)
    n = valid.sum()
    
    if n < 5:
        return {
            'rmse': np.nan, 'mae': np.nan, 'bias': np.nan,
            'median_error': np.nan, 'std': np.nan, 'rstd': np.nan,
            'n_samples': int(n)
        }
    
    errors = pred[valid] - obs[valid]
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)
    median_error = np.median(errors)
    std = np.std(errors, ddof=1)
    
    # Robust STD using MAD (Median Absolute Deviation)
    mad = np.median(np.abs(errors - np.median(errors)))
    rstd = 1.4826 * mad  # Scale factor for normal distribution
    
    return {
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'median_error': median_error,
        'std': std,
        'rstd': rstd,
        'n_samples': int(n)
    }


def compute_roughness_stats(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Compute roughness characterization metrics on a timeseries.
    
    Returns dict with: std, lag1_autocorr, range, iqr
    """
    valid = np.isfinite(timeseries)
    n = valid.sum()
    
    if n < 10:
        return {
            'std': np.nan,
            'lag1_autocorr': np.nan,
            'range': np.nan,
            'iqr': np.nan,
            'n_valid': int(n)
        }
    
    ts = timeseries[valid]
    
    # Standard deviation
    std = np.std(ts, ddof=1)
    
    # Lag-1 autocorrelation
    if len(ts) > 1:
        ts_centered = ts - np.mean(ts)
        numerator = np.sum(ts_centered[:-1] * ts_centered[1:])
        denominator = np.sum(ts_centered**2)
        lag1_autocorr = numerator / denominator if denominator > 0 else np.nan
    else:
        lag1_autocorr = np.nan
    
    # Range
    range_val = np.max(ts) - np.min(ts)
    
    # Interquartile range
    iqr = np.percentile(ts, 75) - np.percentile(ts, 25)
    
    return {
        'std': std,
        'lag1_autocorr': lag1_autocorr,
        'range': range_val,
        'iqr': iqr,
        'n_valid': int(n)
    }


def determine_winner(val1: float, val2: float, lower_is_better: bool = True) -> str:
    """Determine winner between two values."""
    if np.isnan(val1) and np.isnan(val2):
        return 'tie'
    if np.isnan(val1):
        return 'dincae'
    if np.isnan(val2):
        return 'dineof'
    
    if lower_is_better:
        return 'dineof' if val1 < val2 else 'dincae'
    else:
        return 'dineof' if val1 > val2 else 'dincae'


# =============================================================================
# SPATIAL UTILITIES
# =============================================================================

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


def find_center_pixel(lake_mask: np.ndarray, distance_map: np.ndarray) -> Optional[Tuple[int, int]]:
    """Find pixel furthest from shore (lake center)."""
    if distance_map is None or not lake_mask.any():
        return None
    masked_distance = np.where(lake_mask, distance_map, -1)
    return np.unravel_index(np.argmax(masked_distance), masked_distance.shape)


def find_shore_pixel(lake_mask: np.ndarray, distance_map: np.ndarray, 
                     min_distance: int = 3) -> Optional[Tuple[int, int]]:
    """Find pixel near shore (with safety margin)."""
    if distance_map is None or not lake_mask.any():
        return None
    
    target_mask = lake_mask & (distance_map >= min_distance) & (distance_map <= min_distance + 2)
    if not target_mask.any():
        target_mask = lake_mask & (distance_map >= 1) & (distance_map <= 5)
    if not target_mask.any():
        return None
    
    indices = np.where(target_mask)
    return (indices[0][0], indices[1][0])


def find_nearest_pixel(lat_array: np.ndarray, lon_array: np.ndarray,
                       target_lat: float, target_lon: float) -> Tuple[int, int]:
    """Find nearest grid point to target coordinates."""
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    return np.unravel_index(np.argmin(distance), distance.shape)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_selection_csvs(csv_paths: List[str]) -> pd.DataFrame:
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


def extract_pixel_data(ds_dineof: xr.Dataset, ds_dincae: xr.Dataset, 
                       lat_idx: int, lon_idx: int,
                       verbose: bool = False) -> pd.DataFrame:
    """
    Extract reconstruction and satellite observation at a pixel.
    
    Uses 'lswt' for original satellite observations and 'temp_filled' for reconstruction.
    """
    times = pd.to_datetime(ds_dineof['time'].values)
    
    # Get reconstruction (temp_filled)
    dineof_recon = ds_dineof['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values.copy()
    dincae_recon = ds_dincae['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values.copy()
    
    # Get original satellite observation (lswt) - this is the ground truth
    # Try different possible variable names
    satellite_obs = None
    for var_name in ['lswt', 'lake_surface_water_temperature', 'sst']:
        if var_name in ds_dineof:
            satellite_obs = ds_dineof[var_name].isel(lat=lat_idx, lon=lon_idx).values.copy()
            if verbose:
                n_valid = np.sum(np.isfinite(satellite_obs))
                print(f"    Found {var_name}: {n_valid}/{len(satellite_obs)} valid")
            break
    
    if satellite_obs is None:
        if verbose:
            print(f"    WARNING: No satellite obs variable found. Available: {list(ds_dineof.data_vars)}")
        satellite_obs = np.full(len(times), np.nan)
    
    # Convert to Celsius if needed
    if np.nanmean(dineof_recon) > 100:
        dineof_recon = dineof_recon - 273.15
    if np.nanmean(dincae_recon) > 100:
        dincae_recon = dincae_recon - 273.15
    if np.nanmean(satellite_obs) > 100:
        satellite_obs = satellite_obs - 273.15
    
    # is_observed based on whether satellite_obs is valid
    is_observed = np.isfinite(satellite_obs).astype(int)
    
    return pd.DataFrame({
        'date': times,
        'dineof_recon': dineof_recon,
        'dincae_recon': dincae_recon,
        'satellite_obs': satellite_obs,
        'is_observed': is_observed,
    })


# =============================================================================
# ANALYSIS 1: IN-SITU VALIDATION COMPREHENSIVE
# =============================================================================

def collect_insitu_validation(run_root: str, alpha: str) -> pd.DataFrame:
    """
    Collect all in-situ validation results from existing insitu_cv_validation folders.
    
    Loads:
    - *_insitu_stats_site*.csv for aggregated stats
    - *_insitu_timeseries_site*.csv for satellite CV computation
    - *_insitu_metadata_site*.json for location info
    """
    post_dir = os.path.join(run_root, "post")
    results = []
    
    # Find all lake folders
    for lake_folder in os.listdir(post_dir):
        try:
            lake_id = int(lake_folder.lstrip('0') or '0')
            if lake_id <= 0:
                continue
        except:
            continue
        
        insitu_dir = os.path.join(post_dir, lake_folder, alpha, "insitu_cv_validation")
        if not os.path.exists(insitu_dir):
            continue
        
        # Find all sites for this lake
        stats_files = glob(os.path.join(insitu_dir, f"LAKE{lake_folder}_insitu_stats_site*.csv"))
        
        for stats_file in stats_files:
            # Extract site ID from filename
            basename = os.path.basename(stats_file)
            try:
                site_id = int(basename.split('_site')[1].split('.')[0])
            except:
                continue
            
            # Load stats file
            try:
                stats_df = pd.read_csv(stats_file)
            except:
                continue
            
            # Load metadata file
            metadata_file = os.path.join(insitu_dir, f"LAKE{lake_folder}_insitu_metadata_site{site_id}.json")
            metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            # Load timeseries file for satellite CV computation
            ts_file = os.path.join(insitu_dir, f"LAKE{lake_folder}_insitu_timeseries_site{site_id}.csv")
            ts_df = None
            if os.path.exists(ts_file):
                try:
                    ts_df = pd.read_csv(ts_file)
                except:
                    pass
            
            # Extract stats for dineof and dincae from stats_df
            # Stats file has columns: method, data_type, rmse, mae, bias, median, std, rstd, n_samples
            # We want data_type='reconstruction' which is reconstruction vs in-situ
            dineof_mask = (stats_df['method'] == 'dineof') & (stats_df['data_type'] == 'reconstruction')
            dincae_mask = (stats_df['method'] == 'dincae') & (stats_df['data_type'] == 'reconstruction')
            
            if not dineof_mask.any() or not dincae_mask.any():
                continue
            
            dineof_stats = stats_df[dineof_mask].iloc[0]
            dincae_stats = stats_df[dincae_mask].iloc[0]
            
            # Get location info from metadata
            buoy_lat = metadata.get('buoy_lat', np.nan)
            buoy_lon = metadata.get('buoy_lon', np.nan)
            pixel_lat = metadata.get('pixel_lat', np.nan)
            pixel_lon = metadata.get('pixel_lon', np.nan)
            buoy_distance_px = metadata.get('distance_from_shore_px', np.nan)
            
            # Compute distance in degrees
            if np.isfinite(buoy_lat) and np.isfinite(pixel_lat):
                buoy_distance_deg = np.sqrt((buoy_lat - pixel_lat)**2 + (buoy_lon - pixel_lon)**2)
            else:
                buoy_distance_deg = np.nan
            
            # Compute satellite CV stats from timeseries
            sat_dineof_stats = {'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 
                               'median_error': np.nan, 'std': np.nan, 'rstd': np.nan, 'n_samples': 0}
            sat_dincae_stats = sat_dineof_stats.copy()
            
            # Column names in timeseries CSV: satellite_obs_temp, dineof_recon_temp, dincae_recon_temp
            sat_col = 'satellite_obs_temp' if 'satellite_obs_temp' in ts_df.columns else 'satellite_obs'
            dineof_col = 'dineof_recon_temp' if 'dineof_recon_temp' in ts_df.columns else 'dineof_recon'
            dincae_col = 'dincae_recon_temp' if 'dincae_recon_temp' in ts_df.columns else 'dincae_recon'
            
            if ts_df is not None and sat_col in ts_df.columns:
                # Filter to rows where satellite observation exists
                sat_mask = ts_df[sat_col].notna()
                if sat_mask.sum() >= 5:
                    sat_dineof_stats = compute_full_stats(
                        ts_df.loc[sat_mask, dineof_col].values,
                        ts_df.loc[sat_mask, sat_col].values
                    )
                    sat_dincae_stats = compute_full_stats(
                        ts_df.loc[sat_mask, dincae_col].values,
                        ts_df.loc[sat_mask, sat_col].values
                    )
            
            # Build result row
            result = {
                'lake_id': lake_id,
                'site_id': site_id,
                'buoy_lat': buoy_lat,
                'buoy_lon': buoy_lon,
                'pixel_lat': pixel_lat,
                'pixel_lon': pixel_lon,
                'buoy_distance_deg': buoy_distance_deg,
                'buoy_distance_px': buoy_distance_px,
                
                # In-situ validation stats (from existing stats file)
                'insitu_n_samples': int(dineof_stats.get('n_matches', dineof_stats.get('n_samples', dineof_stats.get('n', 0)))),
                'insitu_dineof_rmse': dineof_stats['rmse'],
                'insitu_dincae_rmse': dincae_stats['rmse'],
                'insitu_dineof_mae': dineof_stats['mae'],
                'insitu_dincae_mae': dincae_stats['mae'],
                'insitu_dineof_bias': dineof_stats['bias'],
                'insitu_dincae_bias': dincae_stats['bias'],
                'insitu_dineof_median': dineof_stats['median'],
                'insitu_dincae_median': dincae_stats['median'],
                'insitu_dineof_std': dineof_stats['std'],
                'insitu_dincae_std': dincae_stats['std'],
                'insitu_dineof_rstd': dineof_stats['rstd'],
                'insitu_dincae_rstd': dincae_stats['rstd'],
                
                # In-situ winners
                'insitu_rmse_winner': determine_winner(dineof_stats['rmse'], dincae_stats['rmse']),
                'insitu_mae_winner': determine_winner(dineof_stats['mae'], dincae_stats['mae']),
                'insitu_bias_winner': determine_winner(abs(dineof_stats['bias']), abs(dincae_stats['bias'])),
                'insitu_median_winner': determine_winner(abs(dineof_stats['median']), abs(dincae_stats['median'])),
                'insitu_std_winner': determine_winner(dineof_stats['std'], dincae_stats['std']),
                'insitu_rstd_winner': determine_winner(dineof_stats['rstd'], dincae_stats['rstd']),
                
                # Satellite validation stats (at buoy pixel, computed from timeseries)
                'sat_n_samples': sat_dineof_stats['n_samples'],
                'sat_dineof_rmse': sat_dineof_stats['rmse'],
                'sat_dincae_rmse': sat_dincae_stats['rmse'],
                'sat_dineof_mae': sat_dineof_stats['mae'],
                'sat_dincae_mae': sat_dincae_stats['mae'],
                'sat_dineof_bias': sat_dineof_stats['bias'],
                'sat_dincae_bias': sat_dincae_stats['bias'],
                'sat_dineof_median': sat_dineof_stats['median_error'],
                'sat_dincae_median': sat_dincae_stats['median_error'],
                'sat_dineof_std': sat_dineof_stats['std'],
                'sat_dincae_std': sat_dincae_stats['std'],
                'sat_dineof_rstd': sat_dineof_stats['rstd'],
                'sat_dincae_rstd': sat_dincae_stats['rstd'],
                
                # Satellite winners
                'sat_rmse_winner': determine_winner(sat_dineof_stats['rmse'], sat_dincae_stats['rmse']),
                'sat_mae_winner': determine_winner(sat_dineof_stats['mae'], sat_dincae_stats['mae']),
                'sat_bias_winner': determine_winner(abs(sat_dineof_stats['bias']), abs(sat_dincae_stats['bias'])),
                'sat_median_winner': determine_winner(abs(sat_dineof_stats['median_error']), abs(sat_dincae_stats['median_error'])),
                'sat_std_winner': determine_winner(sat_dineof_stats['std'], sat_dincae_stats['std']),
                'sat_rstd_winner': determine_winner(sat_dineof_stats['rstd'], sat_dincae_stats['rstd']),
            }
            
            results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS 2: THREE-PIXEL ROUGHNESS ANALYSIS
# =============================================================================

def analyze_pixel_roughness(run_root: str, lake_id: int, alpha: str,
                            selection_df: Optional[pd.DataFrame],
                            shore_margin: int = 3,
                            verbose: bool = False) -> List[Dict]:
    """
    Analyze roughness and accuracy at center, shore, and buoy pixels.
    
    Returns list of result dicts (one per pixel location).
    """
    lake_str = f"{lake_id:09d}"
    post_dir = os.path.join(run_root, "post", lake_str, alpha)
    if not os.path.exists(post_dir):
        post_dir = os.path.join(run_root, "post", str(lake_id), alpha)
    
    if not os.path.exists(post_dir):
        return []
    
    dineof_files = glob(os.path.join(post_dir, "*_dineof.nc"))
    dincae_files = glob(os.path.join(post_dir, "*_dincae.nc"))
    
    if not dineof_files or not dincae_files:
        return []
    
    dineof_path = dineof_files[0]
    dincae_path = dincae_files[0]
    
    ds_dineof = xr.open_dataset(dineof_path)
    ds_dincae = xr.open_dataset(dincae_path)
    
    results = []
    
    try:
        lats = ds_dineof['lat'].values
        lons = ds_dineof['lon'].values
        
        # Check what variables are available
        has_lswt = 'lswt' in ds_dineof
        
        if verbose:
            print(f"  First lake diagnostics (Lake {lake_id}):")
            print(f"    Variables in dineof.nc: {list(ds_dineof.data_vars)}")
            print(f"    Has 'lswt': {has_lswt}")
        
        lake_mask = get_lake_mask(ds_dineof)
        if lake_mask is None or not lake_mask.any():
            return []
        
        distance_map = compute_distance_from_shore(lake_mask)
        
        # Define pixel locations to analyze
        pixel_locations = {}
        
        # Center pixel
        center_idx = find_center_pixel(lake_mask, distance_map)
        if center_idx is not None:
            pixel_locations['center'] = {
                'idx': center_idx,
                'lat': float(lats[center_idx[0]]),
                'lon': float(lons[center_idx[1]]),
                'distance_px': float(distance_map[center_idx]),
            }
        
        # Shore pixel
        shore_idx = find_shore_pixel(lake_mask, distance_map, shore_margin)
        if shore_idx is not None:
            pixel_locations['shore'] = {
                'idx': shore_idx,
                'lat': float(lats[shore_idx[0]]),
                'lon': float(lons[shore_idx[1]]),
                'distance_px': float(distance_map[shore_idx]),
            }
        
        # Buoy pixel (if available)
        if selection_df is not None:
            lake_sites = selection_df[selection_df['lake_id'] == lake_id]
            if not lake_sites.empty:
                site = lake_sites.iloc[0]
                buoy_lat = float(site['latitude'])
                buoy_lon = float(site['longitude'])
                buoy_idx = find_nearest_pixel(lats, lons, buoy_lat, buoy_lon)
                
                if lake_mask[buoy_idx]:
                    pixel_locations['buoy'] = {
                        'idx': buoy_idx,
                        'lat': float(lats[buoy_idx[0]]),
                        'lon': float(lons[buoy_idx[1]]),
                        'distance_px': float(distance_map[buoy_idx]),
                        'site_id': int(site['site_id']),
                    }
        
        # Analyze each pixel location
        for loc_name, loc_info in pixel_locations.items():
            idx = loc_info['idx']
            
            # Extract data using corrected function
            pixel_data = extract_pixel_data(ds_dineof, ds_dincae, idx[0], idx[1], 
                                           verbose=(verbose and loc_name == 'center'))
            
            # Roughness metrics on reconstruction (not residuals)
            dineof_roughness = compute_roughness_stats(pixel_data['dineof_recon'].values)
            dincae_roughness = compute_roughness_stats(pixel_data['dincae_recon'].values)
            
            # Comparison vs satellite ground truth (where obs exists)
            sat_mask = pixel_data['is_observed'] == 1
            if sat_mask.sum() >= 5:
                dineof_vs_sat = compute_full_stats(
                    pixel_data.loc[sat_mask, 'dineof_recon'].values,
                    pixel_data.loc[sat_mask, 'satellite_obs'].values
                )
                dincae_vs_sat = compute_full_stats(
                    pixel_data.loc[sat_mask, 'dincae_recon'].values,
                    pixel_data.loc[sat_mask, 'satellite_obs'].values
                )
            else:
                dineof_vs_sat = {k: np.nan for k in ['rmse', 'mae', 'bias', 'median_error', 'std', 'rstd']}
                dineof_vs_sat['n_samples'] = 0
                dincae_vs_sat = dineof_vs_sat.copy()
            
            result = {
                'lake_id': lake_id,
                'pixel_location': loc_name,
                'pixel_lat': loc_info['lat'],
                'pixel_lon': loc_info['lon'],
                'distance_from_shore_px': loc_info['distance_px'],
                'site_id': loc_info.get('site_id', np.nan),
                
                # Roughness metrics (on reconstruction timeseries)
                'dineof_recon_std': dineof_roughness['std'],
                'dincae_recon_std': dincae_roughness['std'],
                'dineof_lag1_autocorr': dineof_roughness['lag1_autocorr'],
                'dincae_lag1_autocorr': dincae_roughness['lag1_autocorr'],
                'dineof_recon_range': dineof_roughness['range'],
                'dincae_recon_range': dincae_roughness['range'],
                'dineof_recon_iqr': dineof_roughness['iqr'],
                'dincae_recon_iqr': dincae_roughness['iqr'],
                'n_valid_recon': dineof_roughness['n_valid'],
                
                # Comparison vs satellite ground truth
                'sat_n_samples': dineof_vs_sat['n_samples'],
                'sat_dineof_rmse': dineof_vs_sat['rmse'],
                'sat_dincae_rmse': dincae_vs_sat['rmse'],
                'sat_dineof_mae': dineof_vs_sat['mae'],
                'sat_dincae_mae': dincae_vs_sat['mae'],
                'sat_dineof_bias': dineof_vs_sat['bias'],
                'sat_dincae_bias': dincae_vs_sat['bias'],
                'sat_dineof_median': dineof_vs_sat['median_error'],
                'sat_dincae_median': dincae_vs_sat['median_error'],
                'sat_dineof_std': dineof_vs_sat['std'],
                'sat_dincae_std': dincae_vs_sat['std'],
                'sat_dineof_rstd': dineof_vs_sat['rstd'],
                'sat_dincae_rstd': dincae_vs_sat['rstd'],
                
                # Winners vs satellite
                'sat_rmse_winner': determine_winner(dineof_vs_sat['rmse'], dincae_vs_sat['rmse']),
                'sat_mae_winner': determine_winner(dineof_vs_sat['mae'], dincae_vs_sat['mae']),
                'sat_std_winner': determine_winner(dineof_vs_sat['std'], dincae_vs_sat['std']),
                
                # Roughness comparison (higher autocorr = smoother, lower std = smoother)
                'smoother_by_autocorr': determine_winner(
                    dineof_roughness['lag1_autocorr'], 
                    dincae_roughness['lag1_autocorr'],
                    lower_is_better=False  # Higher autocorr = smoother
                ),
                'smoother_by_std': determine_winner(
                    dineof_roughness['std'],
                    dincae_roughness['std'],
                    lower_is_better=True  # Lower std = less variable
                ),
            }
            
            results.append(result)
        
        return results
        
    finally:
        ds_dineof.close()
        ds_dincae.close()


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_summary_stats(insitu_df: pd.DataFrame, roughness_df: pd.DataFrame,
                           output_dir: str):
    """Generate summary statistics and save to text file."""
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("COMPREHENSIVE TIMESERIES ANALYSIS - SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # In-situ validation summary
    if len(insitu_df) > 0:
        summary_lines.append("-" * 80)
        summary_lines.append("IN-SITU VALIDATION ANALYSIS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total lake-site pairs with in-situ data: {len(insitu_df)}")
        summary_lines.append(f"Unique lakes: {insitu_df['lake_id'].nunique()}")
        summary_lines.append("")
        
        # Winner counts for in-situ
        summary_lines.append("In-situ ground truth winners:")
        for metric in ['rmse', 'mae', 'bias', 'median', 'std', 'rstd']:
            col = f'insitu_{metric}_winner'
            if col in insitu_df.columns:
                dineof_wins = (insitu_df[col] == 'dineof').sum()
                dincae_wins = (insitu_df[col] == 'dincae').sum()
                ties = (insitu_df[col] == 'tie').sum()
                summary_lines.append(f"  {metric.upper():8s} - DINEOF: {dineof_wins:3d}, DINCAE: {dincae_wins:3d}, Ties: {ties:3d}")
        
        summary_lines.append("")
        
        # Winner counts for satellite (at buoy pixel)
        summary_lines.append("Satellite ground truth (at buoy pixel) winners:")
        for metric in ['rmse', 'mae', 'bias', 'median', 'std', 'rstd']:
            col = f'sat_{metric}_winner'
            if col in insitu_df.columns:
                dineof_wins = (insitu_df[col] == 'dineof').sum()
                dincae_wins = (insitu_df[col] == 'dincae').sum()
                ties = (insitu_df[col] == 'tie').sum()
                summary_lines.append(f"  {metric.upper():8s} - DINEOF: {dineof_wins:3d}, DINCAE: {dincae_wins:3d}, Ties: {ties:3d}")
        
        summary_lines.append("")
        
        # Mean stats vs in-situ
        summary_lines.append("Mean statistics (vs in-situ):")
        summary_lines.append(f"  DINEOF RMSE: {insitu_df['insitu_dineof_rmse'].mean():.3f} ± {insitu_df['insitu_dineof_rmse'].std():.3f}")
        summary_lines.append(f"  DINCAE RMSE: {insitu_df['insitu_dincae_rmse'].mean():.3f} ± {insitu_df['insitu_dincae_rmse'].std():.3f}")
        summary_lines.append(f"  DINEOF MAE:  {insitu_df['insitu_dineof_mae'].mean():.3f} ± {insitu_df['insitu_dineof_mae'].std():.3f}")
        summary_lines.append(f"  DINCAE MAE:  {insitu_df['insitu_dincae_mae'].mean():.3f} ± {insitu_df['insitu_dincae_mae'].std():.3f}")
        summary_lines.append(f"  DINEOF Bias: {insitu_df['insitu_dineof_bias'].mean():.3f} ± {insitu_df['insitu_dineof_bias'].std():.3f}")
        summary_lines.append(f"  DINCAE Bias: {insitu_df['insitu_dincae_bias'].mean():.3f} ± {insitu_df['insitu_dincae_bias'].std():.3f}")
        
        summary_lines.append("")
        
        # Mean stats vs satellite
        valid_sat = insitu_df['sat_n_samples'] > 0
        if valid_sat.sum() > 0:
            summary_lines.append("Mean statistics (vs satellite, at buoy pixel):")
            summary_lines.append(f"  DINEOF RMSE: {insitu_df.loc[valid_sat, 'sat_dineof_rmse'].mean():.3f} ± {insitu_df.loc[valid_sat, 'sat_dineof_rmse'].std():.3f}")
            summary_lines.append(f"  DINCAE RMSE: {insitu_df.loc[valid_sat, 'sat_dincae_rmse'].mean():.3f} ± {insitu_df.loc[valid_sat, 'sat_dincae_rmse'].std():.3f}")
        
        summary_lines.append("")
    
    # Roughness analysis summary
    if len(roughness_df) > 0:
        summary_lines.append("-" * 80)
        summary_lines.append("THREE-PIXEL ROUGHNESS ANALYSIS")
        summary_lines.append("-" * 80)
        
        for loc in ['center', 'shore', 'buoy']:
            loc_df = roughness_df[roughness_df['pixel_location'] == loc]
            if len(loc_df) > 0:
                summary_lines.append(f"\n{loc.upper()} pixels (N={len(loc_df)}):")
                
                # Satellite CV winners
                dineof_wins = (loc_df['sat_rmse_winner'] == 'dineof').sum()
                dincae_wins = (loc_df['sat_rmse_winner'] == 'dincae').sum()
                summary_lines.append(f"  Satellite RMSE winner - DINEOF: {dineof_wins}, DINCAE: {dincae_wins}")
                
                dineof_wins = (loc_df['sat_mae_winner'] == 'dineof').sum()
                dincae_wins = (loc_df['sat_mae_winner'] == 'dincae').sum()
                summary_lines.append(f"  Satellite MAE winner  - DINEOF: {dineof_wins}, DINCAE: {dincae_wins}")
                
                # Mean stats vs satellite
                summary_lines.append(f"  Mean RMSE vs sat      - DINEOF: {loc_df['sat_dineof_rmse'].mean():.4f}, DINCAE: {loc_df['sat_dincae_rmse'].mean():.4f}")
                
                # Roughness stats
                summary_lines.append(f"  Mean lag-1 autocorr   - DINEOF: {loc_df['dineof_lag1_autocorr'].mean():.3f}, DINCAE: {loc_df['dincae_lag1_autocorr'].mean():.3f}")
                summary_lines.append(f"  Mean recon STD        - DINEOF: {loc_df['dineof_recon_std'].mean():.3f}, DINCAE: {loc_df['dincae_recon_std'].mean():.3f}")
                
                # Which is smoother
                smoother_dineof = (loc_df['smoother_by_autocorr'] == 'dineof').sum()
                smoother_dincae = (loc_df['smoother_by_autocorr'] == 'dincae').sum()
                summary_lines.append(f"  Smoother by autocorr  - DINEOF: {smoother_dineof}, DINCAE: {smoother_dincae}")
        
        summary_lines.append("")
    
    summary_lines.append("=" * 80)
    
    # Save summary
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print('\n'.join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive timeseries analysis for DINEOF vs DINCAE comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs:
  comprehensive_timeseries_analysis/
    insitu_validation_comprehensive.csv  - Full stats for lakes with buoy data
    pixel_roughness_analysis.csv         - Roughness metrics at 3 pixel locations
    analysis_summary.txt                 - Summary statistics
        """
    )
    
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--selection-csvs", nargs="+", default=None)
    parser.add_argument("--shore-margin", type=int, default=3, help="Min pixels from shore")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "comprehensive_timeseries_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    selection_csvs = args.selection_csvs or DEFAULT_SELECTION_CSVS
    
    print("=" * 80)
    print("COMPREHENSIVE TIMESERIES ANALYSIS")
    print("=" * 80)
    print(f"Run root: {args.run_root}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load selection CSVs for buoy locations (used for roughness analysis)
    selection_df = load_selection_csvs(selection_csvs)
    if not selection_df.empty:
        print(f"Loaded buoy locations for {selection_df['lake_id'].nunique()} lakes")
    else:
        print("No selection CSVs loaded")
        selection_df = None
    
    # Find all lakes
    post_dir = os.path.join(args.run_root, "post")
    if not os.path.exists(post_dir):
        print(f"ERROR: post directory not found at {post_dir}")
        return 1
    
    lake_ids = []
    for folder in os.listdir(post_dir):
        try:
            lake_id = int(folder.lstrip('0') or '0')
            if lake_id > 0:
                lake_ids.append(lake_id)
        except:
            continue
    
    lake_ids = sorted(lake_ids)
    print(f"Found {len(lake_ids)} lakes to process")
    
    # =========================================================================
    # ANALYSIS 1: IN-SITU VALIDATION (from existing files)
    # =========================================================================
    print("\n" + "-" * 80)
    print("ANALYSIS 1: In-situ validation (from existing insitu_cv_validation files)")
    print("-" * 80)
    
    insitu_df = collect_insitu_validation(args.run_root, args.alpha)
    
    if len(insitu_df) > 0:
        insitu_path = os.path.join(args.output_dir, "insitu_validation_comprehensive.csv")
        insitu_df.to_csv(insitu_path, index=False)
        print(f"✓ Saved {len(insitu_df)} lake-site pairs to: {insitu_path}")
        
        if args.verbose:
            for _, row in insitu_df.iterrows():
                print(f"  Lake {row['lake_id']} site {row['site_id']}: "
                      f"insitu_rmse={row['insitu_rmse_winner']}, sat_rmse={row['sat_rmse_winner']}")
    else:
        print("✗ No in-situ validation results found")
    
    # =========================================================================
    # ANALYSIS 2: THREE-PIXEL ROUGHNESS
    # =========================================================================
    print("\n" + "-" * 80)
    print("ANALYSIS 2: Three-pixel roughness analysis")
    print("-" * 80)
    
    roughness_results = []
    print(f"Processing {len(lake_ids)} lakes...")
    
    first_lake = True
    for i, lake_id in enumerate(lake_ids):
        results = analyze_pixel_roughness(
            args.run_root, lake_id, args.alpha,
            selection_df, shore_margin=args.shore_margin,
            verbose=(args.verbose and first_lake)
        )
        first_lake = False
        roughness_results.extend(results)
        
        if args.verbose and results:
            locs = [r['pixel_location'] for r in results]
            winners = [r['sat_rmse_winner'] for r in results]
            print(f"  Lake {lake_id}: {', '.join(f'{l}={w}' for l, w in zip(locs, winners))}")
    
    if roughness_results:
        roughness_df = pd.DataFrame(roughness_results)
        roughness_path = os.path.join(args.output_dir, "pixel_roughness_analysis.csv")
        roughness_df.to_csv(roughness_path, index=False)
        print(f"\n✓ Saved {len(roughness_df)} pixel analyses to: {roughness_path}")
    else:
        roughness_df = pd.DataFrame()
        print("\n✗ No roughness analysis results")
    
    # =========================================================================
    # GENERATE SUMMARY
    # =========================================================================
    print("\n" + "-" * 80)
    print("GENERATING SUMMARY")
    print("-" * 80)
    
    generate_summary_stats(insitu_df, roughness_df, args.output_dir)
    
    print(f"\n✓ All outputs saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
