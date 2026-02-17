#!/usr/bin/env python3
"""
Comprehensive Timeseries Analysis v2 - Using data_source Flag
==============================================================

Properly separates satellite validation into:
- Training fit (flag=1): Reconstruction vs satellite obs used in training
- CV quality (flag=2): Reconstruction vs satellite obs WITHHELD from training

This enables fair comparison between DINEOF and DINCAE by measuring:
1. How well each method reproduces observations it SAW (training fit)
2. How well each method fills genuine gaps (CV quality)
3. In-situ ground truth comparison (independent validation)

Output files:
- three_location_analysis.csv: Per-pixel analysis at center/shore/buoy locations
- analysis_summary.txt: Summary statistics and winner counts

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

# data_source flag values (from add_data_source_flag.py)
FLAG_TRUE_GAP = 0        # Originally missing (true gap)
FLAG_OBSERVED_SEEN = 1   # Observed and used in training
FLAG_CV_WITHHELD = 2     # CV point (withheld observation)
FLAG_NOT_RECONSTRUCTED = 255  # temp_filled is NaN


# =============================================================================
# STATISTICAL METRICS
# =============================================================================

def compute_full_stats(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics between prediction and observation.
    
    Returns dict with: rmse, mae, bias, median_error, std, rstd, corr, n_samples
    """
    valid = np.isfinite(pred) & np.isfinite(obs)
    n = valid.sum()
    
    if n < 3:
        return {
            'rmse': np.nan, 'mae': np.nan, 'bias': np.nan,
            'median_error': np.nan, 'std': np.nan, 'rstd': np.nan,
            'corr': np.nan, 'n_samples': int(n)
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
    
    # Correlation
    if n >= 5:
        corr = np.corrcoef(pred[valid], obs[valid])[0, 1]
    else:
        corr = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'median_error': median_error,
        'std': std,
        'rstd': rstd,
        'corr': corr,
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
    
    std = np.std(ts, ddof=1)
    
    # Lag-1 autocorrelation
    if len(ts) > 1:
        ts_centered = ts - np.mean(ts)
        numerator = np.sum(ts_centered[:-1] * ts_centered[1:])
        denominator = np.sum(ts_centered**2)
        lag1_autocorr = numerator / denominator if denominator > 0 else np.nan
    else:
        lag1_autocorr = np.nan
    
    range_val = np.max(ts) - np.min(ts)
    iqr = np.percentile(ts, 75) - np.percentile(ts, 25)
    
    return {
        'std': std,
        'lag1_autocorr': lag1_autocorr,
        'range': range_val,
        'iqr': iqr,
        'n_valid': int(n)
    }


def determine_winner(val1: float, val2: float, lower_is_better: bool = True) -> str:
    """Determine winner between two values (DINEOF=val1, DINCAE=val2)."""
    if np.isnan(val1) and np.isnan(val2):
        return 'tie'
    if np.isnan(val1):
        return 'dincae'
    if np.isnan(val2):
        return 'dineof'
    
    if lower_is_better:
        return 'dineof' if val1 < val2 else ('dincae' if val2 < val1 else 'tie')
    else:
        return 'dineof' if val1 > val2 else ('dincae' if val2 > val1 else 'tie')


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
    """Load and combine selection CSVs for buoy locations."""
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


def extract_pixel_data_with_flag(ds_dineof: xr.Dataset, ds_dincae: xr.Dataset, 
                                  lat_idx: int, lon_idx: int,
                                  verbose: bool = False) -> pd.DataFrame:
    """
    Extract reconstruction, satellite observation, and data_source flag at a pixel.
    
    Returns DataFrame with columns:
    - date, dineof_recon, dincae_recon, satellite_obs, data_source
    """
    times = pd.to_datetime(ds_dineof['time'].values)
    
    # Get reconstruction (temp_filled)
    dineof_recon = ds_dineof['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values.copy()
    dincae_recon = ds_dincae['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values.copy()
    
    # Get data_source flag - CRITICAL for proper validation
    data_source = None
    if 'data_source' in ds_dineof:
        data_source = ds_dineof['data_source'].isel(lat=lat_idx, lon=lon_idx).values.copy()
        if verbose:
            n_gap = (data_source == FLAG_TRUE_GAP).sum()
            n_seen = (data_source == FLAG_OBSERVED_SEEN).sum()
            n_cv = (data_source == FLAG_CV_WITHHELD).sum()
            print(f"    data_source flag: gap={n_gap}, seen={n_seen}, cv={n_cv}")
    else:
        if verbose:
            print(f"    WARNING: data_source flag not found in dataset!")
        data_source = np.full(len(times), 255, dtype=np.uint8)
    
    # Get original satellite observation (lake_surface_water_temperature or lswt)
    satellite_obs = None
    for var_name in ['lake_surface_water_temperature', 'lswt', 'sst']:
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
    
    return pd.DataFrame({
        'date': times,
        'dineof_recon': dineof_recon,
        'dincae_recon': dincae_recon,
        'satellite_obs': satellite_obs,
        'data_source': data_source,
    })


# =============================================================================
# IN-SITU VALIDATION RESULTS LOADING
# =============================================================================

def load_insitu_validation_results(run_root: str, alpha: str) -> pd.DataFrame:
    """
    Load in-situ validation results from insitu_cv_validation folders.
    
    Returns DataFrame with columns:
    - lake_id, site_id
    - insitu_dineof_rmse, insitu_dincae_rmse
    - insitu_rmse_winner
    """
    post_dir = os.path.join(run_root, "post")
    results = []
    
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
        
        # Find stats files
        stats_files = glob(os.path.join(insitu_dir, f"LAKE{lake_folder}_insitu_stats_site*.csv"))
        
        for stats_file in stats_files:
            basename = os.path.basename(stats_file)
            # Extract site ID - match site<digits>.csv
            import re
            match = re.search(r'_site(\d+)\.csv$', basename)
            if not match:
                continue
            site_id = int(match.group(1))
            
            try:
                stats_df = pd.read_csv(stats_file)
            except:
                continue
            
            # Get reconstruction stats (reconstruction vs in-situ)
            # data_type should be 'reconstruction' for the main comparison
            dineof_mask = (stats_df['method'] == 'dineof') & (stats_df['data_type'] == 'reconstruction')
            dincae_mask = (stats_df['method'] == 'dincae') & (stats_df['data_type'] == 'reconstruction')
            
            if not dineof_mask.any() or not dincae_mask.any():
                continue
            
            dineof_row = stats_df[dineof_mask].iloc[0]
            dincae_row = stats_df[dincae_mask].iloc[0]
            
            dineof_rmse = dineof_row['rmse']
            dincae_rmse = dincae_row['rmse']
            
            # Determine winner
            if np.isnan(dineof_rmse) or np.isnan(dincae_rmse):
                winner = 'tie'
            elif dineof_rmse < dincae_rmse:
                winner = 'dineof'
            elif dincae_rmse < dineof_rmse:
                winner = 'dincae'
            else:
                winner = 'tie'
            
            results.append({
                'lake_id': lake_id,
                'site_id': site_id,
                'insitu_dineof_rmse': dineof_rmse,
                'insitu_dincae_rmse': dincae_rmse,
                'insitu_dineof_mae': dineof_row.get('mae', np.nan),
                'insitu_dincae_mae': dincae_row.get('mae', np.nan),
                'insitu_dineof_bias': dineof_row.get('bias', np.nan),
                'insitu_dincae_bias': dincae_row.get('bias', np.nan),
                'insitu_n_matches': dineof_row.get('n_matches', dineof_row.get('n', 0)),
                'insitu_rmse_winner': winner,
            })
    
    return pd.DataFrame(results)


# =============================================================================
# THREE-LOCATION ANALYSIS (CENTER, SHORE, BUOY)
# =============================================================================

def analyze_three_locations(run_root: str, lake_id: int, alpha: str,
                            selection_df: Optional[pd.DataFrame],
                            shore_margin: int = 3,
                            verbose: bool = False) -> List[Dict]:
    """
    Analyze at center, shore, and buoy pixels using data_source flag.
    
    For each location, computes:
    - sat_seen_*: Stats at flag=1 points (training data fit)
    - sat_cv_*: Stats at flag=2 points (true CV validation)
    - roughness metrics on full reconstruction timeseries
    
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
        
        # Check for data_source flag
        has_data_source = 'data_source' in ds_dineof
        
        if verbose:
            print(f"  Lake {lake_id} diagnostics:")
            print(f"    Variables: {list(ds_dineof.data_vars)}")
            print(f"    Has data_source flag: {has_data_source}")
        
        if not has_data_source:
            print(f"  WARNING: Lake {lake_id} missing data_source flag - skipping")
            return []
        
        lake_mask = get_lake_mask(ds_dineof)
        if lake_mask is None or not lake_mask.any():
            return []
        
        distance_map = compute_distance_from_shore(lake_mask)
        
        # Define pixel locations
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
            
            # Extract data with flag
            pixel_data = extract_pixel_data_with_flag(
                ds_dineof, ds_dincae, idx[0], idx[1], 
                verbose=(verbose and loc_name == 'center')
            )
            
            # Get masks for each category
            flag = pixel_data['data_source'].values
            mask_seen = (flag == FLAG_OBSERVED_SEEN)  # Training data
            mask_cv = (flag == FLAG_CV_WITHHELD)      # Withheld for CV
            mask_gap = (flag == FLAG_TRUE_GAP)        # True gaps
            
            n_seen = mask_seen.sum()
            n_cv = mask_cv.sum()
            n_gap = mask_gap.sum()
            
            # Roughness metrics on full reconstruction
            dineof_roughness = compute_roughness_stats(pixel_data['dineof_recon'].values)
            dincae_roughness = compute_roughness_stats(pixel_data['dincae_recon'].values)
            
            # --- SATELLITE VALIDATION AT SEEN POINTS (flag=1) ---
            # This measures how well the method reproduced observations it SAW in training
            if n_seen >= 3:
                dineof_vs_sat_seen = compute_full_stats(
                    pixel_data.loc[mask_seen, 'dineof_recon'].values,
                    pixel_data.loc[mask_seen, 'satellite_obs'].values
                )
                dincae_vs_sat_seen = compute_full_stats(
                    pixel_data.loc[mask_seen, 'dincae_recon'].values,
                    pixel_data.loc[mask_seen, 'satellite_obs'].values
                )
            else:
                dineof_vs_sat_seen = {k: np.nan for k in ['rmse', 'mae', 'bias', 'median_error', 'std', 'rstd', 'corr']}
                dineof_vs_sat_seen['n_samples'] = int(n_seen)
                dincae_vs_sat_seen = dineof_vs_sat_seen.copy()
            
            # --- SATELLITE VALIDATION AT CV POINTS (flag=2) ---
            # This is the KEY metric - measures true gap-filling quality
            if n_cv >= 3:
                dineof_vs_sat_cv = compute_full_stats(
                    pixel_data.loc[mask_cv, 'dineof_recon'].values,
                    pixel_data.loc[mask_cv, 'satellite_obs'].values
                )
                dincae_vs_sat_cv = compute_full_stats(
                    pixel_data.loc[mask_cv, 'dincae_recon'].values,
                    pixel_data.loc[mask_cv, 'satellite_obs'].values
                )
            else:
                dineof_vs_sat_cv = {k: np.nan for k in ['rmse', 'mae', 'bias', 'median_error', 'std', 'rstd', 'corr']}
                dineof_vs_sat_cv['n_samples'] = int(n_cv)
                dincae_vs_sat_cv = dineof_vs_sat_cv.copy()
            
            result = {
                'lake_id': lake_id,
                'pixel_location': loc_name,
                'pixel_lat': loc_info['lat'],
                'pixel_lon': loc_info['lon'],
                'distance_from_shore_px': loc_info['distance_px'],
                'site_id': loc_info.get('site_id', np.nan),
                
                # Sample counts (CRITICAL for interpretation)
                'n_seen': int(n_seen),  # flag=1
                'n_cv': int(n_cv),      # flag=2
                'n_gap': int(n_gap),    # flag=0
                'n_total': int(n_seen + n_cv + n_gap),
                
                # === TRAINING FIT (flag=1): Reconstruction vs seen satellite obs ===
                'sat_seen_n': dineof_vs_sat_seen['n_samples'],
                'sat_seen_dineof_rmse': dineof_vs_sat_seen['rmse'],
                'sat_seen_dincae_rmse': dincae_vs_sat_seen['rmse'],
                'sat_seen_dineof_mae': dineof_vs_sat_seen['mae'],
                'sat_seen_dincae_mae': dincae_vs_sat_seen['mae'],
                'sat_seen_dineof_bias': dineof_vs_sat_seen['bias'],
                'sat_seen_dincae_bias': dincae_vs_sat_seen['bias'],
                'sat_seen_dineof_corr': dineof_vs_sat_seen['corr'],
                'sat_seen_dincae_corr': dincae_vs_sat_seen['corr'],
                
                # Training fit winners
                'sat_seen_rmse_winner': determine_winner(dineof_vs_sat_seen['rmse'], dincae_vs_sat_seen['rmse']),
                'sat_seen_mae_winner': determine_winner(dineof_vs_sat_seen['mae'], dincae_vs_sat_seen['mae']),
                'sat_seen_corr_winner': determine_winner(dineof_vs_sat_seen['corr'], dincae_vs_sat_seen['corr'], lower_is_better=False),
                
                # === CV QUALITY (flag=2): Reconstruction vs WITHHELD satellite obs ===
                # THIS IS THE KEY METRIC for gap-filling quality
                'sat_cv_n': dineof_vs_sat_cv['n_samples'],
                'sat_cv_dineof_rmse': dineof_vs_sat_cv['rmse'],
                'sat_cv_dincae_rmse': dincae_vs_sat_cv['rmse'],
                'sat_cv_dineof_mae': dineof_vs_sat_cv['mae'],
                'sat_cv_dincae_mae': dincae_vs_sat_cv['mae'],
                'sat_cv_dineof_bias': dineof_vs_sat_cv['bias'],
                'sat_cv_dincae_bias': dincae_vs_sat_cv['bias'],
                'sat_cv_dineof_corr': dineof_vs_sat_cv['corr'],
                'sat_cv_dincae_corr': dincae_vs_sat_cv['corr'],
                
                # CV quality winners
                'sat_cv_rmse_winner': determine_winner(dineof_vs_sat_cv['rmse'], dincae_vs_sat_cv['rmse']),
                'sat_cv_mae_winner': determine_winner(dineof_vs_sat_cv['mae'], dincae_vs_sat_cv['mae']),
                'sat_cv_corr_winner': determine_winner(dineof_vs_sat_cv['corr'], dincae_vs_sat_cv['corr'], lower_is_better=False),
                
                # === ROUGHNESS metrics (on full reconstruction) ===
                'dineof_recon_std': dineof_roughness['std'],
                'dincae_recon_std': dincae_roughness['std'],
                'dineof_lag1_autocorr': dineof_roughness['lag1_autocorr'],
                'dincae_lag1_autocorr': dincae_roughness['lag1_autocorr'],
                'n_valid_recon': dineof_roughness['n_valid'],
                
                # Roughness comparison
                'smoother_by_autocorr': determine_winner(
                    dineof_roughness['lag1_autocorr'], 
                    dincae_roughness['lag1_autocorr'],
                    lower_is_better=False
                ),
                'smoother_by_std': determine_winner(
                    dineof_roughness['std'],
                    dincae_roughness['std'],
                    lower_is_better=True
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

def generate_summary_stats(results_df: pd.DataFrame, output_dir: str):
    """Generate summary statistics by location type."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("THREE-LOCATION ANALYSIS SUMMARY (Using data_source Flag)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Key distinction:")
    lines.append("  sat_seen_* : Reconstruction vs satellite obs USED in training (flag=1)")
    lines.append("  sat_cv_*   : Reconstruction vs satellite obs WITHHELD from training (flag=2)")
    lines.append("")
    lines.append("sat_cv metrics are the TRUE measure of gap-filling quality!")
    lines.append("")
    
    for loc in ['center', 'shore', 'buoy']:
        loc_df = results_df[results_df['pixel_location'] == loc]
        if len(loc_df) == 0:
            continue
        
        lines.append("-" * 80)
        lines.append(f"{loc.upper()} PIXELS (N={len(loc_df)} lakes)")
        lines.append("-" * 80)
        
        # Sample size statistics
        lines.append(f"\nSample sizes per pixel (median [min-max]):")
        lines.append(f"  Seen (flag=1): {loc_df['n_seen'].median():.0f} [{loc_df['n_seen'].min()}-{loc_df['n_seen'].max()}]")
        lines.append(f"  CV (flag=2):   {loc_df['n_cv'].median():.0f} [{loc_df['n_cv'].min()}-{loc_df['n_cv'].max()}]")
        lines.append(f"  Gap (flag=0):  {loc_df['n_gap'].median():.0f} [{loc_df['n_gap'].min()}-{loc_df['n_gap'].max()}]")
        
        # CV validation results (THE KEY METRIC)
        valid_cv = loc_df['sat_cv_n'] >= 3
        n_valid_cv = valid_cv.sum()
        lines.append(f"\n=== CV VALIDATION (flag=2) - {n_valid_cv} lakes with ≥3 CV points ===")
        if n_valid_cv > 0:
            cv_df = loc_df[valid_cv]
            
            dineof_wins = (cv_df['sat_cv_rmse_winner'] == 'dineof').sum()
            dincae_wins = (cv_df['sat_cv_rmse_winner'] == 'dincae').sum()
            ties = (cv_df['sat_cv_rmse_winner'] == 'tie').sum()
            lines.append(f"  RMSE winner: DINEOF {dineof_wins}, DINCAE {dincae_wins}, Ties {ties}")
            
            dineof_wins = (cv_df['sat_cv_mae_winner'] == 'dineof').sum()
            dincae_wins = (cv_df['sat_cv_mae_winner'] == 'dincae').sum()
            ties = (cv_df['sat_cv_mae_winner'] == 'tie').sum()
            lines.append(f"  MAE winner:  DINEOF {dineof_wins}, DINCAE {dincae_wins}, Ties {ties}")
            
            lines.append(f"\n  Mean RMSE: DINEOF {cv_df['sat_cv_dineof_rmse'].mean():.4f} ± {cv_df['sat_cv_dineof_rmse'].std():.4f}")
            lines.append(f"             DINCAE {cv_df['sat_cv_dincae_rmse'].mean():.4f} ± {cv_df['sat_cv_dincae_rmse'].std():.4f}")
            lines.append(f"  Mean Bias: DINEOF {cv_df['sat_cv_dineof_bias'].mean():.4f}")
            lines.append(f"             DINCAE {cv_df['sat_cv_dincae_bias'].mean():.4f}")
        else:
            lines.append(f"  No lakes with sufficient CV points at {loc} pixel")
        
        # Training fit results (for comparison)
        valid_seen = loc_df['sat_seen_n'] >= 3
        n_valid_seen = valid_seen.sum()
        lines.append(f"\n=== TRAINING FIT (flag=1) - {n_valid_seen} lakes with ≥3 seen points ===")
        if n_valid_seen > 0:
            seen_df = loc_df[valid_seen]
            
            dineof_wins = (seen_df['sat_seen_rmse_winner'] == 'dineof').sum()
            dincae_wins = (seen_df['sat_seen_rmse_winner'] == 'dincae').sum()
            ties = (seen_df['sat_seen_rmse_winner'] == 'tie').sum()
            lines.append(f"  RMSE winner: DINEOF {dineof_wins}, DINCAE {dincae_wins}, Ties {ties}")
            
            lines.append(f"\n  Mean RMSE: DINEOF {seen_df['sat_seen_dineof_rmse'].mean():.4f} ± {seen_df['sat_seen_dineof_rmse'].std():.4f}")
            lines.append(f"             DINCAE {seen_df['sat_seen_dincae_rmse'].mean():.4f} ± {seen_df['sat_seen_dincae_rmse'].std():.4f}")
        
        # Roughness comparison
        lines.append(f"\n=== ROUGHNESS ===")
        smoother_dineof = (loc_df['smoother_by_autocorr'] == 'dineof').sum()
        smoother_dincae = (loc_df['smoother_by_autocorr'] == 'dincae').sum()
        lines.append(f"  Smoother (by autocorr): DINEOF {smoother_dineof}, DINCAE {smoother_dincae}")
        lines.append(f"  Mean lag-1 autocorr: DINEOF {loc_df['dineof_lag1_autocorr'].mean():.3f}, DINCAE {loc_df['dincae_lag1_autocorr'].mean():.3f}")
        lines.append(f"  Mean recon STD:      DINEOF {loc_df['dineof_recon_std'].mean():.3f}, DINCAE {loc_df['dincae_recon_std'].mean():.3f}")
        
        lines.append("")
    
    # Overall summary
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    
    # Aggregate CV results across all locations
    valid_cv_all = results_df['sat_cv_n'] >= 3
    if valid_cv_all.sum() > 0:
        cv_all = results_df[valid_cv_all]
        lines.append(f"\nAcross all locations ({valid_cv_all.sum()} pixel-lake combinations with CV data):")
        
        dineof_wins = (cv_all['sat_cv_rmse_winner'] == 'dineof').sum()
        dincae_wins = (cv_all['sat_cv_rmse_winner'] == 'dincae').sum()
        lines.append(f"  CV RMSE winner: DINEOF {dineof_wins} ({100*dineof_wins/len(cv_all):.1f}%), "
                    f"DINCAE {dincae_wins} ({100*dincae_wins/len(cv_all):.1f}%)")
    
    lines.append("")
    lines.append("=" * 80)
    
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print('\n'.join(lines))
    print(f"\nSummary saved to: {summary_path}")


def generate_lake_by_lake_comparison(results_df: pd.DataFrame, 
                                      insitu_df: pd.DataFrame, 
                                      output_dir: str):
    """
    Generate lake-by-lake comparison of:
    - Smoothness winner (by autocorr at buoy pixel)
    - Satellite CV winner (flag=2 at buoy pixel)
    - In-situ validation winner
    
    Creates a summary table showing if these winners correspond.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("LAKE-BY-LAKE COMPARISON: SMOOTHNESS vs SATELLITE CV vs IN-SITU VALIDATION")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Question: Does smoothness winner correspond to in-situ validation winner?")
    lines.append("")
    
    # Get buoy pixel results only
    buoy_df = results_df[results_df['pixel_location'] == 'buoy'].copy()
    
    if len(buoy_df) == 0:
        lines.append("No buoy pixel data available.")
        return
    
    # Merge with in-situ validation results
    # Take first site per lake if multiple sites
    if len(insitu_df) > 0:
        insitu_first = insitu_df.groupby('lake_id').first().reset_index()
        merged = buoy_df.merge(insitu_first, on='lake_id', how='left')
    else:
        merged = buoy_df.copy()
        merged['insitu_rmse_winner'] = np.nan
        merged['insitu_dineof_rmse'] = np.nan
        merged['insitu_dincae_rmse'] = np.nan
        merged['insitu_n_matches'] = np.nan
    
    # Create comparison table
    lines.append("-" * 100)
    lines.append(f"{'Lake':>6} | {'Smoother':^10} | {'Sat CV':^10} | {'In-situ':^10} | {'Smooth=InSitu?':^14} | {'Notes'}")
    lines.append("-" * 100)
    
    comparison_results = []
    
    for _, row in merged.sort_values('lake_id').iterrows():
        lake_id = int(row['lake_id'])
        
        smoother = row.get('smoother_by_autocorr', 'N/A')
        sat_cv = row.get('sat_cv_rmse_winner', 'N/A')
        insitu = row.get('insitu_rmse_winner', 'N/A')
        
        # Check if smoother matches in-situ winner
        if pd.isna(insitu) or insitu == 'N/A':
            match = 'NO DATA'
        elif smoother == insitu:
            match = 'YES'
        else:
            match = 'NO'
        
        # Notes
        notes = []
        n_cv = row.get('sat_cv_n', 0)
        if n_cv < 10:
            notes.append(f"CV n={int(n_cv)}")
        
        n_insitu = row.get('insitu_n_matches', 0)
        if pd.notna(n_insitu) and n_insitu > 0:
            notes.append(f"insitu n={int(n_insitu)}")
        
        # Autocorrelation values
        dineof_ac = row.get('dineof_lag1_autocorr', np.nan)
        dincae_ac = row.get('dincae_lag1_autocorr', np.nan)
        if pd.notna(dineof_ac) and pd.notna(dincae_ac):
            ac_diff = dineof_ac - dincae_ac
            notes.append(f"Δautocorr={ac_diff:+.3f}")
        
        notes_str = ', '.join(notes) if notes else ''
        
        lines.append(f"{lake_id:>6} | {smoother:^10} | {sat_cv:^10} | {insitu:^10} | {match:^14} | {notes_str}")
        
        comparison_results.append({
            'lake_id': lake_id,
            'smoother_winner': smoother,
            'sat_cv_winner': sat_cv,
            'insitu_winner': insitu,
            'smoother_matches_insitu': match,
            'dineof_autocorr': dineof_ac,
            'dincae_autocorr': dincae_ac,
            'sat_cv_n': n_cv,
            'insitu_n': n_insitu,
        })
    
    lines.append("-" * 100)
    
    # Summary statistics
    comp_df = pd.DataFrame(comparison_results)
    valid_matches = comp_df[comp_df['smoother_matches_insitu'].isin(['YES', 'NO'])]
    
    if len(valid_matches) > 0:
        n_yes = (valid_matches['smoother_matches_insitu'] == 'YES').sum()
        n_no = (valid_matches['smoother_matches_insitu'] == 'NO').sum()
        n_total = len(valid_matches)
        
        lines.append("")
        lines.append("SUMMARY: Does smoothness predict in-situ winner?")
        lines.append(f"  Smoother = In-situ winner:  {n_yes}/{n_total} ({100*n_yes/n_total:.1f}%)")
        lines.append(f"  Smoother ≠ In-situ winner:  {n_no}/{n_total} ({100*n_no/n_total:.1f}%)")
        
        # Also check sat_cv vs insitu
        valid_cv = comp_df[(comp_df['sat_cv_winner'].isin(['dineof', 'dincae'])) & 
                          (comp_df['insitu_winner'].isin(['dineof', 'dincae']))]
        if len(valid_cv) > 0:
            cv_match = (valid_cv['sat_cv_winner'] == valid_cv['insitu_winner']).sum()
            lines.append("")
            lines.append("COMPARISON: Does satellite CV predict in-situ winner?")
            lines.append(f"  Sat CV = In-situ winner:  {cv_match}/{len(valid_cv)} ({100*cv_match/len(valid_cv):.1f}%)")
        
        # Which method wins more in each category
        lines.append("")
        lines.append("WINNER COUNTS:")
        for col, label in [('smoother_winner', 'Smoothness'), 
                           ('sat_cv_winner', 'Satellite CV'),
                           ('insitu_winner', 'In-situ')]:
            dineof_wins = (comp_df[col] == 'dineof').sum()
            dincae_wins = (comp_df[col] == 'dincae').sum()
            lines.append(f"  {label:15}: DINEOF {dineof_wins}, DINCAE {dincae_wins}")
    
    lines.append("")
    lines.append("=" * 100)
    
    # Save comparison table
    comp_path = os.path.join(output_dir, "lake_by_lake_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    
    summary_path = os.path.join(output_dir, "lake_by_lake_comparison.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print('\n'.join(lines))
    print(f"\nComparison saved to: {summary_path}")
    print(f"Comparison CSV: {comp_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three-location analysis using data_source flag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script properly separates satellite validation into:
- sat_seen_*: Reconstruction vs satellite obs USED in training (flag=1)
- sat_cv_*:   Reconstruction vs satellite obs WITHHELD from training (flag=2)

sat_cv metrics are the TRUE measure of gap-filling quality!

Outputs:
  three_location_analysis.csv - Per-pixel stats at center/shore/buoy
  analysis_summary.txt        - Summary statistics
        """
    )
    
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--selection-csvs", nargs="+", default=None)
    parser.add_argument("--shore-margin", type=int, default=3, help="Min pixels from shore")
    parser.add_argument("--lake-ids", nargs="+", type=int, default=None, help="Specific lake IDs")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "three_location_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    selection_csvs = args.selection_csvs or DEFAULT_SELECTION_CSVS
    
    print("=" * 80)
    print("THREE-LOCATION ANALYSIS (Using data_source Flag)")
    print("=" * 80)
    print(f"Run root: {args.run_root}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load buoy locations
    selection_df = load_selection_csvs(selection_csvs)
    if not selection_df.empty:
        print(f"Loaded buoy locations for {selection_df['lake_id'].nunique()} lakes")
    else:
        print("No selection CSVs loaded - buoy pixel analysis will be skipped")
        selection_df = None
    
    # Find lakes
    post_dir = os.path.join(args.run_root, "post")
    if not os.path.exists(post_dir):
        print(f"ERROR: post directory not found at {post_dir}")
        return 1
    
    if args.lake_ids:
        lake_ids = args.lake_ids
    else:
        lake_ids = []
        for folder in os.listdir(post_dir):
            try:
                lake_id = int(folder.lstrip('0') or '0')
                if lake_id > 0:
                    lake_ids.append(lake_id)
            except:
                continue
        lake_ids = sorted(lake_ids)
    
    print(f"Processing {len(lake_ids)} lakes...")
    
    # Run analysis
    all_results = []
    for i, lake_id in enumerate(lake_ids):
        if args.verbose or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(lake_ids)}] Lake {lake_id}")
        
        results = analyze_three_locations(
            args.run_root, lake_id, args.alpha,
            selection_df, shore_margin=args.shore_margin,
            verbose=args.verbose
        )
        all_results.extend(results)
    
    if not all_results:
        print("ERROR: No results generated. Check that output files have data_source flag.")
        return 1
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.output_dir, "three_location_analysis.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved {len(results_df)} pixel analyses to: {results_path}")
    
    # Generate summary
    print("\n" + "-" * 80)
    generate_summary_stats(results_df, args.output_dir)
    
    # Load in-situ validation results and generate lake-by-lake comparison
    print("\n" + "-" * 80)
    print("Loading in-situ validation results...")
    insitu_df = load_insitu_validation_results(args.run_root, args.alpha)
    
    if len(insitu_df) > 0:
        print(f"Found in-situ validation results for {insitu_df['lake_id'].nunique()} lakes")
        
        # Save in-situ summary
        insitu_path = os.path.join(args.output_dir, "insitu_validation_summary.csv")
        insitu_df.to_csv(insitu_path, index=False)
        print(f"✓ Saved in-situ summary to: {insitu_path}")
        
        # Generate lake-by-lake comparison
        print("\n" + "-" * 80)
        generate_lake_by_lake_comparison(results_df, insitu_df, args.output_dir)
    else:
        print("No in-situ validation results found in insitu_cv_validation folders")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())