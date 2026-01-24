#!/usr/bin/env python3
"""
Export Timeseries at Key Pixel Locations
=========================================

Exports reconstruction timeseries and satellite observations at three key
pixel locations for reproducibility and independent verification:

1. CENTER pixel - furthest from shore (lake center)
2. SHORE pixel - 3-5 pixels from shore
3. BUOY pixel - nearest pixel to buoy location (for lakes with in-situ data)

Outputs per lake:
- metadata.json - All metadata for reproducibility
- center_timeseries.csv - Timeseries at center pixel
- shore_timeseries.csv - Timeseries at shore pixel  
- buoy_timeseries.csv - Timeseries at buoy pixel (if available)

Also produces:
- lakes_summary.csv - Quick reference table for all lakes

Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

import argparse
import json
import os
import sys
from datetime import datetime
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
# DATA EXTRACTION
# =============================================================================

def extract_timeseries_at_pixel(ds: xr.Dataset, lat_idx: int, lon_idx: int,
                                 quality_threshold: int = 3) -> pd.DataFrame:
    """
    Extract all relevant timeseries at a pixel.
    
    Returns DataFrame with columns:
    - date
    - satellite_obs (NaN if no observation)
    - quality_level (NaN if no observation)
    - reconstruction value
    - is_observed (1 or 0)
    """
    # Get time axis
    times = pd.to_datetime(ds['time'].values)
    
    # Get reconstruction
    recon = ds['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values
    
    # Convert to Celsius if needed
    if np.nanmean(recon) > 100:
        recon = recon - 273.15
    
    # Get quality level to identify observations
    if 'quality_level' in ds:
        quality = ds['quality_level'].isel(lat=lat_idx, lon=lon_idx).values
        is_observed = (np.isfinite(quality) & (quality >= quality_threshold)).astype(int)
        # Satellite obs = reconstruction value where observed, NaN otherwise
        satellite_obs = np.where(is_observed == 1, recon, np.nan)
    else:
        quality = np.full(len(times), np.nan)
        is_observed = np.zeros(len(times), dtype=int)
        satellite_obs = np.full(len(times), np.nan)
    
    return pd.DataFrame({
        'date': times,
        'satellite_obs': satellite_obs,
        'quality_level': quality,
        'reconstruction': recon,
        'is_observed': is_observed,
    })


def get_time_range_from_attrs(ds: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    """Get config time range from dataset attributes."""
    start = ds.attrs.get('time_start_date', None)
    end = ds.attrs.get('time_end_date', None)
    return start, end


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


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def compute_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Compute RMSE between prediction and observation."""
    valid = np.isfinite(pred) & np.isfinite(obs)
    if valid.sum() < 5:
        return np.nan
    return np.sqrt(np.mean((pred[valid] - obs[valid])**2))


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def export_lake(run_root: str, lake_id: int, alpha: str,
                output_base: str, selection_df: Optional[pd.DataFrame],
                shore_margin: int = 3, quality_threshold: int = 3,
                verbose: bool = False) -> Optional[Dict]:
    """
    Export timeseries for a single lake.
    
    Creates:
    - {output_base}/LAKE{id}/metadata.json
    - {output_base}/LAKE{id}/center_timeseries.csv
    - {output_base}/LAKE{id}/shore_timeseries.csv
    - {output_base}/LAKE{id}/buoy_timeseries.csv (if buoy data available)
    
    Returns summary dict for lakes_summary.csv
    """
    # Find files
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
    
    dineof_path = dineof_files[0]
    dincae_path = dincae_files[0]
    
    # Create output directory
    lake_output_dir = os.path.join(output_base, f"LAKE{lake_str}")
    os.makedirs(lake_output_dir, exist_ok=True)
    
    # Open datasets
    ds_dineof = xr.open_dataset(dineof_path)
    ds_dincae = xr.open_dataset(dincae_path)
    
    try:
        # Get coordinates
        lats = ds_dineof['lat'].values
        lons = ds_dineof['lon'].values
        
        # Get time range from attributes
        time_start, time_end = get_time_range_from_attrs(ds_dineof)
        
        # Get actual timestamps
        times = pd.to_datetime(ds_dineof['time'].values)
        n_timestamps = len(times)
        actual_start = str(times[0])[:10]
        actual_end = str(times[-1])[:10]
        
        # Get lake mask and distance map
        lake_mask = get_lake_mask(ds_dineof)
        if lake_mask is None or not lake_mask.any():
            if verbose:
                print(f"  Lake {lake_id}: no valid lake mask")
            return None
        
        distance_map = compute_distance_from_shore(lake_mask)
        lake_pixels = int(lake_mask.sum())
        max_distance = float(distance_map[lake_mask].max()) if lake_mask.any() else 0
        
        # Initialize metadata
        metadata = {
            "generation_info": {
                "created": datetime.utcnow().isoformat() + "Z",
                "script": "export_pixel_timeseries.py",
                "experiment_root": run_root,
            },
            "lake_info": {
                "lake_id": lake_id,
                "lake_id_str": lake_str,
                "lake_pixels": lake_pixels,
                "max_distance_from_shore_px": max_distance,
            },
            "time_info": {
                "config_start_date": time_start,
                "config_end_date": time_end,
                "actual_start_date": actual_start,
                "actual_end_date": actual_end,
                "n_timestamps": n_timestamps,
                "note": "Timestamps are not daily - gaps exist where no satellite data"
            },
            "source_files": {
                "dineof_nc": os.path.basename(dineof_path),
                "dincae_nc": os.path.basename(dincae_path),
            },
            "units": {
                "temperature": "degrees_Celsius",
                "distance": "pixels",
                "coordinates": "degrees"
            },
            "pixels": {},
            "validation_metrics": {},
            "sample_counts": {},
        }
        
        # Summary dict for lakes_summary.csv
        summary = {
            'lake_id': lake_id,
            'lake_id_str': lake_str,
            'lake_pixels': lake_pixels,
            'max_distance_from_shore_px': max_distance,
            'n_timestamps': n_timestamps,
            'config_start_date': time_start,
            'config_end_date': time_end,
        }
        
        # =====================================================================
        # CENTER PIXEL
        # =====================================================================
        center_idx = find_center_pixel(lake_mask, distance_map)
        if center_idx is not None:
            center_lat = float(lats[center_idx[0]])
            center_lon = float(lons[center_idx[1]])
            center_dist = float(distance_map[center_idx])
            
            # Extract timeseries
            dineof_ts = extract_timeseries_at_pixel(ds_dineof, center_idx[0], center_idx[1], quality_threshold)
            dincae_ts = extract_timeseries_at_pixel(ds_dincae, center_idx[0], center_idx[1], quality_threshold)
            
            # Build combined CSV
            center_df = pd.DataFrame({
                'lake_id': lake_id,
                'pixel_location': f"{center_lat:.6f},{center_lon:.6f}",
                'pixel_indices': f"{center_idx[0]},{center_idx[1]}",
                'date': dineof_ts['date'],
                'satellite_obs': dineof_ts['satellite_obs'],
                'quality_level': dineof_ts['quality_level'],
                'dineof_recon': dineof_ts['reconstruction'],
                'dincae_recon': dincae_ts['reconstruction'],
                'is_observed': dineof_ts['is_observed'],
            })
            
            # Save CSV
            center_csv_path = os.path.join(lake_output_dir, "center_timeseries.csv")
            center_df.to_csv(center_csv_path, index=False)
            
            # Compute validation metrics (recon vs obs at observed times)
            obs_mask = dineof_ts['is_observed'] == 1
            n_observed = int(obs_mask.sum())
            n_missing = int((~obs_mask).sum())
            
            if n_observed >= 5:
                obs_vals = dineof_ts.loc[obs_mask, 'satellite_obs'].values
                dineof_at_obs = dineof_ts.loc[obs_mask, 'reconstruction'].values
                dincae_at_obs = dincae_ts.loc[obs_mask, 'reconstruction'].values
                
                dineof_rmse = compute_rmse(dineof_at_obs, obs_vals)
                dincae_rmse = compute_rmse(dincae_at_obs, obs_vals)
                winner = 'dineof' if dineof_rmse < dincae_rmse else 'dincae'
            else:
                dineof_rmse = np.nan
                dincae_rmse = np.nan
                winner = None
            
            # Update metadata
            metadata['pixels']['center'] = {
                'description': 'Pixel furthest from shore (lake center)',
                'lat_idx': int(center_idx[0]),
                'lon_idx': int(center_idx[1]),
                'lat_deg': center_lat,
                'lon_deg': center_lon,
                'pixel_location_str': f"{center_lat:.6f},{center_lon:.6f}",
                'pixel_indices_str': f"{center_idx[0]},{center_idx[1]}",
                'distance_from_shore_px': center_dist,
            }
            metadata['sample_counts']['center'] = {
                'n_observed': n_observed,
                'n_missing': n_missing,
                'n_total': n_timestamps,
            }
            metadata['validation_metrics']['center'] = {
                'dineof_rmse_vs_satobs': dineof_rmse if not np.isnan(dineof_rmse) else None,
                'dincae_rmse_vs_satobs': dincae_rmse if not np.isnan(dincae_rmse) else None,
                'winner': winner,
            }
            
            # Update summary
            summary['center_lat_deg'] = center_lat
            summary['center_lon_deg'] = center_lon
            summary['center_dist_px'] = center_dist
            summary['center_n_observed'] = n_observed
            summary['center_dineof_rmse'] = dineof_rmse
            summary['center_dincae_rmse'] = dincae_rmse
            summary['center_winner'] = winner
        
        # =====================================================================
        # SHORE PIXEL
        # =====================================================================
        shore_idx = find_shore_pixel(lake_mask, distance_map, shore_margin)
        if shore_idx is not None:
            shore_lat = float(lats[shore_idx[0]])
            shore_lon = float(lons[shore_idx[1]])
            shore_dist = float(distance_map[shore_idx])
            
            # Extract timeseries
            dineof_ts = extract_timeseries_at_pixel(ds_dineof, shore_idx[0], shore_idx[1], quality_threshold)
            dincae_ts = extract_timeseries_at_pixel(ds_dincae, shore_idx[0], shore_idx[1], quality_threshold)
            
            # Build combined CSV
            shore_df = pd.DataFrame({
                'lake_id': lake_id,
                'pixel_location': f"{shore_lat:.6f},{shore_lon:.6f}",
                'pixel_indices': f"{shore_idx[0]},{shore_idx[1]}",
                'date': dineof_ts['date'],
                'satellite_obs': dineof_ts['satellite_obs'],
                'quality_level': dineof_ts['quality_level'],
                'dineof_recon': dineof_ts['reconstruction'],
                'dincae_recon': dincae_ts['reconstruction'],
                'is_observed': dineof_ts['is_observed'],
            })
            
            # Save CSV
            shore_csv_path = os.path.join(lake_output_dir, "shore_timeseries.csv")
            shore_df.to_csv(shore_csv_path, index=False)
            
            # Compute validation metrics
            obs_mask = dineof_ts['is_observed'] == 1
            n_observed = int(obs_mask.sum())
            n_missing = int((~obs_mask).sum())
            
            if n_observed >= 5:
                obs_vals = dineof_ts.loc[obs_mask, 'satellite_obs'].values
                dineof_at_obs = dineof_ts.loc[obs_mask, 'reconstruction'].values
                dincae_at_obs = dincae_ts.loc[obs_mask, 'reconstruction'].values
                
                dineof_rmse = compute_rmse(dineof_at_obs, obs_vals)
                dincae_rmse = compute_rmse(dincae_at_obs, obs_vals)
                winner = 'dineof' if dineof_rmse < dincae_rmse else 'dincae'
            else:
                dineof_rmse = np.nan
                dincae_rmse = np.nan
                winner = None
            
            # Update metadata
            metadata['pixels']['shore'] = {
                'description': f'Pixel {shore_margin}-{shore_margin+2} pixels from shore',
                'lat_idx': int(shore_idx[0]),
                'lon_idx': int(shore_idx[1]),
                'lat_deg': shore_lat,
                'lon_deg': shore_lon,
                'pixel_location_str': f"{shore_lat:.6f},{shore_lon:.6f}",
                'pixel_indices_str': f"{shore_idx[0]},{shore_idx[1]}",
                'distance_from_shore_px': shore_dist,
            }
            metadata['sample_counts']['shore'] = {
                'n_observed': n_observed,
                'n_missing': n_missing,
                'n_total': n_timestamps,
            }
            metadata['validation_metrics']['shore'] = {
                'dineof_rmse_vs_satobs': dineof_rmse if not np.isnan(dineof_rmse) else None,
                'dincae_rmse_vs_satobs': dincae_rmse if not np.isnan(dincae_rmse) else None,
                'winner': winner,
            }
            
            # Update summary
            summary['shore_lat_deg'] = shore_lat
            summary['shore_lon_deg'] = shore_lon
            summary['shore_dist_px'] = shore_dist
            summary['shore_n_observed'] = n_observed
            summary['shore_dineof_rmse'] = dineof_rmse
            summary['shore_dincae_rmse'] = dincae_rmse
            summary['shore_winner'] = winner
        
        # =====================================================================
        # BUOY PIXEL (if available)
        # =====================================================================
        if selection_df is not None:
            lake_sites = selection_df[selection_df['lake_id'] == lake_id]
            if not lake_sites.empty:
                site = lake_sites.iloc[0]
                site_id = int(site['site_id'])
                buoy_lat = float(site['latitude'])
                buoy_lon = float(site['longitude'])
                
                buoy_idx = find_nearest_pixel(lats, lons, buoy_lat, buoy_lon)
                pixel_lat = float(lats[buoy_idx[0]])
                pixel_lon = float(lons[buoy_idx[1]])
                buoy_dist = float(distance_map[buoy_idx]) if distance_map is not None else np.nan
                buoy_on_lake = bool(lake_mask[buoy_idx])
                
                # Extract timeseries
                dineof_ts = extract_timeseries_at_pixel(ds_dineof, buoy_idx[0], buoy_idx[1], quality_threshold)
                dincae_ts = extract_timeseries_at_pixel(ds_dincae, buoy_idx[0], buoy_idx[1], quality_threshold)
                
                # Build combined CSV
                buoy_df = pd.DataFrame({
                    'lake_id': lake_id,
                    'site_id': site_id,
                    'pixel_location': f"{pixel_lat:.6f},{pixel_lon:.6f}",
                    'pixel_indices': f"{buoy_idx[0]},{buoy_idx[1]}",
                    'date': dineof_ts['date'],
                    'satellite_obs': dineof_ts['satellite_obs'],
                    'quality_level': dineof_ts['quality_level'],
                    'dineof_recon': dineof_ts['reconstruction'],
                    'dincae_recon': dincae_ts['reconstruction'],
                    'is_observed': dineof_ts['is_observed'],
                })
                
                # Save CSV
                buoy_csv_path = os.path.join(lake_output_dir, f"buoy_site{site_id}_timeseries.csv")
                buoy_df.to_csv(buoy_csv_path, index=False)
                
                # Compute validation metrics
                obs_mask = dineof_ts['is_observed'] == 1
                n_observed = int(obs_mask.sum())
                n_missing = int((~obs_mask).sum())
                
                if n_observed >= 5:
                    obs_vals = dineof_ts.loc[obs_mask, 'satellite_obs'].values
                    dineof_at_obs = dineof_ts.loc[obs_mask, 'reconstruction'].values
                    dincae_at_obs = dincae_ts.loc[obs_mask, 'reconstruction'].values
                    
                    dineof_rmse = compute_rmse(dineof_at_obs, obs_vals)
                    dincae_rmse = compute_rmse(dincae_at_obs, obs_vals)
                    winner = 'dineof' if dineof_rmse < dincae_rmse else 'dincae'
                else:
                    dineof_rmse = np.nan
                    dincae_rmse = np.nan
                    winner = None
                
                # Update metadata
                metadata['pixels']['buoy'] = {
                    'description': 'Nearest pixel to buoy location',
                    'site_id': site_id,
                    'buoy_lat_deg': buoy_lat,
                    'buoy_lon_deg': buoy_lon,
                    'pixel_lat_idx': int(buoy_idx[0]),
                    'pixel_lon_idx': int(buoy_idx[1]),
                    'pixel_lat_deg': pixel_lat,
                    'pixel_lon_deg': pixel_lon,
                    'pixel_location_str': f"{pixel_lat:.6f},{pixel_lon:.6f}",
                    'pixel_indices_str': f"{buoy_idx[0]},{buoy_idx[1]}",
                    'distance_from_shore_px': buoy_dist,
                    'buoy_on_lake': buoy_on_lake,
                }
                metadata['sample_counts']['buoy'] = {
                    'n_observed': n_observed,
                    'n_missing': n_missing,
                    'n_total': n_timestamps,
                }
                metadata['validation_metrics']['buoy'] = {
                    'dineof_rmse_vs_satobs': dineof_rmse if not np.isnan(dineof_rmse) else None,
                    'dincae_rmse_vs_satobs': dincae_rmse if not np.isnan(dincae_rmse) else None,
                    'winner': winner,
                }
                
                # Update summary
                summary['buoy_site_id'] = site_id
                summary['buoy_lat_deg'] = buoy_lat
                summary['buoy_lon_deg'] = buoy_lon
                summary['buoy_pixel_lat_deg'] = pixel_lat
                summary['buoy_pixel_lon_deg'] = pixel_lon
                summary['buoy_dist_px'] = buoy_dist
                summary['buoy_n_observed'] = n_observed
                summary['buoy_dineof_rmse'] = dineof_rmse
                summary['buoy_dincae_rmse'] = dincae_rmse
                summary['buoy_winner'] = winner
        
        # Save metadata JSON
        metadata_path = os.path.join(lake_output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            center_win = metadata['validation_metrics'].get('center', {}).get('winner', '?')
            shore_win = metadata['validation_metrics'].get('shore', {}).get('winner', '?')
            buoy_win = metadata['validation_metrics'].get('buoy', {}).get('winner', '?')
            print(f"  Lake {lake_id}: center={center_win}, shore={shore_win}, buoy={buoy_win}")
        
        return summary
        
    finally:
        ds_dineof.close()
        ds_dincae.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export timeseries at key pixel locations for reproducibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exports timeseries at three key pixel locations:
- CENTER: Pixel furthest from shore
- SHORE: Pixel 3-5 pixels from shore
- BUOY: Nearest pixel to buoy (for lakes with in-situ data)

Output structure:
  timeseries_pixels/
    LAKE000000015/
      metadata.json
      center_timeseries.csv
      shore_timeseries.csv
      buoy_site1_timeseries.csv
    lakes_summary.csv
        """
    )
    
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--shore-margin", type=int, default=3, help="Min pixels from shore")
    parser.add_argument("--selection-csvs", nargs="+", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "timeseries_pixels")
    os.makedirs(args.output_dir, exist_ok=True)
    
    selection_csvs = args.selection_csvs or DEFAULT_SELECTION_CSVS
    
    print("=" * 70)
    print("EXPORT PIXEL TIMESERIES")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Load selection CSVs for buoy locations
    selection_df = load_selection_csvs(selection_csvs)
    if not selection_df.empty:
        print(f"Loaded buoy locations for {selection_df['lake_id'].nunique()} lakes")
    else:
        print("No selection CSVs loaded - buoy pixels will not be exported")
        selection_df = None
    
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
    print(f"Found {len(lake_ids)} lakes to process")
    
    # Export each lake
    summaries = []
    for lake_id in lake_ids:
        summary = export_lake(
            args.run_root, lake_id, args.alpha,
            args.output_dir, selection_df,
            args.shore_margin, verbose=args.verbose
        )
        if summary is not None:
            summaries.append(summary)
    
    print(f"\nSuccessfully exported {len(summaries)} lakes")
    
    # Save summary CSV
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(args.output_dir, "lakes_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if 'center_winner' in summary_df:
            center_valid = summary_df['center_winner'].notna()
            if center_valid.sum() > 0:
                dineof_wins = (summary_df.loc[center_valid, 'center_winner'] == 'dineof').sum()
                n = center_valid.sum()
                print(f"CENTER: DINEOF wins {dineof_wins}/{n} ({100*dineof_wins/n:.0f}%)")
        
        if 'shore_winner' in summary_df:
            shore_valid = summary_df['shore_winner'].notna()
            if shore_valid.sum() > 0:
                dineof_wins = (summary_df.loc[shore_valid, 'shore_winner'] == 'dineof').sum()
                n = shore_valid.sum()
                print(f"SHORE:  DINEOF wins {dineof_wins}/{n} ({100*dineof_wins/n:.0f}%)")
        
        if 'buoy_winner' in summary_df:
            buoy_valid = summary_df['buoy_winner'].notna()
            if buoy_valid.sum() > 0:
                dineof_wins = (summary_df.loc[buoy_valid, 'buoy_winner'] == 'dineof').sum()
                n = buoy_valid.sum()
                print(f"BUOY:   DINEOF wins {dineof_wins}/{n} ({100*dineof_wins/n:.0f}%)")
    
    print(f"\n✓ All outputs saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
