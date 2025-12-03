#!/usr/bin/env python3
"""
In-Situ Validation Script for Lake CCI Gap-Filling Pipeline

Compares gap-filled LSWT (DINEOF, DINCAE) against in-situ buoy measurements.

Usage:
    python insitu_validation.py --config config.json
    python insitu_validation.py --lake-id 2 --site-id 1 --year 2010

Author: Shaerdan / NCEO / University of Reading
"""

import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================================
# CONFIGURATION DEFAULTS - Edit these to match your setup
# ============================================================================
DEFAULT_CONFIG = {
    "run_root": "/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/anomaly-20251126-c84211-exp1",
    "alpha_slug": "a1000",
    "buoy_dir": "/gws/ssde/j25b/nceo_uor/users/lcarrea01/INSITU/Buoy_Laura/ALL_FILES_QC",
    "selection_csv": "/home/users/shaerdan/general_purposes/insitu_cv/L3S_QL_MDB_2010_selection_converted.csv",
    "output_dir": "/home/users/shaerdan/general_purposes/insitu_cv/insitu_validation_outputs",
    "distance_threshold": 0.05,  # degrees (~5.5 km)
    "quality_threshold": 3,      # minimum quality level for satellite obs
}
# ============================================================================


class InsituValidator:
    """
    Validates gap-filled LSWT against in-situ buoy measurements.
    """
    
    def __init__(self, config: Dict):
        self.run_root = config["run_root"]
        self.alpha_slug = config.get("alpha_slug", "a1000")
        self.buoy_dir = config["buoy_dir"]
        self.selection_csv_path = config["selection_csv"]
        self.output_dir = config["output_dir"]
        self.distance_threshold = config.get("distance_threshold", 0.05)
        self.quality_threshold = config.get("quality_threshold", 3)
        self.plot_all_pixels = config.get("plot_all_pixels", False)  # Plot all lake pixels as spaghetti
        self.max_pixels_to_plot = config.get("max_pixels_to_plot", 500)  # Max pixels for spaghetti plot
        
        # Load selection CSV
        self.selection_df = self._load_selection_csv()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_selection_csv(self) -> pd.DataFrame:
        """Load and clean the selection CSV."""
        df = pd.read_csv(self.selection_csv_path)
        df.columns = df.columns.str.strip()  # Remove trailing spaces
        
        # Ensure time_IS is parsed as datetime
        time_col = [c for c in df.columns if 'time_IS' in c][0]
        if time_col != 'time_IS':
            df = df.rename(columns={time_col: 'time_IS'})
        df['time_IS'] = pd.to_datetime(df['time_IS'])
        
        return df
    
    def get_lake_site_pairs(self, lake_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Get unique (lake_id, lake_id_cci, site_id) pairs from selection CSV."""
        df = self.selection_df
        if lake_ids:
            df = df[df['lake_id'].isin(lake_ids)]
        return df[['lake_id', 'lake_id_cci', 'site_id']].drop_duplicates()
    
    def find_pipeline_outputs(self, lake_id_cci: int) -> Dict[str, Optional[str]]:
        """
        Find all pipeline output files for a lake.
        
        Returns dict with keys: dineof, dincae, eof_filtered, interp_full, eof_filtered_interp_full
        """
        lake_id_str = str(lake_id_cci).zfill(9)
        post_dir = os.path.join(self.run_root, 'post', lake_id_str, self.alpha_slug)
        
        outputs = {
            'dineof': None,
            'dincae': None,
            'eof_filtered': None,
            'interp_full': None,
            'eof_filtered_interp_full': None,
        }
        
        if not os.path.exists(post_dir):
            print(f"WARNING: Post directory not found: {post_dir}")
            return outputs
        
        nc_files = glob(os.path.join(post_dir, "*.nc"))
        
        for nc_file in nc_files:
            basename = os.path.basename(nc_file)
            if '_dincae.nc' in basename:
                outputs['dincae'] = nc_file
            elif '_dineof_eof_filtered_interp_full.nc' in basename:
                outputs['eof_filtered_interp_full'] = nc_file
            elif '_dineof_eof_filtered.nc' in basename:
                outputs['eof_filtered'] = nc_file
            elif '_dineof_eof_interp_full.nc' in basename:
                outputs['interp_full'] = nc_file
            elif '_dineof.nc' in basename:
                outputs['dineof'] = nc_file
        
        return outputs
    
    def get_buoy_filepath(self, lake_id: int, site_id: int) -> Optional[str]:
        """Construct the path to the buoy CSV file."""
        lake_padded = str(lake_id).zfill(6)
        site_padded = str(site_id).zfill(2)
        buoy_file = os.path.join(self.buoy_dir, f"ID{lake_padded}{site_padded}.csv")
        
        if os.path.exists(buoy_file):
            return buoy_file
        return None
    
    def determine_representative_hour(self, lake_id: int, site_id: int) -> Optional[int]:
        """
        Find the hour of day when satellite overpasses most commonly occur
        for this lake/site combination.
        """
        subset = self.selection_df[
            (self.selection_df['lake_id'] == lake_id) & 
            (self.selection_df['site_id'] == site_id)
        ].copy()
        
        if subset.empty:
            return None
        
        subset['time_floor'] = subset['time_IS'].dt.floor('H')
        subset['date'] = subset['time_IS'].dt.date
        subset['hour'] = subset['time_floor'].dt.hour
        
        # Count unique days per hour
        hour_day_counts = subset.groupby('hour')['date'].nunique()
        
        if hour_day_counts.empty:
            return None
        
        representative_hour = hour_day_counts.idxmax()
        print(f"  Representative hour for lake {lake_id}, site {site_id}: {representative_hour}:00 "
              f"({hour_day_counts.max()} unique days)")
        
        return representative_hour
    
    def load_buoy_data(self, lake_id: int, site_id: int, 
                       year_start: Optional[int], year_end: Optional[int],
                       representative_hour: Optional[int]) -> Optional[pd.DataFrame]:
        """
        Load and filter buoy CSV data.
        
        Args:
            lake_id: Lake ID for buoy file lookup
            site_id: Site ID for buoy file lookup
            year_start: Start year (None = no lower limit)
            year_end: End year (None = no upper limit)
            representative_hour: Hour to filter for (None = skip hour filtering, e.g., for daily data)
        """
        buoy_path = self.get_buoy_filepath(lake_id, site_id)
        if buoy_path is None:
            print(f"  Buoy file not found for lake {lake_id}, site {site_id}")
            return None
        
        try:
            df = pd.read_csv(buoy_path, parse_dates=['dateTime'])
        except Exception as e:
            print(f"  Error reading buoy file: {e}")
            return None
        
        initial_count = len(df)
        
        # Detect if data is daily (one reading per day) vs hourly
        readings_per_day = df.groupby(df['dateTime'].dt.date).size()
        median_readings_per_day = readings_per_day.median()
        is_daily_data = median_readings_per_day <= 1.5
        
        if is_daily_data:
            print(f"  Detected DAILY buoy data (median {median_readings_per_day:.1f} readings/day)")
        else:
            print(f"  Detected HOURLY buoy data (median {median_readings_per_day:.1f} readings/day)")
        
        # Filter for year range
        if year_start is not None or year_end is not None:
            years = df['dateTime'].dt.year
            if year_start is not None and year_end is not None:
                df = df[(years >= year_start) & (years <= year_end)]
                print(f"  After year range {year_start}-{year_end} filter: {len(df)} rows")
            elif year_start is not None:
                df = df[years >= year_start]
                print(f"  After year >= {year_start} filter: {len(df)} rows")
            else:
                df = df[years <= year_end]
                print(f"  After year <= {year_end} filter: {len(df)} rows")
        
        # Filter for representative hour (only for hourly data)
        if representative_hour is not None and not is_daily_data:
            df['hour'] = df['dateTime'].dt.hour
            df = df[df['hour'] == representative_hour]
            print(f"  After hour {representative_hour} filter: {len(df)} rows")
        else:
            if is_daily_data:
                print(f"  Skipping hour filter (daily data)")
        
        # Apply quality filter
        if 'qcFlag' in df.columns:
            df = df[df['qcFlag'] == 0]
        elif 'q' in df.columns:
            df = df[df['q'] == 0]
        quality_count = len(df)
        print(f"  After quality filter: {quality_count} rows")
        
        print(f"  Buoy data summary: {initial_count} total → {quality_count} after all filters")
        
        if df.empty:
            return None
        
        return df
    
    def find_nearest_grid_point(self, lat_array: np.ndarray, lon_array: np.ndarray,
                                 target_lat: float, target_lon: float) -> Tuple[Tuple[int, int], float]:
        """Find the nearest grid point to the target coordinates."""
        lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
        distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
        index = np.unravel_index(np.argmin(distance), distance.shape)
        min_distance = np.min(distance)
        return index, min_distance
    
    def extract_temperatures(self, ds: xr.Dataset, grid_idx: Tuple[int, int],
                             dates: List[datetime.date]) -> Dict:
        """
        Extract gap-filled temperatures from a NetCDF dataset for specific dates and grid point.
        
        This only extracts the single pixel time series - very fast!
        
        Returns dict with 'temps' (array of temps), 'matched_dates' (list of dates that were found).
        """
        print(f"      Building date-to-index mapping...")
        
        # Build date-to-index mapping
        time_vals = pd.to_datetime(ds['time'].values)
        date_to_idx = {t.date(): i for i, t in enumerate(time_vals)}
        
        print(f"      NetCDF has {len(date_to_idx)} unique dates")
        print(f"      Buoy has {len(dates)} unique dates to match")
        
        # Get lake size for reference
        lakeid = ds['lakeid'].values
        lake_size = np.count_nonzero(lakeid == 1)
        print(f"      Lake size: {lake_size} pixels")
        
        # Extract ONLY the single pixel time series - this is fast!
        print(f"      Loading pixel time series at grid_idx={grid_idx}...")
        temp_filled_pixel = ds['temp_filled'].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        print(f"      Loaded temp_filled for pixel (shape: {temp_filled_pixel.shape})")
        
        # Count how many dates we can match
        matchable_dates = [d for d in dates if d in date_to_idx]
        print(f"      {len(matchable_dates)}/{len(dates)} buoy dates exist in NetCDF")
        
        temps = []
        matched_dates = []
        
        print(f"      Extracting temperatures...")
        for d in matchable_dates:
            idx = date_to_idx[d]
            temp = temp_filled_pixel[idx]
            
            if np.isnan(temp):
                continue
            
            temps.append(temp)
            matched_dates.append(d)
        
        print(f"      Extraction complete: {len(matched_dates)} valid matches (excluding NaN)")
        
        return {
            'temps': np.array(temps),
            'matched_dates': matched_dates,
            'lake_size': lake_size
        }
    
    def extract_full_pixel_timeseries(self, ds: xr.Dataset, grid_idx: Tuple[int, int]) -> Dict:
        """
        Extract the FULL time series for a single pixel (for interpolated daily files).
        
        Returns dict with 'times' (datetime array), 'temps' (temperature array).
        """
        print(f"      Loading full pixel time series at grid_idx={grid_idx}...")
        
        time_vals = pd.to_datetime(ds['time'].values)
        temp_filled_pixel = ds['temp_filled'].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        
        print(f"      Loaded {len(time_vals)} time steps")
        
        return {
            'times': time_vals,
            'temps': temp_filled_pixel
        }
    
    def extract_all_lake_pixels_timeseries(self, ds: xr.Dataset, 
                                            max_pixels: int = 500) -> Dict:
        """
        Extract time series for all lake pixels (or a random subset if too many).
        
        Args:
            ds: xarray Dataset
            max_pixels: Maximum number of pixels to extract (random sample if exceeded)
        
        Returns dict with 'times', 'temps_all' (2D array: n_pixels x n_times), 
        'pixel_indices', 'n_total_pixels'.
        """
        print(f"      Extracting all lake pixel time series...")
        
        # Get lake mask
        lakeid = ds['lakeid'].values
        lake_mask = lakeid == 1
        n_total_pixels = np.count_nonzero(lake_mask)
        print(f"      Total lake pixels: {n_total_pixels}")
        
        # Get indices of lake pixels
        lake_indices = np.argwhere(lake_mask)  # (n_pixels, 2) array of (lat_idx, lon_idx)
        
        # Sample if too many pixels
        if n_total_pixels > max_pixels:
            print(f"      Sampling {max_pixels} random pixels (out of {n_total_pixels})")
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            sample_idx = rng.choice(n_total_pixels, size=max_pixels, replace=False)
            lake_indices = lake_indices[sample_idx]
        
        n_pixels = len(lake_indices)
        print(f"      Extracting {n_pixels} pixel time series...")
        
        time_vals = pd.to_datetime(ds['time'].values)
        n_times = len(time_vals)
        
        # Extract all selected pixels
        temps_all = np.zeros((n_pixels, n_times), dtype=np.float32)
        for i, (lat_idx, lon_idx) in enumerate(lake_indices):
            temps_all[i, :] = ds['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values
            if (i + 1) % 100 == 0:
                print(f"        Extracted {i + 1}/{n_pixels} pixels...")
        
        print(f"      Extraction complete")
        
        return {
            'times': time_vals,
            'temps_all': temps_all,
            'pixel_indices': lake_indices,
            'n_total_pixels': n_total_pixels,
            'n_sampled_pixels': n_pixels
        }
    
    def _get_year_suffix(self, year_start: Optional[int], year_end: Optional[int]) -> str:
        """Generate a string suffix for filenames based on year range."""
        if year_start is None and year_end is None:
            return "all"
        elif year_start == year_end:
            return str(year_start)
        elif year_start is not None and year_end is not None:
            return f"{year_start}-{year_end}"
        elif year_start is not None:
            return f"{year_start}-present"
        else:
            return f"upto-{year_end}"
    
    def _get_year_str(self, year_start: Optional[int], year_end: Optional[int]) -> str:
        """Generate a human-readable string for year range."""
        if year_start is None and year_end is None:
            return "All Years"
        elif year_start == year_end:
            return str(year_start)
        elif year_start is not None and year_end is not None:
            return f"{year_start}-{year_end}"
        elif year_start is not None:
            return f"{year_start} onwards"
        else:
            return f"up to {year_end}"
    
    def validate_lake_site(self, lake_id: int, lake_id_cci: int, site_id: int,
                           year_start: Optional[int] = None,
                           year_end: Optional[int] = None) -> Optional[Dict]:
        """
        Main validation function for a single lake/site combination.
        
        Args:
            lake_id: Lake ID (for buoy file lookup)
            lake_id_cci: Lake CCI ID (for NetCDF file lookup)
            site_id: Site ID
            year_start: Start year (None = no lower limit)
            year_end: End year (None = no upper limit)
        
        Returns a dict with matched data for all methods.
        """
        year_str = self._get_year_str(year_start, year_end)
        print(f"\n{'='*60}")
        print(f"Processing Lake ID: {lake_id} (CCI: {lake_id_cci}), Site: {site_id}, Year: {year_str}")
        print(f"{'='*60}")
        
        # Step 1: Get representative hour (may be None if no selection data)
        rep_hour = self.determine_representative_hour(lake_id, site_id)
        if rep_hour is None:
            print(f"  Could not determine representative hour from selection CSV.")
            print(f"  Will match by date only (no hour filtering).")
        
        # Step 2: Load buoy data
        buoy_df = self.load_buoy_data(lake_id, site_id, year_start, year_end, rep_hour)
        if buoy_df is None or buoy_df.empty:
            print(f"  No buoy data available. Skipping.")
            return None
        
        # Step 3: Get site coordinates - try selection CSV first, then buoy file
        site_info = self.selection_df[
            (self.selection_df['lake_id'] == lake_id) & 
            (self.selection_df['site_id'] == site_id)
        ]
        
        if not site_info.empty:
            site_lat = site_info.iloc[0]['latitude']
            site_lon = site_info.iloc[0]['longitude']
            print(f"  Site coordinates (from selection CSV): lat={site_lat:.4f}, lon={site_lon:.4f}")
        else:
            # Fall back to buoy file coordinates
            site_lat = buoy_df.iloc[0]['lat']
            site_lon = buoy_df.iloc[0]['lon']
            print(f"  Site coordinates (from buoy file): lat={site_lat:.4f}, lon={site_lon:.4f}")
        
        # Step 4: Find pipeline outputs
        outputs = self.find_pipeline_outputs(lake_id_cci)
        if not any(outputs.values()):
            print(f"  No pipeline outputs found for lake_id_cci={lake_id_cci}. Skipping.")
            return None
        
        print(f"  Found outputs: {[k for k, v in outputs.items() if v]}")
        
        # Step 5: Get buoy dates and temperatures
        buoy_dates = buoy_df['dateTime'].dt.date.tolist()
        buoy_temps = buoy_df['Tw'].values
        
        # Create a date-to-temp mapping for buoy data
        # Handle multiple readings on same date by averaging
        buoy_date_temp = buoy_df.groupby(buoy_df['dateTime'].dt.date)['Tw'].mean().to_dict()
        unique_buoy_dates = list(buoy_date_temp.keys())
        
        # Step 6: Process each output file
        results = {
            'lake_id': lake_id,
            'lake_id_cci': lake_id_cci,
            'site_id': site_id,
            'year_start': year_start,
            'year_end': year_end,
            'year_str': year_str,
            'site_lat': site_lat,
            'site_lon': site_lon,
            'representative_hour': rep_hour,  # May be None if daily data or no selection info
            'methods': {}
        }
        
        grid_idx = None
        
        for method_name, nc_path in outputs.items():
            if nc_path is None:
                continue
            
            print(f"\n  Processing {method_name}...")
            print(f"    Opening: {os.path.basename(nc_path)}")
            
            try:
                ds = xr.open_dataset(nc_path)
                print(f"    Dataset opened. Dims: time={ds.sizes['time']}, lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")
            except Exception as e:
                print(f"    Error opening {nc_path}: {e}")
                continue
            
            # Find nearest grid point (only need to do this once)
            if grid_idx is None:
                lat_array = ds['lat'].values
                lon_array = ds['lon'].values
                grid_idx, distance = self.find_nearest_grid_point(lat_array, lon_array, site_lat, site_lon)
                
                print(f"    Nearest grid point: idx={grid_idx}, distance={distance:.4f}°")
                
                if distance > self.distance_threshold:
                    print(f"    Distance exceeds threshold ({self.distance_threshold}°). Skipping lake.")
                    ds.close()
                    return None
                
                results['grid_idx'] = grid_idx
                results['grid_distance'] = distance
            
            # Extract temperatures at buoy pixel
            extracted = self.extract_temperatures(ds, grid_idx, unique_buoy_dates)
            
            if len(extracted['matched_dates']) == 0:
                print(f"    No matching dates found.")
                ds.close()
                continue
            
            # Get corresponding buoy temps for matched dates
            matched_buoy_temps = np.array([buoy_date_temp[d] for d in extracted['matched_dates']])
            
            # Compute statistics
            diff = extracted['temps'] - matched_buoy_temps
            rmse = np.sqrt(np.mean(diff**2))
            bias = np.mean(diff)
            std = np.std(diff)
            corr = np.corrcoef(extracted['temps'], matched_buoy_temps)[0, 1] if len(diff) > 1 else np.nan
            
            results['methods'][method_name] = {
                'dates': extracted['matched_dates'],
                'satellite_temps': extracted['temps'],
                'insitu_temps': matched_buoy_temps,
                'difference': diff,
                'lake_size': extracted['lake_size'],
                'n_matches': len(diff),
                'rmse': rmse,
                'bias': bias,
                'std': std,
                'correlation': corr,
            }
            
            # For interpolated files (daily data), extract full time series for plotting
            is_interpolated = 'interp' in method_name
            if is_interpolated:
                print(f"    Extracting full time series for interpolated data...")
                full_ts = self.extract_full_pixel_timeseries(ds, grid_idx)
                results['methods'][method_name]['full_timeseries'] = {
                    'times': full_ts['times'],
                    'temps': full_ts['temps']
                }
            
            # Extract all lake pixels if requested (for spaghetti plot)
            if self.plot_all_pixels:
                print(f"    Extracting all lake pixel time series...")
                all_pixels = self.extract_all_lake_pixels_timeseries(ds, self.max_pixels_to_plot)
                results['methods'][method_name]['all_pixels'] = all_pixels
            
            ds.close()
            
            print(f"    Matched: {len(diff)} dates, RMSE: {rmse:.3f}°C, Bias: {bias:.3f}°C, R: {corr:.3f}")
        
        if not results['methods']:
            print(f"  No valid method results. Skipping.")
            return None
        
        return results
    
    def plot_validation(self, results: Dict, save_path: Optional[str] = None):
        """Generate validation plots."""
        lake_id = results['lake_id']
        site_id = results['site_id']
        year_str = results['year_str']
        
        methods = results['methods']
        n_methods = len(methods)
        
        if n_methods == 0:
            return
        
        # Check what data we have
        has_full_ts = any('full_timeseries' in m for m in methods.values())
        has_all_pixels = any('all_pixels' in m for m in methods.values())
        
        # Determine number of plots
        # 1: Matched comparison (all methods)
        # 2-N: Per-method difference plots
        # +1: Full time series plot (if has_full_ts)
        # +1: All pixels spaghetti plot (if has_all_pixels)
        n_extra = (1 if has_full_ts else 0) + (1 if has_all_pixels else 0)
        n_plots = 1 + n_methods + n_extra
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        colors = {
            'dineof': 'blue',
            'dincae': 'orange', 
            'eof_filtered': 'green',
            'interp_full': 'purple',
            'eof_filtered_interp_full': 'brown',
            'insitu': 'red',
        }
        
        plot_idx = 0
        
        # Plot 1: Matched temperature comparison for all methods
        ax = axes[plot_idx]
        plot_idx += 1
        
        for method_name, method_data in methods.items():
            dates = method_data['dates']
            sat_temps = method_data['satellite_temps']
            ax.plot(dates, sat_temps, label=f'{method_name}', 
                   marker='o', linestyle='-', color=colors.get(method_name, 'gray'), 
                   markersize=4, alpha=0.8)
        
        # Plot in-situ (use first method's dates for reference)
        first_method = list(methods.values())[0]
        ax.plot(first_method['dates'], first_method['insitu_temps'], 
               label='In-situ', marker='x', linestyle='None', 
               color=colors['insitu'], markersize=6)
        
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Lake {lake_id} Site {site_id} ({year_str}) - Matched Temperature Comparison')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plots 2-N: Difference for each method
        for method_name, method_data in methods.items():
            ax = axes[plot_idx]
            plot_idx += 1
            
            dates = method_data['dates']
            diff = method_data['difference']
            
            ax.plot(dates, diff, label=f'Diff (Satellite - Insitu)', 
                   marker='s', linestyle='-', color=colors.get(method_name, 'gray'),
                   markersize=4)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.fill_between(dates, diff, 0, alpha=0.3, color=colors.get(method_name, 'gray'))
            ax.set_ylabel('Difference (°C)')
            ax.set_title(f'{method_name}: RMSE={method_data["rmse"]:.3f}°C, '
                        f'Bias={method_data["bias"]:.3f}°C, R={method_data["correlation"]:.3f}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Full time series plot (for interpolated data)
        if has_full_ts:
            ax = axes[plot_idx]
            plot_idx += 1
            
            for method_name, method_data in methods.items():
                if 'full_timeseries' not in method_data:
                    continue
                
                full_ts = method_data['full_timeseries']
                times = full_ts['times']
                temps = full_ts['temps']
                
                color = colors.get(method_name, 'gray')
                ax.plot(times, temps, label=f'{method_name} (daily)', 
                       linestyle='-', color=color, linewidth=0.8, alpha=0.8)
            
            # Overlay in-situ as markers
            ax.plot(first_method['dates'], first_method['insitu_temps'], 
                   label='In-situ', marker='o', linestyle='None', 
                   color=colors['insitu'], markersize=5, zorder=10)
            
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Full Daily Time Series with In-Situ Overlay')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # All pixels spaghetti plot
        if has_all_pixels:
            ax = axes[plot_idx]
            plot_idx += 1
            
            # Find a method with all_pixels data
            for method_name, method_data in methods.items():
                if 'all_pixels' not in method_data:
                    continue
                
                all_pix = method_data['all_pixels']
                times = all_pix['times']
                temps_all = all_pix['temps_all']  # (n_pixels, n_times)
                n_sampled = all_pix['n_sampled_pixels']
                n_total = all_pix['n_total_pixels']
                
                # Plot all pixel time series with low alpha
                for i in range(temps_all.shape[0]):
                    if i == 0:
                        ax.plot(times, temps_all[i, :], color='lightgray', 
                               linewidth=0.5, alpha=0.3, 
                               label=f'Lake pixels ({n_sampled}/{n_total} sampled)')
                    else:
                        ax.plot(times, temps_all[i, :], color='lightgray', 
                               linewidth=0.5, alpha=0.3)
                
                # Plot the buoy pixel in darker color
                if 'full_timeseries' in method_data:
                    full_ts = method_data['full_timeseries']
                    ax.plot(full_ts['times'], full_ts['temps'], 
                           color=colors.get(method_name, 'blue'), linewidth=1.5,
                           label=f'Buoy pixel ({method_name})')
                
                break  # Only need one method for all pixels
            
            # Overlay in-situ
            ax.plot(first_method['dates'], first_method['insitu_temps'], 
                   label='In-situ', marker='o', linestyle='None', 
                   color=colors['insitu'], markersize=6, zorder=10)
            
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'All Lake Pixels (spaghetti) with Buoy Pixel and In-Situ Overlay')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Format x-axis for all plots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        axes[-1].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results_csv(self, results: Dict, save_path: str):
        """Save matched data to CSV."""
        rows = []
        
        for method_name, method_data in results['methods'].items():
            for i, date in enumerate(method_data['dates']):
                rows.append({
                    'lake_id': results['lake_id'],
                    'lake_id_cci': results['lake_id_cci'],
                    'site_id': results['site_id'],
                    'year_start': results['year_start'],
                    'year_end': results['year_end'],
                    'date': date,
                    'method': method_name,
                    'satellite_temp_C': method_data['satellite_temps'][i],
                    'insitu_temp_C': method_data['insitu_temps'][i],
                    'difference_C': method_data['difference'][i],
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"  Saved CSV: {save_path}")
    
    def save_summary_csv(self, all_results: List[Dict], save_path: str):
        """Save summary statistics for all lakes/sites/methods."""
        rows = []
        
        for results in all_results:
            for method_name, method_data in results['methods'].items():
                rows.append({
                    'lake_id': results['lake_id'],
                    'lake_id_cci': results['lake_id_cci'],
                    'site_id': results['site_id'],
                    'year_start': results['year_start'],
                    'year_end': results['year_end'],
                    'site_lat': results['site_lat'],
                    'site_lon': results['site_lon'],
                    'method': method_name,
                    'n_matches': method_data['n_matches'],
                    'rmse': method_data['rmse'],
                    'bias': method_data['bias'],
                    'std': method_data['std'],
                    'correlation': method_data['correlation'],
                    'lake_size_pixels': method_data['lake_size'],
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"\nSaved summary: {save_path}")
    
    def run(self, lake_ids: Optional[List[int]] = None, 
            year_start: Optional[int] = None, year_end: Optional[int] = None):
        """
        Run validation for specified lakes or all lakes in selection CSV.
        
        Args:
            lake_ids: List of lake IDs to process (None = all in selection CSV)
            year_start: Start year (None = no lower limit)
            year_end: End year (None = no upper limit)
        """
        pairs = self.get_lake_site_pairs(lake_ids)
        year_str = self._get_year_str(year_start, year_end)
        print(f"Processing {len(pairs)} lake/site pairs for {year_str}")
        
        all_results = []
        
        for _, row in pairs.iterrows():
            lake_id = row['lake_id']
            lake_id_cci = row['lake_id_cci']
            site_id = row['site_id']
            
            results = self.validate_lake_site(lake_id, lake_id_cci, site_id, year_start, year_end)
            
            if results is not None:
                all_results.append(results)
                
                # Save individual results
                year_suffix = self._get_year_suffix(year_start, year_end)
                base_name = f"lake{lake_id}_site{site_id}_{year_suffix}"
                
                # Plot
                plot_path = os.path.join(self.output_dir, f"{base_name}_comparison.png")
                self.plot_validation(results, plot_path)
                
                # CSV
                csv_path = os.path.join(self.output_dir, f"{base_name}_matched.csv")
                self.save_results_csv(results, csv_path)
        
        # Save summary
        if all_results:
            year_suffix = self._get_year_suffix(year_start, year_end)
            summary_path = os.path.join(self.output_dir, f"validation_summary_{year_suffix}.csv")
            self.save_summary_csv(all_results, summary_path)
        
        print(f"\n{'='*60}")
        print(f"Validation complete: {len(all_results)} lakes processed successfully")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='In-Situ Validation for Lake CCI Gap-Filling Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate specific lake/site for a specific year
    python insitu_validation.py --lake-id 2 --year 2010
    
    # Validate for a range of years (2004 to 2022 inclusive)
    python insitu_validation.py --lake-id 2 --year 2004 2022
    
    # Validate specific lake/site for ALL years in buoy data
    python insitu_validation.py --lake-id 2 --year all
    
    # Validate using lake_id_cci directly (for lakes not in selection CSV)
    python insitu_validation.py --lake-id-cci 20 --site-id 1 --year 2019 2022
    
    # Plot all lake pixels as spaghetti plot with in-situ on top
    python insitu_validation.py --lake-id-cci 20 --site-id 1 --year 2019 2022 --all-pixels
    
    # Limit pixels in spaghetti plot (for very large lakes)
    python insitu_validation.py --lake-id-cci 20 --site-id 1 --all-pixels --max-pixels 200
    
    # Validate all lakes in selection CSV
    python insitu_validation.py --all --year 2010
    
    # Use custom config
    python insitu_validation.py --config my_config.json --lake-id 2
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--lake-id', type=int, nargs='+', help='Lake ID(s) to validate (uses selection CSV)')
    parser.add_argument('--lake-id-cci', type=int, help='Direct lake_id_cci (bypasses selection CSV)')
    parser.add_argument('--site-id', type=int, default=1, help='Site ID (used with --lake-id-cci, default: 1)')
    parser.add_argument('--year', type=str, nargs='+', default=['all'], 
                        help='Target year(s): single year (2010), range (2004 2022), or "all" (default: all)')
    parser.add_argument('--all', action='store_true', help='Validate all lakes in selection CSV')
    parser.add_argument('--run-root', type=str, help='Override run root directory')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--all-pixels', action='store_true', 
                        help='Plot all lake pixel time series (spaghetti plot) with insitu overlay')
    parser.add_argument('--max-pixels', type=int, default=500,
                        help='Max pixels to sample for spaghetti plot (default: 500)')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    
    # Override with CLI args
    if args.run_root:
        config['run_root'] = args.run_root
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.all_pixels:
        config['plot_all_pixels'] = True
    if args.max_pixels:
        config['max_pixels_to_plot'] = args.max_pixels
    
    # Parse year argument
    year_args = args.year
    if len(year_args) == 1 and year_args[0].lower() == 'all':
        year_start = None
        year_end = None
    elif len(year_args) == 1:
        try:
            year_start = int(year_args[0])
            year_end = year_start
        except ValueError:
            print(f"ERROR: Invalid year '{year_args[0]}'. Use a number, range (2004 2022), or 'all'.")
            sys.exit(1)
    elif len(year_args) == 2:
        try:
            year_start = int(year_args[0])
            year_end = int(year_args[1])
            if year_start > year_end:
                year_start, year_end = year_end, year_start  # Swap if reversed
        except ValueError:
            print(f"ERROR: Invalid year range '{year_args}'. Use two numbers like '2004 2022'.")
            sys.exit(1)
    else:
        print(f"ERROR: Invalid year specification. Use single year, range (2004 2022), or 'all'.")
        sys.exit(1)
    
    # Initialize validator
    validator = InsituValidator(config)
    
    # Handle direct lake_id_cci mode (bypasses selection CSV)
    if args.lake_id_cci:
        lake_id_cci = args.lake_id_cci
        site_id = args.site_id
        # For direct mode, assume lake_id == lake_id_cci (user can check buoy files)
        lake_id = lake_id_cci
        
        print(f"Direct mode: lake_id_cci={lake_id_cci}, site_id={site_id}")
        results = validator.validate_lake_site(lake_id, lake_id_cci, site_id, year_start, year_end)
        
        if results:
            year_suffix = validator._get_year_suffix(year_start, year_end)
            base_name = f"lake{lake_id}_site{site_id}_{year_suffix}"
            
            plot_path = os.path.join(validator.output_dir, f"{base_name}_comparison.png")
            validator.plot_validation(results, plot_path)
            
            csv_path = os.path.join(validator.output_dir, f"{base_name}_matched.csv")
            validator.save_results_csv(results, csv_path)
            
            print(f"\nValidation complete. Output: {validator.output_dir}")
            return 0
        return 1
    
    # Standard mode: use selection CSV
    lake_ids = None
    if args.lake_id:
        lake_ids = args.lake_id
    elif not args.all:
        print("ERROR: Specify --lake-id, --lake-id-cci, or --all")
        parser.print_help()
        sys.exit(1)
    
    # Run validation
    results = validator.run(lake_ids=lake_ids, year_start=year_start, year_end=year_end)
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())