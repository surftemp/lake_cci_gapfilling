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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


# ============================================================================
# Plotting Helper Functions (adapted from pipeline lswt_plots.py)
# ============================================================================

def compute_year_ticks(time: pd.DatetimeIndex) -> Tuple[List, List]:
    """Compute yearly tick positions and labels."""
    years = np.unique(time.year)
    y_start = years.min()
    y_end = years.max()
    
    positions, labels = [], []
    for y in range(y_start, y_end + 1):
        if y in years:
            target = pd.Timestamp(y, 1, 1, 12)
            idx = int(np.argmin(np.abs(time - target)))
            positions.append(time[idx])
            labels.append(str(y))
    
    return positions, labels


def split_timeline(time: pd.DatetimeIndex) -> Tuple[int, int, int]:
    """Split timeline into first and second half. Returns (i0, i1, mid_year)."""
    years = np.unique(time.year)
    y_start = years.min()
    y_end = years.max()
    mid_year = y_start + (y_end - y_start) // 2
    
    i0 = int(np.searchsorted(time.values, np.datetime64(f"{y_start}-01-01")))
    i1 = int(np.searchsorted(time.values, np.datetime64(f"{mid_year + 1}-01-01")))
    
    return i0, i1, mid_year


def plot_panel(ax, x_vals, all_pixels, center_pixel, seg_slice, y_lim, 
               tick_pos, tick_lab, title, show_legend=False, 
               insitu_dates=None, insitu_temps=None):
    """
    Plot a single panel with pipeline-style formatting.
    
    Args:
        ax: matplotlib axis
        x_vals: time index
        all_pixels: (time, n_pixels) array, or None to skip
        center_pixel: (time,) array for center/buoy pixel
        seg_slice: slice for this panel
        y_lim: (ymin, ymax)
        tick_pos, tick_lab: tick positions and labels
        title: panel title
        show_legend: whether to show legend
        insitu_dates, insitu_temps: optional in-situ overlay
    """
    ax.yaxis.grid(True, linestyle='--', linewidth=0.3)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.3)
    ax.set_ylim(*y_lim)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    
    x_seg = x_vals[seg_slice]
    center = center_pixel[seg_slice]
    
    # All pixels (faint black lines)
    if all_pixels is not None:
        arr = all_pixels[seg_slice, :]
        for j in range(arr.shape[1]):
            col = arr[:, j]
            valid = np.isfinite(col)
            if valid.any():
                ax.plot(x_seg[valid], col[valid], '-', color='black', alpha=0.15, lw=0.3)
    
    # Center/buoy pixel (bold red line)
    valid_c = np.isfinite(center)
    if valid_c.any():
        ax.plot(x_seg[valid_c], center[valid_c], '-', color='red', lw=0.8, label='Satellite (buoy pixel)')
    
    # In-situ overlay (blue markers)
    if insitu_dates is not None and insitu_temps is not None:
        # Filter insitu to this segment
        seg_start = x_seg.min()
        seg_end = x_seg.max()
        in_seg = [(d, t) for d, t in zip(insitu_dates, insitu_temps) 
                  if pd.Timestamp(d) >= seg_start and pd.Timestamp(d) <= seg_end]
        if in_seg:
            seg_dates, seg_temps = zip(*in_seg)
            ax.plot(seg_dates, seg_temps, 'o', color='blue', markersize=4, 
                   alpha=0.8, label='In-situ', zorder=10)
    
    if show_legend:
        ax.legend(fontsize=8, loc='best')


# ============================================================================
# Main Validator Class
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
        self.plot_all_pixels = config.get("plot_all_pixels", False)
        self.max_pixels_to_plot = config.get("max_pixels_to_plot", 500)
        
        # Load selection CSV
        self.selection_df = self._load_selection_csv()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_selection_csv(self) -> pd.DataFrame:
        """Load and clean the selection CSV."""
        df = pd.read_csv(self.selection_csv_path)
        df.columns = df.columns.str.strip()
        
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
        """Find all pipeline output files for a lake."""
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
        """Find the representative hour for satellite overpasses."""
        subset = self.selection_df[
            (self.selection_df['lake_id'] == lake_id) & 
            (self.selection_df['site_id'] == site_id)
        ].copy()
        
        if subset.empty:
            return None
        
        subset['time_floor'] = subset['time_IS'].dt.floor('H')
        subset['date'] = subset['time_IS'].dt.date
        subset['hour'] = subset['time_floor'].dt.hour
        
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
        """Load and filter buoy CSV data."""
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
        
        # Detect daily vs hourly data
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
            else:
                df = df[years <= year_end]
        
        # Filter for representative hour (only for hourly data)
        if representative_hour is not None and not is_daily_data:
            df['hour'] = df['dateTime'].dt.hour
            df = df[df['hour'] == representative_hour]
            print(f"  After hour {representative_hour} filter: {len(df)} rows")
        elif is_daily_data:
            print(f"  Skipping hour filter (daily data)")
        
        # Apply quality filter
        if 'qcFlag' in df.columns:
            df = df[df['qcFlag'] == 0]
        elif 'q' in df.columns:
            df = df[df['q'] == 0]
        print(f"  After quality filter: {len(df)} rows")
        
        print(f"  Buoy data summary: {initial_count} total → {len(df)} after all filters")
        
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
    
    def extract_matched_temperatures(self, ds: xr.Dataset, grid_idx: Tuple[int, int],
                                      buoy_dates: List) -> Dict:
        """Extract temperatures for dates that match buoy data."""
        time_vals = pd.to_datetime(ds['time'].values)
        date_to_idx = {t.date(): i for i, t in enumerate(time_vals)}
        
        lakeid = ds['lakeid'].values
        lake_size = np.count_nonzero(lakeid == 1)
        
        temp_filled_pixel = ds['temp_filled'].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        
        matchable_dates = [d for d in buoy_dates if d in date_to_idx]
        
        temps = []
        matched_dates = []
        skipped_nan = 0
        
        for d in matchable_dates:
            idx = date_to_idx[d]
            temp = temp_filled_pixel[idx]
            
            if np.isnan(temp):
                skipped_nan += 1
                continue
            
            temps.append(temp)
            matched_dates.append(d)
        
        return {
            'temps': np.array(temps),
            'matched_dates': matched_dates,
            'lake_size': lake_size,
            'n_matchable': len(matchable_dates),
            'n_skipped_nan': skipped_nan
        }
    
    def extract_full_timeseries(self, ds: xr.Dataset, grid_idx: Tuple[int, int]) -> Dict:
        """Extract full time series for a pixel (for interpolated daily data)."""
        time_vals = pd.to_datetime(ds['time'].values)
        temp_filled_pixel = ds['temp_filled'].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        
        return {
            'times': time_vals,
            'temps': temp_filled_pixel
        }
    
    def extract_all_lake_pixels(self, ds: xr.Dataset, max_pixels: int = 500) -> Dict:
        """Extract time series for all lake pixels (for spaghetti plot)."""
        lakeid = ds['lakeid'].values
        lake_mask = lakeid == 1
        n_total_pixels = np.count_nonzero(lake_mask)
        
        lake_indices = np.argwhere(lake_mask)
        
        if n_total_pixels > max_pixels:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n_total_pixels, size=max_pixels, replace=False)
            lake_indices = lake_indices[sample_idx]
        
        n_pixels = len(lake_indices)
        time_vals = pd.to_datetime(ds['time'].values)
        n_times = len(time_vals)
        
        temps_all = np.zeros((n_times, n_pixels), dtype=np.float32)
        for i, (lat_idx, lon_idx) in enumerate(lake_indices):
            temps_all[:, i] = ds['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values
        
        return {
            'times': time_vals,
            'temps_all': temps_all,
            'n_total_pixels': n_total_pixels,
            'n_sampled_pixels': n_pixels
        }
    
    def _get_year_suffix(self, year_start: Optional[int], year_end: Optional[int]) -> str:
        if year_start is None and year_end is None:
            return "all"
        elif year_start == year_end:
            return str(year_start)
        elif year_start is not None and year_end is not None:
            return f"{year_start}-{year_end}"
        else:
            return "partial"
    
    def _get_year_str(self, year_start: Optional[int], year_end: Optional[int]) -> str:
        if year_start is None and year_end is None:
            return "All Years"
        elif year_start == year_end:
            return str(year_start)
        elif year_start is not None and year_end is not None:
            return f"{year_start}-{year_end}"
        else:
            return "partial"
    
    def validate_lake_site(self, lake_id: int, lake_id_cci: int, site_id: int,
                           year_start: Optional[int] = None,
                           year_end: Optional[int] = None) -> Optional[Dict]:
        """Main validation function for a single lake/site combination."""
        year_str = self._get_year_str(year_start, year_end)
        print(f"\n{'='*60}")
        print(f"Processing Lake ID: {lake_id} (CCI: {lake_id_cci}), Site: {site_id}, Year: {year_str}")
        print(f"{'='*60}")
        
        # Step 1: Get representative hour
        rep_hour = self.determine_representative_hour(lake_id, site_id)
        if rep_hour is None:
            print(f"  Could not determine representative hour from selection CSV.")
            print(f"  Will match by date only (no hour filtering).")
        
        # Step 2: Load buoy data
        buoy_df = self.load_buoy_data(lake_id, site_id, year_start, year_end, rep_hour)
        if buoy_df is None or buoy_df.empty:
            print(f"  No buoy data available. Skipping.")
            return None
        
        # Step 3: Get site coordinates
        site_info = self.selection_df[
            (self.selection_df['lake_id'] == lake_id) & 
            (self.selection_df['site_id'] == site_id)
        ]
        
        if not site_info.empty:
            site_lat = site_info.iloc[0]['latitude']
            site_lon = site_info.iloc[0]['longitude']
            print(f"  Site coordinates (from selection CSV): lat={site_lat:.4f}, lon={site_lon:.4f}")
        else:
            site_lat = buoy_df.iloc[0]['lat']
            site_lon = buoy_df.iloc[0]['lon']
            print(f"  Site coordinates (from buoy file): lat={site_lat:.4f}, lon={site_lon:.4f}")
        
        # Step 4: Find pipeline outputs
        outputs = self.find_pipeline_outputs(lake_id_cci)
        if not any(outputs.values()):
            print(f"  No pipeline outputs found for lake_id_cci={lake_id_cci}. Skipping.")
            return None
        
        print(f"  Found outputs: {[k for k, v in outputs.items() if v]}")
        
        # Step 5: Prepare buoy data
        buoy_date_temp = buoy_df.groupby(buoy_df['dateTime'].dt.date)['Tw'].mean().to_dict()
        unique_buoy_dates = list(buoy_date_temp.keys())
        print(f"  Buoy has {len(unique_buoy_dates)} unique dates after grouping")
        
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
            'representative_hour': rep_hour,
            'buoy_date_temp': buoy_date_temp,
            'methods': {}
        }
        
        grid_idx = None
        
        for method_name, nc_path in outputs.items():
            if nc_path is None:
                continue
            
            is_interpolated = 'interp' in method_name
            
            print(f"\n  Processing {method_name}{'(interpolated daily)' if is_interpolated else '(sparse)'}...")
            print(f"    Opening: {os.path.basename(nc_path)}")
            
            try:
                ds = xr.open_dataset(nc_path)
                print(f"    Dataset: time={ds.sizes['time']}, lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")
            except Exception as e:
                print(f"    Error opening {nc_path}: {e}")
                continue
            
            # Find nearest grid point (once)
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
            
            # Extract matched temperatures for statistics
            extracted = self.extract_matched_temperatures(ds, grid_idx, unique_buoy_dates)
            
            if len(extracted['matched_dates']) == 0:
                print(f"    No matching dates found.")
                ds.close()
                continue
            
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
                'is_interpolated': is_interpolated,
            }
            
            # For interpolated files, extract FULL time series
            if is_interpolated:
                print(f"    Extracting full daily time series...")
                full_ts = self.extract_full_timeseries(ds, grid_idx)
                results['methods'][method_name]['full_timeseries'] = full_ts
                
                # Extract all lake pixels if requested
                if self.plot_all_pixels:
                    print(f"    Extracting all lake pixel time series...")
                    all_pixels = self.extract_all_lake_pixels(ds, self.max_pixels_to_plot)
                    results['methods'][method_name]['all_pixels'] = all_pixels
            
            ds.close()
            
            print(f"    Matched: {len(diff)} dates, RMSE: {rmse:.3f}°C, Bias: {bias:.3f}°C, R: {corr:.3f}")
        
        if not results['methods']:
            print(f"  No valid method results. Skipping.")
            return None
        
        return results
    
    def plot_validation(self, results: Dict, save_path: Optional[str] = None):
        """
        Generate validation plots using pipeline-style formatting.
        
        Plot structure:
        1. Sparse methods comparison - satellite at observation dates, ALL in-situ dates
        2. Difference plots for each sparse method (matched dates only)
        3. For each interpolated method: 2-panel full daily time series with ALL in-situ overlay
        4. Difference plots for each interpolated method (matched dates only)
        5. Spaghetti plots (if --all-pixels enabled)
        
        Key principle:
        - Time series plots show ALL in-situ dates to see full buoy coverage
        - Difference plots only show matched dates (where both sat and in-situ exist)
        """
        lake_id = results['lake_id']
        site_id = results['site_id']
        year_str = results['year_str']
        
        methods = results['methods']
        if not methods:
            return
        
        # Separate sparse and interpolated methods
        sparse_methods = {k: v for k, v in methods.items() if not v.get('is_interpolated', False)}
        interp_methods = {k: v for k, v in methods.items() if v.get('is_interpolated', False)}
        
        # Count plots needed
        n_sparse = len(sparse_methods)
        n_interp = len(interp_methods)
        n_spaghetti = sum(1 for v in interp_methods.values() if 'all_pixels' in v)
        
        # Calculate total plots:
        # - Sparse: 1 comparison + N diff plots (only if sparse methods exist)
        # - Interp: 2 panels (full time series) + 1 diff plot per method = 3 per method
        # - Spaghetti: 1 per interp method with all_pixels
        
        n_sparse_plots = (1 + n_sparse) if n_sparse > 0 else 0
        n_interp_plots = 3 * n_interp  # 2 panels + 1 diff plot each
        n_plots = n_sparse_plots + n_interp_plots + n_spaghetti
        
        if n_plots == 0:
            return
        
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
        
        # ============ SPARSE METHODS ONLY ============
        # For sparse methods, show satellite at observation dates, but show ALL in-situ dates
        all_insitu_dates = list(results['buoy_date_temp'].keys())
        all_insitu_temps = list(results['buoy_date_temp'].values())
        print(f"  Total in-situ dates available: {len(all_insitu_dates)}")
        
        if sparse_methods:
            # Plot 1: Sparse methods comparison
            # - Satellite: plotted at dates where observations exist (matched dates)
            # - In-situ: plotted at ALL dates (to show full buoy coverage)
            ax = axes[plot_idx]
            plot_idx += 1
            
            for method_name, method_data in sparse_methods.items():
                dates = method_data['dates']
                sat_temps = method_data['satellite_temps']
                ax.plot(dates, sat_temps, label=method_name, 
                       marker='o', linestyle='-', color=colors.get(method_name, 'gray'), 
                       markersize=3, alpha=0.8, linewidth=0.8)
            
            # In-situ: plot ALL dates (not just matched)
            ax.plot(all_insitu_dates, all_insitu_temps, 
                   label=f'In-situ ({len(all_insitu_dates)} pts)', 
                   marker='x', linestyle='None', 
                   color=colors['insitu'], markersize=5, alpha=0.7)
            
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Lake {lake_id} Site {site_id} ({year_str}) - Sparse Methods vs In-Situ (all buoy dates)')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
            
            # Difference plots for sparse methods (only matched dates - where both exist)
            for method_name, method_data in sparse_methods.items():
                ax = axes[plot_idx]
                plot_idx += 1
                
                dates = method_data['dates']
                diff = method_data['difference']
                
                ax.plot(dates, diff, marker='s', linestyle='-', 
                       color=colors.get(method_name, 'gray'), markersize=3, linewidth=0.8)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.fill_between(dates, diff, 0, alpha=0.3, color=colors.get(method_name, 'gray'))
                ax.set_ylabel('Difference (°C)')
                ax.set_title(f'{method_name}: RMSE={method_data["rmse"]:.3f}°C, '
                            f'Bias={method_data["bias"]:.3f}°C, R={method_data["correlation"]:.3f}, '
                            f'N={method_data["n_matches"]} (matched dates only)')
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
        
        # ============ INTERPOLATED METHODS (Full Daily Time Series) ============
        # For interpolated methods, use ALL in-situ dates (not just matched with sparse)
        # The interpolated data is daily, so it should have values for most in-situ dates
        print(f"  Plotting with ALL {len(all_insitu_dates)} in-situ dates for overlays")
        
        for method_name, method_data in interp_methods.items():
            if 'full_timeseries' not in method_data:
                continue
            
            full_ts = method_data['full_timeseries']
            # Ensure times is a DatetimeIndex for proper handling
            times = pd.DatetimeIndex(full_ts['times'])
            temps = full_ts['temps']
            
            # Split timeline into two halves
            i0, i1, mid_year = split_timeline(times)
            
            if i1 >= len(times) or i0 >= i1:
                # Not enough data to split - skip
                continue
            
            tt0, tt1 = times[i0:i1], times[i1:]
            pos0, lab0 = compute_year_ticks(tt0)
            pos1, lab1 = compute_year_ticks(tt1)
            
            # Compute y-limits from both satellite and ALL in-situ
            finite_temps = temps[np.isfinite(temps)]
            finite_insitu = np.array([t for t in all_insitu_temps if np.isfinite(t)])
            all_vals = np.concatenate([finite_temps, finite_insitu]) if len(finite_temps) > 0 else finite_insitu
            if len(all_vals) > 0:
                y_min, y_max = np.min(all_vals), np.max(all_vals)
                pad = 0.05 * max(1.0, y_max - y_min)
                y_lim = (y_min - pad, y_max + pad)
            else:
                y_lim = (0, 30)
            
            # Get all_pixels if available (for spaghetti background)
            all_pixels = method_data.get('all_pixels', {}).get('temps_all', None)
            
            # First half panel - FULL DAILY DATA with ALL in-situ overlay
            ax = axes[plot_idx]
            plot_idx += 1
            
            plot_panel(ax, times, all_pixels, temps, slice(i0, i1), y_lim,
                      pos0, lab0, 
                      f'{method_name}: Full Daily ({times[i0].year}-{mid_year})',
                      show_legend=True,
                      insitu_dates=all_insitu_dates, insitu_temps=all_insitu_temps)
            ax.set_ylabel('LSWT (°C)')
            
            # Second half panel - FULL DAILY DATA with ALL in-situ overlay
            ax = axes[plot_idx]
            plot_idx += 1
            
            plot_panel(ax, times, all_pixels, temps, slice(i1, None), y_lim,
                      pos1, lab1,
                      f'{method_name}: Full Daily ({mid_year+1}-{times[-1].year})',
                      show_legend=False,
                      insitu_dates=all_insitu_dates, insitu_temps=all_insitu_temps)
            ax.set_ylabel('LSWT (°C)')
            
            # Difference plot - ONLY matched dates (where both satellite and in-situ exist)
            ax = axes[plot_idx]
            plot_idx += 1
            
            # Use matched dates for difference calculation
            matched_dates = method_data['dates']
            diff = method_data['difference']
            
            ax.plot(matched_dates, diff, marker='s', linestyle='-', 
                   color=colors.get(method_name, 'gray'), markersize=3, linewidth=0.8)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.fill_between(matched_dates, diff, 0, alpha=0.3, color=colors.get(method_name, 'gray'))
            ax.set_ylabel('Difference (°C)')
            ax.set_title(f'{method_name} Difference: RMSE={method_data["rmse"]:.3f}°C, '
                        f'Bias={method_data["bias"]:.3f}°C, R={method_data["correlation"]:.3f}, '
                        f'N={method_data["n_matches"]} (matched dates only)')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
        
        # ============ SPAGHETTI PLOTS (if --all-pixels) ============
        for method_name, method_data in interp_methods.items():
            if 'all_pixels' not in method_data:
                continue
            
            ax = axes[plot_idx]
            plot_idx += 1
            
            all_pix = method_data['all_pixels']
            times = pd.DatetimeIndex(all_pix['times'])
            temps_all = all_pix['temps_all']  # (time, n_pixels)
            n_sampled = all_pix['n_sampled_pixels']
            n_total = all_pix['n_total_pixels']
            
            # All pixels (faint black lines - pipeline style)
            for i in range(temps_all.shape[1]):
                col = temps_all[:, i]
                valid = np.isfinite(col)
                if valid.any():
                    label = f'Lake pixels ({n_sampled}/{n_total} sampled)' if i == 0 else None
                    ax.plot(times[valid], col[valid], '-', color='black', 
                           alpha=0.15, lw=0.3, label=label)
            
            # Buoy pixel (bold red line)
            if 'full_timeseries' in method_data:
                full_ts = method_data['full_timeseries']
                buoy_temps = full_ts['temps']
                valid = np.isfinite(buoy_temps)
                ax.plot(times[valid], buoy_temps[valid], '-', color='red', lw=0.8, 
                       label='Buoy pixel (satellite)')
            
            # In-situ overlay - use ALL in-situ dates (blue markers)
            ax.plot(all_insitu_dates, all_insitu_temps, 'o', color='blue', markersize=4,
                   alpha=0.8, label=f'In-situ ({len(all_insitu_dates)} pts)', zorder=10)
            
            ax.set_ylabel('LSWT (°C)')
            ax.set_title(f'{method_name}: All Lake Pixels (spaghetti) with Buoy Pixel and In-Situ')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
        
        # Final formatting
        axes[-1].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
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
        """Run validation for specified lakes."""
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
                
                year_suffix = self._get_year_suffix(year_start, year_end)
                base_name = f"lake{lake_id}_site{site_id}_{year_suffix}"
                
                plot_path = os.path.join(self.output_dir, f"{base_name}_comparison.png")
                self.plot_validation(results, plot_path)
                
                csv_path = os.path.join(self.output_dir, f"{base_name}_matched.csv")
                self.save_results_csv(results, csv_path)
        
        if all_results:
            year_suffix = self._get_year_suffix(year_start, year_end)
            summary_path = os.path.join(self.output_dir, f"validation_summary_{year_suffix}.csv")
            self.save_summary_csv(all_results, summary_path)
        
        print(f"\n{'='*60}")
        print(f"Validation complete: {len(all_results)} lakes processed")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='In-Situ Validation for Lake CCI Gap-Filling Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python insitu_validation.py --lake-id 2 --year 2010
    python insitu_validation.py --lake-id-cci 4503 --site-id 1 --year 2019 2022
    python insitu_validation.py --lake-id-cci 4503 --site-id 1 --year 2019 2022 --all-pixels
    python insitu_validation.py --all --year 2010
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--lake-id', type=int, nargs='+', help='Lake ID(s) (uses selection CSV)')
    parser.add_argument('--lake-id-cci', type=int, help='Direct lake_id_cci')
    parser.add_argument('--site-id', type=int, default=1, help='Site ID (default: 1)')
    parser.add_argument('--year', type=str, nargs='+', default=['all'], 
                        help='Year(s): single, range (2004 2022), or "all"')
    parser.add_argument('--all', action='store_true', help='Validate all lakes in selection CSV')
    parser.add_argument('--run-root', type=str, help='Override run root directory')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--all-pixels', action='store_true', 
                        help='Include spaghetti plots showing all lake pixels')
    parser.add_argument('--max-pixels', type=int, default=500,
                        help='Max pixels for spaghetti plot (default: 500)')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    
    if args.run_root:
        config['run_root'] = args.run_root
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.all_pixels:
        config['plot_all_pixels'] = True
    if args.max_pixels:
        config['max_pixels_to_plot'] = args.max_pixels
    
    # Parse year
    year_args = args.year
    if len(year_args) == 1 and year_args[0].lower() == 'all':
        year_start = None
        year_end = None
    elif len(year_args) == 1:
        year_start = int(year_args[0])
        year_end = year_start
    elif len(year_args) == 2:
        year_start = int(year_args[0])
        year_end = int(year_args[1])
        if year_start > year_end:
            year_start, year_end = year_end, year_start
    else:
        print("ERROR: Invalid year specification")
        sys.exit(1)
    
    # Initialize validator
    validator = InsituValidator(config)
    
    # Handle direct lake_id_cci mode
    if args.lake_id_cci:
        lake_id_cci = args.lake_id_cci
        site_id = args.site_id
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
    
    # Standard mode
    lake_ids = None
    if args.lake_id:
        lake_ids = args.lake_id
    elif not args.all:
        print("ERROR: Specify --lake-id, --lake-id-cci, or --all")
        parser.print_help()
        sys.exit(1)
    
    results = validator.run(lake_ids=lake_ids, year_start=year_start, year_end=year_end)
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())