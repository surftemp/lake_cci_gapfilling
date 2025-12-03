"""
In-Situ Validation Step - Generates validation plots comparing gap-filled LSWT to buoy measurements.

This step runs at the end of postprocessing (after LSWTPlotsStep) and creates:
1. Time series comparison plots (sparse methods vs in-situ, interpolated methods vs in-situ)
2. Difference plots with RMSE/Bias/Correlation statistics
3. CSV files with matched data and summary statistics

Configuration (in experiment JSON):
    "insitu_validation": {
        "enable": true,
        "buoy_dir": "/path/to/buoy/data",
        "selection_csv": "/path/to/selection.csv",
        "distance_threshold": 0.05,
        "quality_threshold": 3
    }

Author: Shaerdan / NCEO / University of Reading
"""

from __future__ import annotations
import os
import json
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from typing import Optional, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    # When used as part of pipeline (from post_steps package)
    from .base import PostProcessingStep, PostContext
except ImportError:
    # When used standalone - define minimal versions
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class PostContext:
        """Minimal PostContext for standalone use."""
        lake_id: Optional[int] = None
        output_path: Optional[str] = None
        experiment_config_path: Optional[str] = None
        lake_path: Optional[str] = None
        dineof_input_path: Optional[str] = None
        dineof_output_path: Optional[str] = None
        output_html_folder: Optional[str] = None
        climatology_path: Optional[str] = None
    
    class PostProcessingStep:
        """Minimal base class for standalone use."""
        @property
        def name(self) -> str:
            return self.__class__.__name__
        
        def should_apply(self, ctx, ds) -> bool:
            return True
        
        def apply(self, ctx, ds):
            raise NotImplementedError


# ============================================================================
# Default Configuration - Used as fallback if not specified in experiment JSON
# ============================================================================
DEFAULT_INSITU_CONFIG = {
    "enable": True,
    "buoy_dir": "/gws/ssde/j25b/nceo_uor/users/lcarrea01/INSITU/Buoy_Laura/ALL_FILES_QC",
    "selection_csv": "/home/users/shaerdan/general_purposes/insitu_cv/L3S_QL_MDB_2010_selection_converted.csv",
    "distance_threshold": 0.05,  # degrees (~5.5 km)
    "quality_threshold": 3,
}

# Alias for backward compatibility with standalone script
INSITU_CONFIG = DEFAULT_INSITU_CONFIG


def load_insitu_config_from_experiment(experiment_config_path: Optional[str]) -> Dict:
    """
    Load insitu_validation config from experiment JSON file.
    Falls back to defaults if not found or on error.
    """
    config = DEFAULT_INSITU_CONFIG.copy()
    
    if not experiment_config_path or not os.path.exists(experiment_config_path):
        return config
    
    try:
        with open(experiment_config_path, 'r') as f:
            exp_config = json.load(f)
        
        # Look for insitu_validation section
        insitu_config = exp_config.get("insitu_validation", {})
        
        # Update with values from config file (if present)
        for key in config.keys():
            if key in insitu_config:
                config[key] = insitu_config[key]
        
        print(f"[InsituValidation] Loaded config from: {os.path.basename(experiment_config_path)}")
        
    except Exception as e:
        print(f"[InsituValidation] Could not load experiment config: {e}, using defaults")
    
    return config


# ============================================================================
# Plotting Helpers
# ============================================================================

def compute_year_ticks(time: pd.DatetimeIndex) -> Tuple[List, List]:
    """Compute yearly tick positions and labels."""
    years = np.unique(time.year)
    y_start, y_end = years.min(), years.max()
    
    positions, labels = [], []
    for y in range(y_start, y_end + 1):
        if y in years:
            target = pd.Timestamp(y, 1, 1, 12)
            idx = int(np.argmin(np.abs(time - target)))
            positions.append(time[idx])
            labels.append(str(y))
    
    return positions, labels


def split_timeline(time: pd.DatetimeIndex) -> Tuple[int, int, int]:
    """Split timeline into first and second half."""
    years = np.unique(time.year)
    y_start, y_end = years.min(), years.max()
    mid_year = y_start + (y_end - y_start) // 2
    
    i0 = int(np.searchsorted(time.values, np.datetime64(f"{y_start}-01-01")))
    i1 = int(np.searchsorted(time.values, np.datetime64(f"{mid_year + 1}-01-01")))
    
    return i0, i1, mid_year


def plot_panel(ax, x_vals, center_pixel, seg_slice, y_lim, tick_pos, tick_lab, 
               title, show_legend=False, insitu_dates=None, insitu_temps=None):
    """Plot a single panel with pipeline-style formatting."""
    ax.yaxis.grid(True, linestyle='--', linewidth=0.3)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.3)
    ax.set_ylim(*y_lim)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    
    x_seg = x_vals[seg_slice]
    center = center_pixel[seg_slice]
    
    # Center/buoy pixel (bold red line)
    valid_c = np.isfinite(center)
    if valid_c.any():
        ax.plot(x_seg[valid_c], center[valid_c], '-', color='red', lw=0.8, 
               label='Satellite (buoy pixel)')
    
    # In-situ overlay (blue markers)
    if insitu_dates is not None and insitu_temps is not None:
        seg_start, seg_end = x_seg.min(), x_seg.max()
        in_seg = [(d, t) for d, t in zip(insitu_dates, insitu_temps) 
                  if pd.Timestamp(d) >= seg_start and pd.Timestamp(d) <= seg_end]
        if in_seg:
            seg_dates, seg_temps = zip(*in_seg)
            ax.plot(seg_dates, seg_temps, 'o', color='blue', markersize=4, 
                   alpha=0.8, label='In-situ', zorder=10)
    
    if show_legend:
        ax.legend(fontsize=8, loc='best')


# ============================================================================
# Main Step Class
# ============================================================================

class InsituValidationStep(PostProcessingStep):
    """
    Generate in-situ validation plots at end of postprocessing.
    
    Configuration can be provided:
    1. In the experiment JSON under "insitu_validation" section (recommended)
    2. Passed directly via config parameter (for standalone use)
    3. Falls back to DEFAULT_INSITU_CONFIG if neither provided
    
    Example JSON config:
        "insitu_validation": {
            "enable": true,
            "buoy_dir": "/path/to/buoy/data",
            "selection_csv": "/path/to/selection.csv",
            "distance_threshold": 0.05,
            "quality_threshold": 3
        }
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # Config can be overridden directly, or loaded from experiment JSON in apply()
        self._override_config = config
        self._selection_df = None
    
    @property
    def name(self) -> str:
        return "InsituValidation"
    
    def _get_config(self, ctx: PostContext) -> Dict:
        """Get configuration, prioritizing: override > experiment JSON > defaults."""
        if self._override_config is not None:
            return self._override_config
        
        # Try to load from experiment config
        experiment_config_path = getattr(ctx, 'experiment_config_path', None)
        return load_insitu_config_from_experiment(experiment_config_path)
    
    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        """Check if we have buoy data for this lake. Always returns True to let apply() handle gracefully."""
        try:
            config = self._get_config(ctx)
            
            # Check enable flag
            if not config.get("enable", True):
                print(f"[InsituValidation] Disabled in config, skipping")
                return False
            
            selection_csv = config.get("selection_csv", "")
            buoy_dir = config.get("buoy_dir", "")
            
            if not selection_csv or not os.path.exists(selection_csv):
                print(f"[InsituValidation] Selection CSV not found: {selection_csv}, skipping")
                return False
            if not buoy_dir or not os.path.exists(buoy_dir):
                print(f"[InsituValidation] Buoy directory not found: {buoy_dir}, skipping")
                return False
            return True
        except Exception as e:
            print(f"[InsituValidation] Error in should_apply: {e}")
            return False
    
    def _load_selection_csv(self) -> pd.DataFrame:
        """Load and cache the selection CSV."""
        if self._selection_df is None:
            df = pd.read_csv(self.config["selection_csv"])
            df.columns = df.columns.str.strip()
            time_col = [c for c in df.columns if 'time_IS' in c][0]
            if time_col != 'time_IS':
                df = df.rename(columns={time_col: 'time_IS'})
            df['time_IS'] = pd.to_datetime(df['time_IS'])
            self._selection_df = df
        return self._selection_df
    
    def _get_lake_sites(self, lake_id_cci: int) -> pd.DataFrame:
        """Get all unique sites for a given lake_id_cci from selection CSV."""
        df = self._load_selection_csv()
        lake_df = df[df['lake_id_cci'] == lake_id_cci]
        
        if lake_df.empty:
            return pd.DataFrame()
        
        # Drop duplicates based on lake_id and site_id only (lat/lon may have tiny variations)
        sites = lake_df[['lake_id', 'site_id', 'latitude', 'longitude']].drop_duplicates(
            subset=['lake_id', 'site_id']
        ).copy()
        
        # Convert to int (may be read as float from CSV)
        sites['lake_id'] = sites['lake_id'].astype(int)
        sites['site_id'] = sites['site_id'].astype(int)
        
        return sites.sort_values('site_id').reset_index(drop=True)
    
    def _get_buoy_filepath(self, lake_id: int, site_id: int) -> Optional[str]:
        """Construct path to buoy CSV file."""
        # Ensure int types for proper string formatting
        lake_id = int(lake_id)
        site_id = int(site_id)
        
        buoy_file = os.path.join(
            self.config["buoy_dir"], 
            f"ID{str(lake_id).zfill(6)}{str(site_id).zfill(2)}.csv"
        )
        
        if os.path.exists(buoy_file):
            return buoy_file
        else:
            print(f"[InsituValidation] Buoy file not found: {os.path.basename(buoy_file)}")
            return None
    
    def _determine_representative_hour(self, lake_id: int, site_id: int) -> Optional[int]:
        """Find representative hour for satellite overpasses."""
        df = self._load_selection_csv()
        subset = df[(df['lake_id'] == lake_id) & (df['site_id'] == site_id)].copy()
        
        if subset.empty:
            return None
        
        subset['hour'] = subset['time_IS'].dt.hour
        subset['date'] = subset['time_IS'].dt.date
        hour_day_counts = subset.groupby('hour')['date'].nunique()
        
        return hour_day_counts.idxmax() if not hour_day_counts.empty else None
    
    def _load_buoy_data(self, lake_id: int, site_id: int, 
                        rep_hour: Optional[int]) -> Optional[pd.DataFrame]:
        """Load and filter buoy CSV data."""
        buoy_path = self._get_buoy_filepath(lake_id, site_id)
        if buoy_path is None:
            return None
        
        try:
            df = pd.read_csv(buoy_path, parse_dates=['dateTime'])
        except Exception as e:
            print(f"[InsituValidation] Error reading buoy file: {e}")
            return None
        
        # Detect daily vs hourly
        readings_per_day = df.groupby(df['dateTime'].dt.date).size()
        is_daily = readings_per_day.median() <= 1.5
        
        # Filter for representative hour (hourly data only)
        if rep_hour is not None and not is_daily:
            df = df[df['dateTime'].dt.hour == rep_hour]
        
        # Quality filter
        if 'qcFlag' in df.columns:
            df = df[df['qcFlag'] == 0]
        elif 'q' in df.columns:
            df = df[df['q'] == 0]
        
        return df if not df.empty else None
    
    def _find_nearest_grid_point(self, lat_array: np.ndarray, lon_array: np.ndarray,
                                  target_lat: float, target_lon: float) -> Tuple[Tuple[int, int], float]:
        """Find nearest grid point to target coordinates."""
        lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
        distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
        index = np.unravel_index(np.argmin(distance), distance.shape)
        return index, np.min(distance)
    
    def _extract_matched_temps(self, ds: xr.Dataset, grid_idx: Tuple[int, int],
                                buoy_dates: List) -> Dict:
        """Extract temperatures for dates matching buoy data."""
        time_vals = pd.to_datetime(ds['time'].values)
        date_to_idx = {t.date(): i for i, t in enumerate(time_vals)}
        
        temp_pixel = ds['temp_filled'].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        
        temps, matched_dates = [], []
        for d in buoy_dates:
            if d not in date_to_idx:
                continue
            temp = temp_pixel[date_to_idx[d]]
            if not np.isnan(temp):
                temps.append(temp)
                matched_dates.append(d)
        
        return {'temps': np.array(temps), 'matched_dates': matched_dates}
    
    def _extract_full_timeseries(self, ds: xr.Dataset, grid_idx: Tuple[int, int]) -> Dict:
        """Extract full time series for a pixel."""
        return {
            'times': pd.to_datetime(ds['time'].values),
            'temps': ds['temp_filled'].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        }
    
    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        """Run in-situ validation for this lake. Fail-safe: errors are logged but don't crash pipeline."""
        
        # Always return valid dataset, even if validation fails
        return_ds = ds if ds is not None else xr.Dataset()
        
        try:
            # Load config (from experiment JSON or defaults)
            self.config = self._get_config(ctx)
            
            lake_id_cci = ctx.lake_id
            if lake_id_cci is None or lake_id_cci <= 0:
                print(f"[InsituValidation] No valid lake_id in context, skipping")
                return return_ds
            
            post_dir = os.path.dirname(ctx.output_path) if ctx.output_path else None
            if not post_dir or not os.path.exists(post_dir):
                print(f"[InsituValidation] Post directory not found, skipping")
                return return_ds
            
            # Output to insitu_cv_validation subfolder
            plot_dir = os.path.join(post_dir, "insitu_cv_validation")
            
            print(f"[InsituValidation] Checking for buoy data for lake {lake_id_cci}")
            
            # Find sites for this lake
            try:
                sites = self._get_lake_sites(lake_id_cci)
            except Exception as e:
                print(f"[InsituValidation] Error loading selection CSV: {e}")
                return return_ds
            
            if sites.empty:
                print(f"[InsituValidation] No buoy sites found for lake {lake_id_cci}, skipping")
                return return_ds
            
            print(f"[InsituValidation] Found {len(sites)} site(s) for lake {lake_id_cci}")
            
            # Find pipeline output files
            outputs = self._find_pipeline_outputs(post_dir, lake_id_cci)
            if not any(outputs.values()):
                print(f"[InsituValidation] No output files found in {post_dir}, skipping")
                return return_ds
            
            # Process each site
            for _, site_row in sites.iterrows():
                try:
                    lake_id = site_row['lake_id']
                    site_id = site_row['site_id']
                    site_lat = site_row['latitude']
                    site_lon = site_row['longitude']
                    
                    print(f"[InsituValidation] Processing site {site_id} at ({site_lat:.4f}, {site_lon:.4f})")
                    
                    # Get representative hour
                    rep_hour = self._determine_representative_hour(lake_id, site_id)
                    
                    # Load buoy data
                    buoy_df = self._load_buoy_data(lake_id, site_id, rep_hour)
                    if buoy_df is None or buoy_df.empty:
                        print(f"[InsituValidation] No buoy data for site {site_id}, skipping site")
                        continue
                    
                    # Prepare buoy data
                    buoy_date_temp = buoy_df.groupby(buoy_df['dateTime'].dt.date)['Tw'].mean().to_dict()
                    if not buoy_date_temp:
                        print(f"[InsituValidation] No valid buoy dates for site {site_id}, skipping site")
                        continue
                    
                    unique_dates = list(buoy_date_temp.keys())
                    print(f"[InsituValidation] Buoy has {len(unique_dates)} unique dates")
                    
                    # Process each output file and generate validation
                    self._validate_site(
                        outputs=outputs,
                        site_lat=site_lat,
                        site_lon=site_lon,
                        buoy_date_temp=buoy_date_temp,
                        lake_id=lake_id,
                        lake_id_cci=lake_id_cci,
                        site_id=site_id,
                        plot_dir=plot_dir
                    )
                except Exception as e:
                    print(f"[InsituValidation] Error processing site {site_row.get('site_id', '?')}: {e}")
                    continue
            
            print(f"[InsituValidation] Completed for lake {lake_id_cci}")
            
        except Exception as e:
            print(f"[InsituValidation] Unexpected error (non-fatal): {e}")
        
        return return_ds
    
    def _find_pipeline_outputs(self, post_dir: str, lake_id_cci: int) -> Dict[str, Optional[str]]:
        """Find all pipeline output files for a lake."""
        lake_id_str = f"LAKE{lake_id_cci:09d}"
        outputs = {
            'dineof': None, 'dincae': None, 'eof_filtered': None,
            'interp_full': None, 'eof_filtered_interp_full': None,
        }
        
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
    
    def _validate_site(self, outputs: Dict, site_lat: float, site_lon: float,
                       buoy_date_temp: Dict, lake_id: int, lake_id_cci: int,
                       site_id: int, plot_dir: str):
        """Validate a single site against all output files. Protected against errors."""
        
        try:
            results = {
                'lake_id': lake_id, 'lake_id_cci': lake_id_cci, 'site_id': site_id,
                'buoy_date_temp': buoy_date_temp, 'methods': {}
            }
            
            grid_idx = None
            unique_buoy_dates = list(buoy_date_temp.keys())
            
            for method_name, nc_path in outputs.items():
                if nc_path is None:
                    continue
                
                is_interpolated = 'interp' in method_name
                
                try:
                    ds = xr.open_dataset(nc_path)
                except Exception as e:
                    print(f"[InsituValidation] Error opening {os.path.basename(nc_path)}: {e}")
                    continue
                
                try:
                    # Find grid point (once)
                    if grid_idx is None:
                        grid_idx, distance = self._find_nearest_grid_point(
                            ds['lat'].values, ds['lon'].values, site_lat, site_lon
                        )
                        if distance > self.config["distance_threshold"]:
                            print(f"[InsituValidation] Distance {distance:.4f}° exceeds threshold, skipping site")
                            ds.close()
                            return
                    
                    # Extract matched temperatures
                    extracted = self._extract_matched_temps(ds, grid_idx, unique_buoy_dates)
                    
                    if len(extracted['matched_dates']) == 0:
                        ds.close()
                        continue
                    
                    matched_buoy = np.array([buoy_date_temp[d] for d in extracted['matched_dates']])
                    
                    # Compute statistics
                    diff = extracted['temps'] - matched_buoy
                    rmse = np.sqrt(np.mean(diff**2))
                    bias = np.mean(diff)
                    corr = np.corrcoef(extracted['temps'], matched_buoy)[0, 1] if len(diff) > 1 else np.nan
                    
                    results['methods'][method_name] = {
                        'dates': extracted['matched_dates'],
                        'satellite_temps': extracted['temps'],
                        'insitu_temps': matched_buoy,
                        'difference': diff,
                        'n_matches': len(diff),
                        'rmse': rmse, 'bias': bias, 'correlation': corr,
                        'is_interpolated': is_interpolated,
                    }
                    
                    # For interpolated files, extract full time series
                    if is_interpolated:
                        results['methods'][method_name]['full_timeseries'] = self._extract_full_timeseries(ds, grid_idx)
                    
                    print(f"[InsituValidation] {method_name}: N={len(diff)}, RMSE={rmse:.3f}°C, Bias={bias:.3f}°C")
                    
                except Exception as e:
                    print(f"[InsituValidation] Error processing {method_name}: {e}")
                finally:
                    ds.close()
            
            # Generate plots and save CSV
            if results['methods']:
                try:
                    self._plot_validation(results, plot_dir)
                except Exception as e:
                    print(f"[InsituValidation] Error generating plots: {e}")
                
                try:
                    self._save_summary_csv(results, plot_dir)
                except Exception as e:
                    print(f"[InsituValidation] Error saving CSV: {e}")
        
        except Exception as e:
            print(f"[InsituValidation] Unexpected error in _validate_site: {e}")
    
    def _plot_validation(self, results: Dict, plot_dir: str):
        """Generate validation plots. Protected against errors."""
        try:
            lake_id = results['lake_id']
            site_id = results['site_id']
            methods = results['methods']
            
            if not methods:
                return
            
            sparse_methods = {k: v for k, v in methods.items() if not v.get('is_interpolated', False)}
            interp_methods = {k: v for k, v in methods.items() if v.get('is_interpolated', False)}
            
            # All in-situ dates
            all_insitu_dates = list(results['buoy_date_temp'].keys())
            all_insitu_temps = list(results['buoy_date_temp'].values())
            
            if not all_insitu_dates:
                print(f"[InsituValidation] No in-situ dates to plot")
                return
            
            # Count plots
            n_sparse = len(sparse_methods)
            n_interp = len(interp_methods)
            n_sparse_plots = (1 + n_sparse) if n_sparse > 0 else 0
            n_interp_plots = 3 * n_interp
            n_plots = n_sparse_plots + n_interp_plots
            
            if n_plots == 0:
                return
            
            fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
            if n_plots == 1:
                axes = [axes]
            
            colors = {
                'dineof': 'blue', 'dincae': 'orange', 'eof_filtered': 'green',
                'interp_full': 'purple', 'eof_filtered_interp_full': 'brown', 'insitu': 'red',
            }
            
            plot_idx = 0
            
            # Sparse methods
            if sparse_methods:
                ax = axes[plot_idx]
                plot_idx += 1
                
                for method_name, method_data in sparse_methods.items():
                    ax.plot(method_data['dates'], method_data['satellite_temps'], 
                           label=method_name, marker='o', linestyle='-', 
                           color=colors.get(method_name, 'gray'), markersize=3, alpha=0.8)
                
                ax.plot(all_insitu_dates, all_insitu_temps, label=f'In-situ ({len(all_insitu_dates)} pts)',
                       marker='x', linestyle='None', color=colors['insitu'], markersize=5, alpha=0.7)
                
                ax.set_ylabel('Temperature (°C)')
                ax.set_title(f'Lake {lake_id} Site {site_id} - Sparse Methods vs In-Situ')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Difference plots
                for method_name, method_data in sparse_methods.items():
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    ax.plot(method_data['dates'], method_data['difference'], marker='s', linestyle='-',
                           color=colors.get(method_name, 'gray'), markersize=3)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.fill_between(method_data['dates'], method_data['difference'], 0, alpha=0.3)
                    ax.set_ylabel('Difference (°C)')
                    ax.set_title(f'{method_name}: RMSE={method_data["rmse"]:.3f}°C, '
                                f'Bias={method_data["bias"]:.3f}°C, R={method_data["correlation"]:.3f}')
                    ax.grid(True, alpha=0.3)
            
            # Interpolated methods
            for method_name, method_data in interp_methods.items():
                if 'full_timeseries' not in method_data:
                    continue
                
                full_ts = method_data['full_timeseries']
                times = pd.DatetimeIndex(full_ts['times'])
                temps = full_ts['temps']
                
                if len(times) < 2:
                    continue
                
                i0, i1, mid_year = split_timeline(times)
                if i1 >= len(times) or i0 >= i1:
                    continue
                
                tt0, tt1 = times[i0:i1], times[i1:]
                pos0, lab0 = compute_year_ticks(tt0)
                pos1, lab1 = compute_year_ticks(tt1)
                
                # Y-limits
                finite_temps = temps[np.isfinite(temps)]
                finite_insitu = np.array([t for t in all_insitu_temps if np.isfinite(t)])
                all_vals = np.concatenate([finite_temps, finite_insitu]) if len(finite_temps) > 0 else finite_insitu
                y_lim = (np.min(all_vals) - 1, np.max(all_vals) + 1) if len(all_vals) > 0 else (0, 30)
                
                # First half
                ax = axes[plot_idx]
                plot_idx += 1
                plot_panel(ax, times, temps, slice(i0, i1), y_lim, pos0, lab0,
                          f'{method_name}: Full Daily ({times[i0].year}-{mid_year})',
                          show_legend=True, insitu_dates=all_insitu_dates, insitu_temps=all_insitu_temps)
                ax.set_ylabel('LSWT (°C)')
                
                # Second half
                ax = axes[plot_idx]
                plot_idx += 1
                plot_panel(ax, times, temps, slice(i1, None), y_lim, pos1, lab1,
                          f'{method_name}: Full Daily ({mid_year+1}-{times[-1].year})',
                          insitu_dates=all_insitu_dates, insitu_temps=all_insitu_temps)
                ax.set_ylabel('LSWT (°C)')
                
                # Difference
                ax = axes[plot_idx]
                plot_idx += 1
                ax.plot(method_data['dates'], method_data['difference'], marker='s', linestyle='-',
                       color=colors.get(method_name, 'gray'), markersize=3)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.fill_between(method_data['dates'], method_data['difference'], 0, alpha=0.3)
                ax.set_ylabel('Difference (°C)')
                ax.set_title(f'{method_name}: RMSE={method_data["rmse"]:.3f}°C, '
                            f'Bias={method_data["bias"]:.3f}°C, N={method_data["n_matches"]}')
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Date')
            plt.tight_layout()
            
            os.makedirs(plot_dir, exist_ok=True)
            save_path = os.path.join(plot_dir, f"LAKE{results['lake_id_cci']:09d}_insitu_validation_site{site_id}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"[InsituValidation] Saved: {os.path.basename(save_path)}")
            
        except Exception as e:
            print(f"[InsituValidation] Error in _plot_validation: {e}")
            plt.close('all')  # Clean up any open figures
    
    def _save_summary_csv(self, results: Dict, plot_dir: str):
        """Save summary statistics to CSV. Protected against errors."""
        try:
            rows = []
            for method_name, method_data in results['methods'].items():
                rows.append({
                    'lake_id': results['lake_id'],
                    'lake_id_cci': results['lake_id_cci'],
                    'site_id': results['site_id'],
                    'method': method_name,
                    'n_matches': method_data['n_matches'],
                    'rmse': method_data['rmse'],
                    'bias': method_data['bias'],
                    'correlation': method_data['correlation'],
                })
            
            if rows:
                os.makedirs(plot_dir, exist_ok=True)
                csv_path = os.path.join(plot_dir, f"LAKE{results['lake_id_cci']:09d}_insitu_stats_site{results['site_id']}.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"[InsituValidation] Saved: {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"[InsituValidation] Error saving CSV: {e}")