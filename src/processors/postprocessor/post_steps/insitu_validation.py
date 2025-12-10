"""
In-Situ Validation Step - Generates validation plots comparing gap-filled LSWT to buoy measurements.

This step runs at the end of postprocessing (after LSWTPlotsStep) and creates:
1. Time series comparison plots (obs vs in-situ, recon vs in-situ, with difference panels)
2. Yearly breakdown plots (same structure, one set of 4 rows per year)
3. CSV files with matched data and summary statistics (overall and per-year)

Enhanced stats include: RMSE, MAE, Bias (mean diff), Median diff, STD, RSTD, Correlation, N

Configuration (in experiment JSON):
    "insitu_validation": {
        "enable": true,
        "buoy_dir": "/path/to/buoy/data",
        "selection_csvs": [
            "/path/to/2010_selection.csv",
            "/path/to/2007_selection.csv",
            "/path/to/2018_selection.csv",
            "/path/to/2020_selection.csv"
        ],
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
from datetime import date

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

# Base directory for selection CSVs
_SELECTION_CSV_DIR = "/home/users/shaerdan/general_purposes/insitu_cv"

DEFAULT_INSITU_CONFIG = {
    "enable": True,
    "buoy_dir": "/gws/ssde/j25b/nceo_uor/users/lcarrea01/INSITU/Buoy_Laura/ALL_FILES_QC",
    # List of selection CSVs in priority order (first match wins)
    "selection_csvs": [
        f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2010_selection.csv",
        f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2007_selection.csv",
        f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2018_selection.csv",
        f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2020_selection.csv",
    ],
    # DEPRECATED but kept for backward compatibility - single CSV path
    "selection_csv": None,
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
# Statistics Helpers
# ============================================================================

def compute_stats(satellite_temps: np.ndarray, insitu_temps: np.ndarray) -> Dict:
    """
    Compute comprehensive comparison statistics.
    
    Returns dict with: rmse, mae, bias, median, std, rstd, correlation, n_matches
    """
    if len(satellite_temps) == 0 or len(insitu_temps) == 0:
        return {
            'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'median': np.nan,
            'std': np.nan, 'rstd': np.nan, 'correlation': np.nan, 'n_matches': 0
        }
    
    diff = satellite_temps - insitu_temps
    n = len(diff)
    
    # Basic stats
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    median = np.median(diff)
    std = np.std(diff, ddof=1) if n > 1 else np.nan
    
    # Robust STD (using IQR)
    if n > 1:
        q75, q25 = np.percentile(diff, [75, 25])
        rstd = (q75 - q25) / 1.349  # IQR to std conversion factor
    else:
        rstd = np.nan
    
    # Correlation
    if n > 1:
        corr = np.corrcoef(satellite_temps, insitu_temps)[0, 1]
    else:
        corr = np.nan
    
    return {
        'rmse': rmse, 'mae': mae, 'bias': bias, 'median': median,
        'std': std, 'rstd': rstd, 'correlation': corr, 'n_matches': n
    }


def ensure_celsius(temps: np.ndarray, label: str = "") -> np.ndarray:
    """
    Ensure temperature array is in Celsius.
    
    Detects Kelvin if mean > 100 and converts by subtracting 273.15.
    
    Args:
        temps: Temperature array
        label: Optional label for logging (e.g., 'obs', 'recon')
    
    Returns:
        Temperature array in Celsius
    """
    if len(temps) == 0:
        return temps
    
    # Check if values look like Kelvin (mean > 100)
    finite_temps = temps[np.isfinite(temps)]
    if len(finite_temps) == 0:
        return temps
    
    mean_temp = np.nanmean(finite_temps)
    
    if mean_temp > 100:
        # Looks like Kelvin, convert to Celsius
        if label:
            print(f"[InsituValidation] Converting {label} from Kelvin to Celsius (mean was {mean_temp:.1f}K)")
        return temps - 273.15
    
    return temps


def format_stats_title(stats: Dict, prefix: str = "") -> str:
    """Format stats dict into a title string."""
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(f"N={stats['n_matches']}")
    parts.append(f"RMSE={stats['rmse']:.3f}°C")
    parts.append(f"MAE={stats['mae']:.3f}°C")
    parts.append(f"Bias={stats['bias']:.3f}°C")
    parts.append(f"Med={stats['median']:.3f}°C")
    parts.append(f"STD={stats['std']:.3f}°C")
    parts.append(f"R={stats['correlation']:.3f}")
    return ", ".join(parts)


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
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # Config can be overridden directly, or loaded from experiment JSON in apply()
        self._override_config = config
        # Cache: selection CSV path -> loaded DataFrame
        self._selection_dfs: Dict[str, pd.DataFrame] = {}
        # Cache: lake_id_cci -> (selection_csv_path, DataFrame)
        self._lake_to_selection: Dict[int, Tuple[str, pd.DataFrame]] = {}
    
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
    
    def _get_selection_csv_list(self) -> List[str]:
        """
        Get list of selection CSVs to search, in priority order.
        Supports both new 'selection_csvs' list and legacy 'selection_csv' single path.
        """
        # New format: list of CSVs
        if "selection_csvs" in self.config and self.config["selection_csvs"]:
            csvs = self.config["selection_csvs"]
            if isinstance(csvs, list):
                return [c for c in csvs if c and os.path.exists(c)]
        
        # Legacy format: single CSV
        if "selection_csv" in self.config and self.config["selection_csv"]:
            csv_path = self.config["selection_csv"]
            if os.path.exists(csv_path):
                return [csv_path]
        
        return []
    
    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        """Check if we have buoy data for this lake. Always returns True to let apply() handle gracefully."""
        try:
            config = self._get_config(ctx)
            
            # Check enable flag
            if not config.get("enable", True):
                print(f"[InsituValidation] Disabled in config, skipping")
                return False
            
            buoy_dir = config.get("buoy_dir", "")
            
            if not buoy_dir or not os.path.exists(buoy_dir):
                print(f"[InsituValidation] Buoy directory not found: {buoy_dir}, skipping")
                return False
            
            # Store config for later use
            self.config = config
            
            # Check that at least one selection CSV exists
            csv_list = self._get_selection_csv_list()
            if not csv_list:
                print(f"[InsituValidation] No valid selection CSVs found, skipping")
                return False
            
            print(f"[InsituValidation] Found {len(csv_list)} selection CSV(s) to search")
            return True
            
        except Exception as e:
            print(f"[InsituValidation] Error in should_apply: {e}")
            return False
    
    def _load_selection_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and cache a selection CSV."""
        if csv_path not in self._selection_dfs:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            time_col = [c for c in df.columns if 'time_IS' in c][0]
            if time_col != 'time_IS':
                df = df.rename(columns={time_col: 'time_IS'})
            df['time_IS'] = pd.to_datetime(df['time_IS'])
            self._selection_dfs[csv_path] = df
        return self._selection_dfs[csv_path]
    
    def _find_selection_for_lake(self, lake_id_cci: int) -> Optional[Tuple[str, pd.DataFrame]]:
        """
        Find which selection CSV contains this lake_id_cci.
        Searches through selection_csvs in order, returns first match.
        """
        # Check cache first
        if lake_id_cci in self._lake_to_selection:
            return self._lake_to_selection[lake_id_cci]
        
        # Search through CSVs in priority order
        csv_list = self._get_selection_csv_list()
        
        for csv_path in csv_list:
            try:
                df = self._load_selection_csv(csv_path)
                if lake_id_cci in df['lake_id_cci'].values:
                    csv_name = os.path.basename(csv_path)
                    print(f"[InsituValidation] Lake {lake_id_cci} found in {csv_name}")
                    self._lake_to_selection[lake_id_cci] = (csv_path, df)
                    return (csv_path, df)
            except Exception as e:
                print(f"[InsituValidation] Error loading {csv_path}: {e}")
                continue
        
        # Not found in any CSV
        print(f"[InsituValidation] Lake {lake_id_cci} not found in any selection CSV")
        self._lake_to_selection[lake_id_cci] = None
        return None
    
    def _get_lake_sites(self, lake_id_cci: int) -> pd.DataFrame:
        """Get all unique sites for a given lake_id_cci from the appropriate selection CSV."""
        result = self._find_selection_for_lake(lake_id_cci)
        if result is None:
            return pd.DataFrame()
        
        csv_path, df = result
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
    
    def _determine_representative_hour(self, lake_id: int, site_id: int, lake_id_cci: int) -> Optional[int]:
        """Find representative hour for satellite overpasses."""
        result = self._find_selection_for_lake(lake_id_cci)
        if result is None:
            return None
        
        csv_path, df = result
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
                                buoy_dates: List, var_name: str = 'temp_filled',
                                quality_threshold: Optional[int] = None) -> Dict:
        """
        Extract temperatures for dates matching buoy data.
        
        Args:
            ds: xarray Dataset
            grid_idx: (lat_idx, lon_idx) tuple
            buoy_dates: list of dates to match
            var_name: variable name to extract ('temp_filled' or 'lake_surface_water_temperature')
            quality_threshold: if provided, only include pixels where quality_level >= threshold
                              (only applies when extracting 'lake_surface_water_temperature')
        
        Returns:
            Dict with 'temps' (in Celsius) and 'matched_dates'
        """
        time_vals = pd.to_datetime(ds['time'].values)
        date_to_idx = {t.date(): i for i, t in enumerate(time_vals)}
        
        if var_name not in ds:
            return {'temps': np.array([]), 'matched_dates': []}
        
        temp_pixel = ds[var_name].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        
        # Get quality level if filtering is requested
        quality_pixel = None
        if quality_threshold is not None and 'quality_level' in ds:
            quality_pixel = ds['quality_level'].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        
        temps, matched_dates = [], []
        for d in buoy_dates:
            if d not in date_to_idx:
                continue
            idx = date_to_idx[d]
            temp = temp_pixel[idx]
            
            if np.isnan(temp):
                continue
            
            # Apply quality filter if requested
            if quality_pixel is not None:
                ql = quality_pixel[idx]
                if np.isnan(ql) or ql < quality_threshold:
                    continue
            
            temps.append(temp)
            matched_dates.append(d)
        
        temps_array = np.array(temps)
        
        # Ensure temperatures are in Celsius
        temps_array = ensure_celsius(temps_array, label=var_name)
        
        return {'temps': temps_array, 'matched_dates': matched_dates}
    
    def _extract_full_timeseries(self, ds: xr.Dataset, grid_idx: Tuple[int, int], 
                                  var_name: str = 'temp_filled') -> Dict:
        """Extract full time series for a pixel, ensuring Celsius."""
        if var_name not in ds:
            return {'times': pd.DatetimeIndex([]), 'temps': np.array([])}
        
        temps = ds[var_name].isel(lat=grid_idx[0], lon=grid_idx[1]).values
        temps = ensure_celsius(temps, label=f"{var_name}_timeseries")
        
        return {
            'times': pd.to_datetime(ds['time'].values),
            'temps': temps
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
            
            # Log quality threshold being used for observation filtering
            quality_threshold = self.config.get("quality_threshold", 3)
            print(f"[InsituValidation] Checking for buoy data for lake {lake_id_cci}")
            print(f"[InsituValidation] Using quality_threshold >= {quality_threshold} for observation filtering")
            
            # Find sites for this lake (now searches across multiple selection CSVs)
            try:
                sites = self._get_lake_sites(lake_id_cci)
            except Exception as e:
                print(f"[InsituValidation] Error finding sites: {e}")
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
                    
                    # Get representative hour (now passes lake_id_cci for proper CSV lookup)
                    rep_hour = self._determine_representative_hour(lake_id, site_id, lake_id_cci)
                    
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
        """Validate a single site against all output files."""
        
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
                    
                    # Extract RECONSTRUCTION (temp_filled) - no quality filter needed, already filtered during preprocessing
                    recon_extracted = self._extract_matched_temps(ds, grid_idx, unique_buoy_dates, 'temp_filled')
                    
                    # Extract OBSERVATION (lake_surface_water_temperature) with quality filter
                    # Use quality_threshold from config to match what was used in preprocessing
                    quality_threshold = self.config.get("quality_threshold", 3)
                    obs_extracted = self._extract_matched_temps(
                        ds, grid_idx, unique_buoy_dates, 'lake_surface_water_temperature',
                        quality_threshold=quality_threshold
                    )
                    
                    # Store reconstruction data
                    if len(recon_extracted['matched_dates']) > 0:
                        matched_buoy_recon = np.array([buoy_date_temp[d] for d in recon_extracted['matched_dates']])
                        recon_stats = compute_stats(recon_extracted['temps'], matched_buoy_recon)
                        
                        results['methods'][method_name] = {
                            'recon': {
                                'dates': recon_extracted['matched_dates'],
                                'satellite_temps': recon_extracted['temps'],
                                'insitu_temps': matched_buoy_recon,
                                'difference': recon_extracted['temps'] - matched_buoy_recon,
                                **recon_stats,
                            },
                            'is_interpolated': is_interpolated,
                        }
                        
                        # For interpolated files, extract full time series
                        if is_interpolated:
                            results['methods'][method_name]['recon']['full_timeseries'] = self._extract_full_timeseries(ds, grid_idx, 'temp_filled')
                        
                        print(f"[InsituValidation] {method_name} (recon): N={recon_stats['n_matches']}, RMSE={recon_stats['rmse']:.3f}°C, Bias={recon_stats['bias']:.3f}°C")
                    
                    # Store observation data (only for sparse methods - obs doesn't exist in interp files)
                    if len(obs_extracted['matched_dates']) > 0 and not is_interpolated:
                        matched_buoy_obs = np.array([buoy_date_temp[d] for d in obs_extracted['matched_dates']])
                        obs_stats = compute_stats(obs_extracted['temps'], matched_buoy_obs)
                        
                        if method_name not in results['methods']:
                            results['methods'][method_name] = {'is_interpolated': is_interpolated}
                        
                        results['methods'][method_name]['obs'] = {
                            'dates': obs_extracted['matched_dates'],
                            'satellite_temps': obs_extracted['temps'],
                            'insitu_temps': matched_buoy_obs,
                            'difference': obs_extracted['temps'] - matched_buoy_obs,
                            **obs_stats,
                        }
                        
                        print(f"[InsituValidation] {method_name} (obs): N={obs_stats['n_matches']}, RMSE={obs_stats['rmse']:.3f}°C, Bias={obs_stats['bias']:.3f}°C")
                    
                except Exception as e:
                    print(f"[InsituValidation] Error processing {method_name}: {e}")
                finally:
                    ds.close()
            
            # Generate plots and save CSV
            if results['methods']:
                try:
                    self._plot_validation(results, plot_dir)
                except Exception as e:
                    print(f"[InsituValidation] Error generating full plots: {e}")
                
                try:
                    self._plot_yearly_validation(results, plot_dir)
                except Exception as e:
                    print(f"[InsituValidation] Error generating yearly plots: {e}")
                
                try:
                    self._save_summary_csv(results, plot_dir)
                except Exception as e:
                    print(f"[InsituValidation] Error saving CSV: {e}")
                
                try:
                    self._save_yearly_csv(results, plot_dir)
                except Exception as e:
                    print(f"[InsituValidation] Error saving yearly CSV: {e}")
        
        except Exception as e:
            print(f"[InsituValidation] Unexpected error in _validate_site: {e}")
    
    def _plot_validation(self, results: Dict, plot_dir: str):
        """
        Generate full time series validation plots.
        Structure: For each method, 4 rows:
          1. Obs vs In-situ
          2. Obs - In-situ difference
          3. Recon vs In-situ  
          4. Recon - In-situ difference
        """
        try:
            lake_id = results['lake_id']
            lake_id_cci = results['lake_id_cci']
            site_id = results['site_id']
            methods = results['methods']
            
            if not methods:
                return
            
            # All in-situ data
            all_insitu_dates = list(results['buoy_date_temp'].keys())
            all_insitu_temps = list(results['buoy_date_temp'].values())
            
            if not all_insitu_dates:
                return
            
            # Count total panels needed
            n_panels = 0
            for method_name, method_data in methods.items():
                has_obs = 'obs' in method_data
                has_recon = 'recon' in method_data
                if has_obs:
                    n_panels += 2  # obs vs insitu + diff
                if has_recon:
                    n_panels += 2  # recon vs insitu + diff
            
            if n_panels == 0:
                return
            
            fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels))
            if n_panels == 1:
                axes = [axes]
            
            colors = {'obs': 'green', 'recon': 'blue', 'insitu': 'red', 'diff': 'purple'}
            plot_idx = 0
            
            for method_name, method_data in methods.items():
                has_obs = 'obs' in method_data
                has_recon = 'recon' in method_data
                
                # --- Observation panels ---
                if has_obs:
                    obs_data = method_data['obs']
                    obs_dates = obs_data['dates']
                    obs_temps = obs_data['satellite_temps']
                    obs_insitu = obs_data['insitu_temps']
                    obs_diff = obs_data['difference']
                    obs_stats = {k: obs_data[k] for k in ['rmse', 'mae', 'bias', 'median', 'std', 'rstd', 'correlation', 'n_matches']}
                    
                    # Panel 1: Obs vs In-situ
                    ax = axes[plot_idx]
                    plot_idx += 1
                    ax.plot(obs_dates, obs_temps, 'o-', color=colors['obs'], markersize=3, label='Observation', alpha=0.8)
                    ax.plot(all_insitu_dates, all_insitu_temps, 'x', color=colors['insitu'], markersize=4, label='In-situ', alpha=0.7)
                    ax.set_ylabel('Temperature (°C)')
                    ax.set_title(f'{method_name} - Observation vs In-situ (N={obs_stats["n_matches"]})')
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Panel 2: Obs - In-situ difference
                    ax = axes[plot_idx]
                    plot_idx += 1
                    ax.plot(obs_dates, obs_diff, 's-', color=colors['diff'], markersize=3)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.fill_between(obs_dates, obs_diff, 0, alpha=0.3, color=colors['diff'])
                    ax.set_ylabel('Obs - In-situ (°C)')
                    ax.set_title(format_stats_title(obs_stats, f'{method_name} Obs-Insitu:'))
                    ax.grid(True, alpha=0.3)
                
                # --- Reconstruction panels ---
                if has_recon:
                    recon_data = method_data['recon']
                    recon_dates = recon_data['dates']
                    recon_temps = recon_data['satellite_temps']
                    recon_insitu = recon_data['insitu_temps']
                    recon_diff = recon_data['difference']
                    recon_stats = {k: recon_data[k] for k in ['rmse', 'mae', 'bias', 'median', 'std', 'rstd', 'correlation', 'n_matches']}
                    
                    # Panel 3: Recon vs In-situ
                    ax = axes[plot_idx]
                    plot_idx += 1
                    ax.plot(recon_dates, recon_temps, 'o-', color=colors['recon'], markersize=3, label='Reconstruction', alpha=0.8)
                    ax.plot(all_insitu_dates, all_insitu_temps, 'x', color=colors['insitu'], markersize=4, label='In-situ', alpha=0.7)
                    ax.set_ylabel('Temperature (°C)')
                    ax.set_title(f'{method_name} - Reconstruction vs In-situ (N={recon_stats["n_matches"]})')
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Panel 4: Recon - In-situ difference
                    ax = axes[plot_idx]
                    plot_idx += 1
                    ax.plot(recon_dates, recon_diff, 's-', color=colors['diff'], markersize=3)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.fill_between(recon_dates, recon_diff, 0, alpha=0.3, color=colors['diff'])
                    ax.set_ylabel('Recon - In-situ (°C)')
                    ax.set_title(format_stats_title(recon_stats, f'{method_name} Recon-Insitu:'))
                    ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Date')
            plt.tight_layout()
            
            os.makedirs(plot_dir, exist_ok=True)
            save_path = os.path.join(plot_dir, f"LAKE{lake_id_cci:09d}_insitu_validation_site{site_id}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"[InsituValidation] Saved: {os.path.basename(save_path)}")
            
        except Exception as e:
            print(f"[InsituValidation] Error in _plot_validation: {e}")
            plt.close('all')
    
    def _plot_yearly_validation(self, results: Dict, plot_dir: str):
        """
        Generate yearly breakdown plots - one file per method.
        
        For each method (dineof, dincae, interp_full, etc.), creates a separate file:
          LAKE{id}_insitu_validation_yearly_{method}_site{N}.png
        
        For sparse methods (with obs): 4 rows per year
          1. Obs vs In-situ
          2. Obs - In-situ difference  
          3. Recon vs In-situ
          4. Recon - In-situ difference
        
        For interpolated methods (no obs): 2 rows per year
          1. Recon vs In-situ
          2. Recon - In-situ difference
        """
        try:
            lake_id = results['lake_id']
            lake_id_cci = results['lake_id_cci']
            site_id = results['site_id']
            methods = results['methods']
            buoy_date_temp = results['buoy_date_temp']
            
            if not methods:
                return
            
            # Find all years with in-situ data
            all_insitu_years = set()
            for d in buoy_date_temp.keys():
                all_insitu_years.add(d.year)
            
            os.makedirs(plot_dir, exist_ok=True)
            colors = {'obs': 'green', 'recon': 'blue', 'insitu': 'red', 'diff': 'purple'}
            
            # Generate one plot file per method
            for method_name, method_data in methods.items():
                try:
                    has_obs = 'obs' in method_data
                    has_recon = 'recon' in method_data
                    
                    if not has_recon:
                        # No reconstruction data, skip this method
                        continue
                    
                    # Find years with data for this method
                    method_years = set(all_insitu_years)  # Start with in-situ years
                    if has_recon:
                        for d in method_data['recon']['dates']:
                            method_years.add(d.year)
                    if has_obs:
                        for d in method_data['obs']['dates']:
                            method_years.add(d.year)
                    
                    method_years = sorted(method_years)
                    
                    if not method_years:
                        continue
                    
                    # Calculate rows needed
                    rows_per_year = (2 if has_obs else 0) + 2  # obs panels + recon panels
                    n_years = len(method_years)
                    n_rows = n_years * rows_per_year
                    
                    if n_rows == 0:
                        continue
                    
                    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.5 * n_rows))
                    if n_rows == 1:
                        axes = [axes]
                    
                    plot_idx = 0
                    
                    for year in method_years:
                        # Filter in-situ data for this year
                        year_buoy = {d: t for d, t in buoy_date_temp.items() if d.year == year}
                        year_insitu_dates = list(year_buoy.keys())
                        year_insitu_temps = list(year_buoy.values())
                        
                        # --- Observation panels for this year (sparse methods only) ---
                        if has_obs:
                            obs_data = method_data['obs']
                            year_obs_mask = np.array([d.year == year for d in obs_data['dates']])
                            year_obs_dates = [d for d, m in zip(obs_data['dates'], year_obs_mask) if m]
                            year_obs_temps = obs_data['satellite_temps'][year_obs_mask]
                            year_obs_insitu = obs_data['insitu_temps'][year_obs_mask]
                            
                            if len(year_obs_temps) > 0:
                                year_obs_diff = year_obs_temps - year_obs_insitu
                                year_obs_stats = compute_stats(year_obs_temps, year_obs_insitu)
                            else:
                                year_obs_diff = np.array([])
                                year_obs_stats = compute_stats(np.array([]), np.array([]))
                            
                            # Panel: Obs vs In-situ
                            ax = axes[plot_idx]
                            plot_idx += 1
                            if len(year_obs_dates) > 0:
                                ax.plot(year_obs_dates, year_obs_temps, 'o-', color=colors['obs'], markersize=4, label='Obs', alpha=0.8)
                            if year_insitu_dates:
                                ax.plot(year_insitu_dates, year_insitu_temps, 'x', color=colors['insitu'], markersize=5, label='In-situ', alpha=0.7)
                            ax.set_ylabel('Temp (°C)')
                            ax.set_title(f'{year} - {method_name} Obs vs In-situ (N={year_obs_stats["n_matches"]})')
                            ax.legend(loc='best', fontsize=7)
                            ax.grid(True, alpha=0.3)
                            
                            # Panel: Obs - In-situ diff
                            ax = axes[plot_idx]
                            plot_idx += 1
                            if len(year_obs_dates) > 0 and len(year_obs_diff) > 0:
                                ax.plot(year_obs_dates, year_obs_diff, 's-', color=colors['diff'], markersize=3)
                                ax.fill_between(year_obs_dates, year_obs_diff, 0, alpha=0.3, color=colors['diff'])
                            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                            ax.set_ylabel('Obs-Insitu (°C)')
                            ax.set_title(format_stats_title(year_obs_stats, f'{year} Obs-Insitu:'))
                            ax.grid(True, alpha=0.3)
                        
                        # --- Reconstruction panels for this year ---
                        recon_data = method_data['recon']
                        year_recon_mask = np.array([d.year == year for d in recon_data['dates']])
                        year_recon_dates = [d for d, m in zip(recon_data['dates'], year_recon_mask) if m]
                        year_recon_temps = recon_data['satellite_temps'][year_recon_mask]
                        year_recon_insitu = recon_data['insitu_temps'][year_recon_mask]
                        
                        if len(year_recon_temps) > 0:
                            year_recon_diff = year_recon_temps - year_recon_insitu
                            year_recon_stats = compute_stats(year_recon_temps, year_recon_insitu)
                        else:
                            year_recon_diff = np.array([])
                            year_recon_stats = compute_stats(np.array([]), np.array([]))
                        
                        # Panel: Recon vs In-situ
                        ax = axes[plot_idx]
                        plot_idx += 1
                        if len(year_recon_dates) > 0:
                            ax.plot(year_recon_dates, year_recon_temps, 'o-', color=colors['recon'], markersize=4, label='Recon', alpha=0.8)
                        if year_insitu_dates:
                            ax.plot(year_insitu_dates, year_insitu_temps, 'x', color=colors['insitu'], markersize=5, label='In-situ', alpha=0.7)
                        ax.set_ylabel('Temp (°C)')
                        ax.set_title(f'{year} - {method_name} Recon vs In-situ (N={year_recon_stats["n_matches"]})')
                        ax.legend(loc='best', fontsize=7)
                        ax.grid(True, alpha=0.3)
                        
                        # Panel: Recon - In-situ diff
                        ax = axes[plot_idx]
                        plot_idx += 1
                        if len(year_recon_dates) > 0 and len(year_recon_diff) > 0:
                            ax.plot(year_recon_dates, year_recon_diff, 's-', color=colors['diff'], markersize=3)
                            ax.fill_between(year_recon_dates, year_recon_diff, 0, alpha=0.3, color=colors['diff'])
                        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                        ax.set_ylabel('Recon-Insitu (°C)')
                        ax.set_title(format_stats_title(year_recon_stats, f'{year} Recon-Insitu:'))
                        ax.grid(True, alpha=0.3)
                    
                    axes[-1].set_xlabel('Date')
                    plt.tight_layout()
                    
                    save_path = os.path.join(plot_dir, f"LAKE{lake_id_cci:09d}_insitu_validation_yearly_{method_name}_site{site_id}.png")
                    plt.savefig(save_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f"[InsituValidation] Saved: {os.path.basename(save_path)}")
                    
                except Exception as e:
                    print(f"[InsituValidation] Error generating yearly plot for {method_name}: {e}")
                    plt.close('all')
                    continue
            
        except Exception as e:
            print(f"[InsituValidation] Error in _plot_yearly_validation: {e}")
            plt.close('all')
    
    def _save_summary_csv(self, results: Dict, plot_dir: str):
        """Save overall summary statistics to CSV."""
        try:
            rows = []
            for method_name, method_data in results['methods'].items():
                # Observation stats
                if 'obs' in method_data:
                    obs = method_data['obs']
                    rows.append({
                        'lake_id': results['lake_id'],
                        'lake_id_cci': results['lake_id_cci'],
                        'site_id': results['site_id'],
                        'method': method_name,
                        'data_type': 'observation',
                        'n_matches': obs['n_matches'],
                        'rmse': obs['rmse'],
                        'mae': obs['mae'],
                        'bias': obs['bias'],
                        'median': obs['median'],
                        'std': obs['std'],
                        'rstd': obs['rstd'],
                        'correlation': obs['correlation'],
                    })
                
                # Reconstruction stats
                if 'recon' in method_data:
                    recon = method_data['recon']
                    rows.append({
                        'lake_id': results['lake_id'],
                        'lake_id_cci': results['lake_id_cci'],
                        'site_id': results['site_id'],
                        'method': method_name,
                        'data_type': 'reconstruction',
                        'n_matches': recon['n_matches'],
                        'rmse': recon['rmse'],
                        'mae': recon['mae'],
                        'bias': recon['bias'],
                        'median': recon['median'],
                        'std': recon['std'],
                        'rstd': recon['rstd'],
                        'correlation': recon['correlation'],
                    })
            
            if rows:
                os.makedirs(plot_dir, exist_ok=True)
                csv_path = os.path.join(plot_dir, f"LAKE{results['lake_id_cci']:09d}_insitu_stats_site{results['site_id']}.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"[InsituValidation] Saved: {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"[InsituValidation] Error saving CSV: {e}")
    
    def _save_yearly_csv(self, results: Dict, plot_dir: str):
        """Save per-year statistics to CSV."""
        try:
            rows = []
            buoy_date_temp = results['buoy_date_temp']
            
            # Find all years
            all_years = set()
            for d in buoy_date_temp.keys():
                all_years.add(d.year)
            for method_name, method_data in results['methods'].items():
                if 'recon' in method_data:
                    for d in method_data['recon']['dates']:
                        all_years.add(d.year)
                if 'obs' in method_data:
                    for d in method_data['obs']['dates']:
                        all_years.add(d.year)
            
            all_years = sorted(all_years)
            
            for method_name, method_data in results['methods'].items():
                for year in all_years:
                    # Observation stats for this year
                    if 'obs' in method_data:
                        obs_data = method_data['obs']
                        year_mask = np.array([d.year == year for d in obs_data['dates']])
                        if year_mask.any():
                            year_temps = obs_data['satellite_temps'][year_mask]
                            year_insitu = obs_data['insitu_temps'][year_mask]
                            stats = compute_stats(year_temps, year_insitu)
                            rows.append({
                                'lake_id': results['lake_id'],
                                'lake_id_cci': results['lake_id_cci'],
                                'site_id': results['site_id'],
                                'year': year,
                                'method': method_name,
                                'data_type': 'observation',
                                **stats,
                            })
                    
                    # Reconstruction stats for this year
                    if 'recon' in method_data:
                        recon_data = method_data['recon']
                        year_mask = np.array([d.year == year for d in recon_data['dates']])
                        if year_mask.any():
                            year_temps = recon_data['satellite_temps'][year_mask]
                            year_insitu = recon_data['insitu_temps'][year_mask]
                            stats = compute_stats(year_temps, year_insitu)
                            rows.append({
                                'lake_id': results['lake_id'],
                                'lake_id_cci': results['lake_id_cci'],
                                'site_id': results['site_id'],
                                'year': year,
                                'method': method_name,
                                'data_type': 'reconstruction',
                                **stats,
                            })
            
            if rows:
                os.makedirs(plot_dir, exist_ok=True)
                csv_path = os.path.join(plot_dir, f"LAKE{results['lake_id_cci']:09d}_insitu_stats_yearly_site{results['site_id']}.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"[InsituValidation] Saved: {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"[InsituValidation] Error saving yearly CSV: {e}")