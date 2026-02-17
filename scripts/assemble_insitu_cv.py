#!/usr/bin/env python3
"""
In-Situ CV Assembly & Statistics
=================================

Assembles all in-situ (buoy) validation pairs from all lakes into a single
file, then computes per-lake and global statistics.

For each lake with buoy data, extracts matched satellite-buoy temperature
pairs from all 6 reconstruction method variants:
    1. dineof               — DINEOF baseline (sparse)
    2. eof_filtered         — DINEOF EOF-filtered (sparse)
    3. interp_full          — DINEOF baseline interpolated to daily
    4. eof_filtered_interp_full — DINEOF EOF-filtered interpolated to daily
    5. dincae               — DINCAE baseline (sparse)
    6. dincae_interp_full   — DINCAE baseline interpolated to daily

Two data_types per non-interpolated method:
    - observation: raw satellite LSWT (quality_level >= 3) vs buoy
    - reconstruction: temp_filled (gap-filled) vs buoy

For reconstruction, also tags each pair as was_observed (True/False),
enabling sub-aggregation into reconstruction_observed and reconstruction_missing.

Observation data_type is skipped for interpolated methods (no original
observations in interpolated files).

Flags per pair:
    - quality_level: satellite QL (3, 4, or 5) — observation data_type only
    - was_observed: whether original satellite obs existed at this date
    - is_temporally_interpolated: whether this date was filled by temporal
      interpolation (daily methods only; cross-refs sparse parent)
    - is_eof_filtered_replaced: did EOF filtering change the value?
    - eof_filter_delta: magnitude of change from EOF filtering

IMPORTANT: Observation data_type ALWAYS requires quality_level >= 3.

Outputs:
    1. Assembly file (one row per matched pair): Parquet + CSV
    2. Per-lake stats CSV (by method × data_type)
    3. Global stats CSV (pooled, and by year/month/season/hemisphere/QL)

Author: Shaerdan / NCEO / University of Reading
Date: February 2026
"""

import argparse
import os
import sys
import time
from datetime import date as dt_date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr

# Import completion check for fair comparison filtering
try:
    from completion_check import get_fair_comparison_lakes, CompletionSummary
    HAS_COMPLETION_CHECK = True
except ImportError:
    HAS_COMPLETION_CHECK = False


# =============================================================================
# Constants
# =============================================================================

# Default paths (JASMIN)
_SELECTION_CSV_DIR = "/home/users/shaerdan/general_purposes/insitu_cv"
DEFAULT_SELECTION_CSVS = [
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2010_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2007_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2018_selection.csv",
    f"{_SELECTION_CSV_DIR}/L3S_QL_MDB_2020_selection.csv",
]
DEFAULT_BUOY_DIR = "/gws/ssde/j25b/nceo_uor/users/lcarrea01/INSITU/Buoy_Laura/ALL_FILES_QC"

# Method variants: key -> (file suffix, is_interpolated, is_eof_filtered)
METHOD_VARIANTS = {
    'dineof':                     ('_dineof.nc',                          False, False),
    'eof_filtered':               ('_dineof_eof_filtered.nc',            False, True),
    'interp_full':                ('_dineof_eof_interp_full.nc',         True,  False),
    'eof_filtered_interp_full':   ('_dineof_eof_filtered_interp_full.nc', True,  True),
    'dincae':                     ('_dincae.nc',                          False, False),
    'dincae_interp_full':         ('_dincae_interp_full.nc',             True,  False),
}

# Parent sparse file for each interpolated method (to compute is_temporally_interpolated)
INTERP_PARENTS = {
    'interp_full':              'dineof',
    'eof_filtered_interp_full': 'eof_filtered',
    'dincae_interp_full':       'dincae',
}

# Quality level threshold (HARD requirement for observation data_type)
QUALITY_LEVEL_THRESHOLD = 3

# Distance threshold for grid-buoy matching (degrees)
DISTANCE_THRESHOLD = 0.05

# Metrics
METRIC_NAMES = [
    'rmse', 'mae', 'bias', 'median', 'std', 'rstd', 'iqr',
    'n', 'correlation', 'r_squared', 'fraction_within_1K',
]


# =============================================================================
# Metric computation (same as satellite CV script)
# =============================================================================

def compute_metrics(diff: np.ndarray,
                    original: np.ndarray = None,
                    reconstructed: np.ndarray = None) -> Dict[str, float]:
    """Compute all metrics from a signed diff vector."""
    n = len(diff)
    if n == 0:
        return {m: float('nan') for m in METRIC_NAMES}

    q75, q25 = np.percentile(diff, [75, 25])
    iqr_val = float(q75 - q25)

    metrics = {
        'rmse':     float(np.sqrt(np.mean(diff ** 2))),
        'mae':      float(np.mean(np.abs(diff))),
        'bias':     float(np.mean(diff)),
        'median':   float(np.median(diff)),
        'std':      float(np.std(diff, ddof=1)) if n > 1 else float('nan'),
        'rstd':     float(iqr_val / 1.349) if n > 1 else float('nan'),
        'iqr':      iqr_val,
        'n':        n,
        'fraction_within_1K': float(np.mean(np.abs(diff) < 1.0)),
    }

    if original is not None and reconstructed is not None and n > 1:
        finite = np.isfinite(original) & np.isfinite(reconstructed)
        if finite.sum() > 1:
            corr = np.corrcoef(original[finite], reconstructed[finite])[0, 1]
            metrics['correlation'] = float(corr) if np.isfinite(corr) else float('nan')
            metrics['r_squared'] = float(corr ** 2) if np.isfinite(corr) else float('nan')
        else:
            metrics['correlation'] = float('nan')
            metrics['r_squared'] = float('nan')
    else:
        metrics['correlation'] = float('nan')
        metrics['r_squared'] = float('nan')

    return metrics


def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Compute metrics from a DataFrame with diff, insitu_temp, satellite_temp."""
    diff = df['diff'].dropna().values
    insitu = df['insitu_temp'].values if 'insitu_temp' in df.columns else None
    sat = df['satellite_temp'].values if 'satellite_temp' in df.columns else None
    return compute_metrics(diff, insitu, sat)


def assign_season(month: int) -> str:
    if month in (12, 1, 2): return 'DJF'
    if month in (3, 4, 5): return 'MAM'
    if month in (6, 7, 8): return 'JJA'
    return 'SON'


def assign_hemisphere(lat: float) -> str:
    return 'N' if lat >= 0 else 'S'


def ensure_celsius(temps: np.ndarray) -> np.ndarray:
    """Convert Kelvin to Celsius if needed."""
    if len(temps) == 0:
        return temps
    finite = temps[np.isfinite(temps)]
    if len(finite) > 0 and np.nanmean(finite) > 100:
        return temps - 273.15
    return temps


# =============================================================================
# Selection CSV & buoy data loading
# =============================================================================

def load_selection_csvs(csv_paths: List[str]) -> List[pd.DataFrame]:
    """Load and parse all selection CSVs."""
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            time_col = [c for c in df.columns if 'time_IS' in c]
            if time_col:
                if time_col[0] != 'time_IS':
                    df = df.rename(columns={time_col[0]: 'time_IS'})
                df['time_IS'] = pd.to_datetime(df['time_IS'])
            df['_source_csv'] = os.path.basename(path)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {path}: {e}")
    return dfs


def get_lake_sites(lake_id_cci: int, selection_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Get all unique buoy sites for a lake across all selection CSVs."""
    all_rows = []
    for df in selection_dfs:
        if 'lake_id_cci' in df.columns and lake_id_cci in df['lake_id_cci'].values:
            lake_df = df[df['lake_id_cci'] == lake_id_cci]
            rows = lake_df[['lake_id', 'site_id', 'latitude', 'longitude']].copy()
            all_rows.append(rows)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # For each (lake_id, site_id), find most frequent (lat, lon)
    result_rows = []
    for (lake_id, site_id), group in combined.groupby(['lake_id', 'site_id']):
        lat_lon_counts = group.groupby(['latitude', 'longitude']).size()
        mode_lat, mode_lon = lat_lon_counts.idxmax()
        result_rows.append({
            'lake_id': int(lake_id),
            'site_id': int(site_id),
            'latitude': mode_lat,
            'longitude': mode_lon,
        })

    return pd.DataFrame(result_rows).sort_values('site_id').reset_index(drop=True)


def determine_representative_hour(lake_id: int, site_id: int, lake_id_cci: int,
                                   selection_dfs: List[pd.DataFrame]) -> Optional[int]:
    """Find representative hour for satellite overpasses."""
    all_subsets = []
    for df in selection_dfs:
        if 'lake_id_cci' not in df.columns:
            continue
        subset = df[(df['lake_id'] == lake_id) & (df['site_id'] == site_id)]
        if not subset.empty:
            all_subsets.append(subset)

    if not all_subsets:
        return None

    combined = pd.concat(all_subsets, ignore_index=True)
    if 'time_IS' not in combined.columns:
        return None

    combined['hour'] = combined['time_IS'].dt.hour
    combined['date'] = combined['time_IS'].dt.date
    hour_day_counts = combined.groupby('hour')['date'].nunique()
    return hour_day_counts.idxmax() if not hour_day_counts.empty else None


def load_buoy_data(lake_id: int, site_id: int, buoy_dir: str,
                   rep_hour: Optional[int] = None) -> Optional[Dict]:
    """Load and filter buoy CSV data."""
    buoy_file = os.path.join(
        buoy_dir,
        f"ID{str(int(lake_id)).zfill(6)}{str(int(site_id)).zfill(2)}.csv"
    )
    if not os.path.exists(buoy_file):
        return None

    try:
        df = pd.read_csv(buoy_file, parse_dates=['dateTime'])
    except Exception as e:
        print(f"    Error reading buoy file: {e}")
        return None

    # Extract source lat/lon (median to handle drift)
    source_lat = float(df['lat'].median()) if 'lat' in df.columns else None
    source_lon = float(df['lon'].median()) if 'lon' in df.columns else None

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

    if df.empty:
        return None

    # Build date→temp dict (mean per day)
    buoy_date_temp = df.groupby(df['dateTime'].dt.date)['Tw'].mean().to_dict()

    return {
        'buoy_date_temp': buoy_date_temp,
        'source_lat': source_lat,
        'source_lon': source_lon,
    }


# =============================================================================
# Grid point finding
# =============================================================================

def find_nearest_grid_point(lat_array: np.ndarray, lon_array: np.ndarray,
                            target_lat: float, target_lon: float) -> Tuple[Tuple[int, int], float]:
    """Find nearest grid point. Returns ((lat_idx, lon_idx), distance_degrees)."""
    target_lat = np.float64(target_lat)
    target_lon = np.float64(target_lon)
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    index = np.unravel_index(np.argmin(distance), distance.shape)
    return index, float(np.min(distance))


# =============================================================================
# Per-lake extraction
# =============================================================================

def extract_lake(run_root: str, lake_id_cci: int, alpha: str,
                 selection_dfs: List[pd.DataFrame], buoy_dir: str,
                 verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Extract all in-situ matched pairs for one lake across all methods and sites.
    """
    lake_id9 = f"{lake_id_cci:09d}"

    # Find post directory
    post_dir = None
    for lake_str in [lake_id9, str(lake_id_cci)]:
        candidate = os.path.join(run_root, "post", lake_str, alpha)
        if os.path.exists(candidate):
            post_dir = candidate
            break
    if post_dir is None:
        return None

    # Find sites
    sites = get_lake_sites(lake_id_cci, selection_dfs)
    if sites.empty:
        return None

    if verbose:
        print(f"  [Lake {lake_id_cci}] {len(sites)} site(s)")

    # Discover method files
    method_files = {}
    for method_key, (suffix, _, _) in METHOD_VARIANTS.items():
        candidates = list(Path(post_dir).glob(f"*{suffix}"))
        if candidates:
            method_files[method_key] = str(candidates[0])

    if not method_files:
        if verbose:
            print(f"  [Lake {lake_id_cci}] No method files found")
        return None

    # Find prepared.nc for was_observed flag
    prepared_path = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
    if not os.path.exists(prepared_path):
        prepared_path = None

    # Load prepared.nc observed mask (once per lake)
    prep_obs_mask = None
    prep_date_to_idx = None
    prep_lat = None
    prep_lon = None
    if prepared_path:
        try:
            ds_prep = xr.open_dataset(prepared_path, decode_times=False)
            prep_data = ds_prep['lake_surface_water_temperature'].values
            prep_obs_mask = ~np.isnan(prep_data)

            raw_times = ds_prep['time'].values
            time_units = ds_prep.attrs.get('time_units', None)
            if time_units and 'days since' in time_units:
                base_str = time_units.replace('days since ', '').strip()
                base_time = pd.Timestamp(base_str)
                prep_times = base_time + pd.to_timedelta(raw_times, unit='D')
                prep_date_to_idx = {t.date(): i for i, t in enumerate(prep_times)}
            prep_lat = ds_prep['lat'].values
            prep_lon = ds_prep['lon'].values
            ds_prep.close()
        except Exception as e:
            if verbose:
                print(f"  [Lake {lake_id_cci}] Error loading prepared.nc: {e}")

    # Pre-load sparse method data for cross-referencing flags
    # We need: sparse file time axes, and sparse file pixel values
    sparse_time_dates = {}   # method_key -> set of dates
    sparse_pixel_vals = {}   # method_key -> {date: temp_filled_value}

    all_rows = []

    for _, site_row in sites.iterrows():
        lake_id = int(site_row['lake_id'])
        site_id = int(site_row['site_id'])
        site_lat = site_row['latitude']
        site_lon = site_row['longitude']

        # Load buoy data
        rep_hour = determine_representative_hour(lake_id, site_id, lake_id_cci, selection_dfs)
        buoy_result = load_buoy_data(lake_id, site_id, buoy_dir, rep_hour)
        if buoy_result is None:
            continue

        buoy_date_temp = buoy_result['buoy_date_temp']
        buoy_lat = buoy_result['source_lat'] or site_lat
        buoy_lon = buoy_result['source_lon'] or site_lon
        buoy_dates = sorted(buoy_date_temp.keys())

        if verbose:
            print(f"    Site {site_id}: {len(buoy_dates)} buoy dates, "
                  f"buoy at ({buoy_lat:.4f}, {buoy_lon:.4f})")

        grid_idx = None
        pixel_lat = None
        pixel_lon = None

        # =====================================================================
        # Pass 1: Extract from each method file
        # =====================================================================
        for method_key, nc_path in method_files.items():
            suffix, is_interpolated, is_eof_filtered_method = METHOD_VARIANTS[method_key]

            try:
                ds = xr.open_dataset(nc_path)
            except Exception as e:
                if verbose:
                    print(f"    {method_key}: error opening: {e}")
                continue

            try:
                time_vals = pd.to_datetime(ds['time'].values)
                date_to_idx = {t.date(): i for i, t in enumerate(time_vals)}

                # Find grid point (once, from first file)
                if grid_idx is None:
                    grid_idx, dist = find_nearest_grid_point(
                        ds['lat'].values, ds['lon'].values, site_lat, site_lon
                    )
                    if dist > DISTANCE_THRESHOLD:
                        if verbose:
                            print(f"    Distance {dist:.4f}° exceeds threshold, skipping site")
                        ds.close()
                        break
                    pixel_lat = float(ds['lat'].values[grid_idx[0]])
                    pixel_lon = float(ds['lon'].values[grid_idx[1]])

                lat_idx, lon_idx = grid_idx

                # Cache sparse file time dates and pixel values for later flag computation
                if not is_interpolated and method_key not in sparse_time_dates:
                    sparse_time_dates[method_key] = set(date_to_idx.keys())
                    if 'temp_filled' in ds:
                        px_ts = ds['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values
                        px_ts = ensure_celsius(px_ts)
                        sv = {}
                        for d, tidx in date_to_idx.items():
                            v = float(px_ts[tidx])
                            if np.isfinite(v):
                                sv[d] = v
                        sparse_pixel_vals[method_key] = sv

                # --- Extract OBSERVATION data_type (non-interpolated only) ---
                if not is_interpolated and 'lake_surface_water_temperature' in ds:
                    obs_pixel = ds['lake_surface_water_temperature'].isel(
                        lat=lat_idx, lon=lon_idx).values
                    obs_pixel = ensure_celsius(obs_pixel)

                    # Get quality_level
                    ql_pixel = None
                    if 'quality_level' in ds:
                        ql_pixel = ds['quality_level'].isel(
                            lat=lat_idx, lon=lon_idx).values

                    for d in buoy_dates:
                        if d not in date_to_idx:
                            continue
                        tidx = date_to_idx[d]
                        obs_val = float(obs_pixel[tidx])
                        if np.isnan(obs_val):
                            continue

                        # Quality level gate: MUST be >= 3
                        ql = float('nan')
                        if ql_pixel is not None:
                            raw_ql = ql_pixel[tidx]
                            if np.isfinite(raw_ql):
                                ql = int(raw_ql)
                        if not (np.isfinite(ql) and ql >= QUALITY_LEVEL_THRESHOLD):
                            continue

                        insitu_val = buoy_date_temp[d]
                        diff = obs_val - insitu_val
                        dt_obj = time_vals[tidx]

                        all_rows.append({
                            'lake_id_cci': lake_id_cci,
                            'lake_id': lake_id,
                            'site_id': site_id,
                            'method': method_key,
                            'data_type': 'observation',
                            'lat_idx': lat_idx,
                            'lon_idx': lon_idx,
                            'pixel_lat': pixel_lat,
                            'pixel_lon': pixel_lon,
                            'buoy_lat': buoy_lat,
                            'buoy_lon': buoy_lon,
                            'date': d,
                            'year': dt_obj.year,
                            'month': dt_obj.month,
                            'doy': dt_obj.dayofyear,
                            'season': assign_season(dt_obj.month),
                            'insitu_temp': insitu_val,
                            'satellite_temp': obs_val,
                            'diff': diff,
                            'quality_level': ql,
                            'was_observed': True,
                            'is_temporally_interpolated': False,
                            'is_eof_filtered_replaced': False,
                            'eof_filter_delta': float('nan'),
                        })

                # --- Extract RECONSTRUCTION data_type ---
                if 'temp_filled' in ds:
                    recon_pixel = ds['temp_filled'].isel(
                        lat=lat_idx, lon=lon_idx).values
                    recon_pixel = ensure_celsius(recon_pixel)

                    for d in buoy_dates:
                        if d not in date_to_idx:
                            continue
                        tidx = date_to_idx[d]
                        recon_val = float(recon_pixel[tidx])
                        if np.isnan(recon_val):
                            continue

                        insitu_val = buoy_date_temp[d]
                        diff = recon_val - insitu_val
                        dt_obj = time_vals[tidx]

                        # Determine was_observed from prepared.nc
                        was_obs = False
                        if prep_obs_mask is not None and prep_date_to_idx is not None:
                            if d in prep_date_to_idx:
                                # Find matching grid cell in prepared.nc
                                if prep_lat is not None and prep_lon is not None:
                                    plat_match = np.argmin(np.abs(prep_lat - pixel_lat))
                                    plon_match = np.argmin(np.abs(prep_lon - pixel_lon))
                                    t_prep = prep_date_to_idx[d]
                                    if (abs(prep_lat[plat_match] - pixel_lat) < 0.01 and
                                        abs(prep_lon[plon_match] - pixel_lon) < 0.01):
                                        was_obs = bool(prep_obs_mask[t_prep, plat_match, plon_match])

                        # Determine is_temporally_interpolated
                        is_temp_interp = False
                        if is_interpolated:
                            parent_key = INTERP_PARENTS.get(method_key)
                            if parent_key and parent_key in sparse_time_dates:
                                # Date NOT in parent sparse file = temporally interpolated
                                is_temp_interp = (d not in sparse_time_dates[parent_key])

                        # Determine is_eof_filtered_replaced and delta
                        is_eof_replaced = False
                        eof_delta = float('nan')
                        if is_eof_filtered_method:
                            # Compare against the non-filtered counterpart
                            if method_key == 'eof_filtered':
                                dineof_vals = sparse_pixel_vals.get('dineof', {})
                            elif method_key == 'eof_filtered_interp_full':
                                dineof_vals = sparse_pixel_vals.get('dineof', {})
                            else:
                                dineof_vals = {}

                            dineof_val = dineof_vals.get(d)
                            if dineof_val is not None and np.isfinite(dineof_val):
                                # For eof_filtered (sparse): compare directly
                                # For eof_filtered_interp_full: compare the sparse eof_filtered
                                # value against sparse dineof value at this date
                                if method_key == 'eof_filtered':
                                    eof_delta = recon_val - dineof_val
                                    is_eof_replaced = abs(eof_delta) > 1e-6
                                elif method_key == 'eof_filtered_interp_full':
                                    # Use sparse eof_filtered value for comparison
                                    eof_sparse_vals = sparse_pixel_vals.get('eof_filtered', {})
                                    eof_sparse_val = eof_sparse_vals.get(d)
                                    if eof_sparse_val is not None:
                                        eof_delta = eof_sparse_val - dineof_val
                                        is_eof_replaced = abs(eof_delta) > 1e-6

                        # Get quality_level for reconstruction (for reference, not gating)
                        ql_recon = float('nan')
                        if 'quality_level' in ds:
                            raw_ql = ds['quality_level'].isel(
                                lat=lat_idx, lon=lon_idx).values[tidx]
                            if np.isfinite(raw_ql):
                                ql_recon = int(raw_ql)

                        all_rows.append({
                            'lake_id_cci': lake_id_cci,
                            'lake_id': lake_id,
                            'site_id': site_id,
                            'method': method_key,
                            'data_type': 'reconstruction',
                            'lat_idx': lat_idx,
                            'lon_idx': lon_idx,
                            'pixel_lat': pixel_lat,
                            'pixel_lon': pixel_lon,
                            'buoy_lat': buoy_lat,
                            'buoy_lon': buoy_lon,
                            'date': d,
                            'year': dt_obj.year,
                            'month': dt_obj.month,
                            'doy': dt_obj.dayofyear,
                            'season': assign_season(dt_obj.month),
                            'insitu_temp': insitu_val,
                            'satellite_temp': recon_val,
                            'diff': diff,
                            'quality_level': ql_recon,
                            'was_observed': was_obs,
                            'is_temporally_interpolated': is_temp_interp,
                            'is_eof_filtered_replaced': is_eof_replaced,
                            'eof_filter_delta': eof_delta,
                        })

            except Exception as e:
                if verbose:
                    print(f"    {method_key}: error: {e}")
                    import traceback
                    traceback.print_exc()
            finally:
                ds.close()

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)

    # Add hemisphere
    if 'pixel_lat' in df.columns:
        df['hemisphere'] = df['pixel_lat'].apply(assign_hemisphere)

    if verbose:
        n_obs = len(df[df['data_type'] == 'observation'])
        n_rec = len(df[df['data_type'] == 'reconstruction'])
        methods = sorted(df['method'].unique())
        print(f"  [Lake {lake_id_cci}] Total: {len(df)} rows "
              f"({n_obs} obs, {n_rec} recon) across {len(methods)} methods")

    return df


# =============================================================================
# Statistics aggregation
# =============================================================================

def compute_stats_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Compute metrics grouped by given columns."""
    results = []
    for group_key, group_df in df.groupby(group_cols, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metrics = compute_metrics_from_df(group_df)
        row = dict(zip(group_cols, group_key))
        row.update(metrics)
        results.append(row)
    return pd.DataFrame(results)


def generate_all_stats(df: pd.DataFrame, output_dir: str):
    """Generate all stats tables from the assembly DataFrame."""

    os.makedirs(output_dir, exist_ok=True)

    # ----- Per-lake stats (by method × data_type × lake) -----
    print("  Per-lake stats...")
    per_lake = compute_stats_table(df, ['method', 'data_type', 'lake_id_cci'])
    per_lake = per_lake.sort_values(['method', 'data_type', 'lake_id_cci'])
    save_csv(per_lake, output_dir, "insitu_cv_per_lake.csv")

    # ----- Global (by method × data_type) -----
    print("  Global stats (pooled)...")
    global_stats = compute_stats_table(df, ['method', 'data_type'])
    save_csv(global_stats, output_dir, "insitu_cv_global.csv")

    # ----- Reconstruction sub-types: observed vs missing -----
    recon = df[df['data_type'] == 'reconstruction']
    if len(recon) > 0:
        print("  Reconstruction: observed vs missing...")
        recon_split = compute_stats_table(recon, ['method', 'was_observed'])
        save_csv(recon_split, output_dir, "insitu_cv_recon_observed_vs_missing.csv")

        # Per-lake version
        recon_split_lake = compute_stats_table(recon, ['method', 'was_observed', 'lake_id_cci'])
        recon_split_lake = recon_split_lake.sort_values(['method', 'was_observed', 'lake_id_cci'])
        save_csv(recon_split_lake, output_dir, "insitu_cv_recon_observed_vs_missing_per_lake.csv")

    # ----- By year -----
    print("  Per-year stats...")
    per_year = compute_stats_table(df, ['method', 'data_type', 'year'])
    per_year = per_year.sort_values(['method', 'data_type', 'year'])
    save_csv(per_year, output_dir, "insitu_cv_per_year.csv")

    # ----- By month -----
    print("  Per-month stats...")
    per_month = compute_stats_table(df, ['method', 'data_type', 'month'])
    per_month = per_month.sort_values(['method', 'data_type', 'month'])
    save_csv(per_month, output_dir, "insitu_cv_per_month.csv")

    # ----- By season -----
    print("  Per-season stats...")
    per_season = compute_stats_table(df, ['method', 'data_type', 'season'])
    save_csv(per_season, output_dir, "insitu_cv_per_season.csv")

    # ----- By hemisphere -----
    if 'hemisphere' in df.columns:
        print("  Per-hemisphere stats...")
        per_hemi = compute_stats_table(df, ['method', 'data_type', 'hemisphere'])
        save_csv(per_hemi, output_dir, "insitu_cv_per_hemisphere.csv")

    # ----- By quality_level (observation only) -----
    obs_df = df[(df['data_type'] == 'observation') & df['quality_level'].notna() & (df['quality_level'] >= 3)]
    if len(obs_df) > 0:
        print("  Per-quality-level stats (observation)...")
        per_ql = compute_stats_table(obs_df, ['method', 'quality_level'])
        per_ql = per_ql.sort_values(['method', 'quality_level'])
        save_csv(per_ql, output_dir, "insitu_cv_obs_per_quality_level.csv")

    # ----- Temporally interpolated vs not (reconstruction from daily methods) -----
    daily_methods = [k for k, (_, is_int, _) in METHOD_VARIANTS.items() if is_int]
    daily_recon = recon[recon['method'].isin(daily_methods)]
    if len(daily_recon) > 0:
        print("  Temporal interpolation impact...")
        interp_stats = compute_stats_table(daily_recon, ['method', 'is_temporally_interpolated'])
        save_csv(interp_stats, output_dir, "insitu_cv_temporal_interp_impact.csv")

    # ----- EOF filter impact (eof_filtered methods) -----
    eof_methods = [k for k, (_, _, is_eof) in METHOD_VARIANTS.items() if is_eof]
    eof_recon = recon[recon['method'].isin(eof_methods)]
    if len(eof_recon) > 0:
        print("  EOF filter impact...")
        eof_stats = compute_stats_table(eof_recon, ['method', 'is_eof_filtered_replaced'])
        save_csv(eof_stats, output_dir, "insitu_cv_eof_filter_impact.csv")


def save_csv(df: pd.DataFrame, output_dir: str, filename: str):
    """Save DataFrame to CSV."""
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"    Saved: {filename}")


# =============================================================================
# Lake discovery
# =============================================================================

def find_lakes_with_post(run_root: str, alpha: str = 'a1000') -> List[int]:
    """Find all lake IDs that have post-processed results."""
    post_dir = os.path.join(run_root, "post")
    if not os.path.exists(post_dir):
        return []
    lake_ids = []
    for folder in os.listdir(post_dir):
        try:
            lake_id = int(folder.lstrip('0') or '0')
            if lake_id > 0:
                alpha_dir = os.path.join(post_dir, folder, alpha)
                if os.path.exists(alpha_dir):
                    lake_ids.append(lake_id)
        except ValueError:
            continue
    return sorted(lake_ids)


def find_lakes_with_insitu(lake_ids: List[int],
                           selection_dfs: List[pd.DataFrame]) -> List[int]:
    """Filter lake IDs to only those with buoy data in selection CSVs."""
    all_lake_ids_in_csvs = set()
    for df in selection_dfs:
        if 'lake_id_cci' in df.columns:
            all_lake_ids_in_csvs.update(df['lake_id_cci'].unique())
    return sorted([lid for lid in lake_ids if lid in all_lake_ids_in_csvs])


# =============================================================================
# Main
# =============================================================================

def _print_summary(assembly: pd.DataFrame):
    """Print breakdown summary of assembly DataFrame."""
    n_lakes = assembly['lake_id_cci'].nunique()
    print(f"Total: {len(assembly):,} rows from {n_lakes} lakes")
    print("\nBreakdown by method × data_type:")
    summary = assembly.groupby(['method', 'data_type']).size().reset_index(name='count')
    for _, row in summary.iterrows():
        print(f"  {row['method']:30s} {row['data_type']:25s} {row['count']:>8,}")


def _print_output_summary(output_dir: str):
    """Print summary of all output files."""
    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp) / 1e6
            print(f"  {f} ({size:.1f} MB)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Assemble in-situ (buoy) CV pairs and compute statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods extracted (6 variants):
    dineof, eof_filtered, interp_full, eof_filtered_interp_full,
    dincae, dincae_interp_full

Data types:
    observation       — satellite LSWT (QL>=3) vs buoy [sparse methods only]
    reconstruction    — temp_filled vs buoy [all methods]
      sub-split by was_observed: reconstruction_observed / reconstruction_missing

Output files (in --output-dir):
    insitu_cv_assembly.parquet / .csv       — Full assembly
    insitu_cv_per_lake.csv                  — Stats by method × data_type × lake
    insitu_cv_global.csv                    — Stats by method × data_type (pooled)
    insitu_cv_recon_observed_vs_missing.csv — Recon stats split by was_observed
    insitu_cv_per_year.csv                  — Stats by method × data_type × year
    insitu_cv_per_month.csv                 — Stats by method × data_type × month
    insitu_cv_per_season.csv                — Stats by method × data_type × season
    insitu_cv_per_hemisphere.csv            — Stats by method × data_type × hemisphere
    insitu_cv_obs_per_quality_level.csv     — Obs stats by method × QL (3,4,5)
    insitu_cv_temporal_interp_impact.csv    — Daily methods: interpolated vs not
    insitu_cv_eof_filter_impact.csv         — EOF-filtered: replaced vs unchanged

Examples:
    # All lakes with buoy data
    python assemble_insitu_cv.py --run-root /path/to/experiment

    # Specific lakes
    python assemble_insitu_cv.py --run-root /path/to/experiment \\
        --lake-ids 88 375 4503
        """
    )
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--phase", choices=["sequential", "extract", "merge", "list-lakes"],
                        default="sequential",
                        help="Execution phase (default: sequential)")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: {run_root}/insitu_cv_assembly/)")
    parser.add_argument("--buoy-dir", default=DEFAULT_BUOY_DIR,
                        help="Buoy data directory")
    parser.add_argument("--selection-csvs", nargs="+", default=None,
                        help="Selection CSV paths (default: standard 4 CSVs)")

    lake_group = parser.add_mutually_exclusive_group()
    lake_group.add_argument("--lake-id", type=int, help="Single lake ID")
    lake_group.add_argument("--lake-ids", type=int, nargs="+", help="Multiple lake IDs")
    lake_group.add_argument("--all", action="store_true", help="Process all lakes with buoy data")

    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Disable fair comparison filtering")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing assembly CSV instead of overwriting. "
                             "Stats are recomputed from the full file.")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip CSV assembly output (Parquet only)")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Skip Parquet assembly output (CSV only)")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "insitu_cv_assembly")
    os.makedirs(args.output_dir, exist_ok=True)

    # Selection CSVs (needed for all phases except merge)
    selection_dfs = []
    if args.phase != 'merge':
        csv_paths = args.selection_csvs or DEFAULT_SELECTION_CSVS
        print("Loading selection CSVs...")
        selection_dfs = load_selection_csvs(csv_paths)
        if not selection_dfs:
            print("Error: No selection CSVs could be loaded")
            sys.exit(1)
        print(f"  Loaded {len(selection_dfs)} selection CSV(s)")

    # Resolve lake IDs (not needed for merge)
    lake_ids_with_insitu = []
    if args.phase != 'merge':
        if args.lake_id:
            lake_ids = [args.lake_id]
        elif args.lake_ids:
            lake_ids = args.lake_ids
        elif args.all:
            if HAS_COMPLETION_CHECK and not args.no_fair_comparison:
                lake_ids, _ = get_fair_comparison_lakes(args.run_root, args.alpha, verbose=True)
                if not lake_ids:
                    print("No lakes with both methods complete.")
                    sys.exit(1)
            else:
                lake_ids = find_lakes_with_post(args.run_root, args.alpha)
        else:
            lake_ids = find_lakes_with_post(args.run_root, args.alpha)

        lake_ids_with_insitu = find_lakes_with_insitu(lake_ids, selection_dfs)
        if not lake_ids_with_insitu:
            print("No lakes with buoy data found")
            sys.exit(1)

    # =====================================================================
    # PHASE: list-lakes
    # =====================================================================
    if args.phase == 'list-lakes':
        for lid in lake_ids_with_insitu:
            print(lid)
        sys.exit(0)

    # =====================================================================
    # PHASE: extract — write per-lake CSVs
    # =====================================================================
    if args.phase == 'extract':
        per_lake_dir = os.path.join(args.output_dir, "per_lake")
        os.makedirs(per_lake_dir, exist_ok=True)

        for lake_id in lake_ids_with_insitu:
            df = extract_lake(
                args.run_root, lake_id, args.alpha,
                selection_dfs, args.buoy_dir, verbose
            )
            if df is not None and len(df) > 0:
                out_path = os.path.join(per_lake_dir, f"insitu_cv_lake_{lake_id}.csv")
                df.to_csv(out_path, index=False)
                print(f"  Saved: {out_path} ({len(df):,} rows)")
                del df
            else:
                if verbose:
                    print(f"  [Lake {lake_id}] No in-situ data")

        sys.exit(0)

    # =====================================================================
    # PHASE: merge — concatenate per-lake CSVs + compute stats
    # =====================================================================
    if args.phase == 'merge':
        per_lake_dir = os.path.join(args.output_dir, "per_lake")
        print(f"\n{'='*70}")
        print("MERGE")
        print(f"{'='*70}")
        t_start = time.time()

        csv_files = sorted(Path(per_lake_dir).glob("insitu_cv_lake_*.csv"))
        if not csv_files:
            print(f"  No per-lake CSVs found in {per_lake_dir}")
            sys.exit(1)

        print(f"  Found {len(csv_files)} per-lake CSVs")

        assembly_csv = os.path.join(args.output_dir, "insitu_cv_assembly.csv")
        header_written = False
        if args.append and os.path.exists(assembly_csv):
            header_written = True

        n_rows = 0
        for csv_file in csv_files:
            chunk = pd.read_csv(csv_file)
            chunk.to_csv(assembly_csv, index=False,
                         mode='a', header=(not header_written))
            header_written = True
            n_rows += len(chunk)
            del chunk

        size_mb = os.path.getsize(assembly_csv) / 1e6
        print(f"  Assembly: {n_rows:,} rows ({size_mb:.1f} MB)")

        print(f"  Reading assembly for statistics...")
        assembly = pd.read_csv(assembly_csv, low_memory=False)
        assembly['date'] = pd.to_datetime(assembly['date'])

        _print_summary(assembly)

        if not args.no_parquet:
            try:
                pq_path = os.path.join(args.output_dir, "insitu_cv_assembly.parquet")
                assembly.to_parquet(pq_path, index=False, engine='pyarrow')
                print(f"  Parquet: {pq_path}")
            except Exception as e:
                print(f"  Parquet failed ({e})")

        if args.no_csv:
            os.remove(assembly_csv)

        print(f"  Computing statistics...")
        generate_all_stats(assembly, args.output_dir)

        del assembly
        _print_output_summary(args.output_dir)
        print(f"  Done ({time.time() - t_start:.1f}s)")
        sys.exit(0)

    # =====================================================================
    # PHASE: sequential (default) — extract + merge in one process
    # =====================================================================
    print("=" * 70)
    print("IN-SITU CV ASSEMBLY (sequential)")
    print("=" * 70)
    print(f"Run root:      {args.run_root}")
    print(f"Lakes (insitu): {len(lake_ids_with_insitu)}")
    print(f"Alpha:          {args.alpha}")
    print(f"Buoy dir:       {args.buoy_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Quality gate:   QL >= {QUALITY_LEVEL_THRESHOLD} (observation data_type)")
    print("=" * 70)

    t_start = time.time()
    assembly_csv = os.path.join(args.output_dir, "insitu_cv_assembly.csv")
    n_lakes_done = 0
    n_rows_total = 0

    if args.append and os.path.exists(assembly_csv):
        header_written = True
        print(f"  Appending to existing assembly")
    else:
        header_written = False
        if os.path.exists(assembly_csv):
            os.remove(assembly_csv)

    for lake_id in lake_ids_with_insitu:
        df = extract_lake(
            args.run_root, lake_id, args.alpha,
            selection_dfs, args.buoy_dir, verbose
        )
        if df is not None and len(df) > 0:
            df.to_csv(assembly_csv, index=False,
                      mode='a', header=(not header_written))
            header_written = True
            n_rows_total += len(df)
            n_lakes_done += 1
            del df

    if n_rows_total == 0 and not (args.append and os.path.exists(assembly_csv)):
        print("\nNo in-situ data extracted")
        sys.exit(1)

    t_extract = time.time() - t_start
    size_mb = os.path.getsize(assembly_csv) / 1e6
    print(f"\nExtraction: {n_rows_total:,} rows from {n_lakes_done} lakes ({t_extract:.1f}s)")

    print(f"\nReading full assembly for statistics...")
    assembly = pd.read_csv(assembly_csv, low_memory=False)
    assembly['date'] = pd.to_datetime(assembly['date'])

    _print_summary(assembly)

    if not args.no_parquet:
        try:
            pq_path = os.path.join(args.output_dir, "insitu_cv_assembly.parquet")
            assembly.to_parquet(pq_path, index=False, engine='pyarrow')
            size_mb_pq = os.path.getsize(pq_path) / 1e6
            print(f"  Parquet: {pq_path} ({size_mb_pq:.1f} MB)")
        except Exception as e:
            print(f"  Parquet failed ({e}), skipping")

    if args.no_csv:
        os.remove(assembly_csv)

    # Compute stats (always from full assembly)
    print(f"\nComputing statistics...")
    generate_all_stats(assembly, args.output_dir)

    t_total = time.time() - t_start
    _print_output_summary(args.output_dir)
    print(f"Total time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
