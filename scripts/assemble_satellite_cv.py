#!/usr/bin/env python3
"""
Satellite CV Assembly & Statistics
===================================

Assembles all cross-validation (withheld observation) pixel pairs from
all lakes into a single file, then computes per-lake and global statistics.

Two modes:
  --mode physical  (default)
      Uses post files (*_dineof.nc, *_dineof_eof_filtered.nc, *_dincae.nc)
      which have data_source flag (0=gap, 1=observed, 2=CV).
      Extracts temp_filled (recon °C) and lake_surface_water_temperature (obs).
      All 3 methods: dineof, eof_filtered, dincae.

  --mode anomaly
      Uses prepared.nc + clouds_index.nc + dineof_results.nc + dincae_results.nc.
      All in detrended anomaly space.
      Only 2 methods: dineof, dincae (eof_filtered doesn't exist in anomaly space).

Outputs:
  1. Assembly file (one row per CV pixel): Parquet + CSV
  2. Per-lake stats CSV
  3. Global stats CSV (pooled by method, and optionally by year/month/season)

Methods compared:
  - dineof (baseline DINEOF reconstruction)
  - eof_filtered (DINEOF with temporal EOF filtering) [physical mode only]
  - dincae (DINCAE U-Net reconstruction)

NOTE: Temporally interpolated products (interp_full) do NOT have CV points
      (no withheld data in interpolated frames), so they are excluded.

Author: Shaerdan / NCEO / University of Reading
Date: February 2026
"""

import argparse
import os
import sys
import time
from datetime import datetime
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

# data_source flag values (from add_data_source_flag.py)
FLAG_TRUE_GAP = 0
FLAG_OBSERVED_SEEN = 1
FLAG_CV_WITHHELD = 2

# Methods to extract in physical mode
PHYSICAL_METHODS = {
    'dineof':       '_dineof.nc',
    'eof_filtered': '_dineof_eof_filtered.nc',
    'dincae':       '_dincae.nc',
}

# Metrics to compute
METRIC_NAMES = [
    'rmse', 'mae', 'bias', 'median', 'std', 'rstd', 'iqr',
    'n', 'correlation', 'r_squared', 'fraction_within_1K',
]


# =============================================================================
# Metric computation
# =============================================================================

def compute_metrics(diff: np.ndarray,
                    original: np.ndarray = None,
                    reconstructed: np.ndarray = None) -> Dict[str, float]:
    """
    Compute all metrics from a signed diff vector.

    diff = reconstructed - original (signed).

    Args:
        diff: signed error vector
        original: original values (for correlation/R²)
        reconstructed: reconstructed values (for correlation/R²)

    Returns:
        dict with all metric values
    """
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

    # Correlation and R² require both original and reconstructed
    if original is not None and reconstructed is not None and n > 1:
        corr = np.corrcoef(original, reconstructed)[0, 1]
        metrics['correlation'] = float(corr) if np.isfinite(corr) else float('nan')
        metrics['r_squared'] = float(corr ** 2) if np.isfinite(corr) else float('nan')
    else:
        metrics['correlation'] = float('nan')
        metrics['r_squared'] = float('nan')

    return metrics


def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Compute metrics from a DataFrame slice (must have diff, original_value, reconstructed_value)."""
    diff = df['diff'].values
    orig = df['original_value'].values if 'original_value' in df.columns else None
    recon = df['reconstructed_value'].values if 'reconstructed_value' in df.columns else None
    return compute_metrics(diff, orig, recon)


# =============================================================================
# Season / temporal helpers
# =============================================================================

def assign_season(month: int) -> str:
    """DJF/MAM/JJA/SON."""
    if month in (12, 1, 2):
        return 'DJF'
    elif month in (3, 4, 5):
        return 'MAM'
    elif month in (6, 7, 8):
        return 'JJA'
    else:
        return 'SON'


def assign_hemisphere(lat: float) -> str:
    return 'N' if lat >= 0 else 'S'


# =============================================================================
# Physical-mode extraction
# =============================================================================

def extract_physical_lake(post_dir: str, lake_id: int,
                          verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Extract CV pixel pairs from post files (physical space, °C).

    For each of the 3 methods, finds pixels where data_source == 2 (CV),
    extracts temp_filled (reconstruction) and lake_surface_water_temperature
    (original observation).

    Also computes eof_filter_replaced flag and delta by cross-referencing
    dineof and eof_filtered values at CV points.
    """
    lake_id9 = f"{lake_id:09d}"
    nc_files = {}

    # Discover post files
    for method_key, suffix in PHYSICAL_METHODS.items():
        candidates = list(Path(post_dir).glob(f"*{suffix}"))
        if candidates:
            nc_files[method_key] = str(candidates[0])

    if not nc_files:
        if verbose:
            print(f"  [Lake {lake_id}] No post files found in {post_dir}")
        return None

    all_rows = []

    # Pre-load dineof values for eof_filter delta comparison
    dineof_cv_lookup = {}  # (lat_idx, lon_idx, time_idx) -> temp_filled value

    for method_key, nc_path in nc_files.items():
        if verbose:
            print(f"  [Lake {lake_id}] {method_key}: {os.path.basename(nc_path)}")

        try:
            # Selectively load only the 4 variables we need, not the entire cube.
            # A large lake file can have 10+ variables at ~1.2GB each.
            keep = {'data_source', 'temp_filled', 'lake_surface_water_temperature',
                    'lswt', 'quality_level'}
            ds = xr.open_dataset(nc_path)
            drop_vars = [v for v in ds.data_vars if v not in keep]
            if drop_vars:
                ds = ds.drop_vars(drop_vars)
            ds.load()  # load remaining variables into memory
        except Exception as e:
            print(f"  [Lake {lake_id}] Error opening {nc_path}: {e}")
            continue

        try:
            if 'data_source' not in ds:
                if verbose:
                    print(f"    data_source flag not found, skipping")
                continue

            if 'temp_filled' not in ds:
                if verbose:
                    print(f"    temp_filled not found, skipping")
                continue

            data_source = ds['data_source'].values  # (time, lat, lon)
            temp_filled = ds['temp_filled'].values
            time_vals = pd.to_datetime(ds['time'].values)
            lat_vals = ds['lat'].values
            lon_vals = ds['lon'].values

            # Get original observation variable
            obs_var = None
            for vname in ['lake_surface_water_temperature', 'lswt']:
                if vname in ds:
                    obs_var = ds[vname].values
                    break

            # Get quality_level variable (from CCI product, copied by CopyOriginalVarsStep)
            ql_var = None
            if 'quality_level' in ds:
                ql_var = ds['quality_level'].values

            # Find all CV points (data_source == 2)
            cv_mask = (data_source == FLAG_CV_WITHHELD)
            cv_indices = np.argwhere(cv_mask)  # (N, 3) → [time_idx, lat_idx, lon_idx]

            if len(cv_indices) == 0:
                if verbose:
                    print(f"    No CV points found (data_source==2)")
                continue

            if verbose:
                print(f"    Found {len(cv_indices):,} CV points")

            for idx in cv_indices:
                t_idx, lat_idx, lon_idx = int(idx[0]), int(idx[1]), int(idx[2])

                recon_val = float(temp_filled[t_idx, lat_idx, lon_idx])
                if np.isnan(recon_val):
                    continue

                # Original observation
                orig_val = float('nan')
                if obs_var is not None:
                    raw = obs_var[t_idx, lat_idx, lon_idx]
                    if np.isfinite(raw):
                        # Convert Kelvin to Celsius if needed
                        orig_val = float(raw)
                        if orig_val > 100:
                            orig_val -= 273.15

                if np.isnan(orig_val):
                    continue

                diff = recon_val - orig_val
                dt = time_vals[t_idx]

                # Extract quality level (should be 3, 4, or 5 for CV points)
                ql = float('nan')
                if ql_var is not None:
                    raw_ql = ql_var[t_idx, lat_idx, lon_idx]
                    if np.isfinite(raw_ql):
                        ql = int(raw_ql)

                row = {
                    'lake_id': lake_id,
                    'method': method_key,
                    'lat_idx': lat_idx,
                    'lon_idx': lon_idx,
                    'time_idx': t_idx,
                    'date': dt.date(),
                    'year': dt.year,
                    'month': dt.month,
                    'doy': dt.dayofyear,
                    'season': assign_season(dt.month),
                    'lat': float(lat_vals[lat_idx]),
                    'lon': float(lon_vals[lon_idx]),
                    'original_value': orig_val,
                    'reconstructed_value': recon_val,
                    'diff': diff,
                    'quality_level': ql,
                }

                all_rows.append(row)

                # Cache dineof values for eof_filter delta
                if method_key == 'dineof':
                    dineof_cv_lookup[(lat_idx, lon_idx, t_idx)] = recon_val

        except Exception as e:
            print(f"  [Lake {lake_id}] Error processing {method_key}: {e}")
            import traceback
            traceback.print_exc()

    if not all_rows:
        if verbose:
            print(f"  [Lake {lake_id}] No valid CV pairs extracted")
        return None

    df = pd.DataFrame(all_rows)

    # Add hemisphere
    if 'lat' in df.columns:
        df['hemisphere'] = df['lat'].apply(assign_hemisphere)

    # Add eof_filter flags for eof_filtered rows
    if 'eof_filtered' in df['method'].values and dineof_cv_lookup:
        eof_mask = df['method'] == 'eof_filtered'
        is_replaced = np.zeros(len(df), dtype=bool)
        delta = np.full(len(df), np.nan)

        for i, row in df[eof_mask].iterrows():
            key = (row['lat_idx'], row['lon_idx'], row['time_idx'])
            dineof_val = dineof_cv_lookup.get(key, np.nan)
            if np.isfinite(dineof_val):
                d = row['reconstructed_value'] - dineof_val
                delta[i] = d
                is_replaced[i] = abs(d) > 1e-6

        df['is_eof_filtered_replaced'] = is_replaced
        df['eof_filter_delta'] = delta
    else:
        df['is_eof_filtered_replaced'] = False
        df['eof_filter_delta'] = np.nan

    if verbose:
        for method in df['method'].unique():
            n = len(df[df['method'] == method])
            print(f"  [Lake {lake_id}] {method}: {n:,} CV pairs")

    return df


# =============================================================================
# Anomaly-mode extraction
# =============================================================================

def extract_anomaly_lake(run_root: str, lake_id: int, alpha: str = 'a1000',
                         verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Extract CV pixel pairs from prepared.nc + results files (anomaly space).

    Uses clouds_index.nc to enumerate CV point coordinates.
    Reads original values from prepared.nc and reconstructions from
    dineof_results.nc / dincae_results.nc.

    Only DINEOF + DINCAE (eof_filtered doesn't exist in anomaly space).
    """
    lake_id9 = f"{lake_id:09d}"

    prepared_nc = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
    clouds_index_nc = os.path.join(run_root, "prepared", lake_id9, "clouds_index.nc")

    if not os.path.exists(prepared_nc) or not os.path.exists(clouds_index_nc):
        if verbose:
            print(f"  [Lake {lake_id}] prepared.nc or clouds_index.nc not found")
        return None

    # Load CV point indices
    ds_cv = xr.open_dataset(clouds_index_nc)
    clouds_index = ds_cv["clouds_index"].values
    iindex = ds_cv["iindex"].values  # lon coordinates (1-based)
    jindex = ds_cv["jindex"].values  # lat coordinates (1-based)
    ds_cv.close()

    # Determine clouds_index layout
    if clouds_index.ndim == 2:
        if clouds_index.shape[1] == 2:
            n_points = clouds_index.shape[0]
            get_m = lambda p: int(clouds_index[p, 0])
            get_t = lambda p: int(clouds_index[p, 1])
        elif clouds_index.shape[0] == 2:
            n_points = clouds_index.shape[1]
            get_m = lambda p: int(clouds_index[0, p])
            get_t = lambda p: int(clouds_index[1, p])
        else:
            print(f"  [Lake {lake_id}] Unexpected clouds_index shape: {clouds_index.shape}")
            return None
    else:
        print(f"  [Lake {lake_id}] Unexpected clouds_index ndim: {clouds_index.ndim}")
        return None

    if verbose:
        print(f"  [Lake {lake_id}] {n_points:,} CV points in clouds_index.nc")

    # Load prepared.nc (original observations in anomaly space)
    ds_prep = xr.load_dataset(prepared_nc)
    original_data = ds_prep['lake_surface_water_temperature'].values  # (time, lat, lon)
    time_vals = pd.to_datetime(ds_prep['time'].values)
    lat_vals = ds_prep['lat'].values
    lon_vals = ds_prep['lon'].values
    ds_prep.close()

    # Load reconstruction files
    recon_files = {}
    dineof_results = os.path.join(run_root, "dineof", lake_id9, alpha, "dineof_results.nc")
    dincae_results = os.path.join(run_root, "dincae", lake_id9, alpha, "dincae_results.nc")

    if os.path.exists(dineof_results):
        ds_din = xr.load_dataset(dineof_results)
        var_name = 'temp_filled' if 'temp_filled' in ds_din else list(ds_din.data_vars)[0]
        recon_files['dineof'] = ds_din[var_name].values
        ds_din.close()
    if os.path.exists(dincae_results):
        ds_din = xr.load_dataset(dincae_results)
        var_name = 'temp_filled' if 'temp_filled' in ds_din else list(ds_din.data_vars)[0]
        recon_files['dincae'] = ds_din[var_name].values
        ds_din.close()

    if not recon_files:
        if verbose:
            print(f"  [Lake {lake_id}] No reconstruction files found")
        return None

    # Try to load quality_level from post file (for QL aggregation)
    # Post files have quality_level copied from original CCI product
    ql_data = None
    ql_time_to_idx = None
    post_dir = os.path.join(run_root, "post", lake_id9, alpha)
    if os.path.exists(post_dir):
        for suffix in ['_dineof.nc', '_dincae.nc']:
            candidates = list(Path(post_dir).glob(f"*{suffix}"))
            if candidates:
                try:
                    ds_post = xr.open_dataset(str(candidates[0]))
                    if 'quality_level' in ds_post:
                        ql_data = ds_post['quality_level'].values
                        ql_times = pd.to_datetime(ds_post['time'].values)
                        ql_time_to_idx = {t.date(): i for i, t in enumerate(ql_times)}
                        # quality_level grid may differ from prepared.nc grid;
                        # store lat/lon for coordinate matching
                        ql_lat = ds_post['lat'].values
                        ql_lon = ds_post['lon'].values
                    ds_post.close()
                    if ql_data is not None:
                        break
                except Exception:
                    pass

    all_rows = []

    for p in range(n_points):
        m = get_m(p)
        t_julia = get_t(p)

        # Convert to 0-based Python indices
        # iindex → lon, jindex → lat (matching dincae_adapter_in.py / add_data_source_flag.py)
        t_idx = t_julia - 1
        lon_idx = int(iindex[m - 1]) - 1
        lat_idx = int(jindex[m - 1]) - 1

        # Bounds check
        if (t_idx < 0 or t_idx >= original_data.shape[0] or
            lat_idx < 0 or lat_idx >= original_data.shape[1] or
            lon_idx < 0 or lon_idx >= original_data.shape[2]):
            continue

        orig_val = float(original_data[t_idx, lat_idx, lon_idx])
        if np.isnan(orig_val):
            continue

        dt = time_vals[t_idx]
        base_row = {
            'lake_id': lake_id,
            'lat_idx': lat_idx,
            'lon_idx': lon_idx,
            'time_idx': t_idx,
            'date': dt.date(),
            'year': dt.year,
            'month': dt.month,
            'doy': dt.dayofyear,
            'season': assign_season(dt.month),
            'lat': float(lat_vals[lat_idx]),
            'lon': float(lon_vals[lon_idx]),
        }

        for method_key, recon_data in recon_files.items():
            if t_idx >= recon_data.shape[0]:
                continue
            recon_val = float(recon_data[t_idx, lat_idx, lon_idx])
            if np.isnan(recon_val):
                continue

            # Try to get quality_level from post file
            ql = float('nan')
            if ql_data is not None and ql_time_to_idx is not None:
                d = dt.date()
                if d in ql_time_to_idx:
                    t_post = ql_time_to_idx[d]
                    # Find matching grid cell in post file by coordinate value
                    plat = float(lat_vals[lat_idx])
                    plon = float(lon_vals[lon_idx])
                    lat_match = np.argmin(np.abs(ql_lat - plat))
                    lon_match = np.argmin(np.abs(ql_lon - plon))
                    if (abs(ql_lat[lat_match] - plat) < 0.01 and
                        abs(ql_lon[lon_match] - plon) < 0.01):
                        raw_ql = ql_data[t_post, lat_match, lon_match]
                        if np.isfinite(raw_ql):
                            ql = int(raw_ql)

            row = {
                **base_row,
                'method': method_key,
                'original_value': orig_val,
                'reconstructed_value': recon_val,
                'diff': recon_val - orig_val,
                'quality_level': ql,
            }
            all_rows.append(row)

    if not all_rows:
        if verbose:
            print(f"  [Lake {lake_id}] No valid CV pairs extracted")
        return None

    df = pd.DataFrame(all_rows)

    # Add hemisphere
    if 'lat' in df.columns:
        df['hemisphere'] = df['lat'].apply(assign_hemisphere)

    # No eof_filter flags in anomaly mode
    df['is_eof_filtered_replaced'] = False
    df['eof_filter_delta'] = np.nan

    if verbose:
        for method in df['method'].unique():
            n = len(df[df['method'] == method])
            print(f"  [Lake {lake_id}] {method}: {n:,} CV pairs")

    return df


# =============================================================================
# Statistics aggregation
# =============================================================================

def compute_stats_table(df: pd.DataFrame, group_cols: List[str],
                        label: str = "") -> pd.DataFrame:
    """
    Compute metrics grouped by given columns.

    Args:
        df: assembly DataFrame (must have 'diff', 'original_value', 'reconstructed_value')
        group_cols: columns to group by (e.g. ['method'], ['method', 'lake_id'])
        label: label for progress reporting

    Returns:
        DataFrame with one row per group, columns = group_cols + metric columns
    """
    results = []
    grouped = df.groupby(group_cols, observed=True)

    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        metrics = compute_metrics_from_df(group_df)

        row = dict(zip(group_cols, group_key))
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results)


def generate_all_stats(df: pd.DataFrame, output_dir: str,
                       mode_suffix: str = "physical"):
    """Generate per-lake, global, and sliced stats tables."""

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{mode_suffix}"

    # --- Per-lake stats (by method × lake) ---
    print("  Computing per-lake stats...")
    per_lake = compute_stats_table(df, ['method', 'lake_id'])
    per_lake = per_lake.sort_values(['method', 'lake_id'])
    per_lake_path = os.path.join(output_dir, f"satellite_cv_per_lake{suffix}.csv")
    per_lake.to_csv(per_lake_path, index=False)
    print(f"  Saved: {per_lake_path}")

    # --- Global stats (by method only — pooled across all lakes) ---
    print("  Computing global stats (pooled)...")
    global_stats = compute_stats_table(df, ['method'])
    global_path = os.path.join(output_dir, f"satellite_cv_global{suffix}.csv")
    global_stats.to_csv(global_path, index=False)
    print(f"  Saved: {global_path}")

    # --- By method × year ---
    print("  Computing per-year stats...")
    per_year = compute_stats_table(df, ['method', 'year'])
    per_year = per_year.sort_values(['method', 'year'])
    per_year_path = os.path.join(output_dir, f"satellite_cv_per_year{suffix}.csv")
    per_year.to_csv(per_year_path, index=False)
    print(f"  Saved: {per_year_path}")

    # --- By method × month ---
    print("  Computing per-month stats...")
    per_month = compute_stats_table(df, ['method', 'month'])
    per_month = per_month.sort_values(['method', 'month'])
    per_month_path = os.path.join(output_dir, f"satellite_cv_per_month{suffix}.csv")
    per_month.to_csv(per_month_path, index=False)
    print(f"  Saved: {per_month_path}")

    # --- By method × season ---
    print("  Computing per-season stats...")
    per_season = compute_stats_table(df, ['method', 'season'])
    per_season_path = os.path.join(output_dir, f"satellite_cv_per_season{suffix}.csv")
    per_season.to_csv(per_season_path, index=False)
    print(f"  Saved: {per_season_path}")

    # --- By method × hemisphere ---
    if 'hemisphere' in df.columns:
        print("  Computing per-hemisphere stats...")
        per_hemi = compute_stats_table(df, ['method', 'hemisphere'])
        per_hemi_path = os.path.join(output_dir, f"satellite_cv_per_hemisphere{suffix}.csv")
        per_hemi.to_csv(per_hemi_path, index=False)
        print(f"  Saved: {per_hemi_path}")

    # --- By method × data_source for eof_filtered analysis ---
    if 'is_eof_filtered_replaced' in df.columns:
        eof_df = df[df['method'] == 'eof_filtered']
        if len(eof_df) > 0:
            print("  Computing eof_filtered replaced vs unchanged stats...")
            eof_stats = compute_stats_table(eof_df, ['is_eof_filtered_replaced'])
            eof_stats['method'] = 'eof_filtered'
            eof_path = os.path.join(output_dir, f"satellite_cv_eof_filter_impact{suffix}.csv")
            eof_stats.to_csv(eof_path, index=False)
            print(f"  Saved: {eof_path}")

    # --- By method × quality_level ---
    if 'quality_level' in df.columns:
        ql_valid = df[df['quality_level'].notna() & (df['quality_level'] >= 3)]
        if len(ql_valid) > 0:
            print("  Computing per-quality-level stats...")
            per_ql = compute_stats_table(ql_valid, ['method', 'quality_level'])
            per_ql = per_ql.sort_values(['method', 'quality_level'])
            per_ql_path = os.path.join(output_dir, f"satellite_cv_per_quality_level{suffix}.csv")
            per_ql.to_csv(per_ql_path, index=False)
            print(f"  Saved: {per_ql_path}")

            # Also by method × quality_level × lake for detailed analysis
            print("  Computing per-lake per-quality-level stats...")
            per_lake_ql = compute_stats_table(ql_valid, ['method', 'lake_id', 'quality_level'])
            per_lake_ql = per_lake_ql.sort_values(['method', 'lake_id', 'quality_level'])
            per_lake_ql_path = os.path.join(output_dir, f"satellite_cv_per_lake_quality_level{suffix}.csv")
            per_lake_ql.to_csv(per_lake_ql_path, index=False)
            print(f"  Saved: {per_lake_ql_path}")

    return per_lake, global_stats


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


def get_post_dir(run_root: str, lake_id: int, alpha: str = 'a1000') -> Optional[str]:
    """Get post directory for a lake, handling both naming conventions."""
    for lake_str in [f"{lake_id:09d}", str(lake_id)]:
        path = os.path.join(run_root, "post", lake_str, alpha)
        if os.path.exists(path):
            return path
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Assemble satellite CV pixel pairs and compute statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  physical  (default) — Extract from post files in °C.
                         All 3 methods: dineof, eof_filtered, dincae.
  anomaly             — Extract from prepared.nc / results files in anomaly space.
                         Only 2 methods: dineof, dincae.

Output files (in --output-dir):
  satellite_cv_assembly_{mode}.parquet   — Full assembly (one row per CV pixel)
  satellite_cv_assembly_{mode}.csv       — Same as above, CSV format
  satellite_cv_per_lake_{mode}.csv       — Stats per lake × method
  satellite_cv_global_{mode}.csv         — Stats pooled by method
  satellite_cv_per_year_{mode}.csv       — Stats by method × year
  satellite_cv_per_month_{mode}.csv      — Stats by method × month
  satellite_cv_per_season_{mode}.csv     — Stats by method × season
  satellite_cv_per_hemisphere_{mode}.csv — Stats by method × hemisphere
  satellite_cv_per_quality_level_{mode}.csv      — Stats by method × QL (3,4,5)
  satellite_cv_per_lake_quality_level_{mode}.csv — Stats by method × lake × QL
  satellite_cv_eof_filter_impact_{mode}.csv      — EOF filter replaced vs unchanged

Examples:
    # Physical space, all lakes
    python assemble_satellite_cv.py --run-root /path/to/experiment

    # Anomaly space, specific lakes
    python assemble_satellite_cv.py --run-root /path/to/experiment \\
        --mode anomaly --lake-ids 88 375 4503

    # Physical + anomaly (both)
    python assemble_satellite_cv.py --run-root /path/to/experiment --mode both
        """
    )
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--mode", choices=["physical", "anomaly", "both"],
                        default="physical",
                        help="Extraction mode (default: physical)")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: {run_root}/satellite_cv_assembly/)")

    lake_group = parser.add_mutually_exclusive_group()
    lake_group.add_argument("--lake-id", type=int, help="Single lake ID")
    lake_group.add_argument("--lake-ids", type=int, nargs="+", help="Multiple lake IDs")
    lake_group.add_argument("--all", action="store_true", help="Process all lakes")

    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Disable fair comparison filtering")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing assembly CSV instead of overwriting. "
                             "Stats are recomputed from the full file.")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip CSV output for assembly (Parquet only)")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Skip Parquet output for assembly (CSV only)")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "satellite_cv_assembly")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine lake IDs
    if args.lake_id:
        lake_ids = [args.lake_id]
    elif args.lake_ids:
        lake_ids = args.lake_ids
    elif args.all:
        if HAS_COMPLETION_CHECK and not args.no_fair_comparison:
            print("=" * 70)
            print("FAIR COMPARISON MODE")
            print("=" * 70)
            lake_ids, _ = get_fair_comparison_lakes(args.run_root, args.alpha, verbose=True)
            if not lake_ids:
                print("No lakes with both methods complete. Use --no-fair-comparison.")
                sys.exit(1)
        else:
            lake_ids = find_lakes_with_post(args.run_root, args.alpha)
    else:
        lake_ids = find_lakes_with_post(args.run_root, args.alpha)

    if not lake_ids:
        print("No lakes to process")
        sys.exit(1)

    modes = []
    if args.mode == 'both':
        modes = ['physical', 'anomaly']
    else:
        modes = [args.mode]

    print("=" * 70)
    print("SATELLITE CV ASSEMBLY")
    print("=" * 70)
    print(f"Run root:   {args.run_root}")
    print(f"Mode(s):    {', '.join(modes)}")
    print(f"Lakes:      {len(lake_ids)}")
    print(f"Alpha:      {args.alpha}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*70}")

        t_start = time.time()

        # Incremental write: append each lake to CSV on disk to avoid
        # accumulating all DataFrames in memory (can be 30+ GB for 120 lakes)
        assembly_csv = os.path.join(args.output_dir,
                                    f"satellite_cv_assembly_{mode}.csv")
        n_lakes_done = 0
        n_rows_total = 0

        # If appending, keep existing file and skip header
        if args.append and os.path.exists(assembly_csv):
            header_written = True
            existing_size = os.path.getsize(assembly_csv) / 1e6
            print(f"  Appending to existing assembly ({existing_size:.1f} MB)")
        else:
            header_written = False
            # Truncate if exists
            if os.path.exists(assembly_csv):
                os.remove(assembly_csv)

        for lake_id in lake_ids:
            if mode == 'physical':
                post_dir = get_post_dir(args.run_root, lake_id, args.alpha)
                if post_dir is None:
                    if verbose:
                        print(f"  [Lake {lake_id}] No post directory found")
                    continue
                df = extract_physical_lake(post_dir, lake_id, verbose)
            else:  # anomaly
                df = extract_anomaly_lake(args.run_root, lake_id, args.alpha, verbose)

            if df is not None and len(df) > 0:
                # Append to CSV on disk
                df.to_csv(assembly_csv, index=False,
                          mode='a', header=(not header_written))
                header_written = True
                n_rows_total += len(df)
                n_lakes_done += 1
                del df  # free memory immediately

        if n_rows_total == 0:
            print(f"\nNo CV data extracted in {mode} mode")
            continue

        t_extract = time.time() - t_start
        size_mb = os.path.getsize(assembly_csv) / 1e6
        print(f"\nExtraction complete: {n_rows_total:,} rows from {n_lakes_done} lakes "
              f"({t_extract:.1f}s, CSV={size_mb:.1f} MB)")

        # Read back for stats (chunked to limit memory)
        print(f"\nReading assembly back for statistics...")
        assembly = pd.read_csv(assembly_csv, low_memory=False)
        assembly['date'] = pd.to_datetime(assembly['date'])

        for method in sorted(assembly['method'].unique()):
            n = len(assembly[assembly['method'] == method])
            print(f"  {method}: {n:,} CV pairs")

        # Optionally save Parquet copy
        if not args.no_parquet:
            try:
                pq_path = os.path.join(args.output_dir,
                                       f"satellite_cv_assembly_{mode}.parquet")
                assembly.to_parquet(pq_path, index=False, engine='pyarrow')
                size_mb = os.path.getsize(pq_path) / 1e6
                print(f"  Parquet: {pq_path} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  Parquet failed ({e}), skipping")

        if args.no_csv:
            # User doesn't want CSV, remove it
            os.remove(assembly_csv)

        # Compute stats
        print(f"\nComputing statistics...")
        generate_all_stats(assembly, args.output_dir, mode_suffix=mode)

        del assembly  # free before next mode
        t_total = time.time() - t_start
        print(f"\n{mode.upper()} mode complete ({t_total:.1f}s)")

    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
        print(f"  {f} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
