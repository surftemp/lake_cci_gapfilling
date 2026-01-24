#!/usr/bin/env python3
"""
Compare In-situ vs Satellite Observation Characteristics
=========================================================

This tests whether in-situ (buoy) measurements are systematically 
"damped" compared to satellite observations.

If the bulk effect hypothesis is true:
- In-situ should have lower amplitude (smaller seasonal swings)
- In-situ might be smoother

We compare FULL timeseries (not matched dates) to capture the true 
characteristics of each measurement type.

For amplitude: Range, IQR, STD - all well-defined for any timeseries
For roughness: Computed but with caveats (irregular sampling)

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
from scipy import stats

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

DEFAULT_BUOY_DIR = "/gws/ssde/j25b/nceo_uor/users/lcarrea01/INSITU/Buoy_Laura/ALL_FILES_QC"


# =============================================================================
# METRICS
# =============================================================================

def compute_amplitude_metrics(x: np.ndarray) -> Dict[str, float]:
    """
    Compute amplitude metrics for a timeseries.
    
    These are well-defined regardless of sampling regularity.
    """
    x = np.asarray(x).flatten()
    x = x[np.isfinite(x)]
    
    if len(x) < 10:
        return {
            'n': len(x),
            'mean': np.nan,
            'std': np.nan,
            'range': np.nan,
            'iqr': np.nan,
            'p05': np.nan,
            'p95': np.nan,
            'p95_p05_range': np.nan,
        }
    
    p05 = np.percentile(x, 5)
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    p95 = np.percentile(x, 95)
    
    return {
        'n': len(x),
        'mean': np.mean(x),
        'std': np.std(x),
        'range': np.max(x) - np.min(x),
        'iqr': p75 - p25,
        'p05': p05,
        'p95': p95,
        'p95_p05_range': p95 - p05,
    }


def compute_roughness(x: np.ndarray) -> float:
    """
    Compute roughness via 1 - lag-1 autocorrelation.
    
    WARNING: This is approximate for irregularly sampled data.
    """
    x = np.asarray(x).flatten()
    x = x[np.isfinite(x)]
    
    if len(x) < 10:
        return np.nan
    
    x_centered = x - np.mean(x)
    autocorr = np.corrcoef(x_centered[:-1], x_centered[1:])[0, 1]
    
    if np.isnan(autocorr):
        return np.nan
    
    return 1 - autocorr


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
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['lake_id', 'site_id'], keep='first')
    return combined


def get_lake_sites(selection_df: pd.DataFrame, lake_id: int) -> pd.DataFrame:
    """Get all buoy sites for a lake."""
    return selection_df[selection_df['lake_id'] == lake_id]


def get_representative_hour(selection_df: pd.DataFrame, lake_id: int, site_id: int) -> Optional[int]:
    """
    Find representative hour for satellite overpasses from selection CSV.
    
    The representative hour is the hour (of in-situ measurement) that appears 
    on the most unique days in the selection CSV.
    """
    subset = selection_df[(selection_df['lake_id'] == lake_id) & 
                          (selection_df['site_id'] == site_id)].copy()
    
    if subset.empty:
        return None
    
    # time_IS is the in-situ timestamp
    if 'time_IS' not in subset.columns:
        return None
    
    try:
        subset['time_IS'] = pd.to_datetime(subset['time_IS'])
        subset['hour'] = subset['time_IS'].dt.hour
        subset['date'] = subset['time_IS'].dt.date
        hour_day_counts = subset.groupby('hour')['date'].nunique()
        return hour_day_counts.idxmax() if not hour_day_counts.empty else None
    except:
        return None


def load_buoy_data(buoy_dir: str, lake_id: int, site_id: int, 
                   rep_hour: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load buoy temperature data from CSV file.
    
    File naming convention: ID{lake_id:06d}{site_id:02d}.csv
    
    Filters to representative hour for hourly data (to match satellite overpass time).
    
    Returns DataFrame with 'date' and 'temp' columns.
    """
    lake_id = int(lake_id)
    site_id = int(site_id)
    
    # Construct filename using the correct convention
    buoy_file = os.path.join(
        buoy_dir,
        f"ID{str(lake_id).zfill(6)}{str(site_id).zfill(2)}.csv"
    )
    
    if not os.path.exists(buoy_file):
        return None
    
    try:
        # Read CSV
        df = pd.read_csv(buoy_file, parse_dates=['dateTime'])
        
        # Temperature column is 'Tw'
        if 'Tw' not in df.columns:
            return None
        
        # Detect daily vs hourly data
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
        
        # Group by date and take mean (handles multiple readings per day)
        daily_temps = df.groupby(df['dateTime'].dt.date)['Tw'].mean()
        
        result = pd.DataFrame({
            'date': pd.to_datetime(daily_temps.index),
            'temp': daily_temps.values
        })
        
        # Convert to Celsius if needed
        if result['temp'].mean() > 100:
            result['temp'] = result['temp'] - 273.15
        
        result = result.dropna()
        return result
        
    except Exception as e:
        return None


def find_nearest_pixel(lat_array: np.ndarray, lon_array: np.ndarray,
                       target_lat: float, target_lon: float) -> Tuple[int, int]:
    """Find nearest grid point."""
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    return np.unravel_index(np.argmin(distance), distance.shape)


def load_satellite_obs_at_pixel(nc_path: str, lat_idx: int, lon_idx: int,
                                 quality_threshold: int = 3) -> np.ndarray:
    """
    Load satellite observations at a pixel, applying quality filter.
    
    Returns array of valid temperature observations.
    """
    with xr.open_dataset(nc_path) as ds:
        # Get temperature
        if 'lswt' in ds:
            temps = ds['lswt'].isel(lat=lat_idx, lon=lon_idx).values
        elif 'temp_filled' in ds:
            # Use original obs from a reconstruction file
            # Need to identify which are original vs filled
            temps = ds['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values
        else:
            return np.array([])
        
        # Apply quality filter if available
        if 'quality_level' in ds:
            quality = ds['quality_level'].isel(lat=lat_idx, lon=lon_idx).values
            temps = np.where(quality >= quality_threshold, temps, np.nan)
        
        # Convert to Celsius
        if np.nanmean(temps) > 100:
            temps = temps - 273.15
        
        # Return only valid values
        return temps[np.isfinite(temps)]


def find_original_lake_file(run_root: str, lake_id: int) -> Optional[str]:
    """Find original satellite timeseries file for a lake."""
    # Common locations for original lake data
    possible_dirs = [
        "/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/lake_ts",
        "/gws/nopw/j04/esacci_lakes/satellite/products/L3S_QL",
    ]
    
    lake_str = f"{lake_id:09d}"
    
    for base_dir in possible_dirs:
        if not os.path.exists(base_dir):
            continue
        
        patterns = [
            os.path.join(base_dir, f"LAKE{lake_str}-*.nc"),
            os.path.join(base_dir, f"LAKE{lake_str}*.nc"),
            os.path.join(base_dir, "**", f"LAKE{lake_str}-*.nc"),
        ]
        
        for pattern in patterns:
            files = glob(pattern, recursive=True)
            if files:
                return files[0]
    
    return None


def load_satellite_from_dineof_output(nc_path: str, lat_idx: int, lon_idx: int,
                                       quality_threshold: int = 3) -> np.ndarray:
    """
    Extract original satellite observations from DINEOF output file.
    
    DINEOF output contains both original obs and filled values.
    Original observations are identified by quality_level >= threshold.
    
    Returns array of valid temperature observations (only original obs, not gap-filled).
    """
    with xr.open_dataset(nc_path) as ds:
        temps = ds['temp_filled'].isel(lat=lat_idx, lon=lon_idx).values
        
        # Identify original observations using quality_level
        if 'quality_level' in ds:
            quality = ds['quality_level'].isel(lat=lat_idx, lon=lon_idx).values
            # Only keep original observations (quality_level >= threshold)
            # NaN quality means it was gap-filled
            valid_mask = np.isfinite(quality) & (quality >= quality_threshold)
            temps = temps[valid_mask]
        else:
            # If no quality level, assume all valid values are observations
            temps = temps[np.isfinite(temps)]
        
        # Convert to Celsius
        if len(temps) > 0 and np.nanmean(temps) > 100:
            temps = temps - 273.15
        
        return temps


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_lake(run_root: str, lake_id: int, alpha: str,
                 selection_df: pd.DataFrame, buoy_dir: str,
                 quality_threshold: int = 3,
                 verbose: bool = False) -> Optional[Dict]:
    """
    Compare in-situ vs satellite characteristics for a lake.
    
    Uses FULL timeseries from each source (not matched dates).
    """
    # Get buoy sites for this lake
    sites = get_lake_sites(selection_df, lake_id)
    if sites.empty:
        if verbose:
            print(f"  Lake {lake_id}: no buoy sites in selection")
        return None
    
    # Use first site
    site_row = sites.iloc[0]
    site_id = site_row['site_id']
    target_lat = site_row['latitude']
    target_lon = site_row['longitude']
    
    # Get representative hour for this site
    rep_hour = get_representative_hour(selection_df, lake_id, site_id)
    
    # Load buoy data with representative hour filtering
    buoy_df = load_buoy_data(buoy_dir, lake_id, site_id, rep_hour)
    if buoy_df is None:
        if verbose:
            buoy_file = os.path.join(buoy_dir, f"ID{str(lake_id).zfill(6)}{str(site_id).zfill(2)}.csv")
            print(f"  Lake {lake_id}: buoy file not found or couldn't load ({os.path.basename(buoy_file)})")
        return None
    
    if len(buoy_df) < 50:
        if verbose:
            print(f"  Lake {lake_id}: insufficient buoy data ({len(buoy_df)} points)")
        return None
    
    insitu_temps = buoy_df['temp'].values
    
    # Find DINEOF output file to extract satellite obs
    lake_str = f"{lake_id:09d}"
    post_dir = os.path.join(run_root, "post", lake_str, alpha)
    if not os.path.exists(post_dir):
        post_dir = os.path.join(run_root, "post", str(lake_id), alpha)
    
    if not os.path.exists(post_dir):
        if verbose:
            print(f"  Lake {lake_id}: post directory not found")
        return None
    
    # Find DINEOF output
    dineof_files = glob(os.path.join(post_dir, "*_dineof.nc"))
    if not dineof_files:
        if verbose:
            print(f"  Lake {lake_id}: no DINEOF output found in {post_dir}")
        return None
    
    dineof_path = dineof_files[0]
    
    # Get pixel indices
    with xr.open_dataset(dineof_path) as ds:
        lat_idx, lon_idx = find_nearest_pixel(
            ds['lat'].values, ds['lon'].values, target_lat, target_lon
        )
    
    # Extract satellite observations
    satellite_temps = load_satellite_from_dineof_output(
        dineof_path, lat_idx, lon_idx, quality_threshold
    )
    
    if len(satellite_temps) < 50:
        if verbose:
            print(f"  Lake {lake_id}: insufficient satellite obs ({len(satellite_temps)} points)")
        return None
    
    # Compute metrics for both
    insitu_metrics = compute_amplitude_metrics(insitu_temps)
    satellite_metrics = compute_amplitude_metrics(satellite_temps)
    
    insitu_roughness = compute_roughness(insitu_temps)
    satellite_roughness = compute_roughness(satellite_temps)
    
    result = {
        'lake_id': lake_id,
        'site_id': site_id,
        'buoy_lat': target_lat,
        'buoy_lon': target_lon,
        
        # In-situ metrics
        'insitu_n': insitu_metrics['n'],
        'insitu_mean': insitu_metrics['mean'],
        'insitu_std': insitu_metrics['std'],
        'insitu_range': insitu_metrics['range'],
        'insitu_iqr': insitu_metrics['iqr'],
        'insitu_p95_p05': insitu_metrics['p95_p05_range'],
        'insitu_roughness': insitu_roughness,
        
        # Satellite metrics
        'satellite_n': satellite_metrics['n'],
        'satellite_mean': satellite_metrics['mean'],
        'satellite_std': satellite_metrics['std'],
        'satellite_range': satellite_metrics['range'],
        'satellite_iqr': satellite_metrics['iqr'],
        'satellite_p95_p05': satellite_metrics['p95_p05_range'],
        'satellite_roughness': satellite_roughness,
    }
    
    if verbose:
        print(f"  Lake {lake_id}: insitu_n={insitu_metrics['n']}, sat_n={satellite_metrics['n']}, "
              f"insitu_std={insitu_metrics['std']:.2f}, sat_std={satellite_metrics['std']:.2f}")
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_plots(results_df: pd.DataFrame, output_dir: str):
    """Create visualizations comparing in-situ vs satellite."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # A) STD comparison
    ax = axes[0, 0]
    x = results_df['satellite_std']
    y = results_df['insitu_std']
    
    ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
    
    insitu_lower = (y < x).sum()
    total = len(x)
    
    ax.set_xlabel('Satellite STD [°C]')
    ax.set_ylabel('In-situ STD [°C]')
    ax.set_title(f'A) Standard Deviation\n'
                 f'In-situ lower: {insitu_lower}/{total} ({100*insitu_lower/total:.0f}%)')
    ax.legend()
    
    # B) IQR comparison
    ax = axes[0, 1]
    x = results_df['satellite_iqr']
    y = results_df['insitu_iqr']
    
    ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    insitu_lower = (y < x).sum()
    
    ax.set_xlabel('Satellite IQR [°C]')
    ax.set_ylabel('In-situ IQR [°C]')
    ax.set_title(f'B) Interquartile Range\n'
                 f'In-situ lower: {insitu_lower}/{total} ({100*insitu_lower/total:.0f}%)')
    
    # C) P95-P05 range comparison
    ax = axes[0, 2]
    x = results_df['satellite_p95_p05']
    y = results_df['insitu_p95_p05']
    
    ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    insitu_lower = (y < x).sum()
    
    ax.set_xlabel('Satellite P95-P05 [°C]')
    ax.set_ylabel('In-situ P95-P05 [°C]')
    ax.set_title(f'C) 90% Range (P95-P05)\n'
                 f'In-situ lower: {insitu_lower}/{total} ({100*insitu_lower/total:.0f}%)')
    
    # D) Roughness comparison (with caveat)
    ax = axes[1, 0]
    x = results_df['satellite_roughness']
    y = results_df['insitu_roughness']
    
    ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    insitu_smoother = (y < x).sum()
    
    ax.set_xlabel('Satellite Roughness (1-autocorr)')
    ax.set_ylabel('In-situ Roughness (1-autocorr)')
    ax.set_title(f'D) Roughness (⚠️ irregular sampling)\n'
                 f'In-situ smoother: {insitu_smoother}/{total} ({100*insitu_smoother/total:.0f}%)')
    
    # E) Summary bar chart - amplitude
    ax = axes[1, 1]
    
    metrics = ['STD', 'IQR', 'P95-P05']
    insitu_lower_counts = [
        (results_df['insitu_std'] < results_df['satellite_std']).sum(),
        (results_df['insitu_iqr'] < results_df['satellite_iqr']).sum(),
        (results_df['insitu_p95_p05'] < results_df['satellite_p95_p05']).sum(),
    ]
    percentages = [100 * c / total for c in insitu_lower_counts]
    
    bars = ax.bar(metrics, percentages, color=['#3498db', '#2ecc71', '#9b59b6'], alpha=0.7)
    ax.axhline(50, color='red', linestyle='--', label='50% (no difference)')
    ax.set_ylabel('% lakes where in-situ is lower')
    ax.set_title('E) Summary: In-situ vs Satellite Amplitude')
    ax.set_ylim(0, 100)
    ax.legend()
    
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 2, f'{pct:.0f}%', ha='center')
    
    # F) Distribution of STD differences
    ax = axes[1, 2]
    diff = results_df['insitu_std'] - results_df['satellite_std']
    
    ax.hist(diff, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal')
    ax.axvline(diff.mean(), color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {diff.mean():.2f}°C')
    
    ax.set_xlabel('In-situ STD - Satellite STD [°C]')
    ax.set_ylabel('Count')
    ax.set_title('F) Distribution of STD Difference\n(Negative = In-situ damped)')
    ax.legend()
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'insitu_vs_satellite_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_summary_report(results_df: pd.DataFrame, output_dir: str):
    """Create text summary."""
    
    report = []
    report.append("=" * 70)
    report.append("IN-SITU vs SATELLITE OBSERVATION CHARACTERISTICS")
    report.append("=" * 70)
    report.append(f"\nTotal lakes analyzed: {len(results_df)}")
    report.append(f"Mean in-situ observations: {results_df['insitu_n'].mean():.0f}")
    report.append(f"Mean satellite observations: {results_df['satellite_n'].mean():.0f}")
    
    n = len(results_df)
    
    report.append("\n" + "-" * 70)
    report.append("AMPLITUDE COMPARISON (Is in-situ damped?)")
    report.append("-" * 70)
    
    for metric, col_suffix, label in [
        ('STD', 'std', 'Standard Deviation'),
        ('IQR', 'iqr', 'Interquartile Range'),
        ('P95-P05', 'p95_p05', '90% Range'),
    ]:
        insitu_col = f'insitu_{col_suffix}'
        sat_col = f'satellite_{col_suffix}'
        
        insitu_lower = (results_df[insitu_col] < results_df[sat_col]).sum()
        insitu_mean = results_df[insitu_col].mean()
        sat_mean = results_df[sat_col].mean()
        diff_mean = (results_df[insitu_col] - results_df[sat_col]).mean()
        
        report.append(f"\n  {label}:")
        report.append(f"    In-situ mean: {insitu_mean:.2f}°C")
        report.append(f"    Satellite mean: {sat_mean:.2f}°C")
        report.append(f"    In-situ lower: {insitu_lower}/{n} ({100*insitu_lower/n:.0f}%)")
        report.append(f"    Mean difference: {diff_mean:.2f}°C")
    
    # Statistical test on STD
    t_stat, p_value = stats.ttest_rel(results_df['insitu_std'], results_df['satellite_std'])
    report.append(f"\n  Paired t-test (STD): t={t_stat:.3f}, p={p_value:.4f}")
    
    insitu_std_lower = (results_df['insitu_std'] < results_df['satellite_std']).sum()
    if p_value < 0.05:
        if insitu_std_lower > n / 2:
            report.append("  --> In-situ is significantly LOWER amplitude (p < 0.05)")
            report.append("  --> CONSISTENT with bulk/damping hypothesis")
        else:
            report.append("  --> In-situ is significantly HIGHER amplitude (p < 0.05)")
            report.append("  --> NOT consistent with bulk/damping hypothesis")
    else:
        report.append("  --> No significant difference (p >= 0.05)")
    
    report.append("\n" + "-" * 70)
    report.append("ROUGHNESS COMPARISON (⚠️ Use with caution - irregular sampling)")
    report.append("-" * 70)
    
    insitu_smoother = (results_df['insitu_roughness'] < results_df['satellite_roughness']).sum()
    report.append(f"\n  In-situ smoother: {insitu_smoother}/{n} ({100*insitu_smoother/n:.0f}%)")
    report.append("  NOTE: Roughness metric is approximate for sparse/irregular data")
    
    report.append("\n" + "-" * 70)
    report.append("INTERPRETATION")
    report.append("-" * 70)
    report.append("\n  If in-situ has LOWER amplitude than satellite:")
    report.append("    - Consistent with bulk effect (buoys measure damped subsurface temp)")
    report.append("    - DINCAE's smoother reconstructions would match in-situ better")
    report.append("    - This could explain why DINCAE improves from 3% to 40% wins")
    report.append("\n  If in-situ has SIMILAR or HIGHER amplitude:")
    report.append("    - Bulk effect hypothesis not supported")
    report.append("    - Need alternative explanation for CV discrepancy")
    
    report.append("\n" + "-" * 70)
    report.append("CAVEATS")
    report.append("-" * 70)
    report.append("  - In-situ and satellite have different sampling times")
    report.append("  - Amplitude differences could be due to spatial scale differences")
    report.append("  - Buoy sensor depth varies between deployments")
    report.append("  - This shows correlation, not causation")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    
    out_path = os.path.join(output_dir, 'insitu_vs_satellite_summary.txt')
    with open(out_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare in-situ vs satellite observation characteristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests whether in-situ measurements are systematically damped compared 
to satellite observations.

Uses FULL timeseries from each source (not matched dates) to capture
true characteristics.

Example:
    python compare_insitu_vs_satellite.py --run-root /path/to/experiment
        """
    )
    
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--buoy-dir", default=DEFAULT_BUOY_DIR, help="Buoy data directory")
    parser.add_argument("--selection-csvs", nargs="+", default=None,
                        help="Selection CSV files")
    parser.add_argument("--quality-threshold", type=int, default=3,
                        help="Quality threshold for satellite obs")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "insitu_vs_satellite_comparison")
    os.makedirs(args.output_dir, exist_ok=True)
    
    selection_csvs = args.selection_csvs or DEFAULT_SELECTION_CSVS
    
    print("=" * 70)
    print("IN-SITU vs SATELLITE COMPARISON")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Buoy dir: {args.buoy_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Load selection CSVs
    selection_df = load_selection_csvs(selection_csvs)
    if selection_df.empty:
        print("ERROR: No selection CSVs loaded")
        return 1
    
    lake_ids = selection_df['lake_id'].unique().tolist()
    print(f"\nFound {len(lake_ids)} lakes with buoy data")
    
    # Analyze each lake
    results = []
    for lake_id in sorted(lake_ids):
        result = analyze_lake(
            args.run_root, lake_id, args.alpha,
            selection_df, args.buoy_dir,
            args.quality_threshold, args.verbose
        )
        if result is not None:
            results.append(result)
    
    print(f"\nSuccessfully analyzed {len(results)} lakes")
    
    if len(results) < 3:
        print("ERROR: Not enough lakes with complete data")
        return 1
    
    results_df = pd.DataFrame(results)
    
    # Save raw data
    csv_path = os.path.join(args.output_dir, 'insitu_vs_satellite_data.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Generate outputs
    print("\nGenerating plots...")
    create_comparison_plots(results_df, args.output_dir)
    create_summary_report(results_df, args.output_dir)
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
