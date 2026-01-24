#!/usr/bin/env python3
"""
Compare DINEOF vs DINCAE Reconstruction Roughness
==================================================

This is the simplest, most unambiguous test:
- Compare full DINEOF and DINCAE reconstructions at the buoy pixel location
- Both are dense daily timeseries, so roughness is well-defined
- No date matching needed - we use the full reconstruction length

Roughness metric: 1 - lag1_autocorrelation
- Higher = rougher (more high-frequency variability)
- Lower = smoother

This answers: "Which method produces smoother reconstructions?"

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
# METRICS
# =============================================================================

def compute_roughness(x: np.ndarray) -> float:
    """
    Compute roughness via 1 - lag-1 autocorrelation.
    
    For dense regular timeseries (like gap-filled reconstructions),
    this is a well-defined measure of high-frequency variability.
    
    Returns:
        1 - lag1_autocorr: Higher = rougher, Lower = smoother
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


def compute_normalized_diff_variance(x: np.ndarray) -> float:
    """
    Alternative roughness: Var(diff) / Var(signal)
    
    Higher = rougher (larger jumps relative to overall variance)
    """
    x = np.asarray(x).flatten()
    x = x[np.isfinite(x)]
    
    if len(x) < 10:
        return np.nan
    
    var_x = np.var(x)
    if var_x < 1e-10:
        return np.nan
    
    diff = np.diff(x)
    var_diff = np.var(diff)
    
    return var_diff / var_x


def compute_timeseries_stats(x: np.ndarray) -> Dict[str, float]:
    """Compute basic timeseries statistics."""
    x = np.asarray(x).flatten()
    x_valid = x[np.isfinite(x)]
    
    if len(x_valid) < 10:
        return {
            'n_valid': len(x_valid),
            'n_total': len(x),
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'range': np.nan,
            'roughness_autocorr': np.nan,
            'roughness_diffvar': np.nan,
        }
    
    return {
        'n_valid': len(x_valid),
        'n_total': len(x),
        'mean': np.mean(x_valid),
        'std': np.std(x_valid),
        'min': np.min(x_valid),
        'max': np.max(x_valid),
        'range': np.max(x_valid) - np.min(x_valid),
        'roughness_autocorr': compute_roughness(x_valid),
        'roughness_diffvar': compute_normalized_diff_variance(x_valid),
    }


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
    # Remove duplicates (keep first occurrence - priority order)
    combined = combined.drop_duplicates(subset=['lake_id', 'site_id'], keep='first')
    return combined


def get_lake_buoy_location(selection_df: pd.DataFrame, lake_id: int) -> Optional[Tuple[float, float]]:
    """Get buoy location for a lake from selection CSV."""
    lake_df = selection_df[selection_df['lake_id'] == lake_id]
    if lake_df.empty:
        return None
    
    # Take first site
    row = lake_df.iloc[0]
    return (row['latitude'], row['longitude'])


def find_nearest_pixel(lat_array: np.ndarray, lon_array: np.ndarray,
                       target_lat: float, target_lon: float) -> Tuple[int, int]:
    """Find nearest grid point to target coordinates."""
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    idx = np.unravel_index(np.argmin(distance), distance.shape)
    return idx


def find_output_files(post_dir: str, lake_id: int) -> Dict[str, Optional[str]]:
    """Find DINEOF and DINCAE output files for a lake."""
    lake_str = f"{lake_id:09d}"
    
    outputs = {
        'dineof': None,
        'dincae': None,
    }
    
    # Try padded lake ID first
    patterns = [
        (f"LAKE{lake_str}-*_dineof.nc", 'dineof'),
        (f"LAKE{lake_str}-*_dincae.nc", 'dincae'),
    ]
    
    for pattern, key in patterns:
        files = glob(os.path.join(post_dir, pattern))
        if files:
            outputs[key] = files[0]
    
    # Try unpadded if not found
    if not outputs['dineof']:
        files = glob(os.path.join(post_dir, f"LAKE*_dineof.nc"))
        if files:
            outputs['dineof'] = files[0]
    
    if not outputs['dincae']:
        files = glob(os.path.join(post_dir, f"LAKE*_dincae.nc"))
        if files:
            outputs['dincae'] = files[0]
    
    return outputs


def extract_pixel_timeseries(nc_path: str, lat_idx: int, lon_idx: int,
                              var_name: str = 'temp_filled') -> np.ndarray:
    """Extract full timeseries at a pixel location."""
    with xr.open_dataset(nc_path) as ds:
        if var_name not in ds:
            return np.array([])
        
        temps = ds[var_name].isel(lat=lat_idx, lon=lon_idx).values
        
        # Convert to Celsius if needed
        if np.nanmean(temps) > 100:
            temps = temps - 273.15
        
        return temps


def get_grid_indices_from_nc(nc_path: str, target_lat: float, target_lon: float) -> Tuple[int, int]:
    """Get pixel indices from a NetCDF file for target coordinates."""
    with xr.open_dataset(nc_path) as ds:
        lat_array = ds['lat'].values
        lon_array = ds['lon'].values
        return find_nearest_pixel(lat_array, lon_array, target_lat, target_lon)


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_lake(run_root: str, lake_id: int, alpha: str,
                 selection_df: pd.DataFrame, verbose: bool = False) -> Optional[Dict]:
    """
    Analyze roughness of DINEOF vs DINCAE full reconstructions at buoy location.
    """
    # Get buoy location
    buoy_loc = get_lake_buoy_location(selection_df, lake_id)
    if buoy_loc is None:
        if verbose:
            print(f"  Lake {lake_id}: no buoy location found")
        return None
    
    target_lat, target_lon = buoy_loc
    
    # Find output files
    lake_str = f"{lake_id:09d}"
    post_dir = os.path.join(run_root, "post", lake_str, alpha)
    if not os.path.exists(post_dir):
        post_dir = os.path.join(run_root, "post", str(lake_id), alpha)
    
    if not os.path.exists(post_dir):
        if verbose:
            print(f"  Lake {lake_id}: post directory not found")
        return None
    
    outputs = find_output_files(post_dir, lake_id)
    
    if not outputs['dineof'] or not outputs['dincae']:
        if verbose:
            missing = []
            if not outputs['dineof']: missing.append('DINEOF')
            if not outputs['dincae']: missing.append('DINCAE')
            print(f"  Lake {lake_id}: missing {', '.join(missing)}")
        return None
    
    # Get pixel indices (use DINEOF file as reference)
    lat_idx, lon_idx = get_grid_indices_from_nc(outputs['dineof'], target_lat, target_lon)
    
    # Extract full timeseries
    dineof_ts = extract_pixel_timeseries(outputs['dineof'], lat_idx, lon_idx)
    dincae_ts = extract_pixel_timeseries(outputs['dincae'], lat_idx, lon_idx)
    
    if len(dineof_ts) < 100 or len(dincae_ts) < 100:
        if verbose:
            print(f"  Lake {lake_id}: timeseries too short (DINEOF={len(dineof_ts)}, DINCAE={len(dincae_ts)})")
        return None
    
    # Compute stats
    dineof_stats = compute_timeseries_stats(dineof_ts)
    dincae_stats = compute_timeseries_stats(dincae_ts)
    
    result = {
        'lake_id': lake_id,
        'buoy_lat': target_lat,
        'buoy_lon': target_lon,
        'lat_idx': lat_idx,
        'lon_idx': lon_idx,
        
        # DINEOF stats
        'dineof_n_valid': dineof_stats['n_valid'],
        'dineof_mean': dineof_stats['mean'],
        'dineof_std': dineof_stats['std'],
        'dineof_range': dineof_stats['range'],
        'dineof_roughness': dineof_stats['roughness_autocorr'],
        'dineof_roughness_diffvar': dineof_stats['roughness_diffvar'],
        
        # DINCAE stats
        'dincae_n_valid': dincae_stats['n_valid'],
        'dincae_mean': dincae_stats['mean'],
        'dincae_std': dincae_stats['std'],
        'dincae_range': dincae_stats['range'],
        'dincae_roughness': dincae_stats['roughness_autocorr'],
        'dincae_roughness_diffvar': dincae_stats['roughness_diffvar'],
    }
    
    if verbose:
        print(f"  Lake {lake_id}: N={dineof_stats['n_valid']}, "
              f"DINEOF roughness={dineof_stats['roughness_autocorr']:.4f}, "
              f"DINCAE roughness={dincae_stats['roughness_autocorr']:.4f}")
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_plot(results_df: pd.DataFrame, output_dir: str):
    """Create visualization comparing DINEOF vs DINCAE roughness."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A) Scatter: DINEOF vs DINCAE roughness
    ax = axes[0, 0]
    x = results_df['dincae_roughness']
    y = results_df['dineof_roughness']
    
    ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    # 1:1 line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
    
    dineof_smoother = (y < x).sum()
    total = len(x)
    
    ax.set_xlabel('DINCAE Roughness (1 - autocorr)')
    ax.set_ylabel('DINEOF Roughness (1 - autocorr)')
    ax.set_title(f'A) Roughness: DINEOF vs DINCAE\n'
                 f'DINEOF smoother: {dineof_smoother}/{total} ({100*dineof_smoother/total:.0f}%)')
    ax.legend()
    
    # B) Histogram of roughness difference
    ax = axes[0, 1]
    diff = results_df['dineof_roughness'] - results_df['dincae_roughness']
    
    ax.hist(diff, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal roughness')
    ax.axvline(diff.mean(), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean diff: {diff.mean():.4f}')
    
    ax.set_xlabel('DINEOF - DINCAE Roughness')
    ax.set_ylabel('Count')
    ax.set_title('B) Distribution of Roughness Difference\n'
                 '(Negative = DINEOF smoother)')
    ax.legend()
    
    # C) Alternative metric: normalized diff variance
    ax = axes[1, 0]
    x = results_df['dincae_roughness_diffvar']
    y = results_df['dineof_roughness_diffvar']
    
    ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    dineof_smoother_alt = (y < x).sum()
    
    ax.set_xlabel('DINCAE Var(diff)/Var(signal)')
    ax.set_ylabel('DINEOF Var(diff)/Var(signal)')
    ax.set_title(f'C) Alt Roughness: Normalized Diff Variance\n'
                 f'DINEOF smoother: {dineof_smoother_alt}/{total} ({100*dineof_smoother_alt/total:.0f}%)')
    
    # D) STD comparison
    ax = axes[1, 1]
    x = results_df['dincae_std']
    y = results_df['dineof_std']
    
    ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    dineof_lower_std = (y < x).sum()
    
    ax.set_xlabel('DINCAE STD [°C]')
    ax.set_ylabel('DINEOF STD [°C]')
    ax.set_title(f'D) Variability: Standard Deviation\n'
                 f'DINEOF lower STD: {dineof_lower_std}/{total} ({100*dineof_lower_std/total:.0f}%)')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'dineof_vs_dincae_roughness.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_summary_report(results_df: pd.DataFrame, output_dir: str):
    """Create text summary."""
    
    report = []
    report.append("=" * 70)
    report.append("DINEOF vs DINCAE RECONSTRUCTION ROUGHNESS COMPARISON")
    report.append("=" * 70)
    report.append(f"\nTotal lakes analyzed: {len(results_df)}")
    report.append(f"Mean timeseries length: {results_df['dineof_n_valid'].mean():.0f} days")
    
    report.append("\n" + "-" * 70)
    report.append("ROUGHNESS (1 - lag1_autocorrelation)")
    report.append("-" * 70)
    report.append("  Higher = rougher, Lower = smoother")
    
    n = len(results_df)
    dineof_mean = results_df['dineof_roughness'].mean()
    dincae_mean = results_df['dincae_roughness'].mean()
    dineof_smoother = (results_df['dineof_roughness'] < results_df['dincae_roughness']).sum()
    
    report.append(f"\n  DINEOF mean roughness: {dineof_mean:.4f}")
    report.append(f"  DINCAE mean roughness: {dincae_mean:.4f}")
    report.append(f"  Difference (DINEOF - DINCAE): {dineof_mean - dincae_mean:.4f}")
    report.append(f"\n  DINEOF smoother: {dineof_smoother}/{n} ({100*dineof_smoother/n:.0f}%)")
    report.append(f"  DINCAE smoother: {n - dineof_smoother}/{n} ({100*(n-dineof_smoother)/n:.0f}%)")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results_df['dineof_roughness'], results_df['dincae_roughness'])
    report.append(f"\n  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        if dineof_mean < dincae_mean:
            report.append("  --> DINEOF is significantly smoother (p < 0.05)")
        else:
            report.append("  --> DINCAE is significantly smoother (p < 0.05)")
    else:
        report.append("  --> No significant difference (p >= 0.05)")
    
    report.append("\n" + "-" * 70)
    report.append("ALTERNATIVE METRIC: Var(diff)/Var(signal)")
    report.append("-" * 70)
    
    dineof_mean_alt = results_df['dineof_roughness_diffvar'].mean()
    dincae_mean_alt = results_df['dincae_roughness_diffvar'].mean()
    dineof_smoother_alt = (results_df['dineof_roughness_diffvar'] < results_df['dincae_roughness_diffvar']).sum()
    
    report.append(f"\n  DINEOF mean: {dineof_mean_alt:.4f}")
    report.append(f"  DINCAE mean: {dincae_mean_alt:.4f}")
    report.append(f"  DINEOF smoother: {dineof_smoother_alt}/{n} ({100*dineof_smoother_alt/n:.0f}%)")
    
    report.append("\n" + "-" * 70)
    report.append("NOTES")
    report.append("-" * 70)
    report.append("  - Analysis uses FULL reconstruction timeseries (all timestamps)")
    report.append("  - No date matching - reconstructions are dense daily data")
    report.append("  - Extracted at buoy pixel location")
    report.append("  - This is unambiguous: which method produces smoother reconstructions?")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    
    out_path = os.path.join(output_dir, 'roughness_comparison_summary.txt')
    with open(out_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare DINEOF vs DINCAE roughness on full reconstructions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script compares the roughness of DINEOF vs DINCAE reconstructions.

Uses FULL reconstruction timeseries at buoy pixel locations.
Both methods produce dense daily data, so roughness is well-defined.

Outputs:
  - dineof_vs_dincae_roughness.png: Visual comparison
  - roughness_comparison_summary.txt: Statistical summary
  - roughness_comparison_data.csv: Raw data

Example:
    python compare_recon_roughness.py --run-root /path/to/experiment
        """
    )
    
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--selection-csvs", nargs="+", default=None,
                        help="Selection CSV files (default: standard 2010, 2007, 2018, 2020)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Defaults
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "roughness_comparison")
    os.makedirs(args.output_dir, exist_ok=True)
    
    selection_csvs = args.selection_csvs or DEFAULT_SELECTION_CSVS
    
    print("=" * 70)
    print("DINEOF vs DINCAE ROUGHNESS COMPARISON")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Alpha: {args.alpha}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Load selection CSVs
    selection_df = load_selection_csvs(selection_csvs)
    if selection_df.empty:
        print("ERROR: No selection CSVs loaded")
        return 1
    
    # Get unique lake IDs from selection
    lake_ids = selection_df['lake_id'].unique().tolist()
    print(f"\nFound {len(lake_ids)} lakes with buoy data in selection CSVs")
    
    # Analyze each lake
    results = []
    for lake_id in sorted(lake_ids):
        result = analyze_lake(
            args.run_root, lake_id, args.alpha,
            selection_df, verbose=args.verbose
        )
        if result is not None:
            results.append(result)
    
    print(f"\nSuccessfully analyzed {len(results)} lakes")
    
    if len(results) < 3:
        print("ERROR: Not enough lakes with complete data")
        return 1
    
    # Build DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw data
    csv_path = os.path.join(args.output_dir, 'roughness_comparison_data.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Generate outputs
    print("\nGenerating plots...")
    create_comparison_plot(results_df, args.output_dir)
    create_summary_report(results_df, args.output_dir)
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
