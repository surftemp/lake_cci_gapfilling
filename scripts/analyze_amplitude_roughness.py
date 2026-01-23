#!/usr/bin/env python3
"""
Amplitude vs Roughness Analysis for Bulk Effect Hypothesis
===========================================================

This script tests whether the CV validation discrepancy (DINCAE wins satellite CV,
DINEOF wins in-situ CV) can be explained by differences in signal characteristics.

Hypothesis:
- In-situ buoys may measure a "damped" signal (lower amplitude, smoother) compared 
  to satellite skin temperature
- If so, the method that produces smoother/lower-amplitude reconstructions would
  match in-situ better

Key distinction:
- AMPLITUDE: Range of signal (peak to trough) - measured by IQR
- ROUGHNESS: High-frequency variability - measured by lag-1 autocorrelation or 
  first-difference variance

Same STD can arise from:
- Large smooth oscillations (high amplitude, low roughness)  
- Flat signal with noisy jumps (low amplitude, high roughness)

Tests:
1. Characterize ground truths: Is in-situ lower amplitude? Smoother? Both?
2. Characterize methods: Is DINEOF lower amplitude? Smoother? Both?
3. What predicts winning: amplitude matching, roughness matching, or both?

Limitations (explicitly stated):
- These tests show CONSISTENCY with hypotheses, not proof
- Correlation ≠ causation
- Alternative explanations may exist

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
import matplotlib.pyplot as plt
from scipy import stats

# Add path for importing from pipeline
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from completion_check import get_fair_comparison_lakes
    HAS_COMPLETION_CHECK = True
except ImportError:
    HAS_COMPLETION_CHECK = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("ERROR: xarray is required. Install with: pip install xarray")
    sys.exit(1)


# =============================================================================
# METRICS: Amplitude and Roughness
# =============================================================================

def compute_amplitude(x: np.ndarray) -> float:
    """
    Compute amplitude as IQR (interquartile range).
    
    IQR is robust to outliers and captures the "typical" range of the signal.
    
    Args:
        x: 1D array of values
    
    Returns:
        IQR (75th - 25th percentile)
    """
    x = np.asarray(x).flatten()
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    return np.percentile(x, 75) - np.percentile(x, 25)


def compute_roughness_autocorr(x: np.ndarray) -> float:
    """
    Compute roughness via lag-1 autocorrelation.
    
    High autocorrelation = smooth (consecutive values similar)
    Low autocorrelation = rough (consecutive values jump around)
    
    Returns 1 - autocorr so that HIGHER = ROUGHER (easier to interpret)
    
    Args:
        x: 1D array of values (should be temporally ordered)
    
    Returns:
        1 - lag1_autocorrelation (0 = perfectly smooth, 1 = no autocorrelation)
    """
    x = np.asarray(x).flatten()
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    
    # Lag-1 autocorrelation
    x_centered = x - np.mean(x)
    autocorr = np.corrcoef(x_centered[:-1], x_centered[1:])[0, 1]
    
    if np.isnan(autocorr):
        return np.nan
    
    # Return 1 - autocorr so higher = rougher
    return 1 - autocorr


def compute_roughness_diffvar(x: np.ndarray) -> float:
    """
    Compute roughness via normalized first-difference variance.
    
    Var(x[t+1] - x[t]) / Var(x)
    
    High value = rough (large jumps relative to overall variance)
    Low value = smooth (small jumps relative to overall variance)
    
    Args:
        x: 1D array of values (should be temporally ordered)
    
    Returns:
        Normalized first-difference variance
    """
    x = np.asarray(x).flatten()
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    
    var_x = np.var(x)
    if var_x < 1e-10:
        return np.nan
    
    diff = np.diff(x)
    var_diff = np.var(diff)
    
    return var_diff / var_x


def compute_all_metrics(x: np.ndarray) -> Dict[str, float]:
    """Compute all signal characterization metrics."""
    return {
        'amplitude_iqr': compute_amplitude(x),
        'roughness_autocorr': compute_roughness_autocorr(x),
        'roughness_diffvar': compute_roughness_diffvar(x),
        'std': np.nanstd(x),
        'n_points': np.sum(np.isfinite(x)),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def ensure_celsius(temps: np.ndarray, threshold: float = 100.0) -> np.ndarray:
    """Convert to Celsius if values appear to be in Kelvin."""
    temps = np.asarray(temps, dtype=float)
    if len(temps) == 0:
        return temps
    if np.nanmean(temps) > threshold:
        return temps - 273.15
    return temps


def find_buoy_info_for_lake(lake_id_cci: int, selection_csvs: List[str]) -> Optional[Tuple[int, int]]:
    """
    Find lake_id and site_id for a given lake_id_cci from selection CSVs.
    
    Selection CSVs have columns: lake_id_cci, lake_id, site_id, latitude, longitude, etc.
    
    Returns (lake_id, site_id) tuple or None if not found.
    """
    for csv_path in selection_csvs:
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            if 'lake_id_cci' not in df.columns:
                continue
            
            lake_rows = df[df['lake_id_cci'] == lake_id_cci]
            if lake_rows.empty:
                continue
            
            # Get first site for this lake
            row = lake_rows.iloc[0]
            lake_id = int(row['lake_id'])
            site_id = int(row['site_id'])
            return (lake_id, site_id)
        except Exception:
            continue
    
    return None


def load_buoy_data(buoy_dir: str, lake_id: int, site_id: int) -> Optional[Dict]:
    """
    Load buoy data for a lake.
    
    Buoy files are named: ID{lake_id:06d}{site_id:02d}.csv
    
    Returns dict with 'dates' and 'temps' arrays, or None if not found.
    """
    # Construct buoy filename
    buoy_filename = f"ID{lake_id:06d}{site_id:02d}.csv"
    buoy_path = os.path.join(buoy_dir, buoy_filename)
    
    if not os.path.exists(buoy_path):
        return None
    
    try:
        df = pd.read_csv(buoy_path, parse_dates=['dateTime'])
        df = df.dropna(subset=['dateTime', 'insituTemp'])
        
        # Filter by QC flag if available (keep only good quality: 0)
        if 'qcFlag' in df.columns:
            df = df[df['qcFlag'] == 0]
        elif 'q' in df.columns:
            df = df[df['q'] == 0]
        
        if df.empty:
            return None
        
        return {
            'dates': df['dateTime'].values,
            'temps': ensure_celsius(df['insituTemp'].values),
        }
    except Exception:
        return None


def load_netcdf_timeseries(nc_path: str, var_name: str) -> Optional[Dict]:
    """
    Load lake-averaged time series from a NetCDF file.
    
    Args:
        nc_path: Path to NetCDF file
        var_name: Variable name to extract ('temp_filled' or 'lake_surface_water_temperature')
    
    Returns dict with 'dates' and 'temps' arrays.
    """
    if not os.path.exists(nc_path):
        return None
    
    try:
        ds = xr.open_dataset(nc_path)
        
        if var_name not in ds:
            ds.close()
            return None
        
        temps = ds[var_name].values
        times = pd.to_datetime(ds['time'].values)
        
        # Get lake mask from lakeid variable
        if 'lakeid' in ds:
            lakeid = ds['lakeid'].values
            if np.nanmax(lakeid) == 1:
                mask = lakeid == 1
            else:
                mask = np.isfinite(lakeid) & (lakeid != 0)
        else:
            # Fallback: use any valid data in first timestep
            mask = np.isfinite(temps[0])
        
        ds.close()
        
        # Average over lake pixels per timestep
        lake_temps = []
        for t in range(temps.shape[0]):
            frame = temps[t]
            lake_vals = frame[mask]
            lake_vals = lake_vals[np.isfinite(lake_vals)]
            if len(lake_vals) > 0:
                lake_temps.append(np.mean(lake_vals))
            else:
                lake_temps.append(np.nan)
        
        return {
            'dates': times,
            'temps': ensure_celsius(np.array(lake_temps)),
        }
    except Exception as e:
        return None


def load_reconstruction_data(post_dir: str, method: str) -> Optional[Dict]:
    """
    Load reconstruction (temp_filled) for a method.
    
    Args:
        post_dir: Path to post/{lake_id}/{alpha}/
        method: 'dineof' or 'dincae'
    
    Returns dict with 'dates' and 'temps' arrays.
    """
    pattern = os.path.join(post_dir, f"*_{method}.nc")
    files = glob(pattern)
    
    if not files:
        return None
    
    return load_netcdf_timeseries(files[0], 'temp_filled')


def load_satellite_observation_data(post_dir: str, method: str = 'dineof') -> Optional[Dict]:
    """
    Load original satellite observations (lake_surface_water_temperature) from post output.
    
    This variable is copied from original lake file to the post-processed output.
    We use the dineof output file by default since it should be present.
    
    Returns lake-averaged time series of observed (non-gap-filled) temperatures.
    """
    pattern = os.path.join(post_dir, f"*_{method}.nc")
    files = glob(pattern)
    
    if not files:
        return None
    
    return load_netcdf_timeseries(files[0], 'lake_surface_water_temperature')


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_lake(
    run_root: str,
    lake_id_cci: int,
    alpha: str,
    buoy_dir: str,
    selection_csvs: List[str],
    verbose: bool = False
) -> Optional[Dict]:
    """
    Analyze amplitude and roughness for a single lake.
    
    Returns dict with metrics for insitu, satellite, dineof, dincae.
    """
    # Paths
    lake_str = f"{lake_id_cci:09d}"
    post_dir = os.path.join(run_root, "post", lake_str, alpha)
    
    if not os.path.exists(post_dir):
        # Try unpadded
        post_dir = os.path.join(run_root, "post", str(lake_id_cci), alpha)
    
    if not os.path.exists(post_dir):
        if verbose:
            print(f"  Lake {lake_id_cci}: post directory not found")
        return None
    
    # Find buoy info (lake_id, site_id) for this lake_id_cci
    buoy_info = find_buoy_info_for_lake(lake_id_cci, selection_csvs)
    if buoy_info is None:
        if verbose:
            print(f"  Lake {lake_id_cci}: not found in selection CSVs")
        return None
    
    lake_id, site_id = buoy_info
    
    # Load buoy data
    buoy_data = load_buoy_data(buoy_dir, lake_id, site_id)
    if buoy_data is None:
        if verbose:
            print(f"  Lake {lake_id_cci}: no buoy data (lake_id={lake_id}, site_id={site_id})")
        return None
    
    # Load satellite observations from post output
    satellite_data = load_satellite_observation_data(post_dir, 'dineof')
    
    # Load reconstructions
    dineof_data = load_reconstruction_data(post_dir, 'dineof')
    dincae_data = load_reconstruction_data(post_dir, 'dincae')
    
    if dineof_data is None or dincae_data is None:
        if verbose:
            print(f"  Lake {lake_id_cci}: missing reconstruction data")
        return None
    
    # Compute metrics for each data source
    result = {
        'lake_id': lake_id_cci,
        'insitu': compute_all_metrics(buoy_data['temps']),
        'dineof': compute_all_metrics(dineof_data['temps']),
        'dincae': compute_all_metrics(dincae_data['temps']),
    }
    
    if satellite_data is not None:
        result['satellite'] = compute_all_metrics(satellite_data['temps'])
    
    if verbose:
        print(f"  Lake {lake_id_cci}: insitu IQR={result['insitu']['amplitude_iqr']:.2f}, "
              f"DINEOF IQR={result['dineof']['amplitude_iqr']:.2f}, "
              f"DINCAE IQR={result['dincae']['amplitude_iqr']:.2f}")
    
    return result


def load_winner_from_validation(run_root: str, alpha: str) -> Dict[int, str]:
    """
    Load which method won (lower RMSE) per lake from validation CSVs.
    
    Returns dict: lake_id -> 'dineof' or 'dincae'
    """
    winners = {}
    
    post_dir = os.path.join(run_root, "post")
    if not os.path.exists(post_dir):
        return winners
    
    for lake_folder in os.listdir(post_dir):
        lake_path = os.path.join(post_dir, lake_folder, alpha)
        if not os.path.isdir(lake_path):
            continue
        
        val_dir = os.path.join(lake_path, "insitu_cv_validation")
        if not os.path.exists(val_dir):
            continue
        
        # Find stats CSV
        csv_files = glob(os.path.join(val_dir, "*_insitu_stats_*.csv"))
        if not csv_files:
            continue
        
        try:
            df = pd.read_csv(csv_files[0])
            
            # Get reconstruction RMSE for each method
            recon_df = df[df['data_type'] == 'reconstruction']
            
            dineof_rmse = recon_df[recon_df['method'] == 'dineof']['rmse'].values
            dincae_rmse = recon_df[recon_df['method'] == 'dincae']['rmse'].values
            
            if len(dineof_rmse) > 0 and len(dincae_rmse) > 0:
                lake_id = int(lake_folder.lstrip('0') or '0')
                if dineof_rmse[0] < dincae_rmse[0]:
                    winners[lake_id] = 'dineof'
                else:
                    winners[lake_id] = 'dincae'
        except Exception:
            continue
    
    return winners


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_test1_plot(results_df: pd.DataFrame, output_dir: str):
    """
    Test 1: Characterize ground truths - is insitu lower amplitude? smoother?
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # A) Amplitude comparison: insitu vs satellite
    ax = axes[0]
    mask = results_df['satellite_amplitude_iqr'].notna()
    if mask.sum() > 2:
        x = results_df.loc[mask, 'satellite_amplitude_iqr']
        y = results_df.loc[mask, 'insitu_amplitude_iqr']
        
        ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # 1:1 line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
        
        # Stats
        below_line = (y < x).sum()
        total = len(x)
        
        ax.set_xlabel('Satellite Amplitude (IQR) [°C]')
        ax.set_ylabel('In-situ Amplitude (IQR) [°C]')
        ax.set_title(f'A) Amplitude: In-situ vs Satellite\n'
                     f'In-situ lower: {below_line}/{total} ({100*below_line/total:.0f}%)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('A) Amplitude: In-situ vs Satellite')
    
    # B) Roughness comparison: insitu vs satellite
    ax = axes[1]
    mask = results_df['satellite_roughness_autocorr'].notna()
    if mask.sum() > 2:
        x = results_df.loc[mask, 'satellite_roughness_autocorr']
        y = results_df.loc[mask, 'insitu_roughness_autocorr']
        
        ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
        
        below_line = (y < x).sum()
        total = len(x)
        
        ax.set_xlabel('Satellite Roughness (1-autocorr)')
        ax.set_ylabel('In-situ Roughness (1-autocorr)')
        ax.set_title(f'B) Roughness: In-situ vs Satellite\n'
                     f'In-situ smoother: {below_line}/{total} ({100*below_line/total:.0f}%)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('B) Roughness: In-situ vs Satellite')
    
    # C) Summary bar chart
    ax = axes[2]
    
    # Compute systematic differences
    amp_diff = results_df['insitu_amplitude_iqr'] - results_df['satellite_amplitude_iqr']
    rough_diff = results_df['insitu_roughness_autocorr'] - results_df['satellite_roughness_autocorr']
    
    categories = ['Amplitude\n(IQR)', 'Roughness\n(1-autocorr)']
    means = [np.nanmean(amp_diff), np.nanmean(rough_diff)]
    sems = [np.nanstd(amp_diff) / np.sqrt(np.sum(np.isfinite(amp_diff))),
            np.nanstd(rough_diff) / np.sqrt(np.sum(np.isfinite(rough_diff)))]
    
    colors = ['green' if m < 0 else 'red' for m in means]
    bars = ax.bar(categories, means, yerr=sems, capsize=5, color=colors, alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('In-situ - Satellite')
    ax.set_title('C) Systematic Difference\n(Negative = In-situ lower)')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'test1_ground_truth_characterization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_test2_plot(results_df: pd.DataFrame, output_dir: str):
    """
    Test 2: Characterize methods - is DINEOF lower amplitude? smoother?
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Color by winner
    colors = results_df['winner'].map({'dineof': '#5DA5DA', 'dincae': '#FAA43A'}).fillna('gray')
    
    # A) Amplitude: DINEOF vs DINCAE
    ax = axes[0]
    x = results_df['dincae_amplitude_iqr']
    y = results_df['dineof_amplitude_iqr']
    
    ax.scatter(x, y, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    dineof_lower = (y < x).sum()
    total = len(x)
    
    ax.set_xlabel('DINCAE Amplitude (IQR) [°C]')
    ax.set_ylabel('DINEOF Amplitude (IQR) [°C]')
    ax.set_title(f'A) Amplitude: DINEOF vs DINCAE\n'
                 f'DINEOF lower: {dineof_lower}/{total} ({100*dineof_lower/total:.0f}%)')
    
    # B) Roughness: DINEOF vs DINCAE
    ax = axes[1]
    x = results_df['dincae_roughness_autocorr']
    y = results_df['dineof_roughness_autocorr']
    
    ax.scatter(x, y, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    dineof_smoother = (y < x).sum()
    
    ax.set_xlabel('DINCAE Roughness (1-autocorr)')
    ax.set_ylabel('DINEOF Roughness (1-autocorr)')
    ax.set_title(f'B) Roughness: DINEOF vs DINCAE\n'
                 f'DINEOF smoother: {dineof_smoother}/{total} ({100*dineof_smoother/total:.0f}%)')
    
    # C) Winner breakdown
    ax = axes[2]
    
    # When DINEOF is smoother, who wins?
    dineof_smoother_mask = results_df['dineof_roughness_autocorr'] < results_df['dincae_roughness_autocorr']
    
    dineof_smoother_dineof_wins = ((dineof_smoother_mask) & (results_df['winner'] == 'dineof')).sum()
    dineof_smoother_dincae_wins = ((dineof_smoother_mask) & (results_df['winner'] == 'dincae')).sum()
    dincae_smoother_dineof_wins = ((~dineof_smoother_mask) & (results_df['winner'] == 'dineof')).sum()
    dincae_smoother_dincae_wins = ((~dineof_smoother_mask) & (results_df['winner'] == 'dincae')).sum()
    
    categories = ['DINEOF\nsmoother', 'DINCAE\nsmoother']
    dineof_wins = [dineof_smoother_dineof_wins, dincae_smoother_dineof_wins]
    dincae_wins = [dineof_smoother_dincae_wins, dincae_smoother_dincae_wins]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x_pos - width/2, dineof_wins, width, label='DINEOF wins', color='#5DA5DA')
    ax.bar(x_pos + width/2, dincae_wins, width, label='DINCAE wins', color='#FAA43A')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Number of lakes')
    ax.set_title('C) Does smoother method win?')
    ax.legend()
    
    # Add counts on bars
    for i, (d, c) in enumerate(zip(dineof_wins, dincae_wins)):
        ax.text(i - width/2, d + 0.3, str(d), ha='center', fontsize=9)
        ax.text(i + width/2, c + 0.3, str(c), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'test2_method_characterization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_test3_plot(results_df: pd.DataFrame, output_dir: str):
    """
    Test 3: What predicts winning - amplitude match or roughness match?
    """
    # Compute matching metrics (smaller = better match with insitu)
    results_df = results_df.copy()
    results_df['dineof_amp_match'] = np.abs(results_df['dineof_amplitude_iqr'] - results_df['insitu_amplitude_iqr'])
    results_df['dincae_amp_match'] = np.abs(results_df['dincae_amplitude_iqr'] - results_df['insitu_amplitude_iqr'])
    results_df['dineof_rough_match'] = np.abs(results_df['dineof_roughness_autocorr'] - results_df['insitu_roughness_autocorr'])
    results_df['dincae_rough_match'] = np.abs(results_df['dincae_roughness_autocorr'] - results_df['insitu_roughness_autocorr'])
    
    # Delta: positive = DINEOF worse match
    results_df['delta_amp_match'] = results_df['dineof_amp_match'] - results_df['dincae_amp_match']
    results_df['delta_rough_match'] = results_df['dineof_rough_match'] - results_df['dincae_rough_match']
    
    # Also load RMSE delta from validation if available
    # For now, use winner as proxy
    results_df['winner_numeric'] = results_df['winner'].map({'dineof': -1, 'dincae': 1})
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    colors = results_df['winner'].map({'dineof': '#5DA5DA', 'dincae': '#FAA43A'}).fillna('gray')
    
    # A) Amplitude match predicts winner?
    ax = axes[0]
    x = results_df['delta_amp_match']
    y = results_df['winner_numeric']
    
    ax.scatter(x, y + np.random.uniform(-0.1, 0.1, len(y)), c=colors, alpha=0.7, 
               edgecolors='black', linewidth=0.5)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Count quadrants
    q1 = ((x > 0) & (y > 0)).sum()  # DINEOF worse match, DINCAE wins - consistent
    q2 = ((x < 0) & (y > 0)).sum()  # DINEOF better match, DINCAE wins - inconsistent
    q3 = ((x < 0) & (y < 0)).sum()  # DINEOF better match, DINEOF wins - consistent
    q4 = ((x > 0) & (y < 0)).sum()  # DINEOF worse match, DINEOF wins - inconsistent
    
    consistent = q1 + q3
    total = len(x)
    
    ax.set_xlabel('Δ(Amplitude match)\n+ = DINEOF worse match')
    ax.set_ylabel('Winner\n(-1=DINEOF, +1=DINCAE)')
    ax.set_title(f'A) Amplitude matching predicts winner?\n'
                 f'Consistent: {consistent}/{total} ({100*consistent/total:.0f}%)')
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['DINEOF', 'DINCAE'])
    
    # B) Roughness match predicts winner?
    ax = axes[1]
    x = results_df['delta_rough_match']
    y = results_df['winner_numeric']
    
    ax.scatter(x, y + np.random.uniform(-0.1, 0.1, len(y)), c=colors, alpha=0.7,
               edgecolors='black', linewidth=0.5)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    q1 = ((x > 0) & (y > 0)).sum()
    q3 = ((x < 0) & (y < 0)).sum()
    consistent = q1 + q3
    
    ax.set_xlabel('Δ(Roughness match)\n+ = DINEOF worse match')
    ax.set_ylabel('Winner\n(-1=DINEOF, +1=DINCAE)')
    ax.set_title(f'B) Roughness matching predicts winner?\n'
                 f'Consistent: {consistent}/{total} ({100*consistent/total:.0f}%)')
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['DINEOF', 'DINCAE'])
    
    # C) Summary: which property matters more?
    ax = axes[2]
    
    # Count how often each property correctly predicts winner
    amp_correct = ((results_df['delta_amp_match'] > 0) & (results_df['winner'] == 'dincae') |
                   (results_df['delta_amp_match'] < 0) & (results_df['winner'] == 'dineof')).sum()
    rough_correct = ((results_df['delta_rough_match'] > 0) & (results_df['winner'] == 'dincae') |
                     (results_df['delta_rough_match'] < 0) & (results_df['winner'] == 'dineof')).sum()
    
    categories = ['Amplitude\nmatching', 'Roughness\nmatching']
    accuracy = [100 * amp_correct / total, 100 * rough_correct / total]
    
    bars = ax.bar(categories, accuracy, color=['#2ecc71', '#3498db'], alpha=0.7)
    ax.axhline(50, color='red', linestyle='--', label='Random chance (50%)')
    
    ax.set_ylabel('Prediction accuracy (%)')
    ax.set_title('C) Which property predicts winner?')
    ax.set_ylim(0, 100)
    ax.legend()
    
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 2, f'{acc:.0f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'test3_what_predicts_winner.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_summary_report(results_df: pd.DataFrame, output_dir: str):
    """Create text summary of findings with explicit limitations."""
    
    report = []
    report.append("=" * 70)
    report.append("AMPLITUDE vs ROUGHNESS ANALYSIS - SUMMARY REPORT")
    report.append("=" * 70)
    report.append(f"\nTotal lakes analyzed: {len(results_df)}")
    report.append(f"Lakes with satellite data: {results_df['satellite_amplitude_iqr'].notna().sum()}")
    
    # Test 1 results
    report.append("\n" + "-" * 70)
    report.append("TEST 1: Ground Truth Characterization (In-situ vs Satellite)")
    report.append("-" * 70)
    
    mask = results_df['satellite_amplitude_iqr'].notna()
    if mask.sum() > 0:
        amp_insitu_lower = (results_df.loc[mask, 'insitu_amplitude_iqr'] < 
                           results_df.loc[mask, 'satellite_amplitude_iqr']).sum()
        rough_insitu_lower = (results_df.loc[mask, 'insitu_roughness_autocorr'] < 
                             results_df.loc[mask, 'satellite_roughness_autocorr']).sum()
        n = mask.sum()
        
        report.append(f"  In-situ has LOWER amplitude: {amp_insitu_lower}/{n} ({100*amp_insitu_lower/n:.0f}%)")
        report.append(f"  In-situ is SMOOTHER:         {rough_insitu_lower}/{n} ({100*rough_insitu_lower/n:.0f}%)")
        
        if amp_insitu_lower / n > 0.6 and rough_insitu_lower / n > 0.6:
            report.append("  --> CONSISTENT with damped signal hypothesis")
        else:
            report.append("  --> NOT consistently supporting damped signal hypothesis")
    
    # Test 2 results
    report.append("\n" + "-" * 70)
    report.append("TEST 2: Method Characterization (DINEOF vs DINCAE)")
    report.append("-" * 70)
    
    n = len(results_df)
    dineof_lower_amp = (results_df['dineof_amplitude_iqr'] < results_df['dincae_amplitude_iqr']).sum()
    dineof_smoother = (results_df['dineof_roughness_autocorr'] < results_df['dincae_roughness_autocorr']).sum()
    
    report.append(f"  DINEOF has LOWER amplitude: {dineof_lower_amp}/{n} ({100*dineof_lower_amp/n:.0f}%)")
    report.append(f"  DINEOF is SMOOTHER:         {dineof_smoother}/{n} ({100*dineof_smoother/n:.0f}%)")
    
    # Test 3 results
    report.append("\n" + "-" * 70)
    report.append("TEST 3: What Predicts Winner?")
    report.append("-" * 70)
    
    # Compute matching
    dineof_amp_match = np.abs(results_df['dineof_amplitude_iqr'] - results_df['insitu_amplitude_iqr'])
    dincae_amp_match = np.abs(results_df['dincae_amplitude_iqr'] - results_df['insitu_amplitude_iqr'])
    dineof_rough_match = np.abs(results_df['dineof_roughness_autocorr'] - results_df['insitu_roughness_autocorr'])
    dincae_rough_match = np.abs(results_df['dincae_roughness_autocorr'] - results_df['insitu_roughness_autocorr'])
    
    amp_correct = ((dineof_amp_match < dincae_amp_match) & (results_df['winner'] == 'dineof') |
                   (dineof_amp_match > dincae_amp_match) & (results_df['winner'] == 'dincae')).sum()
    rough_correct = ((dineof_rough_match < dincae_rough_match) & (results_df['winner'] == 'dineof') |
                     (dineof_rough_match > dincae_rough_match) & (results_df['winner'] == 'dincae')).sum()
    
    report.append(f"  Amplitude matching predicts winner: {amp_correct}/{n} ({100*amp_correct/n:.0f}%)")
    report.append(f"  Roughness matching predicts winner: {rough_correct}/{n} ({100*rough_correct/n:.0f}%)")
    
    # Winner breakdown
    dineof_wins = (results_df['winner'] == 'dineof').sum()
    dincae_wins = (results_df['winner'] == 'dincae').sum()
    report.append(f"\n  Overall winner counts: DINEOF {dineof_wins}, DINCAE {dincae_wins}")
    
    # Limitations
    report.append("\n" + "-" * 70)
    report.append("LIMITATIONS (must acknowledge in any reporting)")
    report.append("-" * 70)
    report.append("  1. These tests show CONSISTENCY with hypotheses, NOT proof")
    report.append("  2. Correlation does not imply causation")
    report.append("  3. Alternative explanations exist:")
    report.append("     - Sensor differences (not necessarily bulk vs skin)")
    report.append("     - Spatial averaging effects")
    report.append("     - Temporal sampling differences")
    report.append("  4. Sample size may be limited")
    report.append("  5. Lake-averaged metrics may mask spatial heterogeneity")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    
    out_path = os.path.join(output_dir, 'amplitude_roughness_summary.txt')
    with open(out_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze amplitude vs roughness for bulk effect hypothesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests whether CV validation discrepancies can be explained by 
differences in signal amplitude and roughness between in-situ and satellite.

Outputs:
  - test1_ground_truth_characterization.png: Is in-situ damped?
  - test2_method_characterization.png: Is DINEOF smoother?
  - test3_what_predicts_winner.png: Amplitude or roughness matching?
  - amplitude_roughness_summary.txt: Summary with explicit limitations

Example:
    python analyze_amplitude_roughness.py --run-root /path/to/experiment
        """
    )
    
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--buoy-dir", default=None, help="Buoy data directory")
    parser.add_argument("--selection-csvs", nargs="+", default=None, help="Selection CSV files")
    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Process all lakes (not just those with both methods)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Defaults
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "amplitude_roughness_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.buoy_dir is None:
        args.buoy_dir = "/gws/ssde/j25b/nceo_uor/users/lcarrea01/INSITU/Buoy_Laura/ALL_FILES_QC"
    
    if args.selection_csvs is None:
        _csv_dir = "/home/users/shaerdan/general_purposes/insitu_cv"
        args.selection_csvs = [
            f"{_csv_dir}/L3S_QL_MDB_2010_selection.csv",
            f"{_csv_dir}/L3S_QL_MDB_2007_selection.csv",
            f"{_csv_dir}/L3S_QL_MDB_2018_selection.csv",
            f"{_csv_dir}/L3S_QL_MDB_2020_selection.csv",
        ]
    
    print("=" * 70)
    print("AMPLITUDE vs ROUGHNESS ANALYSIS")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Alpha: {args.alpha}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Get lake list
    if HAS_COMPLETION_CHECK and not args.no_fair_comparison:
        lake_ids, _ = get_fair_comparison_lakes(args.run_root, args.alpha, verbose=True)
    else:
        # Discover from post directory
        post_dir = os.path.join(args.run_root, "post")
        lake_ids = []
        if os.path.exists(post_dir):
            for folder in os.listdir(post_dir):
                try:
                    lake_id = int(folder.lstrip('0') or '0')
                    if lake_id > 0:
                        lake_ids.append(lake_id)
                except ValueError:
                    continue
        lake_ids = sorted(lake_ids)
    
    print(f"\nAnalyzing {len(lake_ids)} lakes...")
    
    # Load winners from validation
    winners = load_winner_from_validation(args.run_root, args.alpha)
    print(f"Found winner data for {len(winners)} lakes")
    
    # Analyze each lake
    results = []
    for lake_id in lake_ids:
        result = analyze_lake(
            args.run_root, lake_id, args.alpha,
            args.buoy_dir, args.selection_csvs,
            verbose=args.verbose
        )
        if result is not None:
            result['winner'] = winners.get(lake_id, None)
            results.append(result)
    
    print(f"\nSuccessfully analyzed {len(results)} lakes")
    
    if len(results) < 3:
        print("ERROR: Not enough lakes with complete data for analysis")
        return 1
    
    # Build results DataFrame
    rows = []
    for r in results:
        row = {'lake_id': r['lake_id'], 'winner': r['winner']}
        
        for source in ['insitu', 'satellite', 'dineof', 'dincae']:
            if source in r:
                for metric, value in r[source].items():
                    row[f'{source}_{metric}'] = value
        
        rows.append(row)
    
    results_df = pd.DataFrame(rows)
    
    # Filter to lakes with winner data
    results_df = results_df[results_df['winner'].notna()]
    print(f"Lakes with winner data: {len(results_df)}")
    
    if len(results_df) < 3:
        print("ERROR: Not enough lakes with winner data")
        return 1
    
    # Save raw results
    csv_path = os.path.join(args.output_dir, 'amplitude_roughness_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Create plots
    print("\nGenerating plots...")
    create_test1_plot(results_df, args.output_dir)
    create_test2_plot(results_df, args.output_dir)
    create_test3_plot(results_df, args.output_dir)
    
    # Create summary report
    print("\n")
    create_summary_report(results_df, args.output_dir)
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())