#!/usr/bin/env python3
"""
Peak Matching vs Roughness Analysis for Bulk Effect Hypothesis
===============================================================

This script tests whether the CV validation discrepancy (DINEOF wins satellite CV,
DINCAE catches up in in-situ CV) can be explained by differences in signal characteristics.

Hypothesis:
- In-situ buoys may measure a "damped" signal (lower variability, smoother) compared 
  to satellite skin temperature
- If a method produces reconstructions that underestimate peaks/troughs, it would
  match the damped in-situ signal better

Key metrics:
- PEAK MATCHING: Error at extreme times (when insitu is at peaks or troughs)
  - Defined by insitu > P75 (highs) or insitu < P25 (lows)
  - Lower MAE at extremes = better peak matching
- ROUGHNESS: High-frequency variability - measured by 1 - lag-1 autocorrelation
  - Higher value = rougher (more high-frequency noise)

Tests:
1. Characterize ground truths: Is in-situ lower variability? Smoother? 
2. Characterize methods: Is DINEOF worse at peaks? Smoother? Does it underestimate?
3. What predicts winning: better peak matching, or roughness matching?

Limitations (explicitly stated):
- These tests show CONSISTENCY with hypotheses, not proof
- Correlation ≠ causation  
- Alternative explanations may exist
- Roughness metric approximate for irregularly sampled data

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


# =============================================================================
# METRICS: Peak Matching and Roughness
# =============================================================================

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


def compute_extreme_matching_error(method_vals: np.ndarray, insitu_vals: np.ndarray, 
                                    high_mask: np.ndarray, low_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute how well a method matches insitu at extreme times (peaks and troughs).
    
    Extreme times are defined by insitu values:
    - High extremes: insitu > P75(insitu)
    - Low extremes: insitu < P25(insitu)
    
    Args:
        method_vals: Method reconstruction values (matched timestamps)
        insitu_vals: Insitu values (matched timestamps)
        high_mask: Boolean mask for high extreme times (insitu > P75)
        low_mask: Boolean mask for low extreme times (insitu < P25)
    
    Returns:
        Dict with:
        - extreme_mae: MAE at all extreme times (lower = better matching)
        - high_error: mean(method - insitu) at high times (negative = underestimates peaks)
        - low_error: mean(method - insitu) at low times (positive = overestimates troughs)
        - n_extremes: number of extreme points
    """
    method_vals = np.asarray(method_vals).flatten()
    insitu_vals = np.asarray(insitu_vals).flatten()
    
    # Combine masks for all extremes
    extreme_mask = high_mask | low_mask
    n_extremes = extreme_mask.sum()
    
    if n_extremes < 3:
        return {
            'extreme_mae': np.nan,
            'high_error': np.nan,
            'low_error': np.nan,
            'n_extremes': n_extremes,
        }
    
    # Error at all extremes
    extreme_errors = method_vals[extreme_mask] - insitu_vals[extreme_mask]
    extreme_mae = np.mean(np.abs(extreme_errors))
    
    # Separate high and low errors
    high_error = np.nan
    low_error = np.nan
    
    if high_mask.sum() > 0:
        high_error = np.mean(method_vals[high_mask] - insitu_vals[high_mask])
    
    if low_mask.sum() > 0:
        low_error = np.mean(method_vals[low_mask] - insitu_vals[low_mask])
    
    return {
        'extreme_mae': extreme_mae,
        'high_error': high_error,
        'low_error': low_error,
        'n_extremes': n_extremes,
    }


def compute_signal_metrics(x: np.ndarray) -> Dict[str, float]:
    """Compute signal characterization metrics (roughness, std)."""
    x = np.asarray(x).flatten()
    x_valid = x[np.isfinite(x)]
    
    return {
        'roughness_autocorr': compute_roughness_autocorr(x),
        'roughness_diffvar': compute_roughness_diffvar(x),
        'std': np.nanstd(x_valid) if len(x_valid) > 0 else np.nan,
        'n_points': len(x_valid),
    }


# =============================================================================
# DATA LOADING - From timeseries CSVs produced by insitu_validation.py
# =============================================================================

def load_timeseries_csv(run_root: str, lake_id_cci: int, alpha: str) -> Optional[pd.DataFrame]:
    """
    Load matched timeseries CSV produced by insitu_validation.py.
    
    File location: post/{lake_id}/{alpha}/insitu_cv_validation/LAKE{id}_insitu_timeseries_site*.csv
    
    Returns DataFrame with columns:
    - date, insitu_temp, satellite_obs_temp, dineof_recon_temp, dincae_recon_temp, was_observed
    """
    lake_str = f"{lake_id_cci:09d}"
    val_dir = os.path.join(run_root, "post", lake_str, alpha, "insitu_cv_validation")
    
    if not os.path.exists(val_dir):
        # Try unpadded
        val_dir = os.path.join(run_root, "post", str(lake_id_cci), alpha, "insitu_cv_validation")
    
    if not os.path.exists(val_dir):
        return None
    
    # Find timeseries CSV
    pattern = os.path.join(val_dir, f"LAKE{lake_str}_insitu_timeseries_site*.csv")
    files = glob(pattern)
    
    if not files:
        # Try unpadded pattern
        pattern = os.path.join(val_dir, f"LAKE*_insitu_timeseries_site*.csv")
        files = glob(pattern)
    
    if not files:
        return None
    
    try:
        df = pd.read_csv(files[0], parse_dates=['date'])
        return df
    except Exception:
        return None


def ensure_celsius(temps: np.ndarray, threshold: float = 100.0) -> np.ndarray:
    """Convert to Celsius if values appear to be in Kelvin."""
    temps = np.asarray(temps, dtype=float)
    if len(temps) == 0:
        return temps
    if np.nanmean(temps) > threshold:
        return temps - 273.15
    return temps


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_lake(
    run_root: str,
    lake_id_cci: int,
    alpha: str,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Analyze extreme matching and roughness for a single lake.
    
    Extreme matching: How well does each method capture peaks (highs) and troughs (lows)?
    - Defined by insitu values: high = insitu > P75, low = insitu < P25
    - Metric: MAE at extreme times (lower = better peak matching)
    
    Roughness: How smooth/rough is each signal?
    - Metric: 1 - lag1_autocorrelation (higher = rougher)
    
    IMPORTANT: All metrics computed on matched timestamps only.
    
    Returns dict with extreme matching and roughness metrics.
    """
    # Load timeseries CSV
    df = load_timeseries_csv(run_root, lake_id_cci, alpha)
    
    if df is None:
        if verbose:
            print(f"  Lake {lake_id_cci}: no timeseries CSV found")
        return None
    
    # Check we have the required columns
    required_cols = ['insitu_temp', 'dineof_recon_temp', 'dincae_recon_temp']
    for col in required_cols:
        if col not in df.columns:
            if verbose:
                print(f"  Lake {lake_id_cci}: missing column {col}")
            return None
    
    # ==========================================================================
    # Filter to matched timestamps (all three have valid values)
    # ==========================================================================
    mask_recon = (
        df['insitu_temp'].notna() & 
        df['dineof_recon_temp'].notna() & 
        df['dincae_recon_temp'].notna()
    )
    df_matched = df[mask_recon].copy()
    
    if len(df_matched) < 10:
        if verbose:
            print(f"  Lake {lake_id_cci}: too few matched points ({len(df_matched)})")
        return None
    
    # Extract matched arrays
    insitu = df_matched['insitu_temp'].values
    dineof = df_matched['dineof_recon_temp'].values
    dincae = df_matched['dincae_recon_temp'].values
    
    # ==========================================================================
    # Define extreme times based on INSITU (the reference)
    # ==========================================================================
    p25 = np.percentile(insitu, 25)
    p75 = np.percentile(insitu, 75)
    
    high_mask = insitu > p75  # Peak highs (summer)
    low_mask = insitu < p25   # Peak lows (winter)
    
    n_highs = high_mask.sum()
    n_lows = low_mask.sum()
    
    if n_highs < 2 or n_lows < 2:
        if verbose:
            print(f"  Lake {lake_id_cci}: too few extremes (highs={n_highs}, lows={n_lows})")
        return None
    
    # ==========================================================================
    # Compute extreme matching for each method
    # ==========================================================================
    dineof_extreme = compute_extreme_matching_error(dineof, insitu, high_mask, low_mask)
    dincae_extreme = compute_extreme_matching_error(dincae, insitu, high_mask, low_mask)
    
    # ==========================================================================
    # Compute roughness for each series
    # ==========================================================================
    insitu_metrics = compute_signal_metrics(insitu)
    dineof_metrics = compute_signal_metrics(dineof)
    dincae_metrics = compute_signal_metrics(dincae)
    
    # ==========================================================================
    # Build result
    # ==========================================================================
    result = {
        'lake_id': lake_id_cci,
        'n_matched': len(df_matched),
        'n_extremes': n_highs + n_lows,
        'n_highs': n_highs,
        'n_lows': n_lows,
        'insitu_p25': p25,
        'insitu_p75': p75,
        
        # Extreme matching (lower MAE = better peak matching)
        'dineof_extreme_mae': dineof_extreme['extreme_mae'],
        'dincae_extreme_mae': dincae_extreme['extreme_mae'],
        'dineof_high_error': dineof_extreme['high_error'],
        'dincae_high_error': dincae_extreme['high_error'],
        'dineof_low_error': dineof_extreme['low_error'],
        'dincae_low_error': dincae_extreme['low_error'],
        
        # Roughness (higher = rougher)
        'insitu_roughness': insitu_metrics['roughness_autocorr'],
        'dineof_roughness': dineof_metrics['roughness_autocorr'],
        'dincae_roughness': dincae_metrics['roughness_autocorr'],
        
        # Additional metrics
        'insitu_std': insitu_metrics['std'],
        'dineof_std': dineof_metrics['std'],
        'dincae_std': dincae_metrics['std'],
    }
    
    # Add satellite metrics if available
    if 'satellite_obs_temp' in df_matched.columns:
        sat_mask = df_matched['satellite_obs_temp'].notna()
        if sat_mask.sum() >= 10:
            satellite = df_matched.loc[sat_mask, 'satellite_obs_temp'].values
            insitu_sat = df_matched.loc[sat_mask, 'insitu_temp'].values
            
            sat_metrics = compute_signal_metrics(satellite)
            insitu_sat_metrics = compute_signal_metrics(insitu_sat)
            
            result['n_satellite'] = sat_mask.sum()
            result['satellite_roughness'] = sat_metrics['roughness_autocorr']
            result['satellite_std'] = sat_metrics['std']
            result['insitu_sat_roughness'] = insitu_sat_metrics['roughness_autocorr']
            result['insitu_sat_std'] = insitu_sat_metrics['std']
    
    if verbose:
        print(f"  Lake {lake_id_cci}: N={len(df_matched)}, extremes={n_highs}+{n_lows}, "
              f"DINEOF extreme_MAE={dineof_extreme['extreme_mae']:.3f}, "
              f"DINCAE extreme_MAE={dincae_extreme['extreme_mae']:.3f}")
    
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
    Test 1: Characterize ground truths - is insitu lower STD / smoother than satellite?
    
    This tests whether the bulk effect premise holds: buoys should show damped signals.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Use satellite-matched insitu for fair comparison
    mask = results_df['satellite_std'].notna() & results_df['insitu_sat_std'].notna()
    
    # A) STD comparison: insitu vs satellite
    ax = axes[0]
    if mask.sum() > 2:
        x = results_df.loc[mask, 'satellite_std']
        y = results_df.loc[mask, 'insitu_sat_std']
        
        ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # 1:1 line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
        
        below_line = (y < x).sum()
        total = len(x)
        
        ax.set_xlabel('Satellite STD [°C]')
        ax.set_ylabel('In-situ STD [°C]')
        ax.set_title(f'A) Variability: In-situ vs Satellite\n'
                     f'In-situ lower STD: {below_line}/{total} ({100*below_line/total:.0f}%)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('A) Variability: In-situ vs Satellite')
    
    # B) Roughness comparison: insitu vs satellite
    ax = axes[1]
    mask_rough = results_df['satellite_roughness'].notna() & results_df['insitu_sat_roughness'].notna()
    if mask_rough.sum() > 2:
        x = results_df.loc[mask_rough, 'satellite_roughness']
        y = results_df.loc[mask_rough, 'insitu_sat_roughness']
        
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
    
    std_diff = results_df['insitu_sat_std'] - results_df['satellite_std']
    rough_diff = results_df['insitu_sat_roughness'] - results_df['satellite_roughness']
    
    categories = ['STD\n(variability)', 'Roughness\n(1-autocorr)']
    means = [np.nanmean(std_diff), np.nanmean(rough_diff)]
    sems = [np.nanstd(std_diff) / np.sqrt(np.sum(np.isfinite(std_diff))) if np.sum(np.isfinite(std_diff)) > 0 else 0,
            np.nanstd(rough_diff) / np.sqrt(np.sum(np.isfinite(rough_diff))) if np.sum(np.isfinite(rough_diff)) > 0 else 0]
    
    colors = ['green' if m < 0 else 'red' for m in means]
    bars = ax.bar(categories, means, yerr=sems, capsize=5, color=colors, alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('In-situ - Satellite')
    ax.set_title('C) Systematic Difference\n(Negative = In-situ lower/smoother)')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'test1_ground_truth_characterization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_test2_plot(results_df: pd.DataFrame, output_dir: str):
    """
    Test 2: Characterize methods - extreme matching and roughness.
    
    - Is DINEOF worse at extreme matching (higher MAE at peaks/troughs)?
    - Is DINEOF smoother (lower roughness)?
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Color by winner
    colors = results_df['winner'].map({'dineof': '#5DA5DA', 'dincae': '#FAA43A'}).fillna('gray')
    
    # A) Extreme Matching: DINEOF vs DINCAE (lower = better)
    ax = axes[0]
    x = results_df['dincae_extreme_mae']
    y = results_df['dineof_extreme_mae']
    
    ax.scatter(x, y, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    dineof_worse = (y > x).sum()
    total = len(x)
    
    ax.set_xlabel('DINCAE Extreme MAE [°C]')
    ax.set_ylabel('DINEOF Extreme MAE [°C]')
    ax.set_title(f'A) Peak Matching (lower=better)\n'
                 f'DINEOF worse: {dineof_worse}/{total} ({100*dineof_worse/total:.0f}%)')
    
    # B) Roughness: DINEOF vs DINCAE
    ax = axes[1]
    x = results_df['dincae_roughness']
    y = results_df['dineof_roughness']
    
    ax.scatter(x, y, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    dineof_smoother = (y < x).sum()
    
    ax.set_xlabel('DINCAE Roughness (1-autocorr)')
    ax.set_ylabel('DINEOF Roughness (1-autocorr)')
    ax.set_title(f'B) Roughness (lower=smoother)\n'
                 f'DINEOF smoother: {dineof_smoother}/{total} ({100*dineof_smoother/total:.0f}%)')
    
    # C) Peak errors breakdown (high vs low)
    ax = axes[2]
    
    # DINEOF errors at highs and lows
    dineof_high = results_df['dineof_high_error'].mean()
    dineof_low = results_df['dineof_low_error'].mean()
    dincae_high = results_df['dincae_high_error'].mean()
    dincae_low = results_df['dincae_low_error'].mean()
    
    x_pos = np.arange(2)
    width = 0.35
    
    ax.bar(x_pos - width/2, [dineof_high, dineof_low], width, label='DINEOF', color='#5DA5DA', alpha=0.7)
    ax.bar(x_pos + width/2, [dincae_high, dincae_low], width, label='DINCAE', color='#FAA43A', alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['At Highs\n(summer peaks)', 'At Lows\n(winter troughs)'])
    ax.set_ylabel('Mean Error (method - insitu) [°C]')
    ax.set_title('C) Systematic Bias at Extremes\n(negative=underestimate, positive=overestimate)')
    ax.legend()
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'test2_method_characterization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_test3_plot(results_df: pd.DataFrame, output_dir: str):
    """
    Test 3: What predicts winning - better extreme matching or lower roughness?
    
    - Extreme matching: lower MAE at peaks/troughs = better
    - Roughness matching: closer roughness to insitu = better
    """
    results_df = results_df.copy()
    
    # Compute roughness matching (smaller = better match with insitu)
    results_df['dineof_rough_match'] = np.abs(results_df['dineof_roughness'] - results_df['insitu_roughness'])
    results_df['dincae_rough_match'] = np.abs(results_df['dincae_roughness'] - results_df['insitu_roughness'])
    
    # Delta: positive = DINEOF worse
    results_df['delta_extreme'] = results_df['dineof_extreme_mae'] - results_df['dincae_extreme_mae']
    results_df['delta_rough_match'] = results_df['dineof_rough_match'] - results_df['dincae_rough_match']
    
    results_df['winner_numeric'] = results_df['winner'].map({'dineof': -1, 'dincae': 1})
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    colors = results_df['winner'].map({'dineof': '#5DA5DA', 'dincae': '#FAA43A'}).fillna('gray')
    
    # A) Extreme matching predicts winner?
    ax = axes[0]
    x = results_df['delta_extreme']
    y = results_df['winner_numeric']
    total = len(x)
    
    ax.scatter(x, y + np.random.uniform(-0.1, 0.1, len(y)), c=colors, alpha=0.7, 
               edgecolors='black', linewidth=0.5)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Count quadrants
    q1 = ((x > 0) & (y > 0)).sum()  # DINEOF worse extreme, DINCAE wins - consistent
    q3 = ((x < 0) & (y < 0)).sum()  # DINEOF better extreme, DINEOF wins - consistent
    consistent = q1 + q3
    
    ax.set_xlabel('Δ(Extreme MAE)\n+ = DINEOF worse at peaks')
    ax.set_ylabel('Winner\n(-1=DINEOF, +1=DINCAE)')
    ax.set_title(f'A) Peak matching predicts winner?\n'
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
    extreme_correct = ((results_df['delta_extreme'] > 0) & (results_df['winner'] == 'dincae') |
                       (results_df['delta_extreme'] < 0) & (results_df['winner'] == 'dineof')).sum()
    rough_correct = ((results_df['delta_rough_match'] > 0) & (results_df['winner'] == 'dincae') |
                     (results_df['delta_rough_match'] < 0) & (results_df['winner'] == 'dineof')).sum()
    
    categories = ['Peak\nmatching', 'Roughness\nmatching']
    accuracy = [100 * extreme_correct / total, 100 * rough_correct / total]
    
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
    report.append("PEAK MATCHING vs ROUGHNESS ANALYSIS - SUMMARY REPORT")
    report.append("=" * 70)
    report.append(f"\nTotal lakes analyzed: {len(results_df)}")
    report.append(f"Lakes with satellite data: {results_df['satellite_std'].notna().sum()}")
    
    # Test 1 results
    report.append("\n" + "-" * 70)
    report.append("TEST 1: Ground Truth Characterization (In-situ vs Satellite)")
    report.append("-" * 70)
    report.append("  Note: Computed on timestamps where BOTH have valid values")
    
    mask = results_df['satellite_std'].notna() & results_df['insitu_sat_std'].notna()
    if mask.sum() > 0:
        std_insitu_lower = (results_df.loc[mask, 'insitu_sat_std'] < 
                           results_df.loc[mask, 'satellite_std']).sum()
        rough_insitu_lower = (results_df.loc[mask, 'insitu_sat_roughness'] < 
                             results_df.loc[mask, 'satellite_roughness']).sum()
        n = mask.sum()
        
        report.append(f"  In-situ has LOWER STD:    {std_insitu_lower}/{n} ({100*std_insitu_lower/n:.0f}%)")
        report.append(f"  In-situ is SMOOTHER:      {rough_insitu_lower}/{n} ({100*rough_insitu_lower/n:.0f}%)")
        
        if std_insitu_lower / n > 0.6 and rough_insitu_lower / n > 0.6:
            report.append("  --> CONSISTENT with damped signal hypothesis")
        else:
            report.append("  --> NOT consistently supporting damped signal hypothesis")
    else:
        report.append("  Insufficient data for Test 1")
    
    # Test 2 results
    report.append("\n" + "-" * 70)
    report.append("TEST 2: Method Characterization (DINEOF vs DINCAE)")
    report.append("-" * 70)
    report.append("  Note: Computed on matched timestamps")
    
    n = len(results_df)
    dineof_worse_extreme = (results_df['dineof_extreme_mae'] > results_df['dincae_extreme_mae']).sum()
    dineof_smoother = (results_df['dineof_roughness'] < results_df['dincae_roughness']).sum()
    
    report.append(f"  DINEOF worse at extremes:  {dineof_worse_extreme}/{n} ({100*dineof_worse_extreme/n:.0f}%)")
    report.append(f"  DINEOF is SMOOTHER:        {dineof_smoother}/{n} ({100*dineof_smoother/n:.0f}%)")
    
    # Mean errors at extremes
    dineof_high_mean = results_df['dineof_high_error'].mean()
    dineof_low_mean = results_df['dineof_low_error'].mean()
    dincae_high_mean = results_df['dincae_high_error'].mean()
    dincae_low_mean = results_df['dincae_low_error'].mean()
    
    report.append(f"\n  Mean error at highs: DINEOF={dineof_high_mean:+.2f}°C, DINCAE={dincae_high_mean:+.2f}°C")
    report.append(f"  Mean error at lows:  DINEOF={dineof_low_mean:+.2f}°C, DINCAE={dincae_low_mean:+.2f}°C")
    report.append("  (negative at highs = underestimates peaks, positive at lows = overestimates troughs)")
    
    # Test 3 results
    report.append("\n" + "-" * 70)
    report.append("TEST 3: What Predicts Winner?")
    report.append("-" * 70)
    
    # Compute matching
    dineof_rough_match = np.abs(results_df['dineof_roughness'] - results_df['insitu_roughness'])
    dincae_rough_match = np.abs(results_df['dincae_roughness'] - results_df['insitu_roughness'])
    
    extreme_correct = ((results_df['dineof_extreme_mae'] < results_df['dincae_extreme_mae']) & (results_df['winner'] == 'dineof') |
                       (results_df['dineof_extreme_mae'] > results_df['dincae_extreme_mae']) & (results_df['winner'] == 'dincae')).sum()
    rough_correct = ((dineof_rough_match < dincae_rough_match) & (results_df['winner'] == 'dineof') |
                     (dineof_rough_match > dincae_rough_match) & (results_df['winner'] == 'dincae')).sum()
    
    report.append(f"  Peak matching predicts winner:      {extreme_correct}/{n} ({100*extreme_correct/n:.0f}%)")
    report.append(f"  Roughness matching predicts winner: {rough_correct}/{n} ({100*rough_correct/n:.0f}%)")
    report.append(f"  (Random chance would be 50%)")
    
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
    report.append("  5. Extremes defined by percentiles of matched timestamps only")
    report.append("  6. Roughness metric is approximate for irregularly sampled data")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    
    out_path = os.path.join(output_dir, 'peak_roughness_summary.txt')
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
    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Process all lakes (not just those with both methods)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Defaults
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "amplitude_roughness_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PEAK MATCHING vs ROUGHNESS ANALYSIS")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Alpha: {args.alpha}")
    print(f"Output: {args.output_dir}")
    print("NOTE: Requires timeseries CSVs from insitu_validation.py")
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
            verbose=args.verbose
        )
        if result is not None:
            result['winner'] = winners.get(lake_id, None)
            results.append(result)
    
    print(f"\nSuccessfully analyzed {len(results)} lakes")
    
    if len(results) < 3:
        print("ERROR: Not enough lakes with complete data for analysis")
        return 1
    
    # Build results DataFrame (results are already flat dicts)
    results_df = pd.DataFrame(results)
    
    # Filter to lakes with winner data
    results_df = results_df[results_df['winner'].notna()]
    print(f"Lakes with winner data: {len(results_df)}")
    
    if len(results_df) < 3:
        print("ERROR: Not enough lakes with winner data")
        return 1
    
    # Save raw results
    csv_path = os.path.join(args.output_dir, 'peak_roughness_metrics.csv')
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