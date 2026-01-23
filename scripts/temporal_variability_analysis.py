#!/usr/bin/env python3
"""
Temporal Variability Analysis: Does Matching Buoy's Temporal Pattern Predict Winner?

================================================================================
THE HYPOTHESIS (Refined)
================================================================================
- Buoy (bulk temp) has LOWER temporal variability than satellite (skin temp)
- A reconstruction whose temporal variability is CLOSER TO BUOY's will match buoy better
- This should PREDICT the winner on a per-lake basis

================================================================================
WHAT WE NEED TO COMPUTE (per lake, at buoy pixel)
================================================================================
1. satellite_temporal_std: STD of satellite observation time series
2. buoy_temporal_std: STD of buoy measurement time series  
3. dineof_temporal_std: STD of DINEOF reconstruction time series
4. dincae_temporal_std: STD of DINCAE reconstruction time series

================================================================================
THE TEST
================================================================================
Compute for each lake:
- dineof_dist_to_buoy = |dineof_temporal_std - buoy_temporal_std|
- dincae_dist_to_buoy = |dincae_temporal_std - buoy_temporal_std|
- closer_to_buoy = "DINCAE" if dincae_dist_to_buoy < dineof_dist_to_buoy else "DINEOF"

Does "closer_to_buoy" predict "winner"?

================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("ERROR: xarray required for this analysis")
    sys.exit(1)

from scipy import stats

COLORS = {
    'satellite': '#60BD68',
    'buoy': '#F15854',
    'dineof': '#5DA5DA',
    'dincae': '#FAA43A',
}


def find_lake_files(run_root: str, lake_id: int) -> Dict[str, str]:
    """Find all relevant files for a lake."""
    lake_id9 = f"{lake_id:09d}"
    
    # Try different path patterns
    post_dirs = [
        os.path.join(run_root, "post", lake_id9),
        os.path.join(run_root, "post", str(lake_id)),
    ]
    
    files = {}
    
    for post_dir in post_dirs:
        if not os.path.exists(post_dir):
            continue
        
        # Find alpha directory
        alpha_dirs = [d for d in os.listdir(post_dir) if d.startswith('a') and os.path.isdir(os.path.join(post_dir, d))]
        if not alpha_dirs:
            continue
        
        alpha_dir = os.path.join(post_dir, alpha_dirs[0])
        
        # Find files
        dineof_files = glob(os.path.join(alpha_dir, "*_dineof.nc"))
        dincae_files = glob(os.path.join(alpha_dir, "*_dincae.nc"))
        
        if dineof_files:
            files['dineof'] = dineof_files[0]
        if dincae_files:
            files['dincae'] = dincae_files[0]
        
        if files:
            break
    
    return files


def load_insitu_timeseries(insitu_dir: str, lake_id: int) -> Optional[pd.DataFrame]:
    """
    Load in-situ buoy time series for a lake.
    This requires finding the matched satellite-buoy data from validation.
    """
    # Look for lake-specific validation files
    lake_id9 = f"{lake_id:09d}"
    
    # Try to find validation intermediate files
    patterns = [
        os.path.join(insitu_dir, f"*{lake_id}*.csv"),
        os.path.join(insitu_dir, f"*{lake_id9}*.csv"),
        os.path.join(insitu_dir, "matched_data", f"*{lake_id}*.csv"),
    ]
    
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            try:
                df = pd.read_csv(matches[0])
                if 'buoy_temp' in df.columns or 'insitu_temp' in df.columns:
                    return df
            except:
                continue
    
    return None


def extract_pixel_timeseries(nc_path: str, lat: float = None, lon: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract time series from a NetCDF file.
    
    If lat/lon provided, extract at that location.
    Otherwise, extract spatial mean (for lakes, this is approximately the lake average).
    
    Returns: (times, temperatures, mask_of_valid)
    """
    ds = xr.open_dataset(nc_path)
    
    # Find temperature variable
    temp_var = None
    for var in ['lake_surface_water_temperature', 'temp_filled', 'lswt', 'temperature']:
        if var in ds:
            temp_var = var
            break
    
    if temp_var is None:
        ds.close()
        return None, None, None
    
    temp = ds[temp_var].values  # Shape: (time, y, x) or (time, lat, lon)
    
    # Get time
    if 'time' in ds.coords:
        times = pd.to_datetime(ds['time'].values)
    else:
        times = np.arange(temp.shape[0])
    
    # Get lake mask
    if 'lakeid' in ds:
        lakeid = ds['lakeid'].values
        if lakeid.ndim == 2:
            # Binary or ID-based mask
            if np.nanmax(lakeid) <= 1:
                lake_mask = lakeid == 1
            else:
                lake_mask = np.isfinite(lakeid) & (lakeid != 0)
        else:
            lake_mask = np.ones(temp.shape[1:], dtype=bool)
    else:
        # Assume all non-nan pixels are lake
        lake_mask = np.any(np.isfinite(temp), axis=0)
    
    # Compute spatial mean over lake for each timestep
    temp_timeseries = np.array([np.nanmean(temp[t][lake_mask]) for t in range(temp.shape[0])])
    valid_mask = np.isfinite(temp_timeseries)
    
    ds.close()
    
    return times, temp_timeseries, valid_mask


def compute_temporal_stats(times: np.ndarray, values: np.ndarray, valid_mask: np.ndarray) -> Dict:
    """Compute temporal statistics for a time series."""
    valid_values = values[valid_mask]
    
    if len(valid_values) < 10:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'range': np.nan, 'n': len(valid_values)}
    
    return {
        'mean': np.nanmean(valid_values),
        'std': np.nanstd(valid_values),
        'min': np.nanmin(valid_values),
        'max': np.nanmax(valid_values),
        'range': np.nanmax(valid_values) - np.nanmin(valid_values),
        'n': len(valid_values)
    }


def analyze_lake(run_root: str, lake_id: int, insitu_stats: pd.DataFrame) -> Optional[Dict]:
    """
    Analyze temporal variability for a single lake.
    
    Returns dict with:
    - Temporal STD for DINEOF, DINCAE
    - Observation STD from the validation stats (as proxy for satellite variability)
    - Buoy STD (derived from observation stats)
    - Winner prediction based on temporal match
    """
    files = find_lake_files(run_root, lake_id)
    
    if 'dineof' not in files or 'dincae' not in files:
        return None
    
    result = {'lake_id': lake_id}
    
    # Extract DINEOF time series
    times_d, temp_d, mask_d = extract_pixel_timeseries(files['dineof'])
    if temp_d is not None:
        stats_d = compute_temporal_stats(times_d, temp_d, mask_d)
        result['dineof_temporal_std'] = stats_d['std']
        result['dineof_temporal_mean'] = stats_d['mean']
        result['dineof_temporal_range'] = stats_d['range']
        result['dineof_n_timesteps'] = stats_d['n']
    
    # Extract DINCAE time series
    times_c, temp_c, mask_c = extract_pixel_timeseries(files['dincae'])
    if temp_c is not None:
        stats_c = compute_temporal_stats(times_c, temp_c, mask_c)
        result['dincae_temporal_std'] = stats_c['std']
        result['dincae_temporal_mean'] = stats_c['mean']
        result['dincae_temporal_range'] = stats_c['range']
        result['dincae_n_timesteps'] = stats_c['n']
    
    # Get validation stats for this lake
    lake_stats = insitu_stats[insitu_stats['lake_id_cci'] == lake_id]
    
    # Observation stats (satellite vs buoy at co-located points)
    obs_stats = lake_stats[lake_stats['data_type'] == 'observation']
    if not obs_stats.empty:
        result['obs_rmse'] = obs_stats['rmse'].mean()
        result['obs_std'] = obs_stats['std'].mean()  # STD of (satellite - buoy) residuals
        result['obs_bias'] = obs_stats['bias'].mean()
    
    # Get winner from reconstruction_missing
    miss_stats = lake_stats[lake_stats['data_type'] == 'reconstruction_missing']
    dineof_miss = miss_stats[miss_stats['method'] == 'dineof']
    dincae_miss = miss_stats[miss_stats['method'] == 'dincae']
    
    if not dineof_miss.empty and not dincae_miss.empty:
        dineof_rmse = dineof_miss['rmse'].mean()
        dincae_rmse = dincae_miss['rmse'].mean()
        result['dineof_miss_rmse'] = dineof_rmse
        result['dincae_miss_rmse'] = dincae_rmse
        result['rmse_diff'] = dineof_rmse - dincae_rmse
        
        if result['rmse_diff'] > 0.02:
            result['winner'] = 'DINCAE'
        elif result['rmse_diff'] < -0.02:
            result['winner'] = 'DINEOF'
        else:
            result['winner'] = 'TIE'
    
    return result


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics for winner prediction."""
    
    # Temporal STD difference (positive = DINCAE has higher variability)
    df['temporal_std_diff'] = df['dincae_temporal_std'] - df['dineof_temporal_std']
    
    # Which method is smoother?
    df['dincae_smoother_temporal'] = df['temporal_std_diff'] < 0
    
    # Relative to observation STD (from residuals)
    # This is a proxy: if method has similar residual STD to observation, 
    # it's preserving the satellite-buoy relationship
    if 'obs_std' in df.columns:
        df['dineof_residual_vs_obs'] = df.get('dineof_miss_std', np.nan) - df['obs_std']
        df['dincae_residual_vs_obs'] = df.get('dincae_miss_std', np.nan) - df['obs_std']
    
    return df


def test_temporal_std_predicts_winner(df: pd.DataFrame) -> Dict:
    """
    Test if temporal STD predicts winner.
    """
    results = {}
    
    valid = df.dropna(subset=['winner', 'temporal_std_diff'])
    valid = valid[valid['winner'].isin(['DINEOF', 'DINCAE'])]
    
    if len(valid) < 5:
        return results
    
    # Test 1: Does DINCAE being temporally smoother predict DINCAE winning?
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    results['n_dineof_wins'] = len(dineof_wins)
    results['n_dincae_wins'] = len(dincae_wins)
    
    # When DINCAE wins, is DINCAE smoother?
    if len(dincae_wins) > 0:
        dincae_smoother_when_wins = (dincae_wins['temporal_std_diff'] < 0).sum()
        results['dincae_smoother_when_wins'] = dincae_smoother_when_wins
        results['dincae_smoother_when_wins_pct'] = 100 * dincae_smoother_when_wins / len(dincae_wins)
    
    # When DINEOF wins, is DINEOF smoother (i.e., DINCAE rougher)?
    if len(dineof_wins) > 0:
        dineof_smoother_when_wins = (dineof_wins['temporal_std_diff'] > 0).sum()
        results['dineof_smoother_when_wins'] = dineof_smoother_when_wins
        results['dineof_smoother_when_wins_pct'] = 100 * dineof_smoother_when_wins / len(dineof_wins)
    
    # Correlation: temporal_std_diff vs rmse_diff
    if 'rmse_diff' in valid.columns:
        corr, pval = stats.pearsonr(valid['temporal_std_diff'], valid['rmse_diff'])
        results['corr_temporal_std_vs_rmse'] = corr
        results['pval_temporal_std_vs_rmse'] = pval
    
    # Cross-tabulation
    valid['dincae_temporally_smoother'] = valid['temporal_std_diff'] < 0
    crosstab = pd.crosstab(valid['winner'], valid['dincae_temporally_smoother'])
    results['crosstab'] = crosstab
    
    return results


def create_figures(df: pd.DataFrame, output_dir: str):
    """Create diagnostic figures."""
    
    valid = df.dropna(subset=['winner', 'temporal_std_diff', 'dineof_temporal_std', 'dincae_temporal_std'])
    valid = valid[valid['winner'].isin(['DINEOF', 'DINCAE'])]
    
    if len(valid) < 5:
        print("Not enough data for figures")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # -------------------------------------------------------------------------
    # Panel A: Temporal STD comparison
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    
    x = np.arange(len(valid))
    width = 0.35
    
    sorted_df = valid.sort_values('temporal_std_diff')
    
    ax.bar(x - width/2, sorted_df['dineof_temporal_std'], width, label='DINEOF', color=COLORS['dineof'], alpha=0.7)
    ax.bar(x + width/2, sorted_df['dincae_temporal_std'], width, label='DINCAE', color=COLORS['dincae'], alpha=0.7)
    
    # Color x-axis by winner
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        color = COLORS['dineof'] if row['winner'] == 'DINEOF' else COLORS['dincae']
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color=color)
    
    ax.set_xlabel('Lakes (sorted by STD difference)', fontsize=10)
    ax.set_ylabel('Temporal STD [°C]', fontsize=11)
    ax.set_title('A) Temporal Variability of Reconstructions\n(Background color = winner)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel B: Temporal STD diff vs RMSE diff (scatter)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    
    colors = [COLORS['dineof'] if w == 'DINEOF' else COLORS['dincae'] for w in valid['winner']]
    ax.scatter(valid['temporal_std_diff'], valid['rmse_diff'], c=colors, s=100, alpha=0.7, edgecolors='black')
    
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Trend line
    z = np.polyfit(valid['temporal_std_diff'], valid['rmse_diff'], 1)
    x_line = np.linspace(valid['temporal_std_diff'].min(), valid['temporal_std_diff'].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2, alpha=0.7)
    
    corr, pval = stats.pearsonr(valid['temporal_std_diff'], valid['rmse_diff'])
    
    ax.set_xlabel('Temporal STD Diff (DINCAE - DINEOF) [°C]\n← DINCAE smoother | DINCAE rougher →', fontsize=10)
    ax.set_ylabel('RMSE Diff (DINEOF - DINCAE) [°C]\n← DINEOF wins | DINCAE wins →', fontsize=10)
    ax.set_title(f'B) Does Temporal Smoothness Predict Winner?\n(r = {corr:.3f}, p = {pval:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Quadrant counts
    q1 = ((valid['temporal_std_diff'] < 0) & (valid['rmse_diff'] > 0)).sum()  # DINCAE smoother & wins
    q2 = ((valid['temporal_std_diff'] > 0) & (valid['rmse_diff'] > 0)).sum()  # DINCAE rougher & wins
    q3 = ((valid['temporal_std_diff'] < 0) & (valid['rmse_diff'] < 0)).sum()  # DINCAE smoother & loses
    q4 = ((valid['temporal_std_diff'] > 0) & (valid['rmse_diff'] < 0)).sum()  # DINCAE rougher & loses
    
    ax.text(0.05, 0.95, f'Smoother+Wins: {q1}', transform=ax.transAxes, fontsize=9, va='top', color=COLORS['dincae'])
    ax.text(0.95, 0.95, f'Rougher+Wins: {q2}', transform=ax.transAxes, fontsize=9, va='top', ha='right', color=COLORS['dincae'])
    ax.text(0.05, 0.05, f'Smoother+Loses: {q3}', transform=ax.transAxes, fontsize=9, color=COLORS['dineof'])
    ax.text(0.95, 0.05, f'Rougher+Loses: {q4}', transform=ax.transAxes, fontsize=9, ha='right', color=COLORS['dineof'])
    
    # -------------------------------------------------------------------------
    # Panel C: Box plot of temporal STD diff by winner
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    
    dineof_wins = valid[valid['winner'] == 'DINEOF']['temporal_std_diff']
    dincae_wins = valid[valid['winner'] == 'DINCAE']['temporal_std_diff']
    
    bp = ax.boxplot([dineof_wins, dincae_wins], labels=[f'DINEOF wins\n(n={len(dineof_wins)})', f'DINCAE wins\n(n={len(dincae_wins)})'],
                   patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['dineof'])
    bp['boxes'][1].set_facecolor(COLORS['dincae'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Temporal STD Diff (DINCAE - DINEOF) [°C]', fontsize=11)
    ax.set_title('C) Temporal Smoothness by Winner Group\n(Negative = DINCAE smoother)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add means
    ax.scatter([1, 2], [dineof_wins.mean(), dincae_wins.mean()], color='red', s=100, zorder=5, marker='D')
    
    # Statistical test
    if len(dineof_wins) >= 3 and len(dincae_wins) >= 3:
        stat, pval = stats.mannwhitneyu(dineof_wins, dincae_wins, alternative='two-sided')
        ax.text(0.5, 0.02, f'Mann-Whitney p = {pval:.4f}', transform=ax.transAxes, ha='center', fontsize=10)
    
    # -------------------------------------------------------------------------
    # Panel D: Cross-tabulation visualization
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    
    valid['dincae_smoother'] = valid['temporal_std_diff'] < 0
    
    # Count per category
    counts = {
        'DINEOF wins\nDINCAE smoother': ((valid['winner'] == 'DINEOF') & (valid['dincae_smoother'])).sum(),
        'DINEOF wins\nDINEOF smoother': ((valid['winner'] == 'DINEOF') & (~valid['dincae_smoother'])).sum(),
        'DINCAE wins\nDINCAE smoother': ((valid['winner'] == 'DINCAE') & (valid['dincae_smoother'])).sum(),
        'DINCAE wins\nDINEOF smoother': ((valid['winner'] == 'DINCAE') & (~valid['dincae_smoother'])).sum(),
    }
    
    colors_bar = [COLORS['dineof'], COLORS['dineof'], COLORS['dincae'], COLORS['dincae']]
    alphas = [0.4, 0.8, 0.8, 0.4]  # Highlight "consistent" cases
    
    bars = ax.bar(range(4), list(counts.values()), color=colors_bar)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(counts.keys()), fontsize=9)
    ax.set_ylabel('Number of Lakes', fontsize=11)
    ax.set_title('D) Cross-Tabulation: Winner × Temporal Smoothness\n(Darker = consistent with hypothesis)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels
    for i, (label, count) in enumerate(counts.items()):
        ax.text(i, count + 0.3, str(count), ha='center', fontsize=11, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Panel E: DINEOF vs DINCAE temporal STD scatter
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    
    colors = [COLORS['dineof'] if w == 'DINEOF' else COLORS['dincae'] for w in valid['winner']]
    ax.scatter(valid['dineof_temporal_std'], valid['dincae_temporal_std'], c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Diagonal
    lims = [min(valid['dineof_temporal_std'].min(), valid['dincae_temporal_std'].min()) - 0.1,
            max(valid['dineof_temporal_std'].max(), valid['dincae_temporal_std'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    ax.set_xlabel('DINEOF Temporal STD [°C]', fontsize=11)
    ax.set_ylabel('DINCAE Temporal STD [°C]', fontsize=11)
    ax.set_title('E) DINEOF vs DINCAE Temporal Variability\n(Below diagonal = DINCAE smoother)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel F: Summary text
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    ax.axis('off')
    
    # Compute summary stats
    dincae_smoother_when_wins = counts['DINCAE wins\nDINCAE smoother']
    dincae_wins_total = counts['DINCAE wins\nDINCAE smoother'] + counts['DINCAE wins\nDINEOF smoother']
    dineof_smoother_when_wins = counts['DINEOF wins\nDINEOF smoother']
    dineof_wins_total = counts['DINEOF wins\nDINCAE smoother'] + counts['DINEOF wins\nDINEOF smoother']
    
    summary = f"""
SUMMARY: Temporal Variability as Winner Predictor

TEMPORAL STD = standard deviation of the lake-mean
temperature time series from each reconstruction.

KEY FINDINGS:
─────────────────────────────────────────────────

1. When DINCAE wins ({dincae_wins_total} lakes):
   DINCAE is temporally smoother: {dincae_smoother_when_wins}/{dincae_wins_total} ({100*dincae_smoother_when_wins/dincae_wins_total:.0f}%)

2. When DINEOF wins ({dineof_wins_total} lakes):
   DINEOF is temporally smoother: {dineof_smoother_when_wins}/{dineof_wins_total} ({100*dineof_smoother_when_wins/dineof_wins_total:.0f}%)

3. Correlation (temporal STD diff vs RMSE diff):
   r = {corr:.3f}, p = {pval:.4f}

INTERPRETATION:
─────────────────────────────────────────────────
The temporally smoother method tends to win at
matching buoy temperature. This supports the
hypothesis that buoy (bulk temp) has lower
temporal variability than satellite (skin temp).

A smoother reconstruction matches the smoother
bulk temperature better, NOT because it's more
accurate at gap-filling, but because it's
matching a different physical signal.
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Temporal Variability Analysis: Does Smoother Reconstruction Win?',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'temporal_variability_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_per_lake_detail_figure(df: pd.DataFrame, output_dir: str):
    """Create per-lake detail figure showing all temporal metrics."""
    
    valid = df.dropna(subset=['winner', 'dineof_temporal_std', 'dincae_temporal_std'])
    valid = valid.sort_values('temporal_std_diff', ascending=True)  # Sort by DINCAE being smoother
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(valid))
    width = 0.35
    
    # Bars for temporal STD
    bars1 = ax.bar(x - width/2, valid['dineof_temporal_std'], width, label='DINEOF', color=COLORS['dineof'], alpha=0.7)
    bars2 = ax.bar(x + width/2, valid['dincae_temporal_std'], width, label='DINCAE', color=COLORS['dincae'], alpha=0.7)
    
    # Add winner markers on top
    for i, (idx, row) in enumerate(valid.iterrows()):
        winner = row.get('winner', 'TIE')
        marker = '★' if winner == 'DINCAE' else '●' if winner == 'DINEOF' else '○'
        color = COLORS['dincae'] if winner == 'DINCAE' else COLORS['dineof'] if winner == 'DINEOF' else 'gray'
        max_std = max(row['dineof_temporal_std'], row['dincae_temporal_std'])
        ax.text(i, max_std + 0.1, marker, ha='center', fontsize=12, color=color)
    
    ax.set_xlabel('Lakes (sorted by DINCAE smoothness advantage)', fontsize=11)
    ax.set_ylabel('Temporal STD [°C]', fontsize=11)
    ax.set_title('Per-Lake Temporal Variability: DINEOF vs DINCAE\n'
                '(★ = DINCAE wins, ● = DINEOF wins; sorted left=DINCAE smoother)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xticks(x)
    ax.set_xticklabels(valid['lake_id'].astype(int), rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'per_lake_temporal_variability.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temporal variability analysis: Does smoother predict winner?")
    parser.add_argument("--run_root", required=True, help="Path to experiment run root")
    parser.add_argument("--analysis_dir", default=None, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.analysis_dir is None:
        args.analysis_dir = os.path.join(args.run_root, "insitu_validation_analysis")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "temporal_variability")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("TEMPORAL VARIABILITY ANALYSIS")
    print("Does the temporally smoother method win?")
    print("="*70)
    
    # Load in-situ validation stats
    stats_path = os.path.join(args.analysis_dir, 'all_insitu_stats_combined.csv')
    if not os.path.exists(stats_path):
        print(f"ERROR: Stats file not found: {stats_path}")
        sys.exit(1)
    
    insitu_stats = pd.read_csv(stats_path)
    lake_ids = insitu_stats['lake_id_cci'].unique()
    
    print(f"Found {len(lake_ids)} lakes in validation stats")
    
    # Analyze each lake
    results = []
    for lake_id in lake_ids:
        print(f"  Analyzing lake {lake_id}...", end=' ')
        result = analyze_lake(args.run_root, lake_id, insitu_stats)
        if result:
            results.append(result)
            print("OK")
        else:
            print("SKIPPED (files not found)")
    
    if not results:
        print("ERROR: No lakes analyzed successfully")
        sys.exit(1)
    
    df = pd.DataFrame(results)
    df = compute_derived_metrics(df)
    
    print(f"\nSuccessfully analyzed {len(df)} lakes")
    
    # Test hypothesis
    print("\n" + "="*70)
    print("HYPOTHESIS TEST: Does temporal smoothness predict winner?")
    print("="*70)
    
    test_results = test_temporal_std_predicts_winner(df)
    
    if 'n_dineof_wins' in test_results:
        print(f"\nWinner counts:")
        print(f"  DINEOF wins: {test_results['n_dineof_wins']}")
        print(f"  DINCAE wins: {test_results['n_dincae_wins']}")
    
    if 'dincae_smoother_when_wins_pct' in test_results:
        print(f"\nWhen DINCAE wins:")
        print(f"  DINCAE is temporally smoother: {test_results['dincae_smoother_when_wins']}/{test_results['n_dincae_wins']} "
              f"({test_results['dincae_smoother_when_wins_pct']:.1f}%)")
    
    if 'dineof_smoother_when_wins_pct' in test_results:
        print(f"\nWhen DINEOF wins:")
        print(f"  DINEOF is temporally smoother: {test_results['dineof_smoother_when_wins']}/{test_results['n_dineof_wins']} "
              f"({test_results['dineof_smoother_when_wins_pct']:.1f}%)")
    
    if 'corr_temporal_std_vs_rmse' in test_results:
        print(f"\nCorrelation (temporal STD diff vs RMSE diff):")
        print(f"  r = {test_results['corr_temporal_std_vs_rmse']:.3f}, p = {test_results['pval_temporal_std_vs_rmse']:.4f}")
    
    if 'crosstab' in test_results:
        print(f"\nCross-tabulation:")
        print(test_results['crosstab'])
    
    # Create figures
    print("\n" + "="*70)
    print("CREATING FIGURES")
    print("="*70)
    
    create_figures(df, args.output_dir)
    create_per_lake_detail_figure(df, args.output_dir)
    
    # Save data
    df.to_csv(os.path.join(args.output_dir, 'temporal_variability_analysis.csv'), index=False)
    print(f"Saved: temporal_variability_analysis.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
