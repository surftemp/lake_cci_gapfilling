#!/usr/bin/env python3
"""
Investigation: The Smoothness-Beats-Observation Hypothesis

================================================================================
THE HYPOTHESIS
================================================================================
1. Buoy measures BULK temperature, which is temporally SMOOTHER than satellite 
   skin temperature (fewer extreme peaks and troughs due to thermal inertia)

2. A reconstruction method that produces SMOOTHER output will match buoy better,
   not because it's more accurate at gap-filling, but because smoothness 
   qualitatively matches the behavior of bulk temperature

3. THE SMOKING GUN: If reconstruction RMSE < observation RMSE (vs buoy), then
   the reconstruction "beats" the satellite observation itself at matching buoy.
   This should be IMPOSSIBLE if they're measuring the same thing - you can't
   beat ground truth unless you're measuring something different.

================================================================================
KEY METRICS
================================================================================
obs_rmse:   Satellite observation vs Buoy (baseline)
recon_rmse: Reconstruction vs Buoy

obs_std:    Temporal variability of satellite observation
recon_std:  Temporal variability of reconstruction

"Beats observation": recon_rmse < obs_rmse
"Smoother than obs": recon_std < obs_std

================================================================================
PREDICTIONS
================================================================================
If the hypothesis is correct:
1. Methods that are smoother than observation will beat observation at matching buoy
2. DINCAE wins when DINCAE is smoother than both DINEOF and observation
3. The correlation between (recon_std - obs_std) and (recon_rmse - obs_rmse) should be POSITIVE
   (smoother → lower RMSE vs buoy)

================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

COLORS = {
    'observation': '#60BD68',
    'dineof': '#5DA5DA', 
    'dincae': '#FAA43A',
    'winner_dineof': '#5DA5DA',
    'winner_dincae': '#FAA43A',
}


def load_data(analysis_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    return pd.read_csv(csv_path)


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-lake table comparing observation vs reconstruction.
    
    For each lake, we need:
    - obs_rmse, obs_std: Satellite vs Buoy baseline
    - dineof_rmse, dineof_std: DINEOF reconstruction vs Buoy
    - dincae_rmse, dincae_std: DINCAE reconstruction vs Buoy
    
    We compare at reconstruction_missing (true gap-fill) AND reconstruction_observed.
    """
    results = []
    
    for lake_id in df['lake_id_cci'].unique():
        lake_df = df[df['lake_id_cci'] == lake_id]
        row = {'lake_id': int(lake_id)}
        
        # Observation baseline (satellite vs buoy)
        obs = lake_df[lake_df['data_type'] == 'observation']
        if not obs.empty:
            row['obs_rmse'] = obs['rmse'].mean()
            row['obs_std'] = obs['std'].mean()
            row['obs_bias'] = obs['bias'].mean()
            row['obs_mae'] = obs['mae'].mean()
            row['obs_n'] = obs['n_matches'].sum()
        
        # Reconstruction at MISSING pixels (true gap-fill)
        for method in ['dineof', 'dincae']:
            miss = lake_df[(lake_df['data_type'] == 'reconstruction_missing') & 
                          (lake_df['method'] == method)]
            if not miss.empty:
                row[f'{method}_miss_rmse'] = miss['rmse'].mean()
                row[f'{method}_miss_std'] = miss['std'].mean()
                row[f'{method}_miss_bias'] = miss['bias'].mean()
                row[f'{method}_miss_n'] = miss['n_matches'].sum()
        
        # Reconstruction at OBSERVED pixels (training overlap)
        for method in ['dineof', 'dincae']:
            obsv = lake_df[(lake_df['data_type'] == 'reconstruction_observed') & 
                          (lake_df['method'] == method)]
            if not obsv.empty:
                row[f'{method}_obs_rmse'] = obsv['rmse'].mean()
                row[f'{method}_obs_std'] = obsv['std'].mean()
                row[f'{method}_obs_bias'] = obsv['bias'].mean()
                row[f'{method}_obs_n'] = obsv['n_matches'].sum()
        
        results.append(row)
    
    lake_df = pd.DataFrame(results)
    
    # Compute derived metrics
    # Does reconstruction beat observation?
    for method in ['dineof', 'dincae']:
        for loc in ['miss', 'obs']:
            rmse_col = f'{method}_{loc}_rmse'
            std_col = f'{method}_{loc}_std'
            
            if rmse_col in lake_df.columns:
                # RMSE improvement (positive = reconstruction beats observation)
                lake_df[f'{method}_{loc}_beats_obs'] = lake_df['obs_rmse'] - lake_df[rmse_col]
                
                # STD difference (negative = reconstruction is smoother)
                lake_df[f'{method}_{loc}_smoother'] = lake_df['obs_std'] - lake_df[std_col]
    
    # Winner at missing pixels
    if 'dineof_miss_rmse' in lake_df.columns and 'dincae_miss_rmse' in lake_df.columns:
        lake_df['rmse_diff'] = lake_df['dineof_miss_rmse'] - lake_df['dincae_miss_rmse']
        lake_df['winner'] = np.where(lake_df['rmse_diff'] > 0.02, 'DINCAE',
                                     np.where(lake_df['rmse_diff'] < -0.02, 'DINEOF', 'TIE'))
    
    return lake_df


def analyze_beats_observation(lake_df: pd.DataFrame, output_dir: str):
    """
    THE SMOKING GUN TEST:
    Does reconstruction beat observation at matching buoy?
    If yes, is it because it's smoother?
    """
    print("\n" + "="*70)
    print("SMOKING GUN TEST: Does Reconstruction Beat Observation?")
    print("="*70)
    
    print("\nAt MISSING pixels (true gap-fill):")
    print("-" * 60)
    
    for method in ['dineof', 'dincae']:
        col = f'{method}_miss_beats_obs'
        if col not in lake_df.columns:
            continue
        
        valid = lake_df[col].dropna()
        beats = (valid > 0).sum()
        
        print(f"\n{method.upper()}:")
        print(f"  Beats observation: {beats}/{len(valid)} lakes ({100*beats/len(valid):.1f}%)")
        print(f"  Mean improvement: {valid.mean():+.3f}°C (positive = beats obs)")
        print(f"  Range: [{valid.min():.3f}, {valid.max():.3f}]")
    
    print("\n" + "-" * 60)
    print("At OBSERVED pixels (training overlap):")
    print("-" * 60)
    
    for method in ['dineof', 'dincae']:
        col = f'{method}_obs_beats_obs'
        if col not in lake_df.columns:
            continue
        
        valid = lake_df[col].dropna()
        beats = (valid > 0).sum()
        
        print(f"\n{method.upper()}:")
        print(f"  Beats observation: {beats}/{len(valid)} lakes ({100*beats/len(valid):.1f}%)")
        print(f"  Mean improvement: {valid.mean():+.3f}°C")


def analyze_smoothness_beats_observation_correlation(lake_df: pd.DataFrame, output_dir: str):
    """
    KEY TEST: Is there a correlation between being smoother and beating observation?
    
    If smoothness → better buoy match, then:
    (obs_std - recon_std) should correlate with (obs_rmse - recon_rmse)
    Both positive = smoother AND beats obs
    """
    print("\n" + "="*70)
    print("KEY TEST: Does Smoothness → Beating Observation?")
    print("="*70)
    
    correlations = []
    
    for method in ['dineof', 'dincae']:
        for loc in ['miss', 'obs']:
            smoother_col = f'{method}_{loc}_smoother'  # positive = recon is smoother
            beats_col = f'{method}_{loc}_beats_obs'    # positive = recon beats obs
            
            if smoother_col not in lake_df.columns or beats_col not in lake_df.columns:
                continue
            
            valid = lake_df[[smoother_col, beats_col]].dropna()
            if len(valid) < 5:
                continue
            
            r, p = stats.pearsonr(valid[smoother_col], valid[beats_col])
            correlations.append({
                'method': method.upper(),
                'location': loc,
                'r': r,
                'p': p,
                'n': len(valid)
            })
            
            print(f"\n{method.upper()} at {loc.upper()} pixels:")
            print(f"  Correlation (smoother vs beats_obs): r = {r:.3f}, p = {p:.4f}")
            if r > 0.3 and p < 0.05:
                print(f"  *** SIGNIFICANT: Smoother reconstructions DO beat observation! ***")
    
    return correlations


def analyze_winner_smoothness_relationship(lake_df: pd.DataFrame, output_dir: str):
    """
    Does the winner correlate with being smoother than observation?
    """
    print("\n" + "="*70)
    print("WINNER vs SMOOTHNESS RELATIVE TO OBSERVATION")
    print("="*70)
    
    valid = lake_df.dropna(subset=['winner', 'dineof_miss_smoother', 'dincae_miss_smoother'])
    
    print("\nAt MISSING pixels:")
    print("-" * 60)
    
    for winner in ['DINEOF', 'DINCAE']:
        subset = valid[valid['winner'] == winner]
        if len(subset) == 0:
            continue
        
        print(f"\n{winner} wins ({len(subset)} lakes):")
        
        # Is DINEOF smoother than obs?
        dineof_smoother = (subset['dineof_miss_smoother'] > 0).sum()
        print(f"  DINEOF smoother than obs: {dineof_smoother}/{len(subset)} ({100*dineof_smoother/len(subset):.0f}%)")
        
        # Is DINCAE smoother than obs?
        dincae_smoother = (subset['dincae_miss_smoother'] > 0).sum()
        print(f"  DINCAE smoother than obs: {dincae_smoother}/{len(subset)} ({100*dincae_smoother/len(subset):.0f}%)")
        
        # Mean smoothness values
        print(f"  DINEOF smoothness: {subset['dineof_miss_smoother'].mean():+.3f} (pos=smoother)")
        print(f"  DINCAE smoothness: {subset['dincae_miss_smoother'].mean():+.3f}")


def create_smoking_gun_figure(lake_df: pd.DataFrame, output_dir: str):
    """
    Create the key figure showing:
    1. How often each method beats observation
    2. Correlation between smoothness and beating observation
    3. Winner determination by relative smoothness
    """
    print("\n" + "="*70)
    print("CREATING: Smoking Gun Figure")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # =========================================================================
    # Row 1: Does reconstruction beat observation?
    # =========================================================================
    
    # Panel A: Bar chart - how often each method beats observation
    ax = axes[0, 0]
    
    methods = ['dineof', 'dincae']
    locations = ['miss', 'obs']
    
    x = np.arange(len(methods))
    width = 0.35
    
    miss_beats = []
    obs_beats = []
    
    for method in methods:
        miss_col = f'{method}_miss_beats_obs'
        obs_col = f'{method}_obs_beats_obs'
        
        miss_valid = lake_df[miss_col].dropna()
        obs_valid = lake_df[obs_col].dropna()
        
        miss_beats.append(100 * (miss_valid > 0).sum() / len(miss_valid) if len(miss_valid) > 0 else 0)
        obs_beats.append(100 * (obs_valid > 0).sum() / len(obs_valid) if len(obs_valid) > 0 else 0)
    
    ax.bar(x - width/2, miss_beats, width, label='At Missing Pixels', color=['#5DA5DA', '#FAA43A'], alpha=0.7)
    ax.bar(x + width/2, obs_beats, width, label='At Observed Pixels', color=['#5DA5DA', '#FAA43A'], alpha=0.4, hatch='//')
    
    ax.set_ylabel('% of Lakes Where Recon Beats Obs', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['DINEOF', 'DINCAE'])
    ax.set_title('A) How Often Does Reconstruction\nBeat Satellite Observation?', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add percentage labels
    for i, (m, o) in enumerate(zip(miss_beats, obs_beats)):
        ax.text(i - width/2, m + 2, f'{m:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax.text(i + width/2, o + 2, f'{o:.0f}%', ha='center', fontsize=10)
    
    # Panel B: Scatter - Smoothness vs Beats Observation (DINEOF)
    ax = axes[0, 1]
    
    valid = lake_df[['dineof_miss_smoother', 'dineof_miss_beats_obs', 'winner']].dropna()
    colors = [COLORS['winner_dineof'] if w == 'DINEOF' else COLORS['winner_dincae'] if w == 'DINCAE' else 'gray' 
              for w in valid['winner']]
    
    ax.scatter(valid['dineof_miss_smoother'], valid['dineof_miss_beats_obs'], 
              c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='black', linewidth=1, label='Recon = Obs RMSE')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Recon = Obs STD')
    
    # Trend line
    r, p = stats.pearsonr(valid['dineof_miss_smoother'], valid['dineof_miss_beats_obs'])
    z = np.polyfit(valid['dineof_miss_smoother'], valid['dineof_miss_beats_obs'], 1)
    x_line = np.linspace(valid['dineof_miss_smoother'].min(), valid['dineof_miss_smoother'].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('DINEOF Smoothness vs Obs\n(positive = smoother than obs)', fontsize=10)
    ax.set_ylabel('DINEOF Beats Obs by [°C]\n(positive = lower RMSE than obs)', fontsize=10)
    ax.set_title(f'B) DINEOF: Smoothness → Beats Obs?\n(r = {r:.3f}, p = {p:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel C: Scatter - Smoothness vs Beats Observation (DINCAE)
    ax = axes[0, 2]
    
    valid = lake_df[['dincae_miss_smoother', 'dincae_miss_beats_obs', 'winner']].dropna()
    colors = [COLORS['winner_dineof'] if w == 'DINEOF' else COLORS['winner_dincae'] if w == 'DINCAE' else 'gray' 
              for w in valid['winner']]
    
    ax.scatter(valid['dincae_miss_smoother'], valid['dincae_miss_beats_obs'],
              c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    r, p = stats.pearsonr(valid['dincae_miss_smoother'], valid['dincae_miss_beats_obs'])
    z = np.polyfit(valid['dincae_miss_smoother'], valid['dincae_miss_beats_obs'], 1)
    x_line = np.linspace(valid['dincae_miss_smoother'].min(), valid['dincae_miss_smoother'].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('DINCAE Smoothness vs Obs\n(positive = smoother than obs)', fontsize=10)
    ax.set_ylabel('DINCAE Beats Obs by [°C]', fontsize=10)
    ax.set_title(f'C) DINCAE: Smoothness → Beats Obs?\n(r = {r:.3f}, p = {p:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 2: Winner determination
    # =========================================================================
    
    # Panel D: Winner by relative smoothness
    ax = axes[1, 0]
    
    valid = lake_df[['dineof_miss_smoother', 'dincae_miss_smoother', 'winner']].dropna()
    
    # Compute: which method is smoother relative to observation?
    valid['dineof_smoother_than_dincae'] = valid['dineof_miss_smoother'] > valid['dincae_miss_smoother']
    
    # Cross-tabulation
    crosstab = pd.crosstab(valid['winner'], valid['dineof_smoother_than_dincae'])
    crosstab.columns = ['DINCAE\nSmoother', 'DINEOF\nSmoother']
    
    crosstab.plot(kind='bar', ax=ax, color=[COLORS['dincae'], COLORS['dineof']], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Winner', fontsize=11)
    ax.set_ylabel('Number of Lakes', fontsize=11)
    ax.set_title('D) Winner by Which Method is Smoother\n(relative to observation)', fontsize=12, fontweight='bold')
    ax.legend(title='Which is Smoother?')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel E: Direct comparison - DINEOF vs DINCAE smoothness
    ax = axes[1, 1]
    
    valid = lake_df[['dineof_miss_smoother', 'dincae_miss_smoother', 'winner']].dropna()
    colors = [COLORS['winner_dineof'] if w == 'DINEOF' else COLORS['winner_dincae'] if w == 'DINCAE' else 'gray'
              for w in valid['winner']]
    
    ax.scatter(valid['dineof_miss_smoother'], valid['dincae_miss_smoother'],
              c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Diagonal
    lims = [min(valid['dineof_miss_smoother'].min(), valid['dincae_miss_smoother'].min()) - 0.05,
            max(valid['dineof_miss_smoother'].max(), valid['dincae_miss_smoother'].max()) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('DINEOF Smoothness vs Obs [°C]', fontsize=11)
    ax.set_ylabel('DINCAE Smoothness vs Obs [°C]', fontsize=11)
    ax.set_title('E) Relative Smoothness Comparison\n(Above diagonal = DINCAE smoother)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add quadrant counts
    q1 = ((valid['dineof_miss_smoother'] > 0) & (valid['dincae_miss_smoother'] > 0)).sum()
    q2 = ((valid['dineof_miss_smoother'] < 0) & (valid['dincae_miss_smoother'] > 0)).sum()
    q3 = ((valid['dineof_miss_smoother'] < 0) & (valid['dincae_miss_smoother'] < 0)).sum()
    q4 = ((valid['dineof_miss_smoother'] > 0) & (valid['dincae_miss_smoother'] < 0)).sum()
    
    ax.text(0.95, 0.95, f'Both smooth: {q1}', transform=ax.transAxes, ha='right', va='top', fontsize=9)
    ax.text(0.05, 0.95, f'DINCAE smooth: {q2}', transform=ax.transAxes, ha='left', va='top', fontsize=9)
    ax.text(0.05, 0.05, f'Both rough: {q3}', transform=ax.transAxes, ha='left', va='bottom', fontsize=9)
    ax.text(0.95, 0.05, f'DINEOF smooth: {q4}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    
    # Panel F: Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    # Compute summary stats
    dineof_beats = (lake_df['dineof_miss_beats_obs'] > 0).sum()
    dincae_beats = (lake_df['dincae_miss_beats_obs'] > 0).sum()
    total = lake_df['dineof_miss_beats_obs'].notna().sum()
    
    dineof_smoother_overall = (lake_df['dineof_miss_smoother'] > 0).sum()
    dincae_smoother_overall = (lake_df['dincae_miss_smoother'] > 0).sum()
    
    summary = f"""
SUMMARY: The Smoothness-Beats-Observation Hypothesis

KEY FINDING:
At MISSING pixels (true gap-fill), reconstruction often
BEATS satellite observation at matching buoy:

  DINEOF beats obs: {dineof_beats}/{total} lakes ({100*dineof_beats/total:.0f}%)
  DINCAE beats obs: {dincae_beats}/{total} lakes ({100*dincae_beats/total:.0f}%)

This is only possible if buoy measures something different
from what satellite measures (bulk vs skin temperature).

SMOOTHNESS COMPARISON (vs satellite observation):
  DINEOF smoother than obs: {dineof_smoother_overall}/{total} ({100*dineof_smoother_overall/total:.0f}%)
  DINCAE smoother than obs: {dincae_smoother_overall}/{total} ({100*dincae_smoother_overall/total:.0f}%)

INTERPRETATION:
Buoy (bulk temp) is inherently smoother than satellite (skin).
Methods that smooth the reconstruction match buoy better,
not because they're better at gap-filling, but because
smoothness matches the qualitative behavior of bulk temp.
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('The Smoothness Hypothesis: Smoother Reconstructions Beat Observation\n'
                '(Because buoy measures smoother bulk temperature)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'smoothness_beats_observation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_detailed_lake_comparison(lake_df: pd.DataFrame, output_dir: str):
    """
    Create detailed per-lake comparison showing obs vs recon RMSE and STD.
    """
    print("\n" + "="*70)
    print("CREATING: Detailed Per-Lake Comparison")
    print("="*70)
    
    valid = lake_df.dropna(subset=['obs_rmse', 'dineof_miss_rmse', 'dincae_miss_rmse', 
                                    'obs_std', 'dineof_miss_std', 'dincae_miss_std']).copy()
    
    # Sort by how much DINCAE beats observation
    valid = valid.sort_values('dincae_miss_beats_obs', ascending=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    x = np.arange(len(valid))
    width = 0.25
    
    # Panel A: RMSE comparison
    ax = axes[0]
    ax.bar(x - width, valid['obs_rmse'], width, label='Observation', color=COLORS['observation'], alpha=0.7)
    ax.bar(x, valid['dineof_miss_rmse'], width, label='DINEOF', color=COLORS['dineof'], alpha=0.7)
    ax.bar(x + width, valid['dincae_miss_rmse'], width, label='DINCAE', color=COLORS['dincae'], alpha=0.7)
    
    ax.set_ylabel('RMSE vs Buoy [°C]', fontsize=11)
    ax.set_title('A) RMSE Comparison: Observation vs Reconstructions\n'
                '(Lower = better match to buoy; reconstruction CAN beat observation!)', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: STD comparison
    ax = axes[1]
    ax.bar(x - width, valid['obs_std'], width, label='Observation', color=COLORS['observation'], alpha=0.7)
    ax.bar(x, valid['dineof_miss_std'], width, label='DINEOF', color=COLORS['dineof'], alpha=0.7)
    ax.bar(x + width, valid['dincae_miss_std'], width, label='DINCAE', color=COLORS['dincae'], alpha=0.7)
    
    ax.set_xlabel('Lakes (sorted by DINCAE improvement over observation)', fontsize=11)
    ax.set_ylabel('STD [°C]', fontsize=11)
    ax.set_title('B) Temporal Variability (STD) Comparison\n'
                '(Lower = smoother; smoother often means better buoy match)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(valid['lake_id'].astype(int), rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'detailed_lake_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_statistics_summary(lake_df: pd.DataFrame, correlations: list, output_dir: str):
    """
    Generate comprehensive statistics summary.
    """
    print("\n" + "="*70)
    print("STATISTICS SUMMARY")
    print("="*70)
    
    lines = []
    lines.append("="*70)
    lines.append("THE SMOOTHNESS-BEATS-OBSERVATION HYPOTHESIS: STATISTICS")
    lines.append("="*70)
    lines.append("")
    
    # Basic counts
    total = lake_df['dineof_miss_rmse'].notna().sum()
    
    lines.append(f"Total lakes analyzed: {total}")
    lines.append("")
    
    # How often does reconstruction beat observation?
    lines.append("HOW OFTEN DOES RECONSTRUCTION BEAT SATELLITE OBSERVATION?")
    lines.append("-" * 50)
    lines.append("(At MISSING pixels - true gap-fill)")
    
    for method in ['dineof', 'dincae']:
        col = f'{method}_miss_beats_obs'
        valid = lake_df[col].dropna()
        beats = (valid > 0).sum()
        mean_improvement = valid.mean()
        lines.append(f"  {method.upper()}: {beats}/{len(valid)} lakes ({100*beats/len(valid):.1f}%), "
                    f"mean improvement: {mean_improvement:+.3f}°C")
    lines.append("")
    
    # Smoothness relative to observation
    lines.append("SMOOTHNESS RELATIVE TO OBSERVATION (lower STD = smoother)")
    lines.append("-" * 50)
    
    for method in ['dineof', 'dincae']:
        col = f'{method}_miss_smoother'
        valid = lake_df[col].dropna()
        smoother = (valid > 0).sum()
        mean_diff = valid.mean()
        lines.append(f"  {method.upper()} smoother than obs: {smoother}/{len(valid)} lakes ({100*smoother/len(valid):.1f}%), "
                    f"mean: {mean_diff:+.3f}°C")
    lines.append("")
    
    # Correlations
    lines.append("KEY CORRELATIONS")
    lines.append("-" * 50)
    for corr in correlations:
        sig = '***' if corr['p'] < 0.01 else '**' if corr['p'] < 0.05 else '*' if corr['p'] < 0.1 else ''
        lines.append(f"  {corr['method']} at {corr['location'].upper()}: "
                    f"smoothness vs beats_obs: r = {corr['r']:.3f}, p = {corr['p']:.4f} {sig}")
    lines.append("")
    
    # Winner analysis
    lines.append("WINNER ANALYSIS")
    lines.append("-" * 50)
    
    valid = lake_df.dropna(subset=['winner'])
    for winner in ['DINEOF', 'DINCAE']:
        subset = valid[valid['winner'] == winner]
        lines.append(f"  {winner} wins: {len(subset)} lakes")
    lines.append("")
    
    # Interpretation
    lines.append("="*70)
    lines.append("INTERPRETATION")
    lines.append("="*70)
    lines.append("""
The fact that reconstruction can BEAT satellite observation at matching
buoy is the key insight. This should be impossible if satellite and buoy
measure the same thing - you can't be more accurate than ground truth.

This happens because:
1. Satellite measures SKIN temperature (~10μm depth, responds quickly)
2. Buoy measures BULK temperature (~0.5-1m depth, smoother due to thermal inertia)
3. Smoother reconstructions match the smoother bulk temperature better

CONCLUSION:
The in-situ validation does NOT measure gap-filling accuracy.
It measures how well the reconstruction matches a DIFFERENT physical
quantity (bulk vs skin temperature). Smoother methods appear "better"
because bulk temperature is smoother, not because they're better at
reconstructing the original satellite observations.

This explains why:
- Satellite CV: DINEOF wins (tests skin temp reconstruction accuracy)
- In-situ: Mixed results (tests match to bulk temp, which is different)
""")
    
    summary = "\n".join(lines)
    print(summary)
    
    save_path = os.path.join(output_dir, 'smoothness_hypothesis_statistics.txt')
    with open(save_path, 'w') as f:
        f.write(summary)
    print(f"\nSaved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test the smoothness-beats-observation hypothesis")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "smoothness_hypothesis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("TESTING: The Smoothness-Beats-Observation Hypothesis")
    print("="*70)
    
    # Load and prepare data
    df = load_data(args.analysis_dir)
    lake_df = build_comparison_table(df)
    
    print(f"Loaded {len(lake_df)} lakes")
    
    # Run analyses
    analyze_beats_observation(lake_df, args.output_dir)
    correlations = analyze_smoothness_beats_observation_correlation(lake_df, args.output_dir)
    analyze_winner_smoothness_relationship(lake_df, args.output_dir)
    create_smoking_gun_figure(lake_df, args.output_dir)
    create_detailed_lake_comparison(lake_df, args.output_dir)
    generate_statistics_summary(lake_df, correlations, args.output_dir)
    
    # Save data
    lake_df.to_csv(os.path.join(args.output_dir, 'smoothness_analysis_table.csv'), index=False)
    print(f"\nSaved: smoothness_analysis_table.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
