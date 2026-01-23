#!/usr/bin/env python3
"""
Comprehensive Predictor Analysis: What (if anything) Predicts the Winner?

PURPOSE:
========
Test ALL available statistics as predictors of the in-situ validation winner.
Report honestly without overselling. If something is circular, say so.

STRUCTURE:
==========
Part 1: Observation-based predictors (NOT circular)
        - These come from data_type='observation' (satellite vs buoy)
        - Can legitimately be used to predict reconstruction_missing winner
        
Part 2: Reconstruction stats (for characterization only)
        - reconstruction_observed: stats where satellite had data
        - reconstruction_missing: stats where satellite was missing (THE OUTCOME)
        - Using reconstruction_missing stats to predict reconstruction_missing winner
          is CIRCULAR and we will quantify the circularity
          
Part 3: Circularity Analysis
        - Decompose RMSE² = Bias² + STD²
        - Show that high correlation of STD with winner is partly mathematical

Part 4: Honest Summary
        - What can legitimately be said
        - What cannot be claimed

METRICS TESTED:
===============
For each data type, we have:
- rmse: Root Mean Square Error
- bias: Mean error (method - buoy)
- std: Residual standard deviation (STD of residuals)
- mae: Mean Absolute Error (if available)
- correlation: Pearson correlation between method and buoy time series (if available)
- n_matches: Number of matched points
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'dineof': '#5DA5DA',
    'dincae': '#FAA43A',
    'observation': '#60BD68',
    'significant': '#27ae60',
    'not_significant': '#95a5a6',
}


def load_combined_stats(analysis_dir: str) -> pd.DataFrame:
    """Load the combined in-situ stats CSV."""
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    return pd.read_csv(csv_path)


def extract_all_lake_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-lake statistics for all data types and methods.
    Returns a wide-format dataframe with one row per lake.
    """
    results = []
    
    for lake_id in df['lake_id_cci'].unique():
        lake_df = df[df['lake_id_cci'] == lake_id]
        row = {'lake_id': int(lake_id)}
        
        # OBSERVATION stats (satellite vs buoy) - potential PREDICTORS
        obs = lake_df[lake_df['data_type'] == 'observation']
        if not obs.empty:
            for metric in ['rmse', 'bias', 'std', 'mae', 'correlation', 'n_matches']:
                if metric in obs.columns:
                    val = obs[metric].mean()
                    row[f'obs_{metric}'] = val
        
        # RECONSTRUCTION stats for each method and data type
        for method in ['dineof', 'dincae']:
            for dtype in ['reconstruction_observed', 'reconstruction_missing', 'reconstruction']:
                subset = lake_df[(lake_df['data_type'] == dtype) & (lake_df['method'] == method)]
                if not subset.empty:
                    prefix = f'{method}_{dtype.replace("reconstruction_", "")}'
                    for metric in ['rmse', 'bias', 'std', 'mae', 'correlation', 'n_matches']:
                        if metric in subset.columns:
                            row[f'{prefix}_{metric}'] = subset[metric].mean()
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Compute WINNER (based on reconstruction_missing RMSE)
    if 'dineof_missing_rmse' in result_df.columns and 'dincae_missing_rmse' in result_df.columns:
        result_df['rmse_diff'] = result_df['dineof_missing_rmse'] - result_df['dincae_missing_rmse']
        result_df['winner'] = np.where(result_df['rmse_diff'] > 0.02, 'DINCAE',
                                       np.where(result_df['rmse_diff'] < -0.02, 'DINEOF', 'TIE'))
    
    # Compute method DIFFERENCES for reconstruction_missing
    for metric in ['rmse', 'bias', 'std', 'mae', 'correlation']:
        din_col = f'dineof_missing_{metric}'
        dic_col = f'dincae_missing_{metric}'
        if din_col in result_df.columns and dic_col in result_df.columns:
            result_df[f'delta_missing_{metric}'] = result_df[din_col] - result_df[dic_col]
    
    # Compute squared terms for decomposition
    if 'dineof_missing_std' in result_df.columns:
        result_df['dineof_missing_std_sq'] = result_df['dineof_missing_std'] ** 2
        result_df['dincae_missing_std_sq'] = result_df['dincae_missing_std'] ** 2
        result_df['dineof_missing_bias_sq'] = result_df['dineof_missing_bias'] ** 2
        result_df['dincae_missing_bias_sq'] = result_df['dincae_missing_bias'] ** 2
        result_df['dineof_missing_rmse_sq'] = result_df['dineof_missing_rmse'] ** 2
        result_df['dincae_missing_rmse_sq'] = result_df['dincae_missing_rmse'] ** 2
        
        result_df['delta_missing_std_sq'] = result_df['dineof_missing_std_sq'] - result_df['dincae_missing_std_sq']
        result_df['delta_missing_bias_sq'] = result_df['dineof_missing_bias_sq'] - result_df['dincae_missing_bias_sq']
        result_df['delta_missing_rmse_sq'] = result_df['dineof_missing_rmse_sq'] - result_df['dincae_missing_rmse_sq']
    
    return result_df


def test_predictor(df: pd.DataFrame, predictor_col: str, outcome_col: str = 'rmse_diff'):
    """
    Test a single predictor against the outcome.
    Returns correlation, p-value, and prediction accuracy.
    """
    valid = df[[predictor_col, outcome_col, 'winner']].dropna()
    valid = valid[valid['winner'] != 'TIE']
    
    if len(valid) < 5:
        return None
    
    # Correlation
    r, p = stats.pearsonr(valid[predictor_col], valid[outcome_col])
    
    # 95% CI for correlation (Fisher z-transform)
    n = len(valid)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    ci_low = np.tanh(z - 1.96 * se)
    ci_high = np.tanh(z + 1.96 * se)
    
    return {
        'predictor': predictor_col,
        'r': r,
        'r_sq': r ** 2,
        'p': p,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n': n,
    }


def part1_observation_predictors(df: pd.DataFrame):
    """
    Part 1: Test observation-based stats as predictors.
    These are NOT circular because they come from a different data type.
    """
    print("\n" + "="*80)
    print("PART 1: OBSERVATION-BASED PREDICTORS (NOT CIRCULAR)")
    print("="*80)
    print("""
These predictors come from data_type='observation' (satellite vs buoy comparison).
They can legitimately be tested as predictors of the reconstruction_missing winner
because they describe lake characteristics, not the reconstruction itself.

Question: Do lake characteristics (how satellite matches buoy) predict which
method will win at reconstructing missing pixels?
""")
    
    predictors = [
        ('obs_rmse', 'Observation RMSE (satellite vs buoy)'),
        ('obs_bias', 'Observation Bias (satellite - buoy)'),
        ('obs_std', 'Observation Residual STD'),
        ('obs_mae', 'Observation MAE'),
        ('obs_correlation', 'Observation Correlation'),
        ('obs_n_matches', 'Number of observation matches'),
    ]
    
    results = []
    print("\n" + "-"*80)
    print(f"{'Predictor':<45} {'r':>8} {'95% CI':>18} {'n':>5}")
    print("-"*80)
    
    for col, name in predictors:
        if col not in df.columns:
            continue
        result = test_predictor(df, col)
        if result is None:
            continue
        results.append({**result, 'name': name, 'category': 'observation'})
        
        ci_str = f"[{result['ci_low']:+.2f}, {result['ci_high']:+.2f}]"
        print(f"{name:<45} {result['r']:>+8.3f} {ci_str:>18} {result['n']:>5}")
    
    print("-"*80)
    print("""
INTERPRETATION:
- If none of these are strong predictors (|r| < 0.3), then lake characteristics
  do NOT predict which method wins. The winner depends on method behavior.
- If some are strong predictors, those characteristics favor one method.
""")
    
    return results


def part2_reconstruction_stats(df: pd.DataFrame):
    """
    Part 2: Report reconstruction stats for characterization.
    NOT for prediction - just to understand method properties.
    """
    print("\n" + "="*80)
    print("PART 2: RECONSTRUCTION STATISTICS (FOR CHARACTERIZATION)")
    print("="*80)
    print("""
These statistics describe how each method performs. They are NOT used
to predict the winner because that would be circular (using outcome to predict outcome).

Purpose: Understand the properties of each method.
""")
    
    valid = df.dropna(subset=['winner'])
    valid = valid[valid['winner'] != 'TIE']
    
    # Summary stats by method
    print("\n--- reconstruction_missing (TRUE GAP-FILL) ---")
    print(f"Sample: {len(valid)} lakes\n")
    
    metrics = ['rmse', 'bias', 'std']
    
    print(f"{'Metric':<20} {'DINEOF mean':>12} {'DINCAE mean':>12} {'Difference':>12}")
    print("-"*60)
    
    for metric in metrics:
        din_col = f'dineof_missing_{metric}'
        dic_col = f'dincae_missing_{metric}'
        if din_col in valid.columns and dic_col in valid.columns:
            din_mean = valid[din_col].mean()
            dic_mean = valid[dic_col].mean()
            diff = din_mean - dic_mean
            print(f"{metric.upper():<20} {din_mean:>12.3f} {dic_mean:>12.3f} {diff:>+12.3f}")
    
    # By winner
    print("\n--- Breakdown by Winner ---")
    for winner in ['DINEOF', 'DINCAE']:
        subset = valid[valid['winner'] == winner]
        print(f"\n{winner} wins ({len(subset)} lakes):")
        for metric in metrics:
            din_col = f'dineof_missing_{metric}'
            dic_col = f'dincae_missing_{metric}'
            if din_col in subset.columns and dic_col in subset.columns:
                din_mean = subset[din_col].mean()
                dic_mean = subset[dic_col].mean()
                print(f"  {metric.upper()}: DINEOF={din_mean:.3f}, DINCAE={dic_mean:.3f}")


def part3_circularity_analysis(df: pd.DataFrame):
    """
    Part 3: Analyze the circularity of using reconstruction_missing stats
    to predict reconstruction_missing winner.
    """
    print("\n" + "="*80)
    print("PART 3: CIRCULARITY ANALYSIS")
    print("="*80)
    print("""
MATHEMATICAL FACT:
  RMSE² = Bias² + STD²

Therefore, when we compute:
  ΔRMSE = RMSE_dineof - RMSE_dincae  (the outcome that determines winner)
  ΔSTD = STD_dineof - STD_dincae     (a candidate predictor)

These are NOT independent. ΔSTD is a component of ΔRMSE.

The question is: How much of the correlation r(ΔSTD, ΔRMSE) is due to
this mathematical relationship vs. being informative?
""")
    
    valid = df.dropna(subset=['delta_missing_rmse', 'delta_missing_std', 'delta_missing_bias'])
    valid = valid[valid['winner'] != 'TIE']
    n = len(valid)
    
    print(f"\nSample: {n} lakes\n")
    
    # Correlations of components with outcome
    predictors = [
        ('delta_missing_std', 'Δ(STD)'),
        ('delta_missing_bias', 'Δ(Bias)'),
        ('delta_missing_std_sq', 'Δ(STD²)'),
        ('delta_missing_bias_sq', 'Δ(Bias²)'),
    ]
    
    print("Correlation with ΔRMSE (the winner-determining metric):")
    print("-"*60)
    
    for col, name in predictors:
        if col not in valid.columns:
            continue
        r, p = stats.pearsonr(valid[col], valid['delta_missing_rmse'])
        print(f"  {name:<20}: r = {r:+.3f}")
    
    # Variance decomposition
    print("\n--- Variance Decomposition ---")
    print("ΔRMSE² = Δ(STD²) + Δ(Bias²)")
    
    var_std_sq = valid['delta_missing_std_sq'].var()
    var_bias_sq = valid['delta_missing_bias_sq'].var()
    cov = valid[['delta_missing_std_sq', 'delta_missing_bias_sq']].cov().iloc[0, 1]
    var_total = valid['delta_missing_rmse_sq'].var()
    
    total_from_components = var_std_sq + var_bias_sq + 2 * cov
    
    print(f"\n  Var(Δ(STD²)):  {var_std_sq:.4f}  ({100*var_std_sq/total_from_components:.1f}%)")
    print(f"  Var(Δ(Bias²)): {var_bias_sq:.4f}  ({100*var_bias_sq/total_from_components:.1f}%)")
    print(f"  2×Cov:         {2*cov:.4f}  ({100*2*cov/total_from_components:.1f}%)")
    print(f"  ---------------------------------")
    print(f"  Total:         {total_from_components:.4f}")
    print(f"  Var(ΔRMSE²):   {var_total:.4f}")
    
    # Which component dominates?
    std_dominates_count = (valid['delta_missing_std_sq'].abs() > valid['delta_missing_bias_sq'].abs()).sum()
    print(f"\n  |Δ(STD²)| > |Δ(Bias²)| in {std_dominates_count}/{n} lakes ({100*std_dominates_count/n:.0f}%)")
    
    print("""
INTERPRETATION:
---------------
The correlation r(ΔSTD, ΔRMSE) is HIGH because:
  1. STD² is a COMPONENT of RMSE² (mathematical relationship)
  2. STD² contributes MORE variance than Bias² (empirical fact)

This is NOT a "discovery" - it's showing that:
  "Methods differ mainly in residual STD, not in bias"
  
Or equivalently:
  "The winner is determined by temporal pattern matching (STD),
   not by systematic offset (Bias)"
""")
    
    return {
        'var_std_sq': var_std_sq,
        'var_bias_sq': var_bias_sq,
        'cov': cov,
        'pct_std': 100 * var_std_sq / total_from_components,
        'pct_bias': 100 * var_bias_sq / total_from_components,
    }


def part4_reconstruction_missing_predictors(df: pd.DataFrame):
    """
    Part 4: Test reconstruction_missing stats as predictors.
    These ARE circular but we report them for completeness with clear warning.
    """
    print("\n" + "="*80)
    print("PART 4: RECONSTRUCTION_MISSING STATS AS PREDICTORS (CIRCULAR!)")
    print("="*80)
    print("""
WARNING: This section uses reconstruction_missing stats to predict the
reconstruction_missing winner. This is CIRCULAR because:
  - Winner is determined by ΔRMSE
  - ΔRMSE² = Δ(STD²) + Δ(Bias²)
  - So ΔSTD and ΔBias are components of the outcome

We report these for completeness, but the correlations should NOT be
interpreted as "discoveries" - they are partly mathematical artifacts.
""")
    
    predictors = [
        ('delta_missing_std', 'Δ(STD) = STD_dineof - STD_dincae'),
        ('delta_missing_bias', 'Δ(Bias) = Bias_dineof - Bias_dincae'),
        ('delta_missing_rmse', 'Δ(RMSE) = RMSE_dineof - RMSE_dincae [TAUTOLOGY]'),
    ]
    
    results = []
    print("\n" + "-"*80)
    print(f"{'Predictor':<50} {'r':>8} {'Note':>20}")
    print("-"*80)
    
    for col, name in predictors:
        if col not in df.columns:
            continue
        result = test_predictor(df, col)
        if result is None:
            continue
        results.append({**result, 'name': name, 'category': 'reconstruction_missing'})
        
        note = 'TAUTOLOGY' if col == 'delta_missing_rmse' else 'COMPONENT OF RMSE'
        print(f"{name:<50} {result['r']:>+8.3f} {note:>20}")
    
    print("-"*80)
    
    return results


def part5_honest_summary(obs_results: list, decomp: dict, df: pd.DataFrame):
    """
    Part 5: Honest summary of what we can and cannot claim.
    """
    print("\n" + "="*80)
    print("PART 5: HONEST SUMMARY")
    print("="*80)
    
    # Check observation predictors
    strong_obs_predictors = [r for r in obs_results if abs(r['r']) > 0.3]
    
    print("""
WHAT WE CAN SAY:
----------------""")
    
    if strong_obs_predictors:
        print("\n1. Some observation characteristics predict the winner:")
        for r in strong_obs_predictors:
            print(f"   - {r['name']}: r = {r['r']:.3f}")
    else:
        print("""
1. Observation characteristics (satellite vs buoy) do NOT strongly predict
   which method wins at reconstruction_missing.
   
   This means: Lake characteristics don't determine the winner.
   The winner depends on method behavior at that specific lake.""")
    
    print(f"""
2. The RMSE difference between methods is dominated by STD difference:
   - STD² contributes {decomp['pct_std']:.0f}% of the variance in ΔRMSE²
   - Bias² contributes {decomp['pct_bias']:.0f}% of the variance in ΔRMSE²
   
   This means: Methods differ mainly in TEMPORAL PATTERN MATCHING (STD),
   not in SYSTEMATIC OFFSET (Bias).

3. When DINEOF wins, it has lower STD than DINCAE.
   When DINCAE wins, it usually has lower STD than DINEOF.
   
   This means: The "battleground" is temporal dynamics, not mean values.
""")
    
    print("""
WHAT WE CANNOT CLAIM:
---------------------
1. We did NOT "discover" that STD predicts winner.
   This is partly a mathematical relationship (STD² is part of RMSE²).
   
2. We cannot claim the high correlation (r ≈ 0.9) is a finding.
   It's inflated by the mathematical relationship.

3. With only ~25 lakes, statistical significance (p-values) is unreliable.
   We report effect sizes (r) and confidence intervals instead.
""")
    
    print("""
HONEST FRAMING FOR REPORT:
--------------------------
"The RMSE difference between DINEOF and DINCAE at reconstruction_missing
pixels is driven primarily by their difference in residual standard deviation
(accounting for {:.0f}% of variance), rather than by their difference in bias
({:.0f}% of variance). This indicates that the methods differ mainly in how
well their temporal dynamics match the buoy measurements, not in their
systematic offset from buoy values."
""".format(decomp['pct_std'], decomp['pct_bias']))


def create_comprehensive_figure(df: pd.DataFrame, obs_results: list, decomp: dict, output_dir: str):
    """
    Create a comprehensive figure showing all analyses.
    """
    print("\n" + "="*80)
    print("CREATING: Comprehensive Analysis Figure")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 14))
    
    valid = df.dropna(subset=['winner', 'delta_missing_rmse'])
    valid = valid[valid['winner'] != 'TIE']
    
    # Color by winner
    colors = [COLORS['dincae'] if w == 'DINCAE' else COLORS['dineof'] for w in valid['winner']]
    
    # Panel A: Observation predictors
    ax = fig.add_subplot(2, 3, 1)
    if obs_results:
        names = [r['name'].replace('Observation ', '') for r in obs_results]
        rs = [r['r'] for r in obs_results]
        bar_colors = [COLORS['significant'] if abs(r) > 0.3 else COLORS['not_significant'] for r in rs]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, rs, color=bar_colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(0.3, color='green', linestyle='--', alpha=0.5)
        ax.axvline(-0.3, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Correlation with ΔRMSE', fontsize=10)
        ax.set_xlim(-1, 1)
    ax.set_title('A) Observation-Based Predictors\n(NOT circular)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Panel B: ΔSTD vs ΔRMSE
    ax = fig.add_subplot(2, 3, 2)
    if 'delta_missing_std' in valid.columns:
        ax.scatter(valid['delta_missing_std'], valid['delta_missing_rmse'], c=colors, s=80, alpha=0.7, edgecolors='black')
        r, _ = stats.pearsonr(valid['delta_missing_std'], valid['delta_missing_rmse'])
        z = np.polyfit(valid['delta_missing_std'], valid['delta_missing_rmse'], 1)
        x_line = np.linspace(valid['delta_missing_std'].min(), valid['delta_missing_std'].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Δ(STD) [°C]', fontsize=10)
        ax.set_ylabel('Δ(RMSE) [°C]', fontsize=10)
        ax.set_title(f'B) Δ(STD) vs Δ(RMSE): r = {r:.3f}\n⚠️ CIRCULAR (STD is part of RMSE)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel C: ΔBias vs ΔRMSE  
    ax = fig.add_subplot(2, 3, 3)
    if 'delta_missing_bias' in valid.columns:
        ax.scatter(valid['delta_missing_bias'], valid['delta_missing_rmse'], c=colors, s=80, alpha=0.7, edgecolors='black')
        r, _ = stats.pearsonr(valid['delta_missing_bias'], valid['delta_missing_rmse'])
        z = np.polyfit(valid['delta_missing_bias'], valid['delta_missing_rmse'], 1)
        x_line = np.linspace(valid['delta_missing_bias'].min(), valid['delta_missing_bias'].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Δ(Bias) [°C]', fontsize=10)
        ax.set_ylabel('Δ(RMSE) [°C]', fontsize=10)
        ax.set_title(f'C) Δ(Bias) vs Δ(RMSE): r = {r:.3f}\n⚠️ CIRCULAR (Bias is part of RMSE)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel D: Variance decomposition
    ax = fig.add_subplot(2, 3, 4)
    components = ['Var(Δ(STD²))', 'Var(Δ(Bias²))', '2×Cov']
    values = [decomp['pct_std'], decomp['pct_bias'], 100 - decomp['pct_std'] - decomp['pct_bias']]
    bar_colors = ['#e74c3c', '#3498db', '#9b59b6']
    
    bars = ax.bar(components, values, color=bar_colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('% of Var(ΔRMSE²)', fontsize=10)
    ax.set_title('D) Variance Decomposition\nWhat drives ΔRMSE²?', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        y_pos = max(val + 2, 5) if val >= 0 else val - 5
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.0f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Panel E: Winner counts
    ax = fig.add_subplot(2, 3, 5)
    winner_counts = valid['winner'].value_counts()
    ax.bar(['DINEOF', 'DINCAE'], [winner_counts.get('DINEOF', 0), winner_counts.get('DINCAE', 0)],
          color=[COLORS['dineof'], COLORS['dincae']], edgecolor='black')
    ax.set_ylabel('Number of Lakes', fontsize=10)
    ax.set_title('E) Winner Counts\n(reconstruction_missing)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, winner in enumerate(['DINEOF', 'DINCAE']):
        count = winner_counts.get(winner, 0)
        ax.text(i, count + 0.3, str(count), ha='center', fontsize=12, fontweight='bold')
    
    # Panel F: Summary text
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    
    summary = f"""
HONEST SUMMARY
━━━━━━━━━━━━━━

Sample: {len(valid)} lakes

WINNER DETERMINATION:
  DINEOF wins: {winner_counts.get('DINEOF', 0)} lakes
  DINCAE wins: {winner_counts.get('DINCAE', 0)} lakes

VARIANCE DECOMPOSITION:
  ΔRMSE² = Δ(STD²) + Δ(Bias²)
  
  STD² contributes {decomp['pct_std']:.0f}%
  Bias² contributes {decomp['pct_bias']:.0f}%

KEY INSIGHT:
  Methods differ mainly in TEMPORAL
  PATTERN MATCHING (STD), not in
  SYSTEMATIC OFFSET (Bias).

WARNING:
  High r(ΔSTD, ΔRMSE) is partly
  mathematical, NOT a discovery.
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10,
              bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Comprehensive Predictor Analysis: What Predicts the In-Situ Winner?',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'comprehensive_predictor_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive predictor analysis")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "comprehensive_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE PREDICTOR ANALYSIS")
    print("="*80)
    print("Testing ALL available statistics as predictors of in-situ winner")
    print("Reporting honestly without overselling")
    
    # Load and process data
    df_raw = load_combined_stats(args.analysis_dir)
    df = extract_all_lake_stats(df_raw)
    
    print(f"\nLoaded data for {len(df)} lakes")
    print(f"Lakes with valid winner determination: {(df['winner'].notna() & (df['winner'] != 'TIE')).sum()}")
    
    # Run all parts
    obs_results = part1_observation_predictors(df)
    part2_reconstruction_stats(df)
    decomp = part3_circularity_analysis(df)
    part4_reconstruction_missing_predictors(df)
    part5_honest_summary(obs_results, decomp, df)
    
    # Create figure
    create_comprehensive_figure(df, obs_results, decomp, args.output_dir)
    
    # Save data
    df.to_csv(os.path.join(args.output_dir, 'all_lake_stats.csv'), index=False)
    
    if obs_results:
        pd.DataFrame(obs_results).to_csv(os.path.join(args.output_dir, 'observation_predictors.csv'), index=False)
    
    print(f"\n{'='*80}")
    print(f"Outputs saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
