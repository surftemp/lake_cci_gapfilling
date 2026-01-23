#!/usr/bin/env python3
"""
Temporal Variability Analysis: Does Matching Buoy's Temporal Behavior Predict Winner?

GOAL:
=====
For each lake, determine if the WINNER can be predicted by how closely
the reconstruction matches buoy's temporal behavior (smoothness).

HYPOTHESIS:
===========
Buoy (bulk temp) has lower temporal variability than satellite (skin temp).
If a reconstruction's temporal variability is closer to buoy's, it should
match buoy better and thus "win" the in-situ validation.

METRICS PER LAKE:
=================
1. satellite_temporal_std: Temporal STD of satellite at buoy pixel
2. buoy_temporal_std: Temporal STD of buoy measurements  
3. dineof_temporal_std: Temporal STD of DINEOF reconstruction
4. dincae_temporal_std: Temporal STD of DINCAE reconstruction

DERIVED PREDICTORS:
==================
- dineof_closeness_to_buoy = |dineof_temporal_std - buoy_temporal_std|
- dincae_closeness_to_buoy = |dincae_temporal_std - buoy_temporal_std|
- relative_closeness = dineof_closeness - dincae_closeness
  (positive = DINCAE closer to buoy temporal behavior)

PREDICTION:
===========
If relative_closeness > 0 (DINCAE closer to buoy), DINCAE should win.

This requires accessing the actual time series data from the validation CSVs
or the NetCDF files.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from glob import glob
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'observation': '#60BD68',
    'satellite': '#B276B2',
    'buoy': '#F15854',
    'dineof': '#5DA5DA', 
    'dincae': '#FAA43A',
}


def load_combined_stats(analysis_dir: str) -> pd.DataFrame:
    """Load the combined in-situ stats."""
    csv_path = os.path.join(analysis_dir, 'all_insitu_stats_combined.csv')
    return pd.read_csv(csv_path)


def load_detailed_validation_data(analysis_dir: str) -> dict:
    """
    Load the detailed per-lake validation CSVs that contain actual time series.
    
    These should have columns like: time, satellite_temp, buoy_temp, dineof_temp, dincae_temp
    """
    lake_data = {}
    
    # Look for detailed CSVs
    csv_pattern = os.path.join(analysis_dir, 'lake_*_validation.csv')
    csv_files = glob(csv_pattern)
    
    if not csv_files:
        # Try alternative patterns
        csv_pattern = os.path.join(analysis_dir, '**/LAKE*.csv')
        csv_files = glob(csv_pattern, recursive=True)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Extract lake ID from filename
            basename = os.path.basename(csv_file)
            lake_id = int(''.join(filter(str.isdigit, basename.split('_')[0])))
            lake_data[lake_id] = df
        except Exception as e:
            continue
    
    return lake_data


def compute_temporal_stats_from_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-lake temporal variability metrics from the combined stats.
    
    The 'std' column in the combined stats represents the STD of residuals,
    not the temporal STD of the time series. However, we can use the 
    observation std as a proxy for satellite temporal variability at buoy pixel.
    
    For a proper analysis, we need the raw time series data.
    """
    results = []
    
    for lake_id in df['lake_id_cci'].unique():
        lake_df = df[df['lake_id_cci'] == lake_id]
        row = {'lake_id': int(lake_id)}
        
        # Get observation (satellite vs buoy) stats
        obs = lake_df[lake_df['data_type'] == 'observation']
        if not obs.empty:
            # The 'std' here is the STD of (satellite - buoy) residuals
            # This is NOT the temporal STD of satellite or buoy
            row['obs_residual_std'] = obs['std'].mean()
            row['obs_rmse'] = obs['rmse'].mean()
            row['obs_bias'] = obs['bias'].mean()
            row['obs_n'] = obs['n_matches'].sum()
        
        # Get reconstruction stats at MISSING pixels
        for method in ['dineof', 'dincae']:
            miss = lake_df[(lake_df['data_type'] == 'reconstruction_missing') & 
                          (lake_df['method'] == method)]
            if not miss.empty:
                row[f'{method}_residual_std'] = miss['std'].mean()
                row[f'{method}_rmse'] = miss['rmse'].mean()
                row[f'{method}_bias'] = miss['bias'].mean()
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Compute winner
    if 'dineof_rmse' in result_df.columns and 'dincae_rmse' in result_df.columns:
        result_df['rmse_diff'] = result_df['dineof_rmse'] - result_df['dincae_rmse']
        result_df['winner'] = np.where(result_df['rmse_diff'] > 0.02, 'DINCAE',
                                       np.where(result_df['rmse_diff'] < -0.02, 'DINEOF', 'TIE'))
    
    # Compute derived metrics
    # Relative residual STD: positive = DINCAE has lower residual STD (smoother fit)
    if 'dineof_residual_std' in result_df.columns and 'dincae_residual_std' in result_df.columns:
        result_df['residual_std_diff'] = result_df['dineof_residual_std'] - result_df['dincae_residual_std']
    
    return result_df


def analyze_winner_predictors(lake_df: pd.DataFrame, output_dir: str):
    """
    Test which metrics best predict the winner.
    Focus on PER-LAKE prediction, not aggregate statistics.
    """
    print("\n" + "="*70)
    print("WINNER PREDICTION ANALYSIS")
    print("="*70)
    print("\nGoal: Find per-lake metrics that predict whether DINCAE or DINEOF wins")
    
    valid = lake_df.dropna(subset=['winner', 'rmse_diff'])
    valid = valid[valid['winner'] != 'TIE']  # Exclude ties for clearer analysis
    
    # Candidate predictors
    predictors = [
        ('obs_residual_std', 'Observation Residual STD'),
        ('obs_rmse', 'Observation RMSE'),
        ('obs_bias', 'Observation Bias'),
        ('residual_std_diff', 'Residual STD Diff (DINEOF - DINCAE)'),
        ('dineof_residual_std', 'DINEOF Residual STD'),
        ('dincae_residual_std', 'DINCAE Residual STD'),
        ('dineof_bias', 'DINEOF Bias'),
        ('dincae_bias', 'DINCAE Bias'),
    ]
    
    print("\n" + "-"*70)
    print("Correlation with RMSE Difference (positive = DINCAE wins)")
    print("-"*70)
    
    correlations = []
    
    for col, name in predictors:
        if col not in valid.columns:
            continue
        
        subset = valid.dropna(subset=[col])
        if len(subset) < 5:
            continue
        
        r, p = stats.pearsonr(subset[col], subset['rmse_diff'])
        correlations.append({
            'predictor': col,
            'name': name,
            'r': r,
            'p': p,
            'n': len(subset)
        })
        
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"  {name:<40} r={r:+.3f}  p={p:.4f} {sig}")
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['r']), reverse=True)
    
    print("\n" + "-"*70)
    print("TOP PREDICTORS (ranked by |r|)")
    print("-"*70)
    
    for corr in correlations[:5]:
        sig = '***' if corr['p'] < 0.01 else '**' if corr['p'] < 0.05 else '*' if corr['p'] < 0.1 else ''
        direction = "DINCAE wins when higher" if corr['r'] > 0 else "DINEOF wins when higher"
        print(f"  |r|={abs(corr['r']):.3f}  {corr['name']:<35} ({direction}) {sig}")
    
    # Test classification accuracy
    print("\n" + "-"*70)
    print("CLASSIFICATION ACCURACY (Can this metric predict winner?)")
    print("-"*70)
    
    for corr in correlations[:5]:
        col = corr['predictor']
        subset = valid.dropna(subset=[col])
        
        if len(subset) < 5:
            continue
        
        # Simple threshold: use median as cutoff
        median_val = subset[col].median()
        
        if corr['r'] > 0:
            # Higher value → DINCAE wins
            predicted = np.where(subset[col] > median_val, 'DINCAE', 'DINEOF')
        else:
            # Higher value → DINEOF wins
            predicted = np.where(subset[col] > median_val, 'DINEOF', 'DINCAE')
        
        accuracy = (predicted == subset['winner']).mean()
        print(f"  {corr['name']:<40} Accuracy: {accuracy:.1%}")
    
    return correlations


def analyze_residual_std_as_predictor(lake_df: pd.DataFrame, output_dir: str):
    """
    Detailed analysis of residual STD as predictor.
    
    Key insight: The STD in the stats is the STD of (method - buoy) residuals.
    Lower residual STD = tighter fit to buoy = method varies similarly to buoy.
    """
    print("\n" + "="*70)
    print("RESIDUAL STD ANALYSIS (Key Predictor)")
    print("="*70)
    
    valid = lake_df.dropna(subset=['winner', 'residual_std_diff', 
                                    'dineof_residual_std', 'dincae_residual_std'])
    valid = valid[valid['winner'] != 'TIE']
    
    print("\nResidual STD = STD of (reconstruction - buoy)")
    print("Lower residual STD = method varies more like buoy (tighter fit)")
    print()
    
    # By winner group
    for winner in ['DINEOF', 'DINCAE']:
        subset = valid[valid['winner'] == winner]
        print(f"{winner} wins ({len(subset)} lakes):")
        print(f"  DINEOF residual STD: {subset['dineof_residual_std'].mean():.3f}°C")
        print(f"  DINCAE residual STD: {subset['dincae_residual_std'].mean():.3f}°C")
        print(f"  Difference (D-C):    {subset['residual_std_diff'].mean():+.3f}°C")
        print()
    
    # The key question: Does lower residual STD predict winning?
    print("KEY TEST: Does the method with lower residual STD win?")
    
    # For each lake, check if the winner has lower residual STD
    valid['dineof_lower_std'] = valid['dineof_residual_std'] < valid['dincae_residual_std']
    
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    dineof_correct = (dineof_wins['dineof_lower_std']).sum()
    dincae_correct = (~dincae_wins['dineof_lower_std']).sum()
    
    total_correct = dineof_correct + dincae_correct
    total = len(dineof_wins) + len(dincae_wins)
    
    print(f"\n  When DINEOF wins: DINEOF has lower STD in {dineof_correct}/{len(dineof_wins)} cases ({100*dineof_correct/len(dineof_wins):.0f}%)")
    print(f"  When DINCAE wins: DINCAE has lower STD in {dincae_correct}/{len(dincae_wins)} cases ({100*dincae_correct/len(dincae_wins):.0f}%)")
    print(f"\n  OVERALL: Lower residual STD predicts winner in {total_correct}/{total} cases ({100*total_correct/total:.0f}%)")
    
    return total_correct / total


def create_winner_prediction_figure(lake_df: pd.DataFrame, correlations: list, output_dir: str):
    """
    Create figure showing top predictors of winner.
    """
    print("\n" + "="*70)
    print("CREATING: Winner Prediction Figure")
    print("="*70)
    
    valid = lake_df.dropna(subset=['winner', 'rmse_diff'])
    valid = valid[valid['winner'] != 'TIE']
    
    # Get top 4 predictors
    top_predictors = [c for c in correlations if c['predictor'] in valid.columns][:4]
    
    if len(top_predictors) < 2:
        print("Not enough predictors available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, corr in zip(axes, top_predictors):
        col = corr['predictor']
        subset = valid.dropna(subset=[col])
        
        # Color by winner
        colors = [COLORS['dineof'] if w == 'DINEOF' else COLORS['dincae'] 
                  for w in subset['winner']]
        
        ax.scatter(subset[col], subset['rmse_diff'], c=colors, s=100, 
                  alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Trend line
        z = np.polyfit(subset[col], subset['rmse_diff'], 1)
        x_line = np.linspace(subset[col].min(), subset[col].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2, alpha=0.7)
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(subset[col].median(), color='gray', linestyle=':', alpha=0.5)
        
        sig = '***' if corr['p'] < 0.01 else '**' if corr['p'] < 0.05 else '*' if corr['p'] < 0.1 else ''
        ax.set_xlabel(corr['name'], fontsize=11)
        ax.set_ylabel('RMSE Diff (DINEOF - DINCAE) [°C]', fontsize=10)
        ax.set_title(f"r = {corr['r']:.3f}, p = {corr['p']:.4f} {sig}", 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add interpretation
        if corr['r'] > 0:
            ax.text(0.02, 0.98, '↗ Higher → DINCAE wins', transform=ax.transAxes, 
                   fontsize=9, va='top', color='darkorange')
        else:
            ax.text(0.02, 0.98, '↗ Higher → DINEOF wins', transform=ax.transAxes,
                   fontsize=9, va='top', color='blue')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['dineof'], label='DINEOF wins'),
                       Patch(facecolor=COLORS['dincae'], label='DINCAE wins')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11,
              bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('What Predicts the Winner? (Per-Lake Analysis)\n'
                'Each point is one lake; color shows actual winner',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'winner_predictors.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_residual_std_figure(lake_df: pd.DataFrame, output_dir: str):
    """
    Create detailed figure for residual STD analysis.
    """
    print("\n" + "="*70)
    print("CREATING: Residual STD Figure")
    print("="*70)
    
    valid = lake_df.dropna(subset=['winner', 'dineof_residual_std', 'dincae_residual_std'])
    valid = valid[valid['winner'] != 'TIE']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Scatter DINEOF STD vs DINCAE STD
    ax = axes[0]
    colors = [COLORS['dineof'] if w == 'DINEOF' else COLORS['dincae'] 
              for w in valid['winner']]
    
    ax.scatter(valid['dineof_residual_std'], valid['dincae_residual_std'],
              c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Diagonal
    lims = [min(valid['dineof_residual_std'].min(), valid['dincae_residual_std'].min()) - 0.1,
            max(valid['dineof_residual_std'].max(), valid['dincae_residual_std'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal STD')
    
    ax.set_xlabel('DINEOF Residual STD [°C]', fontsize=11)
    ax.set_ylabel('DINCAE Residual STD [°C]', fontsize=11)
    ax.set_title('A) Residual STD Comparison\n(Below diagonal = DINCAE tighter fit)',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add counts per quadrant
    below_diag = (valid['dincae_residual_std'] < valid['dineof_residual_std']).sum()
    above_diag = len(valid) - below_diag
    ax.text(0.95, 0.05, f'DINCAE tighter: {below_diag}\nDINEOF tighter: {above_diag}',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Panel B: By winner - does lower STD predict winner?
    ax = axes[1]
    
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    # For each group, show which method had lower STD
    categories = ['DINEOF\nwins', 'DINCAE\nwins']
    dineof_lower = [(dineof_wins['dineof_residual_std'] < dineof_wins['dincae_residual_std']).sum(),
                    (dincae_wins['dineof_residual_std'] < dincae_wins['dincae_residual_std']).sum()]
    dincae_lower = [len(dineof_wins) - dineof_lower[0], len(dincae_wins) - dineof_lower[1]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, dineof_lower, width, label='DINEOF has lower STD', color=COLORS['dineof'], alpha=0.7)
    ax.bar(x + width/2, dincae_lower, width, label='DINCAE has lower STD', color=COLORS['dincae'], alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Number of Lakes', fontsize=11)
    ax.set_title('B) Does Lower Residual STD Predict Winner?\n(Winner should have lower STD)',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentages
    for i, (dl, dcl) in enumerate(zip(dineof_lower, dincae_lower)):
        total = dl + dcl
        ax.text(i - width/2, dl + 0.2, f'{100*dl/total:.0f}%', ha='center', fontsize=10)
        ax.text(i + width/2, dcl + 0.2, f'{100*dcl/total:.0f}%', ha='center', fontsize=10)
    
    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    
    # Compute accuracy
    correct_dineof = (dineof_wins['dineof_residual_std'] < dineof_wins['dincae_residual_std']).sum()
    correct_dincae = (dincae_wins['dincae_residual_std'] < dincae_wins['dineof_residual_std']).sum()
    total_correct = correct_dineof + correct_dincae
    total = len(dineof_wins) + len(dincae_wins)
    accuracy = total_correct / total
    
    summary = f"""
RESIDUAL STD AS WINNER PREDICTOR

Definition:
  Residual STD = STD of (reconstruction - buoy)
  Lower residual STD = tighter fit = varies more like buoy

Results:
  When DINEOF wins ({len(dineof_wins)} lakes):
    DINEOF has lower STD: {correct_dineof}/{len(dineof_wins)} ({100*correct_dineof/len(dineof_wins):.0f}%)
    
  When DINCAE wins ({len(dincae_wins)} lakes):
    DINCAE has lower STD: {correct_dincae}/{len(dincae_wins)} ({100*correct_dincae/len(dincae_wins):.0f}%)

OVERALL ACCURACY: {total_correct}/{total} ({accuracy:.1%})

Interpretation:
  The method with lower residual STD (tighter fit to buoy)
  wins {accuracy:.0%} of the time.
  
  This suggests the winner is determined by which method
  produces temporal patterns more similar to buoy.
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Residual STD Analysis: Does Tighter Fit to Buoy Predict Winner?',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'residual_std_predictor.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    
    return accuracy


def create_comprehensive_predictor_table(lake_df: pd.DataFrame, output_dir: str):
    """
    Create a comprehensive per-lake table showing all predictors and winner.
    """
    print("\n" + "="*70)
    print("CREATING: Comprehensive Per-Lake Table")
    print("="*70)
    
    valid = lake_df.dropna(subset=['winner'])
    
    # Select columns for display
    cols = ['lake_id', 'winner', 'rmse_diff',
            'dineof_rmse', 'dincae_rmse',
            'dineof_residual_std', 'dincae_residual_std', 'residual_std_diff',
            'obs_residual_std', 'obs_rmse', 'obs_bias']
    
    available_cols = [c for c in cols if c in valid.columns]
    
    # Sort by rmse_diff (DINCAE advantage)
    result = valid[available_cols].sort_values('rmse_diff', ascending=False)
    
    # Add prediction column
    if 'dineof_residual_std' in result.columns and 'dincae_residual_std' in result.columns:
        result['predicted_winner'] = np.where(
            result['dineof_residual_std'] > result['dincae_residual_std'], 
            'DINCAE', 'DINEOF')
        result['prediction_correct'] = result['winner'] == result['predicted_winner']
    
    # Save
    save_path = os.path.join(output_dir, 'winner_prediction_table.csv')
    result.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    
    # Print
    print("\nPer-Lake Winner Prediction (sorted by DINCAE advantage):")
    print("-" * 100)
    print(result.to_string(index=False))
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze what predicts the in-situ validation winner")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "winner_prediction")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("WINNER PREDICTION ANALYSIS")
    print("="*70)
    print("\nGoal: Find per-lake metrics that predict DINCAE vs DINEOF winner")
    print("="*70)
    
    # Load data
    df = load_combined_stats(args.analysis_dir)
    print(f"\nLoaded {len(df)} records from {df['lake_id_cci'].nunique()} lakes")
    
    # Compute per-lake stats
    lake_df = compute_temporal_stats_from_combined(df)
    print(f"Computed stats for {len(lake_df)} lakes")
    
    # Analyze predictors
    correlations = analyze_winner_predictors(lake_df, args.output_dir)
    accuracy = analyze_residual_std_as_predictor(lake_df, args.output_dir)
    
    # Create figures
    create_winner_prediction_figure(lake_df, correlations, args.output_dir)
    create_residual_std_figure(lake_df, args.output_dir)
    
    # Create comprehensive table
    result_table = create_comprehensive_predictor_table(lake_df, args.output_dir)
    
    # Save lake stats
    lake_df.to_csv(os.path.join(args.output_dir, 'lake_stats_for_prediction.csv'), index=False)
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
KEY FINDING:
  Residual STD (how tightly method fits buoy) predicts winner
  with {accuracy:.0%} accuracy.
  
  The method whose temporal pattern more closely matches buoy
  (lower residual STD) wins the in-situ validation.

INTERPRETATION:
  This supports the hypothesis that in-situ validation favors
  methods that produce temporal patterns similar to buoy (bulk temp),
  rather than methods that accurately reconstruct satellite (skin temp).
  
  Satellite CV and in-situ validation are testing DIFFERENT things.
""")
    
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
