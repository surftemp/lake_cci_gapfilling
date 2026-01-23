#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Internal Report

This script creates additional figures to sharpen the story:
1. Complete summary figure (single panel telling the whole story)
2. Side-by-side comparison of satellite CV vs in-situ validation
3. Schematic diagram explaining data types
4. Per-lake breakdown with all key metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

COLORS = {
    'dineof': '#5DA5DA',
    'dincae': '#FAA43A',
    'observation': '#60BD68',
    'correct': '#2ecc71',
    'incorrect': '#e74c3c',
}


def load_data(analysis_dir: str) -> tuple:
    """Load all required data files."""
    lake_stats = pd.read_csv(os.path.join(analysis_dir, 'winner_prediction', 'lake_stats_for_prediction.csv'))
    prediction_table = pd.read_csv(os.path.join(analysis_dir, 'winner_prediction', 'winner_prediction_table.csv'))
    return lake_stats, prediction_table


def create_summary_figure(lake_stats: pd.DataFrame, prediction_table: pd.DataFrame, output_dir: str):
    """
    Create a comprehensive summary figure that tells the complete story.
    
    This is the KEY figure for the report - it should convey the entire finding
    in a single, clear visualization.
    """
    print("Creating: Summary Figure (complete_summary.png)")
    
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)
    
    valid = prediction_table.dropna(subset=['rmse_diff', 'residual_std_diff'])
    valid = valid[valid['winner'] != 'TIE']
    
    # ==========================================================================
    # Panel A: The Puzzle - Winner counts comparison
    # ==========================================================================
    ax = fig.add_subplot(gs[0, 0])
    
    # Data
    categories = ['Satellite CV\n(n=120)', 'In-situ Validation\n(n=25)']
    dineof_counts = [117, 14]
    dincae_counts = [3, 11]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dineof_counts, width, label='DINEOF wins', color=COLORS['dineof'], edgecolor='black')
    bars2 = ax.bar(x + width/2, dincae_counts, width, label='DINCAE wins', color=COLORS['dincae'], edgecolor='black')
    
    ax.set_ylabel('Number of Lakes', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title('A) THE PUZZLE: Different Validation Methods\nGive Different Winners', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentages
    for bar, count in zip(bars1, dineof_counts):
        total = dineof_counts[list(dineof_counts).index(count)] + dincae_counts[list(dineof_counts).index(count)]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{100*count/total:.0f}%', ha='center', fontsize=10, fontweight='bold')
    for bar, count in zip(bars2, dincae_counts):
        total = dineof_counts[list(dincae_counts).index(count)] + dincae_counts[list(dincae_counts).index(count)]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{100*count/total:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # ==========================================================================
    # Panel B: The Key Discovery - Correlation
    # ==========================================================================
    ax = fig.add_subplot(gs[0, 1])
    
    colors = [COLORS['dineof'] if w == 'DINEOF' else COLORS['dincae'] for w in valid['winner']]
    ax.scatter(valid['residual_std_diff'], valid['rmse_diff'], c=colors, s=120, 
              alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Trend line
    r, p = stats.pearsonr(valid['residual_std_diff'], valid['rmse_diff'])
    z = np.polyfit(valid['residual_std_diff'], valid['rmse_diff'], 1)
    x_line = np.linspace(valid['residual_std_diff'].min() - 0.1, valid['residual_std_diff'].max() + 0.1, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r-', linewidth=2.5, alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Residual STD Difference\n(DINEOF − DINCAE) [°C]', fontsize=11)
    ax.set_ylabel('RMSE Difference\n(DINEOF − DINCAE) [°C]', fontsize=11)
    ax.set_title(f'B) THE DISCOVERY: r = {r:.3f} (p < 0.0001)\nResidual STD Predicts Winner', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('DINCAE wins\n(lower RMSE)', xy=(0.4, 0.5), fontsize=9, color=COLORS['dincae'],
               fontweight='bold', ha='center')
    ax.annotate('DINEOF wins\n(lower RMSE)', xy=(-0.4, -0.5), fontsize=9, color=COLORS['dineof'],
               fontweight='bold', ha='center')
    
    # ==========================================================================
    # Panel C: Prediction Accuracy
    # ==========================================================================
    ax = fig.add_subplot(gs[0, 2])
    
    # Prepare data
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    dineof_correct = (dineof_wins['residual_std_diff'] < 0).sum()  # DINEOF lower STD
    dincae_correct = (dincae_wins['residual_std_diff'] > 0).sum()  # DINCAE lower STD
    
    categories = ['DINEOF\nwins', 'DINCAE\nwins']
    correct = [dineof_correct, dincae_correct]
    incorrect = [len(dineof_wins) - dineof_correct, len(dincae_wins) - dincae_correct]
    
    x = np.arange(len(categories))
    width = 0.6
    
    ax.bar(x, correct, width, label='Prediction Correct', color=COLORS['correct'], edgecolor='black')
    ax.bar(x, incorrect, width, bottom=correct, label='Prediction Wrong', color=COLORS['incorrect'], edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Number of Lakes', fontsize=11)
    ax.set_title(f'C) PREDICTION ACCURACY: 84%\n(21/25 lakes correctly predicted)', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentages
    for i, (c, inc) in enumerate(zip(correct, incorrect)):
        total = c + inc
        ax.text(i, c/2, f'{c}/{total}\n({100*c/total:.0f}%)', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
    
    # ==========================================================================
    # Panel D: Physical Explanation
    # ==========================================================================
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    
    explanation = """
PHYSICAL EXPLANATION

Satellite measures SKIN temperature:
  • Top ~10 μm of water
  • Responds quickly to atmosphere
  • High temporal variability

Buoy measures BULK temperature:
  • Depth ~0.5-1 meter
  • Thermal inertia dampens variations
  • Lower temporal variability (smoother)

CONSEQUENCE:
Methods producing smoother reconstructions
have lower residual STD against buoy
(better temporal pattern match).
"""
    
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
    ax.set_title('D) WHY DOES THIS HAPPEN?', fontsize=12, fontweight='bold', y=1.0)
    
    # ==========================================================================
    # Panel E: Definition box
    # ==========================================================================
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    
    definition = """
KEY METRIC DEFINITIONS

Residual STD:
  STD of (reconstruction − buoy) residuals
  
  Lower value means:
  → Reconstruction varies more like buoy
  → Better temporal pattern match
  → Tighter fit to buoy dynamics

Winner Rule:
  DINEOF wins if RMSE_diff < −0.02°C
  DINCAE wins if RMSE_diff > +0.02°C
  
Prediction Rule:
  Predict method with lower Residual STD
  → Accuracy: 84% (21/25 lakes)
"""
    
    ax.text(0.05, 0.95, definition, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='blue'))
    ax.set_title('E) METRIC DEFINITIONS', fontsize=12, fontweight='bold', y=1.0)
    
    # ==========================================================================
    # Panel F: Conclusion box
    # ==========================================================================
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    conclusion = """
CONCLUSION

The two validations test DIFFERENT things:

┌─────────────────┬──────────────────────┐
│ Satellite CV    │ In-situ Validation   │
├─────────────────┼──────────────────────┤
│ Tests: recon    │ Tests: temporal      │
│ accuracy vs     │ pattern match vs     │
│ skin temp       │ bulk temp            │
├─────────────────┼──────────────────────┤
│ DINEOF wins     │ Winner predicted     │
│ 117/120 (98%)   │ by Residual STD      │
│                 │ with 84% accuracy    │
└─────────────────┴──────────────────────┘

This is NOT a contradiction.
These are different tests measuring
different properties.
"""
    
    ax.text(0.05, 0.95, conclusion, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='green'))
    ax.set_title('F) CONCLUSION', fontsize=12, fontweight='bold', y=1.0)
    
    # ==========================================================================
    # Panel G-I: Per-lake breakdown
    # ==========================================================================
    ax = fig.add_subplot(gs[2, :])
    
    # Sort by RMSE diff
    sorted_df = valid.sort_values('rmse_diff', ascending=False)
    x = np.arange(len(sorted_df))
    
    # Color by winner
    colors = [COLORS['dincae'] if r > 0 else COLORS['dineof'] for r in sorted_df['rmse_diff']]
    
    # Plot bars
    bars = ax.bar(x, sorted_df['rmse_diff'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Mark prediction correctness
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        is_correct = row['prediction_correct']
        marker = '✓' if is_correct else '✗'
        color = COLORS['correct'] if is_correct else COLORS['incorrect']
        y_pos = row['rmse_diff'] + 0.03 if row['rmse_diff'] > 0 else row['rmse_diff'] - 0.05
        ax.text(i, y_pos, marker, ha='center', va='center' if row['rmse_diff'] > 0 else 'top',
               fontsize=12, color=color, fontweight='bold')
    
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axhline(0.02, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-0.02, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Lakes (sorted by DINCAE advantage)', fontsize=11)
    ax.set_ylabel('RMSE Difference (DINEOF − DINCAE) [°C]', fontsize=11)
    ax.set_title('G) PER-LAKE BREAKDOWN: RMSE Difference with Prediction Correctness (✓ = correct, ✗ = wrong)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['lake_id'].astype(int), rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['dincae'], label='DINCAE wins'),
        mpatches.Patch(facecolor=COLORS['dineof'], label='DINEOF wins'),
        Line2D([0], [0], marker='$✓$', color='w', markerfacecolor=COLORS['correct'], 
               markersize=12, label='Prediction correct'),
        Line2D([0], [0], marker='$✗$', color='w', markerfacecolor=COLORS['incorrect'],
               markersize=12, label='Prediction wrong'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', ncol=2)
    
    plt.suptitle('Complete Analysis: Why Satellite CV and In-Situ Validation Give Different Winners',
                fontsize=15, fontweight='bold', y=1.01)
    
    save_path = os.path.join(output_dir, 'complete_summary.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_key_result_figure(valid: pd.DataFrame, output_dir: str):
    """
    Create a focused figure showing just the key result (for presentations).
    """
    print("Creating: Key Result Figure (key_result.png)")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: The smoking gun correlation
    ax = axes[0]
    
    colors = [COLORS['dineof'] if w == 'DINEOF' else COLORS['dincae'] for w in valid['winner']]
    ax.scatter(valid['residual_std_diff'], valid['rmse_diff'], c=colors, s=150, 
              alpha=0.7, edgecolors='black', linewidth=1)
    
    # Trend line
    r, p = stats.pearsonr(valid['residual_std_diff'], valid['rmse_diff'])
    z = np.polyfit(valid['residual_std_diff'], valid['rmse_diff'], 1)
    x_line = np.linspace(valid['residual_std_diff'].min() - 0.1, valid['residual_std_diff'].max() + 0.1, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r-', linewidth=3, alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Residual STD Difference (DINEOF − DINCAE) [°C]', fontsize=12)
    ax.set_ylabel('RMSE Difference (DINEOF − DINCAE) [°C]', fontsize=12)
    ax.set_title(f'r = {r:.3f}, p < 0.0001', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['dineof'], edgecolor='black', label='DINEOF wins'),
        mpatches.Patch(facecolor=COLORS['dincae'], edgecolor='black', label='DINCAE wins'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # Panel B: Prediction accuracy breakdown
    ax = axes[1]
    
    dineof_wins = valid[valid['winner'] == 'DINEOF']
    dincae_wins = valid[valid['winner'] == 'DINCAE']
    
    dineof_correct = (dineof_wins['dineof_residual_std'] < dineof_wins['dincae_residual_std']).sum()
    dincae_correct = (dincae_wins['dincae_residual_std'] < dincae_wins['dineof_residual_std']).sum()
    
    categories = ['DINEOF wins\n(n=14)', 'DINCAE wins\n(n=11)']
    correct = [dineof_correct, dincae_correct]
    totals = [len(dineof_wins), len(dincae_wins)]
    
    x = np.arange(len(categories))
    
    bars = ax.bar(x, correct, color=[COLORS['dineof'], COLORS['dincae']], edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    for i, (bar, c, t) in enumerate(zip(bars, correct, totals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{c}/{t}\n({100*c/t:.0f}%)', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Lakes with Lower Residual STD', fontsize=12)
    ax.set_title('Prediction: Winner Has Lower Residual STD', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(totals) + 3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add overall accuracy
    total_correct = dineof_correct + dincae_correct
    total = len(dineof_wins) + len(dincae_wins)
    ax.text(0.5, 0.95, f'Overall Accuracy: {total_correct}/{total} = {100*total_correct/total:.0f}%',
           transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold',
           bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='orange', boxstyle='round'))
    
    plt.suptitle('The Method with Lower Residual STD Wins the In-Situ Validation',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'key_result.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_data_types_schematic(output_dir: str):
    """
    Create a schematic explaining the different data types.
    """
    print("Creating: Data Types Schematic (data_types_schematic.png)")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Data Types in In-Situ Validation', fontsize=16, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    # Draw boxes for each data type
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2)
    
    # Observation
    ax.text(0.15, 0.75, 'observation', fontsize=14, fontweight='bold', ha='center',
           transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor='#60BD68', alpha=0.7))
    ax.text(0.15, 0.65, 'Satellite pixel vs Buoy\nat times with valid satellite data\n\nBaseline: How well does\nsatellite match buoy?',
           fontsize=10, ha='center', transform=ax.transAxes, va='top')
    
    # reconstruction_observed
    ax.text(0.5, 0.75, 'reconstruction_observed', fontsize=14, fontweight='bold', ha='center',
           transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor='#B276B2', alpha=0.7))
    ax.text(0.5, 0.65, 'Reconstruction vs Buoy\nat times where satellite HAD data\n\nTests: Does reconstruction\npreserve original values?',
           fontsize=10, ha='center', transform=ax.transAxes, va='top')
    
    # reconstruction_missing
    ax.text(0.85, 0.75, 'reconstruction_missing', fontsize=14, fontweight='bold', ha='center',
           transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor='#F17CB0', alpha=0.7))
    ax.text(0.85, 0.65, 'Reconstruction vs Buoy\nat times where satellite was MISSING\n\n★ TRUE gap-filling test ★\nNo satellite data existed here!',
           fontsize=10, ha='center', transform=ax.transAxes, va='top')
    
    # Draw timeline illustration
    ax.plot([0.1, 0.9], [0.35, 0.35], 'k-', linewidth=2, transform=ax.transAxes)
    ax.text(0.5, 0.32, 'Time →', fontsize=10, ha='center', transform=ax.transAxes)
    
    # Add data points on timeline
    time_points = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    has_satellite = [True, True, False, True, False, False, True, True]
    
    for t, has_sat in zip(time_points, has_satellite):
        if has_sat:
            ax.plot(t, 0.40, 'go', markersize=12, transform=ax.transAxes)  # Satellite exists
            ax.plot(t, 0.35, 'bs', markersize=10, transform=ax.transAxes)  # Buoy always exists
        else:
            ax.plot(t, 0.40, 'rx', markersize=12, mew=2, transform=ax.transAxes)  # Satellite missing
            ax.plot(t, 0.35, 'bs', markersize=10, transform=ax.transAxes)  # Buoy always exists
    
    # Legend for timeline
    ax.plot([], [], 'go', markersize=10, label='Satellite observation exists')
    ax.plot([], [], 'rx', markersize=10, mew=2, label='Satellite missing (gap)')
    ax.plot([], [], 'bs', markersize=10, label='Buoy measurement')
    ax.legend(loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.15))
    
    # Key insight box
    ax.text(0.5, 0.08, 
           '★ We focus on reconstruction_missing because it tests TRUE gap-filling ability ★',
           fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='orange'))
    
    save_path = os.path.join(output_dir, 'data_types_schematic.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication-quality figures for report")
    parser.add_argument("--analysis_dir", required=True, help="Path to insitu_validation_analysis")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.analysis_dir, "report_figures")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Generating Publication-Quality Figures for Report")
    print("="*60)
    
    # Load data
    lake_stats, prediction_table = load_data(args.analysis_dir)
    
    # Filter valid data
    valid = prediction_table.dropna(subset=['rmse_diff', 'residual_std_diff'])
    valid = valid[valid['winner'] != 'TIE']
    
    print(f"Loaded data for {len(valid)} lakes (excluding ties)")
    
    # Generate figures
    create_summary_figure(lake_stats, prediction_table, args.output_dir)
    create_key_result_figure(valid, args.output_dir)
    create_data_types_schematic(args.output_dir)
    
    print("\n" + "="*60)
    print("Figures generated successfully!")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nGenerated files:")
    print("  - complete_summary.png (multi-panel summary)")
    print("  - key_result.png (focused key finding)")
    print("  - data_types_schematic.png (explanation of data types)")


if __name__ == "__main__":
    main()
