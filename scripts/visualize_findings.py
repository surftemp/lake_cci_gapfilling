#!/usr/bin/env python3
"""
Visualization: DINEOF vs DINCAE Cross-Validation Comparison
============================================================

Creates clear, presentation-ready plots to communicate findings:

1. Shore distance distribution by winner
2. The key finding: Same pixel, different ground truth → different winner
3. Summary of all validation approaches

Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Style settings for clean, professional plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


def load_data(run_root: str):
    """Load all relevant data files."""
    data = {}
    
    # Satellite vs In-situ CV comparison
    cv_path = os.path.join(run_root, "satellite_vs_insitu_cv_comparison.csv")
    if os.path.exists(cv_path):
        data['cv_comparison'] = pd.read_csv(cv_path)
    
    # Spatial analysis (for all lakes)
    spatial_path = os.path.join(run_root, "spatial_analysis", "spatial_analysis_data.csv")
    if os.path.exists(spatial_path):
        data['spatial'] = pd.read_csv(spatial_path)
    
    # In-situ vs satellite amplitude comparison
    amp_path = os.path.join(run_root, "insitu_vs_satellite_comparison", "insitu_vs_satellite_data.csv")
    if os.path.exists(amp_path):
        data['amplitude'] = pd.read_csv(amp_path)
    
    return data


def plot_shore_distance_distribution(df: pd.DataFrame, output_dir: str):
    """
    Plot 1: Buoy distance from shore - distribution by winner
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Get data
    all_distances = df['buoy_distance_from_shore'].dropna()
    dineof_wins = df[df['insitu_winner'] == 'dineof']['buoy_distance_from_shore'].dropna()
    dincae_wins = df[df['insitu_winner'] == 'dincae']['buoy_distance_from_shore'].dropna()
    
    # Common bin edges for consistency
    max_dist = all_distances.max()
    bins = np.arange(0, max_dist + 3, 2)
    
    # A) All lakes
    ax = axes[0]
    ax.hist(all_distances, bins=bins, color='#7f8c8d', edgecolor='black', alpha=0.7)
    ax.axvline(all_distances.median(), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {all_distances.median():.1f}px')
    ax.set_xlabel('Distance from shore (pixels)')
    ax.set_ylabel('Number of lakes')
    ax.set_title(f'A) All In-situ Validation Lakes\n(N={len(all_distances)})')
    ax.legend()
    
    # B) DINEOF wins
    ax = axes[1]
    ax.hist(dineof_wins, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
    if len(dineof_wins) > 0:
        ax.axvline(dineof_wins.median(), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {dineof_wins.median():.1f}px')
    ax.set_xlabel('Distance from shore (pixels)')
    ax.set_ylabel('Number of lakes')
    ax.set_title(f'B) Lakes where DINEOF wins In-situ CV\n(N={len(dineof_wins)})')
    ax.legend()
    
    # C) DINCAE wins
    ax = axes[2]
    ax.hist(dincae_wins, bins=bins, color='#e74c3c', edgecolor='black', alpha=0.7)
    if len(dincae_wins) > 0:
        ax.axvline(dincae_wins.median(), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {dincae_wins.median():.1f}px')
    ax.set_xlabel('Distance from shore (pixels)')
    ax.set_ylabel('Number of lakes')
    ax.set_title(f'C) Lakes where DINCAE wins In-situ CV\n(N={len(dincae_wins)})')
    ax.legend()
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'shore_distance_by_winner.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_key_finding(df: pd.DataFrame, output_dir: str):
    """
    Plot 2: THE KEY FINDING - Same pixel, different ground truth → different winner
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # A) Bar chart: Win rates by CV type
    ax = axes[0]
    
    # Count winners
    n_total = len(df)
    sat_dineof = (df['sat_winner'] == 'dineof').sum()
    sat_valid = df['sat_winner'].notna().sum()
    insitu_dineof = (df['insitu_winner'] == 'dineof').sum()
    
    categories = ['Satellite CV\n(at buoy pixel)', 'In-situ CV\n(at buoy pixel)']
    dineof_pct = [100 * sat_dineof / sat_valid if sat_valid > 0 else 0, 
                  100 * insitu_dineof / n_total]
    dincae_pct = [100 - dineof_pct[0], 100 - dineof_pct[1]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dineof_pct, width, label='DINEOF wins', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, dincae_pct, width, label='DINCAE wins', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Win rate (%)')
    ax.set_title('A) Same Pixel, Different Ground Truth\n→ Different Winner')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 110)
    
    # Add value labels
    for bar, pct in zip(bars1, dineof_pct):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 2, f'{pct:.0f}%', 
                ha='center', va='bottom', fontweight='bold')
    for bar, pct in zip(bars2, dincae_pct):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 2, f'{pct:.0f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # B) Sankey-style flow diagram (simplified as arrow diagram)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('B) What Happens at Each Lake?')
    
    # Count the flows
    both_dineof = ((df['sat_winner'] == 'dineof') & (df['insitu_winner'] == 'dineof')).sum()
    sat_dineof_insitu_dincae = ((df['sat_winner'] == 'dineof') & (df['insitu_winner'] == 'dincae')).sum()
    both_valid = df['sat_winner'].notna() & df['insitu_winner'].notna()
    n_both = both_valid.sum()
    
    # Draw boxes
    # Left: Satellite CV result
    sat_box = mpatches.FancyBboxPatch((0.5, 3), 2.5, 4, boxstyle="round,pad=0.1",
                                       facecolor='#ecf0f1', edgecolor='black', linewidth=2)
    ax.add_patch(sat_box)
    ax.text(1.75, 5, f'Satellite CV\nat buoy pixel\n\nDINEOF: {sat_valid}\nDINCAE: 0', 
            ha='center', va='center', fontsize=10)
    
    # Right top: Both agree DINEOF
    agree_box = mpatches.FancyBboxPatch((6.5, 6), 3, 2, boxstyle="round,pad=0.1",
                                         facecolor='#3498db', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(agree_box)
    ax.text(8, 7, f'Both say\nDINEOF\n({both_dineof})', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Right bottom: Disagree (sat=DINEOF, insitu=DINCAE)
    disagree_box = mpatches.FancyBboxPatch((6.5, 2), 3, 2, boxstyle="round,pad=0.1",
                                            facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(disagree_box)
    ax.text(8, 3, f'Sat=DINEOF\nInsitu=DINCAE\n({sat_dineof_insitu_dincae})', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(6.3, 7), xytext=(3.2, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax.annotate('', xy=(6.3, 3), xytext=(3.2, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
    
    # Key insight text
    ax.text(5, 0.5, 
            f'Key insight: {sat_dineof_insitu_dincae} lakes where DINEOF wins satellite CV\n'
            f'but DINCAE wins in-situ CV — at the EXACT SAME PIXEL',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'key_finding_same_pixel_different_winner.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_comprehensive_summary(data: dict, output_dir: str):
    """
    Plot 3: Comprehensive summary - all validation approaches
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # A) Satellite CV consistency across different approaches
    ax = fig.add_subplot(gs[0, 0])
    
    approaches = ['Large area\nCV (~97%)', 'Center\npixel', 'Shore\npixel', 'Buoy\npixel']
    dineof_wins = [97, 100, 100, 100]  # From the results
    
    colors = ['#3498db'] * 4
    bars = ax.bar(approaches, dineof_wins, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax.set_ylabel('DINEOF win rate (%)')
    ax.set_title('A) Satellite CV: DINEOF Dominates\n(Reconstruction vs Satellite Obs)')
    ax.set_ylim(0, 110)
    
    for bar, val in zip(bars, dineof_wins):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # B) In-situ CV - the discrepancy
    ax = fig.add_subplot(gs[0, 1])
    
    if 'cv_comparison' in data:
        df = data['cv_comparison']
        insitu_dineof = (df['insitu_winner'] == 'dineof').sum()
        n = len(df)
        insitu_pct = 100 * insitu_dineof / n
    else:
        insitu_pct = 58  # From earlier results
        n = 24
    
    categories = ['Satellite CV\n(all approaches)', 'In-situ CV']
    values = [100, insitu_pct]
    colors = ['#3498db', '#9b59b6']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('DINEOF win rate (%)')
    ax.set_title('B) The Discrepancy\n(Same pixel, different ground truth)')
    ax.set_ylim(0, 110)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add annotation for the gap
    ax.annotate('', xy=(1, insitu_pct), xytext=(0, 100),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, (100 + insitu_pct)/2, f'~{100-insitu_pct:.0f}%\ngap', 
            ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    # C) Shore distance comparison (if data available)
    ax = fig.add_subplot(gs[1, 0])
    
    if 'cv_comparison' in data:
        df = data['cv_comparison']
        dineof_wins_df = df[df['insitu_winner'] == 'dineof']
        dincae_wins_df = df[df['insitu_winner'] == 'dincae']
        
        dineof_dist = dineof_wins_df['buoy_distance_from_shore'].dropna()
        dincae_dist = dincae_wins_df['buoy_distance_from_shore'].dropna()
        
        # Box plot
        bp = ax.boxplot([dineof_dist, dincae_dist], 
                        labels=['DINEOF wins\nin-situ CV', 'DINCAE wins\nin-situ CV'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        
        ax.set_ylabel('Buoy distance from shore (pixels)')
        ax.set_title('C) Shore Distance by Winner\n(Weak pattern - not conclusive)')
        
        # Add individual points
        for i, (dist, color) in enumerate([(dineof_dist, '#3498db'), (dincae_dist, '#e74c3c')]):
            x = np.random.normal(i+1, 0.05, len(dist))
            ax.scatter(x, dist, alpha=0.5, color=color, edgecolor='black', s=40, zorder=3)
        
        # Add means
        ax.scatter([1], [dineof_dist.mean()], marker='D', s=100, color='white', 
                   edgecolor='black', zorder=4, label=f'Mean: {dineof_dist.mean():.1f}px')
        ax.scatter([2], [dincae_dist.mean()], marker='D', s=100, color='white', 
                   edgecolor='black', zorder=4)
        
        ax.text(1, dineof_dist.mean() + 1, f'{dineof_dist.mean():.1f}', ha='center', fontsize=9)
        ax.text(2, dincae_dist.mean() + 1, f'{dincae_dist.mean():.1f}', ha='center', fontsize=9)
    
    # D) Key takeaways (text box)
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    
    takeaways = """
    KEY FINDINGS:
    
    1. DINEOF CONSISTENTLY WINS SATELLITE CV
       • Large area CV: ~97%
       • Single pixel (any location): 100%
       • This is ROBUST and RELIABLE
    
    2. IN-SITU CV SHOWS REDUCED ADVANTAGE
       • Same lakes, same pixel: 58% DINEOF
       • 10 lakes flip from DINEOF → DINCAE
       • Discrepancy is about GROUND TRUTH, not method
    
    3. WHAT DOESN'T EXPLAIN THE 40% DINCAE WINS:
       • ✗ Spatial smoothness artifacts
       • ✗ Shore distance (weak, not significant)
       • ✗ Amplitude damping prediction
    
    4. CONCLUSION FOR PAPER:
       "DINEOF demonstrates superior reconstruction
        of satellite skin temperature (97-100%).
        In-situ validation shows reduced advantage,
        reflecting satellite vs buoy measurement
        differences, not methodological issues."
    """
    
    ax.text(0.05, 0.95, takeaways, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2))
    ax.set_title('D) Summary & Conclusions')
    
    plt.suptitle('DINEOF vs DINCAE: Cross-Validation Analysis Summary', fontsize=14, fontweight='bold', y=0.98)
    
    out_path = os.path.join(output_dir, 'comprehensive_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_individual_lake_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot 4: Individual lake scatter - satellite CV RMSE vs in-situ CV RMSE
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # A) DINEOF: Satellite RMSE vs In-situ RMSE
    ax = axes[0]
    
    valid = df['sat_dineof_rmse'].notna() & df['insitu_dineof_rmse'].notna()
    if valid.sum() > 0:
        x = df.loc[valid, 'sat_dineof_rmse']
        y = df.loc[valid, 'insitu_dineof_rmse']
        
        colors = ['#3498db' if w == 'dineof' else '#e74c3c' 
                  for w in df.loc[valid, 'insitu_winner']]
        
        ax.scatter(x, y, c=colors, s=80, edgecolor='black', alpha=0.7)
        
        # Add lake labels
        for i, (xi, yi, lid) in enumerate(zip(x, y, df.loc[valid, 'lake_id'])):
            ax.annotate(str(int(lid)), (xi, yi), fontsize=7, alpha=0.7)
        
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'k--', alpha=0.3)
        
    ax.set_xlabel('DINEOF RMSE vs Satellite Obs (°C)')
    ax.set_ylabel('DINEOF RMSE vs In-situ (°C)')
    ax.set_title('A) DINEOF Performance\n(each point = one lake)')
    
    # Legend
    dineof_patch = mpatches.Patch(color='#3498db', label='DINEOF wins in-situ')
    dincae_patch = mpatches.Patch(color='#e74c3c', label='DINCAE wins in-situ')
    ax.legend(handles=[dineof_patch, dincae_patch])
    
    # B) DINCAE: Satellite RMSE vs In-situ RMSE
    ax = axes[1]
    
    valid = df['sat_dincae_rmse'].notna() & df['insitu_dincae_rmse'].notna()
    if valid.sum() > 0:
        x = df.loc[valid, 'sat_dincae_rmse']
        y = df.loc[valid, 'insitu_dincae_rmse']
        
        colors = ['#3498db' if w == 'dineof' else '#e74c3c' 
                  for w in df.loc[valid, 'insitu_winner']]
        
        ax.scatter(x, y, c=colors, s=80, edgecolor='black', alpha=0.7)
        
        for i, (xi, yi, lid) in enumerate(zip(x, y, df.loc[valid, 'lake_id'])):
            ax.annotate(str(int(lid)), (xi, yi), fontsize=7, alpha=0.7)
        
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'k--', alpha=0.3)
    
    ax.set_xlabel('DINCAE RMSE vs Satellite Obs (°C)')
    ax.set_ylabel('DINCAE RMSE vs In-situ (°C)')
    ax.set_title('B) DINCAE Performance\n(each point = one lake)')
    ax.legend(handles=[dineof_patch, dincae_patch])
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'individual_lake_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_winner_flip_analysis(df: pd.DataFrame, output_dir: str):
    """
    Plot 5: Focus on lakes where winner flips between satellite and in-situ CV
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get lakes where sat=DINEOF but insitu=DINCAE
    flip_mask = (df['sat_winner'] == 'dineof') & (df['insitu_winner'] == 'dincae')
    no_flip_mask = (df['sat_winner'] == 'dineof') & (df['insitu_winner'] == 'dineof')
    
    flip_lakes = df[flip_mask].copy()
    no_flip_lakes = df[no_flip_mask].copy()
    
    # Sort by distance from shore
    all_lakes = pd.concat([flip_lakes, no_flip_lakes]).sort_values('buoy_distance_from_shore')
    
    # Bar chart showing each lake
    y_pos = np.arange(len(all_lakes))
    colors = ['#e74c3c' if w == 'dincae' else '#3498db' for w in all_lakes['insitu_winner']]
    
    bars = ax.barh(y_pos, all_lakes['buoy_distance_from_shore'], color=colors, edgecolor='black', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Lake {int(lid)}" for lid in all_lakes['lake_id']])
    ax.set_xlabel('Buoy distance from shore (pixels)')
    ax.set_title('Lakes Ranked by Buoy Distance from Shore\n(Color = In-situ CV winner)')
    
    # Legend
    dineof_patch = mpatches.Patch(color='#3498db', label='DINEOF wins in-situ CV')
    dincae_patch = mpatches.Patch(color='#e74c3c', label='DINCAE wins in-situ CV (flipped)')
    ax.legend(handles=[dineof_patch, dincae_patch], loc='lower right')
    
    # Add vertical line at threshold
    ax.axvline(5, color='gray', linestyle='--', alpha=0.5, label='5px threshold')
    ax.text(5.2, len(all_lakes)-1, '5px', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'winner_flip_by_shore_distance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_presentation_slide(data: dict, output_dir: str):
    """
    Create a single presentation-ready summary slide
    """
    fig = plt.figure(figsize=(16, 9))
    
    # Title
    fig.suptitle('DINEOF vs DINCAE Gap-Filling: Cross-Validation Summary', 
                 fontsize=18, fontweight='bold', y=0.97)
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.05, right=0.95, top=0.90, bottom=0.08)
    
    # 1. Satellite CV (top left)
    ax = fig.add_subplot(gs[0, 0])
    approaches = ['Area\n(~97%)', 'Center\n(100%)', 'Shore\n(100%)', 'Buoy\n(100%)']
    values = [97, 100, 100, 100]
    ax.bar(approaches, values, color='#3498db', edgecolor='black')
    ax.set_ylim(0, 110)
    ax.set_ylabel('DINEOF win %')
    ax.set_title('Satellite CV\n(vs satellite obs)', fontweight='bold')
    ax.axhline(50, color='red', linestyle='--', alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v+2, f'{v}%', ha='center', fontsize=9, fontweight='bold')
    
    # 2. The discrepancy (top middle)
    ax = fig.add_subplot(gs[0, 1])
    cats = ['Satellite CV', 'In-situ CV']
    vals = [100, 58]
    colors = ['#3498db', '#9b59b6']
    bars = ax.bar(cats, vals, color=colors, edgecolor='black')
    ax.set_ylim(0, 110)
    ax.set_ylabel('DINEOF win %')
    ax.set_title('Same Pixel, Different Result', fontweight='bold')
    ax.axhline(50, color='red', linestyle='--', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v+2, f'{v}%', ha='center', fontsize=11, fontweight='bold')
    
    # Arrow showing the gap
    ax.annotate('', xy=(1, 58), xytext=(0, 100),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, 79, '42%\ngap', ha='center', fontsize=10, color='red', fontweight='bold')
    
    # 3. What flips (top right)
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    
    flip_text = """
    SAME PIXEL ANALYSIS:
    ━━━━━━━━━━━━━━━━━━━━━
    
    23 lakes with both CVs:
    
    • Both say DINEOF:     13
    • Sat=DINEOF →
      Insitu=DINCAE:       10
    
    ━━━━━━━━━━━━━━━━━━━━━
    10 lakes flip winner
    when ground truth changes
    from satellite to buoy
    """
    ax.text(0.1, 0.9, flip_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2))
    
    # 4. Shore distance (bottom left)
    ax = fig.add_subplot(gs[1, 0])
    if 'cv_comparison' in data:
        df = data['cv_comparison']
        dineof_dist = df[df['insitu_winner'] == 'dineof']['buoy_distance_from_shore'].dropna()
        dincae_dist = df[df['insitu_winner'] == 'dincae']['buoy_distance_from_shore'].dropna()
        
        bp = ax.boxplot([dineof_dist, dincae_dist], 
                        labels=['DINEOF\nwins', 'DINCAE\nwins'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        
        ax.scatter(np.ones(len(dineof_dist))*1 + np.random.uniform(-0.1, 0.1, len(dineof_dist)), 
                   dineof_dist, alpha=0.5, color='#3498db', s=30, zorder=3)
        ax.scatter(np.ones(len(dincae_dist))*2 + np.random.uniform(-0.1, 0.1, len(dincae_dist)), 
                   dincae_dist, alpha=0.5, color='#e74c3c', s=30, zorder=3)
        
        ax.set_ylabel('Distance from shore (px)')
        ax.set_title(f'Shore Distance\n(mean: {dineof_dist.mean():.1f} vs {dincae_dist.mean():.1f}px)', fontweight='bold')
    
    # 5. Near vs far shore (bottom middle)
    ax = fig.add_subplot(gs[1, 1])
    if 'cv_comparison' in data:
        df = data['cv_comparison']
        near = df['buoy_distance_from_shore'] < 5
        far = df['buoy_distance_from_shore'] >= 5
        
        near_dineof = (df[near]['insitu_winner'] == 'dineof').sum()
        far_dineof = (df[far]['insitu_winner'] == 'dineof').sum()
        
        n_near = near.sum()
        n_far = far.sum()
        
        cats = [f'Near shore\n(<5px, N={n_near})', f'Far from shore\n(≥5px, N={n_far})']
        vals = [100*near_dineof/n_near if n_near > 0 else 0, 
                100*far_dineof/n_far if n_far > 0 else 0]
        
        bars = ax.bar(cats, vals, color=['#f39c12', '#27ae60'], edgecolor='black')
        ax.set_ylim(0, 100)
        ax.set_ylabel('DINEOF win %')
        ax.set_title('Shore Proximity Effect\n(weak pattern)', fontweight='bold')
        ax.axhline(50, color='red', linestyle='--', alpha=0.3)
        
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v+2, f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')
    
    # 6. Conclusions (bottom right)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    conclusions = """
    CONCLUSIONS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ✓ DINEOF is consistently better
      at reconstructing satellite
      skin temperature (97-100%)
    
    ✓ The 42% in-situ gap is due to
      satellite vs buoy differences,
      NOT methodology issues
    
    ✗ Shore distance doesn't clearly
      explain which lakes flip
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PAPER: Report both CVs honestly.
    DINEOF wins satellite; in-situ
    shows methods are closer.
    """
    ax.text(0.05, 0.95, conclusions, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#d4edda', edgecolor='#28a745', linewidth=2))
    
    out_path = os.path.join(output_dir, 'presentation_slide.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots for findings")
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_root, "visualization_plots")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Load data
    data = load_data(args.run_root)
    
    if 'cv_comparison' not in data:
        print("ERROR: satellite_vs_insitu_cv_comparison.csv not found")
        print("Run compare_satellite_insitu_cv.py first")
        return 1
    
    df = data['cv_comparison']
    print(f"Loaded data for {len(df)} lakes")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    plot_shore_distance_distribution(df, args.output_dir)
    plot_key_finding(df, args.output_dir)
    plot_comprehensive_summary(data, args.output_dir)
    plot_winner_flip_analysis(df, args.output_dir)
    create_presentation_slide(data, args.output_dir)
    
    print(f"\n✓ All plots saved to: {args.output_dir}")
    print("\nPlots created:")
    print("  1. shore_distance_by_winner.png - Distribution comparison")
    print("  2. key_finding_same_pixel_different_winner.png - The main finding")
    print("  3. comprehensive_summary.png - Full 4-panel summary")
    print("  4. winner_flip_by_shore_distance.png - Which lakes flip")
    print("  5. presentation_slide.png - Single slide for sharing")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
