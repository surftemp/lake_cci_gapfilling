#!/usr/bin/env python3
"""
Temporal Gap Analysis for Lake CCI Gap-Filling Project

Computes per-lake temporal gap statistics from prepared.nc observation timestamps.
Gaps are defined as the number of days between consecutive observation dates.

Outputs:
  per_lake/{lake_id}/gap_timeseries.csv   - per-observation gap record
  per_lake/{lake_id}/gap_distribution.png - histogram + boxplot of gaps
  gap_summary_all_lakes.csv               - one row per lake, all gap stats

Usage (interactive, single lake):
  python temporal_gap_analysis.py --exp-dir /path/to/experiment --lake-id 000000020

Usage (interactive, all lakes):
  python temporal_gap_analysis.py --exp-dir /path/to/experiment

Usage (assemble only, after parallel runs):
  python temporal_gap_analysis.py --exp-dir /path/to/experiment --assemble-only

Author: Shaerdan / NCEO / University of Reading
Date: February 2026
"""

import argparse
import os
import sys
import csv
import glob
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Optional imports — fail gracefully if not available (e.g. on compute nodes)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ========== Constants ==========

EPOCH = np.datetime64("1981-01-01T12:00:00")


# ========== Core computation ==========

def load_observation_days(prepared_nc_path: str) -> np.ndarray:
    """
    Load the observation timestamps from prepared.nc as integer days since 1981-01-01.
    
    Only reads the 'time' variable — does NOT load the full dataset (important for
    large files that can be 18+ GB).
    
    Returns:
        Sorted 1D int64 array of days since epoch.
    """
    import xarray as xr
    with xr.open_dataset(prepared_nc_path) as ds:
        time_vals = ds["time"].values
        
        # Handle different encodings
        if np.issubdtype(time_vals.dtype, np.datetime64):
            # Convert datetime64 to days since epoch
            base = np.datetime64("1981-01-01T12:00:00", "ns")
            days = ((time_vals.astype("datetime64[ns]") - base) / np.timedelta64(1, "D")).astype("int64")
        elif np.issubdtype(time_vals.dtype, np.integer):
            # Already integer days
            days = time_vals.astype("int64")
        else:
            raise ValueError(f"Unexpected time dtype: {time_vals.dtype}")
    
    days = np.sort(days)
    return days


def compute_gaps(days: np.ndarray) -> np.ndarray:
    """
    Compute gaps between consecutive observation dates.
    
    Args:
        days: Sorted array of observation days (int64).
        
    Returns:
        1D array of gaps (length = len(days) - 1). Each value is 
        days[i+1] - days[i] for consecutive observations.
    """
    if len(days) < 2:
        return np.array([], dtype="int64")
    return np.diff(days)


def compute_gap_stats(days: np.ndarray, gaps: np.ndarray) -> Dict:
    """
    Compute comprehensive gap statistics for one lake.
    
    Args:
        days: Sorted observation days.
        gaps: Gap array from compute_gaps().
        
    Returns:
        Dictionary of statistics.
    """
    n_obs = len(days)
    
    if n_obs < 2:
        return {
            "n_obs": n_obs,
            "n_total_days": 0,
            "coverage_fraction": 0.0,
            "gap_mean": np.nan,
            "gap_median": np.nan,
            "gap_std": np.nan,
            "gap_robust_std": np.nan,
            "gap_min": np.nan,
            "gap_max": np.nan,
            "gap_p25": np.nan,
            "gap_p75": np.nan,
            "gap_p90": np.nan,
            "gap_p95": np.nan,
            "gap_p99": np.nan,
            "gap_iqr": np.nan,
            "n_gaps_gt7": 0,
            "n_gaps_gt14": 0,
            "n_gaps_gt30": 0,
            "n_gaps_gt60": 0,
            "n_gaps_gt90": 0,
            "longest_gap_start_date": "",
            "longest_gap_end_date": "",
            "first_obs_date": "",
            "last_obs_date": "",
            "time_span_days": 0,
        }
    
    n_total_days = int(days[-1] - days[0] + 1)
    coverage_fraction = n_obs / n_total_days if n_total_days > 0 else 0.0
    
    # Basic stats
    gap_mean = float(np.mean(gaps))
    gap_median = float(np.median(gaps))
    gap_std = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
    
    # Robust std (1.4826 * MAD)
    mad = float(np.median(np.abs(gaps - np.median(gaps))))
    gap_robust_std = 1.4826 * mad
    
    # Percentiles
    gap_p25 = float(np.percentile(gaps, 25))
    gap_p75 = float(np.percentile(gaps, 75))
    gap_p90 = float(np.percentile(gaps, 90))
    gap_p95 = float(np.percentile(gaps, 95))
    gap_p99 = float(np.percentile(gaps, 99))
    gap_iqr = gap_p75 - gap_p25
    
    # Threshold counts
    n_gaps_gt7 = int(np.sum(gaps > 7))
    n_gaps_gt14 = int(np.sum(gaps > 14))
    n_gaps_gt30 = int(np.sum(gaps > 30))
    n_gaps_gt60 = int(np.sum(gaps > 60))
    n_gaps_gt90 = int(np.sum(gaps > 90))
    
    # Longest gap location
    longest_idx = int(np.argmax(gaps))
    longest_start_day = int(days[longest_idx])
    longest_end_day = int(days[longest_idx + 1])
    
    def day_to_datestr(d: int) -> str:
        dt = EPOCH + np.timedelta64(d, "D")
        return str(dt)[:10]
    
    return {
        "n_obs": n_obs,
        "n_total_days": n_total_days,
        "coverage_fraction": round(coverage_fraction, 6),
        "gap_mean": round(gap_mean, 3),
        "gap_median": round(gap_median, 3),
        "gap_std": round(gap_std, 3),
        "gap_robust_std": round(gap_robust_std, 3),
        "gap_min": int(np.min(gaps)),
        "gap_max": int(np.max(gaps)),
        "gap_p25": round(gap_p25, 3),
        "gap_p75": round(gap_p75, 3),
        "gap_p90": round(gap_p90, 3),
        "gap_p95": round(gap_p95, 3),
        "gap_p99": round(gap_p99, 3),
        "gap_iqr": round(gap_iqr, 3),
        "n_gaps_gt7": n_gaps_gt7,
        "n_gaps_gt14": n_gaps_gt14,
        "n_gaps_gt30": n_gaps_gt30,
        "n_gaps_gt60": n_gaps_gt60,
        "n_gaps_gt90": n_gaps_gt90,
        "longest_gap_start_date": day_to_datestr(longest_start_day),
        "longest_gap_end_date": day_to_datestr(longest_end_day),
        "first_obs_date": day_to_datestr(int(days[0])),
        "last_obs_date": day_to_datestr(int(days[-1])),
        "time_span_days": n_total_days,
    }


# ========== Per-lake output ==========

def write_gap_timeseries_csv(
    days: np.ndarray, 
    gaps: np.ndarray, 
    output_path: str,
    lake_id: str
) -> None:
    """
    Write per-observation gap CSV.
    
    Columns:
      observation_index: 0-based index
      date: human-readable date (YYYY-MM-DD)
      days_since_epoch: integer days since 1981-01-01
      gap_days_to_previous: gap from previous observation (NaN for first row)
      gap_days_to_next: gap to next observation (NaN for last row)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n = len(days)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "observation_index", "date", "days_since_epoch",
            "gap_days_to_previous", "gap_days_to_next"
        ])
        
        for i in range(n):
            dt = EPOCH + np.timedelta64(int(days[i]), "D")
            date_str = str(dt)[:10]
            gap_prev = int(gaps[i - 1]) if i > 0 else ""
            gap_next = int(gaps[i]) if i < len(gaps) else ""
            writer.writerow([i, date_str, int(days[i]), gap_prev, gap_next])
    
    print(f"  Wrote {output_path} ({n} observations)")


def plot_gap_distribution(
    gaps: np.ndarray, 
    output_path: str,
    lake_id: str,
    stats: Dict
) -> None:
    """
    Create a combined histogram + boxplot figure for gap distribution.
    """
    if not HAS_MATPLOTLIB:
        print(f"  Skipping plot (matplotlib not available)")
        return
    
    if len(gaps) < 2:
        print(f"  Skipping plot (< 2 gaps)")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), 
                             gridspec_kw={"height_ratios": [3, 1]})
    
    # --- Histogram ---
    ax_hist = axes[0]
    max_gap = int(np.max(gaps))
    
    # Adaptive binning: 1-day bins up to 60, then coarser
    if max_gap <= 60:
        bins = np.arange(0.5, max_gap + 1.5, 1)
    elif max_gap <= 180:
        bins = np.concatenate([
            np.arange(0.5, 60.5, 1),
            np.arange(60.5, max_gap + 5.5, 5)
        ])
    else:
        bins = np.concatenate([
            np.arange(0.5, 30.5, 1),
            np.arange(30.5, 90.5, 5),
            np.arange(90.5, max_gap + 15.5, 15)
        ])
    
    ax_hist.hist(gaps, bins=bins, edgecolor="black", linewidth=0.3,
                 color="steelblue", alpha=0.8)
    ax_hist.set_xlabel("Gap (days)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"Lake {lake_id} — Temporal Gap Distribution\n"
        f"n_obs={stats['n_obs']}, coverage={stats['coverage_fraction']:.1%}, "
        f"median gap={stats['gap_median']:.1f}d, max gap={stats['gap_max']}d"
    )
    
    # Add vertical lines for key percentiles
    for pct, label, color, ls in [
        (stats["gap_median"], "median", "red", "-"),
        (stats["gap_p90"], "p90", "orange", "--"),
        (stats["gap_p99"], "p99", "darkred", ":"),
    ]:
        if np.isfinite(pct):
            ax_hist.axvline(pct, color=color, linestyle=ls, linewidth=1.5, 
                           label=f"{label}={pct:.0f}d")
    ax_hist.legend(fontsize=9)
    
    # --- Boxplot ---
    ax_box = axes[1]
    bp = ax_box.boxplot(gaps, vert=False, widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.5),
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
    ax_box.set_xlabel("Gap (days)")
    ax_box.set_yticks([])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ========== Summary across all lakes ==========

SUMMARY_COLUMNS = [
    "lake_id",
    "n_obs",
    "n_total_days",
    "time_span_days",
    "coverage_fraction",
    "first_obs_date",
    "last_obs_date",
    "gap_mean",
    "gap_median",
    "gap_std",
    "gap_robust_std",
    "gap_min",
    "gap_max",
    "gap_p25",
    "gap_p75",
    "gap_p90",
    "gap_p95",
    "gap_p99",
    "gap_iqr",
    "n_gaps_gt7",
    "n_gaps_gt14",
    "n_gaps_gt30",
    "n_gaps_gt60",
    "n_gaps_gt90",
    "longest_gap_start_date",
    "longest_gap_end_date",
]


def write_summary_csv(all_stats: List[Tuple[str, Dict]], output_path: str) -> None:
    """Write the global summary CSV with one row per lake."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", 
                exist_ok=True)
    
    # Sort by lake_id
    all_stats.sort(key=lambda x: x[0])
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        
        for lake_id, stats in all_stats:
            row = [lake_id] + [stats.get(col, "") for col in SUMMARY_COLUMNS[1:]]
            writer.writerow(row)
    
    print(f"\nWrote summary: {output_path} ({len(all_stats)} lakes)")


def assemble_summary_from_per_lake_csvs(output_dir: str) -> None:
    """
    Re-assemble the global summary CSV by re-reading all per-lake gap_timeseries.csv files.
    Useful after parallel SLURM runs.
    """
    per_lake_dir = os.path.join(output_dir, "per_lake")
    if not os.path.isdir(per_lake_dir):
        print(f"ERROR: {per_lake_dir} does not exist")
        sys.exit(1)
    
    lake_dirs = sorted(glob.glob(os.path.join(per_lake_dir, "*")))
    all_stats = []
    
    for lake_dir in lake_dirs:
        lake_id = os.path.basename(lake_dir)
        csv_path = os.path.join(lake_dir, "gap_timeseries.csv")
        if not os.path.isfile(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        
        # Read days from CSV
        days_list = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                days_list.append(int(row["days_since_epoch"]))
        
        days = np.array(days_list, dtype="int64")
        gaps = compute_gaps(days)
        stats = compute_gap_stats(days, gaps)
        all_stats.append((lake_id, stats))
        print(f"  {lake_id}: {stats['n_obs']} obs, median gap={stats['gap_median']}, max gap={stats['gap_max']}")
    
    summary_path = os.path.join(output_dir, "gap_summary_all_lakes.csv")
    write_summary_csv(all_stats, summary_path)


# ========== Lake discovery ==========

def discover_lakes(exp_dir: str) -> List[Tuple[str, str]]:
    """
    Find all lake prepared.nc files under exp_dir/prepared/.
    
    Returns:
        List of (lake_id, prepared_nc_path) tuples, sorted by lake_id.
    """
    prepared_dir = os.path.join(exp_dir, "prepared")
    if not os.path.isdir(prepared_dir):
        print(f"ERROR: {prepared_dir} does not exist")
        sys.exit(1)
    
    lakes = []
    for entry in sorted(os.listdir(prepared_dir)):
        prep_path = os.path.join(prepared_dir, entry, "prepared.nc")
        if os.path.isfile(prep_path):
            lakes.append((entry, prep_path))
    
    return lakes


# ========== Process one lake ==========

def process_lake(
    lake_id: str,
    prepared_nc_path: str,
    output_dir: str,
    make_plots: bool = True
) -> Optional[Dict]:
    """
    Process a single lake: load times, compute gaps, write CSV and plots.
    
    Returns:
        Gap statistics dict, or None on failure.
    """
    print(f"\n[{lake_id}] Processing...")
    
    try:
        days = load_observation_days(prepared_nc_path)
    except Exception as e:
        print(f"  ERROR loading {prepared_nc_path}: {e}")
        return None
    
    gaps = compute_gaps(days)
    stats = compute_gap_stats(days, gaps)
    
    # Print quick summary
    print(f"  n_obs={stats['n_obs']}, span={stats['first_obs_date']}..{stats['last_obs_date']}, "
          f"coverage={stats['coverage_fraction']:.1%}")
    print(f"  gap: median={stats['gap_median']}, mean={stats['gap_mean']}, "
          f"max={stats['gap_max']}, robust_std={stats['gap_robust_std']}")
    print(f"  gaps >7d: {stats['n_gaps_gt7']}, >14d: {stats['n_gaps_gt14']}, "
          f">30d: {stats['n_gaps_gt30']}, >60d: {stats['n_gaps_gt60']}")
    
    # Write per-lake CSV
    lake_out_dir = os.path.join(output_dir, "per_lake", lake_id)
    csv_path = os.path.join(lake_out_dir, "gap_timeseries.csv")
    write_gap_timeseries_csv(days, gaps, csv_path, lake_id)
    
    # Write per-lake plot
    if make_plots:
        plot_path = os.path.join(lake_out_dir, "gap_distribution.png")
        plot_gap_distribution(gaps, plot_path, lake_id, stats)
    
    return stats


# ========== CLI ==========

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Temporal Gap Analysis for Lake CCI Gap-Filling Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single lake (interactive)
  python temporal_gap_analysis.py --exp-dir /path/to/exp --lake-id 000000020

  # All lakes (interactive)
  python temporal_gap_analysis.py --exp-dir /path/to/exp

  # All lakes, no plots (faster)
  python temporal_gap_analysis.py --exp-dir /path/to/exp --no-plots

  # Assemble summary after parallel SLURM runs
  python temporal_gap_analysis.py --exp-dir /path/to/exp --assemble-only

  # Custom output directory
  python temporal_gap_analysis.py --exp-dir /path/to/exp --output-dir /path/to/gap_analysis
"""
    )
    
    p.add_argument("--exp-dir", required=True,
                   help="Experiment root directory (contains prepared/, post/, dineof/, dincae/)")
    p.add_argument("--lake-id", default=None,
                   help="Process a single lake ID (e.g. 000000020). If omitted, processes all lakes.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Default: {exp-dir}/gap_analysis/")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating per-lake distribution plots (faster)")
    p.add_argument("--assemble-only", action="store_true",
                   help="Only assemble global summary from existing per-lake CSVs")
    
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    exp_dir = args.exp_dir.rstrip("/")
    output_dir = args.output_dir or os.path.join(exp_dir, "gap_analysis")
    
    print(f"Experiment dir: {exp_dir}")
    print(f"Output dir:     {output_dir}")
    
    # Assemble-only mode
    if args.assemble_only:
        print("\n=== Assemble-only mode ===")
        assemble_summary_from_per_lake_csvs(output_dir)
        return
    
    # Discover lakes
    if args.lake_id:
        # Single lake
        prep_path = os.path.join(exp_dir, "prepared", args.lake_id, "prepared.nc")
        if not os.path.isfile(prep_path):
            print(f"ERROR: {prep_path} not found")
            sys.exit(1)
        lakes = [(args.lake_id, prep_path)]
    else:
        # All lakes
        lakes = discover_lakes(exp_dir)
    
    print(f"Found {len(lakes)} lake(s) to process")
    
    # Process
    all_stats = []
    for lake_id, prep_path in lakes:
        stats = process_lake(
            lake_id=lake_id,
            prepared_nc_path=prep_path,
            output_dir=output_dir,
            make_plots=not args.no_plots,
        )
        if stats is not None:
            all_stats.append((lake_id, stats))
    
    # Write global summary
    if all_stats:
        summary_path = os.path.join(output_dir, "gap_summary_all_lakes.csv")
        write_summary_csv(all_stats, summary_path)
    
    print(f"\nDone. Processed {len(all_stats)}/{len(lakes)} lakes.")


if __name__ == "__main__":
    main()
