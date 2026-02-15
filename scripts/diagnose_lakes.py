#!/usr/bin/env python3
"""
diagnose_lakes.py - Quick scan of a pipeline run to identify:
  1. Lakes where ICE replacement actually removed/replaced pixels
  2. Lakes with in-situ validation results

Usage:
  python diagnose_lakes.py --run-root /gws/.../anomaly-20260131-219f0d-exp0_baseline_both/
"""

import argparse
import glob
import os
import re
import sys


def scan_ice_replacement(logs_dir: str) -> dict:
    """
    Scan log files for IceMaskReplacement activity.
    Returns {lake_id: {"n_pixels": int, "n_times": int}} for lakes with replacements.
    """
    results = {}

    # Log files: look for preprocessing logs (stage=pre)
    # Pattern: *_pre_*.out or similar
    log_files = glob.glob(os.path.join(logs_dir, "*.out"))
    if not log_files:
        log_files = glob.glob(os.path.join(logs_dir, "**", "*.out"), recursive=True)

    # Regex to extract lake_id from log filename or content
    # Log filenames often contain the lake ID
    ice_pattern = re.compile(
        r"IceMaskReplacement: replaced (\d+) pixels across (\d+) time"
    )
    lake_id_pattern = re.compile(r"lake[_\s]*(?:id)?[=:\s]*(\d+)", re.IGNORECASE)
    # Also try extracting from filename like: pre_LAKE000000380_...out
    filename_lake_pattern = re.compile(r"LAKE(\d{9})|lake[_-]?(\d+)|row[_-]?(\d+)")

    for log_path in sorted(log_files):
        basename = os.path.basename(log_path)

        try:
            with open(log_path, "r", errors="replace") as f:
                content = f.read()
        except Exception:
            continue

        # Check if this log has ice replacement
        ice_match = ice_pattern.search(content)
        if not ice_match:
            continue

        n_pixels = int(ice_match.group(1))
        n_times = int(ice_match.group(2))

        if n_pixels == 0:
            continue

        # Try to find lake_id
        lake_id = None

        # Method 1: from filename
        fn_match = filename_lake_pattern.search(basename)
        if fn_match:
            for g in fn_match.groups():
                if g is not None:
                    lake_id = int(g)
                    break

        # Method 2: look for lake_id in content (e.g. "lake_id: 380" or attrs)
        if lake_id is None:
            # Look for prepared.nc lake_id attr
            attr_match = re.search(r"['\"]lake_id['\"]:\s*(\d+)", content)
            if attr_match:
                lake_id = int(attr_match.group(1))

        # Method 3: look for LAKE{9digits} in paths within the log
        if lake_id is None:
            path_match = re.search(r"LAKE(\d{9})", content)
            if path_match:
                lake_id = int(path_match.group(1))

        # Method 4: look for /prepared/000000380/ style paths
        if lake_id is None:
            prep_match = re.search(r"/prepared/(\d{9})/", content)
            if prep_match:
                lake_id = int(prep_match.group(1))

        if lake_id is not None:
            results[lake_id] = {
                "n_pixels": n_pixels,
                "n_times": n_times,
                "log_file": basename,
            }
        else:
            # Still record it with filename as key
            results[f"unknown_{basename}"] = {
                "n_pixels": n_pixels,
                "n_times": n_times,
                "log_file": basename,
            }

    return results


def scan_insitu_validation(run_root: str) -> dict:
    """
    Scan post directories for insitu_cv_validation folders with actual results.
    Returns {lake_id: {"path": str, "n_plots": int, "n_csvs": int}}.
    """
    results = {}
    post_base = os.path.join(run_root, "post")

    if not os.path.isdir(post_base):
        return results

    for lake_dir in sorted(os.listdir(post_base)):
        lake_dir_path = os.path.join(post_base, lake_dir)
        if not os.path.isdir(lake_dir_path):
            continue

        try:
            lake_id = int(lake_dir)
        except ValueError:
            continue

        # Search all alpha subdirs
        for alpha_dir in os.listdir(lake_dir_path):
            insitu_dir = os.path.join(lake_dir_path, alpha_dir, "insitu_cv_validation")
            if not os.path.isdir(insitu_dir):
                continue

            pngs = glob.glob(os.path.join(insitu_dir, "*.png"))
            csvs = glob.glob(os.path.join(insitu_dir, "*.csv"))

            if pngs or csvs:
                results[lake_id] = {
                    "path": insitu_dir,
                    "n_plots": len(pngs),
                    "n_csvs": len(csvs),
                    "alpha": alpha_dir,
                }

    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose lakes: ice replacement + insitu validation")
    parser.add_argument("--run-root", required=True, help="Root of the experiment run")
    parser.add_argument("--logs-dir", default=None, help="Override logs directory (default: {run-root}/logs)")
    args = parser.parse_args()

    run_root = args.run_root
    logs_dir = args.logs_dir or os.path.join(run_root, "logs")

    print(f"Run root: {run_root}")
    print(f"Logs dir: {logs_dir}")
    print()

    # === Ice replacement ===
    print("=" * 70)
    print("LAKES WITH ICE PIXEL REPLACEMENT")
    print("=" * 70)

    if os.path.isdir(logs_dir):
        ice_results = scan_ice_replacement(logs_dir)
        if ice_results:
            # Sort by lake_id (numeric ones first)
            numeric = {k: v for k, v in ice_results.items() if isinstance(k, int)}
            other = {k: v for k, v in ice_results.items() if not isinstance(k, int)}

            print(f"\n{'Lake ID':>10}  {'Pixels':>10}  {'Times':>8}  Log File")
            print("-" * 70)
            for lid in sorted(numeric.keys()):
                r = numeric[lid]
                print(f"{lid:>10}  {r['n_pixels']:>10,}  {r['n_times']:>8}  {r['log_file']}")
            for lid in sorted(other.keys()):
                r = other[lid]
                print(f"{str(lid):>10}  {r['n_pixels']:>10,}  {r['n_times']:>8}  {r['log_file']}")

            total_pixels = sum(v["n_pixels"] for v in ice_results.values())
            print(f"\nTotal: {len(numeric)} lakes with ice replacement ({total_pixels:,} pixels total)")
        else:
            print("\nNo ice replacement found in any log files.")
    else:
        print(f"\nLogs directory not found: {logs_dir}")

    # === In-situ validation ===
    print()
    print("=" * 70)
    print("LAKES WITH IN-SITU VALIDATION")
    print("=" * 70)

    insitu_results = scan_insitu_validation(run_root)
    if insitu_results:
        print(f"\n{'Lake ID':>10}  {'Plots':>6}  {'CSVs':>5}  Alpha")
        print("-" * 50)
        for lid in sorted(insitu_results.keys()):
            r = insitu_results[lid]
            print(f"{lid:>10}  {r['n_plots']:>6}  {r['n_csvs']:>5}  {r['alpha']}")

        print(f"\nTotal: {len(insitu_results)} lakes with in-situ validation")
    else:
        print("\nNo in-situ validation results found.")

    # === Summary ===
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ice_ids = {k for k in ice_results if isinstance(k, int)} if os.path.isdir(logs_dir) else set()
    insitu_ids = set(insitu_results.keys())
    both = ice_ids & insitu_ids

    print(f"  Lakes with ice replacement: {len(ice_ids)}")
    print(f"  Lakes with in-situ data:    {len(insitu_ids)}")
    print(f"  Lakes with both:            {len(both)}")
    if both:
        print(f"    IDs: {sorted(both)}")

    # Lakes with insitu but no ice
    insitu_no_ice = insitu_ids - ice_ids
    if insitu_no_ice:
        print(f"  Insitu but no ice:          {len(insitu_no_ice)}")
        print(f"    IDs: {sorted(insitu_no_ice)}")


if __name__ == "__main__":
    main()
