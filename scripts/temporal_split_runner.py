#!/usr/bin/env python3
"""
Temporal Split Runner

Wrapper for processing large lakes that cannot complete in a single time range
due to computational resource limits. Splits the time range into segments,
runs each through the existing pipeline via lswtctl.py, then merges outputs.

Usage:
  # Generate configs + submit SLURM jobs
  python temporal_split_runner.py --base-config exp0_baseline.json --submit

  # After all jobs complete — merge segment outputs
  python temporal_split_runner.py --manifest /path/to/segment_manifest.json --merge

  # Run CV validation on merged outputs
  python temporal_split_runner.py --manifest /path/to/segment_manifest.json --validate

  # Pilot: submit + merge + validate + compare with full-range results
  python temporal_split_runner.py --base-config exp0.json --pilot \\
      --compare-root /gws/.../full_range_run/ --merge --validate

  # All-in-one (submit, poll until done, merge, validate)
  python temporal_split_runner.py --base-config exp0.json --all

Author: Shaerdan / NCEO / University of Reading
Date: February 2026
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ORCHESTRATION_DIR = SCRIPT_DIR.parent / "orchestration"


# =============================================================================
# Config Generation
# =============================================================================

def compute_segment_boundaries(
    start_date: str,
    end_date: str,
    n_segments: int,
) -> List[Tuple[str, str]]:
    """
    Compute non-overlapping segment boundaries by dividing the date range equally.
    Returns list of (start, end) date strings.
    """
    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date)
    total_days = (d1 - d0).days

    boundaries = []
    for i in range(n_segments):
        seg_start = d0 + timedelta(days=int(total_days * i / n_segments))
        if i < n_segments - 1:
            seg_end = d0 + timedelta(days=int(total_days * (i + 1) / n_segments) - 1)
        else:
            seg_end = d1
        boundaries.append((seg_start.isoformat(), seg_end.isoformat()))

    return boundaries


def parse_explicit_segments(segments_str: str) -> List[Tuple[str, str]]:
    """Parse explicit segment ranges from comma-separated string.
    Format: '2000-01-01:2010-12-31,2011-01-01:2022-12-31'
    """
    boundaries = []
    for part in segments_str.split(","):
        start, end = part.strip().split(":")
        boundaries.append((start.strip(), end.strip()))
    return boundaries


def generate_segment_configs(
    base_config_path: str,
    segments: List[Tuple[str, str]],
    large_lake_ids: List[int],
    output_dir: str,
) -> List[Dict]:
    """
    Generate per-segment config files from the base config.

    Each segment config:
    - Has modified start_date / end_date
    - Has modified note / experiment_name with _seg{i} suffix
    - Has custom_lake_ids set to only the large lakes
    - Keeps cv_fraction_target unchanged (5% of each segment = 5% overall)
    """
    with open(base_config_path) as f:
        base = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    seg_configs = []

    for i, (seg_start, seg_end) in enumerate(segments):
        cfg = copy.deepcopy(base)

        # Modify time range
        cfg["preprocessing_options"]["start_date"] = seg_start
        cfg["preprocessing_options"]["end_date"] = seg_end

        # Modify identifiers
        base_note = cfg.get("note", "")
        base_name = cfg.get("experiment_name", "")
        cfg["note"] = f"{base_note}_seg{i}" if base_note else f"seg{i}"
        cfg["experiment_name"] = f"{base_name}_seg{i}" if base_name else f"seg{i}"

        # Restrict to large lakes only
        cfg["dataset_options"]["custom_lake_ids"] = large_lake_ids
        cfg["dataset_options"]["use_custom_lake_ids"] = True

        # Write config
        config_name = f"{base_name}_seg{i}.json" if base_name else f"seg{i}.json"
        config_path = os.path.join(output_dir, config_name)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        seg_configs.append({
            "seg_id": i,
            "start": seg_start,
            "end": seg_end,
            "config_path": os.path.abspath(config_path),
        })
        print(f"  seg{i}: {seg_start} → {seg_end}  [{config_path}]")

    return seg_configs


# =============================================================================
# run_tag resolution (mirrors lswtctl.py logic)
# =============================================================================

def _canonical_dump(conf: dict) -> str:
    """Deterministic JSON dump for hashing (mirrors lswtctl.py)."""
    return json.dumps(conf, sort_keys=True, separators=(",", ":"))


def resolve_run_root(config_path: str) -> str:
    """Resolve run_root for a config, matching lswtctl.py _auto_run_tag logic."""
    with open(config_path) as f:
        conf = json.load(f)

    P = conf.get("paths", {})
    tag = P.get("run_tag")
    if not tag:
        mode = conf.get("mode", "anomaly")
        note = conf.get("note", "")
        dt = datetime.now(timezone.utc).strftime("%Y%m%d")
        h = hashlib.sha1(_canonical_dump(conf).encode()).hexdigest()[:6]
        tag = f"{mode}-{dt}-{h}"
        if note:
            tag = f"{tag}-{note}"

    tpl = P["run_root_template"]
    run_root = tpl.replace("{run_tag}", tag)
    return run_root


# =============================================================================
# SLURM Submission
# =============================================================================

def submit_segment(config_path: str, verbose: bool = True) -> str:
    """Submit a segment config via lswtctl.py submit. Returns run_root."""
    lswtctl = ORCHESTRATION_DIR / "lswtctl.py"
    cmd = [sys.executable, str(lswtctl), "submit", config_path]

    if verbose:
        print(f"  Submitting: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ORCHESTRATION_DIR))

    if result.returncode != 0:
        print(f"  ERROR: lswtctl.py submit failed (rc={result.returncode})")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError(f"Submission failed for {config_path}")

    if verbose and result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-8:]:
            print(f"    {line}")

    # Parse actual run_root from lswtctl.py stdout
    # lswtctl.py prints: "Run root: /gws/.../anomaly-YYYYMMDD-HASH-note"
    run_root = None
    for line in result.stdout.split("\n"):
        if line.strip().startswith("Run root:"):
            run_root = line.split("Run root:", 1)[1].strip()
            break

    if run_root is None:
        # Fallback: try to find manifest.json that lswtctl wrote
        # lswtctl locks the run_tag into the config, so re-read it
        import re
        for line in result.stdout.split("\n"):
            m = re.search(r"run_tag.*?:\s*(\S+)", line)
            if m:
                tag = m.group(1)
                with open(config_path) as f:
                    cfg = json.load(f)
                tpl = cfg["paths"]["run_root_template"]
                run_root = tpl.replace("{run_tag}", tag)
                break

    if run_root is None:
        raise RuntimeError(
            f"Could not determine run_root from lswtctl.py output.\n"
            f"stdout:\n{result.stdout}\n"
            f"You may need to set run_root paths in the manifest manually."
        )

    return run_root


def poll_slurm_jobs(job_prefix: str, poll_interval: int = 120, verbose: bool = True):
    """Poll SLURM until no jobs with the given prefix are running/pending."""
    print(f"\nPolling SLURM for jobs matching prefix '{job_prefix}'...")
    while True:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-n", job_prefix, "-h"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            print("  All jobs completed.")
            return
        n_running = sum(1 for l in lines if " R " in l)
        n_pending = sum(1 for l in lines if " PD " in l)
        print(f"  {datetime.now().strftime('%H:%M:%S')}: {n_running} running, {n_pending} pending, {len(lines)} total")
        time.sleep(poll_interval)


# =============================================================================
# Manifest Management
# =============================================================================

def create_manifest(
    base_config_path: str,
    seg_configs: List[Dict],
    seg_run_roots: List[str],
    segments: List[Tuple[str, str]],
    large_lake_ids: List[int],
    merge_root: str,
) -> Dict:
    """Create and save the segment manifest."""
    manifest = {
        "base_config": os.path.abspath(base_config_path),
        "n_segments": len(segments),
        "full_range": [segments[0][0], segments[-1][1]],
        "segments": [],
        "merge_root": merge_root,
        "lakes": large_lake_ids,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    for i, (sc, rr) in enumerate(zip(seg_configs, seg_run_roots)):
        manifest["segments"].append({
            "seg_id": i,
            "start": sc["start"],
            "end": sc["end"],
            "config_path": sc["config_path"],
            "run_root": rr,
        })

    os.makedirs(merge_root, exist_ok=True)
    manifest_path = os.path.join(merge_root, "segment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    return manifest


# =============================================================================
# Pilot Comparison
# =============================================================================

def compare_with_full_range(
    manifest: Dict,
    compare_root: str,
    alpha: str = "a1000",
    verbose: bool = True,
):
    """Compare merged CV results with existing full-range results."""
    from cv_validation import validate_lake, METRIC_NAMES

    merge_root = manifest["merge_root"]
    lakes = manifest["lakes"]

    print(f"\n{'='*90}")
    print("PILOT COMPARISON: Split-Merged vs Full-Range")
    print(f"{'='*90}")
    print(f"Merged root:     {merge_root}")
    print(f"Full-range root: {compare_root}")
    print()

    for lake_id in lakes:
        print(f"\n--- Lake {lake_id} ---")

        # Get merged results
        merged_results = validate_lake(merge_root, lake_id, alpha, verbose=False)
        # Get full-range results
        full_results = validate_lake(compare_root, lake_id, alpha, verbose=False)

        if not merged_results and not full_results:
            print("  No results for either run")
            continue

        # Build lookup
        merged_by_method = {r.method: r for r in merged_results}
        full_by_method = {r.method: r for r in full_results}

        header = f"  {'Metric':<10}"
        methods = sorted(set(list(merged_by_method.keys()) + list(full_by_method.keys())))
        for m in methods:
            header += f" {m+'(split)':<15} {m+'(full)':<15} {'Δ':<10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for metric in METRIC_NAMES:
            line = f"  {metric:<10}"
            for method in methods:
                mr = merged_by_method.get(method)
                fr = full_by_method.get(method)
                mv = getattr(mr, metric, None) if mr else None
                fv = getattr(fr, metric, None) if fr else None

                def _f(v):
                    if v is None:
                        return "—"
                    if isinstance(v, float) and not np.isnan(v):
                        return f"{v:.6f}"
                    if isinstance(v, int):
                        return str(v)
                    return "—"

                delta = ""
                if mv is not None and fv is not None and isinstance(mv, (int, float)) and isinstance(fv, (int, float)):
                    if not (np.isnan(mv) or np.isnan(fv)):
                        d = mv - fv
                        delta = f"{d:+.6f}" if isinstance(d, float) else f"{d:+d}"

                line += f" {_f(mv):<15} {_f(fv):<15} {delta:<10}"
            print(line)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Split Runner for large lakes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pilot: split-process 3 specific lakes
  python temporal_split_runner.py --base-config exp0.json \\
      --lake-ids 380 88 2 --n-segments 2 --submit

  # Production: all 11 large lakes
  python temporal_split_runner.py --base-config exp0.json \\
      --lake-ids 2 5 6 8 9 11 12 13 15 16 17 --submit

  # Merge after completion
  python temporal_split_runner.py --manifest segment_manifest.json --merge

  # Validate + compare with full-range results
  python temporal_split_runner.py --manifest segment_manifest.json --validate \\
      --compare-root /path/to/full_run/

  # If --lake-ids is omitted, uses custom_lake_ids from --base-config
        """
    )

    # Config sources
    parser.add_argument("--base-config", help="Base JSON config (provides time range, paths, parameters)")
    parser.add_argument("--lake-ids", type=int, nargs="+",
                        help="Lake IDs to split-process (overrides config's custom_lake_ids)")
    parser.add_argument("--manifest", help="Path to existing segment_manifest.json")

    # Segmentation
    parser.add_argument("--n-segments", type=int, default=2,
                        help="Number of time segments (default: 2)")
    parser.add_argument("--segments",
                        help="Explicit ranges: '2000-01-01:2010-12-31,2011-01-01:2022-12-31'")

    # Workflow
    parser.add_argument("--submit", action="store_true", help="Generate configs + submit SLURM")
    parser.add_argument("--merge", action="store_true", help="Merge segment outputs")
    parser.add_argument("--validate", action="store_true", help="Run CV validation")
    parser.add_argument("--all", action="store_true", help="submit + poll + merge + validate")

    # Options
    parser.add_argument("--merge-root", help="Output dir for merged results (auto if omitted)")
    parser.add_argument("--alpha", default="a1000")
    parser.add_argument("--pilot", action="store_true", help="Verbose + extra checks")
    parser.add_argument("--compare-root", help="Full-range experiment root for comparison")
    parser.add_argument("--poll-interval", type=int, default=120,
                        help="SLURM poll interval in seconds (default: 120)")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.all:
        args.submit = True
        args.merge = True
        args.validate = True

    # -------------------------------------------------------------------------
    # Phase 1: Config Generation + Submit
    # -------------------------------------------------------------------------
    manifest = None

    if args.submit:
        if not args.base_config:
            parser.error("--submit requires --base-config")

        # Load base config for time range
        with open(args.base_config) as f:
            base_cfg = json.load(f)
        start_date = base_cfg["preprocessing_options"]["start_date"]
        end_date = base_cfg["preprocessing_options"]["end_date"]

        # Determine lake IDs to split-process
        if args.lake_ids:
            large_lake_ids = args.lake_ids
        else:
            large_lake_ids = base_cfg["dataset_options"]["custom_lake_ids"]
            print(f"  (No --lake-ids provided, using {len(large_lake_ids)} lakes from config)")

        print(f"Base config: {args.base_config}")
        print(f"Time range: {start_date} → {end_date}")
        print(f"Lakes to split: {len(large_lake_ids)} → {large_lake_ids}")
        print(f"Segments: {args.n_segments}")

        # Compute segment boundaries
        if args.segments:
            segments = parse_explicit_segments(args.segments)
        else:
            segments = compute_segment_boundaries(start_date, end_date, args.n_segments)

        for i, (s, e) in enumerate(segments):
            print(f"  seg{i}: {s} → {e}")

        # Determine merge_root
        if args.merge_root:
            merge_root = args.merge_root
        else:
            # Auto-generate based on base config
            base_note = base_cfg.get("note", "experiment")
            mode = base_cfg.get("mode", "anomaly")
            dt = datetime.now(timezone.utc).strftime("%Y%m%d")
            tpl = base_cfg.get("paths", {}).get("run_root_template",
                    "/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/{run_tag}")
            merge_tag = f"{mode}-{dt}-{base_note}_merged"
            merge_root = tpl.replace("{run_tag}", merge_tag)

        configs_dir = os.path.join(merge_root, "configs")

        # Generate segment configs
        print(f"\nGenerating segment configs in {configs_dir}")
        seg_configs = generate_segment_configs(
            args.base_config, segments, large_lake_ids, configs_dir,
        )

        # Submit each segment
        print(f"\nSubmitting {len(seg_configs)} segment jobs...")
        seg_run_roots = []
        for sc in seg_configs:
            print(f"\n--- Segment {sc['seg_id']} ---")
            rr = submit_segment(sc["config_path"], verbose)
            seg_run_roots.append(rr)
            print(f"  run_root: {rr}")

        # Save manifest
        manifest = create_manifest(
            args.base_config, seg_configs, seg_run_roots,
            segments, large_lake_ids, merge_root,
        )

        manifest_path = os.path.join(merge_root, "segment_manifest.json")

        # If --all, poll until jobs complete
        if args.all:
            print("\nWaiting for SLURM jobs to complete...")
            print("(You can also Ctrl+C and run --merge later with --manifest)")
            try:
                # Simple poll: check if all post files exist
                # More robust: poll squeue
                _poll_until_complete(manifest, args.poll_interval, verbose)
            except KeyboardInterrupt:
                print("\nPolling interrupted. Run --merge --manifest later when jobs complete.")
                return

    # -------------------------------------------------------------------------
    # Load manifest if not from submit phase
    # -------------------------------------------------------------------------
    if manifest is None and args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        print(f"Loaded manifest: {args.manifest}")

    if manifest is None and (args.merge or args.validate):
        # Try to find manifest from merge_root
        if args.merge_root:
            manifest_path = os.path.join(args.merge_root, "segment_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
        if manifest is None:
            parser.error("--merge/--validate requires --manifest or prior --submit")

    # -------------------------------------------------------------------------
    # Phase 3: Merge
    # -------------------------------------------------------------------------
    if args.merge and manifest:
        print(f"\n{'='*70}")
        print("MERGING SEGMENT OUTPUTS")
        print(f"{'='*70}")

        from merge_segments import merge_lake, verify_merge

        seg_run_roots = [s["run_root"] for s in manifest["segments"]]
        merge_root = manifest["merge_root"]
        lakes = manifest["lakes"]

        all_summaries = {}
        for lake_id in lakes:
            try:
                summary = merge_lake(lake_id, seg_run_roots, merge_root, args.alpha, verbose)
                all_summaries[lake_id] = summary

                if args.pilot:
                    checks = verify_merge(lake_id, merge_root, args.alpha, verbose)
                    all_summaries[lake_id]["verification"] = checks

            except Exception as e:
                print(f"\nERROR merging lake {lake_id}: {e}")
                if args.pilot:
                    traceback.print_exc()
                all_summaries[lake_id] = {"status": "FAILED", "error": str(e)}

        # Save merge summary
        summary_path = os.path.join(merge_root, "merge_summary.json")
        _save_json(all_summaries, summary_path)
        print(f"\nMerge summary: {summary_path}")

        ok = sum(1 for v in all_summaries.values() if v.get("status") == "ok")
        fail = len(all_summaries) - ok
        print(f"Results: {ok} succeeded, {fail} failed")

    # -------------------------------------------------------------------------
    # Phase 4: Validate
    # -------------------------------------------------------------------------
    if args.validate and manifest:
        print(f"\n{'='*70}")
        print("CV VALIDATION ON MERGED OUTPUTS")
        print(f"{'='*70}")

        merge_root = manifest["merge_root"]
        lakes = manifest["lakes"]

        from cv_validation import validate_lake, METRIC_NAMES
        import csv

        histogram_dir = os.path.join(merge_root, "cv_histograms")
        all_results = []

        for lake_id in lakes:
            results = validate_lake(
                merge_root, lake_id, args.alpha, verbose,
                histogram_dir=histogram_dir,
            )
            all_results.extend(results)

        # Write CSV
        if all_results:
            csv_path = os.path.join(merge_root, "cv_results_merged.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['lake_id', 'alpha', 'method'] + METRIC_NAMES
                writer.writerow(header)
                for r in all_results:
                    row = [r.lake_id, r.alpha, r.method]
                    for m in METRIC_NAMES:
                        val = getattr(r, m)
                        if isinstance(val, float):
                            row.append(f"{val:.6f}" if not np.isnan(val) else "")
                        else:
                            row.append(val)
                    writer.writerow(row)
            print(f"\nCV results: {csv_path}")
            print(f"Histograms: {histogram_dir}")

        # Pilot comparison
        if args.compare_root and manifest:
            compare_with_full_range(manifest, args.compare_root, args.alpha, verbose)


# =============================================================================
# Helpers
# =============================================================================

def _poll_until_complete(
    manifest: Dict,
    poll_interval: int = 120,
    verbose: bool = True,
):
    """
    Poll until all segment outputs exist.
    Checks for the existence of post files (last stage) for all lakes × segments.
    """
    lakes = manifest["lakes"]
    seg_run_roots = [s["run_root"] for s in manifest["segments"]]
    alpha = "a1000"

    total_expected = len(lakes) * len(seg_run_roots) * 2  # dineof + dincae post files

    while True:
        found = 0
        for lake_id in lakes:
            lake_id9 = f"{lake_id:09d}"
            for rr in seg_run_roots:
                post_dir = os.path.join(rr, "post", lake_id9, alpha)
                if os.path.isdir(post_dir):
                    files = os.listdir(post_dir)
                    if any(f.endswith("_dineof.nc") for f in files):
                        found += 1
                    if any(f.endswith("_dincae.nc") for f in files):
                        found += 1

        pct = 100.0 * found / total_expected if total_expected > 0 else 0
        ts = datetime.now().strftime('%H:%M:%S')

        if verbose:
            print(f"  {ts}: {found}/{total_expected} post files exist ({pct:.0f}%)")

        if found >= total_expected:
            print("  All segment outputs complete!")
            return

        time.sleep(poll_interval)


def _save_json(data: dict, path: str):
    """Save dict to JSON, handling numpy types."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_convert)


# Allow import of traceback for pilot mode error reporting
import traceback

if __name__ == "__main__":
    main()
