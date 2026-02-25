#!/usr/bin/env python3
"""
Segment Run Diagnostics

Scans all segment runs referenced by manifest files and reports:
- Which lakes completed fully, partially, or failed
- Failure reasons extracted from chain logs and DINCAE logs
- Actionable summary: which lakes need resubmission with more splits

Usage:
    # Scan all manifests
    python diagnose_segments.py --scan-dir /gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/

    # Scan a specific manifest
    python diagnose_segments.py --manifest /path/to/segment_manifest.json

    # Scan a single segment run (no manifest needed)
    python diagnose_segments.py --run-root /gws/.../anomaly-20260215-3da5c4-exp0_baseline_both_seg0
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from collections import defaultdict


# ─── Failure pattern matchers ────────────────────────────────────────────────

FAILURE_PATTERNS = [
    # SLURM resource limits
    (r"DUE TO TIME LIMIT", "TIMEOUT"),
    (r"CANCELLED", "CANCELLED"),
    (r"slurmstepd.*CANCELLED", "TIMEOUT"),
    (r"time limit", "TIMEOUT"),
    # Out of memory
    (r"oom-kill", "OOM"),
    (r"Out of memory", "OOM"),
    (r"Killed\b", "OOM_LIKELY"),
    (r"MemoryError", "OOM"),
    (r"Cannot allocate memory", "OOM"),
    # DINCAE / GPU
    (r"CUDA out of memory", "GPU_OOM"),
    (r"CUDA error", "GPU_ERROR"),
    (r"OutOfMemoryError", "GPU_OOM"),
    (r"GPU.*error", "GPU_ERROR"),
    (r"CuError", "GPU_ERROR"),
    # Julia / DINCAE specific
    (r"Julia.*error", "JULIA_ERROR"),
    (r"LoadError", "JULIA_ERROR"),
    (r"BoundsError", "JULIA_ERROR"),
    (r"DomainError", "JULIA_ERROR"),
    # Python errors
    (r"Traceback \(most recent call last\)", "PYTHON_ERROR"),
    (r"Exception:", "PYTHON_ERROR"),
    (r"FileNotFoundError", "FILE_NOT_FOUND"),
    # DINEOF
    (r"DINEOF.*fail", "DINEOF_FAIL"),
    (r"Fortran.*error", "DINEOF_FAIL"),
    (r"STOP", "DINEOF_FAIL"),
]


def scan_log_for_errors(log_path, max_lines=200):
    """Scan a log file for failure patterns. Returns list of (pattern_name, line)."""
    if not os.path.exists(log_path):
        return []

    errors = []
    try:
        with open(log_path, 'r', errors='replace') as f:
            # Read last max_lines (errors usually at the end)
            lines = f.readlines()
            tail = lines[-max_lines:] if len(lines) > max_lines else lines

        for line in tail:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            for pattern, label in FAILURE_PATTERNS:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    # Truncate long lines
                    excerpt = line_stripped[:200]
                    errors.append((label, excerpt))
                    break  # one match per line is enough
    except Exception as e:
        errors.append(("READ_ERROR", str(e)))

    return errors


def get_last_lines(log_path, n=5):
    """Get last n non-empty lines from a log file."""
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, 'r', errors='replace') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            return lines[-n:]
    except Exception:
        return []


# ─── Per-lake diagnostics ────────────────────────────────────────────────────

EXPECTED_POST_SUFFIXES = [
    "_dineof.nc",
    "_dineof_eof_filtered.nc",
    "_dineof_eof_interp_full.nc",
    "_dineof_eof_filtered_interp_full.nc",
    "_dincae.nc",
    "_dincae_interp_full.nc",
]

EXPECTED_DINEOF_FILES = [
    "dineof_results.nc",
    "eofs.nc",
    "eofs_filtered.nc",
    "eofs_interpolated.nc",
    "dineof_results_eof_filtered.nc",
    "dineof_results_eof_interp_full.nc",
    "dineof_results_eof_filtered_interp_full.nc",
]

EXPECTED_DINCAE_FILES = [
    "dincae_results.nc",
    "cv_rms.txt",
    "loss_history.json",
]


def diagnose_lake(run_root, lake_id, alpha="a1000"):
    """Diagnose a single lake within a segment run."""
    lake_id9 = f"{lake_id:09d}"
    result = {
        "lake_id": lake_id,
        "status": "UNKNOWN",
        "stages": {},
        "errors": [],
        "post_files": [],
        "missing_post": [],
    }

    # ── 1. Chain log ──
    log_dir = os.path.join(run_root, "logs")
    chain_out = None
    chain_err = None
    if os.path.isdir(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith(f"chain_lake{lake_id}_") and f.endswith(".out"):
                chain_out = os.path.join(log_dir, f)
            if f.startswith(f"chain_lake{lake_id}_") and f.endswith(".err"):
                chain_err = os.path.join(log_dir, f)

    chain_errors = []
    if chain_out:
        chain_errors += scan_log_for_errors(chain_out)
    if chain_err:
        chain_errors += scan_log_for_errors(chain_err)
    if chain_errors:
        result["errors"].extend(chain_errors)

    # Check chain log for stage progress
    if chain_out:
        last = get_last_lines(chain_out, 10)
        result["chain_tail"] = last

    # ── 2. Prepared ──
    prepared_dir = os.path.join(run_root, "prepared", lake_id9)
    prepared_nc = os.path.join(prepared_dir, "prepared.nc")
    result["stages"]["prepared"] = os.path.exists(prepared_nc)

    # ── 3. DINEOF ──
    dineof_dir = os.path.join(run_root, "dineof", lake_id9, alpha)
    dineof_files = {}
    if os.path.isdir(dineof_dir):
        for ef in EXPECTED_DINEOF_FILES:
            dineof_files[ef] = os.path.exists(os.path.join(dineof_dir, ef))
    result["stages"]["dineof"] = dineof_files
    result["stages"]["dineof_complete"] = all(dineof_files.values()) if dineof_files else False

    # ── 4. DINCAE ──
    dincae_dir = os.path.join(run_root, "dincae", lake_id9, alpha)
    dincae_files = {}
    dincae_checkpoints = 0
    if os.path.isdir(dincae_dir):
        for ef in EXPECTED_DINCAE_FILES:
            dincae_files[ef] = os.path.exists(os.path.join(dincae_dir, ef))
        # Count checkpoints to gauge progress
        dincae_checkpoints = len(glob(os.path.join(dincae_dir, "model-checkpoint-*.jld2")))

        # Scan DINCAE logs
        dincae_out = os.path.join(dincae_dir, f"logs_dincae_{lake_id}.out")
        dincae_err = os.path.join(dincae_dir, f"logs_dincae_{lake_id}.err")
        dincae_errors = []
        if os.path.exists(dincae_out):
            dincae_errors += scan_log_for_errors(dincae_out)
        if os.path.exists(dincae_err):
            dincae_errors += scan_log_for_errors(dincae_err)
        if dincae_errors:
            result["errors"].extend([(f"DINCAE_{lab}", msg) for lab, msg in dincae_errors])
        result["dincae_tail"] = get_last_lines(dincae_err, 5) or get_last_lines(dincae_out, 5)

    result["stages"]["dincae"] = dincae_files
    result["stages"]["dincae_checkpoints"] = dincae_checkpoints
    result["stages"]["dincae_complete"] = dincae_files.get("dincae_results.nc", False)

    # ── 5. Post files ──
    post_dir = os.path.join(run_root, "post", lake_id9, alpha)
    if os.path.isdir(post_dir):
        post_ncs = [f for f in os.listdir(post_dir) if f.endswith(".nc")]
        result["post_files"] = sorted(post_ncs)

        for suffix in EXPECTED_POST_SUFFIXES:
            found = any(f.endswith(suffix) for f in post_ncs)
            if not found:
                result["missing_post"].append(suffix)

        has_plots = os.path.isdir(os.path.join(post_dir, "plots"))
        has_insitu = os.path.isdir(os.path.join(post_dir, "insitu_cv_validation"))
        result["stages"]["plots"] = has_plots
        result["stages"]["insitu"] = has_insitu
    else:
        result["missing_post"] = EXPECTED_POST_SUFFIXES[:]
        result["stages"]["plots"] = False
        result["stages"]["insitu"] = False

    # ── Determine status ──
    n_post = len(result["post_files"])
    if n_post >= 6:
        result["status"] = "COMPLETE"
    elif n_post >= 4:
        result["status"] = "PARTIAL_NO_DINCAE"
    elif n_post >= 1:
        result["status"] = "PARTIAL_DINEOF_ONLY"
    elif result["stages"].get("dineof_complete"):
        result["status"] = "DINEOF_OK_POST_FAILED"
    elif any(result["stages"].get("dineof", {}).values()):
        result["status"] = "DINEOF_PARTIAL"
    elif result["stages"]["prepared"]:
        result["status"] = "PREPARED_ONLY"
    else:
        result["status"] = "NO_OUTPUT"

    # Classify primary failure reason
    error_labels = [e[0] for e in result["errors"]]
    if "TIMEOUT" in error_labels or "DINCAE_TIMEOUT" in error_labels:
        result["failure_reason"] = "TIMEOUT"
    elif "OOM" in error_labels or "GPU_OOM" in error_labels or "DINCAE_OOM" in error_labels:
        result["failure_reason"] = "OOM"
    elif "GPU_ERROR" in error_labels or "DINCAE_GPU_ERROR" in error_labels:
        result["failure_reason"] = "GPU_ERROR"
    elif any("ERROR" in e for e in error_labels):
        result["failure_reason"] = "SOFTWARE_ERROR"
    elif result["status"] not in ("COMPLETE",):
        result["failure_reason"] = "UNKNOWN"
    else:
        result["failure_reason"] = None

    return result


# ─── Manifest-level diagnostics ──────────────────────────────────────────────

def diagnose_manifest(manifest_path, verbose=True):
    """Diagnose all lakes across all segments in a manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    merge_root = manifest.get("merge_root", os.path.dirname(manifest_path))
    lakes = manifest["lakes"]
    segments = manifest["segments"]

    print(f"\n{'='*70}")
    print(f"Manifest: {os.path.basename(os.path.dirname(manifest_path))}")
    print(f"  Lakes: {lakes}")
    print(f"  Segments: {len(segments)}")
    print(f"{'='*70}")

    # Per-lake, per-segment diagnostics
    lake_results = {}
    for lid in lakes:
        lake_results[lid] = {"segments": {}, "best_status": "NO_OUTPUT"}

        for seg in segments:
            seg_id = seg["seg_id"]
            rr = seg["run_root"]

            if not os.path.isdir(rr):
                lake_results[lid]["segments"][seg_id] = {
                    "status": "RUN_ROOT_MISSING",
                    "failure_reason": "MISSING_DIR",
                }
                continue

            diag = diagnose_lake(rr, lid)
            lake_results[lid]["segments"][seg_id] = diag

        # Determine best status across segments (for mergability)
        statuses = [d["status"] for d in lake_results[lid]["segments"].values()
                     if isinstance(d, dict)]
        if all(s == "COMPLETE" for s in statuses):
            lake_results[lid]["best_status"] = "ALL_SEGMENTS_COMPLETE"
        elif all(s in ("COMPLETE", "PARTIAL_NO_DINCAE") for s in statuses):
            lake_results[lid]["best_status"] = "DINEOF_MERGEABLE"
        elif any(s in ("COMPLETE", "PARTIAL_NO_DINCAE", "PARTIAL_DINEOF_ONLY") for s in statuses):
            lake_results[lid]["best_status"] = "PARTIALLY_MERGEABLE"
        else:
            lake_results[lid]["best_status"] = "NOT_MERGEABLE"

    # ── Print summary ──
    STATUS_COLORS = {
        "ALL_SEGMENTS_COMPLETE": "✅",
        "DINEOF_MERGEABLE": "🟡",
        "PARTIALLY_MERGEABLE": "🟠",
        "NOT_MERGEABLE": "❌",
    }

    print(f"\n{'─'*70}")
    print("SUMMARY")
    print(f"{'─'*70}")

    for lid in lakes:
        lr = lake_results[lid]
        icon = STATUS_COLORS.get(lr["best_status"], "❓")
        print(f"\n  {icon} Lake {lid:>6d}  [{lr['best_status']}]")

        for seg_id, diag in sorted(lr["segments"].items()):
            if isinstance(diag, dict):
                status = diag.get("status", "?")
                n_post = len(diag.get("post_files", []))
                reason = diag.get("failure_reason", "")
                reason_str = f"  ← {reason}" if reason and reason != "None" else ""

                # DINCAE progress
                ckpts = diag.get("stages", {}).get("dincae_checkpoints", 0)
                dincae_done = diag.get("stages", {}).get("dincae_complete", False)
                dincae_str = f"dincae={'✓' if dincae_done else f'{ckpts}ckpt'}"

                print(f"    seg{seg_id}: {status:<25s} post={n_post}/6  {dincae_str}{reason_str}")

                if verbose and diag.get("errors"):
                    # Show unique error types
                    unique_errors = set()
                    for label, msg in diag["errors"]:
                        key = label
                        if key not in unique_errors:
                            unique_errors.add(key)
                            print(f"           → {label}: {msg[:120]}")

    # ── Actionable recommendations ──
    print(f"\n{'─'*70}")
    print("RECOMMENDATIONS")
    print(f"{'─'*70}")

    complete = [lid for lid in lakes if lake_results[lid]["best_status"] == "ALL_SEGMENTS_COMPLETE"]
    dineof_only = [lid for lid in lakes if lake_results[lid]["best_status"] == "DINEOF_MERGEABLE"]
    partial = [lid for lid in lakes if lake_results[lid]["best_status"] == "PARTIALLY_MERGEABLE"]
    failed = [lid for lid in lakes if lake_results[lid]["best_status"] == "NOT_MERGEABLE"]

    if complete:
        print(f"\n  Ready to merge (all 6 files): {complete}")
        print(f"    → python scripts/merge_segments.py --manifest {manifest_path} --verify")

    if dineof_only:
        print(f"\n  DINEOF complete, DINCAE failed: {dineof_only}")
        print(f"    Option A: Merge DINEOF-only (4 files per lake)")
        print(f"    Option B: Resubmit DINCAE with more splits / longer walltime")

    if partial:
        print(f"\n  Partially complete (some segments missing): {partial}")
        print(f"    → Check individual segment errors above")
        print(f"    → May need resubmission with more splits")

    if failed:
        print(f"\n  Failed entirely: {failed}")
        print(f"    → Likely need 8+ splits or special handling")

    return lake_results


def diagnose_run_root(run_root, verbose=True):
    """Diagnose a single segment run (without manifest)."""
    print(f"\n{'='*70}")
    print(f"Run root: {os.path.basename(run_root)}")
    print(f"{'='*70}")

    # Discover lakes from prepared/ directory
    prepared_dir = os.path.join(run_root, "prepared")
    if not os.path.isdir(prepared_dir):
        print("  No prepared/ directory found")
        return {}

    lake_ids = []
    for d in sorted(os.listdir(prepared_dir)):
        if d.isdigit() or (len(d) == 9 and d.isdigit()):
            lake_ids.append(int(d))

    print(f"  Lakes found: {lake_ids}")

    results = {}
    for lid in lake_ids:
        diag = diagnose_lake(run_root, lid)
        results[lid] = diag

        status = diag["status"]
        n_post = len(diag["post_files"])
        reason = diag.get("failure_reason", "")
        reason_str = f"  ← {reason}" if reason else ""

        ckpts = diag["stages"].get("dincae_checkpoints", 0)
        dincae_done = diag["stages"].get("dincae_complete", False)

        icon = "✅" if status == "COMPLETE" else "🟡" if "PARTIAL" in status else "❌"
        print(f"  {icon} Lake {lid:>6d}: {status:<25s} post={n_post}/6  dincae={'✓' if dincae_done else f'{ckpts}ckpt'}{reason_str}")

        if verbose and diag.get("errors"):
            unique = set()
            for label, msg in diag["errors"]:
                if label not in unique:
                    unique.add(label)
                    print(f"       → {label}: {msg[:120]}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose segment runs: find failures, suggest fixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan-dir",
                       help="Scan directory for all segment_manifest.json files")
    group.add_argument("--manifest",
                       help="Diagnose a specific manifest")
    group.add_argument("--run-root",
                       help="Diagnose a single segment run (no manifest)")

    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress detailed error messages")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.scan_dir:
        # Find all manifests
        manifests = sorted(glob(os.path.join(args.scan_dir, "**/segment_manifest.json"),
                                recursive=True))
        if not manifests:
            print(f"No segment_manifest.json found under {args.scan_dir}")
            sys.exit(1)

        print(f"Found {len(manifests)} manifests")
        all_results = {}
        for mf in manifests:
            key = os.path.basename(os.path.dirname(mf))
            all_results[key] = diagnose_manifest(mf, verbose)

        if args.json:
            # Serialize (skip non-serializable bits)
            print(json.dumps(all_results, indent=2, default=str))

    elif args.manifest:
        results = diagnose_manifest(args.manifest, verbose)
        if args.json:
            print(json.dumps(results, indent=2, default=str))

    elif args.run_root:
        results = diagnose_run_root(args.run_root, verbose)
        if args.json:
            print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
