#!/usr/bin/env python3
"""
segment_pipeline.py - Unified orchestrator for temporal-split lake processing.

Manages the full lifecycle of processing large lakes that require temporal
segmentation: splitting, gap-filling (DINEOF + DINCAE), post-processing,
diagnosis, recovery, merging, validation, and patching back into the main
experiment directory.

Replaces manual use of: temporal_split_runner.py, diagnose_segments.py,
rerun_dincae.py, retrofit_post.py, merge_segments.py

State is tracked in segment_status.json so the pipeline can be re-run
at any point and will pick up where it left off.

Stages:
  process   - Submit segment processing jobs (pre + DINEOF + DINCAE + post)
  diagnose  - Scan all segments, classify status per lake
  recover   - Auto-retry failures (GPU OOM → batch_size/2, RAM OOM → more mem)
  post      - Run post-processing on segments where it's missing
  merge     - Combine segments into unified outputs
  validate  - Generate plots + insitu validation on merged outputs
  finalize  - Verify outputs and prepare for patching to main experiment
  auto      - Walk through all stages in order, skipping completed work

Usage:
  # Initial: create segments + submit processing
  python segment_pipeline.py \\
    --manifest /gws/.../segment_manifest.json \\
    --config configs/exp0_baseline.json \\
    --main-run /gws/.../anomaly-20260131-exp0_baseline_both \\
    --stage auto

  # Check status
  python segment_pipeline.py --manifest ... --stage diagnose

  # Fix failures
  python segment_pipeline.py --manifest ... --stage recover

  # After recovery completes, run remaining stages
  python segment_pipeline.py --manifest ... --stage auto

Author: Shaerdan / NCEO / University of Reading
Date: February 2026
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Imports from sibling scripts (they live in the same scripts/ directory)
# ---------------------------------------------------------------------------
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

EXPECTED_POST_SUFFIXES = [
    "_dineof.nc",
    "_dineof_eof_filtered.nc",
    "_dineof_eof_interp_full.nc",
    "_dineof_eof_filtered_interp_full.nc",
    "_dincae.nc",
    "_dincae_interp_full.nc",
]

EXPECTED_DINCAE_FILES = [
    "dincae_results.nc",
    "cv_rms.txt",
]

FAILURE_PATTERNS = [
    (r"DUE TO TIME LIMIT", "TIMEOUT"),
    (r"CANCELLED", "CANCELLED"),
    (r"slurmstepd.*CANCELLED", "TIMEOUT"),
    (r"time limit", "TIMEOUT"),
    (r"oom-kill", "OOM"),
    (r"Out of memory", "OOM"),
    (r"Killed\b", "OOM_LIKELY"),
    (r"MemoryError", "OOM"),
    (r"Cannot allocate memory", "OOM"),
    (r"Out of GPU memory", "GPU_OOM"),
    (r"CUDA out of memory", "GPU_OOM"),
    (r"CUDA error", "GPU_ERROR"),
    (r"OutOfMemoryError", "GPU_OOM"),
    (r"CuError", "GPU_ERROR"),
    (r"LoadError", "JULIA_ERROR"),
    (r"BoundsError", "JULIA_ERROR"),
    (r"Traceback \(most recent call last\)", "PYTHON_ERROR"),
    (r"FileNotFoundError", "FILE_NOT_FOUND"),
    (r"Signals\.SIGKILL", "OOM"),
]


# ═══════════════════════════════════════════════════════════════════════════
# State file management
# ═══════════════════════════════════════════════════════════════════════════

def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def load_status(manifest_dir: str) -> dict:
    """Load or create segment_status.json."""
    path = os.path.join(manifest_dir, "segment_status.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"lakes": {}, "flags": [], "last_stage": None, "last_run": None}


def save_status(manifest_dir: str, status: dict):
    """Save segment_status.json."""
    status["last_run"] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(manifest_dir, "segment_status.json")
    with open(path, "w") as f:
        json.dump(status, f, indent=2, default=_json_serializer)


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic helpers (from diagnose_segments.py)
# ═══════════════════════════════════════════════════════════════════════════

def scan_log_for_errors(log_path: str, max_lines: int = 200) -> list:
    """Scan a log file for failure patterns."""
    if not os.path.exists(log_path):
        return []
    errors = []
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
            tail = lines[-max_lines:] if len(lines) > max_lines else lines
        for line in tail:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            for pattern, label in FAILURE_PATTERNS:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    errors.append((label, line_stripped[:200]))
                    break
    except Exception as e:
        errors.append(("READ_ERROR", str(e)))
    return errors


def diagnose_lake_segment(run_root: str, lake_id: int, alpha: str = "a1000") -> dict:
    """Diagnose a single lake within a single segment run."""
    lake_id9 = f"{lake_id:09d}"
    result = {
        "lake_id": lake_id,
        "status": "UNKNOWN",
        "errors": [],
        "post_files": [],
        "missing_post": [],
    }

    # --- Chain log ---
    log_dir = os.path.join(run_root, "logs")
    chain_errors = []
    if os.path.isdir(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith(f"chain_lake{lake_id}_") and f.endswith((".out", ".err")):
                chain_errors += scan_log_for_errors(os.path.join(log_dir, f))
    result["errors"].extend(chain_errors)

    # --- Prepared ---
    prepared = os.path.exists(os.path.join(run_root, "prepared", lake_id9, "prepared.nc"))
    result["prepared"] = prepared

    # --- DINEOF ---
    dineof_dir = os.path.join(run_root, "dineof", lake_id9, alpha)
    dineof_results = os.path.exists(os.path.join(dineof_dir, "dineof_results.nc"))
    result["dineof_complete"] = dineof_results

    # --- DINCAE ---
    dincae_dir = os.path.join(run_root, "dincae", lake_id9, alpha)
    dincae_complete = False
    dincae_results_ready = False
    dincae_checkpoints = 0
    if os.path.isdir(dincae_dir):
        dincae_complete = os.path.exists(os.path.join(dincae_dir, "cv_rms.txt"))
        dincae_results_ready = os.path.exists(os.path.join(dincae_dir, "dincae_results.nc"))
        dincae_checkpoints = len(glob(os.path.join(dincae_dir, "model-checkpoint-*.jld2")))

        # Scan DINCAE logs
        for suffix in (".out", ".err"):
            log_path = os.path.join(dincae_dir, f"logs_dincae_{lake_id}{suffix}")
            dincae_errors = scan_log_for_errors(log_path)
            result["errors"].extend(
                [(f"DINCAE_{lab}", msg) for lab, msg in dincae_errors]
            )

    result["dincae_complete"] = dincae_complete
    result["dincae_results_ready"] = dincae_results_ready
    result["dincae_checkpoints"] = dincae_checkpoints

    # --- Post files ---
    post_dir = os.path.join(run_root, "post", lake_id9, alpha)
    if os.path.isdir(post_dir):
        post_ncs = sorted(f for f in os.listdir(post_dir) if f.endswith(".nc"))
        result["post_files"] = post_ncs
        for suffix in EXPECTED_POST_SUFFIXES:
            if not any(f.endswith(suffix) for f in post_ncs):
                result["missing_post"].append(suffix)

    # --- Determine status ---
    n_post = len(result["post_files"])
    if n_post >= 6:
        result["status"] = "COMPLETE"
    elif n_post >= 4:
        result["status"] = "PARTIAL_NO_DINCAE"
    elif n_post >= 1:
        result["status"] = "PARTIAL_DINEOF_ONLY"
    elif dineof_results and dincae_complete:
        result["status"] = "GAPFILL_OK_POST_FAILED"
    elif dineof_results:
        result["status"] = "DINEOF_OK_DINCAE_FAILED"
    elif prepared:
        result["status"] = "PREPARED_ONLY"
    else:
        result["status"] = "NO_OUTPUT"

    # --- Classify failure ---
    # The key question: is the blocker gap-filling or post-processing?
    # If gap-filling is done (DINEOF + DINCAE), any missing post files are a post issue.
    # Old chain log errors are from the gap-filling phase which already succeeded.
    both_gapfill_done = bool(dineof_results) and dincae_complete
    dineof_only_done = bool(dineof_results) and not dincae_complete

    if result["status"] == "COMPLETE":
        result["failure_reason"] = None
    elif both_gapfill_done:
        # Both methods finished — post-processing is the only blocker
        result["failure_reason"] = "POST_NOT_RUN"
    elif dineof_only_done and n_post < 4:
        # DINEOF done, DINCAE failed, and even DINEOF post is incomplete
        # Check if DINCAE had GPU OOM
        error_labels = [e[0] for e in result["errors"]]
        if any(l in error_labels for l in ("GPU_OOM", "DINCAE_GPU_OOM")):
            result["failure_reason"] = "GPU_OOM"
        else:
            result["failure_reason"] = "DINCAE_FAILED"
    elif dineof_only_done and n_post >= 4:
        # DINEOF post done, DINCAE failed
        error_labels = [e[0] for e in result["errors"]]
        if any(l in error_labels for l in ("GPU_OOM", "DINCAE_GPU_OOM")):
            result["failure_reason"] = "GPU_OOM"
        else:
            result["failure_reason"] = "DINCAE_FAILED"
    else:
        error_labels = [e[0] for e in result["errors"]]
        if "TIMEOUT" in error_labels:
            result["failure_reason"] = "TIMEOUT"
        elif any(l in error_labels for l in ("GPU_OOM", "DINCAE_GPU_OOM")):
            result["failure_reason"] = "GPU_OOM"
        elif any(l in error_labels for l in ("OOM", "OOM_LIKELY", "DINCAE_OOM")):
            result["failure_reason"] = "RAM_OOM"
        elif any("ERROR" in l for l in error_labels):
            result["failure_reason"] = "SOFTWARE_ERROR"
        else:
            result["failure_reason"] = "UNKNOWN"

    return result


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: DIAGNOSE
# ═══════════════════════════════════════════════════════════════════════════

def stage_diagnose(manifest: dict, status: dict, lake_ids: list,
                   alpha: str, verbose: bool) -> dict:
    """Scan all segments and classify each lake's status."""
    segments = manifest["segments"]
    all_lakes = lake_ids or manifest["lakes"]

    print(f"\n{'='*70}")
    print(f"STAGE: DIAGNOSE")
    print(f"{'='*70}")
    print(f"  Lakes: {all_lakes}")
    print(f"  Segments: {len(segments)}")

    for lid in all_lakes:
        lid_str = str(lid)
        lake_entry = status["lakes"].setdefault(lid_str, {"segments": {}})
        seg_statuses = []

        for seg in segments:
            seg_id = str(seg["seg_id"])
            rr = seg["run_root"]

            if not os.path.isdir(rr):
                lake_entry["segments"][seg_id] = {
                    "status": "RUN_ROOT_MISSING",
                    "run_root": rr,
                }
                seg_statuses.append("RUN_ROOT_MISSING")
                continue

            diag = diagnose_lake_segment(rr, lid, alpha)
            # Store diagnosis (strip large error messages for state file)
            lake_entry["segments"][seg_id] = {
                "run_root": rr,
                "status": diag["status"],
                "prepared": diag["prepared"],
                "dineof_complete": diag["dineof_complete"],
                "dincae_complete": diag["dincae_complete"],
                "dincae_results_ready": diag["dincae_results_ready"],
                "dincae_checkpoints": diag["dincae_checkpoints"],
                "post_files": len(diag["post_files"]),
                "missing_post": diag["missing_post"],
                "failure_reason": diag["failure_reason"],
                "n_errors": len(diag["errors"]),
            }
            seg_statuses.append(diag["status"])

        # Determine overall lake status
        if all(s == "COMPLETE" for s in seg_statuses):
            lake_entry["overall"] = "ALL_COMPLETE"
        elif all(s in ("COMPLETE", "PARTIAL_NO_DINCAE") for s in seg_statuses):
            lake_entry["overall"] = "DINEOF_MERGEABLE"
        elif any(s == "COMPLETE" for s in seg_statuses):
            lake_entry["overall"] = "PARTIAL"
        else:
            lake_entry["overall"] = "NOT_READY"

    # --- Print summary ---
    STATUS_ICONS = {
        "ALL_COMPLETE": "✅",
        "DINEOF_MERGEABLE": "🟡",
        "PARTIAL": "🟠",
        "NOT_READY": "❌",
    }

    print(f"\n{'─'*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'─'*70}")

    for lid in all_lakes:
        lid_str = str(lid)
        le = status["lakes"].get(lid_str, {})
        overall = le.get("overall", "?")
        icon = STATUS_ICONS.get(overall, "❓")
        print(f"\n  {icon} Lake {lid:>6d}  [{overall}]")

        for seg_id in sorted(le.get("segments", {}).keys(), key=int):
            seg = le["segments"][seg_id]
            s = seg.get("status", "?")
            n_post = seg.get("post_files", 0)
            reason = seg.get("failure_reason", "")
            reason_str = f"  ← {reason}" if reason else ""
            dincae_done = seg.get("dincae_complete", False)
            dincae_res = seg.get("dincae_results_ready", False)
            ckpts = seg.get("dincae_checkpoints", 0)
            if dincae_done and dincae_res:
                dincae_str = "dincae=✓"
            elif dincae_done and not dincae_res:
                dincae_str = "dincae=trained,no_results.nc"
            else:
                dincae_str = f"dincae={ckpts}ckpt"
            print(f"    seg{seg_id}: {s:<28s} post={n_post}/6  {dincae_str}{reason_str}")

    # --- Recommendations ---
    complete = [lid for lid in all_lakes
                if status["lakes"].get(str(lid), {}).get("overall") == "ALL_COMPLETE"]
    needs_recover = [lid for lid in all_lakes
                     if status["lakes"].get(str(lid), {}).get("overall") in ("PARTIAL", "NOT_READY", "DINEOF_MERGEABLE")]
    needs_merge = [lid for lid in all_lakes
                   if status["lakes"].get(str(lid), {}).get("overall") == "ALL_COMPLETE"]

    print(f"\n{'─'*70}")
    print("NEXT STEPS")
    print(f"{'─'*70}")
    if needs_merge:
        print(f"  Ready to merge: {needs_merge}")
    if needs_recover:
        print(f"  Needs recovery: {needs_recover}")
        print(f"    → Run with --stage recover")
    if not needs_recover and not needs_merge:
        print(f"  Nothing to do — all lakes at current best state")

    status["last_stage"] = "diagnose"
    return status


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Recover missing dincae_results.nc from data-avg.nc
# ═══════════════════════════════════════════════════════════════════════════

def _recover_missing_dincae_results(manifest: dict, lake_ids: list,
                                     alpha: str, dry_run: bool):
    """
    Find segments where Julia DINCAE training finished (cv_rms.txt + data-avg.nc
    exist) but dincae_results.nc was never created (chain died before
    dincae_write_out ran, or rerun_dincae.py didn't include the reshape step).

    Submits a SLURM array job to create dincae_results.nc for each such segment.
    CPU-only, no GPU needed, ~32GB RAM, ~10 min per segment.

    Returns the SLURM job ID (str) if jobs were submitted, or None.
    """
    segments = manifest["segments"]
    all_lakes = lake_ids or manifest["lakes"]
    to_fix = []

    # Scan for the pattern
    for seg in segments:
        rr = seg["run_root"]
        if not os.path.isdir(rr):
            continue
        for lid in all_lakes:
            lake_id9 = f"{lid:09d}"
            dincae_dir = os.path.join(rr, "dincae", lake_id9, alpha)
            cv_rms = os.path.join(dincae_dir, "cv_rms.txt")
            results_nc = os.path.join(dincae_dir, "dincae_results.nc")
            data_avg = os.path.join(dincae_dir, "data-avg.nc")
            prepared_nc = os.path.join(rr, "prepared", lake_id9, "prepared.nc")

            if (os.path.exists(cv_rms)
                    and not os.path.exists(results_nc)
                    and os.path.exists(data_avg)
                    and os.path.exists(prepared_nc)):
                to_fix.append((rr, lid, dincae_dir, prepared_nc,
                               seg.get("seg_id", "?")))

    if not to_fix:
        return None

    print(f"\n  DINCAE results recovery: {len(to_fix)} segments need dincae_results.nc")
    for rr, lid, dincae_dir, prepared_nc, seg_id in to_fix:
        print(f"    Lake {lid} seg{seg_id}: data-avg.nc exists, dincae_results.nc missing")

    if dry_run:
        print(f"    [DRY-RUN] Would submit SLURM array for {len(to_fix)} segments")
        return None

    # Write job list
    manifest_dir = os.path.dirname(manifest.get("_manifest_path", ""))
    retrofit_dir = os.path.join(manifest_dir, "retrofit")
    os.makedirs(retrofit_dir, exist_ok=True)

    job_list_path = os.path.join(retrofit_dir, "recover_dincae_results_jobs.txt")
    job_list = []
    for rr, lid, dincae_dir, prepared_nc, seg_id in to_fix:
        # Tab-separated: dincae_dir, prepared_nc, lake_id
        job_list.append(f"{dincae_dir}\t{prepared_nc}\t{lid}")

    with open(job_list_path, "w") as f:
        f.write("\n".join(job_list) + "\n")

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    central_logs = os.path.join(REPO_ROOT, "logs")

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=recover_dincae_results
#SBATCH --array=1-{len(job_list)}
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=eocis_chuk
#SBATCH -o {retrofit_dir}/recover_dincae_results_%a.out
#SBATCH -e {retrofit_dir}/recover_dincae_results_%a.err

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_list_path})
DINCAE_DIR=$(echo "$LINE" | cut -f1)
PREPARED_NC=$(echo "$LINE" | cut -f2)
LAKE_ID=$(echo "$LINE" | cut -f3)

# Symlink to central logs
ln -sf {retrofit_dir}/recover_dincae_results_${{SLURM_ARRAY_TASK_ID}}.out \\
   {central_logs}/recover_dincae_results_lake${{LAKE_ID}}_${{SLURM_JOB_ID}}.out
ln -sf {retrofit_dir}/recover_dincae_results_${{SLURM_ARRAY_TASK_ID}}.err \\
   {central_logs}/recover_dincae_results_lake${{LAKE_ID}}_${{SLURM_JOB_ID}}.err

echo "DINCAE results recovery: lake $LAKE_ID"
echo "  DINCAE dir: $DINCAE_DIR"
echo "  Prepared:   $PREPARED_NC"
echo "  Node: $(hostname), Job: $SLURM_JOB_ID, Start: $(date)"

cd {REPO_ROOT}
python -c "
import sys, json, shutil, pathlib
sys.path.insert(0, 'src')
from dincae_arm import PreparedNC, DincaeArtifacts, write_dineof_shaped_outputs

dincae_dir = pathlib.Path('$DINCAE_DIR')
prepared = PreparedNC(pathlib.Path('$PREPARED_NC'))

arts = DincaeArtifacts(
    dincae_dir=dincae_dir,
    prepared_datetime=dincae_dir / 'prepared_datetime.nc',
    prepared_cropped=dincae_dir / 'prepared_datetime_cropped.nc',
    prepared_cropped_cv=dincae_dir / 'prepared_datetime_cropped_add_clouds.nc',
    prepared_cropped_clean=dincae_dir / 'prepared_datetime_cropped_add_clouds.clean.nc',
    pred_path=dincae_dir / 'data-avg.nc',
)

# Verify intermediate files
for name in ('prepared_cropped', 'prepared_cropped_clean'):
    p = getattr(arts, name)
    if not p.exists():
        print(f'ERROR: missing {{p}}')
        sys.exit(1)

# Load config
dcfg = {{}}
cfg_path = dincae_dir / 'dincae_config.json'
if cfg_path.exists():
    dcfg = json.loads(cfg_path.read_text())

# Run write_dineof_shaped_outputs (same as lswtctl.py)
out = write_dineof_shaped_outputs(
    arts=arts, prepared=prepared,
    post_dir=dincae_dir,
    final_front_name='__tmp_dincae_for_post__',
    cfg=dcfg)

# Copy to dincae_results.nc
tmp = out['output_nc']
results = dincae_dir / 'dincae_results.nc'
if tmp and pathlib.Path(tmp).exists():
    shutil.copy2(tmp, results)
    pathlib.Path(tmp).unlink(missing_ok=True)
    print(f'Created: {{results}} ({{results.stat().st_size / 1e6:.0f}} MB)')
else:
    print('ERROR: write_dineof_shaped_outputs produced no output')
    sys.exit(1)
"

echo "Done: $(date)"
"""

    slurm_path = os.path.join(retrofit_dir, "recover_dincae_results.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

    result = subprocess.run(
        ["sbatch", slurm_path],
        capture_output=True, text=True,
    )
    slurm_job_id = None
    if result.returncode == 0:
        print(f"    Submitted: {result.stdout.strip()}")
        print(f"    Job list:  {job_list_path}")
        print(f"    Logs:      {retrofit_dir}/recover_dincae_results_*.out")
        # Extract job ID for dependency chaining
        # sbatch output: "Submitted batch job 12345"
        parts = result.stdout.strip().split()
        if len(parts) >= 4:
            slurm_job_id = parts[-1]
    else:
        print(f"    ✗ sbatch failed: {result.stderr.strip()}")

    return slurm_job_id


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: RECOVER
# ═══════════════════════════════════════════════════════════════════════════

def stage_recover(manifest: dict, status: dict, lake_ids: list,
                  alpha: str, config: dict, batch_size: int,
                  dry_run: bool, verbose: bool) -> dict:
    """Auto-recover failed segments based on failure classification."""
    segments = manifest["segments"]
    all_lakes = lake_ids or manifest["lakes"]
    manifest_dir = os.path.dirname(manifest.get("_manifest_path", ""))

    print(f"\n{'='*70}")
    print(f"STAGE: RECOVER")
    print(f"{'='*70}")

    # First run diagnose if not already done
    if not any(status["lakes"].get(str(lid), {}).get("segments")
               for lid in all_lakes):
        print("  Running diagnose first...")
        status = stage_diagnose(manifest, status, lake_ids, alpha, verbose)

    # --- Pre-step: recover missing dincae_results.nc ---
    dincae_results_job_id = _recover_missing_dincae_results(
        manifest, lake_ids, alpha, dry_run)

    post_full_jobs = []       # (run_root, lake_id) for FULL post from scratch (post=0/6)
    post_retrofit_jobs = []   # (run_root, lake_id) for retrofit (post=1-5/6)
    gpu_oom_jobs = []       # (run_root, lake_id) for DINCAE resubmit
    timeout_flags = []      # (lake_id, seg_id, run_root) for manual review
    other_flags = []        # everything else

    for lid in all_lakes:
        lid_str = str(lid)
        le = status["lakes"].get(lid_str, {})
        if le.get("overall") == "ALL_COMPLETE":
            continue

        for seg_id_str, seg in le.get("segments", {}).items():
            seg_status = seg.get("status", "")
            failure = seg.get("failure_reason", "")
            rr = seg.get("run_root", "")
            n_post = seg.get("post_files", 0)

            if seg_status == "COMPLETE":
                continue

            if (seg_status == "GAPFILL_OK_POST_FAILED"
                    or failure in ("RAM_OOM", "POST_NOT_RUN")):
                # Gap-filling done (or at least DINEOF done), post needs (re)running
                if seg.get("dineof_complete"):
                    if n_post == 0:
                        post_full_jobs.append((rr, lid))
                    else:
                        post_retrofit_jobs.append((rr, lid))

            elif failure == "GPU_OOM":
                # DINCAE GPU OOM — need to rerun with smaller batch
                gpu_oom_jobs.append((rr, lid))

            elif failure == "TIMEOUT":
                timeout_flags.append((lid, seg_id_str, rr))

            elif seg_status not in ("COMPLETE",) and failure:
                other_flags.append((lid, seg_id_str, rr, failure))

    # --- Report ---
    print(f"\n  Post FULL (from scratch):  {len(post_full_jobs)} jobs")
    print(f"  Post RETROFIT (partial):   {len(post_retrofit_jobs)} jobs")
    print(f"  GPU OOM (DINCAE rerun):    {len(gpu_oom_jobs)} jobs")
    print(f"  Timeout (manual review):   {len(timeout_flags)} flags")
    print(f"  Other (manual review):     {len(other_flags)} flags")

    # --- Handle GPU OOM: resubmit DINCAE with smaller batch ---
    if gpu_oom_jobs:
        print(f"\n  GPU OOM recovery (batch_size={batch_size}):")
        for rr, lid in gpu_oom_jobs:
            lake_id9 = f"{lid:09d}"
            dincae_dir = os.path.join(rr, "dincae", lake_id9, alpha)
            config_path = os.path.join(dincae_dir, "dincae_config.json")

            if not os.path.exists(config_path):
                print(f"    Lake {lid} seg {os.path.basename(rr)}: no dincae_config.json, SKIP")
                continue

            with open(config_path) as f:
                dincae_cfg = json.load(f)

            old_bs = dincae_cfg.get("train", {}).get("batch_size",
                     dincae_cfg.get("batch_size", 32))

            if old_bs <= batch_size and old_bs != 32:
                print(f"    Lake {lid}: batch_size already {old_bs} (≤ {batch_size}), SKIP")
                continue

            print(f"    Lake {lid} in {os.path.basename(rr)}: batch {old_bs} → {batch_size}")

            if not dry_run:
                # Patch config
                train = dincae_cfg.setdefault("train", {})
                train["batch_size"] = batch_size
                if "batch_size" in dincae_cfg:
                    dincae_cfg["batch_size"] = batch_size
                with open(config_path, "w") as f:
                    json.dump(dincae_cfg, f, indent=2)

                # Clear old outputs
                for fname in os.listdir(dincae_dir):
                    if fname in ("data-avg.nc", "cv_rms.txt", "loss_history.json",
                                 "lake_cleanup.nc"):
                        os.remove(os.path.join(dincae_dir, fname))
                    elif fname.startswith("data-avg-epoch") and fname.endswith(".nc"):
                        os.remove(os.path.join(dincae_dir, fname))

                # Try to regenerate and submit DINCAE
                try:
                    from dincae_arm.contracts import DincaeArtifacts
                    from dincae_arm.dincae_runner import _generate_julia_script

                    d = Path(dincae_dir)
                    arts = DincaeArtifacts(
                        dincae_dir=d,
                        prepared_datetime=d / "prepared_datetime.nc",
                        prepared_cropped=d / "prepared_datetime_cropped.nc",
                        prepared_cropped_cv=d / "prepared_datetime_cropped_add_clouds.nc",
                        prepared_cropped_clean=d / "prepared_datetime_cropped_add_clouds.clean.nc",
                    )
                    script_path = _generate_julia_script(arts, dincae_cfg)

                    # Build SLURM job
                    slurm_cfg = dincae_cfg.get("slurm", {})
                    exclude = slurm_cfg.get("exclude", "gpuhost004,gpuhost007,gpuhost012,gpuhost016")
                    julia = dincae_cfg.get("runner", {}).get("julia_exe", "julia")
                    env_lines = []
                    denv = dincae_cfg.get("env", {}).get("dincae", {})
                    if denv.get("module_load"):
                        env_lines.append(denv["module_load"])
                    if denv.get("activate"):
                        env_lines.append(denv["activate"])
                    jp = dincae_cfg.get("runner", {}).get("JULIA_PROJECT")
                    if jp:
                        env_lines.append(f'export JULIA_PROJECT="{jp}"')
                    shared_depot = dincae_cfg.get("runner", {}).get("julia_depot", "$HOME/.julia_cuda_depot")
                    env_lines += [
                        "export JULIA_PKG_PRECOMPILE_AUTO=0",
                        f'export JULIA_DEPOT_PATH="{shared_depot}"',
                        f'mkdir -p "{shared_depot}"',
                        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
                        'export CUDA_PATH="${CONDA_PREFIX}"',
                    ]

                    log_out = d / f"logs_dincae_{lid}.out"
                    log_err = d / f"logs_dincae_{lid}.err"
                    for lp in [log_out, log_err]:
                        if lp.exists():
                            lp.rename(lp.with_suffix(lp.suffix + ".prev"))

                    exclude_line = f"#SBATCH --exclude={exclude}" if exclude else ""
                    gpus = int(slurm_cfg.get("gpus", 1))
                    cpus = int(slurm_cfg.get("cpus", 4))
                    sb = f"""#!/bin/bash
#SBATCH --job-name=dincae_recover_{lid}
{exclude_line}
#SBATCH --time={slurm_cfg.get('time', '24:00:00')}
#SBATCH --mem={slurm_cfg.get('mem', '128G')}
#SBATCH --partition={slurm_cfg.get('partition', 'orchid')}
#SBATCH --account={slurm_cfg.get('account', 'orchid')}
#SBATCH --qos={slurm_cfg.get('qos', 'orchid')}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --chdir={dincae_dir}
#SBATCH -o {log_out}
#SBATCH -e {log_err}

cd {dincae_dir}
{chr(10).join(env_lines)}

set -euo pipefail
echo "DINCAE Recovery (batch_size={batch_size})"
echo "Node: $(hostname), Job: $SLURM_JOB_ID, Start: $(date)"

{julia} -e 'using CUDA, cuDNN, DINCAE; println("OK")'
srun --gres=gpu:{gpus} --ntasks=1 --cpus-per-task={cpus} nvidia-smi
exec srun --gres=gpu:{gpus} --ntasks=1 --cpus-per-task={cpus} stdbuf -oL -eL {julia} --project {script_path.name}
"""
                    jobfile = d / f"recover_dincae_{lid}.slurm"
                    jobfile.write_text(sb)

                    result = subprocess.run(
                        ["sbatch", str(jobfile)],
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        print(f"      Submitted: {result.stdout.strip()}")
                    else:
                        print(f"      sbatch failed: {result.stderr.strip()}")

                except ImportError as e:
                    print(f"      Could not import pipeline code: {e}")
                    print(f"      Use rerun_dincae.py manually for this lake")
                except Exception as e:
                    print(f"      Error: {e}")

    # --- Handle post: FULL from scratch (post=0/6) ---
    full_post_job_id = None
    if post_full_jobs:
        print(f"\n  Full post-processing (from scratch, high QOS + 192G):")
        full_post_job_id = _submit_full_post_jobs(
            post_full_jobs, manifest, config, alpha,
            dry_run, verbose,
            dependency_job_id=dincae_results_job_id)

    # --- Handle post: RETROFIT (post=1-5/6) ---
    if post_retrofit_jobs:
        print(f"\n  Retrofit post-processing (high QOS + 192G):")
        # Depend on both dincae_results and full_post (if any), to avoid
        # SLURM resource contention
        dep_id = full_post_job_id or dincae_results_job_id
        _submit_post_recovery_jobs(post_retrofit_jobs, manifest, config, alpha,
                                    dry_run, verbose,
                                    dependency_job_id=dep_id)

    # --- Handle timeouts ---
    if timeout_flags:
        print(f"\n  ⚠ TIMEOUT flags (max walltime reached, needs more splits):")
        for lid, seg_id, rr in timeout_flags:
            print(f"    Lake {lid} seg{seg_id}: timed out at max walltime")
            print(f"      Segment dir: {rr}")
            status["flags"].append({
                "lake_id": lid,
                "segment": f"seg{seg_id}",
                "issue": "timeout_at_max_walltime",
                "message": f"Lake {lid} seg{seg_id} timed out. Needs more splits.",
                "segment_dir": rr,
            })

    # --- Handle other failures ---
    if other_flags:
        print(f"\n  ⚠ Other failures (manual review needed):")
        for lid, seg_id, rr, reason in other_flags:
            print(f"    Lake {lid} seg{seg_id}: {reason}")
            print(f"      Dir: {rr}")

    if dry_run:
        print(f"\n  [DRY-RUN] No jobs submitted")

    status["last_stage"] = "recover"
    return status


def _submit_full_post_jobs(jobs: list, manifest: dict, config: dict,
                           alpha: str, dry_run: bool, verbose: bool,
                           dependency_job_id: str = None):
    """Submit FULL post-processing SLURM jobs for segments with post=0/6.

    These segments have dineof_results.nc and dincae_results.nc but no post
    directory at all. Runs dineof_postprocessor from scratch for both engines,
    then retrofit_post.py for clamp + dincae_interp + plots + insitu.

    Returns SLURM job ID (str) or None.
    """
    if not jobs:
        return None

    manifest_dir = os.path.dirname(manifest.get("_manifest_path", ""))
    retrofit_dir = os.path.join(manifest_dir, "retrofit")
    os.makedirs(retrofit_dir, exist_ok=True)

    # Resolve ALL paths in Python — no sed gymnastics in bash
    P = config.get("paths", {})
    lake_ts_tpl = P.get("lake_ts_template", "")
    clim_tpl = P.get("climatology_template", "")

    # Build job list: run_root \t lake_id \t lake_ts \t clim_nc
    job_list = []
    for rr, lid in jobs:
        lake_id9 = f"{lid:09d}"
        lake_ts = (lake_ts_tpl
                   .replace("{lake_id9}", lake_id9)
                   .replace("{lake_id}", str(lid)))
        clim_nc = (clim_tpl
                   .replace("{lake_id9}", lake_id9)
                   .replace("{lake_id}", str(lid)))
        job_list.append(f"{rr}\t{lid}\t{lake_ts}\t{clim_nc}")
        if verbose:
            print(f"    Lake {lid} in {os.path.basename(rr)}")
        else:
            print(f"    Lake {lid} in .../{os.path.basename(rr)} (post=0/6, full rebuild)")

    job_list_path = os.path.join(retrofit_dir, "recover_full_post_jobs.txt")
    config_path = manifest.get("base_config", "")

    if dry_run:
        print(f"    [DRY-RUN] Would submit {len(job_list)} full post-processing jobs")
        return None

    with open(job_list_path, "w") as f:
        f.write("\n".join(job_list) + "\n")

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    central_logs = os.path.join(REPO_ROOT, "logs")

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=recover_full_post
#SBATCH --array=1-{len(job_list)}
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=eocis_chuk
#SBATCH -o {retrofit_dir}/recover_full_post_%a.out
#SBATCH -e {retrofit_dir}/recover_full_post_%a.err

set -euo pipefail

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_list_path})
RUN_ROOT=$(echo "$LINE" | cut -f1)
LAKE_ID=$(echo "$LINE" | cut -f2)
LAKE_TS=$(echo "$LINE" | cut -f3)
CLIM_NC=$(echo "$LINE" | cut -f4)

# Symlink to central logs
ln -sf {retrofit_dir}/recover_full_post_${{SLURM_ARRAY_TASK_ID}}.out \\
   {central_logs}/recover_full_post_lake${{LAKE_ID}}_${{SLURM_JOB_ID}}.out
ln -sf {retrofit_dir}/recover_full_post_${{SLURM_ARRAY_TASK_ID}}.err \\
   {central_logs}/recover_full_post_lake${{LAKE_ID}}_${{SLURM_JOB_ID}}.err

echo "Full post-processing recovery: lake $LAKE_ID in $RUN_ROOT"
echo "Node: $(hostname), Job: $SLURM_JOB_ID, Start: $(date)"
echo "LAKE_TS: $LAKE_TS"
echo "CLIM_NC: $CLIM_NC"

cd {REPO_ROOT}

# Resolve paths from run_root + lake_id
LAKE_ID9=$(printf '%09d' $LAKE_ID)
POST_DIR="${{RUN_ROOT}}/post/${{LAKE_ID9}}/{alpha}"
DINEOF_DIR="${{RUN_ROOT}}/dineof/${{LAKE_ID9}}/{alpha}"
DINCAE_DIR="${{RUN_ROOT}}/dincae/${{LAKE_ID9}}/{alpha}"
PREPARED_NC="${{RUN_ROOT}}/prepared/${{LAKE_ID9}}/prepared.nc"
DINEOF_RESULTS="${{DINEOF_DIR}}/dineof_results.nc"
DINCAE_RESULTS="${{DINCAE_DIR}}/dincae_results.nc"
FRONT="LAKE${{LAKE_ID9}}-CCI-L3S-LSWT-CDR-4.5-filled_fine"
POST_DINEOF="${{POST_DIR}}/${{FRONT}}_dineof.nc"
POST_DINCAE="${{POST_DIR}}/${{FRONT}}_dincae.nc"
HTML_DIR="${{RUN_ROOT}}/html/${{LAKE_ID9}}/{alpha}"

# Verify key inputs exist
for f in "$LAKE_TS" "$CLIM_NC" "$PREPARED_NC" "$DINEOF_RESULTS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

mkdir -p "$POST_DIR"
mkdir -p "$HTML_DIR"

echo "============================================================"
echo "Step 1: DINEOF post-processing"
echo "============================================================"
dineof_postprocessor \\
    --lake-path "$LAKE_TS" \\
    --dineof-input-path "$PREPARED_NC" \\
    --dineof-output-path "$DINEOF_RESULTS" \\
    --output-path "$POST_DINEOF" \\
    --output-html-folder "$HTML_DIR" \\
    --config-file {config_path} \\
    --climatology-file "$CLIM_NC" \\
    --units celsius
echo "DINEOF post done: $(ls ${{POST_DIR}}/*_dineof*.nc 2>/dev/null | wc -l) files"

echo "============================================================"
echo "Step 2: DINCAE post-processing"
echo "============================================================"
if [ -f "$DINCAE_RESULTS" ]; then
    dineof_postprocessor \\
        --lake-path "$LAKE_TS" \\
        --dineof-input-path "$PREPARED_NC" \\
        --dineof-output-path "$DINCAE_RESULTS" \\
        --output-path "$POST_DINCAE" \\
        --config-file {config_path} \\
        --climatology-file "$CLIM_NC" \\
        --units celsius \\
        --no-eof-filter \\
        --no-eof-interp \\
        --no-eof-meta \\
        --no-log-meta
    echo "DINCAE post done"
else
    echo "WARNING: dincae_results.nc not found, skipping DINCAE post"
fi

echo "============================================================"
echo "Step 3: Retrofit (clamp + dincae_interp + plots + insitu)"
echo "============================================================"
python scripts/retrofit_post.py \\
    --run-root "$RUN_ROOT" \\
    --lake-ids $LAKE_ID \\
    --config {config_path}

echo "============================================================"
echo "Full post recovery complete: $(date)"
echo "Files: $(ls ${{POST_DIR}}/*.nc 2>/dev/null | wc -l)/6"
echo "============================================================"
"""

    slurm_path = os.path.join(retrofit_dir, "recover_full_post.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

    sbatch_cmd = ["sbatch"]
    if dependency_job_id:
        sbatch_cmd += [f"--dependency=afterok:{dependency_job_id}"]
        print(f"    Full post jobs will wait for DINCAE results job {dependency_job_id}")
    sbatch_cmd.append(slurm_path)

    result = subprocess.run(
        sbatch_cmd,
        capture_output=True, text=True,
    )
    slurm_job_id = None
    if result.returncode == 0:
        print(f"    Submitted: {result.stdout.strip()}")
        parts = result.stdout.strip().split()
        if len(parts) >= 4:
            slurm_job_id = parts[-1]
    else:
        print(f"    sbatch failed: {result.stderr.strip()}")

    return slurm_job_id


def _submit_post_recovery_jobs(jobs: list, manifest: dict, config: dict,
                                alpha: str, dry_run: bool, verbose: bool,
                                dependency_job_id: str = None):
    """Submit post-processing SLURM jobs for recovery.

    If dependency_job_id is provided, post jobs wait for that job to complete
    before starting (SLURM --dependency=afterok:job_id).
    """
    if not jobs:
        return

    manifest_dir = os.path.dirname(manifest.get("_manifest_path", ""))
    retrofit_dir = os.path.join(manifest_dir, "retrofit")
    os.makedirs(retrofit_dir, exist_ok=True)

    # Build job list
    job_list = []
    for rr, lid in jobs:
        job_list.append(f"{rr}\t{lid}")
        if verbose:
            print(f"    Lake {lid} in {os.path.basename(rr)}")

    job_list_path = os.path.join(retrofit_dir, "recover_post_jobs.txt")
    config_path = manifest.get("base_config", "")

    if dry_run:
        print(f"    [DRY-RUN] Would submit {len(job_list)} post-processing jobs")
        return

    with open(job_list_path, "w") as f:
        f.write("\n".join(job_list) + "\n")

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    central_logs = os.path.join(REPO_ROOT, "logs")

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=recover_post
#SBATCH --array=1-{len(job_list)}
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=eocis_chuk
#SBATCH -o {retrofit_dir}/recover_post_%a.out
#SBATCH -e {retrofit_dir}/recover_post_%a.err

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_list_path})
RUN_ROOT=$(echo "$LINE" | cut -f1)
LAKE_ID=$(echo "$LINE" | cut -f2)

# Symlink to central logs
ln -sf {retrofit_dir}/recover_post_${{SLURM_ARRAY_TASK_ID}}.out \\
   {central_logs}/recover_post_lake${{LAKE_ID}}_${{SLURM_JOB_ID}}.out
ln -sf {retrofit_dir}/recover_post_${{SLURM_ARRAY_TASK_ID}}.err \\
   {central_logs}/recover_post_lake${{LAKE_ID}}_${{SLURM_JOB_ID}}.err

echo "Post-processing recovery: lake $LAKE_ID in $RUN_ROOT"
echo "Node: $(hostname), Job: $SLURM_JOB_ID, Start: $(date)"

cd {REPO_ROOT}
python scripts/retrofit_post.py \\
  --run-root "$RUN_ROOT" \\
  --lake-ids $LAKE_ID \\
  --config {config_path}

echo "Done: $(date)"
"""

    slurm_path = os.path.join(retrofit_dir, "recover_post.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

    sbatch_cmd = ["sbatch"]
    if dependency_job_id:
        sbatch_cmd += [f"--dependency=afterok:{dependency_job_id}"]
        print(f"    Post jobs will wait for DINCAE results job {dependency_job_id}")
    sbatch_cmd.append(slurm_path)

    result = subprocess.run(
        sbatch_cmd,
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"    Submitted: {result.stdout.strip()}")
    else:
        print(f"    sbatch failed: {result.stderr.strip()}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: POST
# ═══════════════════════════════════════════════════════════════════════════

def stage_post(manifest: dict, status: dict, lake_ids: list,
               alpha: str, config: dict, config_path: str,
               dry_run: bool, verbose: bool) -> dict:
    """Run post-processing on segments where gap-filling is done but post is missing."""
    segments = manifest["segments"]
    all_lakes = lake_ids or manifest["lakes"]

    print(f"\n{'='*70}")
    print(f"STAGE: POST")
    print(f"{'='*70}")

    jobs = []  # (run_root, lake_id) needing post-processing

    for lid in all_lakes:
        lid_str = str(lid)
        le = status["lakes"].get(lid_str, {})

        for seg_id_str, seg in le.get("segments", {}).items():
            if seg.get("post_files", 0) >= 6:
                continue
            if seg.get("dineof_complete") and seg.get("dincae_complete"):
                jobs.append((seg["run_root"], lid))

    if not jobs:
        print("  No segments need post-processing")
        status["last_stage"] = "post"
        return status

    print(f"  {len(jobs)} segment×lake combinations need post-processing")

    if dry_run:
        for rr, lid in jobs:
            print(f"    [DRY-RUN] Lake {lid} in {os.path.basename(rr)}")
        status["last_stage"] = "post"
        return status

    _submit_post_recovery_jobs(jobs, manifest, config, alpha, dry_run, verbose)

    print(f"\n  After post jobs complete, re-run: --stage diagnose")
    status["last_stage"] = "post"
    return status


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: MERGE
# ═══════════════════════════════════════════════════════════════════════════

def stage_merge(manifest: dict, status: dict, lake_ids: list,
                alpha: str, verbose: bool) -> dict:
    """Merge segment outputs for lakes where all segments are complete."""
    all_lakes = lake_ids or manifest["lakes"]
    seg_run_roots = [s["run_root"] for s in manifest["segments"]]
    merge_root = manifest["merge_root"]

    print(f"\n{'='*70}")
    print(f"STAGE: MERGE")
    print(f"{'='*70}")
    print(f"  Merge root: {merge_root}")

    # Filter to lakes that are ALL_COMPLETE
    mergeable = []
    for lid in all_lakes:
        lid_str = str(lid)
        le = status["lakes"].get(lid_str, {})
        if le.get("overall") == "ALL_COMPLETE":
            mergeable.append(lid)
        else:
            if verbose:
                print(f"  Skipping lake {lid}: {le.get('overall', 'unknown')}")

    if not mergeable:
        print("  No lakes ready to merge (all segments must be COMPLETE)")
        print("  Run --stage diagnose to check status")
        status["last_stage"] = "merge"
        return status

    print(f"  Merging {len(mergeable)} lakes: {mergeable}")

    try:
        from merge_segments import merge_lake, verify_merge
    except ImportError:
        print("  ERROR: Could not import merge_segments.py")
        print("  Make sure it's in the same directory as this script")
        status["last_stage"] = "merge"
        return status

    for lid in mergeable:
        lid_str = str(lid)
        try:
            summary = merge_lake(lid, seg_run_roots, merge_root, alpha, verbose)
            checks = verify_merge(lid, merge_root, alpha, verbose)

            # Collect per-segment metadata
            seg_metadata = _collect_segment_metadata(
                lid, seg_run_roots, manifest["segments"], alpha
            )

            # Write segment_metadata.json alongside merged post files
            lake_id9 = f"{lid:09d}"
            meta_dir = os.path.join(merge_root, "post", lake_id9, alpha)
            os.makedirs(meta_dir, exist_ok=True)
            meta_path = os.path.join(meta_dir, "segment_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(seg_metadata, f, indent=2, default=_json_serializer)

            # Update status
            le = status["lakes"].setdefault(lid_str, {})
            le["merge"] = {
                "status": "complete" if summary.get("status") == "ok" else "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_dir": os.path.join(merge_root, "post", lake_id9, alpha),
                "verification": checks,
            }

        except Exception as e:
            print(f"  ERROR merging lake {lid}: {e}")
            if verbose:
                traceback.print_exc()
            le = status["lakes"].setdefault(lid_str, {})
            le["merge"] = {"status": "failed", "error": str(e)}

    status["last_stage"] = "merge"
    return status


def _collect_segment_metadata(lake_id: int, seg_run_roots: list,
                               segments: list, alpha: str) -> dict:
    """Collect per-segment provenance metadata for a merged lake."""
    lake_id9 = f"{lake_id:09d}"
    meta = {
        "lake_id": lake_id,
        "n_segments": len(seg_run_roots),
        "merge_timestamp": datetime.now(timezone.utc).isoformat(),
        "segments": [],
    }

    for i, (rr, seg_info) in enumerate(zip(seg_run_roots, segments)):
        seg_meta = {
            "seg_id": i,
            "run_root": rr,
            "time_range": [seg_info.get("start", "?"), seg_info.get("end", "?")],
        }

        # DINEOF CV error
        dineof_dir = os.path.join(rr, "dineof", lake_id9, alpha)
        cv_error_file = os.path.join(dineof_dir, "dineof_results.nc")
        if os.path.exists(cv_error_file):
            try:
                import xarray as xr
                ds = xr.open_dataset(cv_error_file)
                if "cv_error" in ds.attrs:
                    seg_meta["dineof_cv_error"] = float(ds.attrs["cv_error"])
                if "optimal_modes" in ds.attrs:
                    seg_meta["dineof_optimal_modes"] = int(ds.attrs["optimal_modes"])
                ds.close()
            except Exception:
                pass

        # DINCAE CV RMS
        dincae_dir = os.path.join(rr, "dincae", lake_id9, alpha)
        cv_rms_file = os.path.join(dincae_dir, "cv_rms.txt")
        if os.path.exists(cv_rms_file):
            try:
                with open(cv_rms_file) as f:
                    for line in f:
                        if "CV_RMS" in line:
                            seg_meta["dincae_cv_rms"] = float(line.split()[-1])
            except Exception:
                pass

        # Detrending parameters (from post files)
        post_dir = os.path.join(rr, "post", lake_id9, alpha)
        if os.path.isdir(post_dir):
            for fname in os.listdir(post_dir):
                if fname.endswith("_dineof.nc"):
                    try:
                        import xarray as xr
                        ds = xr.open_dataset(os.path.join(post_dir, fname))
                        for key in ["detrend_slope_per_day", "detrend_intercept",
                                     "detrend_t0_days"]:
                            if key in ds.attrs:
                                seg_meta[key] = float(ds.attrs[key])
                        ds.close()
                    except Exception:
                        pass
                    break

        # DINCAE config (batch_size, epochs)
        dincae_config_path = os.path.join(dincae_dir, "dincae_config.json")
        if os.path.exists(dincae_config_path):
            try:
                with open(dincae_config_path) as f:
                    dcfg = json.load(f)
                train = dcfg.get("train", {})
                seg_meta["dincae_batch_size"] = train.get("batch_size",
                                                          dcfg.get("batch_size", 32))
                seg_meta["dincae_epochs"] = train.get("epochs",
                                                      dcfg.get("epochs", 300))
            except Exception:
                pass

        meta["segments"].append(seg_meta)

    return meta


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: VALIDATE
# ═══════════════════════════════════════════════════════════════════════════

def stage_validate(manifest: dict, status: dict, lake_ids: list,
                   alpha: str, config: dict, config_path: str,
                   verbose: bool) -> dict:
    """Generate plots and insitu validation on merged outputs."""
    all_lakes = lake_ids or manifest["lakes"]
    merge_root = manifest["merge_root"]

    print(f"\n{'='*70}")
    print(f"STAGE: VALIDATE")
    print(f"{'='*70}")

    # Filter to lakes that have been merged
    validatable = []
    for lid in all_lakes:
        lid_str = str(lid)
        le = status["lakes"].get(lid_str, {})
        merge_info = le.get("merge", {})
        if merge_info.get("status") == "complete":
            validatable.append(lid)

    if not validatable:
        print("  No merged lakes to validate. Run --stage merge first.")
        status["last_stage"] = "validate"
        return status

    print(f"  Validating {len(validatable)} lakes: {validatable}")

    try:
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.lswt_plots import LSWTPlotsStep
        from processors.postprocessor.post_steps.insitu_validation import InsituValidationStep
    except ImportError as e:
        print(f"  ERROR: Could not import pipeline steps: {e}")
        print(f"  Run plots/insitu manually:")
        print(f"    python scripts/run_lswt_plots.py --run-root {merge_root} --all")
        print(f"    python scripts/run_insitu_validation.py --run-root {merge_root} --all")
        status["last_stage"] = "validate"
        return status

    P = config.get("paths", {})
    for lid in validatable:
        lid_str = str(lid)
        lake_id9 = f"{lid:09d}"
        post_dir = os.path.join(merge_root, "post", lake_id9, alpha)

        if not os.path.isdir(post_dir):
            continue

        nc_files = [f for f in os.listdir(post_dir) if f.endswith(".nc")]
        if not nc_files:
            continue

        lake_ts = P.get("lake_ts_template", "").replace("{lake_id9}", lake_id9).replace("{lake_id}", str(lid))
        clim = P.get("climatology_template", "").replace("{lake_id9}", lake_id9).replace("{lake_id}", str(lid))
        prepared = os.path.join(merge_root, "prepared", lake_id9, "prepared.nc")

        ctx = PostContext(
            lake_path=lake_ts,
            dineof_input_path=prepared,
            dineof_output_path="<unused>",
            output_path=os.path.join(post_dir, nc_files[0]),
            output_html_folder=None,
            climatology_path=clim,
            lake_id=lid,
            experiment_config_path=config_path or "",
        )

        le = status["lakes"].setdefault(lid_str, {})
        validate = le.setdefault("validate", {})

        # Plots
        try:
            print(f"  Generating LSWT plots for lake {lid}...")
            LSWTPlotsStep(original_ts_path=lake_ts).apply(ctx, None)
            validate["plots"] = True
        except Exception as e:
            print(f"  ERROR in LSWTPlotsStep for lake {lid}: {e}")
            validate["plots"] = False

        # Insitu
        try:
            print(f"  Running insitu validation for lake {lid}...")
            InsituValidationStep().apply(ctx, None)
            validate["insitu"] = True
        except Exception as e:
            print(f"  ERROR in InsituValidationStep for lake {lid}: {e}")
            validate["insitu"] = False

    status["last_stage"] = "validate"
    return status


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: FINALIZE
# ═══════════════════════════════════════════════════════════════════════════

def stage_finalize(manifest: dict, status: dict, lake_ids: list,
                   alpha: str, main_run: str, verbose: bool) -> dict:
    """Verify merged outputs and prepare for patching to main experiment."""
    all_lakes = lake_ids or manifest["lakes"]
    merge_root = manifest["merge_root"]
    manifest_dir = os.path.dirname(manifest.get("_manifest_path", ""))

    print(f"\n{'='*70}")
    print(f"STAGE: FINALIZE")
    print(f"{'='*70}")

    # Write main_run_target.txt
    if main_run:
        target_path = os.path.join(manifest_dir, "main_run_target.txt")
        with open(target_path, "w") as f:
            f.write(main_run + "\n")
        print(f"  Target experiment: {main_run}")
        print(f"  Written to: {target_path}")

    # Check each lake
    ready = []
    not_ready = []

    for lid in all_lakes:
        lid_str = str(lid)
        lake_id9 = f"{lid:09d}"
        le = status["lakes"].get(lid_str, {})

        merge_info = le.get("merge", {})
        validate_info = le.get("validate", {})

        post_dir = os.path.join(merge_root, "post", lake_id9, alpha)
        has_nc = False
        n_nc = 0
        if os.path.isdir(post_dir):
            ncs = [f for f in os.listdir(post_dir) if f.endswith(".nc")]
            n_nc = len(ncs)
            has_nc = n_nc >= 6

        has_plots = os.path.isdir(os.path.join(post_dir, "plots")) if os.path.isdir(post_dir) else False
        has_insitu = os.path.isdir(os.path.join(post_dir, "insitu_cv_validation")) if os.path.isdir(post_dir) else False
        has_metadata = os.path.exists(os.path.join(post_dir, "segment_metadata.json")) if os.path.isdir(post_dir) else False

        lake_ready = has_nc and has_plots
        status_str = "✅ READY" if lake_ready else "❌ NOT READY"

        detail_parts = [f"nc={n_nc}/6"]
        if has_plots:
            detail_parts.append("plots=✓")
        else:
            detail_parts.append("plots=✗")
        if has_insitu:
            detail_parts.append("insitu=✓")
        else:
            detail_parts.append("insitu=✗")
        if has_metadata:
            detail_parts.append("metadata=✓")

        print(f"  Lake {lid:>6d}: {status_str}  ({', '.join(detail_parts)})")

        if lake_ready:
            ready.append(lid)
        else:
            not_ready.append(lid)

    # Summary
    print(f"\n{'─'*70}")
    print(f"SUMMARY")
    print(f"{'─'*70}")
    print(f"  Ready to patch: {len(ready)}/{len(all_lakes)} lakes")

    if ready and main_run:
        print(f"\n  To patch into main experiment:")
        for lid in ready:
            lake_id9 = f"{lid:09d}"
            src = os.path.join(merge_root, "post", lake_id9)
            dst = os.path.join(main_run, "post", lake_id9)
            print(f"    cp -r {src}/* {dst}/")

        # Also suggest prepared/ symlinks
        print(f"\n  Don't forget prepared/ links:")
        for lid in ready:
            lake_id9 = f"{lid:09d}"
            src = os.path.join(merge_root, "prepared", lake_id9)
            dst = os.path.join(main_run, "prepared", lake_id9)
            print(f"    ln -sf $(readlink {src}) {dst}")

    if not_ready:
        print(f"\n  Not ready: {not_ready}")
        print(f"  Run --stage diagnose to check what's missing")

    status["last_stage"] = "finalize"
    return status


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: INSPECT (show all paths + log tails for specific lakes)
# ═══════════════════════════════════════════════════════════════════════════

def _tail_file(path: str, n: int = 8) -> list:
    """Get last n non-empty lines from a file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", errors="replace") as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
            return lines[-n:]
    except Exception:
        return []


def _file_info(path: str) -> str:
    """Return size + mtime for a file, or 'MISSING'."""
    if not os.path.exists(path):
        return "MISSING"
    st = os.stat(path)
    size_mb = st.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
    if size_mb > 1:
        return f"{size_mb:.1f} MB  ({mtime})"
    else:
        return f"{st.st_size:,} bytes  ({mtime})"


def stage_inspect(manifest: dict, lake_ids: list, alpha: str) -> None:
    """
    Show every file path, log tail, and error for specific lakes across
    all segments. Designed for debugging — answers "where are my files?"
    and "what went wrong?" in one command.
    """
    segments = manifest["segments"]
    all_lakes = lake_ids or manifest["lakes"]
    merge_root = manifest.get("merge_root", "")

    if not lake_ids:
        print("  TIP: Use --lake-ids to inspect specific lakes (inspecting all is verbose)")

    for lid in all_lakes:
        lake_id9 = f"{lid:09d}"

        print(f"\n{'═'*78}")
        print(f"  LAKE {lid}  ({lake_id9})")
        print(f"{'═'*78}")

        for seg in segments:
            seg_id = seg["seg_id"]
            rr = seg["run_root"]
            time_range = f"{seg.get('start', '?')} → {seg.get('end', '?')}"

            print(f"\n  ┌─ Segment {seg_id}  [{time_range}]")
            print(f"  │  Run root: {rr}")

            if not os.path.isdir(rr):
                print(f"  │  ⚠ Run root does not exist!")
                print(f"  └")
                continue

            # ── Prepared ──
            prepared_nc = os.path.join(rr, "prepared", lake_id9, "prepared.nc")
            clouds_nc = os.path.join(rr, "prepared", lake_id9, "clouds_index.nc")
            print(f"  │")
            print(f"  │  Prepared:")
            print(f"  │    prepared.nc:    {_file_info(prepared_nc)}")
            print(f"  │    clouds_index:   {_file_info(clouds_nc)}")

            # ── DINEOF ──
            dineof_dir = os.path.join(rr, "dineof", lake_id9, alpha)
            dineof_results = os.path.join(dineof_dir, "dineof_results.nc")
            print(f"  │")
            print(f"  │  DINEOF:")
            print(f"  │    dir:            {dineof_dir}")
            print(f"  │    results.nc:     {_file_info(dineof_results)}")

            # ── DINCAE ──
            dincae_dir = os.path.join(rr, "dincae", lake_id9, alpha)
            dincae_results = os.path.join(dincae_dir, "dincae_results.nc")
            cv_rms = os.path.join(dincae_dir, "cv_rms.txt")
            dincae_config = os.path.join(dincae_dir, "dincae_config.json")
            print(f"  │")
            print(f"  │  DINCAE:")
            print(f"  │    dir:            {dincae_dir}")
            print(f"  │    results.nc:     {_file_info(dincae_results)}")
            print(f"  │    cv_rms.txt:     {_file_info(cv_rms)}")

            # Show batch_size from config
            if os.path.exists(dincae_config):
                try:
                    with open(dincae_config) as f:
                        dcfg = json.load(f)
                    bs = dcfg.get("train", {}).get("batch_size",
                         dcfg.get("batch_size", "?"))
                    print(f"  │    batch_size:     {bs}")
                except Exception:
                    pass

            # DINCAE logs
            for suffix, label in [(".out", "stdout"), (".err", "stderr")]:
                log_path = os.path.join(dincae_dir, f"logs_dincae_{lid}{suffix}")
                if os.path.exists(log_path):
                    print(f"  │    log ({label}):   {_file_info(log_path)}")
                    tail = _tail_file(log_path, 5)
                    if tail:
                        for line in tail:
                            print(f"  │      │ {line[:120]}")
                # Also check .prev logs (from recovery)
                prev_path = log_path + ".prev"
                if os.path.exists(prev_path):
                    print(f"  │    log.prev:       {_file_info(prev_path)}")

            # ── Post files ──
            post_dir = os.path.join(rr, "post", lake_id9, alpha)
            print(f"  │")
            print(f"  │  Post:")
            print(f"  │    dir:            {post_dir}")
            if os.path.isdir(post_dir):
                ncs = sorted(f for f in os.listdir(post_dir) if f.endswith(".nc"))
                if ncs:
                    for nc in ncs:
                        nc_path = os.path.join(post_dir, nc)
                        # Show just the suffix for brevity
                        suffix = nc.split("LAKE" + lake_id9)[-1] if lake_id9 in nc else nc
                        print(f"  │    {suffix:<45s} {_file_info(nc_path)}")
                else:
                    print(f"  │    (no .nc files)")

                # Check for plots/insitu subdirs
                has_plots = os.path.isdir(os.path.join(post_dir, "plots"))
                has_insitu = os.path.isdir(os.path.join(post_dir, "insitu_cv_validation"))
                if has_plots or has_insitu:
                    parts = []
                    if has_plots:
                        parts.append("plots/")
                    if has_insitu:
                        parts.append("insitu_cv_validation/")
                    print(f"  │    subdirs:        {', '.join(parts)}")
            else:
                print(f"  │    (directory does not exist)")

            # ── Chain logs ──
            log_dir = os.path.join(rr, "logs")
            chain_logs_found = False
            if os.path.isdir(log_dir):
                chain_files = sorted(
                    f for f in os.listdir(log_dir)
                    if f.startswith(f"chain_lake{lid}_") or
                       f.startswith(f"post_dineof_lake{lid}_") or
                       f.startswith(f"post_dincae_lake{lid}_")
                )
                if chain_files:
                    print(f"  │")
                    print(f"  │  Chain/Post logs:")
                    for lf in chain_files:
                        lf_path = os.path.join(log_dir, lf)
                        print(f"  │    {lf}  ({_file_info(lf_path)})")
                        # Show errors and tail
                        errors = scan_log_for_errors(lf_path, max_lines=100)
                        if errors:
                            unique = set()
                            for label, msg in errors:
                                if label not in unique:
                                    unique.add(label)
                                    print(f"  │      ⚠ {label}: {msg[:100]}")
                        else:
                            tail = _tail_file(lf_path, 3)
                            for line in tail:
                                print(f"  │      │ {line[:120]}")
                    chain_logs_found = True

            if not chain_logs_found:
                print(f"  │")
                print(f"  │  Chain logs: (none found in {log_dir})")

            print(f"  └")

        # ── Merged outputs ──
        if merge_root:
            merged_post = os.path.join(merge_root, "post", lake_id9, alpha)
            merged_cv_dineof = os.path.join(merge_root, "dineof", lake_id9, alpha, "cv_pairs_dineof.npz")
            merged_cv_dincae = os.path.join(merge_root, "dincae", lake_id9, alpha, "cv_pairs_dincae.npz")

            print(f"\n  ┌─ Merged outputs")
            print(f"  │  Merge root: {merge_root}")
            print(f"  │")
            print(f"  │  CV pairs:")
            print(f"  │    dineof: {_file_info(merged_cv_dineof)}")
            print(f"  │    dincae: {_file_info(merged_cv_dincae)}")

            if os.path.isdir(merged_post):
                ncs = sorted(f for f in os.listdir(merged_post) if f.endswith(".nc"))
                print(f"  │  Post files: {len(ncs)}")
                for nc in ncs:
                    suffix = nc.split("LAKE" + lake_id9)[-1] if lake_id9 in nc else nc
                    print(f"  │    {suffix}")
                has_plots = os.path.isdir(os.path.join(merged_post, "plots"))
                has_insitu = os.path.isdir(os.path.join(merged_post, "insitu_cv_validation"))
                has_meta = os.path.exists(os.path.join(merged_post, "segment_metadata.json"))
                print(f"  │  plots: {'✓' if has_plots else '✗'}  insitu: {'✓' if has_insitu else '✗'}  metadata: {'✓' if has_meta else '✗'}")
            else:
                print(f"  │  Post dir: (not yet merged)")
            print(f"  └")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: STATUS (scans disk by default; --cached for instant readback)
# ═══════════════════════════════════════════════════════════════════════════

def stage_status(manifest: dict, status: dict, lake_ids: list,
                 alpha: str, verbose: bool, cached: bool = False) -> dict:
    """
    Print overview of pipeline progress.

    Default: scans disk for ground truth (same as diagnose, but with a
    compact summary table). Always safe to run.

    --cached: reads segment_status.json only — instant, but may be stale.
    """
    all_lakes = lake_ids or manifest["lakes"]
    segments = manifest["segments"]
    merge_root = manifest.get("merge_root", "")

    # If not cached, do a fresh disk scan first (updates status in place)
    if not cached:
        print("  Scanning disk for current state...")
        status = stage_diagnose(manifest, status, lake_ids, alpha, verbose=False)
        # Also check merge + validate state from disk
        for lid in all_lakes:
            lid_str = str(lid)
            lake_id9 = f"{lid:09d}"
            le = status["lakes"].setdefault(lid_str, {})

            # Check merged post files
            merged_post_dir = os.path.join(merge_root, "post", lake_id9, alpha)
            if os.path.isdir(merged_post_dir):
                merged_ncs = [f for f in os.listdir(merged_post_dir) if f.endswith(".nc")]
                has_plots = os.path.isdir(os.path.join(merged_post_dir, "plots"))
                has_insitu = os.path.isdir(os.path.join(merged_post_dir, "insitu_cv_validation"))
                has_metadata = os.path.exists(os.path.join(merged_post_dir, "segment_metadata.json"))

                if len(merged_ncs) >= 6:
                    le["merge"] = le.get("merge", {})
                    le["merge"]["status"] = "complete"
                    le["merge"]["post_files"] = len(merged_ncs)

                le["validate"] = le.get("validate", {})
                if has_plots:
                    le["validate"]["plots"] = True
                if has_insitu:
                    le["validate"]["insitu"] = True
    else:
        if not status.get("lakes"):
            print(f"\n  No cached status data. Run --stage status (without --cached) first.")
            return status

    print(f"\n{'='*70}")
    print(f"SEGMENT PIPELINE STATUS {'(cached)' if cached else '(disk scan)'}")
    print(f"{'='*70}")
    print(f"  Manifest:    {manifest.get('_manifest_path', '?')}")
    print(f"  Merge root:  {merge_root}")
    print(f"  Lakes:       {len(all_lakes)}  {all_lakes}")
    print(f"  Segments:    {len(segments)}")
    if status.get("last_run"):
        print(f"  Last saved:  {status.get('last_stage', '?')} at {status['last_run']}")

    if not status.get("lakes"):
        print(f"\n  No status data available.")
        return status

    # ── Per-lake summary table ──
    STATUS_ICONS = {
        "ALL_COMPLETE": "✅",
        "DINEOF_MERGEABLE": "🟡",
        "PARTIAL": "🟠",
        "NOT_READY": "❌",
    }

    # Column headers
    n_segs = len(segments)
    seg_header = "  ".join(f"seg{i}" for i in range(n_segs))
    print(f"\n  {'Lake':>8s}  {'Overall':<18s}  {'Merge':<10s}  {'Valid':<10s}  {seg_header}")
    print(f"  {'─'*8}  {'─'*18}  {'─'*10}  {'─'*10}  {'─' * (6 * n_segs)}")

    # Counters
    counts = {"ALL_COMPLETE": 0, "DINEOF_MERGEABLE": 0, "PARTIAL": 0, "NOT_READY": 0, "?": 0}
    n_merged = 0
    n_validated = 0

    for lid in all_lakes:
        lid_str = str(lid)
        le = status["lakes"].get(lid_str, {})
        overall = le.get("overall", "?")
        icon = STATUS_ICONS.get(overall, "❓")
        counts[overall] = counts.get(overall, 0) + 1

        # Merge status
        merge_info = le.get("merge", {})
        merge_str = merge_info.get("status", "—")
        if merge_str == "complete":
            merge_str = "✓"
            n_merged += 1
        elif merge_str == "failed":
            merge_str = "✗"

        # Validate status
        val_info = le.get("validate", {})
        val_parts = []
        if val_info.get("plots"):
            val_parts.append("P")
        if val_info.get("insitu"):
            val_parts.append("I")
        val_str = "+".join(val_parts) if val_parts else "—"
        if val_info.get("plots"):
            n_validated += 1

        # Per-segment status
        seg_parts = []
        for seg_idx in range(n_segs):
            seg_data = le.get("segments", {}).get(str(seg_idx), {})
            n_post = seg_data.get("post_files", 0)
            seg_status = seg_data.get("status", "?")
            failure = seg_data.get("failure_reason", "")

            if seg_status == "COMPLETE":
                seg_parts.append(f"  {n_post}/6 ")
            elif failure == "POST_NOT_RUN":
                # Gap-filling done, post needs running — show post count
                seg_parts.append(f" P{n_post}/6 ")
            elif failure == "GPU_OOM":
                seg_parts.append("  gOOM ")
            elif failure == "DINCAE_FAILED":
                seg_parts.append(f" D{n_post}/6 ")
            elif failure == "RAM_OOM":
                seg_parts.append("  rOOM ")
            elif failure == "TIMEOUT":
                seg_parts.append("  TIME ")
            elif seg_status == "RUN_ROOT_MISSING":
                seg_parts.append("  MISS ")
            elif n_post > 0:
                seg_parts.append(f"  {n_post}/6  ")
            else:
                short = seg_status[:4] if seg_status else "?"
                seg_parts.append(f"  {short} ")

        seg_str = "  ".join(seg_parts)
        print(f"  {icon} {lid:>6d}  {overall:<18s}  {merge_str:<10s}  {val_str:<10s}  {seg_str}")

    # ── Totals ──
    print(f"\n{'─'*70}")
    print(f"TOTALS")
    print(f"{'─'*70}")
    print(f"  ✅ All segments complete: {counts.get('ALL_COMPLETE', 0)}")
    print(f"  🟡 DINEOF mergeable:     {counts.get('DINEOF_MERGEABLE', 0)}")
    print(f"  🟠 Partial:              {counts.get('PARTIAL', 0)}")
    print(f"  ❌ Not ready:            {counts.get('NOT_READY', 0)}")
    print(f"  Merged:    {n_merged}/{len(all_lakes)}")
    print(f"  Validated: {n_validated}/{len(all_lakes)}")

    # ── Flags ──
    flags = status.get("flags", [])
    if flags:
        print(f"\n{'─'*70}")
        print(f"FLAGS ({len(flags)})")
        print(f"{'─'*70}")
        for fl in flags:
            print(f"  ⚠ Lake {fl.get('lake_id', '?')} {fl.get('segment', '?')}: "
                  f"{fl.get('issue', '?')}")
            if verbose and fl.get("segment_dir"):
                print(f"    Dir: {fl['segment_dir']}")

    # ── Next action suggestion ──
    print(f"\n{'─'*70}")
    print(f"SUGGESTED NEXT ACTION")
    print(f"{'─'*70}")

    if counts.get("NOT_READY", 0) + counts.get("PARTIAL", 0) + counts.get("DINEOF_MERGEABLE", 0) > 0:
        if status.get("last_stage") in (None, "diagnose"):
            print(f"  → Run --stage recover to fix failures")
        else:
            print(f"  → Run --stage diagnose to refresh, then --stage recover")
    elif counts.get("ALL_COMPLETE", 0) > n_merged:
        print(f"  → Run --stage merge ({counts['ALL_COMPLETE'] - n_merged} lakes ready)")
    elif n_merged > n_validated:
        print(f"  → Run --stage validate ({n_merged - n_validated} lakes ready)")
    elif n_merged > 0:
        print(f"  → Run --stage finalize to prepare patch commands")
    else:
        print(f"  → Run --stage status to check current state")

    return status


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: AUTO
# ═══════════════════════════════════════════════════════════════════════════

def stage_auto(manifest: dict, status: dict, lake_ids: list,
               alpha: str, config: dict, config_path: str,
               main_run: str, batch_size: int,
               dry_run: bool, verbose: bool) -> dict:
    """Walk through all stages, skipping completed work, stopping when blocked."""
    all_lakes = lake_ids or manifest["lakes"]

    print(f"\n{'='*70}")
    print(f"STAGE: AUTO")
    print(f"{'='*70}")

    # Step 1: Always diagnose first
    status = stage_diagnose(manifest, status, lake_ids, alpha, verbose)
    save_status(os.path.dirname(manifest["_manifest_path"]), status)

    # Step 2: Check if recovery is needed
    needs_recovery = False
    for lid in all_lakes:
        le = status["lakes"].get(str(lid), {})
        overall = le.get("overall", "")
        if overall in ("PARTIAL", "NOT_READY", "DINEOF_MERGEABLE"):
            # Check if there are actual failures to recover
            for seg in le.get("segments", {}).values():
                reason = seg.get("failure_reason", "")
                if reason in ("RAM_OOM", "GPU_OOM"):
                    needs_recovery = True
                    break

    if needs_recovery:
        status = stage_recover(manifest, status, lake_ids, alpha, config,
                               batch_size, dry_run, verbose)
        save_status(os.path.dirname(manifest["_manifest_path"]), status)
        print(f"\n  Recovery jobs submitted. Re-run --stage auto after they complete.")
        return status

    # Step 3: Check if post-processing is needed
    needs_post = False
    for lid in all_lakes:
        le = status["lakes"].get(str(lid), {})
        for seg in le.get("segments", {}).values():
            if (seg.get("dineof_complete") and seg.get("dincae_complete")
                    and seg.get("post_files", 0) < 6):
                needs_post = True
                break

    if needs_post:
        status = stage_post(manifest, status, lake_ids, alpha, config,
                            config_path, dry_run, verbose)
        save_status(os.path.dirname(manifest["_manifest_path"]), status)
        print(f"\n  Post jobs submitted. Re-run --stage auto after they complete.")
        return status

    # Step 4: Merge
    any_mergeable = any(
        status["lakes"].get(str(lid), {}).get("overall") == "ALL_COMPLETE"
        for lid in all_lakes
    )
    any_unmerged = any(
        status["lakes"].get(str(lid), {}).get("overall") == "ALL_COMPLETE"
        and status["lakes"].get(str(lid), {}).get("merge", {}).get("status") != "complete"
        for lid in all_lakes
    )

    if any_unmerged:
        status = stage_merge(manifest, status, lake_ids, alpha, verbose)
        save_status(os.path.dirname(manifest["_manifest_path"]), status)

    # Step 5: Validate
    any_unvalidated = any(
        status["lakes"].get(str(lid), {}).get("merge", {}).get("status") == "complete"
        and not status["lakes"].get(str(lid), {}).get("validate", {}).get("plots")
        for lid in all_lakes
    )

    if any_unvalidated:
        status = stage_validate(manifest, status, lake_ids, alpha, config,
                                config_path, verbose)
        save_status(os.path.dirname(manifest["_manifest_path"]), status)

    # Step 6: Finalize
    status = stage_finalize(manifest, status, lake_ids, alpha, main_run, verbose)
    save_status(os.path.dirname(manifest["_manifest_path"]), status)

    return status


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified orchestrator for temporal-split lake processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect specific lake: show all paths, logs, errors across segments
  python segment_pipeline.py --manifest .../segment_manifest.json --stage inspect --lake-ids 8

  # Quick status overview (scans disk)
  python segment_pipeline.py --manifest .../segment_manifest.json --stage status

  # Instant status from cached state file (no disk I/O)
  python segment_pipeline.py --manifest .../segment_manifest.json --stage status --cached

  # Check status of all lakes (full diagnostic output)
  python segment_pipeline.py --manifest .../segment_manifest.json --stage diagnose

  # Auto-recover and continue
  python segment_pipeline.py --manifest ... --config exp0.json --stage auto

  # Recover specific lakes
  python segment_pipeline.py --manifest ... --lake-ids 5 6 8 --stage recover

  # Merge completed lakes
  python segment_pipeline.py --manifest ... --stage merge

  # Full auto (diagnose → recover → post → merge → validate → finalize)
  python segment_pipeline.py --manifest ... --config exp0.json \\
    --main-run /gws/.../anomaly-20260131-exp0_baseline_both --stage auto
        """,
    )

    parser.add_argument("--manifest", required=True,
                        help="Path to segment_manifest.json")
    parser.add_argument("--config", default=None,
                        help="Experiment config JSON (needed for post/validate/recover)")
    parser.add_argument("--main-run", default=None,
                        help="Main experiment run root (for finalize stage)")
    parser.add_argument("--stage", required=True,
                        choices=["status", "inspect", "process", "diagnose", "recover",
                                 "post", "merge", "validate", "finalize", "auto"],
                        help="Pipeline stage to run")
    parser.add_argument("--lake-ids", type=int, nargs="+", default=None,
                        help="Subset of lakes to process (default: all)")
    parser.add_argument("--alpha", default="a1000")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="DINCAE batch size for GPU OOM recovery (default: 16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without submitting jobs")
    parser.add_argument("--cached", action="store_true",
                        help="For --stage status: read saved state only (no disk scan)")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    # Load manifest
    if not os.path.isfile(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}")
        sys.exit(1)

    with open(args.manifest) as f:
        manifest = json.load(f)
    manifest["_manifest_path"] = os.path.abspath(args.manifest)

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    # Load config if provided
    config = {}
    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(str(REPO_ROOT), "configs", config_path)
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            print(f"WARNING: config not found: {config_path}")
            config_path = args.config
    else:
        config_path = manifest.get("base_config", "")

    # Load state
    status = load_status(manifest_dir)

    print(f"Manifest: {args.manifest}")
    print(f"Lakes: {args.lake_ids or manifest['lakes']}")
    print(f"Segments: {len(manifest['segments'])}")
    print(f"Stage: {args.stage}")
    if status.get("last_stage"):
        print(f"Last stage run: {status['last_stage']} at {status.get('last_run', '?')}")

    # Dispatch
    if args.stage == "status":
        status = stage_status(manifest, status, args.lake_ids, args.alpha,
                              verbose, cached=args.cached)

    elif args.stage == "inspect":
        stage_inspect(manifest, args.lake_ids, args.alpha)

    elif args.stage == "process":
        print("\n  'process' stage delegates to temporal_split_runner.py")
        print("  Use temporal_split_runner.py --submit for initial processing")
        print("  Then use this script for --stage diagnose onwards")

    elif args.stage == "diagnose":
        status = stage_diagnose(manifest, status, args.lake_ids, args.alpha, verbose)

    elif args.stage == "recover":
        if not config and not args.config:
            parser.error("--stage recover requires --config")
        status = stage_recover(manifest, status, args.lake_ids, args.alpha,
                               config, args.batch_size, args.dry_run, verbose)

    elif args.stage == "post":
        if not config:
            parser.error("--stage post requires --config")
        status = stage_post(manifest, status, args.lake_ids, args.alpha,
                            config, config_path, args.dry_run, verbose)

    elif args.stage == "merge":
        status = stage_merge(manifest, status, args.lake_ids, args.alpha, verbose)

    elif args.stage == "validate":
        if not config:
            parser.error("--stage validate requires --config")
        status = stage_validate(manifest, status, args.lake_ids, args.alpha,
                                config, config_path, verbose)

    elif args.stage == "finalize":
        status = stage_finalize(manifest, status, args.lake_ids, args.alpha,
                                args.main_run, verbose)

    elif args.stage == "auto":
        if not config and not args.config:
            parser.error("--stage auto requires --config")
        status = stage_auto(manifest, status, args.lake_ids, args.alpha,
                            config, config_path, args.main_run, args.batch_size,
                            args.dry_run, verbose)

    # Save state
    save_status(manifest_dir, status)
    print(f"\nState saved: {os.path.join(manifest_dir, 'segment_status.json')}")


if __name__ == "__main__":
    main()
