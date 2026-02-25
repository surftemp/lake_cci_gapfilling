#!/usr/bin/env python3
"""
rerun_dincae.py - Re-run DINCAE with modified parameters for specific lakes.

Designed for cases where DINCAE failed (e.g. GPU OOM) and needs to be retried
with different settings (typically smaller batch_size).

Works with both single run-root and segment manifests.

Usage:
  # Single segment, single lake, reduce batch size
  python rerun_dincae.py \\
    --run-root /gws/.../anomaly-20260215-3da5c4-exp0_baseline_both_seg0 \\
    --lake-ids 8 --batch-size 16

  # All segments from manifest
  python rerun_dincae.py \\
    --manifest /gws/.../exp0_baseline_large_merged_4splits/segment_manifest.json \\
    --lake-ids 8 --batch-size 16

  # Dry-run: show what would be submitted
  python rerun_dincae.py \\
    --manifest /gws/.../segment_manifest.json \\
    --lake-ids 8 --batch-size 16 --dry-run

  # Don't wait for jobs (submit and exit)
  python rerun_dincae.py \\
    --manifest /gws/.../segment_manifest.json \\
    --lake-ids 8 --batch-size 16 --no-wait

Author: Shaerdan / NCEO / University of Reading
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict


def find_dincae_dir(run_root: str, lake_id: int, alpha: str = "a1000") -> Optional[str]:
    """Find the DINCAE directory for a lake within a run."""
    lake_id9 = f"{lake_id:09d}"
    dincae_dir = os.path.join(run_root, "dincae", lake_id9, alpha)
    if os.path.isdir(dincae_dir):
        return dincae_dir
    return None


def load_dincae_config(dincae_dir: str) -> Optional[dict]:
    """Load dincae_config.json from a DINCAE directory."""
    config_path = os.path.join(dincae_dir, "dincae_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        return json.load(f)


def patch_config(cfg: dict, batch_size: Optional[int] = None,
                 epochs: Optional[int] = None) -> dict:
    """Patch DINCAE config with new parameters.
    
    The runner reads from cfg['train']['batch_size'], but we also update
    root-level batch_size if it exists (for display/logging consistency).
    """
    cfg = json.loads(json.dumps(cfg))  # deep copy
    train = cfg.setdefault("train", {})
    if batch_size is not None:
        old_bs = train.get("batch_size", cfg.get("batch_size", 32))
        train["batch_size"] = batch_size
        # Also update root level if it exists there
        if "batch_size" in cfg:
            cfg["batch_size"] = batch_size
        print(f"    batch_size: {old_bs} → {batch_size}")
    if epochs is not None:
        old_ep = train.get("epochs", cfg.get("epochs", 300))
        train["epochs"] = epochs
        if "epochs" in cfg:
            cfg["epochs"] = epochs
        print(f"    epochs: {old_ep} → {epochs}")
    return cfg


def check_existing_results(dincae_dir: str) -> dict:
    """Check what DINCAE outputs already exist."""
    results = {
        "data_avg": os.path.isfile(os.path.join(dincae_dir, "data-avg.nc")),
        "cv_rms": os.path.isfile(os.path.join(dincae_dir, "cv_rms.txt")),
        "checkpoints": 0,
    }
    for f in os.listdir(dincae_dir):
        if f.startswith("data-avg-epoch") and f.endswith(".nc"):
            results["checkpoints"] += 1
    return results


def clear_old_results(dincae_dir: str, dry_run: bool = False):
    """Remove old DINCAE outputs before re-run."""
    to_remove = []
    for f in os.listdir(dincae_dir):
        if f in ("data-avg.nc", "cv_rms.txt", "loss_history.json",
                 "lake_cleanup.nc"):
            to_remove.append(f)
        elif f.startswith("data-avg-epoch") and f.endswith(".nc"):
            to_remove.append(f)

    if to_remove:
        if dry_run:
            print(f"    [DRY-RUN] Would remove {len(to_remove)} old output files")
        else:
            for f in to_remove:
                fp = os.path.join(dincae_dir, f)
                os.remove(fp)
            print(f"    Removed {len(to_remove)} old output files")


def submit_dincae(dincae_dir: str, cfg: dict, lake_id: int,
                  dry_run: bool = False, wait: bool = True) -> Optional[str]:
    """
    Regenerate Julia script and SLURM job, then submit.

    Uses the existing dincae_runner machinery via import if available,
    otherwise generates the SLURM script directly.
    """
    # Save patched config
    config_path = os.path.join(dincae_dir, "dincae_config.json")
    if not dry_run:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"    Saved patched config: {config_path}")

    # Try to use the pipeline's own submission code
    try:
        src_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from dincae_arm.contracts import DincaeArtifacts
        from dincae_arm.dincae_runner import _generate_julia_script, submit_slurm_job

        # Reconstruct DincaeArtifacts from existing files
        arts = _reconstruct_artifacts(dincae_dir)
        if arts is None:
            print(f"    ERROR: Could not reconstruct DINCAE artifacts")
            return None

        if dry_run:
            print(f"    [DRY-RUN] Would submit DINCAE job for lake {lake_id}")
            print(f"    Dir: {dincae_dir}")
            return None

        # Generate script but submit manually (submit_slurm_job uses --wait)
        script_path = _generate_julia_script(arts, cfg)
        print(f"    Generated Julia script: {script_path}")

        # Build and submit SLURM job
        job_id = _submit_slurm_manual(arts, cfg, lake_id, script_path, wait=wait)
        return job_id

    except ImportError as e:
        print(f"    WARNING: Could not import pipeline code ({e})")
        print(f"    Falling back to direct SLURM submission")
        return _submit_slurm_fallback(dincae_dir, cfg, lake_id,
                                       dry_run=dry_run, wait=wait)


def _reconstruct_artifacts(dincae_dir: str):
    """Reconstruct DincaeArtifacts from existing files in a DINCAE directory."""
    try:
        from dincae_arm.contracts import DincaeArtifacts
    except ImportError:
        return None

    d = Path(dincae_dir)

    # All expected files
    prepared_datetime = d / "prepared_datetime.nc"
    prepared_cropped = d / "prepared_datetime_cropped.nc"
    prepared_cropped_cv = d / "prepared_datetime_cropped_add_clouds.nc"
    prepared_cropped_clean = d / "prepared_datetime_cropped_add_clouds.clean.nc"

    # Check required files exist
    for f, label in [
        (prepared_cropped, "prepared_datetime_cropped.nc"),
        (prepared_cropped_cv, "CV file"),
    ]:
        if not f.exists():
            print(f"    Missing: {label} ({f})")
            return None

    arts = DincaeArtifacts(
        dincae_dir=d,
        prepared_datetime=prepared_datetime,
        prepared_cropped=prepared_cropped,
        prepared_cropped_cv=prepared_cropped_cv,
        prepared_cropped_clean=prepared_cropped_clean,
    )
    return arts


def _submit_slurm_manual(arts, cfg: dict, lake_id: int,
                          script_path: Path, wait: bool = True) -> Optional[str]:
    """Submit DINCAE SLURM job (adapted from dincae_runner.submit_slurm_job)."""
    slurm = cfg.get("slurm", {})
    partition = slurm.get("partition", "orchid")
    account = slurm.get("account", "orchid")
    qos = slurm.get("qos", "orchid")
    gpus = int(slurm.get("gpus", 1))
    cpus = int(slurm.get("cpus", 4))
    mem = slurm.get("mem", "128G")
    wall = slurm.get("time", "24:00:00")
    julia = cfg.get("runner", {}).get("julia_exe", "julia")
    exclude = slurm.get("exclude", "gpuhost004,gpuhost007,gpuhost012,gpuhost016")

    env_lines = []
    denv = cfg.get("env", {}).get("dincae", {})
    if denv.get("module_load"):
        env_lines.append(denv["module_load"])
    if denv.get("activate"):
        env_lines.append(denv["activate"])

    julia_project = cfg.get("runner", {}).get("JULIA_PROJECT")
    if julia_project:
        env_lines.append(f'export JULIA_PROJECT="{julia_project}"')

    shared_depot = cfg.get("runner", {}).get("julia_depot", "$HOME/.julia_cuda_depot")
    env_lines += [
        "export JULIA_PKG_PRECOMPILE_AUTO=0",
        f'export JULIA_DEPOT_PATH="{shared_depot}"',
        f'mkdir -p "{shared_depot}"',
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        'export CUDA_PATH="${CONDA_PREFIX}"',
    ]
    env_block = "\n".join(env_lines)

    log_out = Path(arts.dincae_dir) / f"logs_dincae_{lake_id}.out"
    log_err = Path(arts.dincae_dir) / f"logs_dincae_{lake_id}.err"

    # Back up old logs
    for log_path in [log_out, log_err]:
        if log_path.exists():
            backup = log_path.with_suffix(log_path.suffix + ".prev")
            log_path.rename(backup)

    exclude_line = f"#SBATCH --exclude={exclude}" if exclude else ""
    sb = f"""#!/bin/bash
#SBATCH --job-name=dincae_rerun_{lake_id}
{exclude_line}
#SBATCH --time={wall}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --chdir={arts.dincae_dir}
#SBATCH -o {log_out}
#SBATCH -e {log_err}

cd {arts.dincae_dir}
{env_block}

set -euo pipefail

echo "=========================================="
echo "DINCAE Re-run (batch_size patched)"
echo "=========================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Precompile
{julia} -e 'using CUDA, cuDNN, DINCAE; println("All packages loaded successfully")'

# GPU verification
srun --gres=gpu:{gpus} --ntasks=1 --cpus-per-task={cpus} nvidia-smi

# Main execution
echo "=========================================="
echo "Starting DINCAE Training"
echo "Start time: $(date)"
echo "=========================================="
exec srun --gres=gpu:{gpus} --ntasks=1 --cpus-per-task={cpus} stdbuf -oL -eL {julia} --project {script_path.name}
"""

    jobfile = Path(arts.dincae_dir) / f"rerun_dincae_{lake_id}.slurm"
    jobfile.write_text(sb)
    print(f"    SLURM script: {jobfile}")

    wait_flag = ["--wait"] if wait else []
    result = subprocess.run(
        ["sbatch"] + wait_flag + [str(jobfile)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        job_info = result.stdout.strip()
        print(f"    Submitted: {job_info}")
        return job_info
    else:
        print(f"    sbatch failed: {result.stderr.strip()}")
        return None


def _submit_slurm_fallback(dincae_dir: str, cfg: dict, lake_id: int,
                            dry_run: bool = False, wait: bool = True) -> Optional[str]:
    """Fallback: submit existing run_dincae.jl SLURM script with patched config."""
    existing_slurm = os.path.join(dincae_dir, f"run_dincae_{lake_id}.slurm")
    if not os.path.isfile(existing_slurm):
        print(f"    ERROR: No existing SLURM script found: {existing_slurm}")
        print(f"    Cannot submit without pipeline imports")
        return None

    if dry_run:
        print(f"    [DRY-RUN] Would resubmit: {existing_slurm}")
        return None

    # The config is already patched and saved; the Julia script reads it at runtime
    # But run_dincae.jl has hardcoded values, so we need to regenerate it
    print(f"    WARNING: Using existing SLURM script (Julia params may not reflect patched config)")
    print(f"    Consider running from a node with pipeline imports available")

    wait_flag = ["--wait"] if wait else []
    result = subprocess.run(
        ["sbatch"] + wait_flag + [existing_slurm],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        job_info = result.stdout.strip()
        print(f"    Submitted: {job_info}")
        return job_info
    else:
        print(f"    sbatch failed: {result.stderr.strip()}")
        return None


def process_lake(run_root: str, lake_id: int, alpha: str,
                 batch_size: Optional[int], epochs: Optional[int],
                 dry_run: bool, force: bool, wait: bool):
    """Process a single lake in a single run."""
    dincae_dir = find_dincae_dir(run_root, lake_id, alpha)
    if dincae_dir is None:
        print(f"  No DINCAE directory for lake {lake_id} in {run_root}")
        return

    # Load existing config
    cfg = load_dincae_config(dincae_dir)
    if cfg is None:
        print(f"  No dincae_config.json in {dincae_dir}")
        return

    # Check existing results
    existing = check_existing_results(dincae_dir)
    if existing["data_avg"] and existing["cv_rms"] and not force:
        print(f"  DINCAE already complete (data-avg.nc + cv_rms.txt exist)")
        print(f"  Use --force to re-run anyway")
        return

    status_parts = []
    if existing["data_avg"]:
        status_parts.append("data-avg.nc exists (incomplete?)")
    if existing["checkpoints"] > 0:
        status_parts.append(f"{existing['checkpoints']} checkpoints")
    if not status_parts:
        status_parts.append("no outputs")
    print(f"  Current state: {', '.join(status_parts)}")

    # Patch config
    print(f"  Patching config:")
    cfg = patch_config(cfg, batch_size=batch_size, epochs=epochs)

    # Clear old outputs
    clear_old_results(dincae_dir, dry_run=dry_run)

    # Submit
    submit_dincae(dincae_dir, cfg, lake_id, dry_run=dry_run, wait=wait)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run DINCAE with modified parameters (e.g. smaller batch_size for GPU OOM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run, reduce batch size for lake 8
  python rerun_dincae.py --run-root /gws/.../seg0 --lake-ids 8 --batch-size 16

  # All segments from manifest
  python rerun_dincae.py --manifest /gws/.../segment_manifest.json --lake-ids 8 --batch-size 16

  # Dry-run first
  python rerun_dincae.py --manifest /gws/.../segment_manifest.json --lake-ids 8 --batch-size 16 --dry-run

  # Submit all without waiting
  python rerun_dincae.py --manifest /gws/.../segment_manifest.json --lake-ids 8 --batch-size 16 --no-wait
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-root", help="Root of a single experiment run")
    group.add_argument("--manifest", help="Path to segment_manifest.json")

    parser.add_argument("--lake-ids", type=int, nargs="+", required=True,
                        help="Lake IDs to re-run")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="New batch size (default: 16, original was 32)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: keep existing)")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without submitting")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if DINCAE already completed")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit SLURM jobs without waiting for completion")

    args = parser.parse_args()
    wait = not args.no_wait

    # Collect (run_root, lake_id) jobs
    jobs: List[tuple] = []

    if args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"ERROR: manifest not found: {args.manifest}")
            sys.exit(1)
        with open(args.manifest) as f:
            manifest = json.load(f)
        segment_roots = [seg["run_root"] for seg in manifest["segments"]]
        for root in segment_roots:
            for lid in args.lake_ids:
                jobs.append((root, lid))
        print(f"Manifest: {args.manifest}")
        print(f"  Segments: {len(segment_roots)}")
        print(f"  Lakes: {args.lake_ids}")
        print(f"  Total jobs: {len(jobs)}")
    else:
        if not os.path.isdir(args.run_root):
            print(f"ERROR: run-root not found: {args.run_root}")
            sys.exit(1)
        for lid in args.lake_ids:
            jobs.append((args.run_root, lid))

    print(f"Batch size: {args.batch_size}")
    if args.epochs:
        print(f"Epochs: {args.epochs}")
    if args.dry_run:
        print("MODE: DRY-RUN")
    if not wait:
        print("MODE: NO-WAIT (fire and forget)")
    print()

    submitted = []
    for i, (run_root, lake_id) in enumerate(jobs, 1):
        seg_name = os.path.basename(run_root)
        print(f"[{i}/{len(jobs)}] {seg_name} / lake {lake_id}")
        process_lake(
            run_root=run_root,
            lake_id=lake_id,
            alpha=args.alpha,
            batch_size=args.batch_size,
            epochs=args.epochs,
            dry_run=args.dry_run,
            force=args.force,
            wait=wait,
        )
        submitted.append((run_root, lake_id))
        print()

    print(f"{'='*60}")
    print(f"Done: {len(submitted)} DINCAE jobs {'would be ' if args.dry_run else ''}submitted")
    if not args.dry_run and submitted:
        print(f"\nAfter DINCAE completes, run post-processing:")
        if args.manifest:
            print(f"  python scripts/retrofit_post.py \\")
            print(f"    --manifest {args.manifest} \\")
            print(f"    --lake-ids {' '.join(str(lid) for lid in args.lake_ids)} \\")
            print(f"    --config <your_config.json> \\")
            print(f"    --submit-slurm --mem 192G")
        else:
            print(f"  python scripts/retrofit_post.py \\")
            print(f"    --run-root {args.run_root} \\")
            print(f"    --lake-ids {' '.join(str(lid) for lid in args.lake_ids)} \\")
            print(f"    --config <your_config.json>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
