#!/usr/bin/env python3
"""
patch_dincae_interp.py — Fix DINCAE temporal interpolation

The original _dincae_interp_full.nc was created by interpolating final LSWT values
(after trend + climatology were already added back). This is incorrect — interpolation
should happen in anomaly space, matching how DINEOF interp files are produced.

This script:
  1. Reads dincae_results.nc (anomalies on sparse CCI timeline)
  2. Per-pixel linear interpolates anomalies to full daily timeline
  3. Applies AddBackTrend → AddBackClimatology → ClampSubZero (the actual pipeline steps)
  4. Writes corrected _dincae_interp_full.nc

Usage:
  # Single lake (direct)
  python patch_dincae_interp.py --run-root /path/to/exp --config configs/exp0_baseline.json --lake-id 1

  # All lakes — parallel SLURM array
  python patch_dincae_interp.py --run-root /path/to/exp --config configs/exp0_baseline.json --all --submit-slurm

  # Specific lakes — parallel SLURM array
  python patch_dincae_interp.py --run-root /path/to/exp --config configs/exp0_baseline.json --lake-ids 1 3 88 --submit-slurm --force

Author: Lake CCI Gap-Filling Project
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import numpy as np
import xarray as xr
from typing import Optional, List, Dict

# Epoch used throughout the pipeline
EPOCH = np.datetime64("1981-01-01T12:00:00", "ns")


# =============================================================================
# Core: interpolate DINCAE anomalies to full daily timeline
# =============================================================================

def interpolate_dincae_anomalies(
    dincae_results_path: str,
    prepared_path: str,
    verbose: bool = True,
) -> Optional[xr.Dataset]:
    """
    Read dincae_results.nc (anomalies on prepared.nc sparse timeline),
    per-pixel linear interpolate to full daily timeline.

    Returns xr.Dataset with datetime64 time coords and temp_filled in anomaly space.
    Does NOT apply trend/climatology — caller does that via pipeline steps.
    """
    if not os.path.isfile(dincae_results_path):
        print(f"  dincae_results.nc not found: {dincae_results_path}")
        return None

    # Read full_days range from prepared.nc attrs
    with xr.open_dataset(prepared_path) as ds_prep:
        t0 = int(ds_prep.attrs.get("time_start_days"))
        t1 = int(ds_prep.attrs.get("time_end_days"))
        input_attrs = dict(ds_prep.attrs)
        lat_name = "lat" if "lat" in ds_prep.coords else "latitude"
        lon_name = "lon" if "lon" in ds_prep.coords else "longitude"
        lat_vals = ds_prep[lat_name].values
        lon_vals = ds_prep[lon_name].values
        lakeid_data = ds_prep.get("lakeid")
        lakeid_vals = lakeid_data.copy() if lakeid_data is not None else None

    full_days = np.arange(t0, t1 + 1, dtype="int64")
    full_time = EPOCH + full_days.astype("timedelta64[D]")

    # Read DINCAE anomalies
    ds_dincae = xr.open_dataset(dincae_results_path)
    # Time coords are integer days since epoch (same basis as prepared.nc)
    dincae_time = ds_dincae["time"].values
    # Handle both integer and datetime64 time coords
    if np.issubdtype(dincae_time.dtype, np.datetime64):
        # Convert datetime64 to integer days
        dincae_days = ((dincae_time.astype("datetime64[ns]").astype("int64")
                        - EPOCH.astype("int64")) // 86_400_000_000_000).astype("int64")
    else:
        dincae_days = dincae_time.astype("int64")

    temp_anomaly = ds_dincae["temp_filled"].values.astype("float64")  # (T_sparse, lat, lon)
    ds_dincae.close()

    T_full = len(full_days)
    ny, nx = temp_anomaly.shape[1], temp_anomaly.shape[2]
    temp_full = np.full((T_full, ny, nx), np.nan, dtype="float32")

    sparse_x = dincae_days.astype("float64")
    full_x = full_days.astype("float64")

    # Per-pixel linear interpolation (interior only, leave_nan at edges)
    n_interp = 0
    for iy in range(ny):
        for ix in range(nx):
            col = temp_anomaly[:, iy, ix]
            valid = np.isfinite(col)
            if valid.sum() < 2:
                # Not enough points — place available values but don't interpolate
                for i_s, d in enumerate(dincae_days):
                    j = np.searchsorted(full_days, d)
                    if j < T_full and full_days[j] == d and valid[i_s]:
                        temp_full[j, iy, ix] = col[i_s]
                continue

            x_valid = sparse_x[valid]
            y_valid = col[valid]

            # Interior range: first valid to last valid in full timeline
            i0 = np.searchsorted(full_x, x_valid[0])
            i1 = np.searchsorted(full_x, x_valid[-1])
            if i1 < T_full and full_x[i1] == x_valid[-1]:
                i1_end = i1 + 1
            else:
                i1_end = i1

            if i0 < i1_end:
                temp_full[i0:i1_end, iy, ix] = np.interp(
                    full_x[i0:i1_end], x_valid, y_valid
                ).astype("float32")
                n_interp += 1

    if verbose:
        print(f"  Interpolated {n_interp} pixels onto {T_full} daily timesteps (anomaly space)")

    # Build xarray Dataset (matching Pass 3 format)
    ds_out = xr.Dataset()
    ds_out = ds_out.assign_coords({
        "time": full_time,
        lat_name: lat_vals,
        lon_name: lon_vals,
    })

    ds_out["temp_filled"] = xr.DataArray(
        temp_full,
        dims=("time", lat_name, lon_name),
        coords={"time": full_time, lat_name: lat_vals, lon_name: lon_vals},
        attrs={"comment": "DINCAE anomalies interpolated to full daily timeline (before trend/climatology add-back)"},
    )

    if lakeid_vals is not None:
        ds_out["lakeid"] = lakeid_vals

    # Copy prepared.nc attrs (needed by AddBackTrendStep etc.)
    ds_out.attrs.update(input_attrs)
    ds_out.attrs["source_model"] = "DINCAE"
    ds_out.attrs["interpolation_method"] = "per_pixel_linear"
    ds_out.attrs["interpolation_edge_policy"] = "leave_nan"
    ds_out.attrs["interpolation_source"] = os.path.basename(dincae_results_path)

    return ds_out


# =============================================================================
# Apply pipeline steps: trend → climatology → clamp
# =============================================================================

def apply_trend_climatology_clamp(
    ds: xr.Dataset,
    climatology_path: str,
    output_units: str = "kelvin",
    verbose: bool = True,
) -> xr.Dataset:
    """
    Apply AddBackTrend, AddBackClimatology, ClampSubZero to a dataset
    using the actual pipeline step implementations.
    """
    try:
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.add_back_trend import AddBackTrendStep
        from processors.postprocessor.post_steps.add_back_climatology import AddBackClimatologyStep
        from processors.postprocessor.post_steps.clamp_subzero import ClampSubZeroStep
    except ImportError:
        # Try adding src to path
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.add_back_trend import AddBackTrendStep
        from processors.postprocessor.post_steps.add_back_climatology import AddBackClimatologyStep
        from processors.postprocessor.post_steps.clamp_subzero import ClampSubZeroStep

    # Build a minimal PostContext with the fields these steps need
    ctx = PostContext(
        lake_path="<unused>",
        dineof_input_path="<unused>",
        dineof_output_path="<unused>",
        output_path="<unused>",
        output_html_folder=None,
        climatology_path=climatology_path,
        output_units=output_units,
        keep_attrs=True,
    )
    # Steps read trend params from ctx.input_attrs
    ctx.input_attrs = dict(ds.attrs)

    # AddBackTrend
    step_trend = AddBackTrendStep()
    if step_trend.should_apply(ctx, ds):
        if verbose:
            print(f"  Applying: AddBackTrend")
        ds = step_trend.apply(ctx, ds)
    else:
        if verbose:
            print(f"  Skipping: AddBackTrend (no detrend attrs)")

    # AddBackClimatology
    step_clim = AddBackClimatologyStep()
    if step_clim.should_apply(ctx, ds):
        if verbose:
            print(f"  Applying: AddBackClimatology")
        ds = step_clim.apply(ctx, ds)
    else:
        if verbose:
            print(f"  Skipping: AddBackClimatology (no climatology path)")

    # ClampSubZero
    step_clamp = ClampSubZeroStep()
    if step_clamp.should_apply(ctx, ds):
        if verbose:
            print(f"  Applying: ClampSubZero")
        ds = step_clamp.apply(ctx, ds)

    return ds


# =============================================================================
# Path resolution (matches retrofit_post.py logic)
# =============================================================================

def resolve_paths(
    run_root: str, lake_id: int, alpha: str, config: dict
) -> Dict[str, str]:
    """Resolve all paths needed for a lake."""
    lake_id9 = f"{lake_id:09d}"
    P = config.get("paths", {})

    prepared_path = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
    dincae_results = os.path.join(run_root, "dincae", lake_id9, alpha, "dincae_results.nc")
    post_dir = os.path.join(run_root, "post", lake_id9, alpha)

    clim = P.get("climatology_template", "").replace(
        "{lake_id9}", lake_id9).replace("{lake_id}", str(lake_id))

    # Find existing dincae_interp_full.nc in post_dir
    interp_pattern = os.path.join(post_dir, "*_dincae_interp_full.nc")
    existing = glob.glob(interp_pattern)
    if existing:
        out_path = existing[0]
    else:
        # Construct from naming convention
        dineof_files = glob.glob(os.path.join(post_dir, "*_dincae.nc"))
        if dineof_files:
            out_path = dineof_files[0].replace("_dincae.nc", "_dincae_interp_full.nc")
        else:
            out_path = os.path.join(
                post_dir,
                f"LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-filled_fine_dincae_interp_full.nc"
            )

    return {
        "prepared": prepared_path,
        "dincae_results": dincae_results,
        "post_dir": post_dir,
        "climatology": clim,
        "out_path": out_path,
    }


def find_lakes_with_post(run_root: str, alpha: str = "a1000") -> List[int]:
    """Find all lake IDs that have post-processed results."""
    post_dir = os.path.join(run_root, "post")
    if not os.path.isdir(post_dir):
        return []
    lake_ids = []
    for d in sorted(os.listdir(post_dir)):
        try:
            lid = int(d)
            # Check that dincae.nc exists (so interp makes sense)
            if glob.glob(os.path.join(post_dir, d, alpha, "*_dincae.nc")):
                lake_ids.append(lid)
        except ValueError:
            continue
    return lake_ids


# =============================================================================
# Per-lake patching
# =============================================================================

def patch_lake(
    run_root: str,
    lake_id: int,
    config: dict,
    alpha: str = "a1000",
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Patch a single lake's _dincae_interp_full.nc.
    Returns True if successful.
    """
    lake_id9 = f"{lake_id:09d}"
    paths = resolve_paths(run_root, lake_id, alpha, config)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Patching lake {lake_id} ({lake_id9})")
        print(f"{'='*60}")

    # Validate inputs exist
    if not os.path.isfile(paths["prepared"]):
        print(f"  ERROR: prepared.nc not found: {paths['prepared']}")
        return False

    if not os.path.isfile(paths["dincae_results"]):
        print(f"  ERROR: dincae_results.nc not found: {paths['dincae_results']}")
        return False

    if not os.path.isfile(paths["climatology"]):
        print(f"  WARNING: climatology not found: {paths['climatology']}")
        # Continue anyway — AddBackClimatologyStep will skip gracefully

    out_path = paths["out_path"]
    if os.path.isfile(out_path) and not force:
        # Check if already patched (has anomaly interpolation marker)
        try:
            with xr.open_dataset(out_path) as ds_check:
                if ds_check.attrs.get("dincae_interp_anomaly_space", 0) == 1:
                    if verbose:
                        print(f"  Already patched (anomaly space), skipping")
                    return True
        except Exception:
            pass

    t_start = time.time()

    # Step 1: Interpolate anomalies
    if verbose:
        print(f"  Step 1: Interpolating DINCAE anomalies to full daily...")
    ds = interpolate_dincae_anomalies(
        paths["dincae_results"], paths["prepared"], verbose
    )
    if ds is None:
        return False

    # Step 2: Apply trend + climatology + clamp
    if verbose:
        print(f"  Step 2: Applying trend + climatology + clamp...")
    ds = apply_trend_climatology_clamp(
        ds, paths["climatology"], output_units="kelvin", verbose=verbose
    )

    # Mark as patched
    ds.attrs["dincae_interp_anomaly_space"] = 1
    ds.attrs["dincae_interp_patch_version"] = "2026-03-05"

    # Step 3: Write
    if verbose:
        print(f"  Step 3: Writing {os.path.basename(out_path)}...")

    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    if "temp_filled" in ds:
        enc["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path, encoding=enc)

    elapsed = time.time() - t_start
    if verbose:
        tf = ds["temp_filled"]
        n_valid = int(tf.count().values)
        vmin = float(tf.min().values)
        vmax = float(tf.max().values)
        print(f"  Done: {ds.sizes['time']} timesteps, {n_valid:,} non-NaN, "
              f"range [{vmin:.2f}, {vmax:.2f}], {elapsed:.1f}s")

    ds.close()
    return True


# =============================================================================
# SLURM submission
# =============================================================================

def submit_slurm(
    run_root: str,
    config_path: str,
    lake_ids: List[int],
    alpha: str = "a1000",
    force: bool = False,
    account: str = "eocis_chuk",
    partition: str = "standard",
    qos: str = "short",
    mem: str = "64G",
    time_limit: str = "4:00:00",
):
    """Submit a SLURM array job — one task per lake, all parallel."""
    log_dir = os.path.join(run_root, "patch_dincae_interp_logs")
    os.makedirs(log_dir, exist_ok=True)

    lake_list_path = os.path.join(log_dir, "lake_list.txt")
    with open(lake_list_path, "w") as f:
        for lid in lake_ids:
            f.write(f"{lid}\n")

    n_lakes = len(lake_ids)
    script_path = os.path.abspath(__file__)
    force_flag = "--force" if force else ""

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=patch_dincae
#SBATCH --array=1-{n_lakes}
#SBATCH -o {log_dir}/patch_%a.out
#SBATCH -e {log_dir}/patch_%a.err
#SBATCH --mem={mem}
#SBATCH -t {time_limit}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {lake_list_path})
echo "Lake $LAKE_ID started: $(date)"

python -u {script_path} \\
    --run-root {run_root} \\
    --config {config_path} \\
    --alpha {alpha} \\
    --lake-id $LAKE_ID \\
    {force_flag}

echo "Done: $(date)"
"""

    slurm_path = os.path.join(log_dir, "patch_dincae.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

    result = subprocess.run(
        ["sbatch", slurm_path],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted SLURM array: {job_id} ({n_lakes} tasks)")
        print(f"  Lakes: {lake_ids}")
        print(f"  Logs:  {log_dir}/patch_*.out")
        return job_id
    else:
        print(f"sbatch failed: {result.stderr.strip()}")
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Patch _dincae_interp_full.nc: interpolate anomalies instead of final LSWT"
    )
    parser.add_argument("--run-root", required=True,
                        help="Experiment run root")
    parser.add_argument("--config", required=True,
                        help="Experiment config JSON (for climatology template)")
    parser.add_argument("--alpha", default="a1000")
    parser.add_argument("--lake-id", type=int, default=None,
                        help="Single lake ID to patch")
    parser.add_argument("--lake-ids", type=int, nargs="+", default=None,
                        help="Multiple lake IDs to patch")
    parser.add_argument("--all", action="store_true",
                        help="Patch all lakes")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite even if already patched")
    parser.add_argument("--submit-slurm", action="store_true",
                        help="Submit as parallel SLURM array (one task per lake)")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    with open(args.config) as f:
        config = json.load(f)

    # Resolve lake list
    if args.lake_id:
        lake_ids = [args.lake_id]
    elif args.lake_ids:
        lake_ids = args.lake_ids
    elif args.all:
        lake_ids = find_lakes_with_post(args.run_root, args.alpha)
        print(f"Found {len(lake_ids)} lakes with post results")
    else:
        parser.error("Specify --lake-id, --lake-ids, or --all")
        return

    if not lake_ids:
        print("No lakes to process")
        return

    # SLURM mode: submit array job and exit
    if args.submit_slurm:
        submit_slurm(
            run_root=args.run_root,
            config_path=os.path.abspath(args.config),
            lake_ids=lake_ids,
            alpha=args.alpha,
            force=args.force,
        )
        return

    # Direct mode: process lakes sequentially (used by SLURM array tasks)
    n_ok = 0
    n_fail = 0
    for lake_id in lake_ids:
        try:
            ok = patch_lake(
                args.run_root, lake_id, config, args.alpha,
                force=args.force, verbose=verbose
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f"  EXCEPTION for lake {lake_id}: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
