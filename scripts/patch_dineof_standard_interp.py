#!/usr/bin/env python3
"""
patch_dineof_standard_interp.py — Standard DINEOF + pixel-wise interpolation baseline

Creates the "naive" baseline: take raw DINEOF reconstruction (spatially gap-filled,
temporally gappy anomalies) and per-pixel linearly interpolate to full daily timeline,
then apply AddBackTrend → AddBackClimatology → ClampSubZero.

This contrasts with ST-DINEOF which interpolates temporal EOF coefficients (in reduced
EOF space) and reconstructs via spatial EOFs × interpolated temporal coefficients.

Output suffix: _dineof_standard_interp.nc

Usage:
  # Single lake (direct)
  python patch_dineof_standard_interp.py --run-root /path/to/exp --config configs/exp0_baseline.json --lake-id 26

  # All lakes — parallel SLURM array
  python patch_dineof_standard_interp.py --run-root /path/to/exp --config configs/exp0_baseline.json --all --submit-slurm

  # Pilot lake 26, then chain insitu CV
  python patch_dineof_standard_interp.py --run-root /path/to/exp --config configs/exp0_baseline.json --lake-id 26 --chain-cv

Author: Lake CCI Gap-Filling Project
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time as timer
import numpy as np
import xarray as xr
from typing import Optional, List, Dict

# Epoch used throughout the pipeline
EPOCH = np.datetime64("1981-01-01T12:00:00", "ns")
FILL_VALUE = 9999.0


# =============================================================================
# Core: per-pixel interpolate DINEOF anomalies to full daily timeline
# =============================================================================

def interpolate_dineof_anomalies(
    dineof_results_path: str,
    prepared_path: str,
    verbose: bool = True,
) -> Optional[xr.Dataset]:
    """
    Read dineof_results.nc (anomalies on sparse CCI timeline, 9999 fill),
    per-pixel linear interpolate to full daily timeline.

    Returns xr.Dataset with lake_surface_water_temperature_reconstructed in anomaly space on daily timeline.
    """
    if not os.path.isfile(dineof_results_path):
        print(f"  dineof_results.nc not found: {dineof_results_path}")
        return None

    # Read metadata from prepared.nc
    with xr.open_dataset(prepared_path) as ds_prep:
        t0 = int(ds_prep.attrs.get("time_start_days"))
        t1 = int(ds_prep.attrs.get("time_end_days"))
        input_attrs = dict(ds_prep.attrs)
        prep_time = ds_prep["time"].values  # sparse timeline (int days since epoch)

        lat_name = "lat" if "lat" in ds_prep.coords else "latitude"
        lon_name = "lon" if "lon" in ds_prep.coords else "longitude"
        lat_vals = ds_prep[lat_name].values
        lon_vals = ds_prep[lon_name].values
        lakeid_data = ds_prep["lakeid"].copy() if "lakeid" in ds_prep else None

    # Handle time as int days
    if np.issubdtype(prep_time.dtype, np.datetime64):
        sparse_days = ((prep_time.astype("datetime64[ns]").astype("int64")
                        - EPOCH.astype("int64")) // 86_400_000_000_000).astype("int64")
    else:
        sparse_days = prep_time.astype("int64")

    full_days = np.arange(t0, t1 + 1, dtype="int64")
    full_time = EPOCH + full_days.astype("timedelta64[D]")

    # Read DINEOF anomalies (raw output, 9999 fill, no named dims/coords)
    import netCDF4
    nc = netCDF4.Dataset(dineof_results_path)
    _rv = "lake_surface_water_temperature_reconstructed" if "lake_surface_water_temperature_reconstructed" in nc.variables else "temp_filled"
    temp_raw = nc[_rv][:, :, :]  # (T_sparse, lat, lon)
    nc.close()

    # Mask 9999 fill values
    temp_anomaly = temp_raw.astype("float64")
    temp_anomaly[np.abs(temp_anomaly - FILL_VALUE) < 1] = np.nan

    T_sparse, ny, nx = temp_anomaly.shape
    T_full = len(full_days)

    if verbose:
        print(f"  Sparse: {T_sparse} timesteps, Daily: {T_full} timesteps, Grid: {ny}x{nx}")
        n_finite = np.sum(np.isfinite(temp_anomaly[0]))
        print(f"  Lake pixels (t=0): {n_finite}")

    # Validate timeline alignment
    if T_sparse != len(sparse_days):
        print(f"  ERROR: dineof_results has {T_sparse} timesteps but prepared.nc has {len(sparse_days)}")
        return None

    temp_full = np.full((T_full, ny, nx), np.nan, dtype="float32")

    sparse_x = sparse_days.astype("float64")
    full_x = full_days.astype("float64")

    # Per-pixel linear interpolation (interior only, NaN at edges)
    n_interp = 0
    for iy in range(ny):
        for ix in range(nx):
            col = temp_anomaly[:, iy, ix]
            valid = np.isfinite(col)
            if valid.sum() < 2:
                # Place available values without interpolation
                for i_s in range(T_sparse):
                    if valid[i_s]:
                        j = np.searchsorted(full_days, sparse_days[i_s])
                        if j < T_full and full_days[j] == sparse_days[i_s]:
                            temp_full[j, iy, ix] = col[i_s]
                continue

            x_valid = sparse_x[valid]
            y_valid = col[valid]

            # Interior range: first valid to last valid
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

    # Build xarray Dataset
    ds_out = xr.Dataset()
    ds_out = ds_out.assign_coords({
        "time": full_time,
        lat_name: lat_vals,
        lon_name: lon_vals,
    })

    ds_out["lake_surface_water_temperature_reconstructed"] = xr.DataArray(
        temp_full,
        dims=("time", lat_name, lon_name),
        coords={"time": full_time, lat_name: lat_vals, lon_name: lon_vals},
        attrs={"comment": "DINEOF anomalies pixel-wise interpolated to full daily timeline"},
    )

    if lakeid_data is not None:
        ds_out["lakeid"] = lakeid_data

    ds_out.attrs.update(input_attrs)
    ds_out.attrs["source_model"] = "DINEOF_standard"
    ds_out.attrs["interpolation_method"] = "per_pixel_linear"
    ds_out.attrs["interpolation_edge_policy"] = "leave_nan"
    ds_out.attrs["interpolation_source"] = os.path.basename(dineof_results_path)

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
    """Apply AddBackTrend, AddBackClimatology, ClampSubZero."""
    try:
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.add_back_trend import AddBackTrendStep
        from processors.postprocessor.post_steps.add_back_climatology import AddBackClimatologyStep
        from processors.postprocessor.post_steps.clamp_subzero import ClampSubZeroStep
    except ImportError:
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.add_back_trend import AddBackTrendStep
        from processors.postprocessor.post_steps.add_back_climatology import AddBackClimatologyStep
        from processors.postprocessor.post_steps.clamp_subzero import ClampSubZeroStep

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
    ctx.input_attrs = dict(ds.attrs)

    for StepClass, name in [
        (AddBackTrendStep, "AddBackTrend"),
        (AddBackClimatologyStep, "AddBackClimatology"),
        (ClampSubZeroStep, "ClampSubZero"),
    ]:
        step = StepClass()
        if step.should_apply(ctx, ds):
            if verbose:
                print(f"  Applying: {name}")
            ds = step.apply(ctx, ds)
        else:
            if verbose:
                print(f"  Skipping: {name}")

    return ds


# =============================================================================
# Path resolution
# =============================================================================

def resolve_paths(run_root: str, lake_id: int, alpha: str, config: dict) -> Dict[str, str]:
    """Resolve all paths for a lake."""
    lake_id9 = f"{lake_id:09d}"
    P = config.get("paths", {})

    prepared = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
    dineof_results = os.path.join(run_root, "dineof", lake_id9, alpha, "dineof_results.nc")
    post_dir = os.path.join(run_root, "post", lake_id9, alpha)

    clim = P.get("climatology_template", "").replace(
        "{lake_id9}", lake_id9).replace("{lake_id}", str(lake_id))

    out_path = os.path.join(
        post_dir,
        f"LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-filled_fine_dineof_standard_interp.nc"
    )

    return {
        "prepared": prepared,
        "dineof_results": dineof_results,
        "post_dir": post_dir,
        "climatology": clim,
        "out_path": out_path,
    }


def find_lakes_with_dineof(run_root: str, alpha: str = "a1000") -> List[int]:
    """Find all lake IDs that have dineof_results.nc."""
    dineof_dir = os.path.join(run_root, "dineof")
    if not os.path.isdir(dineof_dir):
        return []
    lake_ids = []
    for d in sorted(os.listdir(dineof_dir)):
        try:
            lid = int(d)
            results_path = os.path.join(dineof_dir, d, alpha, "dineof_results.nc")
            if os.path.isfile(results_path):
                lake_ids.append(lid)
        except ValueError:
            continue
    return lake_ids


# =============================================================================
# Per-lake processing
# =============================================================================

def process_lake(
    run_root: str,
    lake_id: int,
    config: dict,
    alpha: str = "a1000",
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """Process a single lake. Returns True if successful."""
    lake_id9 = f"{lake_id:09d}"
    paths = resolve_paths(run_root, lake_id, alpha, config)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Lake {lake_id} ({lake_id9}) — standard DINEOF + pixel-wise interp")
        print(f"{'='*60}")

    # Check inputs
    if not os.path.isfile(paths["prepared"]):
        print(f"  ERROR: prepared.nc not found: {paths['prepared']}")
        return False

    if not os.path.isfile(paths["dineof_results"]):
        print(f"  ERROR: dineof_results.nc not found: {paths['dineof_results']}")
        return False

    if not os.path.isfile(paths["climatology"]):
        print(f"  WARNING: climatology not found: {paths['climatology']}")

    out_path = paths["out_path"]
    if os.path.isfile(out_path) and not force:
        try:
            with xr.open_dataset(out_path) as ds_check:
                if ds_check.attrs.get("dineof_standard_interp_done", 0) == 1:
                    if verbose:
                        print(f"  Already exists, skipping (use --force to overwrite)")
                    return True
        except Exception:
            pass

    t_start = timer.time()

    # Step 1: Per-pixel interpolate anomalies to daily
    if verbose:
        print(f"  Step 1: Per-pixel interpolating DINEOF anomalies to daily...")
    ds = interpolate_dineof_anomalies(
        paths["dineof_results"], paths["prepared"], verbose
    )
    if ds is None:
        return False

    # Step 2: Add back trend + climatology + clamp
    if verbose:
        print(f"  Step 2: Applying trend + climatology + clamp...")
    ds = apply_trend_climatology_clamp(
        ds, paths["climatology"], output_units="kelvin", verbose=verbose
    )

    # Apply lake mask
    if "lakeid" in ds:
        lake_mask = ds["lakeid"].values
        non_lake = np.isnan(lake_mask) | (lake_mask == 0)
        if non_lake.any():
            tf = ds["lake_surface_water_temperature_reconstructed"].values
            tf[:, non_lake] = np.nan
            ds["lake_surface_water_temperature_reconstructed"].values = tf
            if verbose:
                print(f"  Applied lake mask: {int(non_lake.sum())} non-lake pixels NaN'd")

    # Mark as done
    ds.attrs["dineof_standard_interp_done"] = 1
    ds.attrs["dineof_standard_interp_date"] = timer.strftime("%Y-%m-%dT%H:%M:%S", timer.gmtime())

    # Step 3: Write
    if verbose:
        print(f"  Step 3: Writing {os.path.basename(out_path)}...")

    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    if "lake_surface_water_temperature_reconstructed" in ds:
        enc["lake_surface_water_temperature_reconstructed"] = {"dtype": "float32", "zlib": True, "complevel": 5}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path, encoding=enc)

    elapsed = timer.time() - t_start
    if verbose:
        tf = ds["lake_surface_water_temperature_reconstructed"]
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
    chain_cv: bool = False,
    account: str = "eocis_chuk",
    partition: str = "standard",
    qos: str = "standard",
    mem: str = "64G",
    time_limit: str = "01:00:00",
):
    """Submit SLURM array job — one task per lake, fully parallel."""
    log_dir = os.path.join(run_root, "patch_dineof_standard_interp_logs")
    os.makedirs(log_dir, exist_ok=True)

    lake_list_path = os.path.join(log_dir, "lake_list.txt")
    with open(lake_list_path, "w") as f:
        for lid in lake_ids:
            f.write(f"{lid}\n")

    n_lakes = len(lake_ids)
    script_path = os.path.abspath(__file__)
    force_flag = "--force" if force else ""

    # --- Job 1: Interpolation array ---
    interp_script = f"""#!/bin/bash
#SBATCH --job-name=dineof_std_interp
#SBATCH --array=1-{n_lakes}
#SBATCH -o {log_dir}/interp_%a.out
#SBATCH -e {log_dir}/interp_%a.err
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

    interp_slurm_path = os.path.join(log_dir, "interp_array.slurm")
    with open(interp_slurm_path, "w") as f:
        f.write(interp_script)

    result = subprocess.run(["sbatch", "--parsable", interp_slurm_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr.strip()}")
        return

    interp_job_id = result.stdout.strip()
    print(f"Submitted interpolation array: {interp_job_id} ({n_lakes} tasks)")
    print(f"  Lakes: {lake_ids[:10]}{'...' if len(lake_ids) > 10 else ''}")
    print(f"  Logs: {log_dir}/interp_*.out")

    if not chain_cv:
        return

    # --- Job 2: Insitu CV array (depends on interp) ---
    # Uses the existing cv_validation.py / assemble_insitu_cv.py
    cv_script_path = os.path.join(os.path.dirname(script_path), "scripts", "assemble_insitu_cv.py")
    if not os.path.isfile(cv_script_path):
        cv_script_path = os.path.join(os.path.dirname(os.path.dirname(script_path)),
                                       "scripts", "assemble_insitu_cv.py")

    cv_slurm = f"""#!/bin/bash
#SBATCH --job-name=cv_std_interp
#SBATCH --array=1-{n_lakes}
#SBATCH -o {log_dir}/cv_%a.out
#SBATCH -e {log_dir}/cv_%a.err
#SBATCH --mem={mem}
#SBATCH -t {time_limit}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}

source ~/miniforge3/bin/activate
conda activate lake_cci_gapfilling

LAKE_ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {lake_list_path})
echo "Insitu CV for lake $LAKE_ID (dineof_standard_interp): $(date)"

python -u {cv_script_path} \\
    --run-root {run_root} \\
    --config {config_path} \\
    --lake-id $LAKE_ID \\
    --method dineof_standard_interp \\
    --phase extract

echo "Done: $(date)"
"""

    cv_slurm_path = os.path.join(log_dir, "cv_array.slurm")
    with open(cv_slurm_path, "w") as f:
        f.write(cv_slurm)

    result2 = subprocess.run(
        ["sbatch", "--parsable", f"--dependency=afterany:{interp_job_id}", cv_slurm_path],
        capture_output=True, text=True
    )
    if result2.returncode == 0:
        cv_job_id = result2.stdout.strip()
        print(f"Chained insitu CV array: {cv_job_id} (depends on {interp_job_id})")
        print(f"  Logs: {log_dir}/cv_*.out")
    else:
        print(f"CV sbatch failed: {result2.stderr.strip()}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standard DINEOF + pixel-wise interpolation baseline"
    )
    parser.add_argument("--run-root", required=True,
                        help="Experiment run root")
    parser.add_argument("--config", required=True,
                        help="Experiment config JSON (for climatology template)")
    parser.add_argument("--alpha", default="a1000")

    lake_group = parser.add_mutually_exclusive_group(required=True)
    lake_group.add_argument("--lake-id", type=int, help="Single lake ID")
    lake_group.add_argument("--lake-ids", type=int, nargs="+", help="Multiple lake IDs")
    lake_group.add_argument("--all", action="store_true", help="All lakes with dineof_results.nc")

    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--submit-slurm", action="store_true",
                        help="Submit as parallel SLURM array")
    parser.add_argument("--chain-cv", action="store_true",
                        help="Chain insitu CV extraction as dependency job")
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
        lake_ids = find_lakes_with_dineof(args.run_root, args.alpha)
        print(f"Found {len(lake_ids)} lakes with dineof_results.nc")
    else:
        parser.error("Specify --lake-id, --lake-ids, or --all")
        return

    if not lake_ids:
        print("No lakes to process")
        return

    # SLURM mode
    if args.submit_slurm:
        submit_slurm(
            run_root=args.run_root,
            config_path=os.path.abspath(args.config),
            lake_ids=lake_ids,
            alpha=args.alpha,
            force=args.force,
            chain_cv=args.chain_cv,
        )
        return

    # Direct mode (single lake or sequential)
    n_ok = 0
    n_fail = 0
    for lake_id in lake_ids:
        try:
            ok = process_lake(
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
