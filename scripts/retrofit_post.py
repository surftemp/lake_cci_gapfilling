#!/usr/bin/env python3
"""
retrofit_post.py - Apply post-processing updates to existing pipeline outputs.

Updates:
  1. Clamp sub-zero LSWT to 0°C in all post/*.nc files
  2. Create DINCAE daily interpolation (_dincae_interp_full.nc) from _dincae.nc
  3. Re-run LSWT time series plots (discovers all files including new ones)
  4. Re-run in-situ validation (discovers all files including new ones)

Usage:
  # Specific lakes in a single run
  python retrofit_post.py \\
    --run-root /gws/.../anomaly-20260131-exp0_baseline_both/ \\
    --lake-ids 380 88 2

  # All lakes in a run
  python retrofit_post.py \\
    --run-root /gws/.../anomaly-20260131-exp0_baseline_both/ \\
    --all-lakes

  # Segment manifest mode: iterate over all segment run roots
  python retrofit_post.py \\
    --manifest /gws/.../exp0_baseline_large_merged_4splits/segment_manifest.json \\
    --lake-ids 5 6 9 11 \\
    --config configs/exp0_baseline.json

  # Segment manifest + SLURM with higher memory for large lakes
  python retrofit_post.py \\
    --manifest /gws/.../exp0_baseline_large_merged_4splits/segment_manifest.json \\
    --lake-ids 5 6 9 11 \\
    --config configs/exp0_baseline.json \\
    --submit-slurm --mem 192G

  # Dry-run: show what would be changed
  python retrofit_post.py \\
    --run-root /gws/.../anomaly-20260131-exp0_baseline_both/ \\
    --all-lakes --dry-run

  # Skip plot/insitu regeneration (clamp + interp only)
  python retrofit_post.py \\
    --run-root /gws/.../ --all-lakes --no-plots --no-insitu

  # Generate SLURM array job for batch processing
  python retrofit_post.py \\
    --run-root /gws/.../ --all-lakes --submit-slurm --mem 192G

  # Fix EOF filtering (re-filter + full post reprocessing)
  python retrofit_post.py \\
    --run-root /gws/.../ --all-lakes --refilter-eofs --config exp1.json

Author: Shaerdan / NCEO / University of Reading
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import subprocess
import numpy as np
import xarray as xr
from typing import Optional, List, Dict


# =============================================================================
# Step 0: Re-filter EOFs + reconstruct (fix dropped-timestep bug)
# =============================================================================

def refilter_and_reconstruct_eofs(
    dineof_dir: str,
    prepared_path: str,
    lake_id: int,
    config: dict,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Re-run EOF filtering with the current code (interpolation-replacement
    instead of dropping outlier timesteps), then re-reconstruct the filtered
    and filtered-interpolated results.

    Operates entirely in the dineof/{lake}/{alpha}/ directory:
      eofs.nc → eofs_filtered.nc (corrected)
      eofs_filtered.nc → eofs_filtered_interpolated.nc
      eofs_filtered.nc → dineof_results_eof_filtered.nc
      eofs_filtered_interpolated.nc → dineof_results_eof_filtered_interp_full.nc

    Returns True if any files were modified.
    """
    eofs_path = os.path.join(dineof_dir, "eofs.nc")
    if not os.path.isfile(eofs_path):
        print(f"  No eofs.nc in {dineof_dir}, skipping EOF refilter")
        return False

    # Check if already fixed: eofs.nc and eofs_filtered.nc should have same t dim
    filtered_path = os.path.join(dineof_dir, "eofs_filtered.nc")
    if os.path.isfile(filtered_path) and not force:
        try:
            with xr.open_dataset(eofs_path) as ds_orig, \
                 xr.open_dataset(filtered_path) as ds_filt:
                t_orig = ds_orig.sizes.get('t', ds_orig.dims.get('t', -1))
                t_filt = ds_filt.sizes.get('t', ds_filt.dims.get('t', -2))
                if t_orig == t_filt:
                    print(f"  eofs_filtered.nc already has correct timestep count "
                          f"({t_orig}), skipping (use --force to redo)")
                    return False
                else:
                    print(f"  eofs_filtered.nc has {t_filt} timesteps vs {t_orig} in eofs.nc — needs fix")
        except Exception:
            pass

    if dry_run:
        print(f"  [DRY-RUN] Would re-filter EOFs and reconstruct in {dineof_dir}")
        return False

    # Import pipeline steps
    try:
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.filter_eofs import FilterTemporalEOFsStep
        from processors.postprocessor.post_steps.interpolate_temporal_eofs import InterpolateTemporalEOFsStep
        from processors.postprocessor.post_steps.reconstruct_from_eofs import ReconstructFromEOFsStep
    except ImportError:
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.filter_eofs import FilterTemporalEOFsStep
        from processors.postprocessor.post_steps.interpolate_temporal_eofs import InterpolateTemporalEOFsStep
        from processors.postprocessor.post_steps.reconstruct_from_eofs import ReconstructFromEOFsStep

    # Construct minimal PostContext
    dineof_results_path = os.path.join(dineof_dir, "dineof_results.nc")
    if not os.path.isfile(dineof_results_path):
        print(f"  dineof_results.nc not found in {dineof_dir}, skipping")
        return False

    ctx = PostContext(
        lake_path="<unused>",
        dineof_input_path=prepared_path,
        dineof_output_path=dineof_results_path,
        output_path="<unused>",
        output_html_folder=None,
        climatology_path=None,
        lake_id=lake_id,
    )

    # Load full_days from prepared.nc attrs (needed by InterpolateTemporalEOFsStep)
    try:
        with xr.open_dataset(prepared_path) as ds_prep:
            t0 = int(ds_prep.attrs.get("time_start_days"))
            t1 = int(ds_prep.attrs.get("time_end_days"))
        ctx.full_days = np.arange(t0, t1 + 1, dtype="int64")
        ctx.time_start_days = t0
        ctx.time_end_days = t1
    except Exception as e:
        print(f"  Failed to read prepared.nc attrs: {e}")
        return False

    # Read filter params from config (use same settings as original pipeline)
    post_cfg = config.get("post_processing", {})
    eof_selection = post_cfg.get("eof_filter_selection", "variance_threshold")
    variance_threshold = post_cfg.get("eof_filter_variance_threshold", 0.5)
    top_n = post_cfg.get("eof_filter_top_n", 3)
    filter_method = post_cfg.get("eof_filter_method", "robust_sd")
    filter_k = post_cfg.get("eof_filter_k", 4.0)

    modified = False

    # Step 0a: Re-filter EOFs
    print(f"  [0a] Re-filtering EOFs (selection={eof_selection}, method={filter_method}, k={filter_k})...")
    filter_step = FilterTemporalEOFsStep(
        method=filter_method,
        k=filter_k,
        eof_selection=eof_selection,
        variance_threshold=variance_threshold,
        top_n_eofs=top_n,
        overwrite=False,  # writes eofs_filtered.nc (not overwriting eofs.nc)
    )
    if filter_step.should_apply(ctx, None):
        filter_step.apply(ctx, None)
        modified = True
    else:
        print(f"  FilterTemporalEOFsStep: should_apply=False, skipping")
        return False

    # Step 0b: Re-interpolate filtered EOFs onto full daily timeline
    print(f"  [0b] Interpolating filtered EOFs to daily...")
    interp_step = InterpolateTemporalEOFsStep(
        target="full",
        edge_policy="leave_nan",
        source_mode="filtered",
    )
    if interp_step.should_apply(ctx, None):
        interp_step.apply(ctx, None)
        modified = True

    # Step 0c: Reconstruct from filtered EOFs
    print(f"  [0c] Reconstructing from filtered EOFs...")
    recon_filtered = ReconstructFromEOFsStep(source_mode="filtered")
    if recon_filtered.should_apply(ctx, None):
        recon_filtered.apply(ctx, None)
        modified = True

    # Step 0d: Reconstruct from filtered+interpolated EOFs
    print(f"  [0d] Reconstructing from filtered+interpolated EOFs...")
    recon_filtered_interp = ReconstructFromEOFsStep(source_mode="filtered_interp")
    if recon_filtered_interp.should_apply(ctx, None):
        recon_filtered_interp.apply(ctx, None)
        modified = True

    return modified


def rerun_post_processing(
    post_dir: str,
    dineof_dir: str,
    prepared_path: str,
    lake_path: str,
    climatology_path: str,
    config_path: Optional[str] = None,
    config: Optional[dict] = None,
    dry_run: bool = False,
) -> bool:
    """
    Re-run the full post-processing pipeline for a lake.
    This regenerates ALL post files (_dineof.nc, _eof_filtered.nc, etc.)
    from the (corrected) dineof results.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would re-run post-processing pipeline")
        return False

    try:
        from processors.postprocessor.post_process import PostProcessor, PostOptions
    except ImportError:
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from processors.postprocessor.post_process import PostProcessor, PostOptions

    # Find the base _dineof.nc in post_dir to get the output naming pattern
    dineof_nc_files = glob.glob(os.path.join(post_dir, "*_dineof.nc"))
    if not dineof_nc_files:
        print(f"  No *_dineof.nc found in {post_dir}")
        return False
    output_path = dineof_nc_files[0]

    dineof_results_path = os.path.join(dineof_dir, "dineof_results.nc")
    if not os.path.isfile(dineof_results_path):
        print(f"  dineof_results.nc not found in {dineof_dir}")
        return False

    # Build PostOptions from config
    post_cfg = (config or {}).get("post_processing", {})
    options = PostOptions(
        eof_filter_selection=post_cfg.get("eof_filter_selection", "variance_threshold"),
        eof_filter_variance_threshold=post_cfg.get("eof_filter_variance_threshold", 0.5),
        eof_filter_top_n=post_cfg.get("eof_filter_top_n", 3),
        eof_filter_method=post_cfg.get("eof_filter_method", "robust_sd"),
        eof_filter_k=post_cfg.get("eof_filter_k", 4.0),
        output_units=post_cfg.get("output_units", "celsius"),
        clamp_subzero=True,
        dincae_temporal_interp=True,
    )

    print(f"  Running PostProcessor...")
    pp = PostProcessor(
        lake_path=lake_path,
        dineof_input_path=prepared_path,
        dineof_output_path=dineof_results_path,
        output_path=output_path,
        output_html_folder=None,
        climatology_file=climatology_path,
        experiment_config_file=config_path,
        options=options,
    )
    pp.run()
    return True


# =============================================================================
# Step 1: Clamp sub-zero LSWT
# =============================================================================

def clamp_subzero_file(nc_path: str, threshold: float = 0.0, dry_run: bool = False) -> bool:
    """
    Clamp sub-zero temp_filled values in-place.
    Returns True if file was modified, False if skipped.
    
    Uses write-to-temp + os.replace to avoid in-place write issues
    on parallel filesystems (e.g. JASMIN GWS).
    """
    try:
        ds = xr.load_dataset(nc_path)

        # Skip if already clamped
        if ds.attrs.get("subzero_clamped", 0) == 1:
            return False

        if "temp_filled" not in ds:
            return False

        arr = ds["temp_filled"].values
        n_subzero = int(np.nansum(arr < threshold))
        total_valid = int(np.nansum(np.isfinite(arr)))

        if dry_run:
            pct = 100.0 * n_subzero / total_valid if total_valid > 0 else 0
            print(f"  [DRY-RUN] {os.path.basename(nc_path)}: {n_subzero:,} sub-zero values "
                  f"({pct:.2f}% of {total_valid:,})")
            return False

        if n_subzero > 0:
            ds["temp_filled"] = ds["temp_filled"].clip(min=threshold)
            pct = 100.0 * n_subzero / total_valid if total_valid > 0 else 0
            print(f"  Clamped {n_subzero:,} values in {os.path.basename(nc_path)} "
                  f"({pct:.2f}%)")
        else:
            print(f"  No sub-zero values in {os.path.basename(nc_path)}")

        ds.attrs["subzero_clamped"] = 1
        ds.attrs["subzero_threshold"] = threshold

        # Write to temp file, then atomic replace (avoids in-place write issues)
        tmp_path = nc_path + ".tmp"
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        if "temp_filled" in ds:
            enc["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
        ds.to_netcdf(tmp_path, encoding=enc)
        os.replace(tmp_path, nc_path)
        return True

    except Exception as e:
        print(f"  ERROR clamping {os.path.basename(nc_path)}: {e}")
        # Clean up temp file if it exists
        tmp_path = nc_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def clamp_all_post_files(post_dir: str, dry_run: bool = False) -> int:
    """Clamp sub-zero values in all post .nc files. Returns count of modified files."""
    nc_files = sorted(glob.glob(os.path.join(post_dir, "*.nc")))
    n_modified = 0
    for nc_path in nc_files:
        if clamp_subzero_file(nc_path, threshold=0.0, dry_run=dry_run):
            n_modified += 1
    return n_modified


# =============================================================================
# Step 2: DINCAE temporal interpolation
# =============================================================================

def create_dincae_interp(
    dincae_sparse_path: str,
    prepared_path: str,
    out_path: str,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Create full-daily DINCAE output via per-pixel linear interpolation.
    Returns True if file was created.
    """
    if os.path.isfile(out_path) and not force:
        print(f"  DINCAE interp already exists: {os.path.basename(out_path)}")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would create: {os.path.basename(out_path)}")
        return False

    try:
        # Read full_days from prepared.nc attrs
        with xr.open_dataset(prepared_path) as ds_prep:
            t0 = int(ds_prep.attrs.get("time_start_days"))
            t1 = int(ds_prep.attrs.get("time_end_days"))
        full_days = np.arange(t0, t1 + 1, dtype="int64")

        base = np.datetime64("1981-01-01T12:00:00", "ns")
        full_time = base + full_days.astype("timedelta64[D]")

        ds_sparse = xr.load_dataset(dincae_sparse_path)
        sparse_time = ds_sparse["time"].values.astype("datetime64[ns]")
        sparse_days = ((sparse_time.astype("int64") - base.astype("int64"))
                       // 86_400_000_000_000).astype("int64")

        sparse_x = sparse_days.astype("float64")
        full_x = full_days.astype("float64")

        temp_sparse = ds_sparse["temp_filled"].values  # (T_sparse, lat, lon)
        T_full = len(full_days)
        ny, nx = temp_sparse.shape[1], temp_sparse.shape[2]
        temp_full = np.full((T_full, ny, nx), np.nan, dtype="float32")

        n_interp = 0
        for iy in range(ny):
            for ix in range(nx):
                col = temp_sparse[:, iy, ix]
                valid = np.isfinite(col)
                if valid.sum() < 2:
                    for i_s, d in enumerate(sparse_days):
                        j = np.searchsorted(full_days, d)
                        if j < T_full and full_days[j] == d and valid[i_s]:
                            temp_full[j, iy, ix] = col[i_s]
                    continue

                x_valid = sparse_x[valid]
                y_valid = col[valid].astype("float64")

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

        print(f"  Interpolated {n_interp} pixels onto {T_full} daily timesteps")

        lat_name = "lat" if "lat" in ds_sparse.coords else "latitude"
        lon_name = "lon" if "lon" in ds_sparse.coords else "longitude"

        ds_out = xr.Dataset()
        ds_out = ds_out.assign_coords({
            "time": full_time,
            lat_name: ds_sparse[lat_name].values,
            lon_name: ds_sparse[lon_name].values,
        })
        ds_out["temp_filled"] = xr.DataArray(
            temp_full,
            dims=("time", lat_name, lon_name),
            coords={"time": full_time,
                     lat_name: ds_sparse[lat_name].values,
                     lon_name: ds_sparse[lon_name].values},
            attrs={"units": "degree_Celsius",
                    "long_name": "lake surface water temperature (DINCAE, daily interpolated)",
                    "comment": "Per-pixel linear interpolation of sparse DINCAE output to full daily timeline"},
        )

        # Copy lakeid if present
        for var in ("lakeid", "lakeid_original"):
            if var in ds_sparse:
                ds_out[var] = ds_sparse[var]

        # Copy attrs
        ds_out.attrs.update(ds_sparse.attrs)
        ds_out.attrs["source_model"] = "DINCAE"
        ds_out.attrs["interpolation_method"] = "per_pixel_linear"
        ds_out.attrs["interpolation_edge_policy"] = "leave_nan"
        ds_out.attrs["interpolation_source"] = os.path.basename(dincae_sparse_path)
        ds_out.attrs["subzero_clamped"] = 1  # input was already clamped
        ds_out.attrs["subzero_threshold"] = 0.0

        enc = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
        if "temp_filled" in ds_out:
            enc["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
        ds_out.to_netcdf(out_path, encoding=enc)
        print(f"  Wrote: {os.path.basename(out_path)}")

        return True

    except Exception as e:
        print(f"  ERROR creating DINCAE interp: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Steps 3 & 4: Re-run plots and insitu validation
# =============================================================================

def rerun_plots_insitu(
    post_dir: str,
    lake_id: int,
    lake_path: str,
    prepared_path: str,
    climatology_path: str,
    experiment_config: Optional[str] = None,
    do_plots: bool = True,
    do_insitu: bool = True,
    dry_run: bool = False,
):
    """Re-run LSWTPlotsStep and InsituValidationStep for a lake."""
    if dry_run:
        if do_plots:
            print(f"  [DRY-RUN] Would regenerate LSWT plots")
        if do_insitu:
            print(f"  [DRY-RUN] Would regenerate in-situ validation")
        return

    # We need to import the pipeline steps.
    # They may be installed as a package or available on PYTHONPATH.
    try:
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.lswt_plots import LSWTPlotsStep
        from processors.postprocessor.post_steps.insitu_validation import InsituValidationStep
    except ImportError:
        # Try adding src to path
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from processors.postprocessor.post_steps.base import PostContext
        from processors.postprocessor.post_steps.lswt_plots import LSWTPlotsStep
        from processors.postprocessor.post_steps.insitu_validation import InsituValidationStep

    # Find any existing output file to set ctx.output_path (for post_dir derivation)
    nc_files = glob.glob(os.path.join(post_dir, "*.nc"))
    if not nc_files:
        print(f"  No .nc files found in {post_dir}, skipping plots/insitu")
        return

    ctx = PostContext(
        lake_path=lake_path,
        dineof_input_path=prepared_path,
        dineof_output_path="<unused>",
        output_path=nc_files[0],  # any file in post_dir works
        output_html_folder=None,
        climatology_path=climatology_path,
        lake_id=lake_id,
        experiment_config_path=experiment_config,
    )

    if do_plots:
        try:
            LSWTPlotsStep(original_ts_path=lake_path).apply(ctx, None)
        except Exception as e:
            print(f"  ERROR in LSWTPlotsStep: {e}")

    if do_insitu:
        try:
            InsituValidationStep().apply(ctx, None)
        except Exception as e:
            print(f"  ERROR in InsituValidationStep: {e}")


# =============================================================================
# Lake discovery
# =============================================================================

def discover_lakes(run_root: str, alpha_slug: str = "a1000") -> List[int]:
    """Discover all lake IDs that have post output in a run."""
    post_base = os.path.join(run_root, "post")
    if not os.path.isdir(post_base):
        return []
    lake_ids = []
    for entry in sorted(os.listdir(post_base)):
        post_alpha = os.path.join(post_base, entry, alpha_slug)
        if os.path.isdir(post_alpha):
            try:
                lake_id = int(entry)
                lake_ids.append(lake_id)
            except ValueError:
                continue
    return lake_ids


def resolve_paths(
    run_root: str, lake_id: int, alpha_slug: str, config: dict
) -> Dict[str, str]:
    """Resolve all paths needed for a lake."""
    lake_id9 = f"{lake_id:09d}"
    post_dir = os.path.join(run_root, "post", lake_id9, alpha_slug)
    prepared_path = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
    dineof_dir = os.path.join(run_root, "dineof", lake_id9, alpha_slug)

    # Resolve lake_ts and climatology from config templates
    P = config.get("paths", {})

    lake_ts = P.get("lake_ts_template", "").replace("{lake_id9}", lake_id9).replace(
        "{lake_id}", str(lake_id))
    clim = P.get("climatology_template", "").replace("{lake_id9}", lake_id9).replace(
        "{lake_id}", str(lake_id))

    return {
        "post_dir": post_dir,
        "prepared_path": prepared_path,
        "dineof_dir": dineof_dir,
        "lake_ts": lake_ts,
        "climatology": clim,
    }


# =============================================================================
# Per-lake processing
# =============================================================================

def process_lake(
    run_root: str,
    lake_id: int,
    alpha_slug: str,
    config: dict,
    config_path: Optional[str],
    dry_run: bool = False,
    force: bool = False,
    do_plots: bool = True,
    do_insitu: bool = True,
    do_refilter_eofs: bool = False,
):
    """Apply all retrofit steps to a single lake."""
    paths = resolve_paths(run_root, lake_id, alpha_slug, config)
    post_dir = paths["post_dir"]

    if not os.path.isdir(post_dir):
        print(f"  Post dir not found: {post_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Lake {lake_id}")
    print(f"{'='*60}")

    # Step 0: Re-filter EOFs (if requested)
    eofs_modified = False
    if do_refilter_eofs:
        print("\n[Step 0] Re-filtering EOFs...")
        dineof_dir = paths["dineof_dir"]
        if os.path.isdir(dineof_dir):
            eofs_modified = refilter_and_reconstruct_eofs(
                dineof_dir=dineof_dir,
                prepared_path=paths["prepared_path"],
                lake_id=lake_id,
                config=config,
                dry_run=dry_run,
                force=force,
            )
            if eofs_modified and not dry_run:
                print("\n[Step 0+] Re-running full post-processing pipeline...")
                rerun_post_processing(
                    post_dir=post_dir,
                    dineof_dir=dineof_dir,
                    prepared_path=paths["prepared_path"],
                    lake_path=paths["lake_ts"],
                    climatology_path=paths["climatology"],
                    config_path=config_path,
                    config=config,
                    dry_run=dry_run,
                )
                # Post-processing already writes all files including clamped values,
                # so skip the separate clamp step
                print("\n[Step 1] Skipping clamp (already done by post-processing pipeline)")
                # DINCAE interp is also handled by the pipeline
                print("\n[Step 2] Skipping DINCAE interp (already done by post-processing pipeline)")
                # Plots and insitu are also handled
                if do_plots:
                    print("[Step 3] Plots already regenerated by post-processing pipeline")
                if do_insitu:
                    print("[Step 4] In-situ validation already regenerated by post-processing pipeline")
                return
        else:
            print(f"  DINEOF dir not found: {dineof_dir}")

    # Step 1: Clamp sub-zero
    print("\n[Step 1] Clamping sub-zero LSWT...")
    clamp_all_post_files(post_dir, dry_run=dry_run)

    # Step 2: DINCAE interp
    print("\n[Step 2] DINCAE temporal interpolation...")
    dincae_files = glob.glob(os.path.join(post_dir, "*_dincae.nc"))
    if dincae_files:
        dincae_sparse = dincae_files[0]
        dincae_interp = dincae_sparse.replace("_dincae.nc", "_dincae_interp_full.nc")
        prepared = paths["prepared_path"]
        if os.path.isfile(prepared):
            create_dincae_interp(
                dincae_sparse, prepared, dincae_interp,
                dry_run=dry_run, force=force,
            )
        else:
            print(f"  prepared.nc not found: {prepared}")
    else:
        print(f"  No _dincae.nc found in {post_dir}")

    # Steps 3-4: Re-run plots + insitu
    if do_plots or do_insitu:
        print("\n[Step 3-4] Re-running plots and in-situ validation...")
        rerun_plots_insitu(
            post_dir=post_dir,
            lake_id=lake_id,
            lake_path=paths["lake_ts"],
            prepared_path=paths["prepared_path"],
            climatology_path=paths["climatology"],
            experiment_config=config_path,
            do_plots=do_plots,
            do_insitu=do_insitu,
            dry_run=dry_run,
        )


# =============================================================================
# SLURM submission
# =============================================================================

def generate_slurm_script(
    run_root: str,
    lake_ids: List[int],
    alpha_slug: str,
    config_path: str,
    script_dir: str,
    extra_flags: str = "",
    mem: str = "128G",
) -> str:
    """Generate SLURM array job script for batch processing."""
    os.makedirs(script_dir, exist_ok=True)

    # Write lake list
    lake_list_path = os.path.join(script_dir, "retrofit_lake_ids.txt")
    with open(lake_list_path, "w") as f:
        for lid in lake_ids:
            f.write(f"{lid}\n")

    script_path = os.path.join(script_dir, "retrofit_post.slurm")
    this_script = os.path.abspath(__file__)

    script = f"""#!/bin/bash
#SBATCH --job-name=retrofit_post
#SBATCH --array=1-{len(lake_ids)}
#SBATCH --time=48:00:00
#SBATCH --mem={mem}
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=eocis_chuk
#SBATCH --cpus-per-task=1
#SBATCH --chdir={os.path.dirname(this_script)}
#SBATCH -o {script_dir}/logs/retrofit_%a_%j.out
#SBATCH -e {script_dir}/logs/retrofit_%a_%j.err

set -euo pipefail

# Activate conda environment
source ~/miniforge3/bin/activate && conda activate lake_cci_gapfilling

mkdir -p {script_dir}/logs

LAKE_ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {lake_list_path})
echo "[$(date)] Starting retrofit for lake $LAKE_ID (task $SLURM_ARRAY_TASK_ID, job $SLURM_ARRAY_JOB_ID)"

python {this_script} \\
  --run-root {run_root} \\
  --lake-ids $LAKE_ID \\
  --alpha {alpha_slug} \\
  --config {os.path.abspath(config_path)} {extra_flags}

echo "[$(date)] Done: lake $LAKE_ID"
"""

    with open(script_path, "w") as f:
        f.write(script)

    print(f"SLURM script: {script_path}")
    print(f"Lake list: {lake_list_path} ({len(lake_ids)} lakes)")
    print(f"\nSubmit with: sbatch {script_path}")
    return script_path


# =============================================================================
# Main
# =============================================================================

def generate_manifest_slurm_script(
    jobs: List[tuple],
    alpha_slug: str,
    config_path: str,
    script_dir: str,
    extra_flags: str = "",
    mem: str = "128G",
) -> str:
    """Generate SLURM array job for manifest-based retrofit (one task per segment×lake)."""
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(os.path.join(script_dir, "logs"), exist_ok=True)

    # Write job list: each line is "run_root lake_id"
    job_list_path = os.path.join(script_dir, "retrofit_jobs.txt")
    with open(job_list_path, "w") as f:
        for run_root, lake_id in jobs:
            f.write(f"{run_root} {lake_id}\n")

    script_path = os.path.join(script_dir, "retrofit_manifest.slurm")
    this_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "scripts", "retrofit_post.py"))
    # If running from scripts dir already, use __file__ directly
    if not os.path.isfile(this_script):
        this_script = os.path.abspath(__file__)

    script = f"""#!/bin/bash
#SBATCH --job-name=retrofit_manifest
#SBATCH --array=1-{len(jobs)}
#SBATCH --time=48:00:00
#SBATCH --mem={mem}
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=eocis_chuk
#SBATCH --cpus-per-task=1
#SBATCH --chdir={os.path.dirname(this_script)}
#SBATCH -o {script_dir}/logs/retrofit_%a_%j.out
#SBATCH -e {script_dir}/logs/retrofit_%a_%j.err

set -euo pipefail

# Activate conda environment
source ~/miniforge3/bin/activate && conda activate lake_cci_gapfilling

JOB_LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_list_path})
RUN_ROOT=$(echo $JOB_LINE | awk '{{print $1}}')
LAKE_ID=$(echo $JOB_LINE | awk '{{print $2}}')

echo "[$(date)] Starting retrofit for lake $LAKE_ID in $RUN_ROOT (task $SLURM_ARRAY_TASK_ID, job $SLURM_ARRAY_JOB_ID)"

python {this_script} \\
  --run-root $RUN_ROOT \\
  --lake-ids $LAKE_ID \\
  --alpha {alpha_slug} \\
  --config {os.path.abspath(config_path)} {extra_flags}

echo "[$(date)] Done: lake $LAKE_ID in $RUN_ROOT"
"""

    with open(script_path, "w") as f:
        f.write(script)

    print(f"SLURM script: {script_path}")
    print(f"Job list: {job_list_path} ({len(jobs)} jobs)")
    print(f"Memory: {mem}")
    print(f"\nSubmit with: sbatch {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Retrofit post-processing updates to existing pipeline outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specific lakes in a single run
  python retrofit_post.py --run-root /gws/.../run/ --lake-ids 380 88 2

  # All lakes
  python retrofit_post.py --run-root /gws/.../run/ --all-lakes --config exp0.json

  # Segment manifest mode (iterates over all segments)
  python retrofit_post.py --manifest /gws/.../segment_manifest.json --lake-ids 5 6 9 11 --config exp0.json

  # Manifest + SLURM with more memory for large lakes
  python retrofit_post.py --manifest /gws/.../segment_manifest.json --lake-ids 5 6 9 11 --config exp0.json --submit-slurm --mem 192G

  # Dry-run
  python retrofit_post.py --run-root /gws/.../run/ --all-lakes --dry-run
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-root", help="Root of a single experiment run")
    group.add_argument("--manifest", help="Path to segment_manifest.json (iterates over all segment run roots)")
    parser.add_argument("--lake-ids", type=int, nargs="+", help="Specific lake IDs to process")
    parser.add_argument("--all-lakes", action="store_true", help="Process all lakes in the run")
    parser.add_argument("--config", help="Path to experiment JSON config (for path templates)")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    parser.add_argument("--force", action="store_true", help="Overwrite existing DINCAE interp files")
    parser.add_argument("--no-plots", action="store_true", help="Skip LSWT plot regeneration")
    parser.add_argument("--no-insitu", action="store_true", help="Skip in-situ validation regeneration")
    parser.add_argument("--refilter-eofs", action="store_true",
                        help="Re-filter EOFs with interpolation-replacement (fixes dropped-timestep bug). "
                             "Also re-runs full post-processing pipeline for affected passes.")
    parser.add_argument("--submit-slurm", action="store_true", help="Generate + submit SLURM array job")
    parser.add_argument("--mem", default="128G", help="SLURM memory allocation (default: 128G, use 192G for large lakes)")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Manifest mode: expand manifest into list of (run_root, lake_ids) pairs
    # -------------------------------------------------------------------------
    if args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"ERROR: manifest not found: {args.manifest}")
            sys.exit(1)
        with open(args.manifest) as f:
            manifest = json.load(f)

        segment_roots = [seg["run_root"] for seg in manifest["segments"]]
        manifest_lake_ids = manifest.get("lake_ids", [])

        # Determine which lakes to process
        if args.lake_ids:
            target_lakes = args.lake_ids
        elif args.all_lakes:
            target_lakes = manifest_lake_ids
        else:
            target_lakes = manifest_lake_ids
            if target_lakes:
                print(f"Using all {len(target_lakes)} lakes from manifest: {target_lakes}")
            else:
                parser.error("Manifest has no lake_ids; provide --lake-ids or --all-lakes")

        # Validate segment directories exist
        for root in segment_roots:
            if not os.path.isdir(root):
                print(f"WARNING: segment run root not found: {root}")

        # Build list of (run_root, lake_id) jobs
        jobs: List[tuple] = []
        for root in segment_roots:
            for lid in target_lakes:
                jobs.append((root, lid))

        print(f"Manifest: {args.manifest}")
        print(f"  Segments: {len(segment_roots)}")
        print(f"  Lakes: {target_lakes}")
        print(f"  Total jobs: {len(jobs)}")

        # Load config
        config = {}
        if args.config:
            with open(args.config) as f:
                config = json.load(f)

        # SLURM mode for manifest
        if args.submit_slurm:
            if not args.config:
                parser.error("--submit-slurm requires --config")
            manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
            script_dir = os.path.join(manifest_dir, "retrofit")
            script_path = generate_manifest_slurm_script(
                jobs=jobs,
                alpha_slug=args.alpha,
                config_path=args.config,
                script_dir=script_dir,
                extra_flags=" --refilter-eofs" if args.refilter_eofs else "",
                mem=args.mem,
            )
            result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Submitted: {result.stdout.strip()}")
            else:
                print(f"sbatch failed: {result.stderr}")
            return

        # Interactive mode for manifest
        for i, (run_root, lake_id) in enumerate(jobs, 1):
            seg_name = os.path.basename(run_root)
            print(f"\n{'='*60}")
            print(f"[{i}/{len(jobs)}] {seg_name} / lake {lake_id}")
            print(f"{'='*60}")
            process_lake(
                run_root=run_root,
                lake_id=lake_id,
                alpha_slug=args.alpha,
                config=config,
                config_path=args.config,
                dry_run=args.dry_run,
                force=args.force,
                do_plots=not args.no_plots,
                do_insitu=not args.no_insitu,
                do_refilter_eofs=args.refilter_eofs,
            )

        print(f"\n{'='*60}")
        print(f"Manifest retrofit complete: {len(jobs)} jobs across "
              f"{len(segment_roots)} segments × {len(target_lakes)} lakes")
        print(f"{'='*60}")
        return

    # -------------------------------------------------------------------------
    # Single run-root mode (original behavior)
    # -------------------------------------------------------------------------
    if not os.path.isdir(args.run_root):
        print(f"ERROR: run-root not found: {args.run_root}")
        sys.exit(1)

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # If no config but we have manifest.json, try loading from there
    if not config:
        manifest_path = os.path.join(args.run_root, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                config = json.load(f)
            print(f"Loaded config from manifest.json")

    # Determine lake IDs
    if args.all_lakes:
        lake_ids = discover_lakes(args.run_root, args.alpha)
        if not lake_ids:
            print(f"No lakes found in {args.run_root}/post/")
            sys.exit(1)
        print(f"Discovered {len(lake_ids)} lakes")
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        parser.error("Provide --lake-ids or --all-lakes")

    # SLURM submission mode
    if args.submit_slurm:
        if not args.config:
            parser.error("--submit-slurm requires --config")
        script_dir = os.path.join(args.run_root, "retrofit")
        extra_flags = ""
        if args.refilter_eofs:
            extra_flags += " --refilter-eofs"
        script_path = generate_slurm_script(
            args.run_root, lake_ids, args.alpha, args.config, script_dir,
            extra_flags=extra_flags,
            mem=args.mem,
        )
        # Auto-submit
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted: {result.stdout.strip()}")
        else:
            print(f"sbatch failed: {result.stderr}")
        return

    # Process lakes
    print(f"Run root: {args.run_root}")
    print(f"Lakes: {len(lake_ids)}")
    print(f"Alpha: {args.alpha}")
    if args.dry_run:
        print("MODE: DRY-RUN")

    for lake_id in lake_ids:
        process_lake(
            run_root=args.run_root,
            lake_id=lake_id,
            alpha_slug=args.alpha,
            config=config,
            config_path=args.config,
            dry_run=args.dry_run,
            force=args.force,
            do_plots=not args.no_plots,
            do_insitu=not args.no_insitu,
            do_refilter_eofs=args.refilter_eofs,
        )

    print(f"\n{'='*60}")
    print(f"Retrofit complete: {len(lake_ids)} lakes processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
