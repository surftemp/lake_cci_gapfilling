#!/usr/bin/env python3
"""
Segment Merger for Temporal Chunking

Merges per-segment pipeline outputs into a single set of files that
downstream scripts (cv_validation, insitu_validation, analysis) can use.

KEY DESIGN: The ObservationAvailabilityFilterStep removes different pixels
in each time segment (different temporal coverage → different min_observation_percent
threshold outcomes). This means lakeid masks, iindex/jindex, and clouds_index
are segment-specific and CANNOT be naively concatenated.

Solution: compute CV per-segment, concatenate diff vectors, save as .npz.
Post files are on the original CCI grid (not the filtered grid) so overlay works.

Merge operations (per lake):
  1. DINEOF CV  — extract from .dat per segment → concatenate → cv_pairs_dineof.npz
  2. DINCAE CV  — compute from per-segment files → concatenate → cv_pairs_dincae.npz
  3. Post files — overlay non-NaN values from each segment

Files NOT merged (segment-specific masks make this non-trivial and unnecessary):
  - prepared.nc       (different lakeid masks per segment)
  - clouds_index.nc   (different iindex/jindex per segment)
  - dincae_results.nc (different spatial grids per segment)

Author: Shaerdan / NCEO / University of Reading
Date: February 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import xarray as xr

# Import helpers from cv_validation
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from cv_validation import read_gher_file, compute_cv_metrics
except ImportError:
    read_gher_file = None
    compute_cv_metrics = None


# =============================================================================
# DINEOF CV: extract per-segment .dat → concatenate → .npz
# =============================================================================

def merge_dineof_cv_pairs(
    seg_dineof_dirs: List[str],
    out_path: str,
    verbose: bool = True,
) -> Dict:
    """
    Extract CV diff vectors from each segment's DINEOF output,
    concatenate, and save as .npz for cv_validation.py to read.
    
    Checks per segment (in order):
    1. cv_pairs_dineof.npz  (pre-aggregated, e.g. from a prior merge)
    2. CVpoints_*.dat       (standard GHER binary from DINEOF)
    """
    if read_gher_file is None:
        raise ImportError("Could not import read_gher_file from cv_validation.py")

    all_best = []
    all_init = []
    all_diffs = []
    n_per_seg = []

    for seg_i, dineof_dir in enumerate(seg_dineof_dirs):
        # Option 1: existing .npz
        npz_path = os.path.join(dineof_dir, "cv_pairs_dineof.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            diff = data["diff"]
            valid = ~np.isnan(diff)
            diff = diff[valid]
            n_per_seg.append(len(diff))
            all_diffs.append(diff)
            if "best" in data and "init" in data:
                b, i = data["best"], data["init"]
                v = ~np.isnan(b) & ~np.isnan(i)
                all_best.append(b[v])
                all_init.append(i[v])
            if verbose:
                print(f"    seg{seg_i}: {len(diff)} DINEOF CV points (from .npz)")
            continue

        # Option 2: GHER .dat files
        cv_best_path = os.path.join(dineof_dir, "CVpoints_best_estimate.dat")
        cv_init_path = os.path.join(dineof_dir, "CVpoints_initial.dat")

        if not os.path.exists(cv_best_path) or not os.path.exists(cv_init_path):
            if verbose:
                print(f"    seg{seg_i}: no DINEOF CV data found in {dineof_dir}")
            n_per_seg.append(0)
            continue

        best = read_gher_file(cv_best_path, verbose)
        init = read_gher_file(cv_init_path, verbose)

        if best is None or init is None:
            if verbose:
                print(f"    seg{seg_i}: could not read GHER files")
            n_per_seg.append(0)
            continue

        valid = ~np.isnan(best) & ~np.isnan(init)
        b_valid = best[valid]
        i_valid = init[valid]
        n_per_seg.append(len(b_valid))
        all_best.append(b_valid)
        all_init.append(i_valid)
        all_diffs.append(b_valid - i_valid)

        if verbose:
            print(f"    seg{seg_i}: {len(b_valid)} DINEOF CV points (from .dat)")

    if not all_diffs:
        if verbose:
            print("    No DINEOF CV pairs found in any segment")
        return {"success": False, "n_per_seg": n_per_seg}

    diff_all = np.concatenate(all_diffs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_data = {"diff": diff_all, "n_per_segment": np.array(n_per_seg)}
    if all_best:
        save_data["best"] = np.concatenate(all_best)
        save_data["init"] = np.concatenate(all_init)
    np.savez(out_path, **save_data)

    if verbose and compute_cv_metrics:
        metrics = compute_cv_metrics(diff_all)
        print(f"    → cv_pairs_dineof.npz: {len(diff_all)} CV points, "
              f"RMSE={metrics['rmse']:.6f}")
    return {"success": True, "n_per_seg": n_per_seg, "total": len(diff_all)}


# =============================================================================
# DINCAE CV: compute per-segment from prepared.nc + dincae_results.nc +
#            clouds_index.nc → concatenate → .npz
# =============================================================================

def _extract_dincae_cv_from_segment(
    prepared_nc: str,
    dincae_results_nc: str,
    clouds_index_nc: str,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract DINCAE CV diff vector from a single segment.
    Returns diff array (reconstruction - original) at CV point locations, or None.
    """
    if not os.path.exists(dincae_results_nc):
        if verbose:
            print(f"      dincae_results.nc not found")
        return None

    if not os.path.exists(clouds_index_nc):
        if verbose:
            print(f"      clouds_index.nc not found")
        return None

    # Load original
    ds_orig = xr.open_dataset(prepared_nc)
    original = ds_orig["lake_surface_water_temperature"].values
    ds_orig.close()

    # Load reconstruction
    ds_recon = xr.open_dataset(dincae_results_nc)
    recon_var = "temp_filled" if "temp_filled" in ds_recon else list(ds_recon.data_vars)[0]
    reconstructed = ds_recon[recon_var].values
    ds_recon.close()

    # Load CV points
    ds_ci = xr.open_dataset(clouds_index_nc)
    clouds_index = ds_ci["clouds_index"].values  # (2, npoints)
    iindex = ds_ci["iindex"].values
    jindex = ds_ci["jindex"].values
    ds_ci.close()

    npoints = clouds_index.shape[1]

    orig_vals = []
    recon_vals = []

    for p in range(npoints):
        idx = int(clouds_index[0, p])      # spatial index (1-based)
        t = int(clouds_index[1, p]) - 1    # time (0-based)
        i = int(iindex[idx - 1]) - 1       # lon (0-based)
        j = int(jindex[idx - 1]) - 1       # lat (0-based)

        if (0 <= t < original.shape[0] and
            0 <= j < original.shape[1] and
            0 <= i < original.shape[2] and
            t < reconstructed.shape[0]):

            o = original[t, j, i]
            r = reconstructed[t, j, i]

            if not np.isnan(o) and not np.isnan(r):
                orig_vals.append(o)
                recon_vals.append(r)

    if len(orig_vals) == 0:
        return None

    return np.array(recon_vals) - np.array(orig_vals)


def merge_dincae_cv_pairs(
    seg_run_roots: List[str],
    lake_id9: str,
    alpha: str,
    out_path: str,
    verbose: bool = True,
) -> Dict:
    """
    Compute DINCAE CV per-segment, concatenate diff vectors, save as .npz.
    Each segment uses its OWN prepared.nc, clouds_index.nc, and dincae_results.nc
    (with their own consistent lakeid masks).
    """
    all_diffs = []
    n_per_seg = []

    for seg_i, run_root in enumerate(seg_run_roots):
        prepared_nc = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
        clouds_index_nc = os.path.join(run_root, "prepared", lake_id9, "clouds_index.nc")
        dincae_results_nc = os.path.join(run_root, "dincae", lake_id9, alpha, "dincae_results.nc")

        if not os.path.exists(prepared_nc):
            if verbose:
                print(f"    seg{seg_i}: prepared.nc not found, skipping")
            n_per_seg.append(0)
            continue

        if verbose:
            print(f"    seg{seg_i}: extracting DINCAE CV...")

        diff = _extract_dincae_cv_from_segment(
            prepared_nc, dincae_results_nc, clouds_index_nc, verbose
        )

        if diff is not None:
            n_per_seg.append(len(diff))
            all_diffs.append(diff)
            if verbose:
                print(f"    seg{seg_i}: {len(diff)} valid DINCAE CV points")
        else:
            n_per_seg.append(0)
            if verbose:
                print(f"    seg{seg_i}: no valid DINCAE CV points")

    if not all_diffs:
        if verbose:
            print("    No DINCAE CV pairs found in any segment")
        return {"success": False, "n_per_seg": n_per_seg}

    diff_all = np.concatenate(all_diffs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        diff=diff_all,
        n_per_segment=np.array(n_per_seg),
    )

    if verbose:
        metrics = compute_cv_metrics(diff_all)
        print(f"    → cv_pairs_dincae.npz: {len(diff_all)} CV points, "
              f"RMSE={metrics['rmse']:.6f}")
    return {"success": True, "n_per_seg": n_per_seg, "total": len(diff_all)}


# =============================================================================
# Merge: post-processed files (overlay segments)
# =============================================================================

def merge_post_files(
    seg_paths: List[str],
    out_path: str,
    verbose: bool = True,
) -> Dict:
    """
    Overlay post-processed files from each segment.

    Each segment's post file is on the SAME time axis (the original CCI lake
    file's full timeline), but only its time range has non-NaN temp_filled.
    We overlay: use seg0 values where non-NaN, else seg1, etc.
    """
    datasets = []
    for p in seg_paths:
        if not os.path.exists(p):
            if verbose:
                print(f"      WARNING: {p} not found, skipping")
            continue
        datasets.append(xr.open_dataset(p))

    if not datasets:
        if verbose:
            print("      No post files found")
        return {"success": False}

    # Start with a copy of the first dataset
    merged = datasets[0].copy(deep=True)

    # Float variables: fill NaN from merged with seg values
    overlay_vars_float = [
        v for v in merged.data_vars
        if merged[v].dtype.kind == 'f' and merged[v].ndim >= 2
    ]

    for seg_i in range(1, len(datasets)):
        ds_seg = datasets[seg_i]

        for v in overlay_vars_float:
            if v in ds_seg:
                merged_vals = merged[v].values
                seg_vals = ds_seg[v].values
                fill_mask = np.isnan(merged_vals) & ~np.isnan(seg_vals)
                merged_vals[fill_mask] = seg_vals[fill_mask]
                merged[v].values = merged_vals

        # data_source: fill 255 from merged with seg values
        if "data_source" in merged and "data_source" in ds_seg:
            merged_vals = merged["data_source"].values
            seg_vals = ds_seg["data_source"].values
            fill_mask = (merged_vals == 255) & (seg_vals != 255)
            merged_vals[fill_mask] = seg_vals[fill_mask]
            merged["data_source"].values = merged_vals

    # Update attrs
    for i, ds in enumerate(datasets):
        for key in ["detrend_slope_per_day", "detrend_intercept", "detrend_t0_days"]:
            if key in ds.attrs:
                merged.attrs[f"{key}_seg{i}"] = ds.attrs[key]
    merged.attrs["temporal_chunking"] = "merged from segments"
    merged.attrs["n_segments"] = len(datasets)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_netcdf(out_path)

    for ds in datasets:
        ds.close()

    n_filled = 0
    if "temp_filled" in merged:
        n_filled = int(np.count_nonzero(~np.isnan(merged["temp_filled"].values)))

    if verbose:
        print(f"      → merged post file: {n_filled:,} non-NaN temp_filled pixels")
    return {"success": True, "n_filled": n_filled}


# =============================================================================
# Per-lake merge orchestrator
# =============================================================================

def merge_lake(
    lake_id: int,
    seg_run_roots: List[str],
    merge_root: str,
    alpha: str = "a1000",
    verbose: bool = True,
) -> Dict:
    """
    Merge all outputs for a single lake across all segments.

    Strategy:
    - CV validation is computed PER-SEGMENT (each segment has its own
      consistent lakeid/iindex/jindex), then diff vectors are concatenated.
    - Post files are overlaid on the shared CCI time axis.
    - prepared.nc, clouds_index.nc, dincae_results.nc are NOT merged
      (different spatial masks per segment; not needed for downstream).
    """
    lake_id9 = f"{lake_id:09d}"
    summary = {"lake_id": lake_id, "status": "ok", "steps": {}}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Merging lake {lake_id} ({lake_id9})")
        print(f"{'='*60}")

    # --- 1. DINEOF CV pairs (.dat → .npz) ---
    if verbose:
        print(f"\n  [1/3] DINEOF CV (per-segment → concatenate)")
    seg_dineof_dirs = [os.path.join(r, "dineof", lake_id9, alpha) for r in seg_run_roots]
    out_dineof_npz = os.path.join(merge_root, "dineof", lake_id9, alpha, "cv_pairs_dineof.npz")
    info = merge_dineof_cv_pairs(seg_dineof_dirs, out_dineof_npz, verbose)
    summary["steps"]["dineof_cv"] = info

    # --- 2. DINCAE CV pairs (per-segment computation → .npz) ---
    if verbose:
        print(f"\n  [2/3] DINCAE CV (per-segment → concatenate)")
    out_dincae_npz = os.path.join(merge_root, "dincae", lake_id9, alpha, "cv_pairs_dincae.npz")
    info = merge_dincae_cv_pairs(seg_run_roots, lake_id9, alpha, out_dincae_npz, verbose)
    summary["steps"]["dincae_cv"] = info

    # --- 3. Post files (discover ALL .nc files, not just dineof/dincae) ---
    if verbose:
        print(f"\n  [3/4] Post files (overlay)")

    # Discover all unique .nc filenames across all segment post dirs
    all_nc_filenames = set()
    for r in seg_run_roots:
        post_dir = os.path.join(r, "post", lake_id9, alpha)
        if os.path.isdir(post_dir):
            for fname in os.listdir(post_dir):
                if fname.endswith(".nc"):
                    all_nc_filenames.add(fname)

    for nc_fname in sorted(all_nc_filenames):
        seg_post = [os.path.join(r, "post", lake_id9, alpha, nc_fname) for r in seg_run_roots]
        out_post = os.path.join(merge_root, "post", lake_id9, alpha, nc_fname)

        # Determine a short label for logging
        label = nc_fname.split("_")[-1].replace(".nc", "") if "_" in nc_fname else nc_fname
        if verbose:
            print(f"    {label}: {nc_fname}")
        info = merge_post_files(seg_post, out_post, verbose)
        summary["steps"][f"post_{label}"] = info

    # --- 4. Symlink prepared/ from first segment (needed by plot/insitu scripts) ---
    if verbose:
        print(f"\n  [4/4] Prepared directory link")

    prepared_link = os.path.join(merge_root, "prepared", lake_id9)
    if not os.path.exists(prepared_link):
        # Find prepared from first available segment
        for r in seg_run_roots:
            seg_prepared = os.path.join(r, "prepared", lake_id9)
            if os.path.isdir(seg_prepared):
                os.makedirs(os.path.dirname(prepared_link), exist_ok=True)
                os.symlink(seg_prepared, prepared_link)
                if verbose:
                    print(f"    Symlinked: prepared/{lake_id9} → {seg_prepared}")
                summary["steps"]["prepared_link"] = {"success": True, "source": seg_prepared}
                break
        else:
            if verbose:
                print(f"    WARNING: No prepared/ found in any segment")
            summary["steps"]["prepared_link"] = {"success": False}
    else:
        if verbose:
            print(f"    prepared/{lake_id9} already exists")
        summary["steps"]["prepared_link"] = {"success": True, "existing": True}

    return summary


# =============================================================================
# Verify merge
# =============================================================================

def verify_merge(
    lake_id: int,
    merge_root: str,
    alpha: str = "a1000",
    verbose: bool = True,
) -> Dict:
    """Run verification checks on merged outputs."""
    lake_id9 = f"{lake_id:09d}"
    checks = {}

    # Check .npz files exist and have data
    for method in ["dineof", "dincae"]:
        if method == "dineof":
            npz_path = os.path.join(merge_root, "dineof", lake_id9, alpha, "cv_pairs_dineof.npz")
        else:
            npz_path = os.path.join(merge_root, "dincae", lake_id9, alpha, "cv_pairs_dincae.npz")

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            diff = data["diff"]
            n_valid = int(np.sum(~np.isnan(diff)))
            checks[f"{method}_cv_npz_exists"] = True
            checks[f"{method}_cv_n_points"] = n_valid
            if compute_cv_metrics:
                m = compute_cv_metrics(diff[~np.isnan(diff)])
                checks[f"{method}_cv_rmse"] = m["rmse"]
        else:
            checks[f"{method}_cv_npz_exists"] = False

    # Check post files
    for method in ["dineof", "dincae"]:
        post_dir = os.path.join(merge_root, "post", lake_id9, alpha)
        found = False
        if os.path.isdir(post_dir):
            for fname in os.listdir(post_dir):
                if fname.endswith(f"_{method}.nc"):
                    found = True
                    ds = xr.open_dataset(os.path.join(post_dir, fname))
                    if "temp_filled" in ds:
                        tf = ds["temp_filled"].values
                        n_per_t = np.count_nonzero(~np.isnan(tf), axis=(1, 2))
                        nonzero = n_per_t > 0
                        if nonzero.any():
                            first = np.argmax(nonzero)
                            last = len(nonzero) - 1 - np.argmax(nonzero[::-1])
                            interior = nonzero[first:last + 1]
                            checks[f"post_{method}_interior_gaps"] = int(np.sum(~interior))
                    ds.close()
        checks[f"post_{method}_exists"] = found

    if verbose:
        print(f"\n  Verification for lake {lake_id}:")
        for k, v in checks.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            elif isinstance(v, bool):
                status = "✓" if v else "⚠"
                print(f"    {status} {k}: {v}")
            elif isinstance(v, int) and k.endswith("gaps"):
                status = "✓" if v == 0 else "⚠"
                print(f"    {status} {k}: {v}")
            else:
                print(f"    {k}: {v}")

    return checks


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge temporal segment outputs for large lakes",
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to segment_manifest.json")
    parser.add_argument("--lake-id", type=int, default=None,
                        help="Merge only this lake (default: all)")
    parser.add_argument("--alpha", default="a1000")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification checks after merge")
    parser.add_argument("--plots", action="store_true",
                        help="Run LSWT time series plots after merge")
    parser.add_argument("--insitu", action="store_true",
                        help="Run in-situ validation after merge")
    parser.add_argument("--config", default=None,
                        help="Experiment config JSON (needed for --plots/--insitu to find lake_ts, climatology)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    with open(args.manifest) as f:
        manifest = json.load(f)

    seg_run_roots = [s["run_root"] for s in manifest["segments"]]
    merge_root = manifest["merge_root"]
    lakes = manifest["lakes"]

    if args.lake_id:
        lakes = [args.lake_id]

    print(f"Merging {len(lakes)} lakes from {len(seg_run_roots)} segments")
    print(f"Merge root: {merge_root}")

    results = {}
    for lake_id in lakes:
        summary = merge_lake(lake_id, seg_run_roots, merge_root, args.alpha, verbose)
        results[lake_id] = summary

        if args.verify:
            checks = verify_merge(lake_id, merge_root, args.alpha, verbose)
            results[lake_id]["verification"] = checks

    # Write summary
    summary_path = os.path.join(merge_root, "merge_summary.json")
    os.makedirs(merge_root, exist_ok=True)

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

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"\nMerge summary: {summary_path}")

    ok = sum(1 for v in results.values() if v["status"] == "ok")
    fail = sum(1 for v in results.values() if v["status"] != "ok")
    print(f"\nDone: {ok} succeeded, {fail} failed")
    if fail:
        for lid, v in results.items():
            if v["status"] != "ok":
                print(f"  FAILED lake {lid}: {v.get('error', 'unknown')}")

    # Post-merge: run plots and/or insitu validation
    if (args.plots or args.insitu) and ok > 0:
        config = {}
        if args.config:
            with open(args.config) as f:
                config = json.load(f)

        # Import what we need
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_root = os.path.dirname(script_dir)
        src_dir = os.path.join(pipeline_root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        try:
            from processors.postprocessor.post_steps.base import PostContext
            if args.plots:
                from processors.postprocessor.post_steps.lswt_plots import LSWTPlotsStep
            if args.insitu:
                from processors.postprocessor.post_steps.insitu_validation import InsituValidationStep
        except ImportError as e:
            print(f"\nWARNING: Could not import pipeline steps: {e}")
            print("Run plots/insitu manually with the standalone scripts (see below)")
            args.plots = False
            args.insitu = False

        if args.plots or args.insitu:
            P = config.get("paths", {})
            for lake_id in lakes:
                if results.get(lake_id, {}).get("status") != "ok":
                    continue
                lake_id9 = f"{lake_id:09d}"
                post_dir = os.path.join(merge_root, "post", lake_id9, args.alpha)

                nc_files = [f for f in os.listdir(post_dir) if f.endswith(".nc")] if os.path.isdir(post_dir) else []
                if not nc_files:
                    continue

                # Resolve paths
                lake_ts = P.get("lake_ts_template", "").replace("{lake_id9}", lake_id9).replace("{lake_id}", str(lake_id))
                clim = P.get("climatology_template", "").replace("{lake_id9}", lake_id9).replace("{lake_id}", str(lake_id))
                prepared_path = os.path.join(merge_root, "prepared", lake_id9, "prepared.nc")

                ctx = PostContext(
                    lake_path=lake_ts,
                    dineof_input_path=prepared_path,
                    dineof_output_path="<unused>",
                    output_path=os.path.join(post_dir, nc_files[0]),
                    output_html_folder=None,
                    climatology_path=clim,
                    lake_id=lake_id,
                    experiment_config_path=args.config,
                )

                if args.plots:
                    try:
                        print(f"\n  Generating LSWT plots for lake {lake_id}...")
                        LSWTPlotsStep(original_ts_path=lake_ts).apply(ctx, None)
                    except Exception as e:
                        print(f"  ERROR in LSWTPlotsStep for lake {lake_id}: {e}")

                if args.insitu:
                    try:
                        print(f"\n  Running in-situ validation for lake {lake_id}...")
                        InsituValidationStep().apply(ctx, None)
                    except Exception as e:
                        print(f"  ERROR in InsituValidationStep for lake {lake_id}: {e}")

    # Print next steps
    print(f"\n{'='*60}")
    print(f"Merged output: {merge_root}")
    print(f"{'='*60}")
    if not (args.plots or args.insitu):
        print(f"\nNext steps — generate plots and validation:")
        print(f"  python scripts/run_lswt_plots.py --run-root {merge_root} --all")
        print(f"  python scripts/run_insitu_validation.py --run-root {merge_root} --all")
        print(f"\nOr re-run merge with --plots --insitu --config configs/your_config.json")
    print(f"\nTo patch into main experiment:")
    print(f"  cp -r {merge_root}/post/* /path/to/main_experiment/post/")


if __name__ == "__main__":
    main()
