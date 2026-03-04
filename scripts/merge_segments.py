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
    time_chunk: int = 365,
) -> Dict:
    """
    Overlay post-processed files from each segment using chunked I/O.

    Each segment's post file is on the SAME time axis (the original CCI lake
    file's full timeline), but only its time range has non-NaN temp_filled.
    We overlay: use seg0 values where non-NaN, else seg1, etc.

    Memory-bounded: processes `time_chunk` timesteps at a time instead of
    loading all segments fully into RAM.
    """
    import netCDF4

    existing_paths = []
    for p in seg_paths:
        if not os.path.exists(p):
            if verbose:
                print(f"      WARNING: {p} not found, skipping")
            continue
        existing_paths.append(p)

    if not existing_paths:
        if verbose:
            print("      No post files found")
        return {"success": False}

    # Open first dataset to get structure (coords, dims, dtypes, attrs)
    ds0 = xr.open_dataset(existing_paths[0])
    time_name = "time" if "time" in ds0.dims else list(ds0.dims)[0]
    n_time = ds0.sizes[time_name]

    # Identify overlay variables
    overlay_vars_float = [
        v for v in ds0.data_vars
        if ds0[v].dtype.kind == 'f' and ds0[v].ndim >= 2
    ]
    has_data_source = "data_source" in ds0

    # Collect segment attrs for provenance
    seg_attrs = {}
    for i, p in enumerate(existing_paths):
        with xr.open_dataset(p) as ds_tmp:
            for key in ["detrend_slope_per_day", "detrend_intercept", "detrend_t0_days",
                         "trend_params", "trend_model", "trend_added_back"]:
                if key in ds_tmp.attrs:
                    seg_attrs[f"{key}_seg{i}"] = ds_tmp.attrs[key]

    # Prepare output file: write seg0 as base
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds0.data_vars}
    if "temp_filled" in ds0:
        enc["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}

    ds0_full = xr.open_dataset(existing_paths[0])
    ds0_full.attrs.update(seg_attrs)
    ds0_full.attrs["temporal_chunking"] = "merged from segments"
    ds0_full.attrs["n_segments"] = len(existing_paths)
    ds0_full.to_netcdf(out_path, encoding=enc)
    ds0_full.close()
    ds0.close()

    # Overlay remaining segments in time chunks using netCDF4
    if len(existing_paths) > 1:
        nc_out = netCDF4.Dataset(out_path, "r+")
        nc_out.set_auto_mask(False)  # CRITICAL: return raw numpy with NaN, not masked arrays

        for seg_i in range(1, len(existing_paths)):
            n_filled_seg = 0
            ds_seg = xr.open_dataset(existing_paths[seg_i])

            for t_start in range(0, n_time, time_chunk):
                t_end = min(t_start + time_chunk, n_time)
                t_slice = slice(t_start, t_end)

                for v in overlay_vars_float:
                    if v not in ds_seg:
                        continue
                    merged_chunk = nc_out.variables[v][t_slice]
                    seg_chunk = ds_seg[v].values[t_slice]

                    seg_valid = ~np.isnan(seg_chunk)
                    if not seg_valid.any():
                        continue

                    fill_mask = np.isnan(merged_chunk) & seg_valid
                    n_fill = int(fill_mask.sum())
                    if n_fill > 0:
                        merged_chunk[fill_mask] = seg_chunk[fill_mask]
                        nc_out.variables[v][t_slice] = merged_chunk
                        n_filled_seg += n_fill

                if has_data_source and "data_source" in ds_seg:
                    merged_chunk = nc_out.variables["data_source"][t_slice]
                    seg_chunk = ds_seg["data_source"].values[t_slice]
                    fill_mask = (merged_chunk == 255) & (seg_chunk != 255)
                    if fill_mask.any():
                        merged_chunk[fill_mask] = seg_chunk[fill_mask]
                        nc_out.variables["data_source"][t_slice] = merged_chunk

            ds_seg.close()
            if verbose:
                print(f"      overlaid seg{seg_i} ({n_filled_seg:,} pixels filled)")

        nc_out.close()

    # Count filled pixels for summary
    n_filled = 0
    with xr.open_dataset(out_path) as ds_check:
        if "temp_filled" in ds_check:
            for t_start in range(0, n_time, time_chunk):
                t_end = min(t_start + time_chunk, n_time)
                chunk = ds_check["temp_filled"].isel({time_name: slice(t_start, t_end)}).values
                n_filled += int(np.count_nonzero(~np.isnan(chunk)))

    if verbose:
        print(f"      -> merged post file: {n_filled:,} non-NaN temp_filled pixels")
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
        print(f"\n  [1/4] DINEOF CV (per-segment → concatenate)")
    seg_dineof_dirs = [os.path.join(r, "dineof", lake_id9, alpha) for r in seg_run_roots]
    out_dineof_npz = os.path.join(merge_root, "dineof", lake_id9, alpha, "cv_pairs_dineof.npz")
    info = merge_dineof_cv_pairs(seg_dineof_dirs, out_dineof_npz, verbose)
    summary["steps"]["dineof_cv"] = info

    # --- 2. DINCAE CV pairs (per-segment computation → .npz) ---
    if verbose:
        print(f"\n  [2/4] DINCAE CV (per-segment → concatenate)")
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
        for r in seg_run_roots:
            seg_prepared = os.path.join(r, "prepared", lake_id9)
            if os.path.isdir(seg_prepared):
                os.makedirs(os.path.dirname(prepared_link), exist_ok=True)
                os.symlink(seg_prepared, prepared_link)
                if verbose:
                    print(f"    Symlinked: prepared/{lake_id9} -> {seg_prepared}")
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

    # Check post files — discover all .nc, verify each has data in all quarters
    post_dir = os.path.join(merge_root, "post", lake_id9, alpha)
    if os.path.isdir(post_dir):
        nc_files = sorted([f for f in os.listdir(post_dir) if f.endswith(".nc")])
        checks["post_file_count"] = len(nc_files)

        for fname in nc_files:
            label = fname.split("_")[-1].replace(".nc", "") if "_" in fname else fname
            fpath = os.path.join(post_dir, fname)
            ds = xr.open_dataset(fpath)
            if "temp_filled" in ds:
                n_time = ds.sizes.get("time", 0)
                q = n_time // 4 if n_time >= 4 else n_time
                quarter_counts = []
                for qi in range(4):
                    chunk = ds["temp_filled"].isel(time=slice(qi * q, (qi + 1) * q))
                    quarter_counts.append(int(chunk.count().values))
                all_quarters_ok = all(c > 0 for c in quarter_counts)
                checks[f"post_{label}_all_quarters"] = all_quarters_ok
                if not all_quarters_ok:
                    checks[f"post_{label}_quarter_counts"] = quarter_counts
            ds.close()
    else:
        checks["post_file_count"] = 0

    if verbose:
        print(f"\n  Verification for lake {lake_id}:")
        for k, v in checks.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            elif isinstance(v, bool):
                status = "✓" if v else "✗"
                print(f"    {status} {k}: {v}")
            elif isinstance(v, int):
                print(f"    {k}: {v}")
            elif isinstance(v, list):
                print(f"    {k}: {v}")
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


if __name__ == "__main__":
    main()
