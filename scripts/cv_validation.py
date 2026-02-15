#!/usr/bin/env python3
"""
Cross-Validation Error Computation

DINEOF: Reads from GHER binary .dat files (CVpoints_best_estimate.dat, CVpoints_initial.dat)
        OR from cv_pairs_dineof.npz (produced by temporal chunking merge).
        These contain the TRUE CV comparison from the model trained WITHOUT CV points.
        
DINCAE: Computes from dincae_results.nc vs prepared.nc at CV point locations.

Output: CSV file with comprehensive metrics for each lake/method:
        RMSE, MAE, Bias, Median, STD, RSTD, IQR, N_points
        Plus optional error histograms as .npz files.

NOTE: DINEOF's dineof_results.nc is trained WITH CV points (after model selection),
      so we cannot use it for CV validation. We must use the .dat files.

Author: Shaerdan / NCEO / University of Reading
Date: December 2024 (expanded metrics: February 2026)
"""

import argparse
import os
import sys
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np
import xarray as xr
from scipy.io import FortranFile

# Import completion check utilities for fair comparison filtering
try:
    from completion_check import (
        get_fair_comparison_lakes,
        save_exclusion_log,
        CompletionSummary
    )
    HAS_COMPLETION_CHECK = True
except ImportError:
    HAS_COMPLETION_CHECK = False


# =============================================================================
# Metric Definitions
# =============================================================================

METRIC_NAMES = ['rmse', 'mae', 'bias', 'median', 'std', 'rstd', 'iqr', 'n_points']


def compute_cv_metrics(diff: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive CV metrics from a signed error (diff) vector.

    Canonical definitions (consistent with insitu_validation.py:compute_stats):
        diff = reconstruction - original   (signed)
        RMSE    = sqrt(mean(diff²))
        MAE     = mean(|diff|)
        Bias    = mean(diff)               # positive = reconstruction warmer
        Median  = median(diff)
        STD     = std(diff, ddof=1)
        RSTD    = IQR / 1.349             # robust STD via IQR
        IQR     = Q75 - Q25
        N       = len(diff)
    """
    n = len(diff)
    if n == 0:
        return {
            'rmse': float('nan'), 'mae': float('nan'), 'bias': float('nan'),
            'median': float('nan'), 'std': float('nan'), 'rstd': float('nan'),
            'iqr': float('nan'), 'n_points': 0,
        }

    q75, q25 = np.percentile(diff, [75, 25])
    iqr = float(q75 - q25)

    return {
        'rmse':     float(np.sqrt(np.mean(diff ** 2))),
        'mae':      float(np.mean(np.abs(diff))),
        'bias':     float(np.mean(diff)),
        'median':   float(np.median(diff)),
        'std':      float(np.std(diff, ddof=1)) if n > 1 else float('nan'),
        'rstd':     float(iqr / 1.349) if n > 1 else float('nan'),
        'iqr':      iqr,
        'n_points': n,
    }


def save_error_histogram(diff: np.ndarray, output_path: str, n_bins: int = 100):
    """Save error distribution histogram + raw diff vector as .npz."""
    counts, bin_edges = np.histogram(diff, bins=n_bins)
    metrics = compute_cv_metrics(diff)
    np.savez(
        output_path,
        counts=counts,
        bin_edges=bin_edges,
        diff=diff,
        **{f"metric_{k}": v for k, v in metrics.items()},
    )


@dataclass
class CVResult:
    """CV validation result with comprehensive metrics."""
    lake_id: int
    alpha: str
    method: str
    rmse: float
    mae: float
    bias: float
    median: float
    std: float
    rstd: float
    iqr: float
    n_points: int
    diff_vector: Optional[np.ndarray] = field(default=None, repr=False)


# =============================================================================
# GHER Binary Format Reader (for DINEOF CVpoints_*.dat files)
# =============================================================================

def read_gher_file(filepath: str, verbose: bool = False) -> Optional[np.ndarray]:
    """
    Read DINEOF GHER format binary file.
    
    GHER Format (Fortran unformatted):
    1. 10 blank records (padding)
    2. Header: imax, jmax, kmax, iprec, nbmots, valex
    3. Data records
    """
    if not os.path.exists(filepath):
        return None
        
    if verbose:
        print(f"    Reading GHER: {os.path.basename(filepath)}")
    
    # Try big-endian first (typical for DINEOF), then little-endian
    for endian in ['>', '<']:
        try:
            f = FortranFile(filepath, 'r', header_dtype=np.dtype(f'{endian}u4'))
            
            # Skip 10 blank records
            for _ in range(10):
                try:
                    f.read_record(dtype=np.uint8)
                except:
                    pass
            
            # Read header record (24 bytes: 5 int32 + 1 float32)
            header_raw = f.read_record(dtype=np.uint8)
            
            if len(header_raw) >= 24:
                imax = np.frombuffer(header_raw[0:4], dtype=f'{endian}i4')[0]
                jmax = np.frombuffer(header_raw[4:8], dtype=f'{endian}i4')[0]
                kmax = np.frombuffer(header_raw[8:12], dtype=f'{endian}i4')[0]
                iprec = np.frombuffer(header_raw[12:16], dtype=f'{endian}i4')[0]
                nbmots = np.frombuffer(header_raw[16:20], dtype=f'{endian}i4')[0]
                valex = np.frombuffer(header_raw[20:24], dtype=f'{endian}f4')[0]
                
                # Sanity check
                if not (1 <= abs(imax) <= 1e7 and 1 <= abs(jmax) <= 1e7):
                    f.close()
                    continue
                
                # Data type
                dtype = f'{endian}f4' if iprec == 4 else f'{endian}f8'
                
                # Read data records
                total_elements = abs(imax) * abs(jmax) * abs(kmax)
                data_list = []
                elements_read = 0
                
                while elements_read < total_elements:
                    try:
                        rec = f.read_record(dtype=dtype)
                        data_list.append(rec)
                        elements_read += len(rec)
                    except:
                        break
                
                f.close()
                
                if data_list:
                    data = np.concatenate(data_list)
                    # Replace valex with NaN
                    data = np.where(np.abs(data - valex) < 1e-6, np.nan, data)
                    return data
            
            f.close()
        except Exception:
            continue
    
    return None


def compute_dineof_cv(dineof_dir: str, verbose: bool = True) -> Optional[CVResult]:
    """
    Compute DINEOF CV error.

    Checks (in order):
    1. cv_pairs_dineof.npz  — produced by temporal chunking merge
    2. CVpoints_*.dat       — standard GHER binary from DINEOF
    """
    # --- Option A: .npz from temporal chunking merge ---
    npz_path = os.path.join(dineof_dir, "cv_pairs_dineof.npz")
    if os.path.exists(npz_path):
        if verbose:
            print(f"    Reading merged CV pairs from {os.path.basename(npz_path)}")
        data = np.load(npz_path, allow_pickle=True)
        diff = data["diff"]
        valid = ~np.isnan(diff)
        diff = diff[valid]
        if len(diff) == 0:
            if verbose:
                print("    No valid CV points in .npz")
            return None
        metrics = compute_cv_metrics(diff)
        if verbose:
            print(f"    DINEOF CV (npz): RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
                  f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")
        return CVResult(
            lake_id=0, alpha="", method="dineof",
            diff_vector=diff, **metrics,
        )

    # --- Standard: GHER .dat files ---
    cv_best_path = os.path.join(dineof_dir, "CVpoints_best_estimate.dat")
    cv_init_path = os.path.join(dineof_dir, "CVpoints_initial.dat")
    
    if not os.path.exists(cv_best_path) or not os.path.exists(cv_init_path):
        if verbose:
            print(f"    CVpoints_*.dat files not found in {dineof_dir}")
        return None
    
    best = read_gher_file(cv_best_path, verbose)
    init = read_gher_file(cv_init_path, verbose)
    
    if best is None or init is None:
        if verbose:
            print(f"    Could not read GHER files")
        return None
    
    # Filter valid points
    valid = ~np.isnan(best) & ~np.isnan(init)
    best_valid = best[valid]
    init_valid = init[valid]
    
    if len(best_valid) == 0:
        if verbose:
            print(f"    No valid CV points")
        return None
    
    # diff = reconstruction - original
    diff = best_valid - init_valid
    metrics = compute_cv_metrics(diff)

    if verbose:
        print(f"    DINEOF CV: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
              f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")

    return CVResult(
        lake_id=0, alpha="", method="dineof",
        diff_vector=diff, **metrics,
    )


# =============================================================================
# DINCAE CV Validation (from NetCDF files)
# =============================================================================

def load_cv_points(clouds_index_nc: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CV point indices from clouds_index.nc."""
    ds = xr.open_dataset(clouds_index_nc)
    clouds_index = ds["clouds_index"].values  # shape (2, npoints)
    iindex = ds["iindex"].values
    jindex = ds["jindex"].values
    ds.close()
    return clouds_index, iindex, jindex


def compute_dincae_cv(
    prepared_nc: str,
    dincae_results_nc: str,
    clouds_index_nc: str,
    verbose: bool = True
) -> Optional[CVResult]:
    """
    Compute DINCAE CV error via direct comparison.
    DINCAE does NOT retrain after CV, so dincae_results.nc is the CV-validated model.
    """
    if not os.path.exists(dincae_results_nc):
        if verbose:
            print(f"    dincae_results.nc not found")
        return None
    
    # Load original data
    ds_orig = xr.open_dataset(prepared_nc)
    original = ds_orig["lake_surface_water_temperature"].values
    ds_orig.close()
    
    # Load reconstruction
    ds_recon = xr.open_dataset(dincae_results_nc)
    recon_var = "temp_filled" if "temp_filled" in ds_recon else list(ds_recon.data_vars)[0]
    reconstructed = ds_recon[recon_var].values
    ds_recon.close()
    
    # Load CV points
    clouds_index, iindex, jindex = load_cv_points(clouds_index_nc)
    npoints = clouds_index.shape[1]
    
    if verbose:
        print(f"    Loading {npoints} CV points")
    
    # Extract values at CV locations using correct index mapping
    orig_vals = []
    recon_vals = []
    
    for p in range(npoints):
        idx = int(clouds_index[0, p])  # sea point index (1-based)
        t = int(clouds_index[1, p]) - 1  # time (0-based)
        i = int(iindex[idx - 1]) - 1  # lon (0-based)
        j = int(jindex[idx - 1]) - 1  # lat (0-based)
        
        # data[t, j, i] where dims are (time, lat, lon)
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
        if verbose:
            print(f"    No valid CV points extracted")
        return None
    
    orig_vals = np.array(orig_vals)
    recon_vals = np.array(recon_vals)
    
    # diff = reconstruction - original
    diff = recon_vals - orig_vals
    metrics = compute_cv_metrics(diff)
    
    if verbose:
        print(f"    DINCAE CV: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
              f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")

    return CVResult(
        lake_id=0, alpha="", method="dincae",
        diff_vector=diff, **metrics,
    )


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_lake(
    run_root: str,
    lake_id: int,
    alpha: str = "a1000",
    verbose: bool = True,
    histogram_dir: Optional[str] = None,
) -> List[CVResult]:
    """Validate both DINEOF and DINCAE for a single lake.
    
    Checks for pre-computed .npz files first (from temporal chunking merge),
    then falls back to standard computation from NetCDF files.
    """
    
    lake_id9 = f"{lake_id:09d}"
    results = []
    
    prepared_nc = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
    clouds_index_nc = os.path.join(run_root, "prepared", lake_id9, "clouds_index.nc")
    dineof_dir = os.path.join(run_root, "dineof", lake_id9, alpha)
    dincae_dir = os.path.join(run_root, "dincae", lake_id9, alpha)
    
    if verbose:
        print(f"  [Lake {lake_id}]")
    
    # ---- DINEOF CV (from .npz or GHER .dat files) ----
    if os.path.exists(dineof_dir):
        dineof_result = compute_dineof_cv(dineof_dir, verbose)
        if dineof_result:
            dineof_result.lake_id = lake_id
            dineof_result.alpha = alpha
            results.append(dineof_result)
            if histogram_dir and dineof_result.diff_vector is not None:
                os.makedirs(histogram_dir, exist_ok=True)
                save_error_histogram(
                    dineof_result.diff_vector,
                    os.path.join(histogram_dir, f"{lake_id9}_dineof.npz"),
                )
    elif verbose:
        print(f"    DINEOF dir not found")
    
    # ---- DINCAE CV (from .npz or NetCDF files) ----
    # Check for pre-computed .npz first (from temporal chunking merge)
    dincae_npz_path = os.path.join(dincae_dir, "cv_pairs_dincae.npz")
    if os.path.exists(dincae_npz_path):
        if verbose:
            print(f"    Reading merged DINCAE CV pairs from cv_pairs_dincae.npz")
        data = np.load(dincae_npz_path, allow_pickle=True)
        diff = data["diff"]
        valid = ~np.isnan(diff)
        diff = diff[valid]
        if len(diff) > 0:
            metrics = compute_cv_metrics(diff)
            if verbose:
                print(f"    DINCAE CV (npz): RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
                      f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")
            dincae_result = CVResult(
                lake_id=lake_id, alpha=alpha, method="dincae",
                diff_vector=diff, **metrics,
            )
            results.append(dincae_result)
            if histogram_dir:
                os.makedirs(histogram_dir, exist_ok=True)
                save_error_histogram(
                    diff, os.path.join(histogram_dir, f"{lake_id9}_dincae.npz"),
                )
        elif verbose:
            print(f"    No valid CV points in DINCAE .npz")
    else:
        # Standard: compute from NetCDF files (requires prepared.nc + clouds_index.nc)
        dincae_results_nc = os.path.join(dincae_dir, "dincae_results.nc")
        if os.path.exists(dincae_results_nc):
            if not os.path.exists(prepared_nc) or not os.path.exists(clouds_index_nc):
                if verbose:
                    print(f"    DINCAE: prepared.nc or clouds_index.nc not found, skipping")
            else:
                dincae_result = compute_dincae_cv(
                    prepared_nc, dincae_results_nc, clouds_index_nc, verbose
                )
                if dincae_result:
                    dincae_result.lake_id = lake_id
                    dincae_result.alpha = alpha
                    results.append(dincae_result)
                    if histogram_dir and dincae_result.diff_vector is not None:
                        os.makedirs(histogram_dir, exist_ok=True)
                        save_error_histogram(
                            dincae_result.diff_vector,
                            os.path.join(histogram_dir, f"{lake_id9}_dincae.npz"),
                        )
        elif verbose:
            print(f"    DINCAE results not found")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CV Validation - DINEOF from .dat files, DINCAE from NetCDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: DINEOF CV uses CVpoints_*.dat files (or cv_pairs_dineof.npz for temporally
      chunked lakes) because dineof_results.nc is trained WITH CV points after
      model selection. DINCAE does not retrain after CV, so we use dincae_results.nc.

FAIR COMPARISON MODE (default when auto-discovering lakes):
  Only includes lakes where BOTH DINEOF and DINCAE completed processing.
  Use --no-fair-comparison to include all available data.
        """
    )
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--lake-id", type=int, help="Single lake ID")
    parser.add_argument("--lake-ids", type=int, nargs="+", help="Multiple lake IDs")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--output", "-o", default=None, help="Output CSV file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Disable fair comparison filtering")
    parser.add_argument("--histograms", default=None,
                        help="Directory to save error histogram .npz files (default: {run_root}/cv_histograms)")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Determine lake IDs with fair comparison filtering
    lake_ids = []
    completion_summary = None
    
    if args.lake_id:
        lake_ids = [args.lake_id]
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        # Auto-discover - apply fair comparison filter by default
        if HAS_COMPLETION_CHECK and not args.no_fair_comparison:
            print("=" * 70)
            print("FAIR COMPARISON MODE: Getting lakes with both methods complete")
            print("=" * 70)
            
            lake_ids, completion_summary = get_fair_comparison_lakes(
                args.run_root, args.alpha, verbose=True
            )
            
            if not lake_ids:
                print("ERROR: No lakes found with both DINEOF and DINCAE complete!")
                print("Use --no-fair-comparison to include partial results")
                sys.exit(1)
        else:
            # Fallback: discover from prepared directory
            prepared_dir = os.path.join(args.run_root, "prepared")
            if os.path.exists(prepared_dir):
                for d in os.listdir(prepared_dir):
                    if d.isdigit():
                        lake_ids.append(int(d))
            lake_ids.sort()
    
    if not lake_ids:
        print("No lake IDs specified or found")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(args.run_root, f"cv_results_{timestamp}.csv")
    
    # Histogram directory
    histogram_dir = args.histograms
    if histogram_dir is None:
        histogram_dir = os.path.join(args.run_root, "cv_histograms")

    print("=" * 70)
    print("CV Validation (comprehensive metrics)")
    print("  DINEOF: from CVpoints_*.dat or cv_pairs_dineof.npz (GHER/merged)")
    print("  DINCAE: from dincae_results.nc vs prepared.nc")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Lakes: {len(lake_ids)}")
    print(f"Alpha: {args.alpha}")
    print(f"Output: {args.output}")
    print(f"Histograms: {histogram_dir}")
    print(f"Fair comparison: {'DISABLED' if args.no_fair_comparison else 'ENABLED'}")
    print("=" * 70)
    
    all_results = []
    
    for lake_id in lake_ids:
        results = validate_lake(
            args.run_root, lake_id, args.alpha, verbose,
            histogram_dir=histogram_dir,
        )
        all_results.extend(results)
    
    # Write CSV
    if all_results:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, 'w', newline='') as f:
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
        print(f"\nResults written to: {args.output}")
        
        # Save exclusion log if we have completion summary
        if completion_summary is not None:
            log_dir = os.path.dirname(args.output) or "."
            log_path = save_exclusion_log(completion_summary, log_dir, 
                                          filename="cv_excluded_lakes_log.csv")
            print(f"Exclusion log saved: {log_path}")
    
    # Summary table
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    hdr = f"{'Lake':<10} {'Method':<10}"
    for m in METRIC_NAMES:
        hdr += f" {m:<12}"
    print(hdr)
    print("-" * 120)
    
    def _fmt(v):
        if isinstance(v, float) and not np.isnan(v):
            return f"{v:<12.6f}"
        elif isinstance(v, int):
            return f"{v:<12}"
        return f"{'':12}"

    for r in all_results:
        line = f"{r.lake_id:<10} {r.method:<10}"
        for m in METRIC_NAMES:
            line += f" {_fmt(getattr(r, m))}"
        print(line)
    
    print("=" * 120)
    
    # Count summary
    dineof_count = sum(1 for r in all_results if r.method == "dineof")
    dincae_count = sum(1 for r in all_results if r.method == "dincae")
    print(f"\nTotal: {len(all_results)} results ({dineof_count} DINEOF, {dincae_count} DINCAE)")


if __name__ == "__main__":
    main()
