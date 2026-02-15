#!/usr/bin/env python3
"""
Cross-Validation Validation - UPDATED VERSION

CRITICAL FINDING:
DINEOF's dineof_results.nc is produced by a model that was RETRAINED with CV points
included. The reported CV error in the log file is from a DIFFERENT model (trained
without CV points). Therefore, we CANNOT use dineof_results.nc for CV validation.

Solution:
- DINEOF: Read from GHER binary .dat files (CVpoints_best_estimate.dat, CVpoints_initial.dat)
          These contain the TRUE CV comparison from the model trained WITHOUT CV points.
- DINCAE: Compute from dincae_results.nc vs prepared.nc (DINCAE does NOT retrain after CV)

Output CSV format: One row per lake with both methods as columns

Author: Shaerdan / NCEO / University of Reading
Date: December 2024
"""

import argparse
import os
import sys
import csv
import glob
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

try:
    import xarray as xr
except ImportError:
    print("Error: xarray is required. Install with: pip install xarray")
    sys.exit(1)

try:
    from scipy.io import FortranFile
except ImportError:
    print("Error: scipy is required for GHER format. Install with: pip install scipy")
    sys.exit(1)

# Import completion check utilities for fair comparison filtering
try:
    from completion_check import (
        get_fair_comparison_lakes,
        generate_unique_output_dir,
        save_exclusion_log,
        CompletionSummary
    )
    HAS_COMPLETION_CHECK = True
except ImportError:
    HAS_COMPLETION_CHECK = False


@dataclass
class MethodResult:
    """Results for a single method (comprehensive metrics)."""
    rmse: Optional[float] = None
    mae: Optional[float] = None
    bias: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    rstd: Optional[float] = None
    iqr: Optional[float] = None
    n_points: Optional[int] = None
    verified: Optional[bool] = None
    reported_rmse: Optional[float] = None

# Ordered metric names for CSV/display
METRIC_NAMES = ['rmse', 'mae', 'bias', 'median', 'std', 'rstd', 'iqr', 'n_points']


def _compute_full_metrics(diff: np.ndarray) -> dict:
    """Compute all CV metrics from a signed diff vector.
    Consistent with insitu_validation.py:compute_stats definitions."""
    n = len(diff)
    if n == 0:
        return {m: None for m in METRIC_NAMES}
    q75, q25 = np.percentile(diff, [75, 25])
    iqr_val = float(q75 - q25)
    return {
        'rmse':     float(np.sqrt(np.mean(diff**2))),
        'mae':      float(np.mean(np.abs(diff))),
        'bias':     float(np.mean(diff)),
        'median':   float(np.median(diff)),
        'std':      float(np.std(diff, ddof=1)) if n > 1 else None,
        'rstd':     float(iqr_val / 1.349) if n > 1 else None,
        'iqr':      iqr_val,
        'n_points': n,
    }


@dataclass
class LakeCVResult:
    """CV results for a lake (both methods)."""
    lake_id: int
    alpha: str
    dineof: MethodResult = field(default_factory=MethodResult)
    dincae: MethodResult = field(default_factory=MethodResult)


def read_gher_file(filepath: str, verbose: bool = False) -> Optional[np.ndarray]:
    """Read DINEOF GHER format binary file."""
    if not os.path.exists(filepath):
        return None
        
    if verbose:
        print(f"      Reading GHER: {os.path.basename(filepath)}")
    
    for endian in ['>', '<']:
        try:
            f = FortranFile(filepath, 'r', header_dtype=np.dtype(f'{endian}u4'))
            for _ in range(10):
                try:
                    f.read_record(dtype=np.uint8)
                except:
                    pass
            
            header_raw = f.read_record(dtype=np.uint8)
            
            if len(header_raw) >= 24:
                imax = np.frombuffer(header_raw[0:4], dtype=f'{endian}i4')[0]
                jmax = np.frombuffer(header_raw[4:8], dtype=f'{endian}i4')[0]
                kmax = np.frombuffer(header_raw[8:12], dtype=f'{endian}i4')[0]
                iprec = np.frombuffer(header_raw[12:16], dtype=f'{endian}i4')[0]
                valex = np.frombuffer(header_raw[20:24], dtype=f'{endian}f4')[0]
                
                if not (1 <= abs(imax) <= 1e7 and 1 <= abs(jmax) <= 1e7):
                    f.close()
                    continue
                
                dtype = f'{endian}f4' if iprec == 4 else f'{endian}f8'
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
                    data = np.where(np.abs(data - valex) < 1e-6, np.nan, data)
                    return data
            f.close()
        except Exception:
            continue
    return None


def compute_dineof_cv_from_gher(dineof_dir: Path, verbose: bool = True) -> Optional[MethodResult]:
    """Compute DINEOF CV error from CVpoints_*.dat files or cv_pairs_dineof.npz."""
    
    # Check for merged .npz first (from temporal chunking)
    npz_path = dineof_dir / "cv_pairs_dineof.npz"
    if npz_path.exists():
        if verbose:
            print(f"      Reading merged CV pairs from cv_pairs_dineof.npz")
        data = np.load(str(npz_path), allow_pickle=True)
        diff = data["diff"]
        valid = ~np.isnan(diff)
        diff = diff[valid]
        if len(diff) == 0:
            if verbose:
                print(f"      No valid CV points in .npz")
            return None
        metrics = _compute_full_metrics(diff)
        if verbose:
            print(f"      RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
                  f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")
        return MethodResult(**metrics)

    # Standard: GHER .dat files
    cv_best_path = dineof_dir / "CVpoints_best_estimate.dat"
    cv_init_path = dineof_dir / "CVpoints_initial.dat"
    
    if not cv_best_path.exists() or not cv_init_path.exists():
        if verbose:
            print(f"      CVpoints_*.dat files not found")
        return None
    
    best = read_gher_file(str(cv_best_path), verbose)
    init = read_gher_file(str(cv_init_path), verbose)
    
    if best is None or init is None:
        if verbose:
            print(f"      Could not read GHER files")
        return None
    
    valid = ~np.isnan(best) & ~np.isnan(init)
    best_valid = best[valid]
    init_valid = init[valid]
    
    if len(best_valid) == 0:
        if verbose:
            print(f"      No valid CV points")
        return None
    
    diff = best_valid - init_valid
    metrics = _compute_full_metrics(diff)
    
    if verbose:
        print(f"      RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
              f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")
    
    return MethodResult(**metrics)


def load_cv_points(clouds_index_nc: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CV point indices from clouds_index.nc."""
    ds = xr.open_dataset(clouds_index_nc)
    clouds_index = ds["clouds_index"].values
    iindex = ds["iindex"].values
    jindex = ds["jindex"].values
    ds.close()
    return clouds_index, iindex, jindex


def compute_dincae_cv(prepared_nc: Path, dincae_results_nc: Path, 
                      clouds_index_nc: Path, verbose: bool = True) -> Optional[MethodResult]:
    """Compute DINCAE CV error via direct comparison."""
    if not dincae_results_nc.exists():
        if verbose:
            print(f"      dincae_results.nc not found")
        return None
    
    ds_orig = xr.open_dataset(prepared_nc)
    original = ds_orig["lake_surface_water_temperature"].values
    ds_orig.close()
    
    ds_recon = xr.open_dataset(dincae_results_nc)
    recon_var = "temp_filled" if "temp_filled" in ds_recon else list(ds_recon.data_vars)[0]
    reconstructed = ds_recon[recon_var].values
    ds_recon.close()
    
    clouds_index, iindex, jindex = load_cv_points(clouds_index_nc)
    npoints = clouds_index.shape[1]
    
    if verbose:
        print(f"      Loading {npoints} CV points")
    
    orig_vals = []
    recon_vals = []
    
    for p in range(npoints):
        idx = int(clouds_index[0, p])
        t = int(clouds_index[1, p]) - 1
        i = int(iindex[idx - 1]) - 1
        j = int(jindex[idx - 1]) - 1
        
        if (0 <= t < original.shape[0] and 0 <= j < original.shape[1] and 
            0 <= i < original.shape[2] and t < reconstructed.shape[0]):
            o = original[t, j, i]
            r = reconstructed[t, j, i]
            if not np.isnan(o) and not np.isnan(r):
                orig_vals.append(o)
                recon_vals.append(r)
    
    if len(orig_vals) == 0:
        if verbose:
            print(f"      No valid CV points extracted")
        return None
    
    orig_vals = np.array(orig_vals)
    recon_vals = np.array(recon_vals)
    diff = recon_vals - orig_vals
    
    metrics = _compute_full_metrics(diff)
    
    if verbose:
        print(f"      RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
              f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")
    
    return MethodResult(**metrics)


def find_dineof_log(run_root: str, lake_id: int) -> Optional[Path]:
    """Find DINEOF log file in {run_root}/logs/chain_lake{lake_id}_row*.out"""
    logs_dir = Path(run_root) / "logs"
    if not logs_dir.exists():
        return None
    pattern = f"chain_lake{lake_id}_row*.out"
    matches = list(logs_dir.glob(pattern))
    if matches:
        return sorted(matches)[-1]
    return None


def parse_dineof_expected_error(log_path: Path) -> Optional[float]:
    """Parse DINEOF log file to extract the reported CV error."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        pattern = r"expected error calculated by cross-validation\s+([\d.]+)"
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def find_lakes_in_experiment(run_root: str) -> List[int]:
    """Find all lake IDs that have preprocessing results."""
    prepared_dir = os.path.join(run_root, "prepared")
    if not os.path.exists(prepared_dir):
        return []
    lake_ids = []
    for lake_folder in sorted(os.listdir(prepared_dir)):
        lake_path = os.path.join(prepared_dir, lake_folder)
        if os.path.isdir(lake_path):
            try:
                lake_id = int(lake_folder.lstrip('0') or '0')
                if lake_id > 0:
                    lake_ids.append(lake_id)
            except ValueError:
                continue
    return lake_ids


def find_alpha_folders(run_root: str, lake_id: int, method_dir: str = "dineof") -> List[str]:
    """Find all alpha folders for a lake."""
    lake_str_padded = f"{lake_id:09d}"
    for lake_str in [lake_str_padded, str(lake_id)]:
        method_path = os.path.join(run_root, method_dir, lake_str)
        if os.path.exists(method_path):
            break
    else:
        return []
    alphas = []
    for folder in sorted(os.listdir(method_path)):
        folder_path = os.path.join(method_path, folder)
        if os.path.isdir(folder_path) and folder.startswith("a"):
            alphas.append(folder)
    return alphas


def run_cv_validation_for_lake(run_root: str, lake_id: int, verbose: bool = True) -> List[LakeCVResult]:
    """Run CV validation for a single lake.
    
    Checks for pre-computed .npz files first (from temporal chunking merge),
    then falls back to standard computation from NetCDF files.
    """
    lake_id9 = f"{lake_id:09d}"
    results = []
    
    # Locate prepared files (may not exist for temporally chunked merge roots)
    prepared_nc = None
    clouds_index_nc = None
    for lake_str in [lake_id9, str(lake_id)]:
        prepared_dir = Path(run_root) / "prepared" / lake_str
        if prepared_dir.exists():
            p = prepared_dir / "prepared.nc"
            c = prepared_dir / "clouds_index.nc"
            if p.exists() and c.exists():
                prepared_nc = p
                clouds_index_nc = c
            break
    
    has_netcdf_prereqs = (prepared_nc is not None and clouds_index_nc is not None)
    
    # Discover alpha folders from any available method directory
    alphas = find_alpha_folders(run_root, lake_id, "dineof")
    if not alphas:
        alphas = find_alpha_folders(run_root, lake_id, "dincae")
    if not alphas:
        alphas = ["a1000"]
    
    for alpha in alphas:
        if verbose:
            print(f"\n    [{alpha}]")
        
        lake_result = LakeCVResult(lake_id=lake_id, alpha=alpha)
        
        # ---- DINEOF ----
        for lake_str in [lake_id9, str(lake_id)]:
            dineof_dir = Path(run_root) / "dineof" / lake_str / alpha
            if dineof_dir.exists():
                break
        else:
            dineof_dir = None
        
        if dineof_dir and dineof_dir.exists():
            if verbose:
                print(f"    DINEOF:")
            dineof_result = compute_dineof_cv_from_gher(dineof_dir, verbose)
            if dineof_result:
                lake_result.dineof = dineof_result
                log_file = find_dineof_log(run_root, lake_id)
                if log_file:
                    reported = parse_dineof_expected_error(log_file)
                    if reported is not None:
                        lake_result.dineof.reported_rmse = reported
                        diff = abs(dineof_result.rmse - reported)
                        rel_diff = diff / reported if reported > 0 else float('inf')
                        lake_result.dineof.verified = rel_diff < 0.01
                        if verbose:
                            status = "✓ VERIFIED" if lake_result.dineof.verified else "✗ MISMATCH"
                            print(f"      Reported: {reported:.6f}, Computed: {dineof_result.rmse:.6f} - {status}")
        
        # ---- DINCAE ----
        for lake_str in [lake_id9, str(lake_id)]:
            dincae_dir = Path(run_root) / "dincae" / lake_str / alpha
            if dincae_dir.exists():
                break
        else:
            dincae_dir = None
        
        if dincae_dir and dincae_dir.exists():
            # Check for pre-computed .npz first (from temporal chunking merge)
            dincae_npz = dincae_dir / "cv_pairs_dincae.npz"
            if dincae_npz.exists():
                if verbose:
                    print(f"    DINCAE (from cv_pairs_dincae.npz):")
                data = np.load(str(dincae_npz), allow_pickle=True)
                diff = data["diff"]
                valid = ~np.isnan(diff)
                diff = diff[valid]
                if len(diff) > 0:
                    metrics = _compute_full_metrics(diff)
                    lake_result.dincae = MethodResult(**metrics)
                    if verbose:
                        print(f"      RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
                              f"Bias={metrics['bias']:.6f}, Median={metrics['median']:.6f}, N={metrics['n_points']}")
            else:
                # Standard: compute from NetCDF files
                dincae_results_nc = dincae_dir / "dincae_results.nc"
                if dincae_results_nc.exists() and has_netcdf_prereqs:
                    if verbose:
                        print(f"    DINCAE (from NetCDF):")
                    dincae_result = compute_dincae_cv(prepared_nc, dincae_results_nc, clouds_index_nc, verbose)
                    if dincae_result:
                        lake_result.dincae = dincae_result
                elif verbose:
                    if not has_netcdf_prereqs:
                        print(f"    DINCAE: no .npz and no prepared.nc/clouds_index.nc, skipping")
                    else:
                        print(f"    DINCAE: dincae_results.nc not found")
        
        if lake_result.dineof.rmse is not None or lake_result.dincae.rmse is not None:
            results.append(lake_result)
    
    if not results and verbose:
        print(f"  No CV results found for lake {lake_id}")
    
    return results


def write_csv(results: List[LakeCVResult], output_path: str):
    """Write results to CSV file with wide format (comprehensive metrics)."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['lake_id', 'alpha']
        for method in ['dineof', 'dincae']:
            for m in METRIC_NAMES:
                header.append(f"{m}({method})")
            header.append(f"verified({method})" if method == "dineof" else None)
        header = [h for h in header if h is not None]
        writer.writerow(header)
        
        for r in results:
            row = [r.lake_id, r.alpha]
            for method_obj, is_dineof in [(r.dineof, True), (r.dincae, False)]:
                for m in METRIC_NAMES:
                    val = getattr(method_obj, m, None)
                    if val is None:
                        row.append("")
                    elif isinstance(val, float):
                        row.append(f"{val:.6f}")
                    else:
                        row.append(val)
                if is_dineof:
                    row.append(method_obj.verified if method_obj.verified is not None else "")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="CV Validation - DINEOF from .dat files, DINCAE from NetCDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FAIR COMPARISON MODE (default when using --all):
  Only includes lakes where BOTH DINEOF and DINCAE completed processing.
  This ensures CV statistics are computed on the same sample.
  
  Use --no-fair-comparison to include all available data.

Examples:
    # Process all lakes with fair comparison (recommended)
    python run_cv_validation.py --run-root /path/to/experiment --all
    
    # Process specific lake (no filtering needed)
    python run_cv_validation.py --run-root /path/to/experiment --lake-id 4503
    
    # All lakes without fair comparison filter
    python run_cv_validation.py --run-root /path/to/experiment --all --no-fair-comparison
        """
    )
    parser.add_argument("--run-root", required=True, help="Base directory of the experiment")
    
    lake_group = parser.add_mutually_exclusive_group(required=True)
    lake_group.add_argument("--lake-id", type=int, help="Process single lake by ID")
    lake_group.add_argument("--lake-ids", type=int, nargs="+", help="Process multiple lakes by ID")
    lake_group.add_argument("--all", action="store_true", help="Process all lakes")
    
    parser.add_argument("--output", "-o", default=None, help="Output CSV file (default: auto-generated)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--no-fair-comparison", action="store_true",
                        help="Disable fair comparison filtering")
    parser.add_argument("--alpha", default=None, help="Specific alpha slug (e.g., 'a1000')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)
    
    verbose = not args.quiet
    
    # =========================================================================
    # DETERMINE LAKE IDS WITH FAIR COMPARISON FILTERING
    # =========================================================================
    
    completion_summary = None
    
    if args.all:
        # When processing all lakes, apply fair comparison filter by default
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
            # No fair comparison - get all lakes from prepared directory
            lake_ids = find_lakes_in_experiment(args.run_root)
            if not args.no_fair_comparison:
                print("Note: completion_check module not available, processing all lakes")
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        lake_ids = [args.lake_id]
    
    # =========================================================================
    # DETERMINE OUTPUT PATH
    # =========================================================================
    
    if args.output is None:
        # Generate unique output filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(args.run_root, f"cv_results_{timestamp}.csv")
    
    print("=" * 70)
    print("Cross-Validation Validation")
    print("  DINEOF: from CVpoints_*.dat (GHER binary format)")
    print("  DINCAE: from dincae_results.nc vs prepared.nc")
    print("=" * 70)
    print(f"Lakes to process: {len(lake_ids)}")
    print(f"Fair comparison: {'DISABLED' if args.no_fair_comparison else 'ENABLED'}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    all_results = []
    for lake_id in lake_ids:
        print(f"\n[Lake {lake_id}]")
        results = run_cv_validation_for_lake(args.run_root, lake_id, verbose)
        all_results.extend(results)
    
    if all_results:
        write_csv(all_results, args.output)
        print(f"\nResults written to: {args.output}")
        
        # Save exclusion log if we have completion summary
        if completion_summary is not None:
            log_dir = os.path.dirname(args.output) or "."
            log_path = save_exclusion_log(completion_summary, log_dir, 
                                          filename="cv_excluded_lakes_log.csv")
            print(f"Exclusion log saved: {log_path}")
    
    # Summary
    print("\n" + "=" * 140)
    hdr = f"{'Lake':<12} {'Alpha':<8}"
    for method in ['DINEOF', 'DINCAE']:
        for m in ['RMSE', 'MAE', 'Bias', 'Median', 'N']:
            hdr += f" {method+' '+m:<14}"
    print(hdr)
    print("-" * 140)
    
    def _fv(v):
        if v is None: return "-"
        if isinstance(v, float): return f"{v:.6f}"
        return str(v)
    
    for r in all_results:
        line = f"{r.lake_id:<12} {r.alpha:<8}"
        for m_obj in [r.dineof, r.dincae]:
            for attr in ['rmse', 'mae', 'bias', 'median', 'n_points']:
                line += f" {_fv(getattr(m_obj, attr, None)):<14}"
        print(line)
    print("=" * 140)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())