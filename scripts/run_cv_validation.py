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


@dataclass
class MethodResult:
    """Results for a single method."""
    rmse: Optional[float] = None
    mae: Optional[float] = None
    bias: Optional[float] = None
    n_points: Optional[int] = None
    verified: Optional[bool] = None
    reported_rmse: Optional[float] = None


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
    """Compute DINEOF CV error from CVpoints_*.dat files."""
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
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    
    if verbose:
        print(f"      RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f}, N={len(best_valid)}")
    
    return MethodResult(rmse=rmse, mae=mae, bias=bias, n_points=len(best_valid))


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
    
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    
    if verbose:
        print(f"      RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f}, N={len(orig_vals)}")
    
    return MethodResult(rmse=rmse, mae=mae, bias=bias, n_points=len(orig_vals))


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
    """Run CV validation for a single lake."""
    lake_id9 = f"{lake_id:09d}"
    results = []
    
    for lake_str in [lake_id9, str(lake_id)]:
        prepared_dir = Path(run_root) / "prepared" / lake_str
        if prepared_dir.exists():
            break
    else:
        if verbose:
            print(f"  Prepared directory not found")
        return results
    
    prepared_nc = prepared_dir / "prepared.nc"
    clouds_index_nc = prepared_dir / "clouds_index.nc"
    
    if not prepared_nc.exists() or not clouds_index_nc.exists():
        if verbose:
            print(f"  Required files not found")
        return results
    
    alphas = find_alpha_folders(run_root, lake_id, "dineof")
    if not alphas:
        alphas = find_alpha_folders(run_root, lake_id, "dincae")
    if not alphas:
        alphas = ["a1000"]
    
    for alpha in alphas:
        if verbose:
            print(f"\n    [{alpha}]")
        
        lake_result = LakeCVResult(lake_id=lake_id, alpha=alpha)
        
        # DINEOF
        for lake_str in [lake_id9, str(lake_id)]:
            dineof_dir = Path(run_root) / "dineof" / lake_str / alpha
            if dineof_dir.exists():
                break
        else:
            dineof_dir = None
        
        if dineof_dir and dineof_dir.exists():
            if verbose:
                print(f"    DINEOF (from CVpoints_*.dat):")
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
        
        # DINCAE
        for lake_str in [lake_id9, str(lake_id)]:
            dincae_dir = Path(run_root) / "dincae" / lake_str / alpha
            if dincae_dir.exists():
                break
        else:
            dincae_dir = None
        
        if dincae_dir and dincae_dir.exists():
            dincae_results_nc = dincae_dir / "dincae_results.nc"
            if dincae_results_nc.exists():
                if verbose:
                    print(f"    DINCAE (from NetCDF):")
                dincae_result = compute_dincae_cv(prepared_nc, dincae_results_nc, clouds_index_nc, verbose)
                if dincae_result:
                    lake_result.dincae = dincae_result
        
        if lake_result.dineof.rmse is not None or lake_result.dincae.rmse is not None:
            results.append(lake_result)
    
    return results


def write_csv(results: List[LakeCVResult], output_path: str):
    """Write results to CSV file with wide format."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lake_id', 'alpha',
            'rmse(dineof)', 'mae(dineof)', 'bias(dineof)', 'n_points(dineof)', 'verified(dineof)',
            'rmse(dincae)', 'mae(dincae)', 'bias(dincae)', 'n_points(dincae)'
        ])
        for r in results:
            row = [
                r.lake_id, r.alpha,
                f"{r.dineof.rmse:.6f}" if r.dineof.rmse is not None else "",
                f"{r.dineof.mae:.6f}" if r.dineof.mae is not None else "",
                f"{r.dineof.bias:.6f}" if r.dineof.bias is not None else "",
                r.dineof.n_points if r.dineof.n_points is not None else "",
                r.dineof.verified if r.dineof.verified is not None else "",
                f"{r.dincae.rmse:.6f}" if r.dincae.rmse is not None else "",
                f"{r.dincae.mae:.6f}" if r.dincae.mae is not None else "",
                f"{r.dincae.bias:.6f}" if r.dincae.bias is not None else "",
                r.dincae.n_points if r.dincae.n_points is not None else "",
            ]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="CV Validation - DINEOF from .dat files, DINCAE from NetCDF")
    parser.add_argument("--run-root", required=True, help="Base directory of the experiment")
    
    lake_group = parser.add_mutually_exclusive_group(required=True)
    lake_group.add_argument("--lake-id", type=int, help="Process single lake by ID")
    lake_group.add_argument("--lake-ids", type=int, nargs="+", help="Process multiple lakes by ID")
    lake_group.add_argument("--all", action="store_true", help="Process all lakes")
    
    parser.add_argument("--output", "-o", default="cv_results.csv", help="Output CSV file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)
    
    if args.all:
        lake_ids = find_lakes_in_experiment(args.run_root)
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        lake_ids = [args.lake_id]
    
    verbose = not args.quiet
    
    print("=" * 70)
    print("Cross-Validation Validation")
    print("  DINEOF: from CVpoints_*.dat (GHER binary format)")
    print("  DINCAE: from dincae_results.nc vs prepared.nc")
    print("=" * 70)
    
    all_results = []
    for lake_id in lake_ids:
        print(f"\n[Lake {lake_id}]")
        results = run_cv_validation_for_lake(args.run_root, lake_id, verbose)
        all_results.extend(results)
    
    if all_results:
        write_csv(all_results, args.output)
        print(f"\nResults written to: {args.output}")
    
    # Summary
    print("\n" + "=" * 100)
    print(f"{'Lake':<12} {'Alpha':<8} {'DINEOF RMSE':<14} {'DINEOF MAE':<14} {'Verified':<10} {'DINCAE RMSE':<14} {'DINCAE MAE':<14}")
    print("-" * 100)
    for r in all_results:
        print(f"{r.lake_id:<12} {r.alpha:<8} "
              f"{r.dineof.rmse:.6f if r.dineof.rmse else '-':<14} "
              f"{r.dineof.mae:.6f if r.dineof.mae else '-':<14} "
              f"{'✓' if r.dineof.verified else '✗' if r.dineof.verified is False else '-':<10} "
              f"{r.dincae.rmse:.6f if r.dincae.rmse else '-':<14} "
              f"{r.dincae.mae:.6f if r.dincae.mae else '-':<14}")
    print("=" * 100)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())