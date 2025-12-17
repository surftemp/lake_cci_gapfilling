#!/usr/bin/env python3
"""
Cross-Validation Error Computation

DINEOF: Reads from GHER binary .dat files (CVpoints_best_estimate.dat, CVpoints_initial.dat)
        These contain the TRUE CV comparison from the model trained WITHOUT CV points.
        
DINCAE: Computes from dincae_results.nc vs prepared.nc at CV point locations.

Output: CSV file with RMSE, MAE, Bias, N_points for each lake/method.

NOTE: DINEOF's dineof_results.nc is trained WITH CV points (after model selection),
      so we cannot use it for CV validation. We must use the .dat files.

Author: Shaerdan / NCEO / University of Reading
Date: December 2024
"""

import argparse
import os
import sys
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

try:
    import xarray as xr
except ImportError:
    print("Error: xarray required. pip install xarray")
    sys.exit(1)

try:
    from scipy.io import FortranFile
except ImportError:
    print("Error: scipy required for GHER format. pip install scipy")
    sys.exit(1)


@dataclass
class CVResult:
    """CV validation result."""
    lake_id: int
    alpha: str
    method: str
    rmse: float
    mae: float
    bias: float
    n_points: int


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
        except Exception as e:
            continue
    
    return None


def compute_dineof_cv(dineof_dir: str, verbose: bool = True) -> Optional[CVResult]:
    """
    Compute DINEOF CV error from CVpoints_*.dat files (GHER format).
    This is the TRUE CV error from the model trained WITHOUT CV points.
    """
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
    
    # Compute metrics
    diff = best_valid - init_valid
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    
    if verbose:
        print(f"    DINEOF CV: RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f}, N={len(best_valid)}")
    
    return CVResult(
        lake_id=0,  # Will be filled in by caller
        alpha="",
        method="dineof",
        rmse=rmse,
        mae=mae,
        bias=bias,
        n_points=len(best_valid)
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
    
    # Compute metrics
    diff = recon_vals - orig_vals
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    
    if verbose:
        print(f"    DINCAE CV: RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f}, N={len(orig_vals)}")
    
    return CVResult(
        lake_id=0,
        alpha="",
        method="dincae",
        rmse=rmse,
        mae=mae,
        bias=bias,
        n_points=len(orig_vals)
    )


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_lake(
    run_root: str,
    lake_id: int,
    alpha: str = "a1000",
    verbose: bool = True
) -> List[CVResult]:
    """Validate both DINEOF and DINCAE for a single lake."""
    
    lake_id9 = f"{lake_id:09d}"
    results = []
    
    prepared_nc = os.path.join(run_root, "prepared", lake_id9, "prepared.nc")
    clouds_index_nc = os.path.join(run_root, "prepared", lake_id9, "clouds_index.nc")
    dineof_dir = os.path.join(run_root, "dineof", lake_id9, alpha)
    dincae_dir = os.path.join(run_root, "dincae", lake_id9, alpha)
    
    # Check prerequisites
    if not os.path.exists(prepared_nc):
        if verbose:
            print(f"  [Lake {lake_id}] prepared.nc not found, skipping")
        return results
    
    if not os.path.exists(clouds_index_nc):
        if verbose:
            print(f"  [Lake {lake_id}] clouds_index.nc not found, skipping")
        return results
    
    if verbose:
        print(f"  [Lake {lake_id}]")
    
    # DINEOF CV (from GHER .dat files)
    if os.path.exists(dineof_dir):
        dineof_result = compute_dineof_cv(dineof_dir, verbose)
        if dineof_result:
            dineof_result.lake_id = lake_id
            dineof_result.alpha = alpha
            results.append(dineof_result)
    elif verbose:
        print(f"    DINEOF dir not found")
    
    # DINCAE CV (from NetCDF files)
    dincae_results_nc = os.path.join(dincae_dir, "dincae_results.nc")
    if os.path.exists(dincae_results_nc):
        dincae_result = compute_dincae_cv(
            prepared_nc, dincae_results_nc, clouds_index_nc, verbose
        )
        if dincae_result:
            dincae_result.lake_id = lake_id
            dincae_result.alpha = alpha
            results.append(dincae_result)
    elif verbose:
        print(f"    DINCAE results not found")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CV Validation - DINEOF from .dat files, DINCAE from NetCDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: DINEOF CV uses CVpoints_*.dat files because dineof_results.nc is trained
      WITH CV points after model selection. The .dat files contain the TRUE CV
      comparison from the model trained WITHOUT CV points.
      
      DINCAE does not retrain after CV, so we can use dincae_results.nc directly.
        """
    )
    parser.add_argument("--run-root", required=True, help="Experiment root directory")
    parser.add_argument("--lake-id", type=int, help="Single lake ID")
    parser.add_argument("--lake-ids", type=int, nargs="+", help="Multiple lake IDs")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("--output", "-o", default="cv_results.csv", help="Output CSV file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Determine lake IDs
    lake_ids = []
    if args.lake_id:
        lake_ids = [args.lake_id]
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        # Auto-discover from prepared directory
        prepared_dir = os.path.join(args.run_root, "prepared")
        if os.path.exists(prepared_dir):
            for d in os.listdir(prepared_dir):
                if d.isdigit():
                    lake_ids.append(int(d))
        lake_ids.sort()
    
    if not lake_ids:
        print("No lake IDs specified or found")
        sys.exit(1)
    
    print("=" * 70)
    print("CV Validation")
    print("  DINEOF: from CVpoints_*.dat (GHER format)")
    print("  DINCAE: from dincae_results.nc vs prepared.nc")
    print("=" * 70)
    print(f"Run root: {args.run_root}")
    print(f"Lakes: {len(lake_ids)}")
    print(f"Alpha: {args.alpha}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    all_results = []
    
    for lake_id in lake_ids:
        results = validate_lake(args.run_root, lake_id, args.alpha, verbose)
        all_results.extend(results)
    
    # Write CSV
    if all_results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['lake_id', 'alpha', 'method', 'rmse', 'mae', 'bias', 'n_points'])
            for r in all_results:
                writer.writerow([r.lake_id, r.alpha, r.method, 
                                f"{r.rmse:.6f}", f"{r.mae:.6f}", f"{r.bias:.6f}", r.n_points])
        print(f"\nResults written to: {args.output}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Lake':<10} {'Method':<10} {'RMSE':<12} {'MAE':<12} {'Bias':<12} {'N':<10}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r.lake_id:<10} {r.method:<10} {r.rmse:<12.6f} {r.mae:<12.6f} {r.bias:<12.6f} {r.n_points:<10}")
    
    print("=" * 70)
    
    # Count summary
    dineof_count = sum(1 for r in all_results if r.method == "dineof")
    dincae_count = sum(1 for r in all_results if r.method == "dincae")
    print(f"\nTotal: {len(all_results)} results ({dineof_count} DINEOF, {dincae_count} DINCAE)")


if __name__ == "__main__":
    main()