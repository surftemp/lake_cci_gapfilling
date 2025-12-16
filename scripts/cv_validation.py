"""
Cross-Validation Error Computation for Gap-Filling Outputs

This module computes CV RMSE for DINEOF and DINCAE gap-filled outputs by comparing
reconstructed values at CV points against the original observations.

IMPORTANT: CV validation is only meaningful for:
  - dineof_results.nc (raw DINEOF output)
  - dincae_results.nc (DINCAE output)

CV validation is NOT meaningful for EOF-filtered outputs because the filtering
happens AFTER training is complete. See cv_eof_filtered_explanation.md for details.

Author: Shaerdan / NCEO / University of Reading
"""

from __future__ import annotations
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class CVValidationResult:
    """Results from CV validation."""
    rmse: float
    mae: float
    bias: float
    n_points: int
    n_valid: int
    n_skipped_nan_original: int
    n_skipped_nan_recon: int
    
    def __str__(self) -> str:
        return (
            f"CV Validation Results:\n"
            f"  RMSE: {self.rmse:.6f}\n"
            f"  MAE:  {self.mae:.6f}\n"
            f"  Bias: {self.bias:.6f}\n"
            f"  N valid: {self.n_valid} / {self.n_points}"
        )


def load_cv_points(clouds_index_nc: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CV point indices from clouds_index.nc.
    
    Returns:
        clouds_index: (nbpoints, 2) array - [spatial_idx, time_idx] (1-based)
        iindex: (indexcount,) - coordinates for spatial index (1-based)
        jindex: (indexcount,) - coordinates for spatial index (1-based)
    """
    ds = xr.open_dataset(clouds_index_nc)
    
    clouds_index_raw = ds["clouds_index"].values
    
    # Handle dimension ordering - we want (nbpoints, 2)
    if clouds_index_raw.shape[0] == 2:
        clouds_index = clouds_index_raw.T
    else:
        clouds_index = clouds_index_raw
    
    iindex = ds["iindex"].values
    jindex = ds["jindex"].values
    
    ds.close()
    
    return clouds_index, iindex, jindex


def get_cv_coordinates(
    clouds_index: np.ndarray,
    iindex: np.ndarray,
    jindex: np.ndarray
) -> List[Tuple[int, int, int]]:
    """
    Convert CV point indices to (time, lat, lon) coordinates (0-based).
    
    Following the DINCAE adapter convention where iindex->lon, jindex->lat.
    """
    coords = []
    
    for p in range(clouds_index.shape[0]):
        m = int(clouds_index[p, 0])      # spatial index (1-based)
        t_julia = int(clouds_index[p, 1]) # time index (1-based)
        
        # Convert to 0-based Python indices
        t = t_julia - 1
        lon_idx = int(iindex[m - 1]) - 1
        lat_idx = int(jindex[m - 1]) - 1
        
        coords.append((t, lat_idx, lon_idx))
    
    return coords


def compute_cv_error(
    prepared_nc: Path,
    reconstruction_nc: Path,
    clouds_index_nc: Path,
    var_name: str = "lake_surface_water_temperature",
    recon_var_name: str = "temp_filled",
    verbose: bool = True
) -> CVValidationResult:
    """
    Compute CV error for a gap-filled output.
    
    Args:
        prepared_nc: Path to prepared.nc (original observations in anomaly space)
        reconstruction_nc: Path to reconstruction output (dineof_results.nc or dincae_results.nc)
        clouds_index_nc: Path to clouds_index.nc with CV point indices
        var_name: Variable name in prepared.nc
        recon_var_name: Variable name in reconstruction file
        verbose: Print progress info
    
    Returns:
        CVValidationResult with error statistics
    """
    if verbose:
        print(f"[CV] Loading prepared.nc: {prepared_nc}")
        print(f"[CV] Loading reconstruction: {reconstruction_nc}")
    
    # Load original observations
    ds_orig = xr.open_dataset(prepared_nc)
    original = ds_orig[var_name].values
    ds_orig.close()
    
    # Load reconstruction
    ds_recon = xr.open_dataset(reconstruction_nc)
    
    if recon_var_name in ds_recon:
        reconstructed = ds_recon[recon_var_name].values
    else:
        # Find first 3D variable
        for vname in ds_recon.data_vars:
            if ds_recon[vname].ndim == 3:
                reconstructed = ds_recon[vname].values
                if verbose:
                    print(f"[CV] Using variable '{vname}' from reconstruction")
                break
        else:
            ds_recon.close()
            raise ValueError(f"No 3D variable found in {reconstruction_nc}")
    
    ds_recon.close()
    
    # Verify shapes match
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstruction {reconstructed.shape}. "
            f"CV validation requires identical dimensions."
        )
    
    # Load CV points
    clouds_index, iindex, jindex = load_cv_points(clouds_index_nc)
    cv_coords = get_cv_coordinates(clouds_index, iindex, jindex)
    
    if verbose:
        print(f"[CV] Loaded {len(cv_coords)} CV points")
    
    # Compute error at CV points
    n_points = len(cv_coords)
    orig_vals = []
    recon_vals = []
    n_skipped_nan_original = 0
    n_skipped_nan_recon = 0
    
    for t, lat, lon in cv_coords:
        # Bounds check
        if (t < 0 or t >= original.shape[0] or
            lat < 0 or lat >= original.shape[1] or
            lon < 0 or lon >= original.shape[2]):
            n_skipped_nan_original += 1
            continue
        
        orig_val = original[t, lat, lon]
        recon_val = reconstructed[t, lat, lon]
        
        if np.isnan(orig_val):
            n_skipped_nan_original += 1
            continue
        if np.isnan(recon_val):
            n_skipped_nan_recon += 1
            continue
        
        orig_vals.append(orig_val)
        recon_vals.append(recon_val)
    
    orig_vals = np.array(orig_vals)
    recon_vals = np.array(recon_vals)
    n_valid = len(orig_vals)
    
    if n_valid == 0:
        if verbose:
            print(f"[CV] WARNING: No valid CV points!")
        return CVValidationResult(
            rmse=np.nan, mae=np.nan, bias=np.nan,
            n_points=n_points, n_valid=0,
            n_skipped_nan_original=n_skipped_nan_original,
            n_skipped_nan_recon=n_skipped_nan_recon
        )
    
    diff = recon_vals - orig_vals
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    
    if verbose:
        print(f"[CV] RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f} (N={n_valid})")
    
    return CVValidationResult(
        rmse=rmse, mae=mae, bias=bias,
        n_points=n_points, n_valid=n_valid,
        n_skipped_nan_original=n_skipped_nan_original,
        n_skipped_nan_recon=n_skipped_nan_recon
    )


def parse_dineof_expected_error(log_path: Path) -> Optional[float]:
    """
    Parse DINEOF log file to extract the reported CV error.
    
    DINEOF reports: "expected error calculated by cross-validation    X.XXXX"
    """
    import re
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        pattern = r"expected error calculated by cross-validation\s+([\d.]+)"
        match = re.search(pattern, content)
        
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"[CV] Warning: Could not parse DINEOF log: {e}")
    
    return None


def validate_dineof_and_dincae(
    prepared_nc: Path,
    dineof_dir: Path,
    clouds_index_nc: Path,
    dincae_dir: Optional[Path] = None,
    var_name: str = "lake_surface_water_temperature",
    verbose: bool = True
) -> Dict[str, dict]:
    """
    Run CV validation for DINEOF and DINCAE outputs.
    
    This function validates:
    1. dineof_results.nc - compares against DINEOF's internal CV for verification
    2. dincae_results.nc - if available
    
    Args:
        prepared_nc: Path to prepared.nc
        dineof_dir: Directory containing dineof_results.nc
        clouds_index_nc: Path to clouds_index.nc
        dincae_dir: Optional directory containing dincae_results.nc
        var_name: Variable name in prepared.nc
        verbose: Print progress info
    
    Returns:
        Dict with results for each method
    """
    results = {}
    
    # --- DINEOF ---
    dineof_results = dineof_dir / "dineof_results.nc"
    
    if dineof_results.exists():
        if verbose:
            print(f"\n{'='*60}")
            print("DINEOF Cross-Validation")
            print('='*60)
        
        try:
            result = compute_cv_error(
                prepared_nc=prepared_nc,
                reconstruction_nc=dineof_results,
                clouds_index_nc=clouds_index_nc,
                var_name=var_name,
                verbose=verbose
            )
            
            results["dineof"] = {
                "rmse": result.rmse,
                "mae": result.mae,
                "bias": result.bias,
                "n_valid": result.n_valid,
                "n_points": result.n_points,
            }
            
            # Try to verify against DINEOF's internal CV
            dineof_log = dineof_dir / "dineof.out"
            if dineof_log.exists():
                dineof_reported = parse_dineof_expected_error(dineof_log)
                if dineof_reported is not None:
                    diff = abs(result.rmse - dineof_reported)
                    rel_diff = diff / dineof_reported if dineof_reported > 0 else float('inf')
                    
                    results["dineof"]["dineof_reported_rmse"] = dineof_reported
                    results["dineof"]["verification_diff"] = diff
                    results["dineof"]["verified"] = rel_diff < 0.01
                    
                    if verbose:
                        print(f"\n[CV] Verification against DINEOF internal CV:")
                        print(f"     Computed:  {result.rmse:.6f}")
                        print(f"     Reported:  {dineof_reported:.6f}")
                        print(f"     Diff:      {diff:.6f} ({rel_diff*100:.2f}%)")
                        if rel_diff < 0.01:
                            print(f"     ✓ VERIFIED")
                        else:
                            print(f"     ✗ MISMATCH - check implementation")
        
        except Exception as e:
            print(f"[CV] Error processing DINEOF: {e}")
            results["dineof"] = {"error": str(e)}
    
    # --- DINCAE ---
    if dincae_dir is not None:
        dincae_results = dincae_dir / "dincae_results.nc"
        
        if dincae_results.exists():
            if verbose:
                print(f"\n{'='*60}")
                print("DINCAE Cross-Validation")
                print('='*60)
            
            try:
                result = compute_cv_error(
                    prepared_nc=prepared_nc,
                    reconstruction_nc=dincae_results,
                    clouds_index_nc=clouds_index_nc,
                    var_name=var_name,
                    verbose=verbose
                )
                
                results["dincae"] = {
                    "rmse": result.rmse,
                    "mae": result.mae,
                    "bias": result.bias,
                    "n_valid": result.n_valid,
                    "n_points": result.n_points,
                }
                
                # Try to read DINCAE's reported CV RMS
                cv_rms_file = dincae_dir / "cv_rms.txt"
                if cv_rms_file.exists():
                    try:
                        with open(cv_rms_file) as f:
                            line = f.read().strip()
                            if "CV_RMS" in line:
                                dincae_reported = float(line.split()[-1])
                                results["dincae"]["dincae_reported_rmse"] = dincae_reported
                                if verbose:
                                    print(f"\n[CV] DINCAE reported CV RMS: {dincae_reported:.6f}")
                    except:
                        pass
            
            except Exception as e:
                print(f"[CV] Error processing DINCAE: {e}")
                results["dincae"] = {"error": str(e)}
    
    # --- Summary ---
    if verbose and results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"{'Method':<15} {'CV RMSE':>12} {'N Points':>12} {'Verified':>10}")
        print('-'*60)
        
        for method, data in results.items():
            if "rmse" in data:
                verified = "✓" if data.get("verified") else "-"
                print(f"{method:<15} {data['rmse']:>12.6f} {data['n_valid']:>12} {verified:>10}")
            else:
                print(f"{method:<15} {'ERROR':>12}")
    
    return results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for CV validation."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Compute CV error for DINEOF and DINCAE outputs"
    )
    parser.add_argument("lake_id", type=int, help="Lake ID (e.g., 3007)")
    parser.add_argument("run_root", help="Run root directory")
    parser.add_argument("--alpha", default="a1000", help="Alpha slug (default: a1000)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--save-json", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    lake_id9 = f"{args.lake_id:09d}"
    run_root = Path(args.run_root)
    
    # Build paths
    prepared_nc = run_root / "prepared" / lake_id9 / "prepared.nc"
    clouds_index_nc = run_root / "prepared" / lake_id9 / "clouds_index.nc"
    dineof_dir = run_root / "dineof" / lake_id9 / args.alpha
    dincae_dir = run_root / "dincae" / lake_id9 / args.alpha
    
    # Check required files
    if not prepared_nc.exists():
        print(f"ERROR: prepared.nc not found: {prepared_nc}")
        return 1
    
    if not clouds_index_nc.exists():
        print(f"ERROR: clouds_index.nc not found: {clouds_index_nc}")
        return 1
    
    # Run validation
    results = validate_dineof_and_dincae(
        prepared_nc=prepared_nc,
        dineof_dir=dineof_dir,
        clouds_index_nc=clouds_index_nc,
        dincae_dir=dincae_dir if dincae_dir.exists() else None,
        verbose=not args.quiet
    )
    
    if args.save_json and results:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.save_json}")
    
    return 0 if results else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
