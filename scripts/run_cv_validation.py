#!/usr/bin/env python3
"""
Standalone Cross-Validation Runner

Runs CV validation on existing DINEOF and DINCAE pipeline results without
re-running the full pipeline. Compares reconstructed values at CV points
against original observations.

IMPORTANT: CV validation is only meaningful for:
  - dineof_results.nc (raw DINEOF output)  
  - dincae_results.nc (DINCAE output)

CV validation is NOT meaningful for EOF-filtered outputs because the filtering
happens AFTER training is complete. See cv_eof_filtered_explanation.md for details.

Usage:
    # Single lake
    python run_cv_validation.py --run-root /path/to/experiment --lake-id 4503
    
    # All lakes in experiment
    python run_cv_validation.py --run-root /path/to/experiment --all
    
    # Specific lakes
    python run_cv_validation.py --run-root /path/to/experiment --lake-ids 4503 3007 1234
    
    # Save results to JSON
    python run_cv_validation.py --run-root /path/to/experiment --lake-id 4503 --save-json

Author: Shaerdan / NCEO / University of Reading
"""

import argparse
import os
import sys
import json
from glob import glob
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import numpy as np

# Add pipeline src to path for potential imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_PATH = os.path.join(PIPELINE_ROOT, "src", "processors", "postprocessor", "post_steps")

if os.path.exists(SRC_PATH):
    sys.path.insert(0, SRC_PATH)
else:
    sys.path.insert(0, SCRIPT_DIR)

# Try to import xarray
try:
    import xarray as xr
except ImportError:
    print("Error: xarray is required. Install with: pip install xarray")
    sys.exit(1)


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
        m = int(clouds_index[p, 0])       # spatial index (1-based)
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
    """
    if verbose:
        print(f"  Loading prepared.nc...")
        print(f"  Loading reconstruction: {reconstruction_nc.name}")
    
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
                    print(f"  Using variable '{vname}' from reconstruction")
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
        print(f"  Loaded {len(cv_coords)} CV points")
    
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
            print(f"  WARNING: No valid CV points!")
        return CVValidationResult(
            rmse=np.nan, mae=np.nan, bias=np.nan,
            n_points=n_points, n_valid=0,
            n_skipped_nan_original=n_skipped_nan_original,
            n_skipped_nan_recon=n_skipped_nan_recon
        )
    
    diff = recon_vals - orig_vals
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    
    if verbose:
        print(f"  RMSE={rmse:.6f}, MAE={mae:.6f}, Bias={bias:.6f} (N={n_valid})")
    
    return CVValidationResult(
        rmse=rmse, mae=mae, bias=bias,
        n_points=n_points, n_valid=n_valid,
        n_skipped_nan_original=n_skipped_nan_original,
        n_skipped_nan_recon=n_skipped_nan_recon
    )


def parse_dineof_expected_error(log_path: Path) -> Optional[float]:
    """
    Parse DINEOF log file to extract the reported CV error.
    """
    import re
    
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
        print(f"Error: prepared/ directory not found in {run_root}")
        return []
    
    lake_ids = []
    for lake_folder in sorted(os.listdir(prepared_dir)):
        lake_path = os.path.join(prepared_dir, lake_folder)
        if os.path.isdir(lake_path):
            # Handle both "4503" and "000004503" formats
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
    lake_str_plain = str(lake_id)
    
    for lake_str in [lake_str_padded, lake_str_plain]:
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


def run_cv_validation_for_lake(
    run_root: str, 
    lake_id: int,
    verbose: bool = True,
    save_json: bool = False
) -> Dict[str, dict]:
    """Run CV validation for a single lake."""
    
    lake_id9 = f"{lake_id:09d}"
    lake_str_plain = str(lake_id)
    
    # Find prepared.nc and clouds_index.nc
    for lake_str in [lake_id9, lake_str_plain]:
        prepared_dir = Path(run_root) / "prepared" / lake_str
        if prepared_dir.exists():
            break
    else:
        print(f"  Prepared directory not found")
        return {}
    
    prepared_nc = prepared_dir / "prepared.nc"
    clouds_index_nc = prepared_dir / "clouds_index.nc"
    
    if not prepared_nc.exists():
        print(f"  prepared.nc not found")
        return {}
    
    if not clouds_index_nc.exists():
        print(f"  clouds_index.nc not found (CV points not available)")
        return {}
    
    # Find alpha folders
    alphas = find_alpha_folders(run_root, lake_id, "dineof")
    if not alphas:
        print(f"  No alpha folders found")
        return {}
    
    all_results = {}
    
    for alpha in alphas:
        print(f"\n  [{alpha}]")
        results = {}
        
        # Find DINEOF directory
        for lake_str in [lake_id9, lake_str_plain]:
            dineof_dir = Path(run_root) / "dineof" / lake_str / alpha
            if dineof_dir.exists():
                break
        else:
            print(f"    DINEOF directory not found")
            continue
        
        # --- DINEOF ---
        dineof_results = dineof_dir / "dineof_results.nc"
        if dineof_results.exists():
            print(f"\n    DINEOF:")
            try:
                result = compute_cv_error(
                    prepared_nc=prepared_nc,
                    reconstruction_nc=dineof_results,
                    clouds_index_nc=clouds_index_nc,
                    verbose=verbose
                )
                results["dineof"] = asdict(result)
                
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
                            print(f"    Verification: computed={result.rmse:.6f}, reported={dineof_reported:.6f}")
                            if rel_diff < 0.01:
                                print(f"    ✓ VERIFIED (diff={rel_diff*100:.2f}%)")
                            else:
                                print(f"    ✗ MISMATCH (diff={rel_diff*100:.2f}%)")
            
            except Exception as e:
                print(f"    Error: {e}")
                results["dineof"] = {"error": str(e)}
        
        # --- DINCAE ---
        for lake_str in [lake_id9, lake_str_plain]:
            dincae_dir = Path(run_root) / "dincae" / lake_str / alpha
            if dincae_dir.exists():
                break
        else:
            dincae_dir = None
        
        if dincae_dir:
            dincae_results = dincae_dir / "dincae_results.nc"
            if dincae_results.exists():
                print(f"\n    DINCAE:")
                try:
                    result = compute_cv_error(
                        prepared_nc=prepared_nc,
                        reconstruction_nc=dincae_results,
                        clouds_index_nc=clouds_index_nc,
                        verbose=verbose
                    )
                    results["dincae"] = asdict(result)
                
                except Exception as e:
                    print(f"    Error: {e}")
                    results["dincae"] = {"error": str(e)}
        
        # Save results
        if results:
            all_results[alpha] = results
            
            # Save JSON if requested
            if save_json:
                output_dir = dineof_dir / "cv_validation"
                output_dir.mkdir(exist_ok=True)
                json_path = output_dir / "cv_results.json"
                
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                if verbose:
                    print(f"\n    Results saved to: {json_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run CV validation on existing pipeline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single lake
    python run_cv_validation.py --run-root /path/to/exp1 --lake-id 4503
    
    # All lakes
    python run_cv_validation.py --run-root /path/to/exp1 --all
    
    # Specific lakes with JSON output
    python run_cv_validation.py --run-root /path/to/exp1 --lake-ids 4503 3007 --save-json

Output:
    Results printed to console and optionally saved to:
    {run_root}/dineof/{lake_id}/{alpha}/cv_validation/cv_results.json
        """
    )
    
    parser.add_argument("--run-root", required=True,
                        help="Base directory of the experiment")
    
    # Lake selection (mutually exclusive)
    lake_group = parser.add_mutually_exclusive_group(required=True)
    lake_group.add_argument("--lake-id", type=int, 
                            help="Process single lake by ID")
    lake_group.add_argument("--lake-ids", type=int, nargs="+",
                            help="Process multiple lakes by ID")
    lake_group.add_argument("--all", action="store_true",
                            help="Process all lakes in the experiment")
    
    parser.add_argument("--save-json", action="store_true",
                        help="Save results to JSON file")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Less verbose output")
    
    args = parser.parse_args()
    
    # Validate run_root
    if not os.path.exists(args.run_root):
        print(f"Error: Run root does not exist: {args.run_root}")
        sys.exit(1)
    
    # Determine which lakes to process
    if args.all:
        lake_ids = find_lakes_in_experiment(args.run_root)
        print(f"Found {len(lake_ids)} lakes in {args.run_root}")
    elif args.lake_ids:
        lake_ids = args.lake_ids
    else:
        lake_ids = [args.lake_id]
    
    if not lake_ids:
        print("No lakes to process")
        sys.exit(0)
    
    verbose = not args.quiet
    
    # Print header
    print(f"\n{'='*60}")
    print("Cross-Validation Runner")
    print('='*60)
    print(f"Run root: {args.run_root}")
    print(f"Lakes to process: {len(lake_ids)}")
    print(f"Save JSON: {args.save_json}")
    print('='*60)
    
    # Collect all results for summary
    all_lake_results = {}
    success_count = 0
    skip_count = 0
    
    for lake_id in lake_ids:
        print(f"\n[Lake {lake_id}]")
        results = run_cv_validation_for_lake(
            args.run_root, 
            lake_id, 
            verbose=verbose,
            save_json=args.save_json
        )
        
        if results:
            all_lake_results[lake_id] = results
            success_count += 1
        else:
            skip_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Lakes processed: {success_count}")
    print(f"Lakes skipped:   {skip_count}")
    
    if all_lake_results:
        print(f"\n{'Lake':<12} {'Alpha':<8} {'Method':<10} {'CV RMSE':>12} {'N Points':>10} {'Verified':>10}")
        print('-'*64)
        
        for lake_id, alphas in all_lake_results.items():
            for alpha, methods in alphas.items():
                for method, data in methods.items():
                    if "rmse" in data:
                        verified = "✓" if data.get("verified") else "-"
                        print(f"{lake_id:<12} {alpha:<8} {method:<10} {data['rmse']:>12.6f} {data['n_valid']:>10} {verified:>10}")
                    elif "error" in data:
                        print(f"{lake_id:<12} {alpha:<8} {method:<10} {'ERROR':>12}")
    
    print('='*60)
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
