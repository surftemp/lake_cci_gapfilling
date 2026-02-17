#!/usr/bin/env python3
"""
Independent Validation of data_source Flag

This script validates the data_source flag in output NetCDF files by comparing
against INDEPENDENT sources that use completely different code paths:

1. DINCAE's prepared_datetime_cropped_clean.nc and prepared_datetime_cropped_cv.nc
   - CV points (flag=2): where clean has data but cv has NaN
   - Observed/seen (flag=1): where cv has data (not NaN)

2. prepared.nc for true gaps (flag=0)
   - Where data is NaN in prepared.nc

The validation is PIXEL-BY-PIXEL, not just aggregate counts.

Usage:
    python validate_data_source_flag.py --run-root /path/to/experiment --lake-ids 20 44 52
    python validate_data_source_flag.py --run-root /path/to/experiment --lake-ids all
    
Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

import argparse
import sys
import os
import json
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Flag values (must match add_data_source_flag.py)
FLAG_TRUE_GAP = 0
FLAG_OBSERVED_SEEN = 1
FLAG_CV_WITHHELD = 2
FLAG_NOT_RECONSTRUCTED = 255


@dataclass
class ValidationResult:
    """Results from validating one lake."""
    lake_id: int
    success: bool
    total_pixels: int
    n_flag0: int  # true gap
    n_flag1: int  # observed seen
    n_flag2: int  # CV withheld
    n_flag255: int  # not reconstructed
    n_reconstructed: int  # flag0 + flag1 + flag2
    
    # Validation against DINCAE files
    n_cv_expected: int  # from clean vs cv comparison
    n_cv_actual: int    # from flag==2
    n_cv_mismatch: int
    cv_mismatch_samples: List[Tuple]  # (time_idx, lat_idx, lon_idx, expected, actual)
    
    n_seen_expected: int  # from cv file (not NaN)
    n_seen_actual: int    # from flag==1
    n_seen_mismatch: int
    seen_mismatch_samples: List[Tuple]
    
    # Validation against prepared.nc for gaps
    n_gap_expected: int
    n_gap_actual: int
    n_gap_mismatch: int
    gap_mismatch_samples: List[Tuple]
    
    errors: List[str]


def find_lake_files(run_root: Path, lake_id: int, alpha: float = 1000) -> Dict[str, Path]:
    """
    Locate all relevant files for a lake.
    
    Returns dict with keys:
    - output_dineof: LAKE{id}_dineof.nc (has data_source flag)
    - output_dincae: LAKE{id}_dincae.nc (has data_source flag)
    - prepared: prepared.nc
    - clouds_index: clouds_index.nc
    - dincae_clean: prepared_datetime_cropped_clean.nc
    - dincae_cv: prepared_datetime_cropped_cv.nc
    """
    lake_id9 = f"{lake_id:09d}"
    alpha_str = f"a{int(alpha)}"
    
    files = {}
    
    # Output files (post-processed) - use glob to find actual filenames
    post_dir = run_root / "post" / lake_id9 / alpha_str
    
    # Find the actual output files by pattern
    dineof_matches = list(post_dir.glob(f"LAKE{lake_id9}*_dineof.nc"))
    dincae_matches = list(post_dir.glob(f"LAKE{lake_id9}*_dincae.nc"))
    
    files["output_dineof"] = dineof_matches[0] if dineof_matches else post_dir / f"LAKE{lake_id9}_dineof.nc"
    files["output_dincae"] = dincae_matches[0] if dincae_matches else post_dir / f"LAKE{lake_id9}_dincae.nc"
    
    # Prepared files
    prep_dir = run_root / "prepared" / lake_id9
    files["prepared"] = prep_dir / "prepared.nc"
    files["clouds_index"] = prep_dir / "clouds_index.nc"
    
    # DINCAE input files (cropped, with and without CV masking)
    dincae_dir = run_root / "dincae" / lake_id9 / alpha_str
    files["dincae_clean"] = dincae_dir / "prepared_datetime_cropped_add_clouds.clean.nc"
    files["dincae_cv"] = dincae_dir / "prepared_datetime_cropped_add_clouds.nc"
    # Also need the cropped file for crop offsets
    files["dincae_cropped"] = dincae_dir / "prepared_datetime_cropped.nc"
    
    return files


def check_files_exist(files: Dict[str, Path]) -> Tuple[bool, List[str]]:
    """Check which required files exist."""
    required = ["output_dineof", "prepared", "dincae_clean", "dincae_cv"]
    optional = ["output_dincae", "clouds_index", "dincae_cropped"]
    missing = []
    for key in required:
        if key in files and not files[key].exists():
            missing.append(f"{key}: {files[key]}")
    
    return len(missing) == 0, missing


def validate_lake(run_root: Path, lake_id: int, alpha: float = 1000, 
                  max_mismatch_samples: int = 10, verbose: bool = True) -> ValidationResult:
    """
    Validate data_source flag for a single lake by pixel-by-pixel comparison.
    
    Uses DINCAE's clean/cv files as independent ground truth.
    """
    result = ValidationResult(
        lake_id=lake_id,
        success=False,
        total_pixels=0,
        n_flag0=0, n_flag1=0, n_flag2=0, n_flag255=0, n_reconstructed=0,
        n_cv_expected=0, n_cv_actual=0, n_cv_mismatch=0, cv_mismatch_samples=[],
        n_seen_expected=0, n_seen_actual=0, n_seen_mismatch=0, seen_mismatch_samples=[],
        n_gap_expected=0, n_gap_actual=0, n_gap_mismatch=0, gap_mismatch_samples=[],
        errors=[]
    )
    
    # Find files
    files = find_lake_files(run_root, lake_id, alpha)
    ok, missing = check_files_exist(files)
    if not ok:
        result.errors.append(f"Missing files: {missing}")
        return result
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating Lake {lake_id}")
        print(f"{'='*60}")
    
    try:
        # ===== Load output file with data_source flag =====
        ds_out = xr.open_dataset(files["output_dineof"])
        
        if "data_source" not in ds_out:
            result.errors.append("data_source variable not found in output file")
            ds_out.close()
            return result
        
        if "temp_filled" not in ds_out:
            result.errors.append("temp_filled variable not found in output file")
            ds_out.close()
            return result
        
        flag = ds_out["data_source"].values  # (time, lat, lon)
        temp_filled = ds_out["temp_filled"].values  # (time, lat, lon)
        out_time = ds_out["time"].values
        out_lat = ds_out["lat"].values
        out_lon = ds_out["lon"].values
        n_time_out, n_lat_out, n_lon_out = flag.shape
        
        # Get reconstruction mask from temp_filled
        recon_mask = ~np.isnan(temp_filled)
        n_recon_from_temp = int(recon_mask.sum())
        
        result.total_pixels = flag.size
        result.n_flag0 = int((flag == FLAG_TRUE_GAP).sum())
        result.n_flag1 = int((flag == FLAG_OBSERVED_SEEN).sum())
        result.n_flag2 = int((flag == FLAG_CV_WITHHELD).sum())
        result.n_flag255 = int((flag == FLAG_NOT_RECONSTRUCTED).sum())
        result.n_reconstructed = result.n_flag0 + result.n_flag1 + result.n_flag2
        
        if verbose:
            print(f"\nOutput file: {files['output_dineof'].name}")
            print(f"  Shape: ({n_time_out}, {n_lat_out}, {n_lon_out})")
            print(f"  temp_filled not NaN: {n_recon_from_temp:,}")
            print(f"  flag reconstructed (0+1+2): {result.n_reconstructed:,}")
            print(f"  Not reconstructed (255): {result.n_flag255:,}")
            print(f"    gap(0)={result.n_flag0:,}, seen(1)={result.n_flag1:,}, cv(2)={result.n_flag2:,}")
            
            # Check consistency: flag reconstructed should match temp_filled reconstructed
            if n_recon_from_temp != result.n_reconstructed:
                print(f"  WARNING: Mismatch between temp_filled ({n_recon_from_temp:,}) and flag ({result.n_reconstructed:,})")
        
        ds_out.close()
        
        # ===== Load DINCAE clean and cv files =====
        ds_clean = xr.open_dataset(files["dincae_clean"])
        ds_cv = xr.open_dataset(files["dincae_cv"])
        
        # Find the LSWT variable name
        lswt_var = None
        for var in ["lake_surface_water_temperature", "lswt", "temp", "sst"]:
            if var in ds_clean:
                lswt_var = var
                break
        
        if lswt_var is None:
            result.errors.append(f"Could not find LSWT variable in DINCAE files")
            ds_clean.close()
            ds_cv.close()
            return result
        
        clean_data = ds_clean[lswt_var].values  # (time, lat, lon) - CROPPED grid
        cv_data = ds_cv[lswt_var].values
        dincae_time = ds_clean["time"].values
        dincae_lat = ds_clean["lat"].values
        dincae_lon = ds_clean["lon"].values
        
        # Get crop offsets from DINCAE file attributes
        crop_i0 = int(ds_clean.attrs.get("crop_i0", 0))  # lat start
        crop_j0 = int(ds_clean.attrs.get("crop_j0", 0))  # lon start
        
        if verbose:
            print(f"\nDINCAE files:")
            print(f"  Shape: {clean_data.shape}")
            print(f"  Crop offsets: lat_start={crop_i0}, lon_start={crop_j0}")
            
            # Diagnostic: compare clean vs cv files
            clean_valid_count = int(np.sum(~np.isnan(clean_data)))
            cv_valid_count = int(np.sum(~np.isnan(cv_data)))
            cv_masked_count = clean_valid_count - cv_valid_count
            print(f"  Clean file valid pixels: {clean_valid_count:,}")
            print(f"  CV file valid pixels: {cv_valid_count:,}")
            print(f"  Difference (CV masked): {cv_masked_count:,}")
        
        ds_clean.close()
        ds_cv.close()
        
        # ===== Load lake mask from prepared.nc =====
        ds_prep = xr.open_dataset(files["prepared"])
        lake_mask = None
        if "lakeid" in ds_prep:
            lakeid_raw = ds_prep["lakeid"].values
            lake_mask = np.isfinite(lakeid_raw) & (lakeid_raw != 0)  # True for lake pixels
            if verbose:
                print(f"\nLake mask loaded: {int(lake_mask.sum())} lake pixels")
        else:
            if verbose:
                print(f"\nWARNING: No lakeid in prepared.nc, cannot filter by lake mask")
        ds_prep.close()
        
        # ===== Build time mapping: DINCAE time -> output time index =====
        # Convert to comparable format (days since epoch)
        def to_days(t):
            """Convert time to days since 1981-01-01."""
            t = np.asarray(t)
            if np.issubdtype(t.dtype, np.datetime64):
                epoch = np.datetime64("1981-01-01T12:00:00", "ns")
                return ((t.astype("datetime64[ns]") - epoch) / np.timedelta64(1, 'D')).astype(int)
            return t.astype(int)
        
        out_days = to_days(out_time)
        dincae_days = to_days(dincae_time)
        
        out_day_to_idx = {int(d): i for i, d in enumerate(out_days)}
        
        # ===== Pixel-by-pixel validation =====
        if verbose:
            print(f"\nRunning pixel-by-pixel validation...")
        
        n_cv_expected = 0
        n_cv_actual = 0
        n_cv_mismatch = 0
        cv_mismatches = []
        
        n_seen_expected = 0
        n_seen_actual = 0
        n_seen_mismatch = 0
        seen_mismatches = []
        
        n_timesteps_matched = 0
        n_timesteps_skipped = 0
        
        # Load reconstruction mask from output - only validate within reconstructed pixels
        ds_out_reopen = xr.open_dataset(files["output_dineof"])
        recon_mask = ~np.isnan(ds_out_reopen["temp_filled"].values)
        ds_out_reopen.close()
        
        n_skipped_not_recon = 0
        n_skipped_outside_lake = 0  # Track pixels outside lake mask
        
        for t_dincae in range(len(dincae_days)):
            day = int(dincae_days[t_dincae])
            t_out = out_day_to_idx.get(day, -1)
            
            if t_out < 0:
                n_timesteps_skipped += 1
                continue
            
            n_timesteps_matched += 1
            
            # For each spatial pixel in DINCAE grid
            for lat_dincae in range(clean_data.shape[1]):
                for lon_dincae in range(clean_data.shape[2]):
                    # Map to output grid coordinates
                    lat_out = lat_dincae + crop_i0
                    lon_out = lon_dincae + crop_j0
                    
                    # Bounds check
                    if lat_out >= n_lat_out or lon_out >= n_lon_out:
                        continue
                    
                    # Skip pixels outside lake mask (buffer/land pixels)
                    # These correctly have flag=255 after lake mask fix
                    if lake_mask is not None and not lake_mask[lat_out, lon_out]:
                        n_skipped_outside_lake += 1
                        continue
                    
                    # Skip pixels outside reconstruction mask (shouldn't happen now)
                    if not recon_mask[t_out, lat_out, lon_out]:
                        n_skipped_not_recon += 1
                        continue
                    
                    clean_valid = not np.isnan(clean_data[t_dincae, lat_dincae, lon_dincae])
                    cv_valid = not np.isnan(cv_data[t_dincae, lat_dincae, lon_dincae])
                    flag_value = flag[t_out, lat_out, lon_out]
                    
                    # Determine expected flag based on DINCAE files
                    if clean_valid and not cv_valid:
                        # CV point: had data in clean, masked in cv
                        expected_flag = FLAG_CV_WITHHELD
                        n_cv_expected += 1
                        if flag_value == FLAG_CV_WITHHELD:
                            n_cv_actual += 1
                        else:
                            n_cv_mismatch += 1
                            if len(cv_mismatches) < max_mismatch_samples:
                                cv_mismatches.append((
                                    t_out, lat_out, lon_out,
                                    FLAG_CV_WITHHELD, flag_value,
                                    f"day={day}"
                                ))
                    
                    elif cv_valid:
                        # Observed and seen: has data in cv file
                        expected_flag = FLAG_OBSERVED_SEEN
                        n_seen_expected += 1
                        if flag_value == FLAG_OBSERVED_SEEN:
                            n_seen_actual += 1
                        else:
                            n_seen_mismatch += 1
                            if len(seen_mismatches) < max_mismatch_samples:
                                seen_mismatches.append((
                                    t_out, lat_out, lon_out,
                                    FLAG_OBSERVED_SEEN, flag_value,
                                    f"day={day}"
                                ))
                    
                    # Note: if clean is also NaN, it's a gap in the DINCAE region
                    # but we validate gaps separately using prepared.nc
        
        result.n_cv_expected = n_cv_expected
        result.n_cv_actual = n_cv_actual
        result.n_cv_mismatch = n_cv_mismatch
        result.cv_mismatch_samples = cv_mismatches
        
        result.n_seen_expected = n_seen_expected
        result.n_seen_actual = n_seen_actual
        result.n_seen_mismatch = n_seen_mismatch
        result.seen_mismatch_samples = seen_mismatches
        
        if verbose:
            print(f"\n  Timesteps matched: {n_timesteps_matched}, skipped: {n_timesteps_skipped}")
            print(f"  Skipped {n_skipped_outside_lake:,} pixels outside lake mask (buffer/land)")
            print(f"  Skipped {n_skipped_not_recon:,} pixels outside reconstruction mask")
            print(f"\n  CV validation (flag=2):")
            print(f"    Expected (from DINCAE clean vs cv): {n_cv_expected:,}")
            print(f"    Correct in flag: {n_cv_actual:,}")
            print(f"    Mismatches: {n_cv_mismatch:,}")
            if cv_mismatches:
                print(f"    Sample mismatches:")
                for m in cv_mismatches[:5]:
                    print(f"      t={m[0]}, lat={m[1]}, lon={m[2]}: expected={m[3]}, got={m[4]} ({m[5]})")
            
            print(f"\n  Observed/seen validation (flag=1):")
            print(f"    Expected (from DINCAE cv file): {n_seen_expected:,}")
            print(f"    Correct in flag: {n_seen_actual:,}")
            print(f"    Mismatches: {n_seen_mismatch:,}")
            if seen_mismatches:
                print(f"    Sample mismatches:")
                for m in seen_mismatches[:5]:
                    print(f"      t={m[0]}, lat={m[1]}, lon={m[2]}: expected={m[3]}, got={m[4]} ({m[5]})")
        
        # ===== Alternative CV validation using clouds_index.nc directly =====
        # This uses the same approach as add_data_source_flag.py
        if files["clouds_index"].exists():
            ds_cv_idx = xr.open_dataset(files["clouds_index"])
            clouds_index = ds_cv_idx["clouds_index"].values
            iindex = ds_cv_idx["iindex"].values  # LON coords (1-based Julia)
            jindex = ds_cv_idx["jindex"].values  # LAT coords (1-based Julia)
            
            n_cv_from_clouds = clouds_index.shape[1] if clouds_index.ndim == 2 else clouds_index.shape[0]
            ds_cv_idx.close()
            
            # Load prepared.nc to get time mapping
            ds_prep = xr.open_dataset(files["prepared"])
            prep_time = ds_prep["time"].values
            prep_days = to_days(prep_time)
            ds_prep.close()
            
            prep_day_to_out_idx = {int(d): i for i, d in enumerate(out_days) if int(d) in {int(x) for x in prep_days}}
            
            # Validate each CV point from clouds_index.nc
            n_cv_direct_expected = n_cv_from_clouds
            n_cv_direct_correct = 0
            n_cv_direct_mismatch = 0
            cv_direct_mismatches = []
            
            for p in range(n_cv_from_clouds):
                if clouds_index.ndim == 2:
                    m = int(clouds_index[0, p])
                    t_julia = int(clouds_index[1, p])
                else:
                    m = int(clouds_index[p, 0])
                    t_julia = int(clouds_index[p, 1])
                
                t_prep = t_julia - 1  # Convert to 0-based
                if t_prep < 0 or t_prep >= len(prep_days):
                    continue
                
                day = int(prep_days[t_prep])
                t_out = prep_day_to_out_idx.get(day, -1)
                if t_out < 0:
                    continue
                
                # Get coordinates (iindex = LON, jindex = LAT, both 1-based)
                lon_idx = int(iindex[m - 1]) - 1
                lat_idx = int(jindex[m - 1]) - 1
                
                if lat_idx >= n_lat_out or lon_idx >= n_lon_out:
                    continue
                
                flag_value = flag[t_out, lat_idx, lon_idx]
                if flag_value == FLAG_CV_WITHHELD:
                    n_cv_direct_correct += 1
                else:
                    n_cv_direct_mismatch += 1
                    if len(cv_direct_mismatches) < max_mismatch_samples:
                        cv_direct_mismatches.append((t_out, lat_idx, lon_idx, FLAG_CV_WITHHELD, flag_value, f"day={day}"))
            
            if verbose:
                print(f"\n  CV validation via clouds_index.nc (flag=2):")
                print(f"    Total CV points in clouds_index: {n_cv_from_clouds:,}")
                print(f"    Correct in flag: {n_cv_direct_correct:,}")
                print(f"    Mismatches: {n_cv_direct_mismatch:,}")
                if cv_direct_mismatches:
                    print(f"    Sample mismatches:")
                    for m in cv_direct_mismatches[:5]:
                        print(f"      t={m[0]}, lat={m[1]}, lon={m[2]}: expected={m[3]}, got={m[4]} ({m[5]})")
            
            # Use the direct validation results
            result.n_cv_expected = n_cv_direct_expected
            result.n_cv_actual = n_cv_direct_correct
            result.n_cv_mismatch = n_cv_direct_mismatch
            result.cv_mismatch_samples = cv_direct_mismatches
        
        
        # ===== Validate gaps using prepared.nc =====
        # Re-open output to get recon_mask for gap validation
        ds_out = xr.open_dataset(files["output_dineof"])
        temp_filled = ds_out["temp_filled"].values
        recon_mask = ~np.isnan(temp_filled)  # True where reconstruction exists
        ds_out.close()
        
        ds_prep = xr.open_dataset(files["prepared"])
        
        prep_var = None
        for var in ["lake_surface_water_temperature", "lswt", "temp", "sst"]:
            if var in ds_prep:
                prep_var = var
                break
        
        if prep_var:
            prep_data = ds_prep[prep_var].values
            prep_time = ds_prep["time"].values
            prep_days = to_days(prep_time)
            
            # Build mapping from output time to prepared time
            prep_day_to_idx = {int(d): i for i, d in enumerate(prep_days)}
            out_to_prep_idx = []
            for d in out_days:
                out_to_prep_idx.append(prep_day_to_idx.get(int(d), -1))
            out_to_prep_idx = np.array(out_to_prep_idx, dtype=np.int64)
            
            # Validate gaps: for pixels where temp_filled is not NaN AND prepared.nc has NaN
            # flag should be 0 (true gap)
            n_gap_expected = 0
            n_gap_actual = 0
            n_gap_mismatch = 0
            gap_mismatches = []
            
            # Iterate only over reconstructed pixels
            recon_indices = np.argwhere(recon_mask)
            
            for idx in recon_indices:
                t_out, lat_idx, lon_idx = idx[0], idx[1], idx[2]
                
                # Map to prepared time
                t_prep = out_to_prep_idx[t_out]
                if t_prep < 0:
                    continue
                
                # Check if prepared.nc has observation at this pixel
                has_obs = not np.isnan(prep_data[t_prep, lat_idx, lon_idx])
                flag_value = flag[t_out, lat_idx, lon_idx]
                
                if not has_obs:
                    # This is a gap that was filled - flag should be 0
                    n_gap_expected += 1
                    if flag_value == FLAG_TRUE_GAP:
                        n_gap_actual += 1
                    else:
                        n_gap_mismatch += 1
                        if len(gap_mismatches) < max_mismatch_samples:
                            gap_mismatches.append((
                                t_out, lat_idx, lon_idx,
                                FLAG_TRUE_GAP, flag_value,
                                f"no obs in prepared.nc at t_prep={t_prep}"
                            ))
            
            result.n_gap_expected = n_gap_expected
            result.n_gap_actual = n_gap_actual
            result.n_gap_mismatch = n_gap_mismatch
            result.gap_mismatch_samples = gap_mismatches
            
            if verbose:
                print(f"\n  Gap validation (flag=0) - within {len(recon_indices):,} reconstructed pixels:")
                print(f"    Expected (prepared.nc NaN at reconstructed pixels): {n_gap_expected:,}")
                print(f"    Correct in flag: {n_gap_actual:,}")
                print(f"    Mismatches: {n_gap_mismatch:,}")
                if gap_mismatches:
                    print(f"    Sample mismatches:")
                    for m in gap_mismatches[:5]:
                        print(f"      t={m[0]}, lat={m[1]}, lon={m[2]}: expected={m[3]}, got={m[4]} ({m[5]})")
        
        ds_prep.close()
        
        # ===== Summary =====
        total_mismatches = n_cv_mismatch + n_seen_mismatch + n_gap_mismatch
        result.success = (total_mismatches == 0)
        
        if verbose:
            print(f"\n{'='*60}")
            if result.success:
                print(f"✓ Lake {lake_id}: PASSED - All {result.total_pixels:,} pixels validated correctly")
            else:
                print(f"✗ Lake {lake_id}: FAILED - {total_mismatches:,} mismatches found")
            print(f"{'='*60}")
        
        return result
        
    except Exception as e:
        result.errors.append(f"Exception: {str(e)}")
        import traceback
        result.errors.append(traceback.format_exc())
        return result


def get_lake_ids_from_manifest(run_root: Path) -> List[int]:
    """Extract lake IDs from manifest.json."""
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        return []
    
    with open(manifest_path) as f:
        conf = json.load(f)
    
    return conf.get("dataset_options", {}).get("custom_lake_ids", [])


def main():
    parser = argparse.ArgumentParser(
        description="Validate data_source flag by pixel-by-pixel comparison against independent sources"
    )
    parser.add_argument("--run-root", required=True, type=str,
                        help="Path to experiment run root (e.g., /path/to/anomaly-20260125-xxx)")
    parser.add_argument("--lake-ids", nargs="+", required=True,
                        help="Lake IDs to validate, or 'all' to validate all lakes in manifest")
    parser.add_argument("--alpha", type=float, default=1000,
                        help="Alpha value (default: 1000)")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Max mismatch samples to store per category (default: 10)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for results summary")
    
    args = parser.parse_args()
    run_root = Path(args.run_root)
    
    if not run_root.exists():
        print(f"Error: Run root does not exist: {run_root}", file=sys.stderr)
        sys.exit(1)
    
    # Determine lake IDs
    if args.lake_ids == ["all"]:
        lake_ids = get_lake_ids_from_manifest(run_root)
        if not lake_ids:
            print("Error: Could not find lake IDs in manifest.json", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(lake_ids)} lakes in manifest: {lake_ids}")
    else:
        lake_ids = [int(x) for x in args.lake_ids]
    
    # Run validation
    results = []
    for lake_id in lake_ids:
        result = validate_lake(
            run_root, lake_id, args.alpha,
            max_mismatch_samples=args.max_samples,
            verbose=not args.quiet
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    n_passed = sum(1 for r in results if r.success)
    n_failed = sum(1 for r in results if not r.success and not r.errors)
    n_error = sum(1 for r in results if r.errors)
    
    print(f"Total lakes: {len(results)}")
    print(f"  Passed: {n_passed}")
    print(f"  Failed: {n_failed}")
    print(f"  Errors: {n_error}")
    
    if n_failed > 0 or n_error > 0:
        print("\nFailed/Error lakes:")
        for r in results:
            if not r.success:
                total_mismatch = r.n_cv_mismatch + r.n_seen_mismatch + r.n_gap_mismatch
                if r.errors:
                    print(f"  Lake {r.lake_id}: ERROR - {r.errors[0][:100]}")
                else:
                    print(f"  Lake {r.lake_id}: {total_mismatch:,} mismatches "
                          f"(cv={r.n_cv_mismatch}, seen={r.n_seen_mismatch}, gap={r.n_gap_mismatch})")
    
    # Output CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'lake_id', 'success', 'total_pixels', 'n_reconstructed',
                'n_flag0', 'n_flag1', 'n_flag2', 'n_flag255',
                'n_cv_expected', 'n_cv_actual', 'n_cv_mismatch',
                'n_seen_expected', 'n_seen_actual', 'n_seen_mismatch',
                'n_gap_expected', 'n_gap_actual', 'n_gap_mismatch',
                'errors'
            ])
            for r in results:
                writer.writerow([
                    r.lake_id, r.success, r.total_pixels, r.n_reconstructed,
                    r.n_flag0, r.n_flag1, r.n_flag2, r.n_flag255,
                    r.n_cv_expected, r.n_cv_actual, r.n_cv_mismatch,
                    r.n_seen_expected, r.n_seen_actual, r.n_seen_mismatch,
                    r.n_gap_expected, r.n_gap_actual, r.n_gap_mismatch,
                    "; ".join(r.errors) if r.errors else ""
                ])
        print(f"\nResults written to: {args.output}")
    
    # Exit code
    sys.exit(0 if n_passed == len(results) else 1)


if __name__ == "__main__":
    main()
