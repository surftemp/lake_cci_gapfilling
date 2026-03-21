#!/usr/bin/env python3
"""
Fix dincae_interp_full.nc lake mask for all lakes in an experiment,
then re-run insitu CV assembly.

1. For each lake: load dincae_interp_full.nc, apply lake mask from prepared.nc,
   set non-lake pixels to NaN, overwrite file.
2. Re-run assemble_insitu_cv.py to regenerate stats.
"""
import xarray as xr
import numpy as np
import os
import glob
import sys
import argparse


def fix_lake(exp_root, lake_id9, verbose=True):
    """Apply lake mask to dincae_interp_full.nc for one lake."""
    # Find alpha folder
    post_base = os.path.join(exp_root, "post", lake_id9)
    if not os.path.isdir(post_base):
        return "no_post_dir"

    alphas = [d for d in os.listdir(post_base) if d.startswith("a")]
    if not alphas:
        return "no_alpha"

    for alpha in alphas:
        post_dir = os.path.join(post_base, alpha)

        # Find dincae_interp_full file
        pattern = os.path.join(post_dir, "*_dincae_interp_full.nc")
        files = glob.glob(pattern)
        if not files:
            continue

        interp_path = files[0]

        # Load lake mask from prepared.nc
        prep_path = os.path.join(exp_root, "prepared", lake_id9, "prepared.nc")
        if not os.path.isfile(prep_path):
            if verbose:
                print(f"  {lake_id9}: prepared.nc not found")
            return "no_prepared"

        with xr.open_dataset(prep_path) as ds_prep:
            lakeid = ds_prep['lakeid'].values
        lake_mask = (lakeid == 1)
        non_lake = ~lake_mask

        # Load interp file, check and fix
        ds = xr.open_dataset(interp_path)
        tf = ds['lake_surface_water_temperature_reconstructed'].values  # (time, lat, lon)

        # Check one mid timestep for extra pixels
        mid = tf.shape[0] // 2
        extra_before = np.count_nonzero(~np.isnan(tf[mid]) & non_lake)

        if extra_before == 0:
            ds.close()
            if verbose:
                print(f"  {lake_id9}/{alpha}: already clean")
            return "clean"

        # Apply mask
        tf[:, non_lake] = np.nan
        ds['lake_surface_water_temperature_reconstructed'].values = tf

        # Verify
        extra_after = np.count_nonzero(~np.isnan(tf[mid]) & non_lake)

        # Write back
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        if "lake_surface_water_temperature_reconstructed" in ds:
            enc["lake_surface_water_temperature_reconstructed"] = {"dtype": "float32", "zlib": True, "complevel": 5}

        tmp_path = interp_path + ".tmp"
        ds.to_netcdf(tmp_path, encoding=enc)
        ds.close()
        os.replace(tmp_path, interp_path)

        if verbose:
            print(f"  {lake_id9}/{alpha}: fixed {extra_before} -> {extra_after} extra pixels")

    return "fixed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--fix-only", action="store_true",
                        help="Only fix masks, don't reassemble")
    args = parser.parse_args()

    exp_root = args.exp_root.rstrip("/")
    post_dir = os.path.join(exp_root, "post")

    if not os.path.isdir(post_dir):
        print(f"ERROR: {post_dir} not found")
        sys.exit(1)

    # Find all lakes
    lake_dirs = sorted([d for d in os.listdir(post_dir)
                        if d.isdigit() and os.path.isdir(os.path.join(post_dir, d))])

    print(f"Experiment: {exp_root}")
    print(f"Lakes to check: {len(lake_dirs)}")
    print()

    counts = {"fixed": 0, "clean": 0, "no_file": 0, "error": 0}
    for lake_id9 in lake_dirs:
        try:
            result = fix_lake(exp_root, lake_id9)
            if result == "fixed":
                counts["fixed"] += 1
            elif result == "clean":
                counts["clean"] += 1
            else:
                counts["no_file"] += 1
        except Exception as e:
            print(f"  {lake_id9}: ERROR {e}")
            counts["error"] += 1

    print(f"\nMask fix summary:")
    print(f"  Fixed:   {counts['fixed']}")
    print(f"  Clean:   {counts['clean']}")
    print(f"  No file: {counts['no_file']}")
    print(f"  Errors:  {counts['error']}")

    if args.fix_only:
        print("\n--fix-only: skipping reassembly")
        return

    # Re-run insitu CV assembly
    print("\n" + "=" * 60)
    print("Re-running insitu CV assembly...")
    print("=" * 60)

    scripts_dir = os.path.expanduser("~/lake_cci_gapfilling/scripts")
    assembly_script = os.path.join(scripts_dir, "assemble_insitu_cv.py")

    if not os.path.isfile(assembly_script):
        print(f"ERROR: {assembly_script} not found")
        sys.exit(1)

    import subprocess
    cmd = [
        sys.executable, assembly_script,
        "--run-root", exp_root, "--all",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Assembly failed with exit code {result.returncode}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
