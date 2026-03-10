#!/usr/bin/env python3
"""
Convert temp_filled from Kelvin to Celsius in post .nc files.
Operates in-place. Only converts if units indicate Kelvin.
"""
import argparse
import glob
import os
import sys
import numpy as np
import xarray as xr


def convert_lake(post_dir: str, verbose: bool = True) -> int:
    """Convert all .nc files in a lake's post dir. Returns count of files converted."""
    nc_files = sorted(glob.glob(os.path.join(post_dir, "*.nc")))
    n_converted = 0

    for nc_path in nc_files:
        try:
            ds = xr.open_dataset(nc_path)
            if "temp_filled" not in ds:
                ds.close()
                continue

            units = ds["temp_filled"].attrs.get("units", "").lower().strip()
            # Check if Kelvin — either explicit or values > 100
            is_kelvin = "kelvin" in units or units == "k"
            if not is_kelvin:
                # Also check by value range as safety net
                sample = ds["temp_filled"].values.ravel()
                finite = sample[np.isfinite(sample)]
                if len(finite) > 0 and np.nanmean(finite) > 100:
                    is_kelvin = True

            if not is_kelvin:
                if verbose:
                    print(f"  SKIP (already Celsius): {os.path.basename(nc_path)}")
                ds.close()
                continue

            # Convert
            ds["temp_filled"] = (ds["temp_filled"] - 273.15).astype("float32")
            ds["temp_filled"].attrs["units"] = "degree_Celsius"

            # Update clamp threshold if present
            if ds.attrs.get("subzero_clamped", 0) == 1:
                ds.attrs["subzero_threshold"] = 0.0

            # Write to temp then rename (atomic)
            tmp_path = nc_path + ".tmp"
            enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
            if "temp_filled" in ds:
                enc["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
            ds.to_netcdf(tmp_path, encoding=enc)
            ds.close()
            os.replace(tmp_path, nc_path)

            if verbose:
                print(f"  Converted: {os.path.basename(nc_path)}")
            n_converted += 1

        except Exception as e:
            print(f"  ERROR {os.path.basename(nc_path)}: {e}")
            # Clean up temp file
            tmp_path = nc_path + ".tmp"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return n_converted


def main():
    parser = argparse.ArgumentParser(description="Convert temp_filled from Kelvin to Celsius")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--lake-id", type=int, required=True)
    parser.add_argument("--alpha", default="a1000")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    lake_id9 = f"{args.lake_id:09d}"
    post_dir = os.path.join(args.run_root, "post", lake_id9, args.alpha)

    if not os.path.isdir(post_dir):
        print(f"Post dir not found: {post_dir}")
        sys.exit(1)

    verbose = not args.quiet
    if verbose:
        print(f"Lake {args.lake_id} ({lake_id9})")

    n = convert_lake(post_dir, verbose)
    print(f"Done: {n} files converted")


if __name__ == "__main__":
    main()
