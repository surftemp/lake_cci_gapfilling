from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import xarray as xr
from .contracts import PreparedNC, DincaeArtifacts

def _as_datetime64_from_int(time_da: xr.DataArray, epoch: str) -> xr.DataArray:
    if np.issubdtype(time_da.dtype, np.datetime64):
        return time_da
    if not (np.issubdtype(time_da.dtype, np.integer) or np.issubdtype(time_da.dtype, np.floating)):
        return time_da
    base = np.datetime64(epoch.replace("Z", ""), "ns")
    vals = np.asarray(time_da.values, dtype=float)
    out = base + (vals * np.timedelta64(1, "D")).astype("timedelta64[ns]")
    # Filter out CF encoding attributes that would conflict with xarray encoding
    cf_encoding_attrs = {"calendar", "units", "_FillValue"}
    clean_attrs = {k: v for k, v in time_da.attrs.items() if k not in cf_encoding_attrs}
    return xr.DataArray(out, dims=time_da.dims, coords=time_da.coords, name=time_da.name, attrs=clean_attrs)

def convert_time(in_nc: Path, out_nc: Path, epoch: str) -> None:
    ds = xr.load_dataset(in_nc)
    if "time" not in ds:
        raise ValueError("No 'time' coordinate in dataset.")
    ds = ds.assign_coords(time=_as_datetime64_from_int(ds["time"], epoch))
    
    # Remove CF encoding attributes from time coordinate to avoid conflict
    # when xarray tries to encode datetime64 values
    cf_encoding_attrs = {"calendar", "units", "_FillValue"}
    if "time" in ds.coords:
        ds["time"].attrs = {
            k: v for k, v in ds["time"].attrs.items() 
            if k not in cf_encoding_attrs
        }
    
    ds.to_netcdf(out_nc)

def _bbox_from_mask(mask2d: np.ndarray, buffer: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        raise ValueError("Lake mask has no valid (==1) pixels.")
    i0, i1 = int(ys.min()), int(ys.max()) + 1
    j0, j1 = int(xs.min()), int(xs.max()) + 1
    i0 = max(0, i0 - buffer); j0 = max(0, j0 - buffer)
    i1 = min(mask2d.shape[0], i1 + buffer)
    j1 = min(mask2d.shape[1], j1 + buffer)
    return i0, i1, j0, j1

def crop_to_mask(in_nc: Path, out_nc: Path, buffer: int) -> None:
    ds = xr.load_dataset(in_nc)
    if "lakeid" not in ds:
        raise ValueError("Expected variable 'lakeid' to locate the lake mask (1==water).")
    lakeid = ds["lakeid"].load()
    if lakeid.ndim != 2:
        raise ValueError("Expected lakeid(lat,lon) 2D mask.")
    mask = (lakeid == 1).values
    i0, i1, j0, j1 = _bbox_from_mask(mask, buffer)
    lat_name, lon_name = "lat", "lon"
    if lat_name not in ds.dims or lon_name not in ds.dims:
        dims = list(ds.dims)
        if len(dims) >= 2: lat_name, lon_name = dims[-2], dims[-1]
        else: raise ValueError("Cannot determine (lat, lon) dimension names.")
    ds_crop = ds.isel({lat_name: slice(i0, i1), lon_name: slice(j0, j1)}).copy()
    ds_crop.attrs.update({
        "crop_i0": i0, "crop_i1": i1,
        "crop_j0": j0, "crop_j1": j1,
        "crop_buffer_pixels": int(buffer),
    })
    ds_crop.to_netcdf(out_nc)

def add_cv_clouds(
    in_nc: Path,
    out_cv_nc: Path,
    out_clean_nc: Path,
    cv_fraction: float = 0.1,
    random_seed: int | None = 1234,
    variable_name: str = "lake_surface_water_temperature",
    minseafrac: float = 0.05,
    min_frac_missing: float = 0.0,
    max_frac_missing: float = 1.0,
) -> None:
    """
    Old-pipeline-style CV generator:

    - Build sea/land mask from fraction of valid data (minseafrac).
    - Write a *clean* cropped file with mask + count_nomissing (out_clean_nc).
    - Create a CV version (out_cv_nc) by copying realistic cloud patterns
      from candidate timesteps until ~cv_fraction of all valid points
      are turned into CV points.

    Here:
      * cv_fraction is FRACTION OF VALID PIXELS (like old mincvfrac),
        not fraction of timesteps.
      * min_frac_missing / max_frac_missing define which timesteps can be
        used as source cloud patterns.
    """
    import logging

    logger = logging.getLogger(__name__)

    # --- Load base cropped file ---
    ds = xr.open_dataset(in_nc)
    if variable_name not in ds:
        raise KeyError(f"{variable_name!r} not found in {in_nc}")

    data = ds[variable_name].values  # (time, lat, lon)
    if data.ndim != 3:
        raise ValueError(
            f"{variable_name!r} must be 3-D (time, lat, lon); got shape {data.shape}"
        )

    n_time, n_lat, n_lon = data.shape

    # --- 1) Compute mask & count_nomissing (like old add_mask) ---
    count_nomissing = np.sum(~np.isnan(data), axis=0)  # (lat, lon)
    frac_valid = count_nomissing / max(1, n_time)
    mask = (frac_valid > minseafrac).astype(np.int8)   # sea=1, land=0

    # Dimension names (skip time)
    dims_lat, dims_lon = ds[variable_name].dims[1:3]

    ds_clean = ds.copy()
    ds_clean["mask"] = xr.DataArray(
        mask,
        dims=(dims_lat, dims_lon),
        attrs={"long_name": "mask (sea=1, land=0)"},
    )
    ds_clean["count_nomissing"] = xr.DataArray(
        count_nomissing.astype(np.int32),
        dims=(dims_lat, dims_lon),
        attrs={"long_name": "number of present data"},
    )

    # Write the "clean" (no extra CV clouds) file
    ds_clean.to_netcdf(out_clean_nc)

    # --- 2) Create CV file in memory (like old add_cv_points) ---
    ds_cv = ds_clean.copy()
    data_cv = ds_cv[variable_name].values  # view into ds_cv

    # Apply sea mask: anything outside sea set to NaN
    mask_bool = mask == 1
    mask_3d = np.broadcast_to(mask_bool[np.newaxis, :, :], data_cv.shape)
    data_cv[~np.isnan(data_cv) & ~mask_3d] = np.nan

    # Count valid data points (only sea pixels) after applying mask
    nvalid = int(np.sum(~np.isnan(data_cv)))
    if nvalid == 0:
        logger.warning(
            "No valid data after applying sea mask; writing clean file as CV file."
        )
        ds_cv[variable_name].values = data_cv
        ds_cv.to_netcdf(out_cv_nc)
        ds.close()
        return

    target_ncv = cv_fraction * nvalid
    ncv = 0
    dest_count = 0

    # Per-time missing counts & frac_missing
    nmissing = np.sum(np.isnan(data_cv), axis=(1, 2))
    mask_count = int(np.sum(~mask_bool))  # land pixels
    total_pixels = n_lat * n_lon
    denom = total_pixels - mask_count
    if denom <= 0:
        raise ValueError("Denominator for frac_missing is non-positive; check mask.")

    frac_missing = (nmissing - mask_count) / denom

    logger.info(
        "Fraction missing range: %.3f to %.3f",
        float(np.min(frac_missing)),
        float(np.max(frac_missing)),
    )

    # Candidate source timesteps (realistic cloudiness only)
    candidate_mask_index = np.where(
        (min_frac_missing <= frac_missing) & (frac_missing <= max_frac_missing)
    )[0]
    logger.info("Number of candidate masks: %d", candidate_mask_index.size)

    if candidate_mask_index.size == 0:
        logger.error(
            "No candidate masks found for given min_frac_missing/max_frac_missing. "
            "Leaving data without extra CV points."
        )
        ds_cv[variable_name].values = data_cv
        ds_cv.to_netcdf(out_cv_nc)
        ds.close()
        return

    # Sort destination timesteps by missing count (ascending)
    sorted_indices = np.argsort(nmissing)

    # RNG for reproducibility
    rng = np.random.default_rng(random_seed)

    changed_timesteps: list[int] = []

    logger.info(
        "Processing CV points (target %.2f%% of valid pixels).",
        100.0 * cv_fraction,
    )
    for n_dest in sorted_indices:
        # Randomly pick a source mask
        n_source = int(rng.choice(candidate_mask_index))

        # Apply source's missing pattern to destination
        tmp = data_cv[n_dest, :, :].copy()
        nmissing_before = int(np.sum(np.isnan(tmp)))

        tmp[np.isnan(data_cv[n_source, :, :])] = np.nan
        nmissing_after = int(np.sum(np.isnan(tmp)))

        data_cv[n_dest, :, :] = tmp
        changed_timesteps.append(n_dest)

        new_cv_added = nmissing_after - nmissing_before
        if new_cv_added > 0:
            ncv += new_cv_added
            dest_count += 1

        if ncv >= target_ncv:
            break

    percentage = 100.0 * ncv / nvalid if nvalid > 0 else 0.0
    logger.info("Number of CV points: %d (%.2f%% of valid)", ncv, percentage)
    logger.info("Number of corrupted timesteps: %d", dest_count)

    # Attach stats as global attrs (optional)
    ds_cv.attrs["cv_ncv"] = int(ncv)
    ds_cv.attrs["cv_nvalid"] = int(nvalid)
    ds_cv.attrs["cv_percentage"] = float(percentage)
    ds_cv.attrs["cv_dest_count"] = int(dest_count)
    ds_cv.attrs["cv_changed_timesteps"] = np.array(
        changed_timesteps, dtype=np.int32
    )
    ds_cv.attrs["cv_masked_frac"] = float(cv_fraction)

    # Write CV file (with clouds added)
    ds_cv[variable_name].values = data_cv
    ds_cv.to_netcdf(out_cv_nc)

    ds.close()


def apply_dineof_cv_points(
    in_nc: Path,
    clouds_index_nc: Path,
    out_cv_nc: Path,
    out_clean_nc: Path,
    variable_name: str = "lake_surface_water_temperature",
    minseafrac: float = 0.05,
) -> dict:
    """
    Apply DINEOF's CV points (from clouds_index.nc) to DINCAE's cropped input.
    
    This ensures DINCAE uses the exact same CV points as DINEOF for fair comparison.
    
    The process:
    1. Load clouds_index.nc which contains (spatial_index, time_index) pairs
    2. Map spatial_index -> (lat_orig, lon_orig) via iindex/jindex lookup
    3. Apply crop offset: lat_crop = lat_orig - crop_i0, lon_crop = lon_orig - crop_j0  
    4. Mask those pixels as NaN in the output
    
    Args:
        in_nc: Path to cropped input (prepared_datetime_cropped.nc)
        clouds_index_nc: Path to DINEOF's clouds_index.nc
        out_cv_nc: Path for output with CV points masked as NaN
        out_clean_nc: Path for clean output (no CV masking)
        variable_name: Name of the temperature variable
        minseafrac: Minimum fraction of valid data for sea mask
        
    Returns:
        dict with statistics about CV point application
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Load the cropped dataset
    ds = xr.open_dataset(in_nc)
    
    if variable_name not in ds:
        raise KeyError(f"{variable_name!r} not found in {in_nc}")
    
    data = ds[variable_name].values  # (time, lat, lon)
    if data.ndim != 3:
        raise ValueError(f"{variable_name!r} must be 3-D; got shape {data.shape}")
    
    n_time, n_lat, n_lon = data.shape
    
    # --- 1) Compute mask & count_nomissing (same as add_cv_clouds) ---
    count_nomissing = np.sum(~np.isnan(data), axis=0)  # (lat, lon)
    frac_valid = count_nomissing / max(1, n_time)
    
    if "lakeid" in ds:
        mask = (ds["lakeid"].values == 1).astype(np.int8)  # 1 for water, 0 for land
    else:
        mask = (frac_valid > minseafrac).astype(np.int8)  # fallback to recompute mask: defined as any pixels with less than "minseafrac" temporal data is land
    
    dims_lat, dims_lon = ds[variable_name].dims[1:3]
    
    # --- 2) Write the clean file (no CV masking) ---
    ds_clean = ds.copy()
    ds_clean["mask"] = xr.DataArray(
        mask,
        dims=(dims_lat, dims_lon),
        attrs={"long_name": "mask (sea=1, land=0)"},
    )
    ds_clean["count_nomissing"] = xr.DataArray(
        count_nomissing.astype(np.int32),
        dims=(dims_lat, dims_lon),
        attrs={"long_name": "number of present data"},
    )
    ds_clean.to_netcdf(out_clean_nc)
    logger.info(f"Wrote clean file: {out_clean_nc}")
    
    # --- 3) Extract crop metadata ---
    try:
        crop_i0 = int(ds.attrs["crop_i0"])  # LAT start in original grid
        crop_i1 = int(ds.attrs["crop_i1"])  # LAT end in original grid
        crop_j0 = int(ds.attrs["crop_j0"])  # LON start in original grid
        crop_j1 = int(ds.attrs["crop_j1"])  # LON end in original grid
    except KeyError as e:
        raise ValueError(
            f"Cropped file missing crop metadata (crop_i0/i1/j0/j1). "
            f"Was it created with crop_to_mask()? Error: {e}"
        )
    
    logger.info(f"Crop bounds: LAT [{crop_i0}:{crop_i1}], LON [{crop_j0}:{crop_j1}]")
    
    # --- 4) Load DINEOF CV points ---
    clouds_index_nc = Path(clouds_index_nc)
    if not clouds_index_nc.exists():
        raise FileNotFoundError(f"DINEOF CV file not found: {clouds_index_nc}")
    
    ds_cv_idx = xr.open_dataset(clouds_index_nc)
    
    # clouds_index shape in Python/xarray: (2, nbpoints) due to NetCDF dimension ordering
    # Row 0 = spatial index m, Row 1 = time index t (both 1-based from Julia)
    clouds_index = ds_cv_idx["clouds_index"].values  # (2, nbpoints)
    iindex = ds_cv_idx["iindex"].values  # (indexcount,) - LON coords (1-based Julia)
    jindex = ds_cv_idx["jindex"].values  # (indexcount,) - LAT coords (1-based Julia)
    
    npoints_total = clouds_index.shape[1]  # Second dimension is nbpoints
    logger.info(f"Total CV points from DINEOF: {npoints_total}")
    
    ds_cv_idx.close()
    
    # --- 5) Create CV-masked data ---
    ds_cv = ds_clean.copy()
    data_cv = ds_cv[variable_name].values.copy()
    
    # Apply sea mask first (same as add_cv_clouds)
    mask_bool = mask == 1
    mask_3d = np.broadcast_to(mask_bool[np.newaxis, :, :], data_cv.shape)
    data_cv[~np.isnan(data_cv) & ~mask_3d] = np.nan
    
    # Count valid data before CV masking
    nvalid_before = int(np.sum(~np.isnan(data_cv)))
    
    # Track statistics
    points_applied = 0
    points_already_nan = 0
    unique_times = set()
    
    # --- 6) Apply each CV point ---
    for p in range(npoints_total):
        # Get spatial index m and time index t (both 1-based from Julia)
        # clouds_index[0, p] = spatial index, clouds_index[1, p] = time index
        m = int(clouds_index[0, p])
        t_julia = int(clouds_index[1, p])
        
        # Convert time to 0-based Python index
        t = t_julia - 1
        
        # Get original grid coordinates (convert 1-based Julia to 0-based Python)
        # iindex = LON coordinate, jindex = LAT coordinate
        lon_orig = int(iindex[m - 1]) - 1
        lat_orig = int(jindex[m - 1]) - 1
        
        # Transform to cropped coordinates
        lat_crop = lat_orig - crop_i0
        lon_crop = lon_orig - crop_j0
        
        # Apply CV mask
        if np.isnan(data_cv[t, lat_crop, lon_crop]):
            points_already_nan += 1
        else:
            data_cv[t, lat_crop, lon_crop] = np.nan
        
        points_applied += 1
        unique_times.add(t)
    
    # --- 7) Compute statistics ---
    nvalid_after = int(np.sum(~np.isnan(data_cv)))
    ncv_effective = nvalid_before - nvalid_after
    cv_percentage = 100.0 * ncv_effective / nvalid_before if nvalid_before > 0 else 0.0
    
    stats = {
        "total_dineof_cv_points": npoints_total,
        "points_applied": points_applied,
        "points_already_nan": points_already_nan,
        "ncv_effective": ncv_effective,
        "nvalid_before": nvalid_before,
        "nvalid_after": nvalid_after,
        "cv_percentage": cv_percentage,
        "unique_timesteps": len(unique_times),
    }
    
    logger.info(f"CV Application Statistics:")
    logger.info(f"  Total from DINEOF:    {npoints_total}")
    logger.info(f"  Points applied:       {points_applied}")
    logger.info(f"  Already NaN:          {points_already_nan}")
    logger.info(f"  Effective CV points:  {ncv_effective} ({cv_percentage:.2f}%)")
    logger.info(f"  Unique timesteps:     {len(unique_times)}")
    
    # --- 8) Store metadata and write CV file ---
    ds_cv.attrs["cv_source"] = "DINEOF clouds_index.nc"
    ds_cv.attrs["cv_source_path"] = str(clouds_index_nc)
    ds_cv.attrs["cv_ncv"] = ncv_effective
    ds_cv.attrs["cv_nvalid"] = nvalid_before
    ds_cv.attrs["cv_percentage"] = cv_percentage
    ds_cv.attrs["cv_points_applied"] = points_applied
    ds_cv.attrs["cv_points_total"] = npoints_total
    ds_cv.attrs["cv_dest_count"] = len(unique_times)
    
    ds_cv[variable_name].values = data_cv
    ds_cv.to_netcdf(out_cv_nc)
    logger.info(f"Wrote CV file: {out_cv_nc}")
    
    ds.close()
    
    return stats


def build_inputs(prepared, dincae_dir: Path, cfg: Dict, clouds_index_nc: Path = None) -> DincaeArtifacts:
    """
    Build DINCAE input files from prepared.nc.
    
    Args:
        prepared: PreparedNC object with path to prepared.nc
        dincae_dir: Directory for DINCAE intermediate files
        cfg: Configuration dict with epoch, crop, cv settings
        clouds_index_nc: Optional path to DINEOF's clouds_index.nc.
                         If provided, uses DINEOF's CV points instead of generating new ones.
                         This ensures identical CV points for fair DINEOF vs DINCAE comparison.
    
    Returns:
        DincaeArtifacts with paths to all intermediate files
    """
    import logging
    logger = logging.getLogger(__name__)
    
    dincae_dir.mkdir(parents=True, exist_ok=True)
    epoch = cfg.get("epoch", "1981-01-01T12:00:00Z")
    buffer = int(cfg.get("crop", {}).get("buffer_pixels", 2))
    var_name = cfg.get("var_name", "lake_surface_water_temperature")
    
    # Step 1: Time conversion
    p_datetime = dincae_dir / "prepared_datetime.nc"
    convert_time(prepared.path, p_datetime, epoch=epoch)
    
    # Step 2: Crop to mask
    p_crop = dincae_dir / "prepared_datetime_cropped.nc"
    crop_to_mask(p_datetime, p_crop, buffer=buffer)
    
    # Step 3: CV generation
    p_cv = dincae_dir / "prepared_datetime_cropped_add_clouds.nc"
    p_clean = dincae_dir / "prepared_datetime_cropped_add_clouds.clean.nc"
    cv_cfg = cfg.get("cv", {})
    
    # Check if we should use DINEOF CV points
    if clouds_index_nc is not None:
        clouds_index_nc = Path(clouds_index_nc)
        if clouds_index_nc.exists():
            logger.info(f"Using DINEOF CV points from: {clouds_index_nc}")
            apply_dineof_cv_points(
                in_nc=p_crop,
                clouds_index_nc=clouds_index_nc,
                out_cv_nc=p_cv,
                out_clean_nc=p_clean,
                variable_name=var_name,
                minseafrac=float(cv_cfg.get("minseafrac", 0.05)),
            )
        else:
            logger.warning(f"clouds_index_nc specified but not found: {clouds_index_nc}")
            logger.warning("Falling back to add_cv_clouds()")
            add_cv_clouds(
                in_nc=p_crop,
                out_cv_nc=p_cv,
                out_clean_nc=p_clean,
                cv_fraction=float(cv_cfg.get("cv_fraction", 0.1)),
                random_seed=cv_cfg.get("random_seed", 1234),
                variable_name=var_name,
                minseafrac=float(cv_cfg.get("minseafrac", 0.05)),
                min_frac_missing=float(cv_cfg.get("min_frac_missing", 0.05)),
                max_frac_missing=float(cv_cfg.get("max_frac_missing", 0.70)),
            )
    else:
        # Original behavior: generate CV points independently
        logger.info("Generating DINCAE CV points with add_cv_clouds()")
        add_cv_clouds(
            in_nc=p_crop,
            out_cv_nc=p_cv,
            out_clean_nc=p_clean,
            cv_fraction=float(cv_cfg.get("cv_fraction", 0.1)),
            random_seed=cv_cfg.get("random_seed", 1234),
            variable_name=var_name,
            minseafrac=float(cv_cfg.get("minseafrac", 0.05)),
            min_frac_missing=float(cv_cfg.get("min_frac_missing", 0.05)),
            max_frac_missing=float(cv_cfg.get("max_frac_missing", 0.70)),
        )
    
    return DincaeArtifacts(
        dincae_dir=dincae_dir,
        prepared_datetime=p_datetime,
        prepared_cropped=p_crop,
        prepared_cropped_cv=p_cv,
        prepared_cropped_clean=p_clean)