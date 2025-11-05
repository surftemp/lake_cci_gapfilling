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
    return xr.DataArray(out, dims=time_da.dims, coords=time_da.coords, name=time_da.name, attrs=time_da.attrs)

def convert_time(in_nc: Path, out_nc: Path, epoch: str) -> None:
    ds = xr.load_dataset(in_nc)
    if "time" not in ds:
        raise ValueError("No 'time' coordinate in dataset.")
    ds = ds.assign_coords(time=_as_datetime64_from_int(ds["time"], epoch))
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

def add_cv_clouds(in_nc: Path, out_cv_nc: Path, out_clean_nc: Path,
                  cv_fraction: float = 0.1,
                  random_seed: int | None = 1234,
                  variable_name: str = "lake_surface_water_temperature",
                  minseafrac: float = 0.05) -> None:
    ds = xr.load_dataset(in_nc)
    var_data = ds[variable_name].values  # shape: (time, lat, lon)
    count_nomissing = np.sum(~np.isnan(var_data), axis=0)  # (lat, lon)
    n_time = var_data.shape[0]
    frac_valid = count_nomissing / n_time

    # Create land/sea mask: 1 where fraction > minseafrac
    mask = (frac_valid > minseafrac).astype(np.int8)
    
    # Add mask and count_nomissing to dataset
    ds['mask'] = xr.DataArray(
        mask,
        dims=('lat', 'lon'),
        attrs={'long_name': 'mask (sea=1, land=0)'}
    )
    ds['count_nomissing'] = xr.DataArray(
        count_nomissing.astype(np.int32),
        dims=('lat', 'lon'),
        attrs={'long_name': 'number of present data'}
    )    
    
    ds.to_netcdf(out_clean_nc)
    if "time" not in ds: raise ValueError("Dataset must have time dimension for CV masking.")
    if variable_name not in ds: raise ValueError(f"Variable '{variable_name}' not found for CV masking.")
    rng = np.random.default_rng(random_seed)
    ntime = ds.sizes["time"]
    k = max(1, int(round(cv_fraction * ntime)))
    sel_idx = np.sort(rng.choice(ntime, size=k, replace=False))
    var = ds[variable_name].copy()
    if "lakeid" not in ds: raise ValueError("Expected 'lakeid' to identify lake pixels.")
    lake_mask = (ds["lakeid"] == 1)
    var_mask = xr.zeros_like(var, dtype=bool)
    var_mask.loc[dict(time=ds["time"].isel(time=sel_idx))] = True
    lm = lake_mask
    for d in var.dims:
        if d not in lm.dims and d != "time":
            lm = lm.expand_dims({d: var.sizes[d]}) if d in ("lat", "lon") else lm
    masked = var.where(~(var_mask & lm), other=np.nan)
    ds_out = ds.copy()
    ds_out[variable_name] = masked
    ds_out.attrs["cv_masked_frac"] = float(cv_fraction)
    ds_out.attrs["cv_masked_steps"] = int(k)
    ds_out.to_netcdf(out_cv_nc)

def build_inputs(prepared, dincae_dir: Path, cfg: Dict) -> DincaeArtifacts:
    dincae_dir.mkdir(parents=True, exist_ok=True)
    epoch = cfg.get("epoch", "1981-01-01T12:00:00Z")
    buffer = int(cfg.get("crop", {}).get("buffer_pixels", 2))
    var_name = cfg.get("var_name", "lake_surface_water_temperature")
    p_datetime = dincae_dir / "prepared_datetime.nc"
    convert_time(prepared.path, p_datetime, epoch=epoch)
    p_crop = dincae_dir / "prepared_datetime_cropped.nc"
    crop_to_mask(p_datetime, p_crop, buffer=buffer)
    p_cv = dincae_dir / "prepared_datetime_cropped_add_clouds.nc"
    p_clean = dincae_dir / "prepared_datetime_cropped_add_clouds.clean.nc"
    add_cv_clouds(p_crop, p_cv, p_clean,
        cv_fraction=float(cfg.get("cv", {}).get("cv_fraction", 0.1)),
        random_seed=cfg.get("cv", {}).get("random_seed", 1234),
        variable_name=var_name)
    return DincaeArtifacts(
        dincae_dir=dincae_dir,
        prepared_datetime=p_datetime,
        prepared_cropped=p_crop,
        prepared_cropped_cv=p_cv,
        prepared_cropped_clean=p_clean)
