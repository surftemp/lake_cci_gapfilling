from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import xarray as xr
from .contracts import PreparedNC, DincaeArtifacts

def post_map_to_full(pred_nc: Path, cropped_nc: Path, prepared_full_nc: Path, out_full_nc: Path) -> None:
    ds_pred = xr.load_dataset(pred_nc)
    ds_crop = xr.load_dataset(cropped_nc)
    ds_ref = xr.load_dataset(prepared_full_nc)
    try:
        i0 = int(ds_crop.attrs["crop_i0"]); i1 = int(ds_crop.attrs["crop_i1"])
        j0 = int(ds_crop.attrs["crop_j0"]); j1 = int(ds_crop.attrs["crop_j1"])
    except KeyError:
        raise ValueError("Cropped dataset lacks crop indices (crop_i0/1, crop_j0/1).")
    time_name, lat_name, lon_name = "time", "lat", "lon"
    var_name = "lake_surface_water_temperature"
    if var_name not in ds_pred:
        candidates = [v for v in ds_pred.data_vars if ds_pred[v].ndim >= 2]
        if not candidates: raise ValueError("Prediction file has no valid data variable.")
        var_name = candidates[0]
    T, NY, NX = ds_ref.sizes[time_name], ds_ref.sizes[lat_name], ds_ref.sizes[lon_name]
    full_vals = np.full((T, NY, NX), np.nan, dtype=float)
    ds_pred = ds_pred.sortby(time_name); ds_ref = ds_ref.sortby(time_name)
    pred_reindexed = ds_pred[var_name].reindex({time_name: ds_ref[time_name]})
    full_vals[:, i0:i1, j0:j1] = pred_reindexed.values
    da_full = xr.DataArray(
        full_vals, dims=(time_name, lat_name, lon_name),
        coords={time_name: ds_ref[time_name], lat_name: ds_ref[lat_name], lon_name: ds_ref[lon_name]},
        name=var_name)
    out = xr.Dataset({var_name: da_full})
    out[var_name].attrs.update(ds_pred[var_name].attrs)
    out.attrs.update({"source_model": "DINCAE", "mapping": "cropped_to_full", "quality_level": 5})
    out.to_netcdf(out_full_nc)

def to_dineof_shape(full_nc: Path, prepared_full_nc: Path, out_nc: Path, var_name: str) -> None:
    ds_full = xr.load_dataset(full_nc)
    ds_ref = xr.load_dataset(prepared_full_nc)
    # FIX: Identify source variable and rename to 'temp_filled' for DINEOF compatibility
    source_var = var_name
    if source_var not in ds_full:
        candidates = [v for v in ds_full.data_vars]
        if not candidates: raise ValueError("Full product has no data variable to rename.")
        source_var = candidates[0]
    # CRITICAL: Rename to temp_filled
    target_var = "temp_filled"
    if source_var != target_var:
        ds_full = ds_full.rename({source_var: target_var})
    ds_out = ds_full.reindex_like(ds_ref, method=None, copy=True)
    ds_out[target_var].attrs.setdefault("units", ds_ref[var_name].attrs.get("units", "kelvin"))
    ds_out[target_var].attrs.setdefault("long_name", "lake surface water temperature (DINCAE)")
    ds_out.attrs.setdefault("source_model", "DINCAE")
    ds_out.to_netcdf(out_nc)

def make_merged(output_nc: Path, prepared_full_nc: Path, out_merged_nc: Path, var_name: str) -> None:
    ds_pred = xr.load_dataset(output_nc)
    ds_ref = xr.load_dataset(prepared_full_nc)
    # FIX: Use 'temp_filled' for prediction variable
    pred_var = "temp_filled"
    ref_var = var_name
    if pred_var not in ds_pred or ref_var not in ds_ref:
        raise ValueError(f"Expected '{pred_var}' in prediction and '{ref_var}' in reference.")
    obs = ds_ref[ref_var]
    pred = ds_pred[pred_var].reindex_like(obs)
    merged = xr.where(~np.isnan(obs), obs, pred)
    ds_out = xr.Dataset({pred_var: merged})
    ds_out[pred_var].attrs.update(ds_pred[pred_var].attrs)
    ds_out.attrs.update({"merge_rule": "obs_priority_then_dincae", "source_model": "DINCAE"})
    ds_out.to_netcdf(out_merged_nc)

def write_dineof_shaped_outputs(
    arts: DincaeArtifacts,
    prepared: PreparedNC,
    post_dir: Path,
    final_front_name: str,
    cfg: Dict
) -> Dict[str, Optional[Path]]:
    post_dir.mkdir(parents=True, exist_ok=True)
    if arts.pred_path is None or not arts.pred_path.exists():
        raise FileNotFoundError("DINCAE prediction (data-avg.nc) not found.")
    full_nc = arts.dincae_dir / "data-avg-full.nc"
    post_map_to_full(
        pred_nc=arts.pred_path,
        cropped_nc=arts.prepared_cropped,
        prepared_full_nc=prepared.path,
        out_full_nc=full_nc)
    arts.pred_full_path = full_nc
    var_name = cfg.get("var_name", "lake_surface_water_temperature")
    output_nc = post_dir / f"{final_front_name}_dincae.nc"
    to_dineof_shape(full_nc=full_nc, prepared_full_nc=prepared.path, out_nc=output_nc, var_name=var_name)
    merged_nc = None
    if bool(cfg.get("post", {}).get("write_merged", False)):
        merged_nc = post_dir / f"{final_front_name}_merged_dincae.nc"
        make_merged(output_nc=output_nc, prepared_full_nc=prepared.path, out_merged_nc=merged_nc, var_name=var_name)
    for p in [output_nc, merged_nc, full_nc]:
        if p and Path(p).exists():
            ds = xr.load_dataset(p); ds.attrs["model_suffix"] = "_dincae"; ds.to_netcdf(p)
    return {"output_nc": output_nc, "merged_nc": merged_nc, "full_nc": full_nc}