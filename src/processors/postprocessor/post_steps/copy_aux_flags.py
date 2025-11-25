# post_steps/copy_aux_flags.py
from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Optional, Iterable
from .base import PostProcessingStep, PostContext


class CopyAuxFlagsStep(PostProcessingStep):
    """
    Copy auxiliary flag variables (e.g., ice_replaced) from prepared.nc to output.
    """

    def __init__(self, vars_to_copy: Iterable[str] = ("ice_replaced",)):
        super().__init__()
        self.vars_to_copy = tuple(vars_to_copy)

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and len(self.vars_to_copy) > 0

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None

        with xr.open_dataset(ctx.dineof_input_path) as dsi:
            ds_t_len = ds.dims["time"]
            
            if ctx.full_days is not None and ds_t_len == len(ctx.full_days):
                # Full daily timeline (passes 3, 4)
                t_len = len(ctx.full_days)
                idx_map = {int(d): i for i, d in enumerate(ctx.full_days.astype(int))}
                map_idx = np.array([idx_map.get(int(d), -1) for d in ctx.prepared_time_days], dtype="int64")
            else:
                # Original timeline (passes 1, 2)
                t_len = ctx.orig_time_days.size
                map_idx = ctx.map_prepared_to_orig

            for var in self.vars_to_copy:
                if var not in dsi:
                    print(f"[CopyAuxFlags] '{var}' not in prepared.nc; skipping.")
                    continue
                da_sub = dsi[var]
                
                shp = (t_len, da_sub.sizes["lat"], da_sub.sizes["lon"])
                full = np.zeros(shp, dtype=np.uint8)
                sub_vals = da_sub.values.astype(np.uint8)

                for i_sub, i_full in enumerate(map_idx):
                    if i_full >= 0:
                        full[i_full, :, :] = sub_vals[i_sub, :, :]

                ds[var] = xr.DataArray(
                    full,
                    dims=("time", "lat", "lon"),
                    coords={"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
                    attrs=dsi[var].attrs
                )
                print(f"[CopyAuxFlags] Copied '{var}' from prepared.nc")
        return ds


class CopyOriginalVarsStep(PostProcessingStep):
    """
    Copy variables (lake_surface_water_temperature, quality_level) from the 
    original lake time series file to output.
    """

    def __init__(self, vars_to_copy: Iterable[str] = ("lake_surface_water_temperature", "quality_level")):
        super().__init__()
        self.vars_to_copy = tuple(vars_to_copy)

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and len(self.vars_to_copy) > 0

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None

        with xr.open_dataset(ctx.lake_path) as ds_orig:
            orig_days = self.npdatetime_to_days_since_epoch(ds_orig["time"].values)
            ds_t_len = ds.dims["time"]
            
            if ctx.full_days is not None and ds_t_len == len(ctx.full_days):
                # Full daily timeline (passes 3, 4)
                t_len = len(ctx.full_days)
                idx_map = {int(d): i for i, d in enumerate(ctx.full_days.astype(int))}
                map_idx = np.array([idx_map.get(int(d), -1) for d in orig_days], dtype="int64")
            else:
                # Original timeline (passes 1, 2) - direct 1:1 mapping
                t_len = ds_t_len
                map_idx = np.arange(len(orig_days), dtype="int64")

            for var in self.vars_to_copy:
                if var not in ds_orig:
                    print(f"[CopyOriginalVars] '{var}' not in original file; skipping.")
                    continue
                da_sub = ds_orig[var]
                
                shp = (t_len, da_sub.sizes["lat"], da_sub.sizes["lon"])
                if np.issubdtype(da_sub.dtype, np.floating):
                    full = np.full(shp, np.nan, dtype=da_sub.dtype)
                    sub_vals = da_sub.values
                else:
                    full = np.zeros(shp, dtype=np.uint8)
                    sub_vals = da_sub.values.astype(np.uint8)

                for i_sub, i_full in enumerate(map_idx):
                    if 0 <= i_full < t_len:
                        full[i_full, :, :] = sub_vals[i_sub, :, :]

                ds[var] = xr.DataArray(
                    full,
                    dims=("time", "lat", "lon"),
                    coords={"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
                    attrs=ds_orig[var].attrs
                )
                print(f"[CopyOriginalVars] Copied '{var}' from original lake file")
        return ds