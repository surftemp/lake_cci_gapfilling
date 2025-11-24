# post_steps/copy_aux_flags.py
from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Optional, Iterable
from .base import PostProcessingStep, PostContext


class CopyAuxFlagsStep(PostProcessingStep):
    """
    Copy variables from prepared.nc to the output timeline.

    Handles two cases:
      - Original timeline (passes 1, 2): uses ctx.map_prepared_to_orig
      - Full daily timeline (passes 3, 4): uses ctx.full_days to build mapping
    """

    def __init__(self, vars_to_copy: Iterable[str] = ("ice_replaced",)):
        super().__init__()
        self.vars_to_copy = tuple(vars_to_copy)

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and len(self.vars_to_copy) > 0

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None, "CopyAuxFlagsStep expects a dataset from previous step."

        with xr.open_dataset(ctx.dineof_input_path) as dsi:
            # Determine which timeline we're using based on output dataset size
            ds_t_len = ds.dims["time"]
            
            if ctx.full_days is not None and ds_t_len == len(ctx.full_days):
                # Full daily timeline (passes 3, 4)
                t_len = len(ctx.full_days)
                idx_map = {int(d): i for i, d in enumerate(ctx.full_days.astype(int))}
                map_idx = np.array([idx_map.get(int(d), -1) for d in ctx.prepared_time_days], dtype="int64")
            elif ctx.map_prepared_to_orig is not None:
                # Original timeline (passes 1, 2)
                t_len = ctx.orig_time_days.size
                map_idx = ctx.map_prepared_to_orig
            else:
                raise RuntimeError("No time mapping available in context.")

            for var in self.vars_to_copy:
                if var not in dsi:
                    print(f"[CopyAuxFlags] '{var}' not in prepared.nc; skipping.")
                    continue
                da_sub = dsi[var]  # dims: (time, lat, lon)
                
                # init full array and get values
                shp = (t_len, da_sub.sizes.get("lat"), da_sub.sizes.get("lon"))
                if np.issubdtype(da_sub.dtype, np.floating):
                    # Float variables (e.g., lake_surface_water_temperature): use NaN fill
                    full = np.full(shp, np.nan, dtype=da_sub.dtype)
                    sub_vals = da_sub.values
                else:
                    # Integer/byte variables (e.g., ice_replaced, quality_level): use original uint8 behavior
                    full = np.zeros(shp, dtype=np.uint8)
                    sub_vals = da_sub.values.astype(np.uint8)

                # fill where mapping valid
                valid = map_idx >= 0
                for i_sub, ok in enumerate(valid):
                    if not ok:
                        continue
                    i_full = map_idx[i_sub]
                    full[i_full, :, :] = sub_vals[i_sub, :, :]

                ds[var] = xr.DataArray(
                    full,
                    dims=("time", "lat", "lon"),
                    coords={"time": ds["time"].values,
                            "lat": ds["lat"].values,
                            "lon": ds["lon"].values},
                    attrs=dsi[var].attrs
                )
                print(f"[CopyAuxFlags] Copied '{var}' to output")
        return ds