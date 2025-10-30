# post_steps/copy_aux_flags.py
from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Optional, Iterable
from .base import PostProcessingStep, PostContext


class CopyAuxFlagsStep(PostProcessingStep):
    """
    Copy aux flags (e.g., 'ice_replaced') from prepared.nc to the merged timeline.

    Notes:
      - We fill zeros where the prepared time didn't exist in original.
      - dtype: uint8 where possible.
    """

    def __init__(self, vars_to_copy: Iterable[str] = ("ice_replaced",)):
        super().__init__()
        self.vars_to_copy = tuple(vars_to_copy)

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and len(self.vars_to_copy) > 0

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None, "CopyAuxFlagsStep expects a dataset from previous step."

        with xr.open_dataset(ctx.dineof_input_path) as dsi:
            if ctx.orig_time_days is None or ctx.prepared_time_days is None or ctx.map_prepared_to_orig is None:
                raise RuntimeError("Time mapping not available in context; MergeOutputsStep must run first.")

            t_len = ctx.orig_time_days.size
            for var in self.vars_to_copy:
                if var not in dsi:
                    print(f"[CopyAuxFlags] '{var}' not in prepared.nc; skipping.")
                    continue
                da_sub = dsi[var]  # dims: (time, lat, lon)
                # init full array with zeros (uint8)
                shp = (t_len, da_sub.sizes.get("lat"), da_sub.sizes.get("lon"))
                full = np.zeros(shp, dtype=np.uint8)

                # fill where mapping valid
                map_idx = ctx.map_prepared_to_orig
                valid = map_idx >= 0
                sub_vals = da_sub.values.astype(np.uint8)
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
        return ds
