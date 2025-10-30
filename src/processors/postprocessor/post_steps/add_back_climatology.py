# post_steps/add_back_climatology.py
from __future__ import annotations

import os
import numpy as np
import xarray as xr
from typing import Optional
from .base import PostProcessingStep, PostContext


class AddBackClimatologyStep(PostProcessingStep):
    """
    Add back per-pixel climatology: temp_filled = temp_filled + clim(doy, lat, lon)

    Climatology file:
      - Variable candidates: 'lswt_mean_trimmed_345', 'lswt_mean', 'climatology'
      - DOY coordinate can be either:
          * 'doy' in [1..365/366], or
          * 'time' (size 365/366) which we treat as doy (we rename to 'doy')
    """

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        if ds is None:
            return False
        if "temp_filled" not in ds:
            return False
        if not ctx.climatology_path:
            return False
        return os.path.isfile(ctx.climatology_path)

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None
        clim_path = ctx.climatology_path
        try:
            clim_ds = xr.open_dataset(clim_path)
        except Exception as e:
            print(f"[AddBackClimatology] Failed to open climatology: {e}; skipping.")
            return ds

        clim_da = self.pick_first_var(clim_ds, ("lswt_mean_trimmed_345","lswt_mean_trimmed", "lswt_mean", "climatology"))
        if clim_da is None:
            print(f"[AddBackClimatology] No climatology variable found in {clim_path}; skipping.")
            clim_ds.close()
            return ds

        # Normalize to DOY
        if "doy" in clim_da.coords:
            clim = clim_da
        else:
            # accept 'time' of length 365/366 as doy proxy
            if "time" in clim_da.coords and clim_da.sizes.get("time") in (365, 366):
                clim = clim_da.rename({"time": "doy"})
                clim = clim.assign_coords(doy=np.arange(1, clim.sizes["doy"] + 1))
            else:
                print("[AddBackClimatology] Climatology has no 'doy' nor a 365/366-length 'time'; skipping.")
                clim_ds.close()
                return ds

        # Compute DOY for ds.time
        time_npdt = ds["time"].values
        # if already datetime64, convert to doy; else convert from days to dt then doy
        doys = self.doy_from_npdatetime(time_npdt)

        # Expand to (time, lat, lon)
        clim_t = clim.sel(doy=xr.DataArray(doys, dims=("time",)), method="nearest")
        # Align to grid of ds
        clim_t = clim_t.transpose(...)

        # Reindex lat/lon like ds (assumes same names)
        if (clim_t.dims[-2:] != ("lat", "lon")) and all(n in clim_t.dims for n in ("lat", "lon")):
            clim_t = clim_t.transpose("doy", "lat", "lon")
            clim_t = clim_t.drop_vars("doy", errors="ignore")
        # Ensure lat/lon match
        clim_t = clim_t.reindex_like(ds["temp_filled"], method=None)

        # Broadcast time dimension if needed
        if "doy" in clim_t.coords:
            clim_t = clim_t.drop_vars("doy")

        # Add back
        ds["temp_filled"] = (ds["temp_filled"] + clim_t).astype("float32")
        ds.attrs["climatology_added_back"] = 1
        ds.attrs["climatology_file"] = clim_path

        # Units handling
        if ctx.output_units == "celsius":
            ds["temp_filled"] = (ds["temp_filled"] - 273.15).astype("float32")
            ds["temp_filled"].attrs["units"] = "degree_Celsius"
        else:
            ds["temp_filled"].attrs["units"] = "kelvin"

        clim_ds.close()
        return ds
