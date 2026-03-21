# post_steps/add_back_climatology.py
from __future__ import annotations

import os
import numpy as np
import xarray as xr
from typing import Optional
from .base import PostProcessingStep, PostContext


class AddBackClimatologyStep(PostProcessingStep):
    """
    Add back per-pixel climatology: lake_surface_water_temperature_reconstructed = lake_surface_water_temperature_reconstructed + clim(doy, lat, lon)

    Climatology file:
      - Variable candidates: 'lswt_mean_trimmed_345', 'lswt_mean', 'climatology'
      - DOY coordinate can be either:
          * 'doy' in [1..365/366], or
          * 'time' (size 365/366) which we treat as doy (we rename to 'doy')
    """

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        if ds is None:
            return False
        if "lake_surface_water_temperature_reconstructed" not in ds:
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
        doys = self.doy_from_npdatetime(time_npdt)

        # Align climatology grid to output grid (they may differ in size)
        # Reindex clim to match ds lat/lon, then extract as float32 numpy
        clim_aligned = clim.reindex(
            lat=ds["lat"].values, lon=ds["lon"].values, method="nearest"
        )
        clim_np = clim_aligned.values.astype(np.float32)  # (366, nlat, nlon) — small, ~few MB
        del clim_aligned

        # Map DOY [1..366] to 0-based index, clamp to valid range
        doy_idx = np.clip(doys - 1, 0, clim_np.shape[0] - 1).astype(np.intp)

        # Add back — index directly into float32 climatology array, no 3D expansion needed
        recon = ds["lake_surface_water_temperature_reconstructed"].values  # float32
        # Process in time chunks to avoid creating a full (T, lat, lon) clim array
        chunk_size = 500
        n_time = len(doy_idx)
        for t0 in range(0, n_time, chunk_size):
            t1 = min(t0 + chunk_size, n_time)
            recon[t0:t1] += clim_np[doy_idx[t0:t1]]  # fancy index: (chunk, lat, lon)
        del clim_np
        ds["lake_surface_water_temperature_reconstructed"].values = recon
        ds.attrs["climatology_added_back"] = 1
        ds.attrs["climatology_file"] = clim_path

        # Units handling — in-place
        if ctx.output_units == "celsius":
            recon -= np.float32(273.15)
            ds["lake_surface_water_temperature_reconstructed"].values = recon
            ds["lake_surface_water_temperature_reconstructed"].attrs["units"] = "degree_Celsius"
        else:
            ds["lake_surface_water_temperature_reconstructed"].attrs["units"] = "kelvin"

        del recon
        clim_ds.close()
        return ds
