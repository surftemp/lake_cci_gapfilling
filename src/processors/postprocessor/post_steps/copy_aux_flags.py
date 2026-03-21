# post_steps/copy_aux_flags.py
from __future__ import annotations

import numpy as np
import netCDF4
import xarray as xr
from typing import Optional, Iterable
from .base import PostProcessingStep, PostContext, get_current_rss_mb


class CopyAuxFlagsStep(PostProcessingStep):
    """
    Copy auxiliary flag variables (e.g., ice_replaced) from prepared.nc to output.
    RAM-efficient: chunked time reads.
    """

    def __init__(self, vars_to_copy: Iterable[str] = ("ice_replaced",)):
        super().__init__()
        self.vars_to_copy = tuple(vars_to_copy)

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and len(self.vars_to_copy) > 0

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None

        ds_t_len = ds.dims["time"]
        is_full_daily = (ctx.full_days is not None and ds_t_len == len(ctx.full_days))

        dsi = xr.open_dataset(ctx.dineof_input_path)
        try:
            if is_full_daily:
                t_len = len(ctx.full_days)
                idx_map = {int(d): i for i, d in enumerate(ctx.full_days.astype(int))}
                map_idx = np.array([idx_map.get(int(d), -1) for d in ctx.prepared_time_days], dtype="int64")
            else:
                t_len = ctx.orig_time_days.size
                map_idx = ctx.map_prepared_to_orig

            n_src = len(map_idx)
            nlat = ds.dims["lat"]
            nlon = ds.dims["lon"]
            chunk_size = 500

            for var in self.vars_to_copy:
                if var not in dsi:
                    print(f"[CopyAuxFlags] '{var}' not in prepared.nc; skipping.")
                    continue
                da_sub = dsi[var]  # lazy
                var_attrs = dict(da_sub.attrs)

                full = np.zeros((t_len, nlat, nlon), dtype=np.uint8)

                for t0 in range(0, n_src, chunk_size):
                    t1 = min(t0 + chunk_size, n_src)
                    chunk_map = map_idx[t0:t1]
                    valid = chunk_map >= 0
                    if not valid.any():
                        continue
                    src_chunk = da_sub.isel(time=slice(t0, t1)).values.astype(np.uint8)
                    for i_local in np.where(valid)[0]:
                        full[chunk_map[i_local]] = src_chunk[i_local]
                    del src_chunk

                ds[var] = xr.DataArray(
                    full,
                    dims=("time", "lat", "lon"),
                    coords={"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
                    attrs=var_attrs,
                )
                del full
                print(f"[CopyAuxFlags] Copied '{var}' from prepared.nc")
        finally:
            dsi.close()

        return ds


class CopyOriginalVarsStep(PostProcessingStep):
    """
    Copy variables (lake_surface_water_temperature, quality_level) from the
    original lake time series file to output.

    Uses netCDF4 directly (no xarray) for reading to avoid hidden memory spikes
    from xarray's internal decode/copy operations.
    """

    def __init__(self, vars_to_copy: Iterable[str] = ("lake_surface_water_temperature", "quality_level")):
        super().__init__()
        self.vars_to_copy = tuple(vars_to_copy)

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and len(self.vars_to_copy) > 0

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None
        profiling = ctx.profile_memory

        ds_t_len = ds.dims["time"]
        is_full_daily = (ctx.full_days is not None and ds_t_len == len(ctx.full_days))

        # Open with netCDF4 directly — no xarray decode overhead
        # Disable auto mask/scale so we control the decode dtype precisely
        nc = netCDF4.Dataset(ctx.lake_path, "r")
        nc.set_auto_maskandscale(False)
        try:
            # Read time — num2date works on raw values with units/calendar
            nc_time = nc.variables["time"]
            nc.variables["time"].set_auto_maskandscale(True)  # re-enable for time only
            time_vals = netCDF4.num2date(nc_time[:], nc_time.units, nc_time.calendar)
            # Convert to days since 1981-01-01 12:00
            epoch = np.datetime64("1981-01-01T12:00:00", "ns")
            orig_days = np.array([
                int((np.datetime64(t, "ns") - epoch) / np.timedelta64(1, "D"))
                for t in time_vals
            ], dtype="int64")
            n_src = len(orig_days)

            if is_full_daily:
                t_len = len(ctx.full_days)
                idx_map = {int(d): i for i, d in enumerate(ctx.full_days.astype(int))}
                map_idx = np.array([idx_map.get(int(d), -1) for d in orig_days], dtype="int64")
            else:
                t_len = ds_t_len
                map_idx = np.arange(n_src, dtype="int64")

            nlat = ds.dims["lat"]
            nlon = ds.dims["lon"]
            chunk_size = 200  # smaller chunks for tighter memory control

            if profiling:
                print(f"[CopyOriginalVars][MEM] after open+time: RSS={get_current_rss_mb():.0f} MB")

            for var in self.vars_to_copy:
                if var not in nc.variables:
                    print(f"[CopyOriginalVars] '{var}' not in original file; skipping.")
                    continue

                nc_var = nc.variables[var]
                var_attrs = {k: nc_var.getncattr(k) for k in nc_var.ncattrs()
                             if k not in ("_FillValue", "missing_value",
                                          "scale_factor", "add_offset")}
                has_packing = hasattr(nc_var, "scale_factor") or hasattr(nc_var, "add_offset")
                is_float = np.issubdtype(nc_var.dtype, np.floating)
                out_dtype = np.float32 if (is_float or has_packing) else np.uint8
                # Get packing params for manual decode in float32
                scale = np.float32(getattr(nc_var, "scale_factor", 1.0))
                offset = np.float32(getattr(nc_var, "add_offset", 0.0))
                fill_val = getattr(nc_var, "_FillValue", None)

                # Allocate output
                if out_dtype == np.float32:
                    full = np.full((t_len, nlat, nlon), np.nan, dtype=np.float32)
                else:
                    full = np.zeros((t_len, nlat, nlon), dtype=np.uint8)

                # Read and scatter in time chunks
                for t0 in range(0, n_src, chunk_size):
                    t1 = min(t0 + chunk_size, n_src)
                    chunk_map = map_idx[t0:t1]
                    valid_mask = (chunk_map >= 0) & (chunk_map < t_len)
                    if not valid_mask.any():
                        continue
                    # Raw read — no auto decode, stays in native dtype (e.g. int16)
                    raw = nc_var[t0:t1, :, :]
                    if has_packing:
                        # Manual decode in float32: val = raw * scale + offset
                        # Mark fill values as NaN
                        decoded = raw.astype(np.float32) * scale + offset
                        if fill_val is not None:
                            decoded[raw == fill_val] = np.nan
                        raw = decoded
                        del decoded
                    elif out_dtype == np.float32:
                        raw = raw.astype(np.float32)
                    else:
                        raw = raw.astype(np.uint8)
                    valid_idx = np.where(valid_mask)[0]
                    full[chunk_map[valid_idx]] = raw[valid_idx]
                    del raw

                if profiling:
                    print(f"[CopyOriginalVars][MEM] after copy '{var}': RSS={get_current_rss_mb():.0f} MB")

                ds[var] = xr.DataArray(
                    full,
                    dims=("time", "lat", "lon"),
                    coords={"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
                    attrs=var_attrs,
                )
                del full
                print(f"[CopyOriginalVars] Copied '{var}' from original lake file")

                if var == "lake_surface_water_temperature" and ctx.output_units == "celsius":
                    vals = ds[var].values
                    finite_vals = vals[np.isfinite(vals)]
                    if len(finite_vals) > 0 and np.nanmean(finite_vals) > 100:
                        vals -= 273.15
                        ds[var].values = vals.astype(np.float32)
                        ds[var].attrs["units"] = "degree_Celsius"
                        print(f"[CopyOriginalVars] Converted '{var}' to Celsius")
                    del vals, finite_vals
        finally:
            nc.close()

        return ds