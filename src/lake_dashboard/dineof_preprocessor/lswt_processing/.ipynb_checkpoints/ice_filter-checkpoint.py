"""
Ice masking & replacement step (applied in ANOMALY space).

Adds synthetic LSWT anomalies equal to (273.15 K - climatology_mean) where ALL hold:
  1) LIC (smoothed_gap_filled_lic_class == 2), after shrinking by N pixels
  2) NO original LSWT obs at that pixel (NaN)
  3) At least one LSWT obs exists elsewhere in the lake at that time
  4) Per-pixel winter criterion: (climatology_mean - 2*climatology_std) < 273.15 K

Writes:
  - lake_surface_water_temperature with anomaly replacements = 273.15 - clim_mean
  - ice_replaced (uint8, time/lat/lon) = 1 where we set a replacement
"""

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig

import xarray as xr
import numpy as np
import datetime
import os
from scipy.ndimage import binary_erosion
from .stats import get_recorder


class IceMaskReplacementStep(ProcessingStep):
    def should_apply(self, config: ProcessingConfig) -> bool:
        # run only if an ice file path is provided
        return bool(getattr(config, "ice_file", None))

    @property
    def name(self) -> str:
        return "Ice Mask Replacement"

    # ---------- helpers ----------
    @staticmethod
    def _days_to_datetime64(days: np.ndarray) -> np.ndarray:
        # Convert int days since 1981-01-01 12:00:00 to datetime64[ns]
        days = np.asarray(days, dtype="int64")
        ref = np.datetime64("1981-01-01T12:00:00", "ns")
        return ref + days.astype("timedelta64[D]")

    @staticmethod
    def _get_lake_mask(ds: xr.Dataset) -> np.ndarray:
        if "lakeid" in ds:
            return (ds["lakeid"].data > 0).astype(bool)
        if "_mask" in ds.attrs:
            return np.asarray(ds.attrs["_mask"]).astype(bool)
        raise ValueError("No lake mask found (need 'lakeid' variable or '_mask' attribute)")

    @staticmethod
    def _pick_var(ds: xr.Dataset, candidates):
        for name in candidates:
            if name in ds.variables:
                return ds[name]
        return None

    @staticmethod
    def _doy_from_days(days: np.ndarray) -> np.ndarray:
        ref = datetime.datetime(1981, 1, 1, 12, 0)
        return np.array([(ref + datetime.timedelta(days=int(d))).timetuple().tm_yday for d in days], dtype=int)

    # ---------- main ----------
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            var = "lake_surface_water_temperature"
            self.validate_dataset(ds, [var, "time", "lat", "lon"])

            da = ds[var]
            time_days = ds["time"].values.astype("int64")

            # lake mask (2-D)
            lake_mask = self._get_lake_mask(ds)
            if lake_mask.sum() == 0:
                print("IceMaskReplacement: empty lake mask -> skip")
                zr2d = np.zeros_like(da.isel(time=0).values, dtype=np.uint8)
                ds["ice_replaced"] = xr.DataArray(
                    np.tile(zr2d, (da.sizes["time"], 1, 1)),
                    dims=("time", "lat", "lon"),
                    coords=da.coords,
                    attrs={"long_name": "Ice replacement flag", "units": "1"},
                )
                # stats: all zeros
                get_recorder().record_pixel_replacement_daily(self.name, time_days, np.zeros_like(time_days, dtype="int64"))
                return ds

            # load LIC dataset
            lic_path = config.ice_file
            if not os.path.exists(lic_path):
                print(f"IceMaskReplacement: file not found {lic_path} -> skip")
                zr2d = np.zeros_like(da.isel(time=0).values, dtype=np.uint8)
                ds["ice_replaced"] = xr.DataArray(
                    np.tile(zr2d, (da.sizes["time"], 1, 1)),
                    dims=("time", "lat", "lon"),
                    coords=da.coords,
                    attrs={"long_name": "Ice replacement flag", "units": "1"},
                )
                get_recorder().record_pixel_replacement_daily(self.name, time_days, np.zeros_like(time_days, dtype="int64"))
                return ds

            lic_ds = xr.open_dataset(lic_path)
            lic_var = "smoothed_gap_filled_lic_class"
            if lic_var not in lic_ds:
                lic_ds.close()
                raise ValueError(f"'{lic_var}' not found in {lic_path}")

            lic = lic_ds[lic_var]  # (time, lat, lon) cropped to lake area; 2==ice; NaN==non-lake

            # align time: ds.time is numeric days -> convert to datetime64[ns]
            t_dt = self._days_to_datetime64(time_days)
            # select LIC to ds times (nearest within 1 day)
            lic = lic.sel(
                time=xr.DataArray(t_dt, dims=("time",)),
                method="nearest",
                tolerance=np.timedelta64(1, "D"),
            )
            # make LIC 'time' EXACTLY match ds.time (int days)
            lic = lic.assign_coords(time=ds["time"].values)

            # align space: reindex LIC lat/lon onto the full LSWT grid
            lic = lic.reindex_like(da.isel(time=0), method=None)

            # ice boolean (NaNs -> False), then shrink (erode) by N pixels
            lic_bool = (lic == 2).fillna(False)
            shrink_n = int(getattr(config, "ice_shrink_pixels", 1) or 1)
            if shrink_n > 0:
                struct = np.ones((3, 3), dtype=bool)
                ice_e = np.empty_like(lic_bool.values, dtype=bool)
                for i in range(ice_e.shape[0]):
                    ice_e[i] = binary_erosion(
                        lic_bool.values[i],
                        structure=struct,
                        iterations=shrink_n,
                        border_value=0
                    )
                ice_bool = xr.DataArray(ice_e, coords=lic_bool.coords, dims=lic_bool.dims)
            else:
                ice_bool = lic_bool

            lic_ds.close()

            # --- per-pixel DOY climatology mean & std ---
            winter_ok = None
            cmean_for_replace = None  # (time,lat,lon)

            if getattr(config, "climatology_file", None):
                clim_ds = xr.open_dataset(config.climatology_file)

                clim_mean = self._pick_var(clim_ds, [
                    "lswt_mean_trimmed_345","lswt_mean_trimmed","lswt_mean","climatology"
                ])
                clim_std  = self._pick_var(clim_ds, [
                    "lswt_iqr_345_unbiased","lswt_iqr","lswt_std","lswt_sigma","sigma"
                ])
                if clim_mean is None or clim_std is None:
                    print("IceMaskReplacement: climatology mean/std missing -> skipping ALL ice replacements.")
                    clim_ds.close()
                    zr2d = np.zeros_like(da.isel(time=0).values, dtype=np.uint8)
                    ds["ice_replaced"] = xr.DataArray(
                        np.tile(zr2d, (da.sizes["time"], 1, 1)),
                        dims=("time", "lat", "lon"),
                        coords=da.coords,
                        attrs={"long_name": "Ice replacement flag", "units": "1"},
                    )
                    get_recorder().record_pixel_replacement_daily(self.name, time_days, np.zeros_like(time_days, dtype="int64"))
                    return ds

                # Treat climatology 'time' dimension as DOY; rename then attach coord.
                if "doy" not in clim_mean.dims:
                    if "time" in clim_mean.dims:
                        clim_mean = clim_mean.rename({"time": "doy"})
                    elif "day" in clim_mean.dims:
                        clim_mean = clim_mean.rename({"day": "doy"})
                    clim_mean = clim_mean.assign_coords(doy=np.arange(1, clim_mean.sizes["doy"] + 1))

                if "doy" not in clim_std.dims:
                    if "time" in clim_std.dims:
                        clim_std = clim_std.rename({"time": "doy"})
                    elif "day" in clim_std.dims:
                        clim_std = clim_std.rename({"day": "doy"})
                    clim_std = clim_std.assign_coords(doy=np.arange(1, clim_std.sizes["doy"] + 1))

                # Map LSWT times (int days since 1981-01-01 12:00) → DOY (1..366)
                doys = self._doy_from_days(time_days)  # shape: (time,)
                idx = xr.DataArray(doys, dims=("time",), name="doy")

                # Select per-time, per-pixel climatology on the LSWT timeline
                cmean = clim_mean.sel(doy=idx, method="nearest")
                cstd  = clim_std.sel( doy=idx, method="nearest")

                # Align to LSWT grid and ensure (time,lat,lon)
                cmean = cmean.reindex_like(da, method=None).transpose("time", "lat", "lon")
                cstd  = cstd.reindex_like(da, method=None).transpose("time", "lat", "lon")
                clim_ds.close()

                # Winter criterion (Kelvin): mean - 2*std < 273.15; NaNs → False
                winter_ok = (cmean - 2.0 * cstd) < 273.15
                winter_ok = winter_ok.where(~(np.isnan(cmean) | np.isnan(cstd)), False)

                # Keep mean for anomaly replacement: anomaly = 273.15 - mean
                cmean_for_replace = cmean
            else:
                print("IceMaskReplacement: no climatology file given -> skipping ALL ice replacements.")
                zr2d = np.zeros_like(da.isel(time=0).values, dtype=np.uint8)
                ds["ice_replaced"] = xr.DataArray(
                    np.tile(zr2d, (da.sizes["time"], 1, 1)),
                    dims=("time", "lat", "lon"),
                    coords=da.coords,
                    attrs={"long_name": "Ice replacement flag", "units": "1"},
                )
                get_recorder().record_pixel_replacement_daily(self.name, time_days, np.zeros_like(time_days, dtype="int64"))
                return ds

            # --- final condition ---
            is_nan   = xr.apply_ufunc(np.isnan, da)
            lake_da  = xr.DataArray(lake_mask, dims=("lat", "lon"))
            cond = ice_bool & is_nan & (lake_da > 0)  # LIC ice (shrunk) & no obs & in-lake

            # Apply per-pixel winter criterion (already (time,lat,lon))
            cond = cond & winter_ok

            # Require some obs elsewhere in the lake at that time
            cond_vals = cond.values
            valid_any = (np.isfinite(da.values) & lake_mask[None, :, :]).any(axis=(1, 2))
            for i in range(cond_vals.shape[0]):
                if not valid_any[i]:
                    cond_vals[i, :, :] = False
            cond = xr.DataArray(cond_vals, coords=cond.coords, dims=cond.dims)

            n_replace = int(cond.sum().values)
            if n_replace == 0:
                print("IceMaskReplacement: no pixels met all conditions -> nothing replaced")
                ds["ice_replaced"] = xr.DataArray(
                    np.zeros_like(cond.values, dtype=np.uint8),
                    dims=("time", "lat", "lon"),
                    coords=cond.coords,
                    attrs={"long_name": "Ice replacement flag", "units": "1"},
                )
                # stats: zeros per-time
                replaced_per_time = cond.sum(dim=("lat","lon")).values.astype("int64")
                get_recorder().record_pixel_replacement_daily(self.name, time_days, replaced_per_time)
                return ds

            # --- anomaly-consistent replacement ---
            replacement_anom = 273.15 - cmean_for_replace  # (time,lat,lon)
            ds[var] = xr.where(cond, replacement_anom, ds[var])

            ds["ice_replaced"] = xr.DataArray(
                cond.astype(np.uint8).values,
                dims=("time", "lat", "lon"),
                coords=cond.coords,
                attrs={"long_name": "Ice replacement flag (1=replaced to 0°C as anomaly)", "units": "1"},
            )
            ds.attrs["ice_replacement_mode"] = "anomaly_from_climatology"

            # short human-readable log
            t_idx = np.where(cond.sum(dim=("lat", "lon")).values > 0)[0]
            show = t_idx[:10]
            if show.size > 0:
                ref = datetime.datetime(1981, 1, 1, 12)
                print(f"IceMaskReplacement: replaced {n_replace} pixels across {t_idx.size} time(s)")
                for i in show:
                    n_i = int(cond.isel(time=i).sum().values)
                    when = ref + datetime.timedelta(days=int(time_days[i]))
                    print(f"  - {when.date()} : {n_i} px")
                if t_idx.size > show.size:
                    print(f"  ... and {t_idx.size - show.size} more times")
            else:
                print(f"IceMaskReplacement: replaced {n_replace} pixels")

            # ---- stats: per-time replaced counts ----
            replaced_per_time = cond.sum(dim=("lat","lon")).values.astype("int64")
            get_recorder().record_pixel_replacement_daily(self.name, time_days, replaced_per_time)

            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
