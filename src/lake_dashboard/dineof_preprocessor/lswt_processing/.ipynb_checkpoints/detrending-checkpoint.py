"""
Lake-mean robust detrending (Theil–Sen).

Computes a linear trend from the lake *spatial mean* time series using
Theil–Sen (robust) and subtracts it from every pixel.
Only time steps with ≥ coverage_threshold (default 5%) contribute to the fit.

Assumes:
- ds["time"] is numeric (days since 1981-01-01 12:00:00)
- ds["lake_surface_water_temperature"] exists (quality-filtered; typically anomalies after climatology)
- lake mask available as ds["lakeid"] (0/1) or ds.attrs["_mask"] (0/1)
"""

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig
import xarray as xr
import numpy as np
from scipy.stats import theilslopes


class LakeMeanDetrendStep(ProcessingStep):
    def should_apply(self, config: ProcessingConfig) -> bool:
        return bool(getattr(config, "detrend_lake_mean", False))

    @property
    def name(self) -> str:
        return "Lake-mean Theil–Sen Detrending"

    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            var = "lake_surface_water_temperature"
            self.validate_dataset(ds, [var])

            # Mask (prefer current 'lakeid' variable if present)
            if "lakeid" in ds.variables:
                mask = ds["lakeid"].data
            elif "_mask" in ds.attrs:
                mask = np.asarray(ds.attrs["_mask"])
            else:
                raise ValueError("No lake mask found ('lakeid' or '_mask').")
            if mask.ndim != 2 or mask.sum() == 0:
                print("Detrend skipped: empty/invalid mask.")
                return ds

            da = ds[var]
            mask_da = xr.DataArray(mask, dims=("lat", "lon"))
            t = ds["time"].astype("float64")  # days since epoch used in pipeline

            # Coverage per time (fraction of valid lake pixels)
            valid = (~np.isnan(da)) & (mask_da > 0)
            lake_pix = int(mask_da.sum().item())
            if lake_pix == 0:
                print("Detrend skipped: zero lake pixels.")
                return ds

            coverage = (valid.sum(dim=("lat", "lon")) / lake_pix).astype("float64")

            # Select time steps with sufficient coverage
            cov_thr = float(getattr(config, "detrend_coverage_threshold", 0.05) or 0.05)
            sel = coverage >= cov_thr
            if sel.sum().item() == 0:
                print(f"Detrend skipped: no time steps with coverage ≥ {cov_thr:.2f}.")
                return ds

            # Lake-mean (over observed lake pixels only) at selected times
            da_sel = da.where(sel)
            num = (da_sel * mask_da).sum(dim=("lat", "lon"))
            den = valid.where(sel).sum(dim=("lat", "lon")).astype("float64")
            lake_mean = (num / den).where(den > 0).dropna(dim="time")  # 1D

            n_used = lake_mean.sizes.get("time", 0)
            if n_used < 2:
                print("Detrend skipped: fewer than 2 valid time points for fit.")
                return ds

            # Robust linear fit (Theil–Sen) on centered time for stability
            t_sel = t.sel(time=lake_mean["time"]).values
            y_sel = lake_mean.values
            t0 = float(np.median(t_sel))
            x_sel = t_sel - t0
            slope, intercept, _, _ = theilslopes(y_sel, x_sel)  # y ≈ slope*x + intercept
            # logging:
            slope_per_year = slope * 365.25
            print(f"[Detrend] Theil–Sen fit: slope={slope:.6g} per day "
                  f"({slope_per_year:.6g}/yr), intercept={intercept:.6g}")
            
            # Trend span across the dataset (informative)
            tmin = float(t.min().item()); tmax = float(t.max().item())
            trend_span = slope * (tmax - tmin)
            print(f"[Detrend] trend span over period: ~{trend_span:.6g} (units of variable)")
            # end of logging
            
            if not np.isfinite(slope) or not np.isfinite(intercept):
                print("Detrend skipped: non-finite fit coefficients.")
                return ds

            # Build a 1-D DataArray over 'time' and let xarray broadcast
            trend_all = (slope * (t - t0) + intercept).astype("float64")  # dims=('time',)
            print(f"the slope is {slope}")
            detrended = da - trend_all  
            detrended.attrs.update(da.attrs)
            ds[var] = detrended

            # # ===== DEBUG: plot ALL pixels (center highlighted) — before & after detrend =====
            # try:
            #     import os, datetime
            #     import pandas as pd
            #     import matplotlib.pyplot as plt
            
            #     DEBUG_OUTDIR = "/home/users/shaerdan/temporary_plots/"
            #     os.makedirs(DEBUG_OUTDIR, exist_ok=True)
            
            #     def _to_datetimes_from_days(days_float: np.ndarray) -> pd.DatetimeIndex:
            #         ref = datetime.datetime(1981, 1, 1, 12, 0, 0)
            #         return pd.to_datetime([ref + datetime.timedelta(days=float(d)) for d in days_float])
            
            #     def _pick_center_pixel(mask2d: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray):
            #         iy_all, ix_all = np.where(mask2d > 0)
            #         if iy_all.size == 0:
            #             return None
            #         lat_c = np.nanmean(lat2d[iy_all, ix_all])
            #         lon_c = np.nanmean(lon2d[iy_all, ix_all])
            #         d2 = (lat2d[iy_all, ix_all] - lat_c) ** 2 + (lon2d[iy_all, ix_all] - lon_c) ** 2
            #         k = int(np.argmin(d2))
            #         return int(iy_all[k]), int(ix_all[k])
            
            #     def _maybe_to_celsius(arr: np.ndarray) -> np.ndarray:
            #         # Convert from Kelvin only if values are clearly ≳ 200
            #         med = np.nanmedian(arr)
            #         return arr - 273.15 if np.isfinite(med) and med > 200.0 else arr
            
            #     def _plot_all_pixels_time_series(da_in: xr.DataArray,
            #                                     mask2d: np.ndarray,
            #                                     lat: xr.DataArray | np.ndarray,
            #                                     lon: xr.DataArray | np.ndarray,
            #                                     time_days: xr.DataArray,
            #                                     lake_id: int | str,
            #                                     title_suffix: str,
            #                                     outfile: str,
            #                                     y_limits: tuple[float, float] | None = None,
            #                                     trend_series: np.ndarray | None = None):
            #         # lat/lon to 2D
            #         latv = lat.values if isinstance(lat, xr.DataArray) else np.asarray(lat)
            #         lonv = lon.values if isinstance(lon, xr.DataArray) else np.asarray(lon)
            #         if latv.ndim == 1 and lonv.ndim == 1:
            #             lon2d, lat2d = np.meshgrid(lonv, latv)
            #         elif latv.ndim == 2 and lonv.ndim == 2:
            #             lat2d, lon2d = latv, lonv
            #         else:
            #             raise ValueError("lat/lon must both be 1D or both 2D")
            
            #         # center pixel
            #         center = _pick_center_pixel(mask2d, lat2d, lon2d)
            #         if center is None:
            #             print("DEBUG plot skipped: empty lake mask.")
            #             return
            #         iy_c, ix_c = center
            
            #         # stack masked pixels → (time, space)
            #         da_masked = da_in.where(mask2d)
            #         arr_all = da_masked.stack(space=("lat", "lon")).dropna(dim="space", how="all").values  # (T, N)
            #         ts_center = da_in.isel(lat=iy_c, lon=ix_c).values  # (T,)
            
            #         # time → pandas
            #         tdays = np.asarray(time_days.values, dtype=np.float64)
            #         time = _to_datetimes_from_days(tdays)
            
            #         # choose year range and split point (like your style)
            #         years_unique = pd.Index(time.year.unique()).sort_values()
            #         y_start, y_end = int(years_unique.min()), int(years_unique.max())
            #         ym = 1995 if 1995 in years_unique else y_start
            #         y_last = 2022 if 2022 in years_unique else y_end
            #         years = np.arange(ym, y_last + 1)
            #         mid = ym + (len(years) - 1) // 2
            
            #         # indices for split
            #         indt0 = int(np.searchsorted(time.values, np.datetime64(f"{ym}-01-01T00:00:00")))
            #         indt1 = int(np.searchsorted(time.values, np.datetime64(f"{mid + 1}-01-01T00:00:00")))
            #         tt0 = time[indt0:indt1]
            #         tt1 = time[indt1:]
            
            #         # ticks at ~Jan 1 12:00 each year
            #         tick_pos_0, tick_lab_0, tick_pos_1, tick_lab_1 = [], [], [], []
            #         for y in years:
            #             target = pd.Timestamp(y, 1, 1, 12)
            #             i = int(np.argmin(np.abs(time - target)))
            #             (tick_pos_0 if y <= mid else tick_pos_1).append(time[i])
            #             (tick_lab_0 if y <= mid else tick_lab_1).append(str(y))
            
            #         # style params (as in your function)
            #         line_size = 0.2
            #         resolution = 200
            #         dot_size = 3.5
            #         dot_line = 0.2
            
            #         # unit handling
            #         arr_all_plot = _maybe_to_celsius(arr_all)
            #         ts_center_plot = _maybe_to_celsius(ts_center)
            #         y_units = (da_in.attrs.get("units") or "LSWT")
            #         if np.nanmedian(arr_all) > 200.0:
            #             y_units = "LSWT (°C)"  # Kelvin converted to Celsius
            
            #         print(f"Plotting Lake {lake_id} ({title_suffix}): Npix={arr_all.shape[1]}")
            
            #         fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
            #         fig.suptitle(f"Lake {lake_id} — {title_suffix}", fontsize=14)
            
            #         for a in ax:
            #             a.yaxis.grid(True, which='major', linestyle='--', linewidth=0.2)
            #             a.xaxis.grid(True, which='both', linestyle='--', linewidth=0.2)
            
            #         # panel 0
            #         if len(tt0) > 0:
            #             seg0 = slice(indt0, indt1)
            #             if np.any(arr_all_plot[seg0, :] > 400):
            #                 print("Warning: values > 400 in first panel")
            #             for j in range(arr_all_plot.shape[1]):
            #                 L = arr_all_plot[seg0, j]
            #                 maskv = np.isfinite(L) & (L > -1000)
            #                 ax[0].plot(tt0[maskv], L[maskv], '-', color='black', alpha=0.2, lw=line_size)
            #             Lc0 = ts_center_plot[seg0]
            #             maskc0 = np.isfinite(Lc0) & (Lc0 > -1000)
            #             ax[0].plot(tt0[maskc0], Lc0[maskc0], '-', color='red', alpha=1, lw=line_size*2,
            #                        mfc='red', ms=dot_size-2, mew=dot_line, mec='darkred', label='center pixel')
            #             if trend_series is not None:
            #                 trend_plot = _maybe_to_celsius(trend_series)[seg0]
            #                 mt = np.isfinite(trend_plot) & (trend_plot > -1000)
            #                 ax[0].plot(tt0[mt], trend_plot[mt], '-', color='blue', lw=0.8, alpha=0.9, label='trend')     
                            
            #         # panel 1
            #         if len(tt1) > 0:
            #             seg1 = slice(indt1, None)
            #             if np.any(arr_all_plot[seg1, :] > 400):
            #                 print("Warning: values > 400 in second panel")
            #             for j in range(arr_all_plot.shape[1]):
            #                 L = arr_all_plot[seg1, j]
            #                 maskv = np.isfinite(L) & (L > -1000)
            #                 ax[1].plot(tt1[maskv], L[maskv], '-', color='black', alpha=0.2, lw=line_size)
            #             Lc1 = ts_center_plot[seg1]
            #             maskc1 = np.isfinite(Lc1) & (Lc1 > -1000)
            #             ax[1].plot(tt1[maskc1], Lc1[maskc1], '-', color='red', alpha=1, lw=line_size*2,
            #                        mfc='red', ms=dot_size-2, mew=dot_line, mec='darkred')
                        
            #             if trend_series is not None:
            #                 tr1 = _maybe_to_celsius(trend_series)[seg1]
            #                 mt1 = np.isfinite(tr1) & (tr1 > -1000)
            #                 ax[1].plot(tt1[mt1], tr1[mt1], '-', color='blue', lw=0.8, alpha=0.9, label='trend')
            #                 ax[1].legend(fontsize=8) 
                        
            #         ax[1].set_xlabel('Time', fontsize=12)
            #         ax[0].set_ylabel(y_units, fontsize=12)
            #         ax[1].set_ylabel(y_units, fontsize=12)
            #         ax[0].set_xticks(tick_pos_0); ax[0].set_xticklabels(tick_lab_0, fontsize=8)
            #         ax[1].set_xticks(tick_pos_1); ax[1].set_xticklabels(tick_lab_1, fontsize=8)
            #         ax[0].legend(fontsize=8)
            #         # If fixed y-limits provided, enforce on both panels
            #         if y_limits is not None and np.all(np.isfinite(y_limits)):
            #             ax[0].set_ylim(y_limits)
            #             ax[1].set_ylim(y_limits)            
            #         plt.tight_layout()
            #         plt.savefig(outfile, dpi=resolution, bbox_inches='tight')
            #         plt.close(fig)
            #         print(f"Saved: {outfile}")
            
            #     # gather common inputs
            #     lake_id_any = ds.attrs.get("_lake_id_value", ds.attrs.get("lake_id", "unknown"))
            #     mask_np = ds["lakeid"].data if "lakeid" in ds.variables else np.asarray(mask)
            #     lat_da = ds["lat"]; lon_da = ds["lon"]; time_da = ds["time"]
            
            #     # Compute common y-limits (before & after) and the trend series on the full time grid
            #     tdays = np.asarray(time_da.values, dtype=np.float64)
            #     trend_full = (slope * (tdays - t0) + intercept).astype("float64")  # (T,)
            
            #     # Extract stacked arrays for y-range computation
            #     arr_before = da.where(mask_np).stack(space=("lat","lon")).dropna(dim="space", how="all").values
            #     arr_after  = detrended.where(mask_np).stack(space=("lat","lon")).dropna(dim="space", how="all").values
            
            #     # Use the same Kelvin→°C decision for both (based on BEFORE median)
            #     use_celsius = np.isfinite(np.nanmedian(arr_before)) and (np.nanmedian(arr_before) > 200.0)
            #     if use_celsius:
            #         arr_before = arr_before - 273.15
            #         arr_after  = arr_after  - 273.15
            #         trend_full = trend_full - 273.15
            
            #     # Common y-limits with a small padding
            #     mskB = np.isfinite(arr_before) & (arr_before > -1000)
            #     mskA = np.isfinite(arr_after)  & (arr_after  > -1000)
            #     mskT = np.isfinite(trend_full) & (trend_full > -1000)
            #     y_min = np.nanmin([
            #         np.nanmin(arr_before[mskB]) if mskB.any() else np.nan,
            #         np.nanmin(arr_after[mskA])  if mskA.any() else np.nan,
            #         np.nanmin(trend_full[mskT]) if mskT.any() else np.nan,
            #     ])
            #     y_max = np.nanmax([
            #         np.nanmax(arr_before[mskB]) if mskB.any() else np.nan,
            #         np.nanmax(arr_after[mskA])  if mskA.any() else np.nan,
            #         np.nanmax(trend_full[mskT]) if mskT.any() else np.nan,
            #     ])
            #     if np.isfinite(y_min) and np.isfinite(y_max):
            #         pad = 0.02 * max(1.0, (y_max - y_min))
            #         common_ylim = (y_min - pad, y_max + pad)
            #     else:
            #         common_ylim = None
            
            #     # BEFORE
            #     out_before = os.path.join(
            #         DEBUG_OUTDIR, f"LAKE{int(lake_id_any):09d}_allpix_before_detrend.png" if str(lake_id_any).isdigit()
            #         else f"LAKE_{lake_id_any}_allpix_before_detrend.png"
            #     )
            #     _plot_all_pixels_time_series(da_in=da, mask2d=mask_np, lat=lat_da, lon=lon_da,
            #                                  time_days=time_da, lake_id=lake_id_any,
            #                                  title_suffix="ALL pixels — BEFORE detrend",
            #                                  outfile=out_before,
            #                                  y_limits=common_ylim,
            #                                  trend_series=trend_full)
            
            #     # AFTER
            #     out_after = os.path.join(
            #         DEBUG_OUTDIR, f"LAKE{int(lake_id_any):09d}_allpix_after_detrend.png" if str(lake_id_any).isdigit()
            #         else f"LAKE_{lake_id_any}_allpix_after_detrend.png"
            #     )
            #     _plot_all_pixels_time_series(da_in=detrended, mask2d=mask_np, lat=lat_da, lon=lon_da,
            #                                  time_days=time_da, lake_id=lake_id_any,
            #                                  title_suffix="ALL pixels — AFTER detrend",
            #                                  outfile=out_after,
            #                                  y_limits=common_ylim)
            
            # except Exception as _e:
            #     print(f"DEBUG all-pixel plot failed: {_e}")
            # # ===== END DEBUG =====
            

            # Metadata
            ds.attrs["detrend_method"] = "lake-mean Theil–Sen (linear)"
            ds.attrs["detrend_coverage_threshold"] = cov_thr
            ds.attrs["detrend_n_times_used"] = int(n_used)
            ds.attrs["detrend_slope_per_day"] = float(slope)   # units: (var units)/day
            ds.attrs["detrend_intercept"] = float(intercept)
            ds.attrs["detrend_t0_days"] = float(t0)  

            
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
