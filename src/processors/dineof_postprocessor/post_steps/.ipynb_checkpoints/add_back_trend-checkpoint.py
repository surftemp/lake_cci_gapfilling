# post_steps/add_back_trend.py
from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Optional
from .base import PostProcessingStep, PostContext


class AddBackTrendStep(PostProcessingStep):
    """
    Add back the detrended lake-mean trend using attrs recorded by the preprocessor.

    Expected attrs in prepared.nc (stored in ctx.input_attrs):
      - detrend_method: e.g., "lake_mean_theil_sen" or "none"
      - detrend_slope_per_day: float (K/day for anomalies)
      - detrend_intercept: float (K at reference x; see t0)
      - detrend_t0_days: int or float (reference days since epoch used for fit) [optional]

    We compute trend(t) = slope * (days - t0) + intercept   (if t0 present)
                      or = slope * days + intercept         (if t0 missing)
    Then: temp_filled += trend(t)
    """

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        if ds is None:
            return False
        ia = ctx.input_attrs or {}
        method = str(ia.get("detrend_method", "none")).lower()
        if method in ("", "none", "false", "0", "off"):
            return False
        return "temp_filled" in ds

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None

        ia = ctx.input_attrs or {}
        slope = ia.get("detrend_slope_per_day", None)
        intercept = ia.get("detrend_intercept", None)
        t0 = ia.get("detrend_t0_days", None)

        if slope is None or intercept is None:
            print("[AddBackTrend] Missing slope/intercept in attrs; skipping trend add-back.")
            return ds

        # Compute days since epoch for ds.time
        time_npdt = ds["time"].values
        days = self.npdatetime_to_days_since_epoch(time_npdt).astype("float64")
        if t0 is not None:
            x = days - float(t0)
        else:
            x = days

        trend = float(slope) * x + float(intercept)  # shape (time,)
        # Broadcast to 3D
        trend3d = xr.DataArray(trend, dims=("time",), coords={"time": ds["time"].values})
        trend3d = trend3d.broadcast_like(ds["temp_filled"])

        ds["temp_filled"] = (ds["temp_filled"] + trend3d).astype("float32")
        ds.attrs["trend_added_back"] = 1
        ds.attrs["trend_model"] = "linear"
        ds.attrs["trend_params"] = f"slope_per_day={slope}, intercept={intercept}, t0_days={t0}"

        return ds
