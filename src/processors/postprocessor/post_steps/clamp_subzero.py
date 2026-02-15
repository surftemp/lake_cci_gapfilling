# post_steps/clamp_subzero.py
"""
Clamp sub-zero LSWT values to 0°C (or 273.15 K if output is in Kelvin).

Placed after AddBackClimatologyStep in the pipeline so that trend + climatology
have already been restored and units are final.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Optional
from .base import PostProcessingStep, PostContext


class ClampSubZeroStep(PostProcessingStep):
    """
    Clip temp_filled so that no pixel has LSWT below the freezing point.

    The threshold depends on ctx.output_units:
      - "celsius"  ->  0.0
      - "kelvin"   ->  273.15
    """

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and "temp_filled" in ds

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None

        if ctx.output_units == "kelvin":
            threshold = 273.15
        else:
            threshold = 0.0

        arr = ds["temp_filled"].values
        n_subzero = int(np.nansum(arr < threshold))
        total_valid = int(np.nansum(np.isfinite(arr)))

        if n_subzero > 0:
            ds["temp_filled"] = ds["temp_filled"].clip(min=threshold)
            pct = 100.0 * n_subzero / total_valid if total_valid > 0 else 0
            print(f"[ClampSubZero] Clamped {n_subzero:,} values below {threshold} "
                  f"({pct:.2f}% of {total_valid:,} valid pixels)")
        else:
            print(f"[ClampSubZero] No sub-zero values found ({total_valid:,} valid pixels)")

        ds.attrs["subzero_clamped"] = 1
        ds.attrs["subzero_threshold"] = threshold
        return ds
