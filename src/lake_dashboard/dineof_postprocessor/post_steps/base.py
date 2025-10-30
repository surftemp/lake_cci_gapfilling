# post_steps/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import xarray as xr
import os

# Epoch consistent with the rest of the pipeline
EPOCH_DT64_NS = np.datetime64("1981-01-01T12:00:00", "ns")
EPOCH_NS = EPOCH_DT64_NS.astype("int64")  # integer ns since Unix epoch

@dataclass
class PostContext:
    lake_path: str
    dineof_input_path: str
    dineof_output_path: str
    output_path: str
    output_html_folder: Optional[str]
    climatology_path: Optional[str]
    output_units: str = "kelvin"  # "kelvin" or "celsius"
    keep_attrs: bool = True
    # <- add this to align with your post_process.py caller
    experiment_config_path: Optional[str] = None

    # Filled by steps (shared state)
    input_attrs: dict | None = None
    lake_id: Optional[int] = None
    test_id: Optional[str] = None

    # cached time mappings
    orig_time_days: Optional[np.ndarray] = None        # original time as days since epoch
    prepared_time_days: Optional[np.ndarray] = None    # prepared time as days since epoch
    map_prepared_to_orig: Optional[np.ndarray] = None  # len(prepared_time_days) -> indices in original

    # dims cache
    lat_name: str = "lat"
    lon_name: str = "lon"
    time_name: str = "time"

    # timeline references
    time_units: str = "days since 1981-01-01 12:00:00"
    time_start_days: Optional[int] = None
    time_end_days: Optional[int] = None
    full_days: Optional[np.ndarray] = None    


class PostProcessingStep:
    """Base class for pipeline steps."""
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return True

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        raise NotImplementedError("apply must be implemented by subclasses")

    # ---------- time helpers (robust to int days or datetime64) ----------
    @staticmethod
    def ensure_npdatetime64(arr) -> np.ndarray:
        """
        Return a datetime64[ns] array for any input:
          - If numeric, treat as 'days since 1981-01-01 12:00' and convert.
          - If datetime64-like, cast to ns precision.
        """
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.number):
            return (EPOCH_DT64_NS + a.astype("timedelta64[D]")).astype("datetime64[ns]")
        return a.astype("datetime64[ns]")

    @staticmethod
    def npdatetime_to_days_since_epoch(arr) -> np.ndarray:
        """
        Convert any time array to int64 'days since 1981-01-01 12:00':
          - If arr is numeric already, just cast to int64.
          - If arr is datetime64-like, compute delta days to EPOCH.
        """
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.number):
            return a.astype(np.int64)
        dt = a.astype("datetime64[ns]")
        delta_ns = dt.astype("int64") - EPOCH_NS
        return (delta_ns // 86_400_000_000_000).astype(np.int64)  # 1 day = 86_400 * 1e9 ns

    @staticmethod
    def days_since_epoch_to_npdatetime(days: np.ndarray) -> np.ndarray:
        """Convert int days since EPOCH to numpy datetime64[ns]."""
        d = np.asarray(days, dtype="int64")
        return EPOCH_DT64_NS + d.astype("timedelta64[D]")

    @staticmethod
    def doy_from_any_time(arr) -> np.ndarray:
        """
        Compute day-of-year (1..366) for either:
          - numeric 'days since 1981-01-01 12:00'
          - datetime64 array
        Fully vectorized with datetime64 arithmetic.
        """
        dt = PostProcessingStep.ensure_npdatetime64(arr)
        return ((dt.astype("datetime64[D]") - dt.astype("datetime64[Y]"))
                .astype("timedelta64[D]").astype(np.int64) + 1)

    # Back-compat for any step that still calls doy_from_npdatetime(...)
    @staticmethod
    def doy_from_npdatetime(arr) -> np.ndarray:
        return PostProcessingStep.doy_from_any_time(arr)

    # ---------- misc helpers ----------
    @staticmethod
    def safe_mkdir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def pick_first_var(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[xr.DataArray]:
        for name in candidates:
            if name in ds.variables:
                return ds[name]
        return None
