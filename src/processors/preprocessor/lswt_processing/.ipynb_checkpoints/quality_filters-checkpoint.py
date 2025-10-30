"""
Quality filtering processing steps for LSWT data.

Contains all quality-related filtering operations including:
- Basic quality threshold filtering
- AVHRR Quality Level 3 filtering for post-2007 data
- Z-score based outlier detection and removal
"""

import xarray as xr
import numpy as np
from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig
from .stats import get_recorder


class QualityFilterStep(ProcessingStep):
    """Apply quality threshold filtering to LSWT data"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return config.quality_threshold > 0
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            self.validate_dataset(ds, ["lake_surface_water_temperature", "quality_level"])
            
            quality_level = ds["quality_level"]
            lswt = ds["lake_surface_water_temperature"]
            
            # Log before statistics
            valid_before = int((~np.isnan(lswt)).sum())
            print(f"Before quality filter: {valid_before} valid pixels")
            print(f"Temperature range: {np.nanmin(lswt):.2f} to {np.nanmax(lswt):.2f}")
            
            # Apply quality level filter
            lswt_filtered = lswt.where(quality_level >= config.quality_threshold, np.nan)
            ds["lake_surface_water_temperature"] = lswt_filtered
            
            # Per-time removed counts (stats)
            valid_before_t = (~np.isnan(lswt)).sum(dim=("lat","lon")).values.astype("int64")
            valid_after_t  = (~np.isnan(lswt_filtered)).sum(dim=("lat","lon")).values.astype("int64")
            removed_t = (valid_before_t - valid_after_t).clip(min=0)
            time_days = ds["time"].values.astype("int64")
            get_recorder().record_pixel_filter_daily(self.name, time_days, removed_t)
            
            # Log after statistics
            valid_after = int((~np.isnan(lswt_filtered)).sum())
            removed = valid_before - valid_after
            percentage = (removed / valid_before) * 100 if valid_before > 0 else 0
            print(f"After quality filter: {valid_after} valid pixels")
            print(f"Removed {removed} pixels ({percentage:.2f}%) below quality threshold {config.quality_threshold}")
            
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "Quality Threshold Filtering"


class AVHRRQualityLevel3FilterStep(ProcessingStep):
    """Remove AVHRR Quality Level 3 data for post-2007 observations"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return config.remove_avhrr_ql3
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            self.validate_dataset(ds, ["lake_surface_water_temperature", "quality_level", "obs_instr"])
            
            print("Applying AVHRR QL3 filtering for post-2007 data")
            
            # Keep a copy for per-time stats (before)
            lswt_before = ds["lake_surface_water_temperature"].copy()

            # Convert time coordinates to years
            baseline = np.datetime64('1981-01-01T12:00:00')
            years = np.array([(baseline + np.timedelta64(int(t), 'D')).item().year 
                             for t in ds.coords["time"].data])
            
            obs_instr_array = ds["obs_instr"].data.astype(np.int64)
            quality_array = ds["quality_level"].data
            total_changed = 0
            
            # Process each time step
            for idx, yr in enumerate(years):
                if yr >= 2007:
                    # Check for specific instrument combinations (bits 8 and 16)
                    has_8b = (obs_instr_array[idx] & 8) != 0
                    has_16b = (obs_instr_array[idx] & 16) != 0
                    combo_mask = has_8b | has_16b

                    if np.any(combo_mask):
                        quality_valid = quality_array[idx] == 3
                        adjust_mask = combo_mask & quality_valid
                        
                        # Apply filter to temperature data
                        temp_slice = ds["lake_surface_water_temperature"].isel(time=idx)
                        orig_vals = temp_slice.data
                        temp_adjusted = temp_slice.where(~adjust_mask, np.nan)
                        
                        # Count changes using original method
                        num_changed = int(np.count_nonzero(temp_adjusted.data - orig_vals))
                        if num_changed > 0:
                            print(f"  [Year {yr}, Frame {idx}] Filtered {num_changed} pixels")
                        
                        total_changed += num_changed
                        ds["lake_surface_water_temperature"].data[idx, :, :] = temp_adjusted.data
                        
            print(f"Total pixels filtered across all post-2007 frames: {total_changed}")

            # Per-time removed counts (stats)
            lswt_after = ds["lake_surface_water_temperature"]
            valid_before_t = (~np.isnan(lswt_before)).sum(dim=("lat","lon")).values.astype("int64")
            valid_after_t  = (~np.isnan(lswt_after)).sum(dim=("lat","lon")).values.astype("int64")
            removed_t = (valid_before_t - valid_after_t).clip(min=0)
            time_days = ds["time"].values.astype("int64")
            get_recorder().record_pixel_filter_daily(self.name, time_days, removed_t)

            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "AVHRR Quality Level 3 Filtering"


class ZScoreFilterStep(ProcessingStep):
    """
    Single-mode outlier filtering:
      - outlier_mode='zscore'   : mean/std z-score
      - outlier_mode='robust'   : median/MAD robust z-score (MAD*1.4826)
      - outlier_mode='quantile' : clip outside [quantile_low, quantile_high]

    All filtered values -> NaN (keeps availability logic consistent).
    """

    def __init__(self, default_z_threshold: float = 2.5):
        self.default_z_threshold = default_z_threshold

    @property
    def name(self) -> str:
        return "Outlier Filtering"

    def _infer_mode(self, config: ProcessingConfig) -> str:
        # Back-compat: if old flag is set and no explicit mode, treat as 'zscore'
        mode = (getattr(config, "outlier_mode", None) or "").lower()
        if not mode:
            if getattr(config, "apply_zscore_filter", False):
                mode = "zscore"
            else:
                mode = "off"
        if mode not in ("off", "zscore", "robust", "quantile"):
            print(f"Unknown outlier_mode='{mode}', forcing 'off'")
            mode = "off"
        return mode

    def should_apply(self, config: ProcessingConfig) -> bool:
        mode = getattr(config, "outlier_mode", None)
        if mode is None:
            return False
        mode = str(mode).lower()
        return mode not in ("off", "none", "false", "0")

    def _valid_mask(self, da: xr.DataArray, fillvalue: float) -> xr.DataArray:
        m = ~xr.apply_ufunc(np.isnan, da)
        if np.isfinite(fillvalue):
            m = m & (da != fillvalue)
        return m

    def _center_scale(self, valid_da: xr.DataArray, mode: str) -> tuple[xr.DataArray, xr.DataArray]:
        if mode == "robust":
            center = valid_da.median()
            mad = (valid_da - center).abs().median()
            scale = mad * 1.4826
            print("Z-score mode=robust (center=median, scale=1.4826*MAD)")
        else: # standard zscore
            center = valid_da.mean()
            scale  = valid_da.std()
            print("Z-score mode=zscore (center=mean, scale=std)")
        # guard tiny scale
        scale = xr.where(scale < 1e-12, 1e-12, scale)
        return center, scale

    def _quantile_bounds(self, valid_da: xr.DataArray, qlow: float, qhigh: float) -> tuple[float, float]:
        stacked = valid_da.stack(sample=("time", "lat", "lon")).dropna(dim="sample", how="all")
        qs = stacked.quantile([qlow, qhigh], dim="sample")
        return float(qs.sel(quantile=qlow)), float(qs.sel(quantile=qhigh))

    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            var = "lake_surface_water_temperature"
            self.validate_dataset(ds, [var])
            da = ds[var]

            mode = self._infer_mode(config)
            fillvalue = getattr(config, "fillvalue", np.nan)
            valid_mask = self._valid_mask(da, fillvalue)
            valid_da = da.where(valid_mask)

            # capture pre counts for per-time stats
            valid_before_t = (~np.isnan(da)).sum(dim=("lat","lon")).values.astype("int64")

            if mode in ("zscore", "robust"):
                zthr = float(getattr(config, "z_threshold", self.default_z_threshold) or self.default_z_threshold)
                center, scale = self._center_scale(valid_da, mode)
                z = xr.apply_ufunc(np.abs, (da - center) / scale)
                out = (z > zthr) & valid_mask
                n_out = int(out.sum().values); n_tot = int(valid_mask.sum().values)
                print(f"{mode} z-score thr={zthr}: removed {n_out}/{n_tot} ({100*n_out/max(n_tot,1):.2f}%)")
                ds[var] = da.where(~out, np.nan)

                # per-time stats
                valid_after_t  = (~np.isnan(ds[var])).sum(dim=("lat","lon")).values.astype("int64")
                removed_t = (valid_before_t - valid_after_t).clip(min=0)
                time_days = ds["time"].values.astype("int64")
                get_recorder().record_pixel_filter_daily(self.name, time_days, removed_t)

                n_out = int(out.sum().values)
                n_tot = int(valid_mask.sum().values)
                print(f"[Outlier] removed={n_out:,}/{n_tot:,} ({(100*n_out/max(n_tot,1)):.2f}%), "
                      f"remaining_valid={int((~np.isnan(ds[var])).sum().values):,}")                
                return ds

            if mode == "quantile":
                qlow  = float(getattr(config, "quantile_low", 0.05) or 0.05)
                qhigh = float(getattr(config, "quantile_high", 0.95) or 0.95)
                if not (0.0 < qlow < qhigh < 1.0):
                    raise ValueError(f"quantile_low/high must satisfy 0<low<high<1 (got {qlow}, {qhigh})")
                ql, qh = self._quantile_bounds(valid_da, qlow, qhigh)
                out = ((da < ql) | (da > qh)) & valid_mask
                n_out = int(out.sum().values); n_tot = int(valid_mask.sum().values)
                print(f"quantile [{qlow:.2f},{qhigh:.2f}] -> bounds [{ql:.3f},{qh:.3f}]: "
                      f"removed {n_out}/{n_tot} ({100*n_out/max(n_tot,1):.2f}%)")
                ds[var] = da.where(~out, np.nan)

                # per-time stats
                valid_after_t  = (~np.isnan(ds[var])).sum(dim=("lat","lon")).values.astype("int64")
                removed_t = (valid_before_t - valid_after_t).clip(min=0)
                time_days = ds["time"].values.astype("int64")
                get_recorder().record_pixel_filter_daily(self.name, time_days, removed_t)

                n_out = int(out.sum().values)
                n_tot = int(valid_mask.sum().values)
                print(f"[Outlier] removed={n_out:,}/{n_tot:,} ({(100*n_out/max(n_tot,1)):.2f}%), "
                      f"remaining_valid={int((~np.isnan(ds[var])).sum().values):,}")                   
                return ds

            # mode == 'off'
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
