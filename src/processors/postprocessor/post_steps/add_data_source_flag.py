# post_steps/add_data_source_flag.py
"""
Add Data Source Flag Step

Creates a `data_source` variable in the output that indicates for each (time, lat, lon):
    0 = true gap (originally missing, not observed)
    1 = observed and seen in training (not withheld as CV)
    2 = CV point (withheld observation used for cross-validation)

This flag enables proper separation of validation metrics:
- Satellite CV should only use flag==2 points
- "Observed" reconstruction quality uses flag==1 points
- Gap-filling quality uses flag==0 points

Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

from __future__ import annotations

import os
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional

from .base import PostProcessingStep, PostContext


# Flag values
FLAG_TRUE_GAP = 0        # Originally missing (true gap)
FLAG_OBSERVED_SEEN = 1   # Observed and used in training
FLAG_CV_WITHHELD = 2     # CV point (withheld observation)


class AddDataSourceFlagStep(PostProcessingStep):
    """
    Add a data_source flag variable indicating the source/status of each pixel.
    
    Reads:
    - prepared.nc: to determine which (time, lat, lon) have valid observations
    - clouds_index.nc: to identify which observations were withheld as CV points
    
    Output variable `data_source`:
        0 = true gap (originally missing)
        1 = observed and seen in training
        2 = CV point (withheld observation)
    """
    
    def __init__(self, lswt_var: str = "lake_surface_water_temperature"):
        super().__init__()
        self.lswt_var = lswt_var
    
    @property
    def name(self) -> str:
        return "AddDataSourceFlag"
    
    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        if ds is None:
            return False
        
        # Check if clouds_index.nc exists
        clouds_index_path = self._get_clouds_index_path(ctx)
        if clouds_index_path is None or not os.path.exists(clouds_index_path):
            print(f"[{self.name}] clouds_index.nc not found; skipping")
            return False
        
        return True
    
    def _get_clouds_index_path(self, ctx: PostContext) -> Optional[str]:
        """
        Locate clouds_index.nc in the prepared directory.
        
        The prepared directory is the parent of dineof_input_path (prepared.nc).
        """
        prepared_dir = os.path.dirname(ctx.dineof_input_path)
        clouds_index_path = os.path.join(prepared_dir, "clouds_index.nc")
        
        if os.path.exists(clouds_index_path):
            return clouds_index_path
        
        # Also check if path is stored in prepared.nc attrs
        try:
            with xr.open_dataset(ctx.dineof_input_path) as ds_prep:
                if "dineof_cv_path" in ds_prep.attrs:
                    cv_path = ds_prep.attrs["dineof_cv_path"]
                    if os.path.exists(cv_path):
                        return cv_path
        except Exception:
            pass
        
        return None
    
    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None
        
        print(f"[{self.name}] Creating data_source flag variable...")
        
        # Get output dimensions
        time_vals = ds["time"].values
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
        n_time = len(time_vals)
        n_lat = len(lat_vals)
        n_lon = len(lon_vals)
        
        # Initialize flag array: default to FLAG_TRUE_GAP (0)
        flag = np.zeros((n_time, n_lat, n_lon), dtype=np.uint8)
        
        # --- Step 1: Load prepared.nc to get observation mask ---
        # Observations are where the LSWT variable is NOT NaN
        with xr.open_dataset(ctx.dineof_input_path) as ds_prep:
            # Get the prepared time indices
            prep_time_days = self.npdatetime_to_days_since_epoch(ds_prep["time"].values)
            
            # Get observation mask from prepared data
            if self.lswt_var in ds_prep:
                prep_data = ds_prep[self.lswt_var].values  # (time, lat, lon)
            else:
                # Fallback to common variable names
                for var_name in ["lake_surface_water_temperature", "temp", "sst"]:
                    if var_name in ds_prep:
                        prep_data = ds_prep[var_name].values
                        break
                else:
                    print(f"[{self.name}] Warning: Could not find LSWT variable in prepared.nc")
                    prep_data = None
            
            if prep_data is not None:
                # Observation mask: where data is valid (not NaN)
                obs_mask_prep = ~np.isnan(prep_data)
                print(f"[{self.name}] Prepared data shape: {prep_data.shape}")
                print(f"[{self.name}] Valid observations in prepared.nc: {obs_mask_prep.sum():,}")
        
        # --- Step 2: Map prepared time to output time ---
        # The output may have a different (expanded) time axis
        out_time_days = self.npdatetime_to_days_since_epoch(time_vals)
        
        # Build mapping: for each prepared time, find corresponding output time index
        out_time_set = {int(d): i for i, d in enumerate(out_time_days)}
        prep_to_out_idx = []
        for d in prep_time_days:
            prep_to_out_idx.append(out_time_set.get(int(d), -1))
        prep_to_out_idx = np.array(prep_to_out_idx, dtype=np.int64)
        
        # --- Step 3: Set FLAG_OBSERVED_SEEN where observations exist ---
        if prep_data is not None:
            for t_prep in range(len(prep_time_days)):
                t_out = prep_to_out_idx[t_prep]
                if t_out >= 0:
                    # Mark observed pixels as FLAG_OBSERVED_SEEN
                    flag[t_out, :, :] = np.where(
                        obs_mask_prep[t_prep, :, :],
                        FLAG_OBSERVED_SEEN,
                        FLAG_TRUE_GAP
                    )
        
        # --- Step 4: Load CV points and mark them as FLAG_CV_WITHHELD ---
        clouds_index_path = self._get_clouds_index_path(ctx)
        if clouds_index_path and os.path.exists(clouds_index_path):
            n_cv_marked = self._mark_cv_points(
                flag, clouds_index_path, prep_time_days, prep_to_out_idx, n_lat, n_lon
            )
            print(f"[{self.name}] Marked {n_cv_marked:,} CV points as withheld")
        
        # --- Step 5: Add flag to dataset ---
        ds["data_source"] = xr.DataArray(
            flag,
            dims=("time", "lat", "lon"),
            coords={"time": time_vals, "lat": lat_vals, "lon": lon_vals},
            attrs={
                "long_name": "data source flag",
                "flag_values": [0, 1, 2],
                "flag_meanings": "true_gap observed_seen cv_withheld",
                "comment": (
                    "0=true gap (originally missing), "
                    "1=observed and seen in training, "
                    "2=CV point (withheld observation)"
                ),
            }
        )
        
        # Summary statistics
        n_gap = (flag == FLAG_TRUE_GAP).sum()
        n_observed = (flag == FLAG_OBSERVED_SEEN).sum()
        n_cv = (flag == FLAG_CV_WITHHELD).sum()
        total = flag.size
        
        print(f"[{self.name}] Data source flag summary:")
        print(f"[{self.name}]   True gaps:      {n_gap:>12,} ({100*n_gap/total:.2f}%)")
        print(f"[{self.name}]   Observed/seen:  {n_observed:>12,} ({100*n_observed/total:.2f}%)")
        print(f"[{self.name}]   CV withheld:    {n_cv:>12,} ({100*n_cv/total:.2f}%)")
        
        return ds
    
    def _mark_cv_points(
        self,
        flag: np.ndarray,
        clouds_index_path: str,
        prep_time_days: np.ndarray,
        prep_to_out_idx: np.ndarray,
        n_lat: int,
        n_lon: int,
    ) -> int:
        """
        Mark CV points in the flag array.
        
        clouds_index.nc structure (from dineof_cvp.jl):
        - clouds_index: (nbpoints, 2) or (2, nbpoints) depending on NetCDF dimension order
          - [*, 0] or [0, *] = spatial index m (1-based)
          - [*, 1] or [1, *] = time index t (1-based)
        - iindex: (indexcount,) = first dimension (i) coordinates (1-based)
        - jindex: (indexcount,) = second dimension (j) coordinates (1-based)
        
        Note: Julia uses column-major order. The prepared.nc data is (time, lat, lon).
        In the Julia code, the loop is: for i=1:imax, for j=1:jmax
        where i iterates over the first spatial dimension and j over the second.
        
        After careful analysis of the code:
        - In Julia's dineof_cvp.jl: SST is loaded as SST[:,:,:] = (dim1, dim2, time)
        - The mask loop: for i=1:imax, for j=1:jmax where mask[i,j]
        - iindex[m] = i (first dimension), jindex[m] = j (second dimension)
        
        In Python with xarray (row-major):
        - prepared.nc is typically (time, lat, lon)
        - lat corresponds to second spatial dimension (j in Julia) via jindex
        - lon corresponds to first spatial dimension (i in Julia) via iindex
        
        Therefore (matching dincae_adapter_in.py):
        - iindex → lon index
        - jindex → lat index
        
        Returns number of CV points marked.
        """
        ds_cv = xr.open_dataset(clouds_index_path)
        
        try:
            clouds_index = ds_cv["clouds_index"].values
            iindex = ds_cv["iindex"].values  # lon coordinates (1-based, from Julia first dim)
            jindex = ds_cv["jindex"].values  # lat coordinates (1-based, from Julia second dim)
            
            # Determine dimension order of clouds_index
            # It should be (nbpoints, 2) but xarray might transpose
            if clouds_index.ndim == 2:
                if clouds_index.shape[1] == 2:
                    # Shape is (nbpoints, 2): [point_idx, 0] = m, [point_idx, 1] = t
                    n_points = clouds_index.shape[0]
                    get_m = lambda p: int(clouds_index[p, 0])
                    get_t = lambda p: int(clouds_index[p, 1])
                elif clouds_index.shape[0] == 2:
                    # Shape is (2, nbpoints): [0, point_idx] = m, [1, point_idx] = t
                    n_points = clouds_index.shape[1]
                    get_m = lambda p: int(clouds_index[0, p])
                    get_t = lambda p: int(clouds_index[1, p])
                else:
                    print(f"[{self.name}] Unexpected clouds_index shape: {clouds_index.shape}")
                    return 0
            else:
                print(f"[{self.name}] Unexpected clouds_index ndim: {clouds_index.ndim}")
                return 0
            
            print(f"[{self.name}] Loading {n_points:,} CV points from clouds_index.nc")
            
            n_marked = 0
            n_skipped_time = 0
            n_skipped_bounds = 0
            n_already_gap = 0
            
            for p in range(n_points):
                # Get spatial index m and time index t (both 1-based from Julia)
                m = get_m(p)
                t_julia = get_t(p)
                
                # Convert time to 0-based prepared index
                t_prep = t_julia - 1
                
                # Check time bounds
                if t_prep < 0 or t_prep >= len(prep_time_days):
                    n_skipped_time += 1
                    continue
                
                # Map prepared time to output time
                t_out = prep_to_out_idx[t_prep]
                if t_out < 0:
                    n_skipped_time += 1
                    continue
                
                # Get spatial coordinates (convert 1-based Julia to 0-based Python)
                # iindex → lon, jindex → lat (matching dincae_adapter_in.py)
                lon_idx = int(iindex[m - 1]) - 1
                lat_idx = int(jindex[m - 1]) - 1
                
                # Check spatial bounds
                if lat_idx < 0 or lat_idx >= n_lat or lon_idx < 0 or lon_idx >= n_lon:
                    n_skipped_bounds += 1
                    continue
                
                # Mark as CV point (should currently be FLAG_OBSERVED_SEEN)
                current_flag = flag[t_out, lat_idx, lon_idx]
                if current_flag == FLAG_TRUE_GAP:
                    # This shouldn't happen - CV points should have been observations
                    n_already_gap += 1
                else:
                    flag[t_out, lat_idx, lon_idx] = FLAG_CV_WITHHELD
                    n_marked += 1
            
            if n_skipped_time > 0:
                print(f"[{self.name}] Skipped {n_skipped_time} CV points (time mismatch)")
            if n_skipped_bounds > 0:
                print(f"[{self.name}] Skipped {n_skipped_bounds} CV points (out of bounds)")
            if n_already_gap > 0:
                print(f"[{self.name}] Warning: {n_already_gap} CV points were already gaps")
            
            return n_marked
            
        finally:
            ds_cv.close()
