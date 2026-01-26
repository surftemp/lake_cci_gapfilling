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
        
        # Initialize flag array: default to FLAG_NOT_RECONSTRUCTED (255)
        FLAG_NOT_RECONSTRUCTED = 255
        flag = np.full((n_time, n_lat, n_lon), FLAG_NOT_RECONSTRUCTED, dtype=np.uint8)
        
        # Get the reconstruction mask from temp_filled
        # Only pixels where temp_filled is NOT NaN were actually reconstructed
        if "temp_filled" not in ds:
            print(f"[{self.name}] Warning: temp_filled not found in output, cannot create flag")
            return ds
        
        temp_filled = ds["temp_filled"].values  # (time, lat, lon)
        recon_mask = ~np.isnan(temp_filled)  # True where reconstruction exists
        n_reconstructed = recon_mask.sum()
        
        print(f"[{self.name}] Output shape: {temp_filled.shape}")
        print(f"[{self.name}] Reconstructed pixels (temp_filled not NaN): {n_reconstructed:,}")
        
        # --- Step 1: Load prepared.nc to get observation mask ---
        with xr.open_dataset(ctx.dineof_input_path) as ds_prep:
            prep_time_days = self.npdatetime_to_days_since_epoch(ds_prep["time"].values)
            
            # Get observation mask from prepared data
            prep_data = None
            for var_name in [self.lswt_var, "lake_surface_water_temperature", "temp", "sst"]:
                if var_name in ds_prep:
                    prep_data = ds_prep[var_name].values  # (time, lat, lon)
                    break
            
            if prep_data is None:
                print(f"[{self.name}] Warning: Could not find LSWT variable in prepared.nc")
                return ds
            
            obs_mask_prep = ~np.isnan(prep_data)
            print(f"[{self.name}] Prepared data shape: {prep_data.shape}")
            print(f"[{self.name}] Valid observations in prepared.nc: {obs_mask_prep.sum():,}")
        
        # --- Step 2: Build time mappings ---
        out_time_days = self.npdatetime_to_days_since_epoch(time_vals)
        
        # Map: output time index -> prepared time index (for looking up observations)
        prep_day_to_idx = {int(d): i for i, d in enumerate(prep_time_days)}
        out_to_prep_idx = []
        for d in out_time_days:
            out_to_prep_idx.append(prep_day_to_idx.get(int(d), -1))
        out_to_prep_idx = np.array(out_to_prep_idx, dtype=np.int64)
        
        # Map: prepared time index -> output time index (for CV marking)
        out_day_to_idx = {int(d): i for i, d in enumerate(out_time_days)}
        prep_to_out_idx = []
        for d in prep_time_days:
            prep_to_out_idx.append(out_day_to_idx.get(int(d), -1))
        prep_to_out_idx = np.array(prep_to_out_idx, dtype=np.int64)
        
        # --- Step 3: For each reconstructed pixel, determine its source ---
        # Iterate only where reconstruction exists
        recon_indices = np.argwhere(recon_mask)  # (N, 3) array of [t, lat, lon]
        
        print(f"[{self.name}] Processing {len(recon_indices):,} reconstructed pixels...")
        
        for idx in recon_indices:
            t_out, lat_idx, lon_idx = idx[0], idx[1], idx[2]
            
            # Map to prepared time
            t_prep = out_to_prep_idx[t_out]
            if t_prep < 0:
                # Timestep not in prepared.nc - shouldn't happen for reconstructed pixels
                # but mark as gap if it does
                flag[t_out, lat_idx, lon_idx] = FLAG_TRUE_GAP
                continue
            
            # Check if this pixel had an observation in prepared.nc
            if obs_mask_prep[t_prep, lat_idx, lon_idx]:
                # Had observation - mark as OBSERVED_SEEN for now
                # Will be overwritten to CV_WITHHELD later if it's a CV point
                flag[t_out, lat_idx, lon_idx] = FLAG_OBSERVED_SEEN
            else:
                # No observation - true gap
                flag[t_out, lat_idx, lon_idx] = FLAG_TRUE_GAP
        
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
                "flag_values": [0, 1, 2, 255],
                "flag_meanings": "true_gap observed_seen cv_withheld not_reconstructed",
                "comment": (
                    "0=true gap (originally missing, gapfilled by method), "
                    "1=observed and seen in training (reconstruction of observation), "
                    "2=CV point (withheld observation, reconstruction of CV point), "
                    "255=not reconstructed (temp_filled is NaN at this pixel)"
                ),
            }
        )
        
        # Summary statistics - only count within reconstructed pixels (temp_filled not NaN)
        n_not_recon = int((flag == 255).sum())
        n_gap = int((flag == FLAG_TRUE_GAP).sum())
        n_observed = int((flag == FLAG_OBSERVED_SEEN).sum())
        n_cv = int((flag == FLAG_CV_WITHHELD).sum())
        n_flagged_recon = n_gap + n_observed + n_cv
        
        # Verification: flag counts should match recon_mask
        n_temp_filled_valid = int(recon_mask.sum())
        if n_flagged_recon != n_temp_filled_valid:
            print(f"[{self.name}] WARNING: Flag count mismatch!")
            print(f"[{self.name}]   temp_filled valid pixels: {n_temp_filled_valid:,}")
            print(f"[{self.name}]   flag 0+1+2 pixels: {n_flagged_recon:,}")
            print(f"[{self.name}]   Difference: {n_flagged_recon - n_temp_filled_valid:,}")
        
        print(f"[{self.name}] Data source flag summary:")
        print(f"[{self.name}]   temp_filled valid (recon_mask): {n_temp_filled_valid:>12,}")
        print(f"[{self.name}]   Not reconstructed (255): {n_not_recon:>12,} (temp_filled is NaN)")
        print(f"[{self.name}]   --- Within reconstructed pixels ({n_flagged_recon:,} flagged as 0/1/2) ---")
        print(f"[{self.name}]   True gaps (0):     {n_gap:>12,} ({100*n_gap/n_flagged_recon:.2f}%)")
        print(f"[{self.name}]   Observed/seen (1): {n_observed:>12,} ({100*n_observed/n_flagged_recon:.2f}%)")
        print(f"[{self.name}]   CV withheld (2):   {n_cv:>12,} ({100*n_cv/n_flagged_recon:.2f}%)")
        
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
            n_skipped_not_recon = 0
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
                if current_flag == 255:
                    # Not reconstructed (temp_filled is NaN here) - skip
                    n_skipped_not_recon += 1
                    continue
                elif current_flag == FLAG_TRUE_GAP:
                    # This shouldn't happen - CV points should have been observations
                    n_already_gap += 1
                elif current_flag == FLAG_OBSERVED_SEEN:
                    flag[t_out, lat_idx, lon_idx] = FLAG_CV_WITHHELD
                    n_marked += 1
                else:
                    # Already marked as CV (duplicate?)
                    pass
            
            if n_skipped_time > 0:
                print(f"[{self.name}] Skipped {n_skipped_time} CV points (time mismatch)")
            if n_skipped_bounds > 0:
                print(f"[{self.name}] Skipped {n_skipped_bounds} CV points (spatial out of bounds)")
            if n_skipped_not_recon > 0:
                print(f"[{self.name}] Skipped {n_skipped_not_recon} CV points (not reconstructed, flag=255)")
            if n_already_gap > 0:
                print(f"[{self.name}] Warning: {n_already_gap} CV points were already gaps")
            
            return n_marked
            
        finally:
            ds_cv.close()
