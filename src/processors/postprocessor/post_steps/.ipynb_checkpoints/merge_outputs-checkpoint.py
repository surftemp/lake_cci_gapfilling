# post_steps/merge_outputs.py
from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Optional, Dict
from .base import PostProcessingStep, PostContext


class MergeOutputsStep(PostProcessingStep):
    """
    Merge DINEOF 'temp_filled' back onto the original lake time axis.

    - Reads:
        * ctx.lake_path
        * ctx.dineof_input_path (prepared.nc, carries attrs and the subset time samples)
        * ctx.dineof_output_path (dineof_results.nc with 'temp_filled')
    - Writes:
        * Dataset with variables: time, lat, lon, lakeid (if present in original), temp_filled
        * ctx.input_attrs, ctx.lake_id, ctx.test_id
        * ctx.orig_time_days, ctx.prepared_time_days, ctx.map_prepared_to_orig
    """

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        # Open original lake (for canonical time axis and grid)
        ds_orig = xr.open_dataset(ctx.lake_path)
        if ctx.time_name not in ds_orig.coords:
            raise ValueError(f"Original file missing time coordinate: {ctx.lake_path}")
    
        # Determine dims
        lat_name = "lat" if "lat" in ds_orig.coords else ctx.lat_name
        lon_name = "lon" if "lon" in ds_orig.coords else ctx.lon_name
    
        # original time -> int days
        orig_time_npdt = ds_orig[ctx.time_name].values
        orig_days = self.npdatetime_to_days_since_epoch(orig_time_npdt)
        ctx.orig_time_days = orig_days
    
        # Read prepared input (for attrs + optional mapping metadata)
        with xr.open_dataset(ctx.dineof_input_path) as ds_in:
            ctx.input_attrs = dict(ds_in.attrs)
            ctx.lake_id = int(ds_in.attrs.get("lake_id", -1))
            ctx.test_id = str(ds_in.attrs.get("test_id", ""))
            prep_days = self.npdatetime_to_days_since_epoch(ds_in[ctx.time_name].values)
            ctx.prepared_time_days = prep_days
    
        # DINEOF output (unfiltered or filtered)
        ds_out = xr.open_dataset(ctx.dineof_output_path)
        try:
            if "temp_filled" not in ds_out:
                raise KeyError(f"'temp_filled' not found in {ctx.dineof_output_path}")
    
            # Use the OUTPUT file's time if present; else fallback to prepared (clipped)
            if ctx.time_name in ds_out.coords:
                sub_time_npdt = ds_out[ctx.time_name].values
                sub_days = self.npdatetime_to_days_since_epoch(sub_time_npdt)
            else:
                # For files without time coordinate, we need to handle different cases
                temp_shape = ds_out["temp_filled"].shape[0]
                
                # Check if this might be an interpolated reconstruction (full timeline)
                if temp_shape == len(ctx.full_days):
                    # This is likely the interpolated reconstruction with full daily timeline
                    sub_days = ctx.full_days
                    print(f"[MergeOutputs] Using full daily timeline ({len(sub_days)} days) for interpolated reconstruction")
                else:
                    # Regular case: use prepared timeline truncated to actual data length
                    sub_days = ctx.prepared_time_days[:temp_shape]
                    print(f"[MergeOutputs] Using prepared timeline truncated to {temp_shape} timesteps")
    
            # Mapping dict: day -> index in original full axis
            idx_map: Dict[int, int] = {int(d): i for i, d in enumerate(orig_days)}
    
            # Allocate final cube on full time axis
            t_len = orig_days.size
            lat_len = ds_orig.dims[lat_name]
            lon_len = ds_orig.dims[lon_name]
            out = np.full((t_len, lat_len, lon_len), np.nan, dtype="float32")
    
            temp_out = ds_out["temp_filled"].values  # (t_sub, lat, lon)
    
            # Ensure sub_days and temp_out have matching lengths
            actual_timesteps = min(len(sub_days), temp_out.shape[0])
            if len(sub_days) != temp_out.shape[0]:
                print(f"[MergeOutputs] Length mismatch: sub_days={len(sub_days)}, temp_out.shape[0]={temp_out.shape[0]}")
                print(f"[MergeOutputs] Using first {actual_timesteps} timesteps")
    
            # Fill using the actual available timesteps
            misses = 0
            for i_sub in range(actual_timesteps):
                i_full = idx_map.get(int(sub_days[i_sub]), -1)
                if i_full >= 0:
                    out[i_full, :, :] = temp_out[i_sub, :, :]
                else:
                    misses += 1
            if misses:
                print(f"[MergeOutputs] Skipped {misses} sub timesteps not present in original timeline.")
    
            # Optional: keep prepared→original mapping as metadata (do NOT use it to fill)
            map_idx = np.array([idx_map.get(int(d), -1) for d in prep_days], dtype="int64")
            if np.any(map_idx < 0):
                missing = int(np.sum(map_idx < 0))
                print(f"[MergeOutputs] WARNING: {missing} prepared times not found in original timeline.")
            ctx.map_prepared_to_orig = map_idx
    
            # Build merged dataset
            ds_merged = xr.Dataset()
            ds_merged = ds_merged.assign_coords({
                ctx.time_name: ds_orig[ctx.time_name].values,
                lat_name: ds_orig[lat_name].values,
                lon_name: ds_orig[lon_name].values
            })
            if "lakeid" in ds_orig:
                ds_merged["lakeid"] = ds_orig["lakeid"]
    
            ds_merged["temp_filled"] = xr.DataArray(
                out,
                dims=(ctx.time_name, lat_name, lon_name),
                coords={ctx.time_name: ds_orig[ctx.time_name].values,
                        lat_name: ds_orig[lat_name].values,
                        lon_name: ds_orig[lon_name].values},
                attrs={"comment": "DINEOF-filled anomalies (before trend/climatology add-back)"}
            )
    
            # copy attrs
            if ctx.keep_attrs:
                ds_merged.attrs.update(ds_orig.attrs)
            ds_merged.attrs["prepared_source"] = ctx.dineof_input_path
            ds_merged.attrs["dineof_source"] = ctx.dineof_output_path
            if ctx.test_id is not None:
                ds_merged.attrs["test_id"] = ctx.test_id
            if ctx.lake_id is not None and ctx.lake_id >= 0:
                ds_merged.attrs["lake_id"] = ctx.lake_id
    
            return ds_merged
        finally:
            ds_out.close()
            ds_orig.close()

