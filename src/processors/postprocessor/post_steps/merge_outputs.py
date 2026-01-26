# post_steps/merge_outputs.py
from __future__ import annotations

import os
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
    
            # Use the OUTPUT file's time if present; else fallback to EOF file or prepared
            if ctx.time_name in ds_out.coords:
                sub_time_npdt = ds_out[ctx.time_name].values
                sub_days = self.npdatetime_to_days_since_epoch(sub_time_npdt)
            else:
                # For files without time coordinate, we need to handle different cases
                temp_shape = ds_out["temp_filled"].shape[0]
                
                # Check if this might be an interpolated reconstruction (full timeline)
                if ctx.full_days is not None and temp_shape == len(ctx.full_days):
                    # This is likely the interpolated reconstruction with full daily timeline
                    sub_days = ctx.full_days
                    print(f"[MergeOutputs] Using full daily timeline ({len(sub_days)} days) for interpolated reconstruction")
                elif temp_shape == len(ctx.prepared_time_days):
                    # Exact match with prepared - use prepared times directly
                    sub_days = ctx.prepared_time_days
                    print(f"[MergeOutputs] Using prepared timeline ({len(sub_days)} timesteps)")
                else:
                    # Length mismatch - this is likely a filtered reconstruction
                    # Try to find the corresponding EOF file to get correct times
                    sub_days = self._get_times_from_eof_file(ctx, temp_shape)
                    if sub_days is not None:
                        print(f"[MergeOutputs] Using EOF file timeline ({len(sub_days)} timesteps)")
                    else:
                        # Last resort: truncate prepared (this was the old buggy behavior)
                        sub_days = ctx.prepared_time_days[:temp_shape]
                        print(f"[MergeOutputs] WARNING: Falling back to truncated prepared timeline ({temp_shape} timesteps)")
                        print(f"[MergeOutputs] This may cause incorrect time mapping for filtered reconstructions!")
    
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
    
            # Apply lake mask: ensure only lake pixels have values
            # This is critical for DINCAE which fills the entire cropped rectangle
            # including buffer pixels outside the lake mask
            # Use lakeid from prepared.nc (same mask DINEOF uses) for consistency
            lake_mask_applied = False
            with xr.open_dataset(ctx.dineof_input_path) as ds_prep:
                if "lakeid" in ds_prep:
                    lakeid_raw = ds_prep["lakeid"].values
                    lake_mask_2d = np.isfinite(lakeid_raw) & (lakeid_raw != 0)  # True for lake pixels
                    non_lake_2d = ~lake_mask_2d
                    
                    # Count how many non-lake pixels had values (for logging)
                    n_non_lake_with_value = 0
                    for t in range(t_len):
                        frame = out[t, :, :]
                        n_non_lake_with_value += int(np.sum(~np.isnan(frame) & non_lake_2d))
                        out[t, non_lake_2d] = np.nan
                    
                    lake_mask_applied = True
                    if n_non_lake_with_value > 0:
                        print(f"[MergeOutputs] Applied lake mask from prepared.nc: removed {n_non_lake_with_value:,} values outside lake mask")
                    else:
                        print(f"[MergeOutputs] Lake mask applied (no non-lake values found)")
                else:
                    print(f"[MergeOutputs] WARNING: No 'lakeid' in prepared.nc, cannot apply lake mask")
            
            if not lake_mask_applied:
                # Fallback to original lake file
                if "lakeid" in ds_orig:
                    lakeid_raw = ds_orig["lakeid"].values
                    lake_mask_2d = np.isfinite(lakeid_raw) & (lakeid_raw != 0)
                    non_lake_2d = ~lake_mask_2d
                    
                    n_non_lake_with_value = 0
                    for t in range(t_len):
                        frame = out[t, :, :]
                        n_non_lake_with_value += int(np.sum(~np.isnan(frame) & non_lake_2d))
                        out[t, non_lake_2d] = np.nan
                    
                    if n_non_lake_with_value > 0:
                        print(f"[MergeOutputs] Applied lake mask from original file: removed {n_non_lake_with_value:,} values outside lake mask")
                else:
                    print(f"[MergeOutputs] WARNING: No 'lakeid' found in any source file")
    
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
                lakeid_raw = ds_orig["lakeid"].values
                lakeid_binary = np.where(np.isfinite(lakeid_raw) & (lakeid_raw != 0), 1, 0).astype(np.int32)
                ds_merged["lakeid"] = xr.DataArray(
                    lakeid_binary,
                    dims=ds_orig["lakeid"].dims,
                    coords=ds_orig["lakeid"].coords,
                    attrs={"long_name": "lake mask", "flag_values": "0=land, 1=lake"}
                )
    
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

    def _get_times_from_eof_file(self, ctx: PostContext, expected_length: int) -> Optional[np.ndarray]:
        """
        Try to find the corresponding EOF file and extract its time coordinate.
        
        This handles the case where dineof_results_eof_filtered.nc was created from
        eofs_filtered.nc, which has fewer timesteps than prepared.nc due to outlier filtering.
        
        Returns the time coordinate as days since epoch, or None if not found.
        """
        base_dir = os.path.dirname(ctx.dineof_output_path)
        output_basename = os.path.basename(ctx.dineof_output_path)
        
        # Determine which EOF file to look for based on the output filename
        if "eof_filtered" in output_basename:
            # For filtered results, look for eofs_filtered.nc
            eof_candidates = [
                os.path.join(base_dir, "eofs_filtered.nc"),
            ]
        elif "eof_interp" in output_basename:
            # For interpolated results, look for eofs_interpolated.nc or eofs_filtered_interpolated.nc
            eof_candidates = [
                os.path.join(base_dir, "eofs_interpolated.nc"),
                os.path.join(base_dir, "eofs_filtered_interpolated.nc"),
            ]
        else:
            # For regular results, look for eofs.nc
            eof_candidates = [
                os.path.join(base_dir, "eofs.nc"),
            ]
        
        for eof_path in eof_candidates:
            if os.path.exists(eof_path):
                try:
                    with xr.open_dataset(eof_path) as eof_ds:
                        # Check for time coordinate (might be named 'time' or stored differently)
                        if 'time' in eof_ds.coords:
                            eof_times = eof_ds['time'].values
                            eof_days = self.npdatetime_to_days_since_epoch(eof_times)
                            
                            if len(eof_days) == expected_length:
                                print(f"[MergeOutputs] Found matching EOF times in {os.path.basename(eof_path)}")
                                return eof_days
                            else:
                                print(f"[MergeOutputs] EOF file {os.path.basename(eof_path)} has {len(eof_days)} times, expected {expected_length}")
                except Exception as e:
                    print(f"[MergeOutputs] Could not read EOF file {eof_path}: {e}")
        
        return None