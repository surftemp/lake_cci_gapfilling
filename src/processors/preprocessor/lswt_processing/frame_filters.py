"""
Frame filtering processing steps for LSWT data.

Contains steps for:
- Empty frame removal (frames with no valid data)
- Sparse frame removal (frames below completeness threshold)
"""

import xarray as xr
import numpy as np
from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig
from .stats import get_recorder


class EmptyFrameRemovalStep(ProcessingStep):
    """Remove empty frames or frames below specified completeness threshold"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return config.remove_empty or config.remove_threshold > 0.0
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            print("Analyzing frames for removal...")
            
            self.validate_dataset(ds, ["lake_surface_water_temperature"])            
            # Get lake mask and temperature data
            mask = ds["lakeid"].data if "lakeid" in ds.variables else np.asarray(ds.attrs["_mask"]).astype(int)
            lswt = ds["lake_surface_water_temperature"]
            converted_dates = ds.coords["time"].data
            
            # Calculate total lake pixels
            frame_pixels = np.count_nonzero(mask)
            print(f"Total lake pixels in mask: {frame_pixels}")
            
            if frame_pixels == 0:
                print("Warning: No lake pixels found in mask!")
                return ds
            
            # Analyze each frame
            time_select = []
            empty_frames = 0
            sparse_frames = 0
            frame_stats = []
            
            print(f"Analyzing {len(converted_dates)} frames...")
            
            for time_index in range(len(converted_dates)):
                # Get temperature data for this frame
                frame = lswt[time_index, :, :].data
                
                # Legacy behavior: counts nonzero in (frame * mask), NaNs count as nonzero
                filled_pixels = np.count_nonzero(np.isfinite(frame) & (mask > 0))
                
                # Calculate fill ratio
                fill_ratio = filled_pixels / frame_pixels if frame_pixels > 0 else 0
                
                # Store frame statistics
                frame_stats.append({
                    'index': time_index,
                    'filled_pixels': filled_pixels,
                    'fill_ratio': fill_ratio,
                    'keep': True
                })
                
                # Determine if frame should be removed
                remove_frame = False
                removal_reason = ""
                
                if config.remove_empty and filled_pixels == 0:
                    remove_frame = True
                    removal_reason = "empty"
                    empty_frames += 1
                elif config.remove_threshold > 0.0 and fill_ratio < config.remove_threshold:
                    remove_frame = True
                    removal_reason = f"sparse (fill ratio: {fill_ratio:.3f} < {config.remove_threshold:.3f})"
                    sparse_frames += 1
                
                if remove_frame:
                    frame_stats[-1]['keep'] = False
                    if time_index < 10 or len(converted_dates) - time_index <= 10:  # Log first/last 10
                        print(f"  Frame {time_index}: REMOVE - {removal_reason}")
                else:
                    time_select.append(time_index)
                    if time_index < 5:  # Log first few kept frames
                        print(f"  Frame {time_index}: KEEP - {filled_pixels} pixels ({fill_ratio:.3f} fill ratio)")
            
            # Calculate overall statistics
            total_frames = len(converted_dates)
            kept_frames = len(time_select)
            removed_frames = total_frames - kept_frames
            removal_percentage = (removed_frames / total_frames) * 100 if total_frames > 0 else 0
            
            print(f"\nFrame filtering results:")
            print(f"  - Total frames: {total_frames}")
            print(f"  - Frames kept: {kept_frames}")
            print(f"  - Frames removed: {removed_frames} ({removal_percentage:.1f}%)")
            if config.remove_empty:
                print(f"    * Empty frames: {empty_frames}")
            if config.remove_threshold > 0.0:
                print(f"    * Sparse frames (< {config.remove_threshold:.1f} fill): {sparse_frames}")
            
            # Record per-step removed timesteps (stats)
            all_days = ds["time"].values.astype("int64")
            removed_idx = [i for i in range(len(all_days)) if i not in time_select]
            removed_days = [int(all_days[i]) for i in removed_idx]
            get_recorder().record_timesteps_removed(self.name, removed_days)
            
            # Apply frame selection to dataset
            if removed_frames > 0:
                print(f"Applying frame selection...")
                ds = ds.isel(time=time_select)
                
                # Verify the selection worked
                new_time_size = ds.dims.get('time', 0)
                if new_time_size != kept_frames:
                    print(f"Warning: Expected {kept_frames} frames, got {new_time_size}")
                
                print(f"Dataset reduced from {total_frames} to {new_time_size} frames")
            else:
                print("No frames removed - dataset unchanged")
            
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "Empty Frame Removal"
