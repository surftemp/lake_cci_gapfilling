"""
Spatial filtering processing steps for LSWT data.

Contains steps for:
- Shore pixel removal based on distance to land
- Observation availability filtering based on temporal coverage
"""

import xarray as xr
import numpy as np
import os
from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig
from .stats import get_recorder


class ShoreRemovalStep(ProcessingStep):
    """Remove shore pixels based on distance to land thresholds"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return config.remove_shore is not None and config.remove_shore > 0
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            print(f"Applying shore removal with distance threshold: {config.remove_shore}")
            
            self.validate_dataset(ds, ["lake_surface_water_temperature"])
            
            if "_mask" in ds.attrs:
                current_mask = np.asarray(ds.attrs["_mask"])
            elif "lakeid" in ds.variables:
                current_mask = ds["lakeid"].data
            else:
                raise ValueError("No lake mask found ('_mask' attr or 'lakeid' variable).")

            if "_lake_id_value" not in ds.attrs:
                raise ValueError("Missing lake id (_lake_id_value).")
            lakeid = int(ds.attrs["_lake_id_value"])
            print(f"Processing lake ID: {lakeid}")
            
            # Determine distance to land directory
            distance_to_land_dir = config.distance_to_land_dir or "distance_to_land_mask"
            dtl_filename = f"lake_{lakeid}_DTL.nc"
            dtl_filepath = os.path.join(distance_to_land_dir, dtl_filename)
            
            print(f"Looking for distance to land file: {dtl_filepath}")
            
            if not os.path.exists(dtl_filepath):
                print(f"Warning: Distance to land file not found: {dtl_filepath}")
                print(f"Proceeding without shore removal for lake {lakeid}")
                return ds
            
            # Load distance to land data
            print("Loading distance to land data...")
            try:
                dtl_ds = xr.open_dataset(dtl_filepath)
            except Exception as e:
                raise ValueError(f"Failed to load distance to land file: {e}")
            
            # Find distance to land variable (try common names)
            dtl_var_candidates = [
                "distance_to_land",
                "distance_to_shore", 
                "dist_to_land",
                "dtl",
                "distance"
            ]
            
            distance_to_land = None
            for var_name in dtl_var_candidates:
                if var_name in dtl_ds.variables:
                    distance_to_land = dtl_ds[var_name]
                    print(f"Using distance variable: {var_name}")
                    break
            
            if distance_to_land is None:
                available_vars = list(dtl_ds.variables.keys())
                dtl_ds.close()
                raise ValueError(f"No recognized distance variable found. Available: {available_vars}")
            
            # Validate dimensions match
            if distance_to_land.shape != current_mask.shape:
                dtl_ds.close()
                raise ValueError(
                    f"Dimension mismatch: distance data {distance_to_land.shape} "
                    f"vs lake mask {current_mask.shape}"
                )
            
            # Create distance mask (True for pixels with distance > threshold)
            print(f"Creating distance mask with threshold > {config.remove_shore}")
            
            # NaNs → False
            distance_mask = (distance_to_land > config.remove_shore).fillna(False)
            distance_mask_array = distance_mask.values.astype(bool)
            
            # Calculate distance statistics for current lake pixels
            lake_pixel_mask = current_mask.astype(bool)
            if np.any(lake_pixel_mask):
                lake_distances = distance_to_land.values[lake_pixel_mask]
                valid_distances = lake_distances[~np.isnan(lake_distances)]
                if len(valid_distances) > 0:
                    print(f"Distance statistics for lake pixels:")
                    print(f"  - Min distance: {np.min(valid_distances):.2f}")
                    print(f"  - Max distance: {np.max(valid_distances):.2f}")
                    print(f"  - Mean distance: {np.mean(valid_distances):.2f}")
            
            # Intersect original lake mask with distance mask
            cur  = np.asarray(current_mask).astype(bool)
            new_mask = (cur & distance_mask_array).astype(np.int32)
            
            # Calculate removal statistics
            original_lake_pixels = np.count_nonzero(current_mask)
            remaining_lake_pixels = np.count_nonzero(new_mask)
            removed_pixels = original_lake_pixels - remaining_lake_pixels
            removal_percentage = (removed_pixels / original_lake_pixels) * 100 if original_lake_pixels > 0 else 0
            
            print(f"Shore removal statistics:")
            print(f"  - Original lake pixels: {original_lake_pixels}")
            print(f"  - Remaining lake pixels: {remaining_lake_pixels}")
            print(f"  - Removed pixels: {removed_pixels} ({removal_percentage:.1f}%)")
            
            # Record spatial mask pruning
            get_recorder().record_spatial_mask_prune(self.name, int(removed_pixels))

            # Update dataset + running mask
            ds["lakeid"] = (('lat', 'lon'), new_mask)
            ds.attrs["_mask"] = new_mask
            
            # Close distance to land dataset
            dtl_ds.close()
            
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "Shore Pixel Removal"


class ObservationAvailabilityFilterStep(ProcessingStep):
    """Filter pixels based on minimum observation availability over time"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return config.min_observation_percent is not None and config.min_observation_percent > 0
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            threshold_percent = config.min_observation_percent * 100
            print(f"Applying observation availability filtering with minimum {threshold_percent:.1f}% threshold")
            
            self.validate_dataset(ds, ["lake_surface_water_temperature"])
            
            if "_mask" in ds.attrs:
                current_mask = np.asarray(ds.attrs["_mask"])
            elif "lakeid" in ds.variables:
                current_mask = ds["lakeid"].data
            else:
                raise ValueError("No lake mask found ('_mask' attr or 'lakeid' variable).")
            
            # Get LSWT data for availability calculation
            lswt_data = ds["lake_surface_water_temperature"]
            
            print(f"Analyzing observation availability across {lswt_data.sizes['time']} time steps...")
            
            # Calculate availability for each pixel
            valid_observations = (~np.isnan(lswt_data)).sum(dim="time")
            total_time_steps = lswt_data.sizes["time"]
            
            # Calculate fraction of available observations for each pixel (0.0-1.0)
            availability_fraction = valid_observations / total_time_steps
            
            # Create availability mask (True for pixels with sufficient observations)
            availability_mask = availability_fraction >= config.min_observation_percent
            
            # Convert to numpy for intersection calculation
            availability_mask_array = availability_mask.values.astype(int)
            
            # Calculate availability statistics for current lake pixels
            lake_pixel_mask = current_mask.astype(bool)
            if np.any(lake_pixel_mask):
                lake_availability = availability_fraction.values[lake_pixel_mask]
                valid_lake_availability = lake_availability[~np.isnan(lake_availability)]
                
                if len(valid_lake_availability) > 0:
                    print(f"Availability statistics for current lake pixels:")
                    print(f"  - Mean availability: {np.mean(valid_lake_availability)*100:.1f}%")
                    print(f"  - Min availability: {np.min(valid_lake_availability)*100:.1f}%")
                    print(f"  - Max availability: {np.max(valid_lake_availability)*100:.1f}%")
                    
                    # Count pixels below threshold
                    below_threshold = np.sum(valid_lake_availability < config.min_observation_percent)
                    print(f"  - Pixels with <{threshold_percent:.1f}% availability: {below_threshold}")
            
            # Intersect current mask with availability mask
            cur   = np.asarray(current_mask).astype(bool)
            avail = availability_mask.values.astype(bool)
            new_mask = (cur & avail).astype(np.int32)

            # Calculate filtering statistics
            before_availability_pixels = np.count_nonzero(current_mask)
            after_availability_pixels = np.count_nonzero(new_mask)
            removed_by_availability = before_availability_pixels - after_availability_pixels
            availability_removal_percentage = (
                (removed_by_availability / before_availability_pixels) * 100 
                if before_availability_pixels > 0 else 0
            )
            
            print(f"Observation availability filtering results:")
            print(f"  - Total time steps: {total_time_steps}")
            print(f"  - Minimum required observations: {int(total_time_steps * config.min_observation_percent)}")
            print(f"  - Lake pixels before filtering: {before_availability_pixels}")
            print(f"  - Lake pixels after filtering: {after_availability_pixels}")
            print(f"  - Pixels removed: {removed_by_availability} ({availability_removal_percentage:.1f}%)")
            
            # Record spatial mask pruning
            get_recorder().record_spatial_mask_prune(self.name, int(removed_by_availability))

            # Update dataset with new mask
            ds["lakeid"] = (('lat', 'lon'), new_mask)
            
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "Observation Availability Filtering"
