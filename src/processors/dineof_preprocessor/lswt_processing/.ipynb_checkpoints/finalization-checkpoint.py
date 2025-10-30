"""
Dataset finalization processing steps for LSWT data.

Contains steps for:
- Final metadata addition
- Dataset attribute configuration
- Processing summary generation
"""

import xarray as xr
import numpy as np
from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


class DatasetFinalizerStep(ProcessingStep):
    """Finalize dataset with metadata, attributes, and processing summary"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return True  # Always needed as the final step
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            print("Finalizing dataset...")
            
            if "lakeid" not in ds.variables:
                if "_mask" in ds.attrs:
                    ds["lakeid"] = (("lat","lon"), np.asarray(ds.attrs["_mask"]).astype(np.int32))
                else:
                    raise ValueError("Missing lake mask (_mask) and 'lakeid'; cannot finalize.")

            self.validate_dataset(ds, ["lake_surface_water_temperature", "lakeid"])
            
            # Apply final time slicing (matching original behavior)
            if '_time_start' in ds.attrs and '_time_stop' in ds.attrs:
                time_start = ds.attrs['_time_start']
                time_stop = ds.attrs['_time_stop']
                print(f"Applying final time slice: {time_start} to {time_stop}")
                ds = ds.sel(time=slice(time_start, time_stop))
                
                # Clean up temporary attributes
                del ds.attrs['_time_start']
                del ds.attrs['_time_stop']
            
            # Get lake ID from the data
            lakeid = int(ds.attrs.get("_lake_id_value", np.nanmax(ds["lakeid"].data)))
            print(f"Finalizing dataset for lake ID: {lakeid}")
            
            # Calculate final dataset statistics
            total_pixels = np.count_nonzero(ds["lakeid"].data)
            time_steps = ds.dims.get('time', 0)
            lswt_data = ds["lake_surface_water_temperature"]
            valid_observations = int((~np.isnan(lswt_data)).sum())
            total_possible = total_pixels * time_steps
            data_completeness = (valid_observations / total_possible) * 100 if total_possible > 0 else 0
            
            # Temperature statistics
            if valid_observations > 0:
                temp_min = float(np.nanmin(lswt_data))
                temp_max = float(np.nanmax(lswt_data))
                temp_mean = float(np.nanmean(lswt_data))
                temp_std = float(np.nanstd(lswt_data))
            else:
                temp_min = temp_max = temp_mean = temp_std = np.nan
            
            # Set global attributes
            ds.attrs.update({
                "lake_id": lakeid,
                "test_id": config.test_id,
                "processing_software": "LSWT Preprocessor v2.0",
                "processing_date": str(np.datetime64('now')),
                "source_file": config.input_file,
                
                # Dataset statistics
                "final_lake_pixels": total_pixels,
                "final_time_steps": time_steps,
                "total_valid_observations": valid_observations,
                "data_completeness_percent": f"{data_completeness:.2f}",
                
                # Temperature statistics
                "temperature_min": temp_min,
                "temperature_max": temp_max,
                "temperature_mean": temp_mean,
                "temperature_std": temp_std,
            })
            
            # Create processing summary
            processing_summary = self._create_processing_summary(config)
            ds.attrs["processing_summary"] = processing_summary
            # Exact legacy string for downstream compatibility
            ds.attrs["lake_dashboard_prepare_data"] = (
                f"prepare_data: qthreshold:{config.quality_threshold} "
                f"remove_empty:{config.remove_empty} "
                f"remove_threshold:{config.remove_threshold} "
                f"remove_shore:{config.remove_shore} "
                f"min_obs_percent:{config.min_observation_percent}"
            )
            
            # Update variable attributes
            self._update_variable_attributes(ds, config)
            
            # Final validation
            self._validate_final_dataset(ds)

            # --- add canonical integer-time window attrs for post-processing
            epoch_units = "days since 1981-01-01 12:00:00"
            ds.attrs["time_units"] = epoch_units
            
            def _to_days_int64(da_or_scalar, units):
                # robust helper: accepts datetime64 array or scalar "YYYY-MM-DD"
                if isinstance(da_or_scalar, str):
                    t = np.datetime64(da_or_scalar)
                    return int(np.rint((t - np.datetime64("1981-01-01T12:00:00")) / np.timedelta64(1, "D")))
                # assume CF-decoded datetime64 array
                vals = da_or_scalar.values if hasattr(da_or_scalar, "values") else da_or_scalar
                vals = vals.astype("datetime64[ns]")
                base = np.datetime64("1981-01-01T12:00:00").astype("datetime64[ns]")
                days = ((vals - base) / np.timedelta64(1, "D")).astype("int64")
                return int(days[0]) if np.ndim(days) else int(days)
            
            # record the chosen start/end window from config (daily, inclusive)
            ds.attrs["time_start_date"] = config.start_date
            ds.attrs["time_end_date"]   = config.end_date
            ds.attrs["time_start_days"] = _to_days_int64(config.start_date, epoch_units)
            ds.attrs["time_end_days"]   = _to_days_int64(config.end_date,   epoch_units)
            
            print(f"Dataset finalization complete:")
            print(f"  - Lake ID: {lakeid}")
            print(f"  - Final dimensions: {dict(ds.dims)}")
            print(f"  - Lake pixels: {total_pixels}")
            print(f"  - Time steps: {time_steps}")
            print(f"  - Valid observations: {valid_observations:,}")
            print(f"  - Data completeness: {data_completeness:.2f}%")
            if valid_observations > 0:
                print(f"  - Temperature range: {temp_min:.2f} to {temp_max:.2f}°C")
            
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    def _create_processing_summary(self, config: ProcessingConfig) -> str:
        """Create a summary string of all applied processing steps"""
        summary_parts = [
            f"prepare_data: qthreshold:{config.quality_threshold}",
            f"remove_empty:{config.remove_empty}",
            f"remove_threshold:{config.remove_threshold}",
            f"remove_avhrr_ql3:{config.remove_avhrr_ql3}",
            f"apply_zscore_filter:{config.apply_zscore_filter}",
        ]
        
        if config.climatology_file:
            summary_parts.append("climatology_subtraction:True")
        
        if config.remove_shore is not None:
            summary_parts.append(f"remove_shore:{config.remove_shore}")
        
        if config.min_observation_percent is not None:
            summary_parts.append(f"min_obs_percent:{config.min_observation_percent}")
        
        if config.start_date or config.end_date:
            date_range = f"{config.start_date or 'start'}_{config.end_date or 'end'}"
            summary_parts.append(f"date_range:{date_range}")
        
        return " ".join(summary_parts)
    
    def _update_variable_attributes(self, ds: xr.Dataset, config: ProcessingConfig) -> None:
        """Update attributes for key variables"""
        
        # Update lake surface water temperature attributes
        if "lake_surface_water_temperature" in ds.variables:
            lswt_attrs = {
                "long_name": "lake surface water temperature",
                "standard_name": "surface_temperature",
                "units": "degrees_Celsius",
                "_FillValue": config.fillvalue,
                "missing_value": config.fillvalue,
                "valid_range": [-50.0, 60.0],  # Reasonable temperature range for lakes
                "comment": "Quality filtered and processed lake surface water temperature data"
            }
            
            # Add processing-specific comments
            processing_notes = []
            if config.quality_threshold > 0:
                processing_notes.append(f"quality filtered (threshold >= {config.quality_threshold})")
            if config.apply_zscore_filter:
                processing_notes.append("outlier filtered (z-score < 2.5)")
            if config.climatology_file:
                processing_notes.append("climatology subtracted (anomalies)")
            
            if processing_notes:
                lswt_attrs["processing_applied"] = "; ".join(processing_notes)
            
            ds["lake_surface_water_temperature"].attrs.update(lswt_attrs)
        
        # Update lakeid attributes
        if "lakeid" in ds.variables:
            ds["lakeid"].attrs.update({
                "long_name": "lake identification mask",
                "description": "Binary mask identifying lake pixels (1=lake, 0=not lake)",
                "processing_note": "Mask may be modified by spatial filtering steps"
            })
        
        # Update coordinate attributes
        if "time" in ds.coords:
            ds["time"].attrs.update({
                "calendar": "standard",
                "axis": "T"
            })
        
        for coord in ["lat", "lon"]:
            if coord in ds.coords:
                ds[coord].attrs.update({
                    "axis": "Y" if coord == "lat" else "X",
                    "standard_name": "latitude" if coord == "lat" else "longitude",
                    "units": "degrees_north" if coord == "lat" else "degrees_east"
                })
    
    def _validate_final_dataset(self, ds: xr.Dataset) -> None:
        """Perform final validation checks on the dataset"""
        
        # Check required variables
        required_vars = ["lake_surface_water_temperature", "lakeid", "time"]
        missing_vars = [var for var in required_vars if var not in ds.variables and var not in ds.coords]
        if missing_vars:
            raise ValueError(f"Final dataset missing required variables: {missing_vars}")
        
        # Check for reasonable data ranges
        lswt = ds["lake_surface_water_temperature"]
        if lswt.sizes['time'] == 0:
            raise ValueError("Final dataset has no time steps")
        
        lake_pixels = np.count_nonzero(ds["lakeid"].data)
        if lake_pixels == 0:
            raise ValueError("Final dataset has no lake pixels")
        
        # Check temperature data reasonableness
        valid_temps = lswt.where(~np.isnan(lswt))
        if valid_temps.size > 0:
            temp_range = [float(valid_temps.min()), float(valid_temps.max())]
            if temp_range[0] < -100 or temp_range[1] > 100:
                print(f"Warning: Temperature range {temp_range} seems unrealistic")
        
        print("Final dataset validation passed")
    
    @property
    def name(self) -> str:
        return "Dataset Finalization"