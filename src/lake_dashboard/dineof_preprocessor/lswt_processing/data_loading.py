"""
Data loading and preparation processing steps for LSWT data.

Contains steps for:
- Initial dataset loading and variable filtering
- Time coordinate conversion to days since 1981-01-01 12:00:00
"""

import xarray as xr
import numpy as np
import datetime
from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


class DataLoaderStep(ProcessingStep):
    """Load dataset and perform initial setup including temporal subsetting and variable filtering"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return True  # Always needed as the first step
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            print(f"Loading dataset: {config.input_file}")
            print(f"Original dataset dimensions: {dict(ds.dims)}")
            
            # Apply temporal subsetting if specified
            original_time_size = ds.dims.get('time', 0)
            
            if config.start_date is not None and config.end_date is not None:
                ds = ds.sel(time=slice(config.start_date, config.end_date))
                print(f"Applied date range: {config.start_date} to {config.end_date}")
            elif config.start_date is not None:
                ds = ds.sel(time=slice(config.start_date, None))
                print(f"Applied start date: {config.start_date}")
            elif config.end_date is not None:
                ds = ds.sel(time=slice(None, config.end_date))
                print(f"Applied end date: {config.end_date}")
            
            new_time_size = ds.dims.get('time', 0)
            if original_time_size != new_time_size:
                print(f"Time dimension: {original_time_size} -> {new_time_size} frames")
            
            # Filter to keep only required variables (lakeid will be handled separately)
            output_variables = {
                "time", "lat", "lon", "lake_surface_water_temperature", 
                "lakeid", "quality_level", "obs_instr"
            }
            
            original_vars = set(ds.variables.keys())
            to_remove = [v for v in original_vars if v not in output_variables]
            
            if to_remove:
                print(f"Removing {len(to_remove)} unnecessary variables: {to_remove}")
                for var in to_remove:
                    del ds[var]
            
            print(f"Retained variables: {list(ds.variables.keys())}")
            
            # Validate required variables are present
            required_vars = ["lake_surface_water_temperature", "quality_level", "lakeid"]
            missing_vars = [var for var in required_vars if var not in ds.variables]
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")
            
            lakeids = ds["lakeid"].data
            lakeid = int(np.nanmax(lakeids))
            mask0 = (lakeids == lakeid).astype(np.int32)
            
            ds.attrs["_mask"] = mask0  # the lake mask to carry forward
            del ds["lakeid"]            #  no need to store the lakeid grid       
            ds.attrs["_lake_id_value"] = int(lakeid)            
            print(f"Dataset loaded successfully with {len(ds.variables)} variables")
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "Data Loading and Variable Filtering"


class TimeCoordinateConverterStep(ProcessingStep):
    """Convert time coordinates to days since 1981-01-01 12:00:00"""
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return True  # Always needed for consistent time representation
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            print("Converting time coordinates to days since 1981-01-01 12:00:00")
            
            # Reference time
            jan01_1981 = datetime.datetime(1981, 1, 1, 12, 0, 0)
            
            # Date converter function
            def convert_date(npdt):
                """Convert numpy datetime64 to days since reference"""
                try:
                    odt = datetime.datetime.utcfromtimestamp(npdt/1e9)
                    if odt.hour != 12 or odt.minute != 0 or odt.second != 0:
                        raise ValueError(f"Expected midday timestamp, got: {odt}")
                    return (odt - jan01_1981).days
                except Exception as e:
                    raise ValueError(f"Error converting timestamp {npdt}: {e}")
            
            # Get original time data and attributes
            original_times = ds.coords["time"].data.tolist()
            time_attrs = ds["time"].attrs.copy()
            
            print(f"Converting {len(original_times)} time coordinates...")
            
            # Convert all timestamps
            try:
                converted_dates = [convert_date(npdt) for npdt in original_times]
            except Exception as e:
                raise ValueError(f"Time conversion failed: {e}")
            
            # Validate converted dates are sequential
            if len(converted_dates) > 1:
                if not all(converted_dates[i] <= converted_dates[i+1] for i in range(len(converted_dates)-1)):
                    print("Warning: Converted dates are not in chronological order")
            
            # Update time attributes
            time_attrs.update({
                "long_name": "days since 1981-01-01 12:00:00",
                "standard_name": "time",
                "units": "days since 1981-01-01 12:00:00"
            })
            
            # Update time coordinates in dataset
            ds.coords["time"] = xr.IndexVariable("time", np.array(converted_dates), time_attrs)
            
            # Store converted dates for final slicing (matching original behavior)
            ds.attrs['_time_start'] = converted_dates[0]
            ds.attrs['_time_stop'] = converted_dates[-1]
        
            
            # Log conversion results
            time_range = f"{converted_dates[0]} to {converted_dates[-1]}"
            print(f"Time conversion complete: {time_range} (days since 1981-01-01)")
            
            # Convert back to check dates for logging
            start_date = jan01_1981 + datetime.timedelta(days=converted_dates[0])
            end_date = jan01_1981 + datetime.timedelta(days=converted_dates[-1])
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "Time Coordinate Conversion"