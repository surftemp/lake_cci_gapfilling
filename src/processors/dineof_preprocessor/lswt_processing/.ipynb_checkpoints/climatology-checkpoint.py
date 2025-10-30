'''Climatology processing steps for LSWT data.

Contains steps for:
- Climatology subtraction to create temperature anomalies
- Day-of-year alignment for climatological data
'''

import xarray as xr
import numpy as np
import datetime
from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


class ClimatologySubtractionStep(ProcessingStep):
    
    def should_apply(self, config: ProcessingConfig) -> bool:
        return config.climatology_file is not None
    
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            print(f"Subtracting climatology from: {config.climatology_file}")
            
            self.validate_dataset(ds, ["lake_surface_water_temperature"])
            
            # Reference time for day-of-year calculations
            jan01_1981 = datetime.datetime(1981, 1, 1, 12, 0, 0)
            
            # Load climatology dataset
            print("Loading climatology dataset...")
            clim_ds = xr.open_dataset(config.climatology_file)
            
            # Identify climatology variable (try common names)
            clim_var_candidates = [
                "lswt_mean_trimmed_345",
                "lswt_mean_trimmed",
                "climatology", 
                "mean_temperature",
                "lswt_mean",
                "temperature_climatology"
            ]
            
            clim_var = None
            for var_name in clim_var_candidates:
                if var_name in clim_ds.variables:
                    clim_var = clim_ds[var_name]
                    print(f"Using climatology variable: {var_name}")
                    break
            
            if clim_var is None:
                available_vars = list(clim_ds.variables.keys())
                raise ValueError(f"No recognized climatology variable found. Available: {available_vars}")
            
            # Ensure climatology has day-of-year coordinate
            if "doy" not in clim_var.coords:
                print("Converting climatology time coordinate to day-of-year...")
                if "time" in clim_var.coords:
                    clim_var = clim_var.rename({"time": "doy"})
                    # Assume the time dimension represents days 1-365/366
                    doy_values = np.arange(1, clim_var.sizes["doy"] + 1)
                    clim_var = clim_var.assign_coords(doy=doy_values)
                else:
                    raise ValueError("Climatology data must have either 'doy' or 'time' coordinate")
            
            print(f"Climatology shape: {clim_var.shape}")
            print(f"Day-of-year range: {clim_var.coords['doy'].min().values} to {clim_var.coords['doy'].max().values}")
            
            # Convert dataset time coordinates to day-of-year
            converted_dates = ds.coords["time"].data
            print("Computing day-of-year for each time step...")
            
            doy_list = []
            for day_offset in converted_dates:
                actual_date = jan01_1981 + datetime.timedelta(days=int(day_offset))
                doy = actual_date.timetuple().tm_yday
                doy_list.append(doy)
            
            print(f"Dataset day-of-year range: {min(doy_list)} to {max(doy_list)}")
            
            # Assign day-of-year as coordinate to the main dataset
            ds = ds.assign_coords(doy=("time", doy_list))
            
            # Align climatology with dataset day-of-year
            print("Aligning climatology with dataset timeline...")
            clim_aligned = clim_var.sel(doy=ds["doy"], method="nearest")
            
            # Get original temperature data
            orig_lswt = ds["lake_surface_water_temperature"]
            
            # Statistics before subtraction
            valid_orig = (~np.isnan(orig_lswt)).sum()
            valid_clim = (~np.isnan(clim_aligned)).sum()
            print(f"Valid pixels - Original: {valid_orig.values}, Climatology: {valid_clim.values}")
            
            # Custom subtraction logic (matching original exactly):
            # Original comment: "if original == notvalid then result=notvalid; if climatology is NaN then result=valid; else result = orig - climatology"
            # Original code: xr.where((orig == np.nan) | (np.isnan(clim_aligned)), np.nan, orig - clim_aligned)
            # Note: (orig == np.nan) is always False, so effectively: xr.where(np.isnan(clim_aligned), np.nan, orig - clim_aligned)
            print("Performing climatology subtraction...")
            
            new_lswt = xr.where((orig_lswt == np.nan) | (np.isnan(clim_aligned)), np.nan, orig_lswt - clim_aligned)
            
            # Update dataset with anomaly data
            ds["lake_surface_water_temperature"] = new_lswt
            
            # Calculate and log statistics  
            valid_after = (~np.isnan(new_lswt)).sum()
            
            print(f"Climatology subtraction complete:")
            print(f"  - Valid pixels after processing: {valid_after.values}")
            
            # Log temperature range changes
            if valid_after.values > 0:
                orig_range = f"{np.nanmin(orig_lswt):.2f} to {np.nanmax(orig_lswt):.2f}"
                anom_range = f"{np.nanmin(new_lswt):.2f} to {np.nanmax(new_lswt):.2f}"
                print(f"  - Temperature range changed from {orig_range}°C to {anom_range}°C")
            
            # Close climatology dataset
            clim_ds.close()
            
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))
    
    @property
    def name(self) -> str:
        return "Climatology Subtraction"