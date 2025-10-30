"""
Wrapper for accessing a lake's data to obtain masks, temperature data and temperature time series.
This class encapsulates an xarray dataset (e.g., from a netCDF file) and exposes methods to
retrieve various data slices, convert arrays to Python lists, and handle memory management.
"""

from .bokeh_utils import create_logger
import numpy as np
import xarray as xr

logger = create_logger(__name__)

class Dataset(object):
    count = 0

    def __init__(self, dashboard, xr_dataset):
        """
        Initialize a Dataset instance.
        
        This wraps an xarray dataset to help with memory management and provides helper methods to extract
        data (masks, temperature values, time series) as Python lists rather than numpy arrays.

        :param dashboard: The owning dashboard instance which holds configuration such as variable names.
        :param xr_dataset: An xarray Dataset object (typically loaded from a netCDF4 file).
        """
        self.dashboard = dashboard      # Reference to the dashboard for configuration access.
        self.xr_dataset = xr_dataset    # Store the xarray dataset.

        # Check if this is an ice cover dataset (which uses "lake_ice_cover_class")
        if "lake_ice_cover_class" in self.xr_dataset.variables:
            # This is an ice cover dataset: create a full-grid mask (all ones).
            lat_data = self.xr_dataset.variables["lat"].data
            lon_data = self.xr_dataset.variables["lon"].data
            self.mask = np.ones((len(lat_data), len(lon_data)))
            logger.info("Ice cover dataset detected; using full-grid mask.")
        else:
            # For temperature datasets, try to use the configured mask variable.
            mask_var = dashboard.lake_mask_variable
            if mask_var not in self.xr_dataset.variables and "lakeid_GloboLakes" in self.xr_dataset.variables:
                mask_var = "lakeid_GloboLakes"
            if mask_var in self.xr_dataset.variables:
                self.mask = self.xr_dataset.variables[mask_var].data[:, :]
                logger.info("Using mask variable: %s", mask_var)
            else:
                # If no mask variable is found, create a full-grid mask.
                lat_data = self.xr_dataset.variables["lat"].data
                lon_data = self.xr_dataset.variables["lon"].data
                self.mask = np.ones((len(lat_data), len(lon_data)))
                logger.warning("No mask variable found; using full-grid mask.")

        # Retrieve latitude and longitude arrays from the dataset.
        if "lat" in self.xr_dataset.variables:
            self.lats = self.xr_dataset.variables["lat"].data.tolist()
        else:
            self.lats = []
        if "lon" in self.xr_dataset.variables:
            self.lons = self.xr_dataset.variables["lon"].data.tolist()
        else:
            self.lons = []

        self.shapeYX = self.mask.shape

        Dataset.count += 1
        logger.info("Creating Dataset, open count is now %d" % (Dataset.count))
        
        # Convert the time coordinate data to a Python list.
        self.times = self.xr_dataset.coords["time"].data.tolist()

    def get_mask(self, slice=None):
        """
        Retrieve the lake mask array.
        
        :param slice: Optional tuple (y_min, y_max, x_min, x_max) to subset the mask.
        :return: The mask array (or its subset) where data is available.
        """
        if slice:
            return self.mask[slice[0]: slice[1] + 1, slice[2]: slice[3] + 1]
        else:
            return self.mask

    def get_shape_YX(self):
        """
        Return the shape (rows, columns) of the mask (or data array).
        """
        return self.shapeYX

    def get_data(self, field_name, time_index, slice):
        """
        Retrieve a slice of data for a given field and time index.
        
        :param field_name: The name of the variable/field to extract.
        :param time_index: The index corresponding to a specific time.
        :param slice: A tuple (y_min, y_max, x_min, x_max) for spatial subsetting.
        :return: A subset of the data array at the given time and spatial slice.
        """
        return self.xr_dataset.variables[field_name][time_index, slice[0]:slice[1] + 1, slice[2]:slice[3] + 1].data

    def get_min_value(self, field_name):
        """
        Return the minimum value for a given field, skipping NaN values.
        
        :param field_name: The variable name.
        :return: The minimum value as a scalar.
        """
        return self.xr_dataset[field_name].min(skipna=True).item()

    def get_max_value(self, field_name):
        """
        Return the maximum value for a given field, skipping NaN values.
        
        :param field_name: The variable name.
        :return: The maximum value as a scalar.
        """
        return self.xr_dataset[field_name].max(skipna=True).item()

    def get_max_difference(self, field_name1, field_name2):
        """
        Compute the maximum absolute difference between two fields in the dataset.
        
        :param field_name1: First variable name.
        :param field_name2: Second variable name.
        :return: The maximum absolute difference as a float.
        """
        v1 = self.xr_dataset[field_name1].data
        v2 = self.xr_dataset[field_name2].data
        return float(np.nanmax(np.abs(v1 - v2)))

    def get_max_diff(self, field_name, other_dataset):
        """
        Compute the maximum absolute difference between the same field in this dataset and another.
        
        :param field_name: The variable name.
        :param other_dataset: Another Dataset instance to compare against.
        :return: The maximum absolute difference as a float.
        """
        v1 = self.xr_dataset[field_name].data
        v2 = other_dataset.xr_dataset[field_name].data
        return float(np.nanmax(np.abs(v1 - v2)))

    def get_data_at_time(self, field_name, time_value, slice):
        """
        Retrieve a slice of data for a given field at a specific time value.
        
        :param field_name: The variable name.
        :param time_value: The time value (must match one of the values in self.times).
        :param slice: A tuple (y_min, y_max, x_min, x_max) for spatial subsetting.
        :return: The data slice; if the time_value is not found, returns an array filled with NaN.
        """
        try:
            time_index = self.times.index(time_value)
            return self.xr_dataset.variables[field_name].data[
                time_index, slice[0]:slice[1] + 1, slice[2]:slice[3] + 1
            ]
        except ValueError as ex:
            arr = self.xr_dataset.variables[field_name].data[
                0, slice[0]:slice[1] + 1, slice[2]:slice[3] + 1
            ]
            arr.fill(np.nan)
            return arr

    def get_timeseries(self, field_name, x_index, y_index):
        """
        Extract the entire time series for a given field at a specific spatial coordinate.
        
        :param field_name: The variable name.
        :param x_index: The x-index (column) in the dataset.
        :param y_index: The y-index (row) in the dataset.
        :return: A list representing the time series at that pixel.
        """
        return self.xr_dataset.variables[field_name].data[:, y_index, x_index].tolist()

    def get_lats(self):
        """
        Return the list of latitudes.
        """
        return self.lats

    def get_lons(self):
        """
        Return the list of longitudes.
        """
        return self.lons

    def get_times(self):
        """
        Return the list of time coordinate values.
        """
        return self.xr_dataset.coords["time"].data.tolist()

    def close(self):
        """
        Close the underlying xarray dataset and decrease the open count.
        
        This helps with releasing memory when the dataset is no longer needed.
        """
        Dataset.count -= 1
        logger.info("Closing Dataset, open count is now %d" % (Dataset.count))
        self.xr_dataset.close()
        self.xr_dataset = None

    def __del__(self):
        """
        Destructor called when the Dataset object is garbage collected.
        Logs that the dataset is being deleted.
        """
        logger.info("Deleting Dataset")
