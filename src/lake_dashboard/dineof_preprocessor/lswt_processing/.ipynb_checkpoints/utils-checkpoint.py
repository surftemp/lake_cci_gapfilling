"""
Common utility functions for LSWT processing pipeline.

Contains shared utility functions used across multiple processing steps.
"""

import numpy as np
import datetime
import xarray as xr
from typing import Tuple, List, Optional, Union


def calculate_statistics(data: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """
    Calculate comprehensive statistics for a data array.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    mask : np.ndarray, optional
        Boolean mask to apply to data (True = include, False = exclude)
        
    Returns:
    --------
    dict
        Dictionary containing statistical measures
    """
    if mask is not None:
        valid_data = data[mask & ~np.isnan(data)]
    else:
        valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'percentile_25': np.nan,
            'percentile_75': np.nan
        }
    
    return {
        'count': len(valid_data),
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'median': float(np.median(valid_data)),
        'percentile_25': float(np.percentile(valid_data, 25)),
        'percentile_75': float(np.percentile(valid_data, 75))
    }


def format_statistics(stats: dict, name: str = "Data", units: str = "") -> str:
    """
    Format statistics dictionary into a readable string.
    
    Parameters:
    -----------
    stats : dict
        Statistics dictionary from calculate_statistics()
    name : str
        Name of the data being described
    units : str, optional
        Units for the data
        
    Returns:
    --------
    str
        Formatted statistics string
    """
    if stats['count'] == 0:
        return f"{name}: No valid data"
    
    unit_str = f" {units}" if units else ""
    
    return (
        f"{name} statistics ({stats['count']} valid values):\n"
        f"  Mean: {stats['mean']:.3f}{unit_str}\n"
        f"  Std:  {stats['std']:.3f}{unit_str}\n"
        f"  Min:  {stats['min']:.3f}{unit_str}\n"
        f"  Max:  {stats['max']:.3f}{unit_str}\n"
        f"  Median: {stats['median']:.3f}{unit_str}"
    )


def days_since_reference_to_date(days: Union[int, float], 
                                reference_date: str = "1981-01-01 12:00:00") -> datetime.datetime:
    """
    Convert days since reference date to datetime object.
    
    Parameters:
    -----------
    days : int or float
        Days since reference date
    reference_date : str
        Reference date in YYYY-MM-DD HH:MM:SS format
        
    Returns:
    --------
    datetime.datetime
        Corresponding datetime object
    """
    ref_dt = datetime.datetime.strptime(reference_date, "%Y-%m-%d %H:%M:%S")
    return ref_dt + datetime.timedelta(days=int(days))


def date_to_days_since_reference(date: datetime.datetime,
                                reference_date: str = "1981-01-01 12:00:00") -> int:
    """
    Convert datetime object to days since reference date.
    
    Parameters:
    -----------
    date : datetime.datetime
        Date to convert
    reference_date : str
        Reference date in YYYY-MM-DD HH:MM:SS format
        
    Returns:
    --------
    int
        Days since reference date
    """
    ref_dt = datetime.datetime.strptime(reference_date, "%Y-%m-%d %H:%M:%S")
    return (date - ref_dt).days


def extract_lake_id(dataset: xr.Dataset) -> int:
    """
    Extract lake ID from dataset.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Dataset containing lakeid variable
        
    Returns:
    --------
    int
        Lake ID number
        
    Raises:
    -------
    ValueError
        If lakeid variable not found or invalid
    """
    if "lakeid" not in dataset.variables:
        raise ValueError("Dataset does not contain 'lakeid' variable")
    
    lakeid_data = dataset["lakeid"].data
    unique_ids = np.unique(lakeid_data[lakeid_data > 0])
    
    if len(unique_ids) == 0:
        raise ValueError("No valid lake ID found in dataset")
    elif len(unique_ids) > 1:
        print(f"Warning: Multiple lake IDs found: {unique_ids}. Using maximum.")
    
    return int(np.max(unique_ids))


def create_lake_mask(dataset: xr.Dataset, lake_id: int) -> np.ndarray:
    """
    Create binary lake mask for specified lake ID.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Dataset containing lakeid variable
    lake_id : int
        Lake ID to create mask for
        
    Returns:
    --------
    np.ndarray
        Binary mask (1=lake, 0=not lake)
    """
    if "lakeid" not in dataset.variables:
        raise ValueError("Dataset does not contain 'lakeid' variable")
    
    lakeid_data = dataset["lakeid"].data
    return np.where(lakeid_data == lake_id, 1, 0)


def calculate_data_completeness(temperature_data: xr.DataArray, 
                               lake_mask: np.ndarray) -> Tuple[int, int, float]:
    """
    Calculate data completeness statistics.
    
    Parameters:
    -----------
    temperature_data : xr.DataArray
        Temperature data array
    lake_mask : np.ndarray
        Binary lake mask
        
    Returns:
    --------
    Tuple[int, int, float]
        (valid_observations, total_possible, completeness_percentage)
    """
    lake_pixels = np.count_nonzero(lake_mask)
    time_steps = temperature_data.sizes.get('time', 0)
    total_possible = lake_pixels * time_steps
    
    # Count valid observations within lake pixels
    valid_mask = (~np.isnan(temperature_data)) & (lake_mask > 0)
    valid_observations = int(valid_mask.sum())
    
    completeness_percentage = (valid_observations / total_possible * 100) if total_possible > 0 else 0
    
    return valid_observations, total_possible, completeness_percentage


def validate_temperature_range(temperature_data: xr.DataArray, 
                              min_temp: float = -50.0, 
                              max_temp: float = 60.0) -> Tuple[bool, List[str]]:
    """
    Validate temperature data is within reasonable ranges.
    
    Parameters:
    -----------
    temperature_data : xr.DataArray
        Temperature data to validate
    min_temp : float
        Minimum reasonable temperature (°C)
    max_temp : float  
        Maximum reasonable temperature (°C)
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_warnings)
    """
    warnings = []
    is_valid = True
    
    valid_temps = temperature_data.where(~np.isnan(temperature_data))
    
    if valid_temps.size == 0:
        warnings.append("No valid temperature data found")
        is_valid = False
        return is_valid, warnings
    
    actual_min = float(valid_temps.min())
    actual_max = float(valid_temps.max())
    
    if actual_min < min_temp:
        warnings.append(f"Temperature below reasonable minimum: {actual_min:.2f}°C < {min_temp}°C")
        is_valid = False
    
    if actual_max > max_temp:
        warnings.append(f"Temperature above reasonable maximum: {actual_max:.2f}°C > {max_temp}°C")
        is_valid = False
    
    # Check for suspicious values
    if actual_max - actual_min > 80:
        warnings.append(f"Very large temperature range: {actual_max - actual_min:.1f}°C")
    
    return is_valid, warnings


def log_processing_step(step_name: str, 
                       before_count: Optional[int] = None, 
                       after_count: Optional[int] = None,
                       additional_info: Optional[str] = None) -> None:
    """
    Standardized logging for processing steps.
    
    Parameters:
    -----------
    step_name : str
        Name of the processing step
    before_count : int, optional
        Count before processing
    after_count : int, optional
        Count after processing
    additional_info : str, optional
        Additional information to log
    """
    print(f"\n--- {step_name} ---")
    
    if before_count is not None and after_count is not None:
        change = before_count - after_count
        percentage = (change / before_count) * 100 if before_count > 0 else 0
        print(f"  Before: {before_count:,}")
        print(f"  After:  {after_count:,}")
        print(f"  Change: -{change:,} ({percentage:.1f}% reduction)")
    
    if additional_info:
        print(f"  Info: {additional_info}")


def safe_divide(numerator: Union[int, float], 
               denominator: Union[int, float], 
               default: float = 0.0) -> float:
    """
    Safe division that handles zero denominators.
    
    Parameters:
    -----------
    numerator : int or float
        Numerator value
    denominator : int or float
        Denominator value
    default : float
        Value to return if denominator is zero
        
    Returns:
    --------
    float
        Result of division or default value
    """
    if denominator == 0:
        return default
    return float(numerator / denominator)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Parameters:
    -----------
    size_bytes : int
        Size in bytes
        
    Returns:
    --------
    str
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.1f} GB"


def create_processing_metadata(config, processing_time: float = None) -> dict:
    """
    Create standardized processing metadata dictionary.
    
    Parameters:
    -----------
    config : ProcessingConfig
        Configuration object
    processing_time : float, optional
        Processing time in seconds
        
    Returns:
    --------
    dict
        Metadata dictionary
    """
    metadata = {
        "processing_software": "LSWT Preprocessor v2.0",
        "processing_date": str(np.datetime64('now')),
        "source_file": config.input_file,
        "test_id": config.test_id,
        "configuration": {
            "quality_threshold": config.quality_threshold,
            "remove_empty": config.remove_empty,
            "remove_threshold": config.remove_threshold,
            "apply_zscore_filter": config.apply_zscore_filter,
            "remove_avhrr_ql3": config.remove_avhrr_ql3,
        }
    }
    
    if config.climatology_file:
        metadata["configuration"]["climatology_file"] = config.climatology_file
    
    if config.remove_shore is not None:
        metadata["configuration"]["remove_shore"] = config.remove_shore
    
    if config.min_observation_percent is not None:
        metadata["configuration"]["min_observation_percent"] = config.min_observation_percent
    
    if processing_time is not None:
        metadata["processing_time_seconds"] = processing_time
    
    return metadata