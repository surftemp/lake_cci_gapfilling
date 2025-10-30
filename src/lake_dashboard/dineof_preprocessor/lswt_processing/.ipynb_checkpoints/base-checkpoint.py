"""
Base classes and interfaces for LSWT processing pipeline.

Contains the abstract base class that all processing steps must implement.
"""

import xarray as xr
from abc import ABC, abstractmethod
from .config import ProcessingConfig


class ProcessingStep(ABC):
    """Abstract base class for all processing steps"""
    
    @abstractmethod
    def should_apply(self, config: ProcessingConfig) -> bool:
        """
        Determine if this step should be applied based on configuration.
        
        Parameters:
        -----------
        config : ProcessingConfig
            Configuration object containing all processing parameters
            
        Returns:
        --------
        bool
            True if this step should be applied, False otherwise
        """
        pass
    
    @abstractmethod
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        """
        Apply the processing step to the dataset.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input dataset to process
        config : ProcessingConfig
            Configuration object containing processing parameters
            
        Returns:
        --------
        xr.Dataset
            Processed dataset
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name for this processing step.
        
        Returns:
        --------
        str
            Name of the processing step for logging purposes
        """
        pass
    
    def validate_dataset(self, ds: xr.Dataset, required_variables: list = None) -> None:
        """
        Validate that the dataset contains required variables.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset to validate
        required_variables : list, optional
            List of variable names that must be present in the dataset
            
        Raises:
        -------
        ValueError
            If required variables are missing from the dataset
        """
        if required_variables is None:
            required_variables = ["lake_surface_water_temperature"]
        
        missing_vars = [var for var in required_variables if var not in ds.variables]
        if missing_vars:
            raise ValueError(f"Missing required variables in dataset: {missing_vars}")
    
    def log_statistics(self, message: str, before_count: int = None, after_count: int = None) -> None:
        """
        Helper method for logging processing statistics.
        
        Parameters:
        -----------
        message : str
            Base message to log
        before_count : int, optional
            Count before processing
        after_count : int, optional
            Count after processing
        """
        if before_count is not None and after_count is not None:
            change = before_count - after_count
            percentage = (change / before_count) * 100 if before_count > 0 else 0
            print(f"{message}: {before_count} -> {after_count} (removed: {change}, {percentage:.1f}%)")
        else:
            print(message)


class ProcessingError(Exception):
    """Custom exception for processing step errors"""
    
    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        self.message = message
        super().__init__(f"Error in {step_name}: {message}")