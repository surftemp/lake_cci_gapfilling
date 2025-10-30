"""
LSWT Processing Package

A modular processing pipeline for Lake Surface Water Temperature data preprocessing.
This package provides a flexible, extensible framework for applying various
quality control and preprocessing steps to satellite-derived LSWT data.

Main Components:
- config: Configuration dataclass for all processing parameters
- base: Abstract base class for processing steps
- Individual processing modules for specific operations

Usage:
    from lswt_processing import ProcessingConfig
    from lswt_converter import Converter
    
    converter = Converter()
    converter.convert(...)
"""

# Version information
__version__ = "2.0.0"
__author__ = "Niall McCarroll, Shaerdan Shataer"
__institution__ = "University of Reading, National Centre for Earth Observation"

# Import main components for easy access
from .config import ProcessingConfig
from .base import ProcessingStep, ProcessingError

# Import all processing steps
from .data_loading import DataLoaderStep, TimeCoordinateConverterStep
from .quality_filters import QualityFilterStep, AVHRRQualityLevel3FilterStep, ZScoreFilterStep
from .climatology import ClimatologySubtractionStep
from .spatial_filters import ShoreRemovalStep, ObservationAvailabilityFilterStep
from .frame_filters import EmptyFrameRemovalStep
from .finalization import DatasetFinalizerStep

# Define what gets exported when using "from lswt_processing import *"
__all__ = [
    # Core classes
    'ProcessingConfig',
    'ProcessingStep',
    'ProcessingError',
    
    # Processing steps
    'DataLoaderStep',
    'TimeCoordinateConverterStep',
    'QualityFilterStep',
    'AVHRRQualityLevel3FilterStep',
    'ZScoreFilterStep',
    'ClimatologySubtractionStep',
    'ShoreRemovalStep',
    'ObservationAvailabilityFilterStep',
    'EmptyFrameRemovalStep',
    'DatasetFinalizerStep',
]

# Package-level constants
DEFAULT_FILLVALUE = 9999.0
DEFAULT_Z_THRESHOLD = 2.5
REFERENCE_DATE = "1981-01-01 12:00:00"

# Supported processing steps in typical execution order
PROCESSING_STEP_ORDER = [
    'DataLoaderStep',
    'TimeCoordinateConverterStep', 
    'QualityFilterStep',
    'AVHRRQualityLevel3FilterStep',
    'ZScoreFilterStep',
    'ClimatologySubtractionStep',
    'ShoreRemovalStep',
    'ObservationAvailabilityFilterStep',
    'EmptyFrameRemovalStep',
    'DatasetFinalizerStep',
]