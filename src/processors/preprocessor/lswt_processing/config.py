"""
Configuration module for LSWT processing pipeline.

Contains the main configuration dataclass that holds all processing parameters.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessingConfig:
    """Configuration object for all processing parameters"""
    
    # Required parameters
    test_id: str
    input_file: str
    output_file: str
    
    # Quality filtering parameters
    quality_threshold: int = 0
    remove_avhrr_ql3: bool = False
    apply_zscore_filter: bool = False
    
    # Temporal filtering parameters
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    remove_empty: bool = False
    remove_threshold: float = 0.0
    
    # Spatial filtering parameters
    remove_shore: Optional[float] = None
    distance_to_land_dir: Optional[str] = None
    min_observation_percent: Optional[float] = None
    
    # Scientific processing parameters
    climatology_file: Optional[str] = None
    ice_file: Optional[str] = None
    
    # Technical parameters
    fillvalue: float = 9999.0    

    # Lake-mean detrending
    detrend_lake_mean: bool = False
    detrend_coverage_threshold: float = 0.05    

    # Zscore filter
    outlier_mode: str | None = None          # "zscore" | "robust" | "quantile" | "off"
    z_threshold: float | None = None         # for mode=zscore
    mad_threshold: float | None = None       # for mode=robust
    quantile_low: float | None = None        # for mode=quantile
    quantile_high: float | None = None       # for mode=quantile

    # ice
    ice_shrink_pixels: int = 1
    ice_value_k: float = 273.15   
    ice_lic_var: str = "smoothed_gap_filled_lic_class"      # default lic ice mask variable

    
    # DINEOF cross-validation generation settings
    cv_enable: bool = False
    cv_data_var: str | None = None
    cv_mask_file: str | None = None
    cv_mask_var: str | None = None
    cv_nbclean: int | None = None
    cv_seed: int | None = 1234  # Default seed for reproducibility
    cv_out: str | None = None
    cv_varname: str | None = None
    cv_fraction_target: Optional[float] = None
    cv_min_cloud_frac: float = 0.05  # Min cloud fraction for source timesteps
    cv_max_cloud_frac: float = 0.70  # Max cloud fraction for source timesteps  

    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        if self.remove_threshold < 0.0 or self.remove_threshold > 1.0:
            raise ValueError("remove_threshold must be between 0.0 and 1.0")
        
        if self.min_observation_percent is not None:
            if self.min_observation_percent < 0.0 or self.min_observation_percent > 1.0:
                raise ValueError("min_observation_percent must be between 0.0 and 1.0")
        
        if self.remove_shore is not None and self.remove_shore < 0.0:
            raise ValueError("remove_shore threshold must be >= 0.0")