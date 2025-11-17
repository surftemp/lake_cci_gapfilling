"""
Lake Surface Water Temperature Data Converter (Modular Architecture)

Main converter module with pipeline architecture.
Individual processing steps are imported from separate modules.

Author: Niall McCarroll (main), Shaerdan Shataer
Institution: University of Reading, National Centre for Earth Observation
"""

import os
import json
import argparse
from typing import List

import xarray as xr
import numpy as np

# Import processing steps from separate modules
from .lswt_processing.config import ProcessingConfig
from .lswt_processing.base import ProcessingStep
from .lswt_processing.data_loading import DataLoaderStep, TimeCoordinateConverterStep
from .lswt_processing.quality_filters import (
    QualityFilterStep,
    AVHRRQualityLevel3FilterStep,
    ZScoreFilterStep,
)
from .lswt_processing.climatology import ClimatologySubtractionStep
from .lswt_processing.ice_filter import IceMaskReplacementStep
from .lswt_processing.detrending import LakeMeanDetrendStep
from .lswt_processing.spatial_filters import (
    ShoreRemovalStep,
    ObservationAvailabilityFilterStep,
)
from .lswt_processing.frame_filters import EmptyFrameRemovalStep
from .lswt_processing.finalization import DatasetFinalizerStep
from .lswt_processing.dineof_cv import DineofCVGenerationStep
from .lswt_processing.dineof_safety import DineofInitSafetyAdjustStep
from .lswt_processing.stats import get_recorder
from .lswt_processing.final_stats import FinalStatsWriteStep

class Converter:
    """Main converter class with modular processing pipeline"""

    def __init__(self, fillvalue: float = 9999.0):
        self.fillvalue = fillvalue
        self.missingvalue = fillvalue
        self.processing_steps = self._create_processing_pipeline()

    def _create_processing_pipeline(self) -> List[ProcessingStep]:
        """Create the ordered list of processing steps"""
        return [
            # Data loading and preparation
            DataLoaderStep(),
            TimeCoordinateConverterStep(),

            # Quality filtering
            QualityFilterStep(),
            AVHRRQualityLevel3FilterStep(),

            # Scientific preprocessing (order matters)
            ClimatologySubtractionStep(),
            LakeMeanDetrendStep(),   # robust lake-mean detrend
            ZScoreFilterStep(),      # outlier filtering (zscore/robust/quantile)
            
            # Spatial filtering
            ShoreRemovalStep(),
            ObservationAvailabilityFilterStep(),

            # Temporal filtering
            EmptyFrameRemovalStep(),

            # Ice replacement AFTER all filtering (operates in anomaly space if pipeline is set that way)
            IceMaskReplacementStep(),

            # Finalization
            DatasetFinalizerStep(),

            # Stats write (CSV files beside prepared.nc)
            FinalStatsWriteStep(),

            # generate cross-validation mask for customised cv
            DineofCVGenerationStep(),

            # safety check to reduce nevs and ncv if either is > min(size(X_data))
            DineofInitSafetyAdjustStep(),
        ]

    def convert(
        self,
        test_id,
        input_file_name,
        output_file_name,
        qthreshold,
        remove_empty,
        remove_avhrr_ql3,
        remove_threshold,
        start_date,
        end_date,
        climatology_file,
        ice_file,
        apply_zscore_filter,
        remove_shore,
        distance_to_land_dir,
        min_observation_percent,
        detrend_lake_mean=False,
        detrend_coverage_threshold=0.05,
        outlier_mode=None,
        z_threshold=None,
        mad_threshold=None,
        quantile_low=None,
        quantile_high=None,
        ice_shrink_pixels: int = 1,
        ice_value_k: float = 273.15,
        cv_enable: bool = False,
        cv_data_var: str | None = None,
        cv_mask_file: str | None = None,
        cv_mask_var: str | None = None,
        cv_nbclean: int | None = None,
        cv_seed: int | None = None,
        cv_out: str | None = None,
        cv_varname: str | None = None,
        cv_fraction_target: float | None = None,
    ):
        """Main conversion method - maintains original interface"""

        # Create configuration object from parameters
        config = ProcessingConfig(
            test_id=test_id,
            input_file=input_file_name,
            output_file=output_file_name,
            quality_threshold=qthreshold,
            remove_empty=remove_empty,
            remove_avhrr_ql3=remove_avhrr_ql3,
            remove_threshold=remove_threshold,
            start_date=start_date,
            end_date=end_date,
            climatology_file=climatology_file,
            ice_file=ice_file,
            apply_zscore_filter=apply_zscore_filter,
            remove_shore=remove_shore,
            distance_to_land_dir=distance_to_land_dir,
            min_observation_percent=min_observation_percent,
            fillvalue=self.fillvalue,
            detrend_lake_mean=detrend_lake_mean,
            detrend_coverage_threshold=detrend_coverage_threshold,
            outlier_mode=outlier_mode,
            z_threshold=z_threshold,
            mad_threshold=mad_threshold,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            ice_shrink_pixels=ice_shrink_pixels,
            ice_value_k=ice_value_k,
            cv_enable=cv_enable,
            cv_data_var=cv_data_var,
            cv_mask_file=cv_mask_file,
            cv_mask_var=cv_mask_var,
            cv_nbclean=cv_nbclean,
            cv_seed=cv_seed,
            cv_out=cv_out,
            cv_varname=cv_varname,
            cv_fraction_target=cv_fraction_target,        
        )

        # Load initial dataset
        print(f"Loading dataset from {config.input_file}")
        ds = xr.open_dataset(config.input_file)
        
        # Initialize stats recorder context
        rec = get_recorder()
        rec.reset()  # Prevent contamination from previous runs

        # Set initial metadata BEFORE any processing
        rec.set_lake_meta(ds)
        
        # Track whether we've captured the original denominator and baseline
        baseline_captured = False
        
        # Find the CV generation step index
        cv_step_index = None
        for idx, step in enumerate(self.processing_steps):
            if isinstance(step, DineofCVGenerationStep):
                cv_step_index = idx
                break
        
        # Process through pipeline
        for idx, step in enumerate(self.processing_steps):
            # Write file BEFORE CV generation runs
            if cv_step_index is not None and idx == cv_step_index and step.should_apply(config):
                print(f"\n{'='*60}")
                print(f"Writing prepared.nc BEFORE CV generation...")
                print(f"{'='*60}")
                
                # Clean up any remaining temporary attributes before writing
                temp_attrs = [attr for attr in list(ds.attrs.keys()) if attr.startswith("_")]
                for attr in temp_attrs:
                    if attr in ds.attrs:
                        del ds.attrs[attr]

                for v in ("lake_surface_water_temperature", "lat", "lon"):
                    if v in ds.variables:
                        ds[v].attrs.pop("_FillValue", None)
                        ds[v].attrs.pop("missing_value", None)
                
                # Ensure output directory exists
                outdir = os.path.dirname(config.output_file)
                if outdir:
                    os.makedirs(outdir, exist_ok=True)
                
                # Write the file with proper encoding
                encoding = {
                    "lat": {"_FillValue": -32768},
                    "lon": {"_FillValue": -32768},
                    "lake_surface_water_temperature": {
                        "_FillValue": self.fillvalue,
                        "missing_value": self.missingvalue,
                    },
                }
                ds.to_netcdf(config.output_file, encoding=encoding)
                print(f"Intermediate prepared.nc written: {config.output_file}")
                
                # Re-open to ensure proper encoding metadata is available
                ds.close()
                ds = xr.open_dataset(config.output_file, decode_times=False)
                print(f"Dataset reopened from disk for CV generation")
                print(f"{'='*60}\n")
            
            # Apply the processing step
            if step.should_apply(config):
                print(f"Applying: {step.name}")
                ds = step.apply(ds, config)
                
                # CRITICAL: Capture baseline after DataLoaderStep
                # This step does date filtering but no quality filtering yet
                if not baseline_captured and step.name == "Data Loading and Variable Filtering":
                    rec.set_original_denominator_from_ds(ds)
                    if "time" in ds.dims:
                        rec.set_total_time_before(int(ds.dims["time"]))
                    baseline_captured = True
            else:
                print(f"Skipping: {step.name}")

        # Validate that we captured the baseline
        if not baseline_captured:
            print("Warning: Processing baseline was not captured for stats calculation")

        # Clean up any remaining temporary attributes
        temp_attrs = [attr for attr in list(ds.attrs.keys()) if attr.startswith("_")]
        for attr in temp_attrs:
            if attr in ds.attrs:
                del ds.attrs[attr]

        for v in ("lake_surface_water_temperature", "lat", "lon"):
            if v in ds.variables:
                ds[v].attrs.pop("_FillValue", None)
                ds[v].attrs.pop("missing_value", None)

        # Ensure output directory exists
        outdir = os.path.dirname(config.output_file)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        # Record final time count for stats
        rec.set_total_time_after(int(ds.dims.get("time", 0)))
        
        # Only write if we haven't already (CV generation writes it early)
        if cv_step_index is None or not getattr(config, 'cv_enable', False):
            ds.to_netcdf(
                config.output_file,
                encoding={
                    "lat": {"_FillValue": -32768},
                    "lon": {"_FillValue": -32768},
                    "lake_surface_water_temperature": {
                        "_FillValue": self.fillvalue,
                        "missing_value": self.missingvalue,
                    },
                },
            )
            print(f"Conversion complete: {config.input_file} => {config.output_file}")
        else:
            # File already written before CV generation
            ds.close()
            print(f"Conversion complete (file written before CV): {config.input_file} => {config.output_file}")


def _merge_cfg(json_cfg: dict, args: argparse.Namespace) -> dict:
    """Merge JSON config with CLI; CLI overrides JSON when not None."""
    cli = vars(args).copy()
    merged = dict(json_cfg or {})

    # Positional keys
    if cli.get("test_id") is not None:
        merged["test_id"] = cli["test_id"]
    if cli.get("input_file") is not None:
        merged["input_file"] = cli["input_file"]
    if cli.get("output_file") is not None:
        merged["output_file"] = cli["output_file"]

    # Optional flags / options
    for k_cli, k_cfg in [
        ("quality_threshold", "quality_threshold"),
        ("remove_avhrr_ql3", "remove_avhrr_ql3"),
        ("apply_zscore_filter", "apply_zscore_filter"),
        ("outlier_mode", "outlier_mode"),
        ("z_threshold", "z_threshold"),
        ("mad_threshold", "mad_threshold"),
        ("quantile_low", "quantile_low"),
        ("quantile_high", "quantile_high"),
        ("start_date", "start_date"),
        ("end_date", "end_date"),
        ("remove_empty", "remove_empty"),
        ("remove_threshold", "remove_threshold"),
        ("remove_shore", "remove_shore"),
        ("distance_to_land_dir", "distance_to_land_dir"),
        ("min_observation_percent", "min_observation_percent"),
        ("climatology_file", "climatology_file"),
        ("ice_file", "ice_file"),
        ("detrend_lake_mean", "detrend_lake_mean"),
        ("detrend_coverage_threshold", "detrend_coverage_threshold"),
        ("ice_shrink_pixels", "ice_shrink_pixels"),
        ("ice_value_k", "ice_value_k"),
        ("cv_enable", "cv_enable"),
        ("cv_data_var", "cv_data_var"),
        ("cv_mask_file", "cv_mask_file"),
        ("cv_mask_var", "cv_mask_var"),
        ("cv_nbclean", "cv_nbclean"),
        ("cv_seed", "cv_seed"),
        ("cv_out", "cv_out"),
        ("cv_varname", "cv_varname"),  
        ("cv_fraction_target", "cv_fraction_target"),                
    ]:
        val = cli.get(k_cli, None)
        # For booleans from argparse with default None, store only if provided
        if val is not None:
            merged[k_cfg] = val

    # Required fields
    for k in ("test_id", "input_file", "output_file"):
        if k not in merged or merged[k] in (None, ""):
            raise SystemExit(f"Missing required field '{k}' in config or CLI.")

    return merged


def main():
    """Main function with JSON config support (CLI still works, CLI overrides JSON)."""
    parser = argparse.ArgumentParser(
        description="Process lake surface water temperature data for DINEOF analysis"
    )

    # Allow JSON config
    parser.add_argument("--config", type=str, help="Path to JSON config file with all options")

    # Make positional args optional to allow --config-only runs
    parser.add_argument("test_id", nargs="?", default=None, help="Provide a unique identifier for the test")
    parser.add_argument("input_file", nargs="?", default=None, help="The input netCDF4 file")
    parser.add_argument("output_file", nargs="?", default=None, help="The output netCDF4 file")

    # Quality filtering arguments
    quality_group = parser.add_argument_group("Quality Filtering")
    quality_group.add_argument("--quality-threshold", type=int, default=None,
                               help="Quality threshold - use data where quality >= this value")
    quality_group.add_argument("--remove-avhrr-ql3", action="store_true",
                               help="Remove AVHRR QL3 pixels for post-2007 data")
    quality_group.add_argument("--apply-zscore-filter", action="store_true",
                               help="(Back-compat) Apply z-score filtering toggle")
    quality_group.add_argument("--outlier-mode", type=str, default=None,
                               choices=["zscore", "robust", "quantile", "off"],
                               help="Choose one outlier mode: zscore | robust | quantile | off")
    quality_group.add_argument("--z-threshold", type=float, default=None,
                               help="Z-score threshold (mode=zscore)")
    quality_group.add_argument("--mad-threshold", type=float, default=None,
                               help="Robust z threshold in MAD units (mode=robust)")
    quality_group.add_argument("--q-low", dest="quantile_low", type=float, default=None,
                               help="Lower quantile (mode=quantile)")
    quality_group.add_argument("--q-high", dest="quantile_high", type=float, default=None,
                               help="Upper quantile (mode=quantile)")

    # Temporal filtering arguments
    temporal_group = parser.add_argument_group("Temporal Filtering")
    temporal_group.add_argument("--start-date", type=str, default=None,
                                help="Start date for time selection (YYYY-MM-DD)")
    temporal_group.add_argument("--end-date", type=str, default=None,
                                help="End date for time selection (YYYY-MM-DD)")
    temporal_group.add_argument("--remove-empty", action="store_true",
                                help="Remove frames with no valid data")
    temporal_group.add_argument("--remove-threshold", type=float, default=None,
                                help="Remove frames with less than this fraction of valid values (0-1)")

    # Spatial filtering arguments
    spatial_group = parser.add_argument_group("Spatial Filtering")
    spatial_group.add_argument("--remove-shore", type=float, default=None,
                               help="Remove shore pixels with distance to land <= threshold")
    spatial_group.add_argument("--distance-to-land-dir", type=str, default=None,
                               help="Directory containing distance to land files")
    spatial_group.add_argument("--min-observation-percent", type=float, default=None,
                               help="Minimum fraction of valid observations required for pixel inclusion (0-1)")

    # Scientific processing arguments
    science_group = parser.add_argument_group("Scientific Processing")
    science_group.add_argument("--climatology-file", type=str, default=None,
                               help="Path to climatology netCDF file")
    science_group.add_argument("--ice-file", type=str, default=None,
                               help="Path to LIC ice mask netCDF file")
    science_group.add_argument("--detrend-lake-mean", action="store_true",
                               default=None,
                               help="Robust (Theil–Sen) detrend using lake-mean time series")
    science_group.add_argument("--detrend-coverage-threshold", type=float, default=None,
                               help="Min fraction of lake pixels present to include a time step")
    science_group.add_argument("--ice-shrink-pixels", type=int, default=None,
                               help="Binary erosion pixels to shrink ice (default 1)")
    science_group.add_argument("--ice-value-k", type=float, default=None,
                               help="Replacement value for ice (Kelvin, default 273.15)")

    # cv mask generator arguments
    cv_group = parser.add_argument_group("DINEOF CV (optional)")
    cv_group.add_argument("--cv-enable", action="store_true", default=None,
                          help="Generate DINEOF cross-validation pairs from the input file")
    cv_group.add_argument("--cv-data-var", type=str, default=None,
                          help="Variable name in the input file (default: lake_surface_water_temperature)")
    cv_group.add_argument("--cv-mask-file", type=str, default=None,
                          help="Path to land/sea mask NetCDF (lat,lon)")
    cv_group.add_argument("--cv-mask-var", type=str, default=None,
                          help="Variable in the mask file (sea==1/True)")
    cv_group.add_argument("--cv-nbclean", type=int, default=None,
                          help="Number of clean frames that receive pasted donor clouds (default 3)")
    cv_group.add_argument("--cv-seed", type=int, default=None,
                          help="RNG seed (default 123)")
    cv_group.add_argument("--cv-out", type=str, default=None,
                          help="Output NetCDF path for CV pairs (default: <output_dir>/cv_pairs.nc)")
    cv_group.add_argument("--cv-varname", type=str, default=None,
                          help="Variable name in CV NetCDF (default: cv_pairs)")
    

    args = parser.parse_args()

    # Load JSON if provided
    json_cfg = None
    if args.config:
        with open(args.config, "r") as f:
            json_cfg = json.load(f)

    # Merge JSON + CLI (CLI overrides where provided)
    cfg = _merge_cfg(json_cfg, args)

    print(f"Converting {cfg['input_file']} => {cfg['output_file']}")

    # Ensure output directory exists
    outdir = os.path.dirname(cfg["output_file"])
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Run conversion with merged config values
    Converter().convert(
        cfg["test_id"],
        cfg["input_file"],
        cfg["output_file"],
        cfg.get("quality_threshold", 0),
        cfg.get("remove_empty", False),
        cfg.get("remove_avhrr_ql3", False),
        cfg.get("remove_threshold", 0.0),
        cfg.get("start_date"),
        cfg.get("end_date"),
        cfg.get("climatology_file"),
        cfg.get("ice_file"),
        cfg.get("apply_zscore_filter", False),
        cfg.get("remove_shore"),
        cfg.get("distance_to_land_dir"),
        cfg.get("min_observation_percent"),
        cfg.get("detrend_lake_mean", False),
        cfg.get("detrend_coverage_threshold", 0.05),
        cfg.get("outlier_mode"),
        cfg.get("z_threshold"),
        cfg.get("mad_threshold"),
        cfg.get("quantile_low"),
        cfg.get("quantile_high"),
        cfg.get("ice_shrink_pixels", 1),
        cfg.get("ice_value_k", 273.15),
        cfg.get("cv_enable", False),
        cfg.get("cv_data_var"),
        cfg.get("cv_mask_file"),
        cfg.get("cv_mask_var"),
        cfg.get("cv_nbclean"),
        cfg.get("cv_seed"),
        cfg.get("cv_out"),
        cfg.get("cv_varname"),
        cfg.get("cv_fraction_target"),        
    )

    # OPTIONAL: provenance sidecar JSON (effective config + detrend params)
    try:
        ds_out = xr.open_dataset(cfg["output_file"])
        prov = {
            "test_id": cfg["test_id"],
            "input_file": cfg["input_file"],
            "output_file": cfg["output_file"],
            "time_basis": "days since 1981-01-01 12:00:00",
            "effective_config": cfg,
            "processing": {
                "detrend": {
                    "applied": "detrend_method" in ds_out.attrs,
                    "method": ds_out.attrs.get("detrend_method"),
                    "t0_days": ds_out.attrs.get("detrend_t0_days"),
                    "slope_per_day": ds_out.attrs.get("detrend_slope_per_day"),
                    "intercept": ds_out.attrs.get("detrend_intercept"),
                    "coverage_threshold": ds_out.attrs.get("detrend_coverage_threshold"),
                    "n_times_used": ds_out.attrs.get("detrend_n_times_used"),
                },
                "ice_replacement": {
                    "applied": "ice_replaced" in ds_out.variables,
                    "mode": ds_out.attrs.get("ice_replacement_mode"),
                    "total_pixels": int(ds_out["ice_replaced"].sum().values) if "ice_replaced" in ds_out else 0,
                },
                "outlier_filter": {
                    "mode": cfg.get("outlier_mode"),
                    "z_threshold": cfg.get("z_threshold"),
                    "mad_threshold": cfg.get("mad_threshold"),
                    "quantile_low": cfg.get("quantile_low"),
                    "quantile_high": cfg.get("quantile_high"),
                },
            },
        }
        sidecar = cfg["output_file"] + ".provenance.json"
        with open(sidecar, "w") as f:
            json.dump(prov, f, indent=2)
        ds_out.close()
        print(f"Wrote provenance: {sidecar}")
    except Exception as e:
        print(f"(non-fatal) failed to write provenance JSON: {e}")


if __name__ == "__main__":
    main()