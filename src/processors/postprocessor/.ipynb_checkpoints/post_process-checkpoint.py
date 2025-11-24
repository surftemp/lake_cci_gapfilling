# post_process.py
"""
Modular Post-Processor for DINEOF outputs (factorized pipeline)

- No per-lake JSON required.
- Reads detrend/provenance attrs from the DINEOF input file (prepared.nc).
- Climatology path is taken from CLI if provided, else from the preprocessor JSON path
  stored in prepared.nc attrs (preprocess_config_path -> 'climatology_file').

Pipeline (default order):
  1) MergeOutputsStep         : merge DINEOF results back to original time axis
  2) CopyAuxFlagsStep         : copy 'ice_replaced' (and other aux) from prepared.nc where aligned
  3) AddBackTrendStep         : add trend back using attrs encoded at preprocessing
  4) AddBackClimatologyStep   : add climatology back using day-of-year
  5) AddInitMetadataStep      : parse .init and attach dineof_* attrs + provenance
  6) AddEOFsMetadataStep      : read eof(s) file and annotate number of EOFs
  7) AddDineofLogMetadataStep : parse most recent DINEOF .out for CV and missing stats
  8) QAPlotsStep              : optional plots (NewReconstructor + PlotExporter)

CLI:
  --lake-path            original LSWT file (for final timeline/metadata)
  --dineof-input-path    prepared.nc (input to DINEOF, contains attrs)
  --dineof-output-path   Output_.../dineof_results.nc (result from DINEOF)
  --output-path          merged output path
  --output-html-folder   optional; if given, QA plots are generated
  --climatology-file     optional; overrides any value found via preprocess_config_path
  --config-file          optional; path to experiment_settings.json (recorded for provenance)

Author: refactor based on your original post-processor
"""

from __future__ import annotations

import argparse
import json
import os
import xarray as xr
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# ==== step imports (factorized) ====
# Ensure these modules exist under lake_dashboard/dineof_postprocessor/post_steps/
from .post_steps.base import PostProcessingStep, PostContext
from .post_steps.merge_outputs import MergeOutputsStep
from .post_steps.copy_aux_flags import CopyAuxFlagsStep
from .post_steps.add_back_trend import AddBackTrendStep
from .post_steps.add_back_climatology import AddBackClimatologyStep
from .post_steps.add_metadata_init import AddInitMetadataStep
from .post_steps.add_metadata_eofs import AddEOFsMetadataStep
from .post_steps.add_metadata_log import AddDineofLogMetadataStep
from .post_steps.qa_plots import QAPlotsStep
from .post_steps.filter_eofs import FilterTemporalEOFsStep
from .post_steps.reconstruct_from_eofs import ReconstructFromEOFsStep
from .post_steps.interpolate_temporal_eofs import InterpolateTemporalEOFsStep
from .post_steps.lswt_plots import LSWTPlotsStep


# ===== helpers for finding climatology via prepared.nc =====

def _load_preprocess_config_from_input_attrs(dineof_input_path: str) -> Optional[dict]:
    """
    Try to read 'preprocess_config_path' from prepared.nc attrs and load that JSON.
    Returns dict or None.
    """
    try:
        with xr.open_dataset(dineof_input_path) as ds_in:
            cfg_path = ds_in.attrs.get("preprocess_config_path")
        if not cfg_path or not os.path.isfile(cfg_path):
            return None
        with open(cfg_path, "r") as fp:
            return json.load(fp)
    except Exception as e:
        print(f"[Post] Failed to load preprocess config via attrs: {e}")
        return None


def _resolve_climatology_path(
    cli_clim_path: Optional[str],
    dineof_input_path: str
) -> Optional[str]:
    """
    Priority:
      1) CLI --climatology-file if provided
      2) 'climatology_file' from the preprocessor JSON referenced by prepared.nc attrs
      3) None (skip add-back climatology step)
    """
    if cli_clim_path:
        return cli_clim_path
    cfg = _load_preprocess_config_from_input_attrs(dineof_input_path)
    return cfg.get("climatology_file") if cfg else None

def _with_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"
    
# ===== Pipeline Orchestrator =====

@dataclass
class PostOptions:
    add_back_trend: bool = True
    add_back_climatology: bool = True
    copy_aux_flags: bool = True
    write_init_metadata: bool = True
    write_eof_metadata: bool = True
    write_dineof_log_metadata: bool = True
    run_qa_plots: bool = True
    output_units: str = "kelvin"     # "kelvin" or "celsius"
    keep_attrs: bool = True          # keep/preserve attrs where reasonable
    eof_filter_enable: bool = True
    eof_filter_method: str = "robust_sd"            # "robust_sd" | "quantile"
    eof_filter_k: float = 4.0
    eof_filter_quantiles: tuple = (0.005, 0.995)
    eof_filter_temporal_prefix: str = "temporal_eof"
    eof_filter_overwrite: bool = False
    eof_filter_output_suffix: str = "_filtered"
    recon_after_eof_filter: bool = True
    recon_when_interp_eofs: bool = False   # placeholder for your future interpolation case    
    eof_interp_enable: bool = True
    eof_interp_edge: str = "leave_nan"
    eof_filter_selection: str = "variance_threshold"               # "all" | "variance_threshold" | "top_n"
    eof_filter_variance_threshold: float = 0.5     # cumulative variance explained threshold
    eof_filter_top_n: int = 3                       # number of top EOFs to filter
    
    recon_after_eof_filter: bool = True

class PostProcessor:
    """
    Orchestrates the post-processing pipeline using modular steps.
    """

    def __init__(
        self,
        *,
        lake_path: str,
        dineof_input_path: str,
        dineof_output_path: str,
        output_path: str,
        output_html_folder: Optional[str] = None,
        climatology_file: Optional[str] = None,
        experiment_config_file: Optional[str] = None,
        options: Optional[PostOptions] = None,
    ):
        self.lake_path = lake_path
        self.dineof_input_path = dineof_input_path
        self.dineof_output_path = dineof_output_path
        self.output_path = output_path
        self.output_html_folder = output_html_folder
        self.climatology_file = climatology_file
        self.experiment_config_file = experiment_config_file
        self.options = options or PostOptions()

        # Create a context that's passed through steps
        self.ctx = PostContext(
            lake_path=self.lake_path,
            dineof_input_path=self.dineof_input_path,
            dineof_output_path=self.dineof_output_path,
            output_path=self.output_path,
            output_html_folder=self.output_html_folder,
            climatology_path=self.climatology_file,  # may be None; replaced later if resolvable
            output_units=self.options.output_units,
            keep_attrs=self.options.keep_attrs,
            experiment_config_path=self.experiment_config_file,
        )

        # Build pipeline
        self.pipeline: List[PostProcessingStep] = self._create_pipeline()

    def _create_pipeline(self) -> List[PostProcessingStep]:
        steps: List[PostProcessingStep] = []

        # Merge DINEOF output onto original time axis
        steps.append(MergeOutputsStep())

        # Copy "lake_surface_water_temperature" and "quality_level" from prepared.nc onto merged timeline
        steps.append(CopyAuxFlagsStep(vars_to_copy=(
            "lake_surface_water_temperature",
            "quality_level",
        )))
        
        # Copy aux flags (e.g., 'ice_replaced') from prepared.nc onto merged timeline
        if self.options.copy_aux_flags:
            steps.append(CopyAuxFlagsStep(vars_to_copy=("ice_replaced",)))

        # Optional QA plots
        if self.options.run_qa_plots and self.output_html_folder:
            steps.append(QAPlotsStep())        
        
        # Add back trend (if detrended)
        if self.options.add_back_trend:
            steps.append(AddBackTrendStep())

        # Add back climatology
        if self.options.add_back_climatology:
            steps.append(AddBackClimatologyStep())

        # Add DINEOF init metadata (alpha, nev, etc.) + provenance
        if self.options.write_init_metadata:
            steps.append(AddInitMetadataStep())
            
        if self.options.eof_filter_enable:
            steps.append(
                FilterTemporalEOFsStep(
                    method=self.options.eof_filter_method,
                    k=self.options.eof_filter_k,
                    quantiles=self.options.eof_filter_quantiles,
                    temporal_var_prefix=self.options.eof_filter_temporal_prefix,
                    output_suffix=self.options.eof_filter_output_suffix,
                    overwrite=self.options.eof_filter_overwrite,
                    eof_selection=self.options.eof_filter_selection,
                    variance_threshold=self.options.eof_filter_variance_threshold,
                    top_n_eofs=self.options.eof_filter_top_n,                    
                )
            )

        if self.options.eof_interp_enable:
            # Interpolate raw EOFs to full daily
            steps.append(InterpolateTemporalEOFsStep(
                target="full", 
                edge_policy=self.options.eof_interp_edge,
                source_mode="raw"  # eofs.nc -> eofs_interpolated.nc
            ))
            # Interpolate filtered EOFs to full daily
            steps.append(InterpolateTemporalEOFsStep(
                target="full", 
                edge_policy=self.options.eof_interp_edge,
                source_mode="filtered"  # eofs_filtered.nc -> eofs_filtered_interpolated.nc
            ))
        
        # Reconstruct from filtered EOFs (sparse timestamps)
        if self.options.recon_after_eof_filter:
            steps.append(ReconstructFromEOFsStep(source_mode="filtered"))
        
        # Reconstruct from interpolated EOFs (full daily from raw)
        if self.options.eof_interp_enable:
            steps.append(ReconstructFromEOFsStep(source_mode="interp"))
            # Reconstruct from filtered interpolated EOFs (full daily from filtered)
            steps.append(ReconstructFromEOFsStep(source_mode="filtered_interp"))
            

        # Parse DINEOF log for CV metrics
        if self.options.write_dineof_log_metadata:
            steps.append(AddDineofLogMetadataStep())

        # Add EOFs metadata
        if self.options.write_eof_metadata:
            steps.append(AddEOFsMetadataStep())            

        # Note: LSWTPlotsStep is NOT added to the pipeline here.
        # It is called once at the end of run() after all passes complete,
        # to avoid redundant plot generation during intermediate passes.
        
        return steps

    def run(self) -> None:
        # Resolve climatology path if CLI didn't provide it
        if not self.ctx.climatology_path:
            self.ctx.climatology_path = _resolve_climatology_path(
                cli_clim_path=None,
                dineof_input_path=self.dineof_input_path,
            )
            if self.ctx.climatology_path:
                print(f"[Post] Using climatology from preprocess JSON: {self.ctx.climatology_path}")
            else:
                print("[Post] No climatology file available; will skip AddBackClimatologyStep if present.")
                        
        ds: Optional[xr.Dataset] = None

        # Load prepared attrs to construct full daily axis
        with xr.open_dataset(self.dineof_input_path) as ds_in:
            tu = ds_in.attrs.get("time_units", "days since 1981-01-01 12:00:00")
            t0 = int(ds_in.attrs.get("time_start_days"))
            t1 = int(ds_in.attrs.get("time_end_days"))
        self.ctx.time_units = tu
        self.ctx.time_start_days = t0
        self.ctx.time_end_days = t1
        self.ctx.full_days = np.arange(t0, t1 + 1, dtype="int64")        

        # Run pipeline
        for step in self.pipeline:
            if not step.should_apply(self.ctx, ds):
                print(f"[Post] Skipping: {step.name}")
                continue
            print(f"[Post] Applying: {step.name}")
            ds = step.apply(self.ctx, ds)

        # Final write (if the last step hasn’t already written)
        if ds is not None:
            # record provenance paths on the final dataset
            try:
                # propagate preprocess_config_path if present on prepared.nc
                with xr.open_dataset(self.dineof_input_path) as ds_in:
                    pcfg = ds_in.attrs.get("preprocess_config_path")
                    if pcfg:
                        ds.attrs["preprocess_config_path"] = str(pcfg)
            except Exception:
                pass
            if self.experiment_config_file:
                ds.attrs["experiment_config_file"] = str(self.experiment_config_file)

            # sensible default encoding
            enc = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
            # Guarantee float32 for temp_filled (matches earlier behavior)
            if "temp_filled" in ds:
                enc["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            ds.to_netcdf(self.output_path, encoding=enc)
            print(f"[Post] Wrote merged output: {self.output_path}")

        # the second pass creates eof filtered recon
        base_dir = os.path.dirname(self.dineof_output_path)
        filtered_results = os.path.join(base_dir, "dineof_results_eof_filtered.nc")
        
        if os.path.isfile(filtered_results):
            print(f"[Post] Found filtered reconstruction: {filtered_results}")
        
            # Temporarily switch source results and output path
            orig_results = self.ctx.dineof_output_path
            orig_output  = self.ctx.output_path
            orig_html = self.ctx.output_html_folder
            if orig_html:
                self.ctx.output_html_folder = _with_suffix(orig_html, "_eof_filtered") 
                
            self.ctx.dineof_output_path = filtered_results
            self.ctx.output_path        = _with_suffix(orig_output, "_eof_filtered")
        
            # Skip the EOF steps on this second pass (we already filtered & reconstructed)
            skip_steps = {
                "FilterTemporalEOFs", 
                "InterpolateTemporalEOFs_raw",
                "InterpolateTemporalEOFs_filtered",
                "ReconstructFromEOFs_filtered", 
                "ReconstructFromEOFs_interp",
                "ReconstructFromEOFs_filtered_interp",
            }
        
            ds2 = None
            for step in self.pipeline:
                if step.name in skip_steps:
                    continue
                if not step.should_apply(self.ctx, ds2):
                    continue
                ds2 = step.apply(self.ctx, ds2)
        
            if ds2 is not None:
                # same provenance + encoding as your first write
                try:
                    with xr.open_dataset(self.dineof_input_path) as ds_in:
                        pcfg = ds_in.attrs.get("preprocess_config_path")
                        if pcfg:
                            ds2.attrs["preprocess_config_path"] = str(pcfg)
                except Exception:
                    pass
                if self.experiment_config_file:
                    ds2.attrs["experiment_config_file"] = str(self.experiment_config_file)
        
                enc2 = {v: {"zlib": True, "complevel": 4} for v in ds2.data_vars}
                if "temp_filled" in ds2:
                    enc2["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
                os.makedirs(os.path.dirname(self.ctx.output_path), exist_ok=True)
                ds2.to_netcdf(self.ctx.output_path, encoding=enc2)
                print(f"[Post] Wrote filtered final output: {self.ctx.output_path}")
        
            # restore
            self.ctx.dineof_output_path = orig_results
            self.ctx.output_path        = orig_output            
            self.ctx.output_html_folder = orig_html

        # the third pass that creates eof temporally interpolated recon  
        base_dir = os.path.dirname(self.dineof_output_path)
        interp_results = os.path.join(base_dir, "dineof_results_eof_interp_full.nc")
        if os.path.isfile(interp_results):
            print(f"[Post] Found full-daily interpolated reconstruction: {interp_results}")
        
            # stash originals
            orig_results = self.ctx.dineof_output_path
            orig_output  = self.ctx.output_path
            orig_html    = self.ctx.output_html_folder
        
            # switch to interpolated result; suffix output
            self.ctx.dineof_output_path = interp_results
            self.ctx.output_path        = _with_suffix(orig_output, "_eof_interp_full")
            if orig_html:
                self.ctx.output_html_folder = _with_suffix(orig_html, "_eof_interp_full")
        
            # Skip MergeOutputs because we want the full daily timeline, not original lake timeline
            skip_steps = {
                "FilterTemporalEOFs", 
                "InterpolateTemporalEOFs_raw",
                "InterpolateTemporalEOFs_filtered", 
                "MergeOutputsStep",
                "ReconstructFromEOFs_filtered", 
                "ReconstructFromEOFs_interp",
                "ReconstructFromEOFs_filtered_interp",
            }
        
            # Create initial dataset with full daily timeline instead of using MergeOutputsStep
            with xr.open_dataset(interp_results) as ds_interp:
                # Read prepared.nc for metadata and coordinates
                with xr.open_dataset(self.dineof_input_path) as ds_in:
                    self.ctx.input_attrs = dict(ds_in.attrs)
                    self.ctx.lake_id = int(ds_in.attrs.get("lake_id", -1))
                    self.ctx.test_id = str(ds_in.attrs.get("test_id", ""))
                    lat_name = "lat" if "lat" in ds_in.coords else self.ctx.lat_name
                    lon_name = "lon" if "lon" in ds_in.coords else self.ctx.lon_name
                    lat_vals = ds_in[lat_name].values
                    lon_vals = ds_in[lon_name].values
                    lakeid_data = ds_in.get("lakeid")
                
                # Convert full_days (integer days since epoch) to datetime64
                base_time = np.datetime64("1981-01-01T12:00:00")
                full_time = base_time + self.ctx.full_days.astype('timedelta64[D]')
                
                # Get temp_filled from interpolated results
                temp_data = ds_interp["temp_filled"].values
                
                # Build dataset with full daily timeline
                ds3 = xr.Dataset()
                ds3 = ds3.assign_coords({
                    self.ctx.time_name: full_time,
                    lat_name: lat_vals,
                    lon_name: lon_vals
                })
                
                if lakeid_data is not None:
                    ds3["lakeid"] = lakeid_data
                
                ds3["temp_filled"] = xr.DataArray(
                    temp_data,
                    dims=(self.ctx.time_name, lat_name, lon_name),
                    coords={self.ctx.time_name: full_time,
                            lat_name: lat_vals,
                            lon_name: lon_vals},
                    attrs={"comment": "DINEOF-filled anomalies (daily interpolated, before trend/climatology add-back)"}
                )
                
                # Copy attributes
                if self.ctx.keep_attrs:
                    ds3.attrs.update(self.ctx.input_attrs)
                ds3.attrs["prepared_source"] = self.dineof_input_path
                ds3.attrs["dineof_source"] = self.ctx.dineof_output_path
                if self.ctx.test_id is not None:
                    ds3.attrs["test_id"] = self.ctx.test_id
                if self.ctx.lake_id is not None and self.ctx.lake_id >= 0:
                    ds3.attrs["lake_id"] = self.ctx.lake_id
                
                print(f"[Post] Created daily interpolated dataset with {len(full_time)} timesteps (full daily timeline)")
        
            # Now run remaining steps (AddBackTrend, AddBackClimatology, etc.) on the daily timeline
            for step in self.pipeline:
                if step.name in skip_steps:
                    continue
                if not step.should_apply(self.ctx, ds3):
                    continue
                ds3 = step.apply(self.ctx, ds3)
        
            if ds3 is not None:
                try:
                    with xr.open_dataset(self.dineof_input_path) as ds_in:
                        pcfg = ds_in.attrs.get("preprocess_config_path")
                        if pcfg:
                            ds3.attrs["preprocess_config_path"] = str(pcfg)
                except Exception:
                    pass
                if self.experiment_config_file:
                    ds3.attrs["experiment_config_file"] = str(self.experiment_config_file)
        
                enc3 = {v: {"zlib": True, "complevel": 4} for v in ds3.data_vars}
                if "temp_filled" in ds3:
                    enc3["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
                os.makedirs(os.path.dirname(self.ctx.output_path), exist_ok=True)
                ds3.to_netcdf(self.ctx.output_path, encoding=enc3)
                print(f"[Post] Wrote full-daily interpolated final output: {self.ctx.output_path}")
        
            # restore
            self.ctx.dineof_output_path = orig_results
            self.ctx.output_path        = orig_output
            self.ctx.output_html_folder = orig_html
        
        # ==================== PASS 4: filtered interpolated (full daily from filtered) ====================
        base_dir = os.path.dirname(self.dineof_output_path)
        filtered_interp_results = os.path.join(base_dir, "dineof_results_eof_filtered_interp_full.nc")
        if os.path.isfile(filtered_interp_results):
            print(f"[Post] Found filtered-interpolated reconstruction: {filtered_interp_results}")
        
            # stash originals
            orig_results = self.ctx.dineof_output_path
            orig_output  = self.ctx.output_path
            orig_html    = self.ctx.output_html_folder
        
            # switch to filtered interpolated result; suffix output
            self.ctx.dineof_output_path = filtered_interp_results
            self.ctx.output_path        = _with_suffix(orig_output, "_eof_filtered_interp_full")
            if orig_html:
                self.ctx.output_html_folder = _with_suffix(orig_html, "_eof_filtered_interp_full")
        
            # Skip same steps as pass 3
            skip_steps = {
                "FilterTemporalEOFs", 
                "InterpolateTemporalEOFs_raw",
                "InterpolateTemporalEOFs_filtered", 
                "MergeOutputsStep",
                "ReconstructFromEOFs_filtered", 
                "ReconstructFromEOFs_interp",
                "ReconstructFromEOFs_filtered_interp",
            }
        
            # Create initial dataset with full daily timeline instead of using MergeOutputsStep
            with xr.open_dataset(filtered_interp_results) as ds_interp:
                # Read prepared.nc for metadata and coordinates
                with xr.open_dataset(self.dineof_input_path) as ds_in:
                    self.ctx.input_attrs = dict(ds_in.attrs)
                    self.ctx.lake_id = int(ds_in.attrs.get("lake_id", -1))
                    self.ctx.test_id = str(ds_in.attrs.get("test_id", ""))
                    lat_name = "lat" if "lat" in ds_in.coords else self.ctx.lat_name
                    lon_name = "lon" if "lon" in ds_in.coords else self.ctx.lon_name
                    lat_vals = ds_in[lat_name].values
                    lon_vals = ds_in[lon_name].values
                    lakeid_data = ds_in.get("lakeid")
                
                # Convert full_days (integer days since epoch) to datetime64
                base_time = np.datetime64("1981-01-01T12:00:00")
                full_time = base_time + self.ctx.full_days.astype('timedelta64[D]')
                
                # Get temp_filled from interpolated results
                temp_data = ds_interp["temp_filled"].values
                
                # Build dataset with full daily timeline
                ds4 = xr.Dataset()
                ds4 = ds4.assign_coords({
                    self.ctx.time_name: full_time,
                    lat_name: lat_vals,
                    lon_name: lon_vals
                })
                
                if lakeid_data is not None:
                    ds4["lakeid"] = lakeid_data
                
                ds4["temp_filled"] = xr.DataArray(
                    temp_data,
                    dims=(self.ctx.time_name, lat_name, lon_name),
                    coords={self.ctx.time_name: full_time,
                            lat_name: lat_vals,
                            lon_name: lon_vals},
                    attrs={"comment": "DINEOF-filled anomalies (filtered + daily interpolated, before trend/climatology add-back)"}
                )
                
                # Copy attributes
                if self.ctx.keep_attrs:
                    ds4.attrs.update(self.ctx.input_attrs)
                ds4.attrs["prepared_source"] = self.dineof_input_path
                ds4.attrs["dineof_source"] = self.ctx.dineof_output_path
                if self.ctx.test_id is not None:
                    ds4.attrs["test_id"] = self.ctx.test_id
                if self.ctx.lake_id is not None and self.ctx.lake_id >= 0:
                    ds4.attrs["lake_id"] = self.ctx.lake_id
                
                print(f"[Post] Created filtered-interpolated dataset with {len(full_time)} timesteps (full daily timeline)")
        
            # Now run remaining steps (AddBackTrend, AddBackClimatology, etc.) on the daily timeline
            for step in self.pipeline:
                if step.name in skip_steps:
                    continue
                if not step.should_apply(self.ctx, ds4):
                    continue
                ds4 = step.apply(self.ctx, ds4)
        
            if ds4 is not None:
                try:
                    with xr.open_dataset(self.dineof_input_path) as ds_in:
                        pcfg = ds_in.attrs.get("preprocess_config_path")
                        if pcfg:
                            ds4.attrs["preprocess_config_path"] = str(pcfg)
                except Exception:
                    pass
                if self.experiment_config_file:
                    ds4.attrs["experiment_config_file"] = str(self.experiment_config_file)
        
                enc4 = {v: {"zlib": True, "complevel": 4} for v in ds4.data_vars}
                if "temp_filled" in ds4:
                    enc4["temp_filled"] = {"dtype": "float32", "zlib": True, "complevel": 5}
                os.makedirs(os.path.dirname(self.ctx.output_path), exist_ok=True)
                ds4.to_netcdf(self.ctx.output_path, encoding=enc4)
                print(f"[Post] Wrote filtered-interpolated final output: {self.ctx.output_path}")
        
            # restore
            self.ctx.dineof_output_path = orig_results
            self.ctx.output_path        = orig_output
            self.ctx.output_html_folder = orig_html
        
        LSWTPlotsStep(original_ts_path=self.lake_path).apply(self.ctx, None)


# ===== CLI =====

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Modular post-processor for DINEOF outputs")
    p.add_argument("--lake-path", required=True, help="Path to the original lake netCDF")
    p.add_argument("--dineof-input-path", required=True, help="Path to the prepared.nc used as DINEOF input")
    p.add_argument("--dineof-output-path", required=True, help="Path to the DINEOF result (dineof_results.nc)")
    p.add_argument("--output-path", required=True, help="Path to the merged output (netCDF)")
    p.add_argument("--output-html-folder", required=False, default=None,
                   help="Optional folder to write QA plots (if provided, QA step runs)")
    p.add_argument("--climatology-file", required=False, default=None,
                   help="Optional override for climatology path (otherwise inferred via prepared.nc attrs)")
    p.add_argument("--config-file", required=False, default=None,
                   help="Optional experiment_settings.json path to record in output attrs")

    p.add_argument("--no-trend", action="store_true", help="Disable add-back-trend step")
    p.add_argument("--no-climatology", action="store_true", help="Disable add-back-climatology step")
    p.add_argument("--no-aux", action="store_true", help="Disable copy of aux flags (e.g., 'ice_replaced')")
    p.add_argument("--no-init-meta", action="store_true", help="Disable writing INIT/alpha metadata")
    p.add_argument("--no-eof-meta", action="store_true", help="Disable writing EOF metadata")
    p.add_argument("--no-log-meta", action="store_true", help="Disable parsing DINEOF log metadata")
    p.add_argument("--no-qa", action="store_true", help="Disable QA plots even if output-html-folder is provided")
    p.add_argument("--units", choices=("kelvin", "celsius"), default="kelvin",
                   help="Units for final output; default Kelvin.")

    # EOF-filter options
    p.add_argument("--no-eof-filter", action="store_true", help="Disable temporal EOF filtering")
    p.add_argument("--eof-filter-method", choices=("robust_sd","quantile"), default="robust_sd")
    p.add_argument("--eof-filter-k", type=float, default=4.0)
    p.add_argument("--eof-filter-quantiles", type=str, default="0.005,0.995",
                   help="Comma-separated low,high (used when method=quantile)")
    p.add_argument("--eof-filter-overwrite", action="store_true", help="Overwrite eofs.nc in place")
    p.add_argument("--eof-filter-prefix", default="temporal_eof")
    p.add_argument("--eof-filter-suffix", default="_filtered")  
    p.add_argument("--eof-filter-selection", choices=("all", "variance_threshold", "top_n"), 
                   default="variance_threshold",
                   help="Method for selecting which EOFs to filter: all, variance_threshold, or top_n")
    p.add_argument("--eof-filter-variance-threshold", type=float, default=0.5,
                   help="Cumulative variance threshold for EOF selection (0-1, default 0.5 = 50%%)")
    p.add_argument("--eof-filter-top-n", type=int, default=3,
                   help="Number of top EOFs to filter when using top_n selection")
    
    p.add_argument("--no-recon-after-filter", action="store_true",
                   help="Do not reconstruct dineof_results_eof_filtered.nc after EOF filtering")

    p.add_argument("--no-eof-interp", action="store_true", help="Disable temporal EOF interpolation step")
    p.add_argument("--eof-interp-edge", choices=("leave_nan","nearest"), default="leave_nan")
    
    
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.eof_filter_quantiles:
        q_lo, q_hi = [float(x) for x in args.eof_filter_quantiles.split(",")]
    else:
        q_lo, q_hi = (0.005, 0.995)
        
    opts = PostOptions(
        add_back_trend=not args.no_trend,
        add_back_climatology=not args.no_climatology,
        copy_aux_flags=not args.no_aux,
        write_init_metadata=not args.no_init_meta,
        write_eof_metadata=not args.no_eof_meta,
        write_dineof_log_metadata=not args.no_log_meta,
        run_qa_plots=(not args.no_qa) and bool(args.output_html_folder),
        output_units=args.units,
        keep_attrs=True,
        eof_filter_enable=not args.no_eof_filter,
        eof_filter_method=args.eof_filter_method,
        eof_filter_k=args.eof_filter_k,
        eof_filter_quantiles=(q_lo, q_hi),
        eof_filter_temporal_prefix=args.eof_filter_prefix,
        eof_filter_overwrite=args.eof_filter_overwrite,
        eof_filter_output_suffix=args.eof_filter_suffix,
        recon_after_eof_filter = not args.no_recon_after_filter, 
        eof_interp_enable=not args.no_eof_interp,        
        eof_interp_edge=args.eof_interp_edge,            
    )

    proc = PostProcessor(
        lake_path=args.lake_path,
        dineof_input_path=args.dineof_input_path,
        dineof_output_path=args.dineof_output_path,
        output_path=args.output_path,
        output_html_folder=args.output_html_folder,
        climatology_file=args.climatology_file,
        experiment_config_file=args.config_file,
        options=opts,
    )
    proc.run()


if __name__ == "__main__":
    main()