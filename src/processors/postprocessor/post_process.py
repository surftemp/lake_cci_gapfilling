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
import gc
import json
import os
import uuid
from datetime import datetime, timezone
import xarray as xr
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import glob

# ==== step imports (factorized) ====
# Ensure these modules exist under lake_dashboard/dineof_postprocessor/post_steps/
from .post_steps.base import PostProcessingStep, PostContext, get_current_rss_mb, get_peak_rss_mb
from .post_steps.merge_outputs import MergeOutputsStep
from .post_steps.copy_aux_flags import CopyAuxFlagsStep, CopyOriginalVarsStep
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
from .post_steps.insitu_validation import InsituValidationStep
from .post_steps.add_data_source_flag import AddDataSourceFlagStep
from .post_steps.clamp_subzero import ClampSubZeroStep


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

def _cci_product_filename(lake_path: str, output_dir: str) -> str:
    """
    Derive CCI-aligned final product filename from the input lake file path.

    Input:  LAKE000001241-CCI-L3S-LSWT-CDR-4.5-fv01.0.nc
    Output: LAKE000001241-CCI-L3S-LSWT-CDR-4.5-fv01.0-FILLED.nc
    """
    base = os.path.basename(lake_path)
    root, ext = os.path.splitext(base)
    return os.path.join(output_dir, f"{root}-FILLED{ext}")
    
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
    eof_filter_replacement_mode: str = "blanket"    # "blanket" | "per_eof"
    
    add_data_source_flag: bool = True  # Add data_source flag (CV/observed/gap)
    clamp_subzero: bool = True         # Clamp sub-zero LSWT to freezing point
    dincae_temporal_interp: bool = True  # Linear temporal interpolation for DINCAE output

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

        # Cache original CCI coordinate attrs for output formatting
        self._orig_coord_attrs = self._load_orig_coord_attrs()

        # Build pipeline
        self.pipeline: List[PostProcessingStep] = self._create_pipeline()

    def _create_pipeline(self) -> List[PostProcessingStep]:
        steps: List[PostProcessingStep] = []

        # Merge DINEOF output onto original time axis
        steps.append(MergeOutputsStep())

        # Copy lake_surface_water_temperature and quality_level from original lake file
        steps.append(CopyOriginalVarsStep())
        
        # Copy aux flags (e.g., 'ice_replaced') from prepared.nc
        if self.options.copy_aux_flags:
            steps.append(CopyAuxFlagsStep())

        # Add data_source flag (CV/observed/gap)
        if self.options.add_data_source_flag:
            steps.append(AddDataSourceFlagStep())

        # Optional QA plots
        if self.options.run_qa_plots and self.output_html_folder:
            steps.append(QAPlotsStep())        
        
        # Add back trend (if detrended)
        if self.options.add_back_trend:
            steps.append(AddBackTrendStep())

        # Add back climatology
        if self.options.add_back_climatology:
            steps.append(AddBackClimatologyStep())

        # Clamp sub-zero LSWT to freezing point (after units are final)
        if self.options.clamp_subzero:
            steps.append(ClampSubZeroStep())

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
                    replacement_mode=self.options.eof_filter_replacement_mode,
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

    # ===== Output formatting helpers =====

    def _load_orig_coord_attrs(self) -> dict:
        """Load coordinate attributes from the original ESA CCI lake file."""
        attrs = {"lat": {}, "lon": {}, "time": {}}
        try:
            with xr.open_dataset(self.lake_path) as ds_orig:
                for coord in ("lat", "lon", "time"):
                    if coord in ds_orig.coords:
                        attrs[coord] = dict(ds_orig[coord].attrs)
        except Exception as e:
            print(f"[Post] Warning: could not load coordinate attrs from {self.lake_path}: {e}")
        return attrs

    def _build_output_encoding(self, ds: xr.Dataset) -> dict:
        """Build netCDF encoding dict that matches ESA CCI format conventions."""
        enc = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}

        # Time: match ESA CCI input (seconds since 1970-01-01, gregorian, int32)
        enc["time"] = {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "gregorian",
            "dtype": "int32",
        }

        # Coordinates: suppress xarray's auto-added _FillValue
        enc["lat"] = {"_FillValue": None}
        enc["lon"] = {"_FillValue": None}

        # Bounds: no _FillValue, no compression needed
        for bnd in ("lat_bounds", "lon_bounds"):
            if bnd in ds:
                enc[bnd] = {"_FillValue": None}

        # crs scalar: no compression
        if "crs" in ds:
            enc["crs"] = {"dtype": "int32"}

        # Reconstructed LSWT: float32, compressed
        RECON_VAR = "lake_surface_water_temperature_reconstructed"
        if RECON_VAR in ds:
            enc[RECON_VAR] = {"dtype": "float32", "zlib": True, "complevel": 5}

        # lake_surface_water_temperature: float32
        if "lake_surface_water_temperature" in ds:
            enc["lake_surface_water_temperature"] = {"dtype": "float32", "zlib": True, "complevel": 4}

        # quality_level: int8 to match input (byte)
        if "quality_level" in ds:
            enc["quality_level"] = {"dtype": "int8", "_FillValue": np.int8(0), "zlib": True, "complevel": 4}

        # Flag variables: uint8
        for flag_var in ("data_source", "ice_replaced"):
            if flag_var in ds:
                enc[flag_var] = {"dtype": "uint8", "zlib": True, "complevel": 4}

        # Lake ID masks: int32
        for lid_var in ("lakeid", "lakeid_original"):
            if lid_var in ds:
                enc[lid_var] = {"dtype": "int32", "zlib": True, "complevel": 4}

        return enc

    def _fix_output_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Restore coordinate attributes and add CCI-compliant metadata."""
        RECON_VAR = "lake_surface_water_temperature_reconstructed"

        # --- Coordinate attributes from original CCI file ---
        for coord in ("lat", "lon", "time"):
            if coord in ds.coords and self._orig_coord_attrs.get(coord):
                ds[coord].attrs.pop("_FillValue", None)
                for k, v in self._orig_coord_attrs[coord].items():
                    ds[coord].attrs[k] = v

        # --- Lat/lon bounds ---
        if "lat" in ds.coords and "lat_bounds" not in ds:
            lat = ds["lat"].values
            lon = ds["lon"].values
            if len(lat) > 1 and len(lon) > 1:
                dlat = np.abs(np.diff(lat[:2])[0]) / 2
                dlon = np.abs(np.diff(lon[:2])[0]) / 2
                ds["lat_bounds"] = xr.DataArray(
                    np.column_stack([lat - dlat, lat + dlat]),
                    dims=("lat", "nv"),
                )
                ds["lon_bounds"] = xr.DataArray(
                    np.column_stack([lon - dlon, lon + dlon]),
                    dims=("lon", "nv"),
                )
                ds["lat"].attrs["bounds"] = "lat_bounds"
                ds["lon"].attrs["bounds"] = "lon_bounds"

        # --- CRS grid mapping (WGS84) ---
        if "crs" not in ds:
            ds["crs"] = xr.DataArray(
                np.int32(0),
                attrs={
                    "grid_mapping_name": "latitude_longitude",
                    "semi_major_axis": 6378137.0,
                    "inverse_flattening": 298.257223563,
                    "longitude_of_prime_meridian": 0.0,
                },
            )
        # Add grid_mapping to all data variables
        for var in list(ds.data_vars):
            if var not in ("crs", "lat_bounds", "lon_bounds", "lakeid", "lakeid_original"):
                ds[var].attrs.setdefault("grid_mapping", "crs")

        # --- Reconstructed LSWT metadata ---
        if RECON_VAR in ds:
            tf_attrs = ds[RECON_VAR].attrs
            tf_attrs["long_name"] = (
                "complete two-dimensional lake surface water temperature "
                "reconstructed by robust-DINEOF"
            )
            tf_attrs["standard_name"] = "lake_surface_water_temperature_reconstructed"
            tf_attrs.setdefault("units", "degree_Celsius")
            tf_attrs["ancillary_variables"] = "quality_level data_source"

        return ds

    def _write_output(self, ds: xr.Dataset, path: str, extra_attrs: Optional[dict] = None) -> None:
        """Write output dataset with proper encoding, coordinate attrs, and provenance."""
        self._fix_output_attrs(ds)

        # Ensure output variable has the standard name (fallback for any edge case)
        RECON_VAR = "lake_surface_water_temperature_reconstructed"
        if RECON_VAR not in ds and "temp_filled" in ds:
            ds = ds.rename({"temp_filled": RECON_VAR})

        # --- CCI-required global attributes ---
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fresh UUID per output file (do NOT inherit from input)
        ds.attrs["tracking_id"] = str(uuid.uuid4())

        # Production timestamp (overwrite inherited input date)
        ds.attrs["date_created"] = now_str

        # Append gap-filling provenance to history
        old_history = ds.attrs.get("history", "")
        ds.attrs["history"] = (
            f"{old_history}; {now_str} gap-filled by robust-DINEOF "
            f"(lake_cci_gapfilling pipeline)"
        )

        # CCI data standards compliance
        ds.attrs["format_version"] = "CCI Data Standards v2.3"
        ds.attrs["naming_authority"] = "lakes.esa-cci"
        ds.attrs["Conventions"] = "CF-1.8"

        # Temporal resolution (daily for all products)
        ds.attrs["time_coverage_resolution"] = "P1D"

        # Provenance paths
        try:
            with xr.open_dataset(self.dineof_input_path) as ds_in:
                pcfg = ds_in.attrs.get("preprocess_config_path")
                if pcfg:
                    ds.attrs["preprocess_config_path"] = str(pcfg)
        except Exception:
            pass
        if self.experiment_config_file:
            ds.attrs["experiment_config_file"] = str(self.experiment_config_file)

        if extra_attrs:
            ds.attrs.update(extra_attrs)

        enc = self._build_output_encoding(ds)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds.to_netcdf(path, encoding=enc, unlimited_dims=["time"])
        print(f"[Post] Wrote: {path}")

    # ==================== Memory-profiled step runner ====================

    def _run_steps(self, ds: Optional[xr.Dataset], skip_steps: Optional[set] = None,
                   pass_label: str = "") -> Optional[xr.Dataset]:
        """Run pipeline steps on ds, with optional memory profiling and step skipping."""
        skip = skip_steps or set()
        profiling = self.ctx.profile_memory
        for step in self.pipeline:
            if step.name in skip:
                continue
            if not step.should_apply(self.ctx, ds):
                if profiling:
                    print(f"[Post] Skipping: {step.name}")
                continue
            if profiling:
                rss_before = get_current_rss_mb()
            print(f"[Post]{' ' + pass_label + ' --' if pass_label else ''} Applying: {step.name}")
            ds = step.apply(self.ctx, ds)
            if profiling:
                rss_after = get_current_rss_mb()
                print(f"[MEM] {step.name}: RSS {rss_before:.0f} -> {rss_after:.0f} MB "
                      f"(delta {rss_after - rss_before:+.0f} MB)")
        return ds

    def _log_pass_memory(self, pass_name: str) -> None:
        """Log memory at pass boundary if profiling is enabled."""
        if self.ctx.profile_memory:
            rss = get_current_rss_mb()
            peak = get_peak_rss_mb()
            print(f"[MEM] === {pass_name} complete: RSS={rss:.0f} MB, peak={peak:.0f} MB ===")

    # ==================== Individual pass methods ====================

    def _pass1_baseline(self) -> None:
        """Pass 1: Baseline DINEOF reconstruction on original sparse timeline."""
        ds = self._run_steps(None, pass_label="Pass1")
        if ds is not None:
            self._write_output(ds, self.output_path)
        del ds
        gc.collect()
        self._log_pass_memory("Pass 1 (baseline)")

    def _pass2_eof_filtered(self) -> None:
        """Pass 2: EOF-filtered reconstruction on original sparse timeline."""
        base_dir = os.path.dirname(self.dineof_output_path)
        filtered_results = os.path.join(base_dir, "dineof_results_eof_filtered.nc")
        if not os.path.isfile(filtered_results):
            return

        print(f"[Post] Found filtered reconstruction: {filtered_results}")

        orig_results = self.ctx.dineof_output_path
        orig_output  = self.ctx.output_path
        orig_html    = self.ctx.output_html_folder
        if orig_html:
            self.ctx.output_html_folder = _with_suffix(orig_html, "_eof_filtered")
        self.ctx.dineof_output_path = filtered_results
        self.ctx.output_path        = _with_suffix(orig_output, "_eof_filtered")

        skip_steps = {
            "FilterTemporalEOFs",
            "InterpolateTemporalEOFs_raw",
            "InterpolateTemporalEOFs_filtered",
            "ReconstructFromEOFs_filtered",
            "ReconstructFromEOFs_interp",
            "ReconstructFromEOFs_filtered_interp",
        }

        ds = self._run_steps(None, skip_steps=skip_steps, pass_label="Pass2")
        if ds is not None:
            self._write_output(ds, self.ctx.output_path)

        # restore
        self.ctx.dineof_output_path = orig_results
        self.ctx.output_path        = orig_output
        self.ctx.output_html_folder = orig_html
        del ds
        gc.collect()
        self._log_pass_memory("Pass 2 (eof_filtered)")

    def _build_full_daily_dataset(self, interp_nc_path: str, comment: str) -> xr.Dataset:
        """Build an xr.Dataset on the full daily timeline from an interpolated results file."""
        base_time = np.datetime64("1981-01-01T12:00:00")
        full_time = base_time + self.ctx.full_days.astype('timedelta64[D]')

        with xr.open_dataset(interp_nc_path) as ds_interp:
            with xr.open_dataset(self.dineof_input_path) as ds_in:
                self.ctx.input_attrs = dict(ds_in.attrs)
                self.ctx.lake_id = int(ds_in.attrs.get("lake_id", -1))
                self.ctx.test_id = str(ds_in.attrs.get("test_id", ""))
                lat_name = "lat" if "lat" in ds_in.coords else self.ctx.lat_name
                lon_name = "lon" if "lon" in ds_in.coords else self.ctx.lon_name
                lat_vals = ds_in[lat_name].values
                lon_vals = ds_in[lon_name].values
                lakeid_data = ds_in.get("lakeid")

            temp_data = ds_interp["lake_surface_water_temperature_reconstructed"].values

        ds = xr.Dataset()
        ds = ds.assign_coords({
            self.ctx.time_name: full_time,
            lat_name: lat_vals,
            lon_name: lon_vals
        })
        if lakeid_data is not None:
            ds["lakeid"] = lakeid_data
        ds["lake_surface_water_temperature_reconstructed"] = xr.DataArray(
            temp_data,
            dims=(self.ctx.time_name, lat_name, lon_name),
            coords={self.ctx.time_name: full_time,
                    lat_name: lat_vals,
                    lon_name: lon_vals},
            attrs={"comment": comment}
        )
        if self.ctx.keep_attrs:
            ds.attrs.update(self.ctx.input_attrs)
        ds.attrs["prepared_source"] = self.dineof_input_path
        ds.attrs["dineof_source"] = self.ctx.dineof_output_path
        if self.ctx.test_id is not None:
            ds.attrs["test_id"] = self.ctx.test_id
        if self.ctx.lake_id is not None and self.ctx.lake_id >= 0:
            ds.attrs["lake_id"] = self.ctx.lake_id

        print(f"[Post] Created dataset with {len(full_time)} timesteps (full daily timeline)")
        return ds

    _FULL_DAILY_SKIP_STEPS = {
        "FilterTemporalEOFs",
        "InterpolateTemporalEOFs_raw",
        "InterpolateTemporalEOFs_filtered",
        "MergeOutputsStep",
        "ReconstructFromEOFs_filtered",
        "ReconstructFromEOFs_interp",
        "ReconstructFromEOFs_filtered_interp",
    }

    def _pass3_interp_full(self) -> None:
        """Pass 3: Raw EOFs interpolated to full daily timeline."""
        base_dir = os.path.dirname(self.dineof_output_path)
        interp_results = os.path.join(base_dir, "dineof_results_eof_interp_full.nc")
        if not os.path.isfile(interp_results):
            return

        print(f"[Post] Found full-daily interpolated reconstruction: {interp_results}")

        orig_results = self.ctx.dineof_output_path
        orig_output  = self.ctx.output_path
        orig_html    = self.ctx.output_html_folder
        self.ctx.dineof_output_path = interp_results
        self.ctx.output_path        = _with_suffix(orig_output, "_eof_interp_full")
        if orig_html:
            self.ctx.output_html_folder = _with_suffix(orig_html, "_eof_interp_full")

        ds = self._build_full_daily_dataset(
            interp_results,
            "DINEOF-filled anomalies (daily interpolated, before trend/climatology add-back)")
        ds = self._run_steps(ds, skip_steps=self._FULL_DAILY_SKIP_STEPS, pass_label="Pass3")
        if ds is not None:
            self._write_output(ds, self.ctx.output_path)

        # restore
        self.ctx.dineof_output_path = orig_results
        self.ctx.output_path        = orig_output
        self.ctx.output_html_folder = orig_html
        del ds
        gc.collect()
        self._log_pass_memory("Pass 3 (interp_full)")

    def _pass4_filtered_interp_full(self) -> None:
        """Pass 4: Filtered EOFs interpolated to full daily timeline (final product)."""
        base_dir = os.path.dirname(self.dineof_output_path)
        filtered_interp_results = os.path.join(base_dir, "dineof_results_eof_filtered_interp_full.nc")
        if not os.path.isfile(filtered_interp_results):
            return

        print(f"[Post] Found filtered-interpolated reconstruction: {filtered_interp_results}")

        orig_results = self.ctx.dineof_output_path
        orig_output  = self.ctx.output_path
        orig_html    = self.ctx.output_html_folder
        self.ctx.dineof_output_path = filtered_interp_results
        self.ctx.output_path        = _with_suffix(orig_output, "_eof_filtered_interp_full")
        if orig_html:
            self.ctx.output_html_folder = _with_suffix(orig_html, "_eof_filtered_interp_full")

        ds = self._build_full_daily_dataset(
            filtered_interp_results,
            "DINEOF-filled anomalies (filtered + daily interpolated, before trend/climatology add-back)")
        ds = self._run_steps(ds, skip_steps=self._FULL_DAILY_SKIP_STEPS, pass_label="Pass4")

        if ds is not None:
            self._write_output(ds, self.ctx.output_path)
            # Also write the CCI-aligned final product file
            post_dir = os.path.dirname(self.ctx.output_path)
            final_path = _cci_product_filename(self.lake_path, post_dir)
            self._write_output(ds, final_path)
            print(f"[Post] Final CCI product: {final_path}")

        # restore
        self.ctx.dineof_output_path = orig_results
        self.ctx.output_path        = orig_output
        self.ctx.output_html_folder = orig_html
        del ds
        gc.collect()
        self._log_pass_memory("Pass 4 (filtered_interp_full)")

    def _pass_cv(self) -> None:
        """CV passes: _for_cv baseline and eof_filtered variants for satellite CV."""
        base_dir = os.path.dirname(self.dineof_output_path)
        cv_results = os.path.join(base_dir, "dineof_results_for_cv.nc")
        cv_eofs = os.path.join(base_dir, "eofs_for_cv.nc")

        if not os.path.isfile(cv_results):
            return

        print(f"[Post] Found CV-withheld reconstruction: {cv_results}")

        # --- Filter _for_cv EOFs and reconstruct filtered _for_cv results ---
        if os.path.isfile(cv_eofs) and self.options.eof_filter_enable:
            cv_filter_step = FilterTemporalEOFsStep(
                method=self.options.eof_filter_method,
                k=self.options.eof_filter_k,
                quantiles=self.options.eof_filter_quantiles,
                temporal_var_prefix=self.options.eof_filter_temporal_prefix,
                output_suffix=self.options.eof_filter_output_suffix,
                overwrite=self.options.eof_filter_overwrite,
                eof_selection=self.options.eof_filter_selection,
                variance_threshold=self.options.eof_filter_variance_threshold,
                top_n_eofs=self.options.eof_filter_top_n,
                replacement_mode=self.options.eof_filter_replacement_mode,
                eofs_basename="eofs_for_cv",
            )
            # These steps operate on files, ds=None is fine
            if cv_filter_step.should_apply(self.ctx, None):
                print(f"[Post] Filtering CV-withheld EOFs: eofs_for_cv.nc")
                cv_filter_step.apply(self.ctx, None)

            if self.options.recon_after_eof_filter:
                cv_recon_step = ReconstructFromEOFsStep(
                    source_mode="filtered",
                    eofs_prefix="eofs_for_cv",
                    output_cv_suffix="_for_cv",
                )
                if cv_recon_step.should_apply(self.ctx, None):
                    print(f"[Post] Reconstructing from filtered CV-withheld EOFs")
                    cv_recon_step.apply(self.ctx, None)

        skip_steps = {
            "FilterTemporalEOFs",
            "InterpolateTemporalEOFs_raw",
            "InterpolateTemporalEOFs_filtered",
            "ReconstructFromEOFs_filtered",
            "ReconstructFromEOFs_interp",
            "ReconstructFromEOFs_filtered_interp",
            "QAPlotsStep",
            "LSWTPlotsStep",
        }

        # Enable CV point marking for _for_cv passes
        self.ctx.mark_cv_points = True

        # --- CV-1: baseline _for_cv ---
        orig_results = self.ctx.dineof_output_path
        orig_output  = self.ctx.output_path
        orig_html    = self.ctx.output_html_folder

        self.ctx.dineof_output_path = cv_results
        self.ctx.output_path        = _with_suffix(orig_output, "_for_cv")

        ds = self._run_steps(None, skip_steps=skip_steps, pass_label="CV-1")
        if ds is not None:
            self._write_output(ds, self.ctx.output_path, {"cv_withheld": 1})

        self.ctx.dineof_output_path = orig_results
        self.ctx.output_path        = orig_output
        self.ctx.output_html_folder = orig_html
        del ds
        gc.collect()
        self._log_pass_memory("CV-1 (baseline_for_cv)")

        # --- CV-2: eof_filtered _for_cv ---
        cv_filtered_results = os.path.join(base_dir, "dineof_results_eof_filtered_for_cv.nc")
        if os.path.isfile(cv_filtered_results):
            print(f"[Post] Found CV-withheld filtered reconstruction: {cv_filtered_results}")

            orig_results = self.ctx.dineof_output_path
            orig_output  = self.ctx.output_path
            orig_html    = self.ctx.output_html_folder

            self.ctx.dineof_output_path = cv_filtered_results
            self.ctx.output_path        = _with_suffix(orig_output, "_eof_filtered_for_cv")

            ds = self._run_steps(None, skip_steps=skip_steps, pass_label="CV-2")
            if ds is not None:
                self._write_output(ds, self.ctx.output_path, {"cv_withheld": 1})

            self.ctx.dineof_output_path = orig_results
            self.ctx.output_path        = orig_output
            self.ctx.output_html_folder = orig_html
            del ds
            gc.collect()
            self._log_pass_memory("CV-2 (eof_filtered_for_cv)")

        # Disable CV marking after _for_cv passes are done
        self.ctx.mark_cv_points = False

    def _pass5_dincae_interp(self) -> None:
        """Pass 5: DINCAE temporal interpolation to full daily timeline."""
        if not self.options.dincae_temporal_interp:
            return

        dincae_results_path = self._find_dincae_results()
        if dincae_results_path is None:
            print("[Post] dincae_results.nc not found; skipping DINCAE interp")
            return

        post_dir = os.path.dirname(self.output_path)
        dincae_sparse_files = glob.glob(os.path.join(post_dir, "*_dincae.nc"))
        if dincae_sparse_files:
            dincae_interp_path = dincae_sparse_files[0].replace(
                "_dincae.nc", "_dincae_interp_full.nc")
        else:
            dincae_interp_path = os.path.join(
                post_dir, _with_suffix(
                    os.path.basename(self.output_path), "_dincae_interp_full"
                ).replace("_dineof", ""))

        if os.path.isfile(dincae_interp_path):
            print(f"[Post] DINCAE interp already exists: {os.path.basename(dincae_interp_path)}")
            return

        print(f"[Post] Creating DINCAE daily interpolation from anomalies: "
              f"{os.path.basename(dincae_results_path)}")

        ds = self._interpolate_dincae_anomalies(dincae_results_path)
        if ds is None:
            return

        orig_output = self.ctx.output_path
        orig_html = self.ctx.output_html_folder
        self.ctx.output_path = dincae_interp_path
        if orig_html:
            self.ctx.output_html_folder = _with_suffix(orig_html, "_dincae_interp_full")

        if not hasattr(self.ctx, 'input_attrs') or not self.ctx.input_attrs:
            with xr.open_dataset(self.dineof_input_path) as ds_in:
                self.ctx.input_attrs = dict(ds_in.attrs)

        skip_steps = {
            "FilterTemporalEOFs",
            "InterpolateTemporalEOFs_raw",
            "InterpolateTemporalEOFs_filtered",
            "MergeOutputsStep",
            "ReconstructFromEOFs_filtered",
            "ReconstructFromEOFs_interp",
            "ReconstructFromEOFs_filtered_interp",
            "CopyOriginalVarsStep",
            "CopyAuxFlagsStep",
            "AddDataSourceFlagStep",
            "QAPlotsStep",
            "AddInitMetadataStep",
            "AddEOFsMetadataStep",
            "AddDineofLogMetadataStep",
        }

        ds = self._run_steps(ds, skip_steps=skip_steps, pass_label="Pass5-DINCAE")
        if ds is not None:
            self._write_output(ds, dincae_interp_path, {"dincae_interp_anomaly_space": 1})

        self.ctx.output_path = orig_output
        self.ctx.output_html_folder = orig_html
        del ds
        gc.collect()
        self._log_pass_memory("Pass 5 (dincae_interp)")

    # ==================== Main entry point ====================

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

        # Load prepared attrs to construct full daily axis
        with xr.open_dataset(self.dineof_input_path) as ds_in:
            tu = ds_in.attrs.get("time_units", "days since 1981-01-01 12:00:00")
            t0 = int(ds_in.attrs.get("time_start_days"))
            t1 = int(ds_in.attrs.get("time_end_days"))
        self.ctx.time_units = tu
        self.ctx.time_start_days = t0
        self.ctx.time_end_days = t1
        self.ctx.full_days = np.arange(t0, t1 + 1, dtype="int64")

        if self.ctx.profile_memory:
            rss = get_current_rss_mb()
            print(f"[MEM] === Post-processing start: RSS={rss:.0f} MB ===")

        # Each pass runs in its own method scope, so the dataset is freed after each pass.
        self._pass1_baseline()
        self._pass2_eof_filtered()
        self._pass3_interp_full()
        self._pass4_filtered_interp_full()
        self._pass_cv()
        self._pass5_dincae_interp()

        if self.ctx.profile_memory:
            rss = get_current_rss_mb()
            peak = get_peak_rss_mb()
            print(f"[MEM] === Post-processing complete: RSS={rss:.0f} MB, lifetime peak={peak:.0f} MB ===")

    # ==================== DINCAE interpolation helpers ====================

    def _find_dincae_results(self) -> Optional[str]:
        # find dincae_results.nc (anomalies), deriving path from prepared.nc location
        # prepared.nc is at {run_root}/prepared/{lake_id9}/prepared.nc
        # dincae_results.nc is at {run_root}/dincae/{lake_id9}/{alpha}/dincae_results.nc
        prepared_dir = os.path.dirname(self.dineof_input_path)  # .../prepared/{lake_id9}
        lake_id9 = os.path.basename(prepared_dir)
        run_root = os.path.dirname(os.path.dirname(prepared_dir))  # up 2 levels

        # Get alpha from output_path: .../post/{lake_id9}/{alpha}/LAKE...nc
        post_dir = os.path.dirname(self.output_path)
        alpha = os.path.basename(post_dir)

        path = os.path.join(run_root, "dincae", lake_id9, alpha, "dincae_results.nc")
        if os.path.isfile(path):
            return path
        return None

    def _interpolate_dincae_anomalies(self, dincae_results_path: str) -> Optional[xr.Dataset]:
        # read dincae_results.nc (anomalies on prepared.nc sparse timeline),
        # per-pixel linear interpolate to full daily timeline.
        
        # returns xr.Dataset with datetime64 time coords, lake_surface_water_temperature_reconstructed in ANOMALY space.
        # Pipeline steps (trend, climatology, clamp) are applied by the caller.

        base = np.datetime64("1981-01-01T12:00:00", "ns")
        full_days = self.ctx.full_days
        if full_days is None:
            print("[Post] DINCAE interp: full_days not available, skipping")
            return None

        full_time = base + full_days.astype("timedelta64[D]")

        try:
            ds_dincae = xr.open_dataset(dincae_results_path)

            # Handle integer or datetime64 time coords
            dincae_time = ds_dincae["time"].values
            if np.issubdtype(dincae_time.dtype, np.datetime64):
                dincae_days = ((dincae_time.astype("datetime64[ns]").astype("int64")
                                - base.astype("int64")) // 86_400_000_000_000).astype("int64")
            else:
                dincae_days = dincae_time.astype("int64")

            _rv = "lake_surface_water_temperature_reconstructed" if "lake_surface_water_temperature_reconstructed" in ds_dincae else "temp_filled"
            temp_anomaly = ds_dincae[_rv].values.astype("float64")
            ds_dincae.close()

            T_full = len(full_days)
            ny, nx = temp_anomaly.shape[1], temp_anomaly.shape[2]
            temp_full = np.full((T_full, ny, nx), np.nan, dtype="float32")

            sparse_x = dincae_days.astype("float64")
            full_x = full_days.astype("float64")

            # Per-pixel linear interpolation (interior only)
            n_interp = 0
            for iy in range(ny):
                for ix in range(nx):
                    col = temp_anomaly[:, iy, ix]
                    valid = np.isfinite(col)
                    if valid.sum() < 2:
                        for i_s, d in enumerate(dincae_days):
                            j = np.searchsorted(full_days, d)
                            if j < T_full and full_days[j] == d and valid[i_s]:
                                temp_full[j, iy, ix] = col[i_s]
                        continue

                    x_valid = sparse_x[valid]
                    y_valid = col[valid]

                    i0 = np.searchsorted(full_x, x_valid[0])
                    i1 = np.searchsorted(full_x, x_valid[-1])
                    if i1 < T_full and full_x[i1] == x_valid[-1]:
                        i1_end = i1 + 1
                    else:
                        i1_end = i1

                    if i0 < i1_end:
                        temp_full[i0:i1_end, iy, ix] = np.interp(
                            full_x[i0:i1_end], x_valid, y_valid
                        ).astype("float32")
                        n_interp += 1

            print(f"[Post] DINCAE interp: interpolated {n_interp} pixels onto "
                  f"{T_full} daily timesteps (anomaly space)")

            # Read prepared.nc for coords and metadata
            with xr.open_dataset(self.dineof_input_path) as ds_in:
                lat_name = "lat" if "lat" in ds_in.coords else self.ctx.lat_name
                lon_name = "lon" if "lon" in ds_in.coords else self.ctx.lon_name
                lat_vals = ds_in[lat_name].values
                lon_vals = ds_in[lon_name].values
                lakeid_data = ds_in.get("lakeid")

            # Build dataset (matching Pass 3 format)
            ds_out = xr.Dataset()
            ds_out = ds_out.assign_coords({
                self.ctx.time_name: full_time,
                lat_name: lat_vals,
                lon_name: lon_vals,
            })

            ds_out["lake_surface_water_temperature_reconstructed"] = xr.DataArray(
                temp_full,
                dims=(self.ctx.time_name, lat_name, lon_name),
                coords={self.ctx.time_name: full_time,
                        lat_name: lat_vals,
                        lon_name: lon_vals},
                attrs={"comment": "DINCAE anomalies interpolated to full daily "
                       "(before trend/climatology add-back)"},
            )

            if lakeid_data is not None:
                ds_out["lakeid"] = lakeid_data

            # Copy attrs from prepared.nc (needed by AddBackTrendStep etc.)
            if self.ctx.keep_attrs and hasattr(self.ctx, 'input_attrs') and self.ctx.input_attrs:
                ds_out.attrs.update(self.ctx.input_attrs)
            else:
                with xr.open_dataset(self.dineof_input_path) as ds_in:
                    ds_out.attrs.update(dict(ds_in.attrs))

            ds_out.attrs["source_model"] = "DINCAE"
            ds_out.attrs["interpolation_method"] = "per_pixel_linear"
            ds_out.attrs["interpolation_edge_policy"] = "leave_nan"
            ds_out.attrs["interpolation_source"] = os.path.basename(dincae_results_path)

            return ds_out

        except Exception as e:
            print(f"[Post] DINCAE anomaly interpolation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


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
    p.add_argument("--eof-filter-replacement-mode", choices=("blanket", "per_eof"),
                   default="blanket",
                   help="'blanket': replace all EOFs at any flagged timestep. "
                        "'per_eof': replace each selected EOF only at its own outlier timesteps.")
    
    p.add_argument("--no-recon-after-filter", action="store_true",
                   help="Do not reconstruct dineof_results_eof_filtered.nc after EOF filtering")

    p.add_argument("--no-eof-interp", action="store_true", help="Disable temporal EOF interpolation step")
    p.add_argument("--eof-interp-edge", choices=("leave_nan","nearest"), default="leave_nan")
    
    p.add_argument("--no-clamp-subzero", action="store_true",
                   help="Disable clamping of sub-zero LSWT values")
    p.add_argument("--no-dincae-interp", action="store_true",
                   help="Disable DINCAE temporal interpolation to full daily")
    p.add_argument("--profile-memory", action="store_true",
                   help="Log VmRSS before/after each step and at pass boundaries")

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
        eof_filter_replacement_mode=args.eof_filter_replacement_mode,
        recon_after_eof_filter = not args.no_recon_after_filter, 
        eof_interp_enable=not args.no_eof_interp,        
        eof_interp_edge=args.eof_interp_edge,
        clamp_subzero=not args.no_clamp_subzero,
        dincae_temporal_interp=not args.no_dincae_interp,
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
    proc.ctx.profile_memory = args.profile_memory
    proc.run()


if __name__ == "__main__":
    main()