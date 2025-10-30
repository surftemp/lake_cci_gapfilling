# Preprocessor (`src/lake_dashboard/dineof_preprocessor/lswt_processing/…`)
**Base & config**
- `ProcessingStep` (abstract): contract for steps → `should_apply(cfg)`, `apply(ds, cfg)`, `name()`. Also validation + simple logging.
- `ProcessingConfig` (dataclass): all knobs (paths, thresholds, flags, CV options, detrend/climatology settings). Validates ranges.

**Load & normalize**
- `DataLoaderStep` → **in:** source LSWT dataset(s)  
  **out:** `xr.Dataset` with selected variables/time window; attaches basic attrs.  
- `TimeCoordinateConverterStep` → **in:** dataset with `time` as datetimes  
  **out:** `time` converted to **days since 1981-01-01 12:00:00** (epoch preserved in attrs).

**Quality & frame filtering**
- `QualityFilterStep` → **in:** LSWT + quality flags  
  **out:** LSWT with values below quality threshold set to NaN (or masked); records counts.
- `EmptyFrameRemovalStep` → **in:** LSWT cube  
  **out:** removes frames with <X% observations or empty; updates stats.
- `FrameCompletenessFilterStep` (in `frame_filters.py`) → **in:** LSWT  
  **out:** keeps frames above `min_observation_percent`.

**Spatial / pixel-wise filters**
- `SpatialMedianFilterStep` (in `spatial_filters.py`) → **in:** LSWT  
  **out:** median-filtered LSWT (kernel per config).
- (Other spatial frame filters are grouped similarly; selection via config.)

**Physical/ice logic**
- `IceFilterStep` (in `ice_filter.py`) → **in:** LSWT + ice mask/logic  
  **out:** replaces iced pixels with NaN or “ice_replaced” flag per cfg.

**Climatology & detrending**
- `ClimatologyStep` (in `climatology.py`) → **in:** LSWT  
  **out:** per-pixel climatology (typically DOY series in attrs or a variable); stores where saved.
- `DetrendingStep` (in `detrending.py`) → **in:** LSWT  
  **out:** lake-mean (or per-pixel, per cfg) trend removed; **saves slope/intercept/t0** in attrs for post-add-back.

**Finalization & stats**
- `FinalizeDatasetStep` (in `finalization.py`) → **in:** processed LSWT  
  **out:** standardized **prepared.nc** with canonical attrs (e.g., `time_units = "days since 1981-01-01 12:00:00"`), variable attrs, integrity checks.
- `StatsRecorder` (in `stats.py`) → accumulates counts, time extents, lake pixel counts; writes CSVs (daily/monthly/seasonal/summary) and keeps epoch-days helpers.

**Preprocessor net effect**
- **Produces:** `prepared.nc` (+ optional `prepared_datetime*.nc`), stats CSVs, and attrs encoding:
  - epoch time system,
  - detrend parameters,
  - path to climatology (if written),
  - optional CV masks/seeds if configured.

---

# DINEOF (external stage)
- **Consumes:** `prepared.nc`
- **Produces:** EOF recon (`output.nc`, `merged.nc` or similar) with reconstructed anomalies on a possibly **subset or EOF-time grid** (often missing the full original time axis).

---

# Post-processor (`src/lake_dashboard/dineof_postprocessor/…`)
**Pipeline driver**
- `PostProcessor` (in `post_process.py`)  
  Orchestrates modular steps (default order below) using `PostOptions` (dataclass of flags/paths/units).  

**Shared base/context**
- `PostContext` & `PostProcessingStep` (in `post_steps/base.py`)  
  Utilities for time/DOY conversions, units (K/°C), safe mkdir, reading/writing, and helpers like `pick_first_var`. Keeps canonical epoch (`1981-01-01 12:00:00`).

**Default step order & I/O**
1) `MergeOutputsStep` → **in:** DINEOF results + `prepared.nc`  
   **out:** DINEOF field re-indexed to **original full time axis**; writes merged to working ds.
2) `CopyAuxFlagsStep` → **in:** merged ds + `prepared.nc`  
   **out:** copies aux vars (e.g., `ice_replaced`, masks) where aligned.
3) `AddBackTrendStep` → **in:** detrended recon + trend attrs from `prepared.nc`  
   **out:** **trend restored** using saved slope/intercept/t0 (handles missing t0 fallback).  
4) `AddBackClimatologyStep` → **in:** anomaly + DOY climatology (var or external file, DOY axis)  
   **out:** **climatology restored**; renames/aligns DOY as needed.
5) `AddInitMetadataStep` → **in:** access to original `.init`/provenance + `prepared.nc`  
   **out:** attaches `dineof_*` attrs, provenance trail.  
6) `AddEOFsMetadataStep` → **in:** EOF outputs  
   **out:** EOF meta (rank, variance explained, etc.) recorded in attrs.  
7) `QAPlotsStep` → **in:** final/merged ds + paths  
   **out:** static QA figures to `output_dir`.  
8) (Optional) `FilterEOFsStep` (in `filter_eofs.py`) → **in:** temporal EOF series (`temporal_eofK*`)  
   **out:** tail/outlier-filtered EOFs (robust SD or quantile); can **re-reconstruct** after filter if `recon_after_eof_filter=True`.  
9) (Optional) `InterpolateTemporalEOFsStep` (in `interpolate_temporal_eofs.py`) → **in:** temporal EOF series + desired day grid  
   **out:** `eofs_interpolated.nc` respecting edge policy (`leave_nan|nearest`), then (optionally) `ReconstructFromEOFsStep`.  
10) (Optional) `ReconstructFromEOFsStep` → **in:** (filtered/interpolated) EOFs + spatial modes  
    **out:** rebuilt LSWT field.

**Post-processor net effect**
- **Produces:** a final **filled LSWT** on the original time axis (units per option: K/°C), EOF diagnostics, QA plots, and enriched attrs (trend/clim restored, provenance, EOF metadata). Intermediate files (`*_filtered`, `eofs_interpolated.nc`) are controlled by options.

---

# Orchestration (`orchestration/lswtctl.py`)
- CLI that renders SLURM scripts and **submits stages** (`preprocess → dineof → postprocess`) with job-dependency wiring. Helpers for:
  - path resolution, idempotent skips, run-tagging, rendering `stage.slurm`,
  - batching over lake IDs, detect/calc `test_id`, grid expansion, and shell execution.

---

## How the branches link (data contract)
1) **Preprocessor → DINEOF**  
   - Input: raw LSWT + flags → **prepared.nc**  
   - Contract: epoch-days time, masked LSWT variable(s), attrs encoding detrend/climatology, optional CV/masks, and canonical names (`lake_surface_water_temperature`, `lakeid`, etc.).
2) **DINEOF → Post-processor**  
   - Input: **prepared.nc** + DINEOF outputs (`output.nc` / `merged.nc`)  
   - Contract: Post merges DINEOF back to **full original time**, then **adds back trend and climatology** using attrs saved by the preprocessor; copies aux flags; runs EOF filtering/QA as configured.
