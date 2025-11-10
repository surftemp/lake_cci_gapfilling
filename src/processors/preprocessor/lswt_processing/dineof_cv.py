# lswt_processing/dineof_cv.py
"""
DINEOF Cross-Validation (CV) Mask Generator
============================================

PURPOSE
-------
Generates synthetic cross-validation test points for DINEOF gap-filling validation.
DINEOF (Data Interpolating Empirical Orthogonal Functions) uses EOF decomposition to 
reconstruct missing data. To validate reconstruction quality, we need known "ground truth" 
values that can be compared against reconstructions. This module creates such test points 
by artificially masking pixels that originally had valid data.

ALGORITHM OVERVIEW
------------------
The CV mask generation follows this procedure:

1. **Identify Clean Frames**: 
   - Analyze all time frames to calculate cloud coverage (fraction of missing lake pixels)
   - Select the N cleanest frames (frames with least missing data) as "recipient" frames
   - These frames have many valid pixels available to serve as ground truth

2. **Select Donor Frames**:
   - From the remaining cloudy frames, randomly select N "donor" frames
   - Each donor frame provides a realistic cloud pattern (spatial distribution of gaps)
   - Randomization ensures diverse cloud geometries are tested

3. **Paste Cloud Patterns**:
   - For each clean→donor pair:
     * Take the cloud mask (NaN pattern) from the donor frame
     * Paste it onto the clean frame, masking previously valid pixels
     * The newly masked pixels become CV test points (ground truth known, artificially hidden)
   
4. **Record CV Test Points**:
   - Track the (space, time) coordinates of all newly masked pixels
   - Convert to DINEOF's linear indexing format: (m, t) pairs where:
     * m = spatial index (1-based, column-major ocean pixel numbering)
     * t = time index (1-based frame number)
   - Output as NetCDF variable with shape (2, nbpoints)

MATHEMATICAL DETAILS
--------------------
For a dataset with dimensions (time=T, lat=nlat, lon=nlon):

- **Lake mask (sea)**: Boolean array (nlat, lon) where True = lake pixel, False = land
- **Ocean pixels (M)**: Number of valid lake pixels = sum(sea)
- **Cloud coverage**: cloudcov[t] = (NaN_count[t] - nbland) / M
  where NaN_count[t] = number of NaN values at time t, nbland = number of land pixels

- **Clean frames**: clean = argsort(cloudcov)[:nbclean]  # Indices of nbclean least cloudy frames
- **Donor frames**: randomly selected from frames with cloudcov > 0, excluding clean frames

For each (clean_t, donor_t) pair:
  - S_original[clean_t, i, j] = valid data (known ground truth)
  - S_modified[clean_t, i, j] = NaN where donor_t was NaN (artificially masked)
  - CV test point: (m[i,j], clean_t) where m is DINEOF's spatial index

DINEOF LINEAR INDEXING
----------------------
DINEOF uses column-major (Fortran-style) linear indexing for ocean pixels:
  - Outer loop over longitude (ii = 0 to nlon-1)
  - Inner loop over latitude (jj = 0 to nlat-1)
  - m = m(lon, lat) increments only for lake pixels (where sea[jj, ii] == True)
  - Indexing is 1-based for compatibility with DINEOF Fortran code

OUTPUT FORMAT
-------------
Creates NetCDF file 'cv_pairs.nc' with variable 'cv_pairs':
  - Dimensions: (index=2, nbpoints=number_of_test_points)
  - index=0: spatial indices (m) in DINEOF linear numbering
  - index=1: temporal indices (t) as 1-based frame numbers
  - Attributes stored in processed dataset:
    * dineof_cv_path: path to cv_pairs.nc
    * dineof_cv_var: variable name in cv_pairs.nc
    * dineof_cv_total_points: total number of CV test points
    * dineof_cv_affected_frames: number of frames with CV points
    * dineof_cv_M_ocean_pixels: total lake pixels (M)
    * dineof_cv_T_frames: total time frames (T)

USAGE IN DINEOF
---------------
The generated cv_pairs.nc file is passed to DINEOF via the clouds parameter:
  clouds = 'cv_pairs.nc#cv_pairs'

DINEOF will:
  1. Mask the specified (m,t) points during reconstruction
  2. Reconstruct values at those points using EOF decomposition
  3. Compare reconstructed vs. true values to compute RMS error
  4. Use cross-validation error to determine optimal number of EOFs

CONFIGURATION
-------------
Required in preprocessing_options:
  - cv_enable: true                 # Enable CV generation
  - cv_mask_var: "lakeid"          # Variable name for lake/land mask

Optional (with defaults):
  - cv_mask_file: <input_file>     # Defaults to input time series file
  - cv_data_var: "lake_surface_water_temperature"  # Temperature variable name
  - cv_nbclean: 3                  # Number of clean frames to use
  - cv_seed: 123                   # Random seed for reproducibility
  - cv_out: "<output_dir>/cv_pairs.nc"  # Output path
  - cv_varname: "cv_pairs"         # Variable name in output NetCDF

REFERENCES
----------
- Alvera-Azcárate et al. (2005): "Reconstruction of incomplete oceanographic data sets 
  using empirical orthogonal functions: application to the Adriatic Sea surface temperature"
- Beckers & Rixen (2003): "EOF calculations and data filling from incomplete oceanographic datasets"
- DINEOF User Guide: http://modb.oce.ulg.ac.be/mediawiki/index.php/DINEOF

AUTHOR
------
Shaerdan Shataer, with architecture design by Niall McCarroll
National Centre for Earth Observation, University of Reading
"""

import os
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


class DineofCVGeneratorCore:
    """
    Core CV pair generator compatible with DINEOF cross-validation requirements.
    
    This class implements the algorithm for creating synthetic cross-validation test points
    by pasting realistic cloud patterns onto clean frames. It handles:
    
    1. Loading and CF-scaling of NetCDF data
    2. Cloud pattern extraction from donor frames
    3. Artificial masking of clean frames
    4. Conversion to DINEOF's column-major linear indexing (1-based)
    5. Export to NetCDF format
    
    The output CV pairs file can be passed directly to DINEOF via:
        clouds = 'cv_pairs.nc#cv_pairs'
    
    DINEOF will use these points to validate reconstruction quality and determine
    the optimal number of EOFs through cross-validation error minimization.
    
    Methods
    -------
    load(data_nc, data_var, mask_nc, mask_var)
        Load temperature data and lake/land mask from NetCDF
    generate(S, sea, nbclean, seed)
        Generate CV test points by pasting cloud patterns
    save_pairs_netcdf(pairs_1based, out_nc, varname)
        Save CV pairs to NetCDF in DINEOF format
    
    Examples
    --------
    >>> core = DineofCVGeneratorCore()
    >>> data, mask = core.load("lake_ts.nc", "temperature", "lake_ts.nc", "lakeid")
    >>> pairs, meta = core.generate(data, mask, nbclean=3, seed=123)
    >>> path, var = core.save_pairs_netcdf(pairs, "cv_pairs.nc")
    >>> print(f"Created {meta['total_cv_points']} CV test points")
    """

    @staticmethod
    def _normalize_mask_da(mask_da: xr.DataArray) -> np.ndarray:
        """
        Convert mask DataArray to boolean numpy array (True = lake/ocean, False = land).
        
        Handles both NaN-based masks and value-based masks (>0.5 = valid).
        """
        m = mask_da.load().astype("float64")
        if set(m.dims) != {"lat", "lon"}:
            raise ValueError(f"mask must be (lat,lon), got {m.dims}")
        return (~np.isnan(m.values)) if np.isnan(m.values).any() else (m.values > 0.5)

    @staticmethod
    def _load_cf_scaled(da: xr.DataArray) -> xr.DataArray:
        """
        Load data array and apply CF convention scale_factor and add_offset.
        
        Also handles _FillValue and missing_value by converting to NaN.
        """
        scale = float(da.attrs.get("scale_factor", 1.0))
        offset = float(da.attrs.get("add_offset", 0.0))
        fill = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))
        arr = da.astype("float64")
        if fill is not None:
            arr = arr.where(arr != fill)
        if scale != 1.0 or offset != 0.0:
            arr = arr * scale + offset
        return arr

    def load(self, data_nc: str, data_var: str, mask_nc: str, mask_var: str) -> Tuple[xr.DataArray, np.ndarray]:
        """
        Load temperature data and lake/land mask from NetCDF files.
        
        Parameters
        ----------
        data_nc : str
            Path to NetCDF file containing temperature data
        data_var : str
            Variable name for temperature (e.g., "lake_surface_water_temperature")
        mask_nc : str
            Path to NetCDF file containing lake/land mask (often same as data_nc)
        mask_var : str
            Variable name for mask (e.g., "lakeid")
            
        Returns
        -------
        da : xr.DataArray
            Temperature data with dimensions (time, lat, lon), CF-scaled
        sea : np.ndarray
            Boolean mask (lat, lon) where True = lake pixel, False = land
            
        Raises
        ------
        ValueError
            If variables not found, dimensions incorrect, or spatial sizes don't match
        """
        ds_data = xr.open_dataset(data_nc, decode_times=False)
        if data_var not in ds_data.variables:
            raise ValueError(f"'{data_var}' not found in {data_nc}")
        da = ds_data[data_var]
        if tuple(da.dims) != ("time", "lat", "lon"):
            raise ValueError(f"{data_var} must be (time,lat,lon), got {da.dims}")
        da = self._load_cf_scaled(da)

        ds_mask = xr.open_dataset(mask_nc, decode_times=False)
        if mask_var not in ds_mask.variables:
            raise ValueError(f"'{mask_var}' not found in {mask_nc}")
        sea = self._normalize_mask_da(ds_mask[mask_var])

        # sanity check: spatial dimensions must match
        if ds_mask[mask_var].sizes.get("lat") != da.sizes["lat"] or ds_mask[mask_var].sizes.get("lon") != da.sizes["lon"]:
            raise ValueError("Mask (lat,lon) size must match data.")
        ds_data.close()
        ds_mask.close()
        return da, sea

    def generate(self, S: xr.DataArray, sea: np.ndarray, nbclean: int = 3, seed: int = 123) -> Tuple[np.ndarray, dict]:
        """
        Generate cross-validation test points by pasting cloud patterns onto clean frames.
        EXACT COPY from working JupyterHub code.
        """
        # Apply lake mask
        S = S.where(sea)
        T, nlat, nlon = S.sizes["time"], S.sizes["lat"], S.sizes["lon"]
        M = int(sea.sum())
        nbland = int((~sea).sum())

        # Calculate cloud coverage per frame
        nan_counts = np.isnan(S).sum(dim=("lat", "lon")).values
        cloudcov = (nan_counts - nbland) / M

        if not (1 <= nbclean < T):
            raise ValueError("nbclean must be >=1 and < number of time steps.")

        # Select clean frames and donors
        clean = np.argsort(cloudcov)[:nbclean]
        donors_pool = np.where((cloudcov > 0) & (~np.isin(np.arange(T), clean)))[0]
        if donors_pool.size == 0:
            raise ValueError("No cloudy donor frames available.")

        rng = np.random.default_rng(seed)
        donors = rng.choice(donors_pool, size=nbclean, replace=True)

        # Paste cloud patterns (EXACT JupyterHub code)
        S_np = S.values
        S2_np = S_np.copy()
        for t_clean, t_donor in zip(clean, donors):
            donor_nan = np.isnan(S_np[t_donor, :, :])
            img = S2_np[t_clean, :, :]
            img[donor_nan] = np.nan
            S2_np[t_clean, :, :] = img

        # Find newly masked pixels
        newly = np.isnan(S2_np) & ~np.isnan(S_np)

        # Build DINEOF linear index (EXACT JupyterHub code)
        mindex_lonlat = np.zeros((nlon, nlat), dtype=np.int32)
        c = 0
        for ii in range(nlon):          # lon outer
            for jj in range(nlat):      # lat inner
                if sea[jj, ii]:
                    c += 1
                    mindex_lonlat[ii, jj] = c
        
        # Extract CV coordinates (EXACT JupyterHub code)
        t_idx, i_idx, j_idx = np.where(newly)
        if t_idx.size == 0:
            raise RuntimeError("No newly masked points; try larger nbclean or ensure donors have clouds.")
        
        space_idx = mindex_lonlat[j_idx, i_idx]
        time_idx  = (t_idx + 1).astype(np.int32)
        clouds_mat = np.column_stack([space_idx, time_idx]).astype(np.int32)

        meta = {
            "clean_frames": clean,
            "donor_frames": donors,
            "total_cv_points": int(clouds_mat.shape[0]),
            "affected_frames": int(np.unique(time_idx).size),
            "mean_cv_per_frame_pct": 100.0 * (clouds_mat.shape[0] / float(T)) / M,
        }
        return clouds_mat, meta

    def save_pairs_netcdf(self, pairs_1based: np.ndarray, out_nc: str, varname: str = "cv_pairs") -> Tuple[str, str]:
        """
        Save CV pairs to NetCDF in DINEOF format.
        EXACT COPY from working JupyterHub code.
        """
        pairs_t = pairs_1based.astype(np.float32).T  # (2, nbpoints)
        nbpoints = pairs_t.shape[1]
        ds = xr.Dataset(
            {varname: (("index", "nbpoints"), pairs_t)},
            coords={
                "index": np.array([1, 2], dtype=np.int32),
                "nbpoints": np.arange(1, nbpoints + 1, dtype=np.int32),
            },
        )
        
        out_nc = str(out_nc)
        os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
        if os.path.exists(out_nc):
            os.remove(out_nc)
        
        ds.to_netcdf(out_nc, mode="w")
        ds.close()
        return out_nc, varname


class DineofCVGenerationStep(ProcessingStep):
    """
    Pipeline-compatible step for generating DINEOF cross-validation pairs:
      - no-op unless config.cv_enable is True
      - reads *raw* input file (config.input_file) to build CV pairs
      - cv_mask_file defaults to config.input_file (typically they're the same)
      - writes NetCDF to cv_pairs.nc, leaves ds unchanged
      - records the path/varname in ds.attrs for downstream use
      
    Required config:
      - cv_enable: True
      - cv_mask_var: name of mask variable (e.g., "lakeid")
    
    Optional config (with defaults):
      - cv_mask_file: path to mask file (defaults to input_file)
      - cv_data_var: variable name (defaults to "lake_surface_water_temperature")
      - cv_nbclean: number of clean frames (defaults to 3)
      - cv_seed: random seed (defaults to 123)
      - cv_out: output path (defaults to <output_dir>/cv_pairs.nc)
      - cv_varname: variable name in output (defaults to "cv_pairs")
    """

    def should_apply(self, config: ProcessingConfig) -> bool:
        return bool(getattr(config, "cv_enable", False))

    @property
    def name(self) -> str:
        return "DINEOF CV Generation"

    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        """Use processed dataset, NOT input file - this is the critical fix!"""
        try:
            if not config.cv_mask_var:
                raise ValueError("cv_enable set but cv_mask_var is not provided.")

            data_var = config.cv_data_var or "lake_surface_water_temperature"
            mask_var = config.cv_mask_var
            nbclean = int(config.cv_nbclean or 3)
            seed = int(config.cv_seed or 123)
            out_nc = config.cv_out or os.path.join(os.path.dirname(config.output_file) or ".", "cv_pairs.nc")
            varname = config.cv_varname or "cv_pairs"
            
            print(f"[CV] Generating cross-validation pairs from PROCESSED dataset")
            print(f"[CV] Using mask variable '{mask_var}' and data variable '{data_var}'")

            # Validate variables exist
            if data_var not in ds.variables:
                raise ValueError(f"'{data_var}' not found in processed dataset")
            if mask_var not in ds.variables:
                raise ValueError(f"'{mask_var}' not found in processed dataset")
            
            # Extract from PROCESSED dataset
            S = ds[data_var].load()
            mask_da = ds[mask_var].load()
            
            # Convert mask to boolean
            core = DineofCVGeneratorCore()
            sea = core._normalize_mask_da(mask_da)
            
            M = int(sea.sum())
            print(f"[CV] Lake pixels in processed data: {M}")
            
            # Generate CV pairs from processed data
            pairs, meta = core.generate(S, sea, nbclean=nbclean, seed=seed)
            
            # Verify spatial indices are valid
            max_spatial_idx = int(pairs[:, 0].max())
            if max_spatial_idx > M:
                raise ValueError(
                    f"CV generation produced spatial index {max_spatial_idx} "
                    f"but only {M} lake pixels exist!"
                )
            
            path, vname = core.save_pairs_netcdf(pairs, out_nc, varname=varname)

            # Record in attrs
            ds.attrs["dineof_cv_path"] = path
            ds.attrs["dineof_cv_var"] = vname
            ds.attrs["dineof_cv_total_points"] = meta["total_cv_points"]
            ds.attrs["dineof_cv_affected_frames"] = meta["affected_frames"]

            print(f"[CV] Saved CV NetCDF: {path}#{vname} ({meta['total_cv_points']} points)")
            print(f"[CV] Spatial indices: 1..{max_spatial_idx} (valid range: 1..{M})")
            print(f"[CV] Add to dineof.init:\n      clouds = '{path}#{vname}'")
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))