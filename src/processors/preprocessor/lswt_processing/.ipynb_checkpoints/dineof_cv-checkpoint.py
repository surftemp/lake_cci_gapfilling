# lswt_processing/dineof_cv.py
"""
DINEOF Cross-Validation (CV) Mask Generator — Julia-aligned, DINEOF-accurate

- Spatial enumeration: lat-outer, lon-inner (Fortran/MATLAB-like), matching DINEOF.
- NetCDF pairs: (nbpoints, index) == (N, 2), columns [m, t], both 1-based.
- Logic of cloud generation and indexing mirrors the Julia dineof_cvp code:
    * Choose nbclean cleanest time frames (lowest cloud fraction).
    * Randomly choose donor frames (avoiding clean frames).
    * Paste donor cloud patterns into clean frames to create new missing values.
    * Build (m, t) pairs for the newly missing points using a DINEOF-style mindex.

- The outer pipeline still supports:
    * cv_fraction_target and optional cv_absolute_cap,
    * computing a valid-pool size using strict RAW/CF semantics on prepared.nc,
    * saving metadata into the xarray.Dataset attributes.
"""

import os
import tempfile
from typing import Tuple, Optional

import numpy as np
import xarray as xr

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


# ------------------------------ helpers ------------------------------
def _build_mindex_lat_outer(sea: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fortran/MATLAB-like enumeration: lat outer, lon inner.

    sea: 2D boolean or {0,1} array (lat, lon), True/1 for water/valid, False/0 for land.
    Returns:
        mindex: (lat, lon) -> m (1..M), 0 for land
        inv:    (M+1, 2)   with inv[m] = (j, i) in 0-based (lat, lon) index
    """
    nlat, nlon = sea.shape
    mindex = np.zeros((nlat, nlon), dtype=np.int32)
    c = 0
    for jj in range(nlat):       # lat outer
        for ii in range(nlon):   # lon inner
            if sea[jj, ii]:
                c += 1
                mindex[jj, ii] = c
    I, J = np.where(mindex > 0)
    inv = np.empty((c + 1, 2), np.int32)  # 1..M
    inv[mindex[I, J]] = np.stack([I, J], axis=1)  # (lat, lon)
    return mindex, inv


def _write_temp_prepared_snapshot(ds: xr.Dataset, data_var: str, target_dir: Optional[str]) -> str:
    """
    Create a temp prepared file from in-memory ds that behaves like a real 'prepared.nc'
    for both RAW and CF reads:
      - NaNs are encoded as a real fill value (prefer attr, else 9999.0).
      - _FillValue/missing_value set on the variable.
      - CF view will show NaN; RAW view shows the numeric fill (e.g., 9999.0).
    Only writes coords (time, lat, lon) and the data_var to keep it small.
    """
    A = ds[data_var]
    fill_value = 9999.0 

    # minimal Dataset
    coords = {}
    for c in ("time", "lat", "lon"):
        if c in ds:
            coords[c] = ds[c].copy()

    raw_vals = A.values.astype(np.float32, copy=True)
    # Encode NaNs as fill_value for RAW semantics
    nan_mask = ~np.isfinite(raw_vals)
    if nan_mask.any():
        raw_vals[nan_mask] = fill_value

    A_out = xr.DataArray(
        raw_vals,
        dims=A.dims,
        coords=coords,
        name=data_var,
        attrs=dict(A.attrs),  # <-- this was the buggy line
    )
    # Ensure fill attrs present
    A_out.attrs["_FillValue"] = fill_value
    A_out.attrs["missing_value"] = fill_value

    ds_out = xr.Dataset({data_var: A_out}, coords=coords)

    # Write file
    if target_dir and target_dir.strip():
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_prepared_", suffix=".nc", dir=target_dir)
    else:
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_prepared_", suffix=".nc")
        
    os.close(fd)


    # Enforce dtype + encodings so CF/RAW behave as intended
    enc = {data_var: {"_FillValue": fill_value, "dtype": "float32"}}
    ds_out.to_netcdf(tmp_path, encoding=enc)
    return tmp_path


def _ensure_prepared_path(ds: xr.Dataset, data_var: str, prepared_file_path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Ensure a readable on-disk file for strict RAW/CF validation:
    - If prepared_file_path exists: use it; return (path, None)
    - Else: write a TEMP snapshot (NaNs encoded as fill); return (tmp_path, tmp_path) so caller can delete it.
    """
    if prepared_file_path and os.path.exists(prepared_file_path):
        return prepared_file_path, None
    base_dir = os.path.dirname(prepared_file_path) if prepared_file_path else None
    tmp_path = _write_temp_prepared_snapshot(ds, data_var, base_dir)
    return tmp_path, tmp_path


# ------------------------------ core generator ------------------------------

class DineofCVGeneratorCore:
    def generate(
        self,
        S: xr.DataArray,
        sea: np.ndarray,
        nbclean: int,
        seed: int,
        prepared_file_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """

        Mirrors the Julia dineof_cvp logic:

        1) Apply sea mask: set non-sea pixels to NaN in all frames.
        2) Compute cloudcov(t): fraction of NaNs (minus static land) per time.
        3) Select nbclean cleanest frames (lowest cloudcov).
        4) Randomly choose donor frames (avoiding clean frames).
        5) Paste donor clouds (NaNs) into clean frames, creating newly-masked points.
        6) Build DINEOF mindex (lat-outer, lon-inner).
        7) For every newly masked point, record (m, t) with 1-based time index.

        Arguments
        ---------
        S : xr.DataArray
            Data array with dims ("time", "lat", "lon").
        sea : np.ndarray
            2D boolean array (lat, lon). True for water (mask==1 in Julia).
        nbclean : int
            Number of cleanest images to be covered with clouds.
        seed : int
            RNG seed for donor selection.
        prepared_file_path : Optional[str]
            Kept for API compatibility, unused in this simplified core.

        Returns
        -------
        clouds_mat : np.ndarray of shape (N, 2)
            Each row is [m, t], both 1-based, DINEOF-style.
        meta : dict
            Metadata, including counts and some basic diagnostics.
        """

        # 1. Shapes and mask

        T = int(S.sizes["time"])
        nlat = int(S.sizes["lat"])
        nlon = int(S.sizes["lon"])

        S_np = S.values.astype(np.float64, copy=True)  # (time, lat, lon)

        if sea.shape != (nlat, nlon):
            raise ValueError(f"'sea' mask shape {sea.shape} does not match (lat,lon)=({nlat},{nlon})")

        land = ~sea
        # Apply sea mask: land -> NaN at all times
        for t in range(T):
            img = S_np[t, :, :]
            img[land] = np.nan
            S_np[t, :, :] = img

        nbland = int(land.sum())
        mmax = int(sea.sum())
        if mmax == 0:
            raise ValueError("No sea pixels in 'sea' mask.")

        # 2. Cloud coverage per time frame
        nan_counts = np.isnan(S_np).sum(axis=(1, 2))  # shape (T,)
        cloudcov = (nan_counts - nbland) / float(mmax)

        if not (1 <= nbclean < T):
            raise ValueError("nbclean must be >=1 and < number of time steps.")

        clean = np.argsort(cloudcov)[:nbclean]  # indices of cleanest frames

        # 3. Random donor indices, avoiding clean
        rng = np.random.default_rng(seed)
        Ntime = T
        donors = rng.integers(0, Ntime, size=nbclean)
        while np.any(np.isin(donors, clean)):
            donors = rng.integers(0, Ntime, size=nbclean)

        # 4. Build S2 by pasting donor clouds into clean frames
        S2_np = S_np.copy()
        for t_clean, t_donor in zip(clean, donors):
            donor_nan = np.isnan(S_np[t_donor, :, :])
            img = S2_np[t_clean, :, :]
            img[donor_nan] = np.nan
            S2_np[t_clean, :, :] = img

        # 5. Newly masked points: NaN in S2 but finite in original S
        newly = np.isnan(S2_np) & ~np.isnan(S_np)
        t_idx, jj_idx, ii_idx = np.where(newly)  # (time, lat, lon)
        nbpoints = t_idx.size
        if nbpoints == 0:
            raise RuntimeError("No newly masked points; increase nbclean or ensure donors have clouds.")

        # 6. Spatial enumeration (mindex) as in Julia
        mindex, _ = _build_mindex_lat_outer(sea)

        clouds_mat = np.zeros((nbpoints, 2), dtype=np.int32)
        out_count = 0
        for l, (ti, jj, ii) in enumerate(zip(t_idx, jj_idx, ii_idx)):
            m = mindex[jj, ii]
            if m <= 0:
                continue
            clouds_mat[out_count, 0] = int(m)        # m index (1..M)
            clouds_mat[out_count, 1] = int(ti + 1)   # time index, 1-based
            out_count += 1

        clouds_mat = clouds_mat[:out_count, :]

        # 7. Diagnostics
        nbgood = int(np.isfinite(S_np).sum())
        nbgood2 = int(np.isfinite(S2_np).sum())
        pct_added = 100.0 * (nbgood - nbgood2) / max(nbgood, 1)

        meta = {
            "clean_frames": clean,
            "donor_frames": donors,
            "total_cv_points": int(clouds_mat.shape[0]),
            "affected_frames": int(np.unique(clouds_mat[:, 1]).size) if clouds_mat.size else 0,
            "mean_cv_per_frame_pct": 100.0 * (clouds_mat.shape[0] / float(T)) / max(mmax, 1),
            "cv_keep_t": 0,                  # kept for API / print compatibility
            "cv_shift_t_plus_1": 0,
            "cv_dropped": 0,
            "spatial_order": "lat_outer",
            "pct_cloud_cover_added": pct_added,
        }
        return clouds_mat, meta

    @staticmethod
    def save_pairs_netcdf(pairs_1based: np.ndarray, out_nc: str, varname: str = "cv_pairs") -> Tuple[str, str]:
        """
        Save CV pairs to NetCDF in DINEOF/Julia-compatible layout:

        - pairs_1based: array of shape (N, 2), columns [m, t], both 1-based.
        - NetCDF variable shape: (nbpoints, index) == (N, 2).
        """
        p = np.asarray(pairs_1based, dtype=np.int32)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(f"pairs must have shape (N,2); got {p.shape}")

        N = p.shape[0]

        ds = xr.Dataset(
            {varname: (("nbpoints", "index"), p)},  # shape (N, 2)
            coords={
                "nbpoints": np.arange(1, N + 1, dtype=np.int32),   # 1..N
                "index": np.array([1, 2], dtype=np.int32),         # 1=m, 2=t
            },
        )

        if os.path.exists(out_nc):
            os.remove(out_nc)

        ds[varname] = ds[varname].astype("int32")
        ds[varname].encoding["_FillValue"] = None
        ds["nbpoints"] = ds["nbpoints"].astype("int32")
        ds["index"] = ds["index"].astype("int32")

        ds.to_netcdf(out_nc, mode="w", engine="netcdf4")
        ds.close()
        return out_nc, varname


# ------------------------------ pipeline step ------------------------------

class DineofCVGenerationStep(ProcessingStep):
    def should_apply(self, config: ProcessingConfig) -> bool:
        return bool(getattr(config, "cv_enable", False))

    @property
    def name(self) -> str:
        return "DINEOF CV Generation"

    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            if not config.cv_mask_var:
                raise ValueError("cv_enable is set but cv_mask_var is not provided.")

            data_var = config.cv_data_var or "lake_surface_water_temperature"
            mask_var = config.cv_mask_var
            nbclean = int(config.cv_nbclean or 3)
            seed = int(config.cv_seed or 123)
            out_nc = config.cv_out or os.path.join(os.path.dirname(config.output_file) or ".", "cv_pairs.nc")
            varname = config.cv_varname or "cv_pairs"

            print(f"[CV] Generating CV pairs: data='{data_var}', mask='{mask_var}', nbclean={nbclean}, seed={seed}")

            if data_var not in ds.variables:
                raise ValueError(f"'{data_var}' not found in processed dataset.")
            if mask_var not in ds.variables:
                raise ValueError(f"'{mask_var}' not found in processed dataset.")

            S = ds[data_var].load()
            mask_da = ds[mask_var].load()

            mask = mask_da.astype("float64").values
            sea = np.isfinite(mask) & (mask > 0)

            M = int(sea.sum())
            print(f"[CV] Sea pixels (M): {M}")

            core = DineofCVGeneratorCore()
            pairs, meta = core.generate(
                S,
                sea,
                nbclean=nbclean,
                seed=seed,
                prepared_file_path=getattr(config, "output_file", None),
            )

            # --- aim for cv_fraction_target (closest from below), with optional absolute cap ---
            cv_fraction_target = float(getattr(config, "cv_fraction_target", 1.0))
            cv_absolute_cap    = getattr(config, "cv_absolute_cap", None)

            # Build strict-valid pool using RAW/CF semantics on prepared file
            tmp_path = None
            try:
                backing_path, tmp_path = _ensure_prepared_path(
                    S.to_dataset(name=S.name), S.name, getattr(config, "output_file", None)
                )
                ds_cf  = xr.open_dataset(backing_path, decode_times=False, mask_and_scale=True)
                ds_raw = xr.open_dataset(backing_path, decode_times=False, mask_and_scale=False)

                A_cf  = ds_cf[S.name].values  # (T,lat,lon)
                A_raw = ds_raw[S.name].values
                T = A_cf.shape[0]

                FILL = 9999.0
                
                SEA = np.broadcast_to(sea[None, :, :], (T,) + sea.shape)
                
                # CF view: finite at t and t-1
                valid_cf = np.isfinite(A_cf)
                valid_cf_tm1 = np.vstack([np.zeros((1,) + sea.shape, dtype=bool), valid_cf[:-1]])
                valid = SEA & valid_cf & valid_cf_tm1
                
                # RAW view: explicitly reject 9999.0 at t and t-1
                raw = A_raw.astype(np.float64)
                eq_fill_t = np.isclose(raw, FILL, rtol=0, atol=1e-6)
                eq_fill_tm1 = np.vstack([np.zeros((1,) + sea.shape, dtype=bool), eq_fill_t[:-1]])
                valid &= ~eq_fill_t & ~eq_fill_tm1

                valid[0, :, :] = False  # need t-1
                valid_pool_size = int(valid.sum())
            finally:
                try:
                    ds_cf.close(); ds_raw.close()
                except Exception:
                    pass
                if tmp_path and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception:
                        pass

            # --- convert fraction & cap into a soft target K_target ---
            K_target = int(np.floor(cv_fraction_target * valid_pool_size))
            if cv_absolute_cap is not None:
                K_target = min(K_target, int(cv_absolute_cap))
            K_target = max(0, K_target)

            k0 = int(pairs.shape[0])
            print(f"[CV] Initial CV pairs: {k0}, target K_target={K_target}, valid_pool_size={valid_pool_size}")

            # downsample only if overshoot. never re-generate with new nbclean if not enough.
            if K_target < k0:
                rng = np.random.default_rng(seed)
                if K_target > 0:
                    take = rng.choice(k0, size=K_target, replace=False)
                    pairs = pairs[take, :]
                else:
                    pairs = pairs[:0, :]
                meta["total_cv_points"] = int(pairs.shape[0])
            else:
                # Under target: accept what nbclean produced
                meta["total_cv_points"] = k0


            if pairs.size:
                max_m = int(pairs[:, 0].max())
                if max_m > M:
                    raise ValueError(f"CV pairs produced m={max_m} but only {M} sea pixels exist.")

            path, vname = core.save_pairs_netcdf(pairs, out_nc, varname=varname)

            ds.attrs["dineof_cv_path"] = path
            ds.attrs["dineof_cv_var"] = vname
            ds.attrs["dineof_cv_total_points"] = meta["total_cv_points"]
            ds.attrs["dineof_cv_affected_frames"] = meta["affected_frames"]
            ds.attrs["dineof_cv_M_ocean_pixels"] = M
            ds.attrs["dineof_cv_T_frames"] = int(S.sizes["time"])
            ds.attrs["dineof_cv_fraction_target"] = float(getattr(config, "cv_fraction_target", 1.0))
            if getattr(config, "cv_absolute_cap", None) is not None:
                ds.attrs["dineof_cv_absolute_cap"] = int(getattr(config, "cv_absolute_cap"))

            print(f"[CV] Saved: {path}#{vname} ({meta['total_cv_points']} pts)")
            print(f"[CV] Spatial order: {meta['spatial_order']}")
            print(f"[CV] kept t: {meta['cv_keep_t']}, kept (t+1): {meta['cv_shift_t_plus_1']}, dropped: {meta['cv_dropped']}")
            print(f"[CV] Add to dineof.init: clouds = '{path}#{vname}'")
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
