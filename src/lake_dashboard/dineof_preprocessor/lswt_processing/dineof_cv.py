# lswt_processing/dineof_cv.py
import os
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


class DineofCVGeneratorCore:
    """
    Lightweight, embedded generator core (no external import).
    Builds (m,t) CV pairs compatible with your DINEOF build:
      - variable shape: (2, nbpoints)
      - dims: ("index","nbpoints")
      - index 1 = m (ocean-linear), index 2 = t (1-based time)
    """

    @staticmethod
    def _normalize_mask_da(mask_da: xr.DataArray) -> np.ndarray:
        m = mask_da.load().astype("float64")
        if set(m.dims) != {"lat", "lon"}:
            raise ValueError(f"mask must be (lat,lon), got {m.dims}")
        return (~np.isnan(m.values)) if np.isnan(m.values).any() else (m.values > 0.5)

    @staticmethod
    def _load_cf_scaled(da: xr.DataArray) -> xr.DataArray:
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

        # sanity
        if ds_mask[mask_var].sizes.get("lat") != da.sizes["lat"] or ds_mask[mask_var].sizes.get("lon") != da.sizes["lon"]:
            raise ValueError("Mask (lat,lon) size must match data.")
        ds_data.close()
        ds_mask.close()
        return da, sea

    def generate(self, S: xr.DataArray, sea: np.ndarray, nbclean: int = 3, seed: int = 123) -> Tuple[np.ndarray, dict]:
        # enforce land→NaN, compute donors
        S = S.where(sea)
        T, nlat, nlon = S.sizes["time"], S.sizes["lat"], S.sizes["lon"]
        M = int(sea.sum())
        nbland = int((~sea).sum())

        nan_counts = np.isnan(S).sum(dim=("lat", "lon")).values
        cloudcov = (nan_counts - nbland) / M

        if not (1 <= nbclean < T):
            raise ValueError("nbclean must be >=1 and < number of time steps.")

        clean = np.argsort(cloudcov)[:nbclean]
        donors_pool = np.where((cloudcov > 0) & (~np.isin(np.arange(T), clean)))[0]
        if donors_pool.size == 0:
            raise ValueError("No cloudy donor frames available; dataset may be fully clean.")

        rng = np.random.default_rng(seed)
        donors = rng.choice(donors_pool, size=nbclean, replace=True)

        S_np = S.values
        S2_np = S_np.copy()
        for t_clean, t_donor in zip(clean, donors):
            donor_nan = np.isnan(S_np[t_donor, :, :])
            img = S2_np[t_clean, :, :]
            img[donor_nan] = np.nan
            S2_np[t_clean, :, :] = img

        newly = np.isnan(S2_np) & ~np.isnan(S_np)

        # DINEOF linear indexing: outer over lon, inner over lat → m = m(lon,lat)
        mindex_lonlat = np.zeros((nlon, nlat), dtype=np.int32)
        c = 0
        for ii in range(nlon):
            for jj in range(nlat):
                if sea[jj, ii]:
                    c += 1
                    mindex_lonlat[ii, jj] = c

        t_idx, i_idx, j_idx = np.where(newly)  # i=lat, j=lon
        if t_idx.size == 0:
            raise RuntimeError("No newly masked points; increase nbclean or check donors.")
        space_idx = mindex_lonlat[j_idx, i_idx].astype(np.int32)
        time_idx = (t_idx + 1).astype(np.int32)  # 1-based

        pairs = np.column_stack([space_idx, time_idx]).astype(np.int32)
        meta = {
            "clean_frames": clean,
            "donor_frames": donors,
            "total_cv_points": int(pairs.shape[0]),
            "affected_frames": int(np.unique(time_idx).size),
            "M_ocean_pixels": M,
            "T_frames": int(T),
        }
        return pairs, meta

    def save_pairs_netcdf(self, pairs_1based: np.ndarray, out_nc: str, varname: str = "cv_pairs") -> Tuple[str, str]:
        pairs_t = pairs_1based.astype(np.float32).T  # (2, nbpoints)
        nbpoints = pairs_t.shape[1]
        ds = xr.Dataset(
            {varname: (("index", "nbpoints"), pairs_t)},
            coords={
                "index": np.array([1, 2], dtype=np.int32),
                "nbpoints": np.arange(1, nbpoints + 1, dtype=np.int32),
            },
        )
        os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
        if os.path.exists(out_nc):
            os.remove(out_nc)
        ds.to_netcdf(out_nc, mode="w")
        ds.close()
        return out_nc, varname


class DineofCVGenerationStep(ProcessingStep):
    """
    Pipeline-compatible step:
      - no-op unless config.cv_enable is True
      - reads *raw* input file (config.input_file) to build CV pairs
      - writes NetCDF, leaves ds unchanged
      - records the path/varname in ds.attrs for downstream use
    """

    def should_apply(self, config: ProcessingConfig) -> bool:
        return bool(getattr(config, "cv_enable", False))

    @property
    def name(self) -> str:
        return "DINEOF CV Generation"

    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            if not config.cv_mask_file or not config.cv_mask_var:
                raise ValueError("cv_enable set but cv_mask_file and cv_mask_var are not provided.")

            data_nc = config.input_file            # per your requirement: same input path used elsewhere
            data_var = config.cv_data_var or "lake_surface_water_temperature"
            mask_nc = config.cv_mask_file
            mask_var = config.cv_mask_var
            nbclean = int(config.cv_nbclean or 3)
            seed = int(config.cv_seed or 123)
            out_nc = config.cv_out or os.path.join(os.path.dirname(config.output_file) or ".", "cv_pairs.nc")
            varname = config.cv_varname or "cv_pairs"

            core = DineofCVGeneratorCore()
            S, sea = core.load(data_nc, data_var, mask_nc, mask_var)
            pairs, meta = core.generate(S, sea, nbclean=nbclean, seed=seed)
            path, vname = core.save_pairs_netcdf(pairs, out_nc, varname=varname)

            # record in attrs (non-invasive)
            ds.attrs["dineof_cv_path"] = path
            ds.attrs["dineof_cv_var"] = vname
            ds.attrs["dineof_cv_total_points"] = meta["total_cv_points"]
            ds.attrs["dineof_cv_affected_frames"] = meta["affected_frames"]
            ds.attrs["dineof_cv_M_ocean_pixels"] = meta["M_ocean_pixels"]
            ds.attrs["dineof_cv_T_frames"] = meta["T_frames"]

            print(f"[CV] Saved CV NetCDF: {path}#{vname} ({meta['total_cv_points']} points)")
            print(f"[CV] Add to dineof.init:\n      clouds = '{path}#{vname}'")
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
