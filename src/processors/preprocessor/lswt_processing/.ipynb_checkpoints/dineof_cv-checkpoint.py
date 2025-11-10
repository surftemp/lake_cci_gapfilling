# lswt_processing/dineof_cv.py
"""
DINEOF Cross-Validation (CV) Mask Generator
"""

import os
from typing import Tuple
import numpy as np
import xarray as xr

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


class DineofCVGeneratorCore:
    @staticmethod
    def _load_cf_scaled(da: xr.DataArray) -> xr.DataArray:
        """Apply CF scale_factor/add_offset and convert fill/missing to NaN."""
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

        # Exact DINEOF predicate
        mask = ds_mask[mask_var].load().astype("float64").values
        sea = np.isfinite(mask) & (mask > 0)

        if ds_mask[mask_var].sizes.get("lat") != da.sizes["lat"] or ds_mask[mask_var].sizes.get("lon") != da.sizes["lon"]:
            raise ValueError("Mask (lat,lon) size must match data.")
        ds_data.close(); ds_mask.close()
        return da, sea

    def generate(self, S: xr.DataArray, sea: np.ndarray, nbclean: int = 3, seed: int = 123) -> Tuple[np.ndarray, dict]:
        """
        Generate CV pairs by pasting donor cloud patterns onto clean frames,
        then choose t' per pair so that BOTH S[t'] and S[t'-1] are finite.
        """
        # Apply lake mask; compute sizes
        S = S.where(sea)
        T, nlat, nlon = S.sizes["time"], S.sizes["lat"], S.sizes["lon"]
        M = int(sea.sum()); nbland = int((~sea).sum())

        # Cloud coverage
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

        # Paste donor clouds onto the clean frames
        S_np = S.values
        S2_np = S_np.copy()
        for t_clean, t_donor in zip(clean, donors):
            donor_nan = np.isnan(S_np[t_donor, :, :])
            img = S2_np[t_clean, :, :]
            img[donor_nan] = np.nan
            S2_np[t_clean, :, :] = img

        newly = np.isnan(S2_np) & ~np.isnan(S_np)
        t_idx, i_idx, j_idx = np.where(newly)
        if t_idx.size == 0:
            raise RuntimeError("No newly masked points; try larger nbclean or ensure donors have clouds.")

        # DINEOF spatial indexing: lon-outer, lat-inner (1-based)
        mindex_lonlat = np.zeros((nlon, nlat), dtype=np.int32)
        c = 0
        for ii in range(nlon):
            for jj in range(nlat):
                if sea[jj, ii]:
                    c += 1
                    mindex_lonlat[ii, jj] = c

        # Natural (1-based) indices from numpy indices
        space_idx = mindex_lonlat[j_idx, i_idx].astype(np.int32)
        time_idx  = (t_idx + 1).astype(np.int32)  # 1-based

        # ---------- Per-pair robust alignment (binary-agnostic) ----------
        # Build inverse m -> (lat,lon)
        I, J = np.where(mindex_lonlat > 0)
        inv = np.empty((M + 1, 2), np.int32)  # 1..M
        inv[mindex_lonlat[I, J]] = np.stack([J, I], 1)  # (lat,lon)

        kept_m, kept_t = [], []
        n_keep_t = 0       # wrote t' = t
        n_shift_t1 = 0     # wrote t' = t+1
        n_drop = 0

        for mi, ti in zip(space_idx.tolist(), time_idx.tolist()):
            jj, ii = inv[int(mi)]

            # Choice A: write t' = t  (DINEOF could read t' or t'-1 ⇒ need S[t] and S[t-1] finite)
            ok_keep = (2 <= ti <= T) and np.isfinite(S_np[ti - 1, jj, ii]) and np.isfinite(S_np[ti - 2, jj, ii])

            # Choice B: write t' = t+1 (DINEOF could read t' or t'-1 ⇒ need S[t+1] and S[t] finite)
            ok_shift = (ti + 1 <= T) and np.isfinite(S_np[ti, jj, ii]) and np.isfinite(S_np[ti - 1, jj, ii])

            if ok_keep:
                kept_m.append(int(mi)); kept_t.append(int(ti));       n_keep_t += 1
            elif ok_shift:
                kept_m.append(int(mi)); kept_t.append(int(ti + 1));   n_shift_t1 += 1
            else:
                n_drop += 1

        if not kept_m:
            raise ValueError("All CV pairs unsafe (no t' with both frames finite).")

        clouds_mat = np.column_stack([
            np.array(kept_m, np.int32),
            np.array(kept_t, np.int32)
        ]).astype(np.int32)

        # Hard guard: every kept pair has BOTH S[t'] and S[t'-1] finite
        bad_final = 0
        for mi, ti in clouds_mat:
            jj, ii = inv[int(mi)]
            if not ((1 <= ti <= T) and np.isfinite(S_np[ti - 1, jj, ii]) and
                    (2 <= ti <= T) and np.isfinite(S_np[ti - 2, jj, ii])):
                bad_final += 1
        assert bad_final == 0, f"CV contains {bad_final} unsafe points after alignment."

        meta = {
            "clean_frames": clean,
            "donor_frames": donors,
            "total_cv_points": int(clouds_mat.shape[0]),
            "affected_frames": int(np.unique(np.asarray(clouds_mat[:, 1])).size),
            "mean_cv_per_frame_pct": 100.0 * (clouds_mat.shape[0] / float(T)) / M,
            "cv_keep_t": n_keep_t,
            "cv_shift_t_plus_1": n_shift_t1,
            "cv_dropped": n_drop,
        }
        return clouds_mat, meta

    def save_pairs_netcdf(self, pairs_1based: np.ndarray, out_nc: str, varname: str = "cv_pairs") -> Tuple[str, str]:
        """Save CV pairs as (index=2, nbpoints=N) float32; rows: 0->m, 1->t."""
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
    def should_apply(self, config: ProcessingConfig) -> bool:
        return bool(getattr(config, "cv_enable", False))

    @property
    def name(self) -> str:
        return "DINEOF CV Generation"

    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        """Generate CV pairs from the **processed** dataset (must match DINEOF input)."""
        try:
            if not config.cv_mask_var:
                raise ValueError("cv_enable set but cv_mask_var is not provided.")

            data_var = config.cv_data_var or "lake_surface_water_temperature"
            mask_var = config.cv_mask_var
            nbclean  = int(config.cv_nbclean or 3)
            seed     = int(config.cv_seed or 123)
            out_nc   = config.cv_out or os.path.join(os.path.dirname(config.output_file) or ".", "cv_pairs.nc")
            varname  = config.cv_varname or "cv_pairs"

            print(f"[CV] Generating cross-validation pairs from PROCESSED dataset")
            print(f"[CV] Using mask variable '{mask_var}' and data variable '{data_var}'")

            if data_var not in ds.variables:
                raise ValueError(f"'{data_var}' not found in processed dataset")
            if mask_var not in ds.variables:
                raise ValueError(f"'{mask_var}' not found in processed dataset")

            S = ds[data_var].load()
            mask_da = ds[mask_var].load()

            # Exact DINEOF predicate
            mask = mask_da.astype("float64").values
            sea  = np.isfinite(mask) & (mask > 0)

            M = int(sea.sum())
            print(f"[CV] Lake pixels in processed data: {M}")

            core = DineofCVGeneratorCore()
            pairs, meta = core.generate(S, sea, nbclean=nbclean, seed=seed)

            max_spatial_idx = int(pairs[:, 0].max())
            if max_spatial_idx > M:
                raise ValueError(
                    f"CV generation produced spatial index {max_spatial_idx} "
                    f"but only {M} lake pixels exist!"
                )

            path, vname = core.save_pairs_netcdf(pairs, out_nc, varname=varname)

            # record
            ds.attrs["dineof_cv_path"] = path
            ds.attrs["dineof_cv_var"] = vname
            ds.attrs["dineof_cv_total_points"] = meta["total_cv_points"]
            ds.attrs["dineof_cv_affected_frames"] = meta["affected_frames"]
            ds.attrs["dineof_cv_M_ocean_pixels"] = M
            ds.attrs["dineof_cv_T_frames"] = int(S.sizes["time"])

            print(f"[CV] Saved CV NetCDF: {path}#{vname} ({meta['total_cv_points']} points)")
            print(f"[CV] kept t: {meta['cv_keep_t']}, kept (t+1): {meta['cv_shift_t_plus_1']}, dropped: {meta['cv_dropped']}")
            print(f"[CV] Spatial indices: 1..{max_spatial_idx} (valid range: 1..{M})")
            print(f"[CV] Add to dineof.init:\n      clouds = '{path}#{vname}'")
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
