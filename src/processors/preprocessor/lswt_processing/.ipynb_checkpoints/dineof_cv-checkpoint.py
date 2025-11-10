# lswt_processing/dineof_cv.py
"""
DINEOF Cross-Validation (CV) Mask Generator
- lon-outer linearisation (matches DINEOF expectation)
- Per-pair time alignment (robust to t vs t-1 checks)
- STRICT filtering against both CF-NaN and RAW fill tokens (e.g., 9999)
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
        """Load data + mask. Returns (S, sea_bool)."""
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

        # Exact DINEOF predicate: lake pixels are finite and > 0
        mask = ds_mask[mask_var].load().astype("float64").values
        sea = np.isfinite(mask) & (mask > 0)

        if ds_mask[mask_var].sizes.get("lat") != da.sizes["lat"] or ds_mask[mask_var].sizes.get("lon") != da.sizes["lon"]:
            raise ValueError("Mask (lat,lon) size must match data.")
        ds_data.close(); ds_mask.close()
        return da, sea

    @staticmethod
    def _build_mindex_lon_outer(sea: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build spatial linear index m (1..M) from a (lat,lon) boolean mask.
        lon-outer, lat-inner. Returns (mindex[lon,lat], inv[1..M]=(lat,lon)).
        """
        nlat, nlon = sea.shape
        mindex = np.zeros((nlon, nlat), dtype=np.int32)  # addressed [lon,lat]
        c = 0
        for ii in range(nlon):          # lon outer
            for jj in range(nlat):      # lat inner
                if sea[jj, ii]:
                    c += 1
                    mindex[ii, jj] = c
        I, J = np.where(mindex > 0)
        inv = np.empty((c + 1, 2), np.int32)            # 1..M
        inv[mindex[I, J]] = np.stack([J, I], 1)         # (lat,lon)
        return mindex, inv

    @staticmethod
    def _per_pair_time_align_masked(Snp: np.ndarray, inv: np.ndarray, T: int,
                                    space_idx: np.ndarray, time_idx: np.ndarray):
        """
        Choose t' per pair so that BOTH S[t'] and S[t'-1] are finite (CF-masked view).
        If not possible, drop the pair. Returns kept arrays + counters.
        """
        kept_m, kept_t = [], []
        n_keep_t = 0       # wrote t' = t
        n_shift_t1 = 0     # wrote t' = t+1
        n_drop = 0

        for mi, ti in zip(space_idx.tolist(), time_idx.tolist()):
            jj, ii = inv[int(mi)]

            # Option A: keep t' = t  → require S[t], S[t-1] finite
            ok_keep = (2 <= ti <= T) and np.isfinite(Snp[ti - 1, jj, ii]) and np.isfinite(Snp[ti - 2, jj, ii])

            # Option B: shift t' = t+1 → require S[t+1], S[t] finite
            ok_shift = (ti + 1 <= T) and np.isfinite(Snp[ti, jj, ii]) and np.isfinite(Snp[ti - 1, jj, ii])

            if ok_keep:
                kept_m.append(int(mi)); kept_t.append(int(ti));       n_keep_t += 1
            elif ok_shift:
                kept_m.append(int(mi)); kept_t.append(int(ti + 1));   n_shift_t1 += 1
            else:
                n_drop += 1

        return kept_m, kept_t, n_keep_t, n_shift_t1, n_drop

    @staticmethod
    def _strict_filter_cf_and_raw(prepared_path: str, vname: str, inv: np.ndarray, clouds_mat: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        STRICT filter against:
          - CF-masked NaNs at t' and at t'-1
          - RAW fill tokens (e.g., 9999, 1e20, etc.) at t' and at t'-1
        """
        if clouds_mat.size == 0:
            return clouds_mat, 0

        # Open in both modes
        ds_plus = xr.open_dataset(prepared_path, decode_times=False, mask_and_scale=True)
        ds_raw  = xr.open_dataset(prepared_path, decode_times=False, mask_and_scale=False)

        A_masked = ds_plus[vname].values  # CF-masked/scaled
        A_raw    = ds_raw [vname].values  # RAW numbers
        T = A_masked.shape[0]

        # Collect fill candidates from RAW attrs + usual sentinels
        atts = ds_raw[vname].attrs
        fills = []
        for key in ("_FillValue", "missing_value"):
            if key in atts:
                val = atts[key]
                if np.isscalar(val):
                    fills.append(float(val))
                else:
                    fills.extend([float(x) for x in np.atleast_1d(val)])
        # Add common sentinels
        fills.extend([9.96921e36, 1e20, -1e20, 9999.0, -9999.0])
        fv = np.array(sorted(set(fills)), dtype=np.float64)

        mi = clouds_mat[:, 0].astype(int)
        ti = clouds_mat[:, 1].astype(int)

        # Map to indices
        jj = inv[mi, 0]; ii = inv[mi, 1]
        tt = ti - 1
        tm1 = ti - 2

        keep = (ti >= 2) & (ti <= T)            # ensure t'-1 exists and t' in range

        # CF finite at t' and t'-1
        keep &= np.isfinite(A_masked[tt,  jj, ii])
        keep &= np.isfinite(A_masked[tm1, jj, ii])

        # RAW not equal to any fill (with tolerance)
        if fv.size:
            raw_t  = A_raw[tt,  jj, ii].astype(np.float64)
            raw_m1 = A_raw[tm1, jj, ii].astype(np.float64)
            is_fill_t  = np.any(np.isclose(raw_t[:,  None], fv[None, :], rtol=0, atol=1e-6), axis=1)
            is_fill_m1 = np.any(np.isclose(raw_m1[:, None], fv[None, :], rtol=0, atol=1e-6), axis=1)
            keep &= ~is_fill_t & ~is_fill_m1

        filtered = clouds_mat[keep]
        dropped = int((~keep).sum())

        ds_plus.close(); ds_raw.close()
        return filtered, dropped

    @staticmethod
    def _validate_no_nan_masked(Snp: np.ndarray, inv: np.ndarray, T: int, clouds_mat: np.ndarray) -> int:
        """Return count of pairs where NOT (S[t'] finite AND S[t'-1] finite)."""
        if clouds_mat.size == 0:
            return 0
        bad = 0
        for mi, ti in clouds_mat:
            jj, ii = inv[int(mi)]
            ok_t   = (1 <= ti <= T) and np.isfinite(Snp[ti - 1, jj, ii])
            ok_tm1 = (2 <= ti <= T) and np.isfinite(Snp[ti - 2, jj, ii])
            if not (ok_t and ok_tm1):
                bad += 1
        return bad

    def generate(self, S: xr.DataArray, sea: np.ndarray, nbclean: int = 3, seed: int = 123) -> Tuple[np.ndarray, dict]:
        """
        Generate CV pairs by pasting donor cloud patterns onto clean frames,
        then align each pair in time and strictly filter against CF-NaNs and RAW fills.
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

        # lon-outer spatial index
        mindex, inv = self._build_mindex_lon_outer(sea)

        # Natural (1-based) indices; mindex is addressed [lon,lat]
        space_idx = mindex[j_idx, i_idx].astype(np.int32)
        time_idx  = (t_idx + 1).astype(np.int32)  # 1-based

        # Per-pair time alignment using CF-masked array (S_np)
        kept_m, kept_t, n_keep_t, n_shift_t1, n_drop = self._per_pair_time_align_masked(
            S_np, inv, T, space_idx, time_idx
        )
        if not kept_m:
            raise ValueError("All candidate CV pairs were dropped in masked alignment.")

        clouds_mat = np.column_stack([np.array(kept_m, np.int32),
                                      np.array(kept_t, np.int32)]).astype(np.int32)

        # STRICT filter against CF-NaN and RAW fills (e.g., 9999)
        # Find backing file path of S
        prep_path = (S.encoding.get("source")
                     or S.encoding.get("filename_or_obj")
                     or S.encoding.get("filepath", None))
        if not prep_path or not os.path.exists(prep_path):
            raise RuntimeError("Cannot locate backing file for prepared dataset; strict RAW/CF filtering requires file path.")

        clouds_mat, dropped_raw = self._strict_filter_cf_and_raw(
            prepared_path=prep_path, vname=S.name, inv=inv, clouds_mat=clouds_mat
        )

        # Final masked validation guard (paranoia)
        bad_masked = self._validate_no_nan_masked(S_np, inv, T, clouds_mat)
        if bad_masked:
            raise ValueError(f"Internal guard: masked view still has {bad_masked} unsafe pairs after strict filtering.")

        meta = {
            "clean_frames": clean,
            "donor_frames": donors,
            "total_cv_points": int(clouds_mat.shape[0]),
            "affected_frames": int(np.unique(clouds_mat[:, 1]).size) if clouds_mat.size else 0,
            "mean_cv_per_frame_pct": 100.0 * (clouds_mat.shape[0] / float(T)) / M if M > 0 else 0.0,
            "cv_keep_t": n_keep_t,
            "cv_shift_t_plus_1": n_shift_t1,
            "cv_dropped": n_drop + dropped_raw,
            "spatial_order": "lon_outer",
        }
        return clouds_mat, meta

    def save_pairs_netcdf(self, pairs_1based, out_nc, varname="cv_pairs"):
        """
        Save as int32 with dims (index, nbpoints) → (2, N), matches earlier successful layout.
        """
        p = np.asarray(pairs_1based, dtype=np.int32)  # (N,2) [m,t]
        pairs_t = p.T  # (2, N)

        ds = xr.Dataset(
            {varname: (("index", "nbpoints"), pairs_t)},
            coords={
                "index":    np.array([1, 2], dtype=np.int32),
                "nbpoints": np.arange(1, pairs_t.shape[1] + 1, dtype=np.int32),
            },
        )
        if os.path.exists(out_nc):
            os.remove(out_nc)
        # lock types; no _FillValue on the data variable
        ds[varname] = ds[varname].astype("int32")
        ds[varname].encoding["_FillValue"] = None
        ds["index"]    = ds["index"].astype("int32")
        ds["nbpoints"] = ds["nbpoints"].astype("int32")
        ds.to_netcdf(out_nc, mode="w", engine="netcdf4")
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

            max_spatial_idx = int(pairs[:, 0].max()) if pairs.size else 0
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
            print(f"[CV] spatial order chosen: {meta['spatial_order']}")
            print(f"[CV] kept t: {meta['cv_keep_t']}, kept (t+1): {meta['cv_shift_t_plus_1']}, dropped: {meta['cv_dropped']}")
            print(f"[CV] Spatial indices: 1..{max_spatial_idx} (valid range: 1..{M})")
            print(f"[CV] Add to dineof.init:\n      clouds = '{path}#{vname}'")
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
