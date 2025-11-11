# lswt_processing/dineof_cv.py
"""
DINEOF Cross-Validation (CV) Mask Generator — robust & DINEOF-accurate

- Spatial enumeration: lat-outer, lon-inner (Fortran/MATLAB-like), matching DINEOF.
- NetCDF pairs: (nbpoints, index) == (N, 2), columns [m, t], both 1-based.
- Strict filtering on BOTH CF (masked/scaled) and RAW (mask_and_scale=False):
  reject any pair where S[t] or S[t-1] is NaN OR equals known fill sentinels.
- If prepared.nc doesn’t exist yet, we emit a TEMP prepared snapshot where NaNs
  are ENCODED as a real fill value (e.g., 9999.0) and _FillValue/missing_value
  attrs are set, so RAW view exposes 9999 and CF view exposes NaN.
"""

import os
import tempfile
from typing import Tuple, Optional

import numpy as np
import xarray as xr

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig


# ------------------------------ helpers ------------------------------

def _collect_fill_values(da: xr.DataArray) -> np.ndarray:
    fills = []
    atts = da.attrs
    for key in ("_FillValue", "missing_value"):
        if key in atts:
            val = atts[key]
            if np.isscalar(val):
                fills.append(float(val))
            else:
                fills.extend([float(x) for x in np.atleast_1d(val)])
    fills.extend([9.96921e36, 1.0e36, 1.0e30, -1.0e30, 1.0e20, -1.0e20, 9999.0, -9999.0])
    if not fills:
        return np.array([], dtype=np.float64)
    return np.array(sorted(set(fills)), dtype=np.float64)


def _build_mindex_lat_outer(sea: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fortran/MATLAB-like enumeration: lat outer, lon inner."""
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
    if data_var not in ds:
        raise ValueError(f"{data_var} not in dataset")
    A = ds[data_var]

    # Determine fill to use
    fill_candidates = _collect_fill_values(A)
    if fill_candidates.size:
        fill_value = float(fill_candidates[0])
    else:
        fill_value = 9999.0  # sensible default for our temp snapshot

    # Build minimal Dataset
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
        attrs=dict(A.attrs),  # copy scale/add_offset if any
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
    try:
        os.close(fd)
    except Exception:
        pass

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
        prepared_file_path: Optional[str],
    ) -> Tuple[np.ndarray, dict]:
        """
        1) Paste donor-clouds into clean frames to create 'newly' masked points.
        2) Build DINEOF m (lat-outer, lon-inner).
        3) Per-pair time choice: keep t if S[t] and S[t-1] finite, else try t+1, else drop.
        4) STRICT filter on a real-on-disk prepared file (CF + RAW), rejecting NaNs and sentinels.
        """
        # Apply sea mask and shapes
        S = S.where(sea)
        T, nlat, nlon = S.sizes["time"], S.sizes["lat"], S.sizes["lon"]
        M = int(sea.sum())
        nbland = int((~sea).sum())

        nan_counts = np.isnan(S).sum(dim=("lat", "lon")).values
        cloudcov = (nan_counts - nbland) / max(M, 1)

        if not (1 <= nbclean < T):
            raise ValueError("nbclean must be >=1 and < number of time steps.")

        clean = np.argsort(cloudcov)[:nbclean]
        donors_pool = np.where((cloudcov > 0) & (~np.isin(np.arange(T), clean)))[0]
        if donors_pool.size == 0:
            raise ValueError("No cloudy donor frames available.")
        rng = np.random.default_rng(seed)
        donors = rng.choice(donors_pool, size=nbclean, replace=True)

        # Paste donor clouds
        S_np = S.values  # (time, lat, lon)
        S2_np = S_np.copy()
        for t_clean, t_donor in zip(clean, donors):
            donor_nan = np.isnan(S_np[t_donor, :, :])
            img = S2_np[t_clean, :, :]
            img[donor_nan] = np.nan
            S2_np[t_clean, :, :] = img

        newly = np.isnan(S2_np) & ~np.isnan(S_np)
        t_idx, jj_idx, ii_idx = np.where(newly)  # (time, lat, lon)
        if t_idx.size == 0:
            raise RuntimeError("No newly masked points; increase nbclean or ensure donors have clouds.")

        # Spatial indexing: DINEOF-compatible
        mindex, inv = _build_mindex_lat_outer(sea)
        space_idx = mindex[jj_idx, ii_idx].astype(np.int32)
        time_idx = (t_idx + 1).astype(np.int32)  # 1-based

        kept_m, kept_t, n_keep_t, n_shift_t1, n_drop = self._per_pair_time_align_masked(
            S_np, inv, T, space_idx, time_idx
        )
        if not kept_m:
            raise ValueError("All candidate CV pairs were dropped by masked-domain alignment.")

        clouds_mat = np.column_stack([np.array(kept_m, np.int32),
                                      np.array(kept_t, np.int32)]).astype(np.int32)

        # Hard drop any t<=1 (DINEOF needs t-1)
        if clouds_mat.size:
            clouds_mat = clouds_mat[clouds_mat[:, 1] >= 2]

        # STRICT filter using an on-disk prepared file (real RAW + CF semantics)
        tmp_path = None
        try:
            backing_path, tmp_path = _ensure_prepared_path(S.to_dataset(name=S.name), S.name, prepared_file_path)
            clouds_mat, dropped_raw = self._strict_filter_cf_and_raw(
                prepared_path=backing_path, vname=S.name, inv=inv, clouds_mat=clouds_mat
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass

        meta = {
            "clean_frames": clean,
            "donor_frames": donors,
            "total_cv_points": int(clouds_mat.shape[0]),
            "affected_frames": int(np.unique(clouds_mat[:, 1]).size) if clouds_mat.size else 0,
            "mean_cv_per_frame_pct": 100.0 * (clouds_mat.shape[0] / float(T)) / max(M, 1),
            "cv_keep_t": n_keep_t,
            "cv_shift_t_plus_1": n_shift_t1,
            "cv_dropped": n_drop + int((space_idx.size - clouds_mat.shape[0])),
            "spatial_order": "lat_outer",
        }
        return clouds_mat, meta

    @staticmethod
    def _per_pair_time_align_masked(
        Snp: np.ndarray,
        inv: np.ndarray,
        T: int,
        space_idx: np.ndarray,
        time_idx: np.ndarray,
    ):
        kept_m, kept_t = [], []
        n_keep_t = 0
        n_shift_t1 = 0
        n_drop = 0

        for mi, ti in zip(space_idx.tolist(), time_idx.tolist()):
            jj, ii = inv[int(mi)]
            ok_keep = (2 <= ti <= T) and np.isfinite(Snp[ti - 1, jj, ii]) and np.isfinite(Snp[ti - 2, jj, ii])
            ok_shift = (ti + 1 <= T) and np.isfinite(Snp[ti, jj, ii]) and np.isfinite(Snp[ti - 1, jj, ii])

            if ok_keep:
                kept_m.append(int(mi)); kept_t.append(int(ti)); n_keep_t += 1
            elif ok_shift:
                kept_m.append(int(mi)); kept_t.append(int(ti + 1)); n_shift_t1 += 1
            else:
                n_drop += 1

        return kept_m, kept_t, n_keep_t, n_shift_t1, n_drop

    @staticmethod
    def _strict_filter_cf_and_raw(
        prepared_path: str,
        vname: str,
        inv: np.ndarray,
        clouds_mat: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        if clouds_mat.size == 0:
            return clouds_mat, 0

        ds_cf = xr.open_dataset(prepared_path, decode_times=False, mask_and_scale=True)
        ds_raw = xr.open_dataset(prepared_path, decode_times=False, mask_and_scale=False)

        if vname not in ds_cf.variables:
            raise ValueError(f"Variable '{vname}' not found in prepared file {prepared_path}")

        A_cf = ds_cf[vname].values
        A_raw = ds_raw[vname].values
        T = A_cf.shape[0]

        fill_vec = _collect_fill_values(ds_raw[vname])

        mi = clouds_mat[:, 0].astype(int)
        ti = clouds_mat[:, 1].astype(int)

        jj = inv[mi, 0]
        ii = inv[mi, 1]
        tt = ti - 1
        tm1 = ti - 2

        keep = (ti >= 2) & (ti <= T)

        # CF finite checks (reject NaN)
        keep &= np.isfinite(A_cf[tt, jj, ii])
        keep &= np.isfinite(A_cf[tm1, jj, ii])

        # RAW sentinel checks (reject any match to fill values)
        if fill_vec.size:
            raw_t = A_raw[tt, jj, ii].astype(np.float64)
            raw_m1 = A_raw[tm1, jj, ii].astype(np.float64)
            is_fill_t = np.any(np.isclose(raw_t[:, None], fill_vec[None, :], rtol=0, atol=1e-12), axis=1)
            is_fill_m1 = np.any(np.isclose(raw_m1[:, None], fill_vec[None, :], rtol=0, atol=1e-12), axis=1)
            keep &= ~is_fill_t
            keep &= ~is_fill_m1

        filtered = clouds_mat[keep]
        dropped = int((~keep).sum())

        ds_cf.close(); ds_raw.close()
        return filtered, dropped

    @staticmethod
    def save_pairs_netcdf(pairs_1based: np.ndarray, out_nc: str, varname: str = "cv_pairs") -> Tuple[str, str]:
        p = np.asarray(pairs_1based, dtype=np.int32)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(f"pairs must have shape (N,2); got {p.shape}")
        
        # TRANSPOSE to (2, N) for Fortran dimension reversal
        p_t = p.T  # Shape (2, N)
        
        ds = xr.Dataset(
            {varname: (("index", "nbpoints"), p_t)},  # Changed order and use transposed
            coords={
                "index": np.array([1, 2], dtype=np.int32),  # Swapped order
                "nbpoints": np.arange(1, p.shape[0] + 1, dtype=np.int32),
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

            print(f"[CV] Saved: {path}#{vname} ({meta['total_cv_points']} pts)")
            print(f"[CV] Spatial order: {meta['spatial_order']}")
            print(f"[CV] kept t: {meta['cv_keep_t']}, kept (t+1): {meta['cv_shift_t_plus_1']}, dropped: {meta['cv_dropped']}")
            print(f"[CV] Add to dineof.init: clouds = '{path}#{vname}'")
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
