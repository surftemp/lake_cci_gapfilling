# post_steps/reconstruct_from_eofs.py
from __future__ import annotations

import os
import glob
import numpy as np
import xarray as xr
from typing import Optional, List, Tuple, Dict, Any
from .base import PostProcessingStep, PostContext


class ReconstructFromEOFsStep(PostProcessingStep):
    """
    Reconstruct (anomaly) field from temporal/spatial EOFs (+ eigenvalues)
    and write a new file 'dineof_results_eof_filtered.nc' that mirrors the
    structure/attrs of the original dineof_results.nc.

    Trigger use-cases:
      1) EOF filtering applied (preferred input = filtered EOFs).
      2) Interpolation of EOFs required (dummy toggle supported).

    Notes:
      - Does NOT modify the main merged ds. Operates on disk side artifacts.
      - No new attrs are invented; global/var attrs are copied from the original results file.
    """

    name = "ReconstructFromEOFs"

    def __init__(self, *, require_when_filtered: bool = True, require_when_interp: bool = False):
        self.require_when_filtered = require_when_filtered
        self.require_when_interp = require_when_interp

    # ---------- plumbing ----------

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        # Apply if:
        #  (A) filtering was enabled AND produced/left an EOFs file, or
        #  (B) user asked for the (future) "interpolated EOFs" path (dummy toggle here).
        base_dir = os.path.dirname(ctx.dineof_output_path)
        have_results = os.path.isfile(ctx.dineof_output_path)
        if not have_results:
            return False

        eofs_path = self._prefer_filtered_eofs(ctx, base_dir)
        if self.require_when_filtered and eofs_path:
            return True
        if self.require_when_interp:
            # placeholder for later: if/when an "interpolated EOFs" artifact exists
            return True
        return False

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        base_dir = os.path.dirname(ctx.dineof_output_path)
        eofs_path = self._prefer_filtered_eofs(ctx, base_dir)
        print("eof path", eofs_path)
        if eofs_path.endswith("eofs_interpolated.nc"):
            print("eofs_interpolated found")
            target_path = os.path.join(base_dir, "dineof_results_eof_interp_full.nc")
        elif eofs_path.endswith("eofs_filtered.nc"):
            print("eofs_filtered found")
            target_path = os.path.join(base_dir, "dineof_results_eof_filtered.nc")
        else:
            target_path = os.path.join(base_dir, "dineof_results_from_eofs.nc")      
            
        if eofs_path is None:
            print(f"[{self.name}] No EOFs file found to reconstruct from; skipping.")
            return ds if ds is not None else xr.Dataset()

        # target_path = os.path.join(base_dir, "dineof_results_eof_filtered.nc")

        # --- open EOFs
        try:
            eofs = xr.open_dataset(eofs_path)
        except Exception as e:
            print(f"[{self.name}] Failed to open EOFs: {eofs_path}: {e}")
            return ds if ds is not None else xr.Dataset()

        # detect available modes
        temporal_vars = [v for v in eofs.data_vars
                         if v.startswith("temporal_eof") and ("t" in eofs[v].dims)]
        spatial_vars  = [v for v in eofs.data_vars
                         if v.startswith("spatial_eof") and (set(("y","x")) <= set(eofs[v].dims))]

        if not temporal_vars or not spatial_vars or "eigenvalues" not in eofs:
            print(f"[{self.name}] Missing EOF components; skipping.")
            eofs.close()
            return ds if ds is not None else xr.Dataset()

        # sort by suffix index to maintain consistent pairing
        def _mode_id(name: str) -> int:
            return int(name.split("eof")[-1])

        temporal_vars.sort(key=_mode_id)
        spatial_vars.sort(key=_mode_id)

        modes = [ _mode_id(v) for v in temporal_vars ]
        # align lists in case of partial overlap
        spatial_vars = [f"spatial_eof{m}" for m in modes if f"spatial_eof{m}" in spatial_vars]
        temporal_vars = [f"temporal_eof{m}" for m in modes if f"temporal_eof{m}" in temporal_vars]
        modes = [m for m in modes if (f"temporal_eof{m}" in temporal_vars and f"spatial_eof{m}" in spatial_vars)]

        if not modes:
            print(f"[{self.name}] No matching temporal/spatial EOF mode pairs; skipping.")
            eofs.close()
            return ds if ds is not None else xr.Dataset()

        # --- build temporal (time x mode)
        # Prefer real time coord if present; fall back to t index
        if "time" in eofs.coords and ("t" in eofs["time"].dims or eofs["time"].sizes.get("time") == eofs.sizes.get("t")):
            time_coord = eofs["time"].values
        else:
            time_coord = np.arange(eofs.dims["t"], dtype=np.int64)

        T = np.stack([eofs[v].values for v in temporal_vars], axis=1)  # (t, K)

        # --- build spatial (mode x y x x)
        S_list = [eofs[v].values for v in spatial_vars]                # each (y, x)
        S = np.stack(S_list, axis=0)                                   # (K, y, x)

        # --- singular values from eigenvalues
        eig = eofs["eigenvalues"].values
        # guard length
        K = T.shape[1]
        sigma = np.sqrt(eig[:K])
        # scale temporal by sigma
        T_scaled = T * sigma[np.newaxis, :]

        # --- reconstruct via tensordot: (t,K) · (K,y,x) -> (t,y,x)
        recon = np.tensordot(T_scaled, S, axes=([1], [0]))             # (t, y, x)

        # --- wrap in DataArray with coords
        y_name, x_name = self._infer_yx_names(eofs)
        da_recon = xr.DataArray(
            data=recon.astype("float32"),
            dims=["time", y_name, x_name],
            coords={
                "time": time_coord,
                y_name: eofs[y_name].values if y_name in eofs.coords else np.arange(recon.shape[1]),
                x_name: eofs[x_name].values if x_name in eofs.coords else np.arange(recon.shape[2]),
            },
            name="temp_filled",
        )

        eofs.close()

        # --- mirror original dineof_results.nc structure
        try:
            with xr.open_dataset(ctx.dineof_output_path) as orig:
                # align to template using 'temp_filled' (if orig doesn’t have it, this still works)
                da_recon = self._align_to_template(da_recon, orig, "temp_filled")
            
                ds_out = xr.Dataset({"temp_filled": da_recon})
            
                # copy global attrs
                ds_out.attrs = dict(orig.attrs)
                # copy var attrs if present on the original
                if "temp_filled" in orig:
                    ds_out["temp_filled"].attrs = dict(orig["temp_filled"].attrs)
            
                enc = {"temp_filled": {"dtype": "float32", "zlib": True, "complevel": 4}}
                ds_out.to_netcdf(target_path, encoding=enc)
                print(f"[{self.name}] Wrote {target_path} (var='temp_filled')")
        except Exception as e:
            print(f"[{self.name}] Failed to mirror/write result: {e}")

        return ds if ds is not None else xr.Dataset()

    # ---------- helpers ----------

    def _prefer_filtered_eofs(self, ctx: PostContext, base_dir: str) -> Optional[str]:
        # prefer interpolated → filtered → raw
        p_interp = os.path.join(base_dir, "eofs_interpolated.nc")
        if os.path.isfile(p_interp): return p_interp
        p_filt = os.path.join(base_dir, "eofs_filtered.nc")
        if os.path.isfile(p_filt): return p_filt
        p_raw  = os.path.join(base_dir, "eofs.nc")
        if os.path.isfile(p_raw):  return p_raw
        return None

    def _infer_yx_names(self, ds: xr.Dataset) -> Tuple[str, str]:
        for yx in (("y","x"), ("lat","lon")):
            if all(n in ds.dims or n in ds.coords for n in yx):
                return yx
        # fallback
        return ("y", "x")

    def _pick_main_var(self, ds: xr.Dataset) -> Optional[str]:
        # pick first 3D var with ('time', y, x)-like dims
        for name, da in ds.data_vars.items():
            dims = tuple(da.dims)
            if len(dims) == 3 and "time" in dims:
                return name
        return None

    def _align_to_template(self, da: xr.DataArray, tmpl: xr.Dataset, var_name: str) -> xr.DataArray:
        # Reindex/rename dims to match template variable (preserve lon/lat grid & time)
        if var_name not in tmpl:
            return da

        t_da = tmpl[var_name]
        # rename dims if necessary
        rename_map = {}
        dims_target = t_da.dims
        dims_src = da.dims
        # assume structure ('time', y, x) both sides; just map second/third dims by name similarity
        for src, tgt in zip(dims_src, dims_target):
            if src != tgt:
                rename_map[src] = tgt
        if rename_map:
            da = da.rename(rename_map)

        # reindex to template coords where available
        for coord in t_da.dims:
            if coord in t_da.coords and coord in da.coords:
                da = da.reindex({coord: t_da.coords[coord]}, method=None)

        return da
