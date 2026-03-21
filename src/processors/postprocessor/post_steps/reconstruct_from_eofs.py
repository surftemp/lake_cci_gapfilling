# post_steps/reconstruct_from_eofs.py
# FIXED: Adds back the mean offset that DINEOF subtracts before SVD
from __future__ import annotations

import os
import glob
import numpy as np
import netCDF4
import xarray as xr
from typing import Optional, List, Tuple, Dict, Any
from .base import PostProcessingStep, PostContext, get_current_rss_mb


class ReconstructFromEOFsStep(PostProcessingStep):
    """
    Reconstruct (anomaly) field from temporal/spatial EOFs (+ eigenvalues)
    and write a dineof_results_*.nc file that mirrors the structure/attrs 
    of the original dineof_results.nc.

    Source modes:
      - 'filtered'        -> reads eofs_filtered.nc, writes dineof_results_eof_filtered.nc
      - 'interp'          -> reads eofs_interpolated.nc, writes dineof_results_eof_interp_full.nc
      - 'filtered_interp' -> reads eofs_filtered_interpolated.nc, writes dineof_results_eof_filtered_interp_full.nc

    FIXED: Now computes and adds back the mean offset that DINEOF subtracts before SVD.
    This ensures reconstructed outputs match the original DINEOF output when no filtering is applied.

    Notes:
      - Does NOT modify the main merged ds. Operates on disk side artifacts.
      - No new attrs are invented; global/var attrs are copied from the original results file.
    """

    name = "ReconstructFromEOFs"  # Base name, will be overridden in __init__

    def __init__(self, *, source_mode: str = "filtered",
                 eofs_prefix: str = "eofs", output_cv_suffix: str = ""):
        assert source_mode in ("filtered", "interp", "filtered_interp")
        self.source_mode = source_mode
        self.eofs_prefix = eofs_prefix
        self.output_cv_suffix = output_cv_suffix
        # Dynamic name for skip_steps logic in multi-pass
        self.name = f"ReconstructFromEOFs_{source_mode}"
        if output_cv_suffix:
            self.name += output_cv_suffix

    # ---------- plumbing ----------

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        base_dir = os.path.dirname(ctx.dineof_output_path)
        have_results = os.path.isfile(ctx.dineof_output_path)
        if not have_results:
            return False

        eofs_path = self._get_eofs_source(base_dir)
        return eofs_path is not None

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        profiling = ctx.profile_memory
        base_dir = os.path.dirname(ctx.dineof_output_path)
        eofs_path = self._get_eofs_source(base_dir)
        target_path = os.path.join(base_dir, self._get_output_filename())

        print(f"[{self.name}] source_mode={self.source_mode}, eofs_path={eofs_path}")

        if eofs_path is None:
            print(f"[{self.name}] No EOFs file found for source_mode={self.source_mode}; skipping.")
            return ds if ds is not None else xr.Dataset()

        mean_offset = self._compute_mean_offset(ctx, base_dir)
        if mean_offset is not None:
            print(f"[{self.name}] Mean offset to add back: {mean_offset:.6f}")

        try:
            eofs = xr.open_dataset(eofs_path)
        except Exception as e:
            print(f"[{self.name}] Failed to open EOFs: {eofs_path}: {e}")
            return ds if ds is not None else xr.Dataset()

        temporal_vars = [v for v in eofs.data_vars
                         if v.startswith("temporal_eof") and ("t" in eofs[v].dims)]
        spatial_vars  = [v for v in eofs.data_vars
                         if v.startswith("spatial_eof") and (set(("y","x")) <= set(eofs[v].dims))]

        if not temporal_vars or not spatial_vars or "eigenvalues" not in eofs:
            print(f"[{self.name}] Missing EOF components; skipping.")
            eofs.close()
            return ds if ds is not None else xr.Dataset()

        def _mode_id(name: str) -> int:
            return int(name.split("eof")[-1])

        temporal_vars.sort(key=_mode_id)
        spatial_vars.sort(key=_mode_id)

        modes = [_mode_id(v) for v in temporal_vars]
        spatial_vars = [f"spatial_eof{m}" for m in modes if f"spatial_eof{m}" in spatial_vars]
        temporal_vars = [f"temporal_eof{m}" for m in modes if f"temporal_eof{m}" in temporal_vars]
        modes = [m for m in modes if (f"temporal_eof{m}" in temporal_vars and f"spatial_eof{m}" in spatial_vars)]

        if not modes:
            print(f"[{self.name}] No matching temporal/spatial EOF mode pairs; skipping.")
            eofs.close()
            return ds if ds is not None else xr.Dataset()

        if "time" in eofs.coords and ("t" in eofs["time"].dims or eofs["time"].sizes.get("time") == eofs.sizes.get("t")):
            time_coord = eofs["time"].values
        else:
            time_coord = np.arange(eofs.dims["t"], dtype=np.int64)

        T_mat = np.stack([eofs[v].values for v in temporal_vars], axis=1)  # (t, K)
        S = np.stack([eofs[v].values for v in spatial_vars], axis=0)       # (K, y, x)
        eig = eofs["eigenvalues"].values
        K = T_mat.shape[1]
        sigma = eig[:K]
        T_scaled = T_mat * sigma[np.newaxis, :]

        y_name, x_name = self._infer_yx_names(eofs)
        y_vals = eofs[y_name].values if y_name in eofs.coords else np.arange(S.shape[1])
        x_vals = eofs[x_name].values if x_name in eofs.coords else np.arange(S.shape[2])
        eofs.close()
        del T_mat

        if profiling:
            print(f"[{self.name}][MEM] after EOF load: RSS={get_current_rss_mb():.0f} MB")

        # --- reconstruct via tensordot: (t,K) . (K,y,x) -> (t,y,x)
        recon = np.tensordot(T_scaled, S, axes=([1], [0]))  # (t, y, x)
        del T_scaled, S

        if mean_offset is not None:
            recon += mean_offset

        recon = recon.astype("float32")

        if profiling:
            print(f"[{self.name}][MEM] after reconstruct: RSS={get_current_rss_mb():.0f} MB")

        # --- Get global/var attrs from original dineof_results.nc ---
        global_attrs = {}
        var_attrs = {}
        try:
            with xr.open_dataset(ctx.dineof_output_path) as orig:
                global_attrs = dict(orig.attrs)
                if "lake_surface_water_temperature_reconstructed" in orig:
                    var_attrs = dict(orig["lake_surface_water_temperature_reconstructed"].attrs)
        except Exception:
            pass

        if mean_offset is not None:
            global_attrs["mean_offset_applied"] = float(mean_offset)

        # --- Write using chunked netCDF4 to avoid xarray memory spike ---
        n_time = recon.shape[0]
        n_y = recon.shape[1]
        n_x = recon.shape[2]
        chunk_size = 500

        try:
            nc = netCDF4.Dataset(target_path, "w", format="NETCDF4")
            nc.createDimension("time", None)  # unlimited
            nc.createDimension(y_name, n_y)
            nc.createDimension(x_name, n_x)

            # Global attrs
            for k, v in global_attrs.items():
                try:
                    nc.setncattr(k, v)
                except TypeError:
                    nc.setncattr(k, str(v))

            # Time coordinate
            nc_time = nc.createVariable("time", time_coord.dtype, ("time",), fill_value=False)
            nc_time[:] = time_coord

            # Spatial coordinates
            nc_y = nc.createVariable(y_name, y_vals.dtype, (y_name,), fill_value=False)
            nc_y[:] = y_vals
            nc_x = nc.createVariable(x_name, x_vals.dtype, (x_name,), fill_value=False)
            nc_x[:] = x_vals

            # Main variable — chunked write
            nc_var = nc.createVariable(
                "lake_surface_water_temperature_reconstructed",
                "f4", ("time", y_name, x_name),
                zlib=True, complevel=4)
            for k, v in var_attrs.items():
                if k != "_FillValue":
                    nc_var.setncattr(k, v)

            for t0 in range(0, n_time, chunk_size):
                t1 = min(t0 + chunk_size, n_time)
                nc_var[t0:t1] = recon[t0:t1]

            nc.close()

            if profiling:
                print(f"[{self.name}][MEM] after chunked write: RSS={get_current_rss_mb():.0f} MB")

            print(f"[{self.name}] Wrote {target_path} (var='lake_surface_water_temperature_reconstructed')")
        except Exception as e:
            print(f"[{self.name}] Failed to write result: {e}")

        del recon
        return ds if ds is not None else xr.Dataset()

    # ---------- helpers ----------

    def _compute_mean_offset(self, ctx: PostContext, base_dir: str) -> Optional[float]:
        """
        Get the mean offset that DINEOF subtracted before SVD.
        
        DINEOF stores this in meandata.val in Fortran scientific notation (e.g., 0.3894E-01).
        
        Returns:
            The scalar mean offset to add back, or None if file not found.
        """
        meandata_path = os.path.join(base_dir, "meandata.val")
        
        if not os.path.isfile(meandata_path):
            print(f"[{self.name}] meandata.val not found, cannot add mean offset")
            return None
        
        try:
            with open(meandata_path, 'r') as f:
                content = f.read().strip()
            
            # Parse Fortran scientific notation (e.g., "0.3894E-01")
            mean_offset = float(content)
            return mean_offset
            
        except Exception as e:
            print(f"[{self.name}] Failed to read meandata.val: {e}")
            return None

    def _get_eofs_source(self, base_dir: str) -> Optional[str]:
        """Explicit source selection based on source_mode and eofs_prefix."""
        if self.source_mode == "filtered":
            p = os.path.join(base_dir, f"{self.eofs_prefix}_filtered.nc")
            return p if os.path.isfile(p) else None
        elif self.source_mode == "interp":
            p = os.path.join(base_dir, f"{self.eofs_prefix}_interpolated.nc")
            return p if os.path.isfile(p) else None
        elif self.source_mode == "filtered_interp":
            p = os.path.join(base_dir, f"{self.eofs_prefix}_filtered_interpolated.nc")
            return p if os.path.isfile(p) else None
        return None

    def _get_output_filename(self) -> str:
        """Output filename based on source_mode and output_cv_suffix."""
        s = self.output_cv_suffix  # e.g. "" or "_for_cv"
        if self.source_mode == "filtered":
            return f"dineof_results_eof_filtered{s}.nc"
        elif self.source_mode == "interp":
            return f"dineof_results_eof_interp_full{s}.nc"
        elif self.source_mode == "filtered_interp":
            return f"dineof_results_eof_filtered_interp_full{s}.nc"
        return f"dineof_results_from_eofs{s}.nc"

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
        dims_target = t_da.dims
        dims_src = da.dims
        
        # Check if template has duplicate dimension names (e.g., dim002, dim001, dim001)
        # This happens when DINEOF Fortran writes a square grid where lat==lon size
        if len(dims_target) != len(set(dims_target)):
            print(f"[{self.name}] Template has duplicate dim names {dims_target}, correcting...")
            # Create corrected target dims: (dim002, dim001, dim001) -> (dim003, dim002, dim001)
            # We know the structure should be (time, lat, lon) so we assign unique generic names
            dims_target = ('dim003', 'dim002', 'dim001')
            print(f"[{self.name}] Corrected target dims to {dims_target}")
        
        # rename dims if necessary
        rename_map = {}
        # assume structure ('time', y, x) both sides; just map second/third dims by name similarity
        for src, tgt in zip(dims_src, dims_target):
            if src != tgt:
                # Check if target name already exists as a coord to avoid conflict
                if tgt in da.coords and tgt not in da.dims:
                    # Drop the conflicting coord first
                    da = da.drop_vars(tgt)
                rename_map[src] = tgt
        if rename_map:
            da = da.rename(rename_map)

        # reindex to template coords where available - skip if template had duplicate dims
        # since the corrected dims won't have matching coords in template
        for coord in t_da.dims:
            if coord in t_da.coords and coord in da.coords:
                da = da.reindex({coord: t_da.coords[coord]}, method=None)

        return da