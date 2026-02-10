# post_steps/filter_eofs.py
from __future__ import annotations

import os
import glob
import numpy as np
import xarray as xr
from typing import Optional, Tuple, List, Dict, Any
from .base import PostProcessingStep, PostContext


class FilterTemporalEOFsStep(PostProcessingStep):
    """
    Filter outlier timesteps in eofs.nc based on temporal EOF outlier rules,
    replacing their values with linear interpolation on physical time (days since epoch).
    
    - If ANY selected temporal EOF flags a time step as outlier, ALL temporal EOF values
      at that timestep are replaced with linearly interpolated values from neighboring
      good timesteps (using physical time, not array index, to respect irregular gaps).
    - The output eofs_filtered.nc has the SAME number of timesteps as eofs.nc.
    - At edge timesteps (first/last), np.interp clamps to the nearest good value.
    - Supports three EOF selection methods for outlier detection:
      1. "all": Apply filtering to all temporal EOFs
      2. "variance_threshold": Only filter EOFs that cumulatively explain Y% of variance
      3. "top_n": Only filter the first N EOFs
    - Writes a new filtered file by default (suffix), or overwrites if configured.
    - Does not modify the main LSWT dataset; returns ds unchanged.
    - Writes filtering statistics to CSV file.
    """

    name = "FilterTemporalEOFs"

    def __init__(
        self,
        *,
        method: str = "robust_sd",                 # "robust_sd" or "quantile"
        k: float = 4.0,                            # for robust_sd: k * RSD (RSD = 1.4826*MAD)
        quantiles: Tuple[float, float] = (0.005, 0.995),
        temporal_var_prefix: str = "temporal_eof",
        output_suffix: str = "_filtered",
        overwrite: bool = False,
        # NEW: EOF selection parameters
        eof_selection: str = "all",                # "all", "variance_threshold", or "top_n"
        variance_threshold: float = 0.95,          # for variance_threshold: cumulative variance explained
        top_n_eofs: int = 3,                      # for top_n: number of EOFs to consider
    ):
        self.method = method
        self.k = float(k)
        self.q_lo, self.q_hi = quantiles
        self.temporal_var_prefix = temporal_var_prefix
        self.output_suffix = output_suffix
        self.overwrite = overwrite
        
        # EOF selection parameters
        self.eof_selection = eof_selection
        self.variance_threshold = variance_threshold
        self.top_n_eofs = top_n_eofs

        # populated after apply()
        self.info: Dict[str, Any] = {}
        self.per_eof_stats: List[Dict[str, Any]] = []
        self.variance_explained: Dict[str, float] = {}  # NEW: track variance explained per EOF

    # ---------- plumbing ----------

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        # We filter EOFs file independently of ds; run once per pipeline if eofs exists.
        eofs_path = self._find_eofs_path(ctx)
        return eofs_path is not None and os.path.isfile(eofs_path)

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        eofs_path = self._find_eofs_path(ctx)
        assert eofs_path is not None

        try:
            eofs = xr.open_dataset(eofs_path)
        except Exception as e:
            print(f"[{self.name}] Failed to open {eofs_path}: {e}; skipping.")
            return ds if ds is not None else xr.Dataset()

        temporal_vars = self._get_temporal_vars(eofs)
        if not temporal_vars:
            print(f"[{self.name}] No temporal EOF series matching '{self.temporal_var_prefix}*'; skipping.")
            eofs.close()
            return ds if ds is not None else xr.Dataset()

        # Select which EOFs to use for filtering based on variance explained
        selected_vars = self._select_eofs_by_variance(eofs, temporal_vars)
        
        print(f"[{self.name}] EOF selection method: {self.eof_selection}")
        print(f"[{self.name}] Selected {len(selected_vars)}/{len(temporal_vars)} EOFs for filtering")
        if self.variance_explained:
            total_var_selected = sum(self.variance_explained[v] for v in selected_vars if v in self.variance_explained)
            print(f"[{self.name}] Selected EOFs explain {total_var_selected:.1%} of total variance")

        # Build flag mask across SELECTED temporal EOFs
        flagged_any, per_eof_flagged = self._build_flag_mask_with_stats(eofs, temporal_vars, selected_vars)
        replaced_idx = np.flatnonzero(flagged_any)

        # Calculate statistics
        total_timesteps = len(flagged_any)
        total_replaced = int(flagged_any.sum())
        fraction_replaced = total_replaced / total_timesteps if total_timesteps > 0 else 0.0

        # Store per-EOF statistics (now includes whether EOF was used for filtering)
        self.per_eof_stats = []
        for i, var_name in enumerate(temporal_vars):
            eof_flagged_count = int(per_eof_flagged[i].sum())
            eof_fraction = eof_flagged_count / total_timesteps if total_timesteps > 0 else 0.0
            was_selected = var_name in selected_vars
            self.per_eof_stats.append({
                'eof_name': var_name,
                'timesteps_flagged': eof_flagged_count,
                'fraction_flagged': eof_fraction,
                'used_for_filtering': was_selected,
                'variance_explained': self.variance_explained.get(var_name, np.nan)
            })

        # Replace outlier temporal EOF values with linear interpolation on physical time.
        # This preserves the same number of timesteps (unlike the old approach which dropped them).
        eofs_filt = eofs.copy(deep=True)
        if flagged_any.any():
            phys_days = self._get_physical_days(ctx, eofs)
            good_mask = ~flagged_any
            n_good = int(good_mask.sum())
            if n_good >= 2:
                for v in temporal_vars:
                    vals = eofs_filt[v].values.copy()
                    original_outlier_vals = vals[flagged_any].copy()
                    # np.interp: at edges, clamps to nearest good value (flat extrapolation)
                    vals[flagged_any] = np.interp(
                        phys_days[flagged_any],
                        phys_days[good_mask],
                        vals[good_mask]
                    )
                    eofs_filt[v] = (eofs_filt[v].dims, vals)
                    # Log per-EOF replacement summary
                    max_change = np.max(np.abs(vals[flagged_any] - original_outlier_vals)) if flagged_any.any() else 0.0
                    print(f"[{self.name}] {v}: replaced {total_replaced} outlier values "
                          f"(max abs change: {max_change:.6f})")
            else:
                print(f"[{self.name}] WARNING: only {n_good} good timesteps, cannot interpolate. "
                      f"Outlier values left unchanged.")

        # Decide where to write
        if self.overwrite:
            out_path = eofs_path
        else:
            root, ext = os.path.splitext(eofs_path)
            out_path = f"{root}{self.output_suffix}{ext}"

        # Write — same timestep count as original eofs.nc, with outliers replaced
        enc = {v: {"zlib": True, "complevel": 4} for v in eofs_filt.data_vars}
        try:
            eofs_filt.to_netcdf(out_path, encoding=enc)
            print(f"[{self.name}] Wrote filtered EOFs -> {out_path} "
                  f"({total_replaced}/{total_timesteps} outlier timesteps replaced, "
                  f"{total_timesteps} total timesteps preserved)")
        except Exception as e:
            print(f"[{self.name}] Failed to write {out_path}: {e}")

        # Write statistics CSV
        output_dir = os.path.dirname(ctx.output_path)
        stats_file = os.path.join(output_dir, "eof_filtering_stats.csv")
        self._write_filtering_stats(stats_file, total_timesteps, total_replaced, fraction_replaced)

        # Summarize for later steps/QA
        self.info = {
            "eofs_input": eofs_path,
            "eofs_output": out_path,
            "method": self.method,
            "k": self.k,
            "quantiles": (self.q_lo, self.q_hi),
            "temporal_prefix": self.temporal_var_prefix,
            "n_time": int(total_timesteps),
            "n_replaced": int(total_replaced),
            "replaced_indices": replaced_idx.tolist(),
            "fraction_replaced": fraction_replaced,
            "eof_selection": self.eof_selection,
            "n_eofs_selected": len(selected_vars),
            "n_eofs_total": len(temporal_vars),
        }
        
        if self.eof_selection == "variance_threshold":
            self.info["variance_threshold"] = self.variance_threshold
        elif self.eof_selection == "top_n":
            self.info["top_n_eofs"] = self.top_n_eofs
            
        # Attach to context for downstream use
        if not hasattr(ctx, "eof_filter_info"):
            setattr(ctx, "eof_filter_info", self.info)
        else:
            ctx.eof_filter_info = self.info

        eofs.close()
        eofs_filt.close()

        return ds if ds is not None else xr.Dataset()

    # ---------- helpers ----------

    def _find_eofs_path(self, ctx: PostContext) -> Optional[str]:
        """
        Heuristics: look in the dineof_output_path directory for a file named *eofs*.nc
        Prefer exact 'eofs.nc' if present.
        """
        base_dir = os.path.dirname(ctx.dineof_output_path)
        candidates = [
            os.path.join(base_dir, "eofs.nc"),
            os.path.join(base_dir, "EOFs.nc"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

        globs = glob.glob(os.path.join(base_dir, "*eofs*.nc"))
        if globs:
            # deterministic choice: shortest name, then lexicographic
            globs.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
            return globs[0]
        return None

    def _get_physical_days(self, ctx: PostContext, eofs: xr.Dataset) -> np.ndarray:
        """
        Get physical time axis (days since 1981-01-01) for the EOF timesteps.
        
        Needed for proper linear interpolation that respects actual time gaps
        between observations (not array index which assumes equal spacing).
        
        Tries in order:
          1. ctx.prepared_time_days (set by MergeOutputsStep, most reliable)
          2. 'time' coordinate in eofs.nc itself
          3. Read from prepared.nc directly
          4. Raise error (cannot proceed without physical time)
        """
        n_t = eofs.dims["t"]
        
        # 1. Try ctx.prepared_time_days (set by MergeOutputsStep which runs before us)
        if hasattr(ctx, "prepared_time_days") and ctx.prepared_time_days is not None:
            days = ctx.prepared_time_days
            if len(days) == n_t:
                print(f"[{self.name}] Using physical time from ctx.prepared_time_days ({n_t} timesteps)")
                return days.astype("float64")
            else:
                print(f"[{self.name}] WARNING: ctx.prepared_time_days length {len(days)} != eofs t dim {n_t}")
        
        # 2. Try 'time' coord in eofs.nc
        if "time" in eofs.coords:
            vals = eofs["time"].values
            if np.issubdtype(vals.dtype, np.datetime64):
                base = np.datetime64("1981-01-01T12:00:00", "ns")
                days = ((vals.astype("datetime64[ns]") - base) / np.timedelta64(1, "D")).astype("float64")
                if len(days) == n_t:
                    print(f"[{self.name}] Using physical time from eofs.nc 'time' coord ({n_t} timesteps)")
                    return days
            elif np.issubdtype(vals.dtype, np.integer) or np.issubdtype(vals.dtype, np.floating):
                days = vals.astype("float64")
                if len(days) == n_t:
                    print(f"[{self.name}] Using physical time from eofs.nc numeric 'time' coord ({n_t} timesteps)")
                    return days
        
        # 3. Read from prepared.nc directly
        try:
            with xr.open_dataset(ctx.dineof_input_path) as ds_in:
                time_vals = ds_in[ctx.time_name].values.astype("datetime64[ns]")
                base = np.datetime64("1981-01-01T12:00:00", "ns")
                days = ((time_vals - base) / np.timedelta64(1, "D")).astype("float64")
                if len(days) == n_t:
                    print(f"[{self.name}] Using physical time from prepared.nc ({n_t} timesteps)")
                    return days
                else:
                    print(f"[{self.name}] WARNING: prepared.nc time length {len(days)} != eofs t dim {n_t}")
        except Exception as e:
            print(f"[{self.name}] WARNING: Failed to read prepared.nc for time: {e}")
        
        # 4. Cannot proceed without physical time — do NOT fall back to array index
        raise ValueError(
            f"[{self.name}] Cannot determine physical time axis for EOF interpolation. "
            f"eofs t dim = {n_t}, none of the time sources matched."
        )

    def _get_temporal_vars(self, ds: xr.Dataset) -> List[str]:
        """Get all temporal EOF variables sorted by mode number."""
        vars = [
            name for name, da in ds.data_vars.items()
            if name.startswith(self.temporal_var_prefix) and ("t" in da.dims)
        ]
        # Sort by EOF index (0, 1, 2, ...)
        vars.sort(key=lambda x: int(x.replace(self.temporal_var_prefix, '')))
        return vars

    def _select_eofs_by_variance(self, ds: xr.Dataset, temporal_vars: List[str]) -> List[str]:
        """
        Select which EOFs to use for filtering based on variance explained.
        
        Returns:
            List of temporal EOF variable names to use for filtering
        """
        if self.eof_selection == "all":
            # Original behavior: use all EOFs
            return temporal_vars
        
        # Need eigenvalues for variance-based selection
        if "eigenvalues" not in ds:
            print(f"[{self.name}] WARNING: No eigenvalues found, using all EOFs")
            return temporal_vars
        
        eigenvalues = ds["eigenvalues"].values
        total_variance = np.sum(eigenvalues)
        
        # Calculate variance explained by each EOF
        self.variance_explained = {}
        for i, var_name in enumerate(temporal_vars):
            mode_idx = int(var_name.replace(self.temporal_var_prefix, ''))
            if mode_idx < len(eigenvalues):
                self.variance_explained[var_name] = eigenvalues[mode_idx] / total_variance
            else:
                self.variance_explained[var_name] = 0.0
        
        if self.eof_selection == "variance_threshold":
            # Select EOFs until cumulative variance threshold is reached
            cumulative_variance = 0.0
            selected = []
            for var_name in temporal_vars:
                if cumulative_variance < self.variance_threshold:
                    selected.append(var_name)
                    cumulative_variance += self.variance_explained.get(var_name, 0)
                else:
                    break
            return selected
            
        elif self.eof_selection == "top_n":
            # Select the first N EOFs
            return temporal_vars[:min(self.top_n_eofs, len(temporal_vars))]
        
        else:
            print(f"[{self.name}] Unknown EOF selection method: {self.eof_selection}, using all")
            return temporal_vars

    def _build_flag_mask_with_stats(self, ds: xr.Dataset, all_vars: List[str], selected_vars: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Build flag mask and return both combined mask and per-EOF masks.
        Only selected_vars contribute to the combined mask.
        """
        if self.method == "robust_sd":
            return self._rule_robust_sd_with_stats(ds, all_vars, selected_vars)
        elif self.method == "quantile":
            return self._rule_quantile_with_stats(ds, all_vars, selected_vars)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _rule_robust_sd_with_stats(self, ds: xr.Dataset, all_vars: List[str], selected_vars: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        t_len = ds.dims["t"]
        flagged_any = np.zeros(t_len, dtype=bool)
        per_eof_flagged = []

        for v in all_vars:
            x = ds[v].values  # (t,)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            rsd = 1.4826 * mad
            if not np.isfinite(rsd) or rsd == 0:
                flagged = np.zeros(t_len, dtype=bool)
            else:
                flagged = np.abs(x - med) > (self.k * rsd)
            
            per_eof_flagged.append(flagged)
            
            # Only contribute to combined mask if this EOF was selected
            if v in selected_vars:
                flagged_any |= flagged

        return flagged_any, per_eof_flagged

    def _rule_quantile_with_stats(self, ds: xr.Dataset, all_vars: List[str], selected_vars: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        t_len = ds.dims["t"]
        flagged_any = np.zeros(t_len, dtype=bool)
        per_eof_flagged = []

        for v in all_vars:
            x = ds[v].values
            lo = np.nanquantile(x, self.q_lo)
            hi = np.nanquantile(x, self.q_hi)
            flagged = (x < lo) | (x > hi)
            
            per_eof_flagged.append(flagged)
            
            # Only contribute to combined mask if this EOF was selected
            if v in selected_vars:
                flagged_any |= flagged

        return flagged_any, per_eof_flagged

    def _write_filtering_stats(self, stats_file: str, total_timesteps: int, total_replaced: int, fraction_replaced: float):
        """Write filtering statistics to CSV file."""
        try:
            import csv
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            
            with open(stats_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['eof_name', 'variance_explained', 'used_for_filtering', 'timesteps_flagged', 'fraction_flagged'])
                
                # Per-EOF statistics
                for stat in self.per_eof_stats:
                    writer.writerow([
                        stat['eof_name'],
                        f"{stat['variance_explained']:.6f}" if not np.isnan(stat['variance_explained']) else 'N/A',
                        'Yes' if stat['used_for_filtering'] else 'No',
                        stat['timesteps_flagged'], 
                        f"{stat['fraction_flagged']:.6f}"
                    ])
                
                # Summary row
                writer.writerow([
                    'TOTAL_COMBINED',
                    '',
                    '',
                    total_replaced,
                    f"{fraction_replaced:.6f}"
                ])
                
                # Metadata rows
                writer.writerow([])  # Empty row separator
                writer.writerow(['# Filtering Parameters'])
                writer.writerow(['method', self.method])
                writer.writerow(['eof_selection', self.eof_selection])
                writer.writerow(['replacement_method', 'linear_interpolation_on_physical_time'])
                
                if self.eof_selection == "variance_threshold":
                    writer.writerow(['variance_threshold', f"{self.variance_threshold:.2%}"])
                elif self.eof_selection == "top_n":
                    writer.writerow(['top_n_eofs', self.top_n_eofs])
                    
                if self.method == "robust_sd":
                    writer.writerow(['k_threshold', f"{self.k:.1f}"])
                elif self.method == "quantile":
                    writer.writerow(['quantile_low', f"{self.q_lo:.3f}"])
                    writer.writerow(['quantile_high', f"{self.q_hi:.3f}"])
                    
                writer.writerow(['total_timesteps', total_timesteps])
                writer.writerow(['total_replaced', total_replaced])
            
            print(f"[{self.name}] Wrote filtering stats: {stats_file}")
            
        except Exception as e:
            print(f"[{self.name}] Failed to write stats file {stats_file}: {e}")