from __future__ import annotations
import os, glob
import numpy as np
import xarray as xr
from typing import Optional, Tuple, List, Dict
from .base import PostProcessingStep, PostContext

class InterpolateTemporalEOFsStep(PostProcessingStep):
    """
    Build 'eofs_interpolated.nc' with temporal_eofK dense on a chosen integer-day axis:
      - target = 'prepared'  -> prepared.nc timeline (trimmed)
      - target = 'full'      -> daily from ctx.time_start_days..ctx.time_end_days
    Edge policy: leave_nan | nearest
    """

    name = "InterpolateTemporalEOFs"

    def __init__(self, *, target: str = "full", edge_policy: str = "leave_nan"):
        assert target in ("prepared", "full")
        assert edge_policy in ("leave_nan", "nearest")
        self.target = target
        self.edge_policy = edge_policy

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        base_dir = os.path.dirname(ctx.dineof_output_path)
        return os.path.isfile(self._pick_eofs_src(base_dir)) and ctx.full_days is not None

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        base_dir = os.path.dirname(ctx.dineof_output_path)
        src_path = self._pick_eofs_src(base_dir)
        if not src_path:
            print(f"[{self.name}] No eofs source found; skipping.")
            return ds if ds is not None else xr.Dataset()

        # target axis in integer days
        prepared_days = self._read_prepared_days(ctx)
        target_days = prepared_days if (self.target == "prepared") else ctx.full_days

        E = xr.open_dataset(src_path)
        try:
            avail_days = self._infer_eofs_days(E, ctx, prepared_days)
            pos_in_target = {int(d): i for i, d in enumerate(target_days)}
            modes = sorted([int(v.split("temporal_eof")[-1]) for v in E.data_vars if v.startswith("temporal_eof")])

            # allocate dataset
            coords = {"t": target_days}
            out_vars = {}
            for k in modes:
                vname = f"temporal_eof{k}"
                src = E[vname].values  # (t_src,)

                # build out vector
                out = np.full((target_days.size,), np.nan, dtype=src.dtype)

                # map available days into target positions
                for i_src, d in enumerate(avail_days):
                    j = pos_in_target.get(int(d), -1)
                    if j >= 0:
                        out[j] = src[i_src]

                # interpolate internal gaps on integer x
                x = target_days.astype("float64")
                y = out.astype("float64")
                m = np.isfinite(y)

                if m.sum() >= 2:
                    i0 = np.argmax(m)
                    i1 = len(m) - 1 - np.argmax(m[::-1])
                    y[i0:i1+1] = np.interp(x[i0:i1+1], x[m], y[m])
                    if self.edge_policy == "nearest":
                        if i0 > 0:
                            y[:i0] = y[i0]
                        if i1 < len(y) - 1:
                            y[i1+1:] = y[i1]
                # Guard: with <2 anchors, leave as-is (NaNs) rather than polluting
                out_vars[vname] = (("t",), y.astype("float32"))

            # carry spatial EOFs & eigenvalues verbatim
            for name, da in E.data_vars.items():
                if name.startswith("spatial_eof") or name == "eigenvalues":
                    out_vars[name] = da

            out = xr.Dataset(out_vars, coords=coords)
            # add attrs
            out.attrs.update(E.attrs)
            out.attrs["eofs_interpolated"] = 1
            out.attrs["eof_interp_method"] = "linear"
            out.attrs["eof_interp_edge"] = self.edge_policy
            out.attrs["target_time_from"] = "prepared.nc" if (self.target == "prepared") else "preprocessor attrs (full daily)"
            out = out.assign_coords(t=target_days)

            # write
            target_path = os.path.join(base_dir, "eofs_interpolated.nc")
            comp = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
            out.to_netcdf(target_path, encoding=comp)
            print(f"[{self.name}] Wrote {target_path}")

            # stash path on context for downstream
            ctx.eofs_interpolated_path = target_path
        finally:
            E.close()

        return ds if ds is not None else xr.Dataset()

    # ---- helpers ----
    def _pick_eofs_src(self, base_dir: str) -> Optional[str]:
        for name in ("eofs_filtered.nc", "eofs.nc", "EOFs.nc"):
            p = os.path.join(base_dir, name)
            if os.path.isfile(p):
                return p
        # any *eofs*.nc
        cand = sorted(glob.glob(os.path.join(base_dir, "*eofs*.nc")))
        return cand[0] if cand else None

    def _read_prepared_days(self, ctx: PostContext) -> np.ndarray:
        with xr.open_dataset(ctx.dineof_input_path) as ds_in:
            # prepared is decoded datetime64; convert to int days using same basis
            vals = ds_in[ctx.time_name].values.astype("datetime64[ns]")
        base = np.datetime64("1981-01-01T12:00:00").astype("datetime64[ns]")
        return ((vals - base) / np.timedelta64(1, "D")).astype("int64")

    def _infer_eofs_days(self, E: xr.Dataset, ctx: PostContext, prepared_days: np.ndarray) -> np.ndarray:
        # If a 'time' coord exists, interpret it carefully.
        if "time" in E.coords:
            vals = E["time"].values
            # True datetime axis
            if np.issubdtype(vals.dtype, np.datetime64):
                base = np.datetime64("1981-01-01T12:00:00").astype("datetime64[ns]")
                vals_ns = vals.astype("datetime64[ns]")
                return ((vals_ns - base) / np.timedelta64(1, "D")).astype("int64")
            # Numeric → assume already integer days since epoch
            if np.issubdtype(vals.dtype, np.integer) or np.issubdtype(vals.dtype, np.floating):
                return vals.astype("int64")
            # Fallback: align by length to prepared
            return prepared_days[: E.sizes.get("t", vals.size)]

        # Otherwise, infer via companion results or fall back to prepared
        if "t" in E.dims:
            base_dir = os.path.dirname(ctx.dineof_output_path)
            for fname in ("dineof_results_eof_filtered.nc", "dineof_results.nc"):
                p = os.path.join(base_dir, fname)
                if os.path.isfile(p):
                    with xr.open_dataset(p) as R:
                        if ctx.time_name in R.coords and np.issubdtype(R[ctx.time_name].dtype, np.datetime64):
                            vals = R[ctx.time_name].values.astype("datetime64[ns]")
                            base = np.datetime64("1981-01-01T12:00:00").astype("datetime64[ns]")
                            days = ((vals - base) / np.timedelta64(1, "D")).astype("int64")
                            return days[: E.dims["t"]]
            # fallback: assume 1–1 with prepared
            return prepared_days[: E.dims["t"]]

        raise ValueError("EOFs file carries neither 'time' coord nor 't' dim")
