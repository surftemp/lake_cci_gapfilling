# post_steps/lswt_production_plots.py
"""
RAM-efficient paper-quality LSWT timeseries plots for production pipeline.
Reads FILLED.nc via netCDF4 directly. Handles masked arrays properly.
Uses LineCollection for per-pixel plots (fast rendering).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import netCDF4 as nc4

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "figure.figsize": (8, 4),
    "figure.dpi": 400,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "axes.linewidth": 0.3,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "ytick.major.width": 0.5,
    "ytick.major.size": 2,
    "xtick.major.width": 0.5,
    "xtick.major.size": 2,
    "lines.linewidth": 1,
    "lines.markersize": 3,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

from .base import PostProcessingStep, PostContext, get_current_rss_mb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_var(nc_var, slices):
    """Read from netCDF4 variable, convert masked array to NaN-filled numpy."""
    raw = nc_var[slices]
    if isinstance(raw, np.ma.MaskedArray):
        return raw.filled(np.nan).astype(np.float32)
    return np.asarray(raw, dtype=np.float32)


def _nc_times(ds):
    """Convert netCDF time to python datetime list and pandas DatetimeIndex."""
    tv = ds.variables["time"]
    cfdates = nc4.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
    py = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in cfdates]
    return py, pd.DatetimeIndex(py)


def _to_celsius(arr):
    """Convert to Celsius if Kelvin (mean > 100)."""
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and np.nanmean(finite) > 100:
        return arr - 273.15
    return arr


def _find_centre(ds, lake_iy, lake_ix, lat, lon, lake_id, csv_path):
    """Find centre pixel. CSV lookup with lakeid validation, fallback to centroid."""
    if csv_path and os.path.isfile(csv_path) and lake_id is not None:
        try:
            df = pd.read_csv(csv_path)
            row = df.loc[df["CCI ID"] == int(lake_id)]
            if len(row) > 0:
                clat = float(row.iloc[0]["LAT CENTRE"])
                clon = float(row.iloc[0]["LON CENTRE"])
                # Find nearest lake pixel to these coords
                d2 = (lat[lake_iy] - clat)**2 + (lon[lake_ix] - clon)**2
                nearest = np.argmin(d2)
                ci, cj = int(lake_iy[nearest]), int(lake_ix[nearest])
                print(f"[LSWTPlots] Centre from CSV (nearest lake pixel): ({ci},{cj}) "
                      f"lat={lat[ci]:.4f} lon={lon[cj]:.4f}")
                return ci, cj
        except Exception as e:
            print(f"[LSWTPlots] CSV lookup failed: {e}")

    # Fallback: centroid of lake pixels
    ci = int(lake_iy[len(lake_iy) // 2])
    cj = int(lake_ix[len(lake_ix) // 2])
    print(f"[LSWTPlots] Centre from centroid: ({ci},{cj})")
    return ci, cj


# ---------------------------------------------------------------------------
# LSWTProductionPlotsStep
# ---------------------------------------------------------------------------

class LSWTProductionPlotsStep(PostProcessingStep):
    """Generate paper-quality LSWT timeseries plots from FILLED.nc."""

    def __init__(self, lake_centre_csv: str = "", perpixel_plots: bool = False):
        super().__init__()
        self.lake_centre_csv = lake_centre_csv
        self.perpixel_plots = perpixel_plots

    def generate(self, ctx: PostContext, filled_nc_path: str, output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        lake_id = ctx.lake_id or 0
        saved = []

        ds = nc4.Dataset(filled_nc_path, "r")
        try:
            py_times, time_idx = _nc_times(ds)

            # Find data variable
            var_name = None
            for candidate in ["lake_surface_water_temperature_reconstructed", "temp_filled"]:
                if candidate in ds.variables:
                    var_name = candidate
                    break
            if var_name is None:
                print("[LSWTPlots] No data variable found")
                return saved

            lat = np.asarray(ds.variables["lat"][:])
            lon = np.asarray(ds.variables["lon"][:])
            nt = len(time_idx)

            # Lake mask
            if "lakeid" in ds.variables:
                lid = np.asarray(ds.variables["lakeid"][:, :])
                lake_iy, lake_ix = np.where(lid == 1)
            else:
                mid_slab = _read_var(ds.variables[var_name], (nt // 2, slice(None), slice(None)))
                lake_iy, lake_ix = np.where(np.isfinite(mid_slab))
            n_lake = len(lake_iy)
            print(f"[LSWTPlots] Lake pixels: {n_lake}, timesteps: {nt}")

            if n_lake == 0:
                print("[LSWTPlots] No lake pixels found")
                return saved

            # Centre pixel
            ci, cj = _find_centre(ds, lake_iy, lake_ix, lat, lon, lake_id, self.lake_centre_csv)
            centre_ts = _to_celsius(_read_var(ds.variables[var_name], (slice(None), ci, cj)))
            print(f"[LSWTPlots] Centre pixel valid: {np.isfinite(centre_ts).sum()}/{nt}")

            # Min/max envelope — chunked
            ts_min = np.full(nt, np.nan, dtype=np.float32)
            ts_max = np.full(nt, np.nan, dtype=np.float32)
            for t0 in range(0, nt, 500):
                t1 = min(t0 + 500, nt)
                # Read only lake pixels for this time chunk
                for k in range(n_lake):
                    px = _read_var(ds.variables[var_name], (slice(t0, t1), lake_iy[k], lake_ix[k]))
                    px = _to_celsius(px)
                    if k == 0:
                        ts_min[t0:t1] = px
                        ts_max[t0:t1] = px
                    else:
                        ts_min[t0:t1] = np.fmin(ts_min[t0:t1], px)
                        ts_max[t0:t1] = np.fmax(ts_max[t0:t1], px)

            print(f"[LSWTPlots] Shade valid: {np.isfinite(ts_min).sum()}/{nt}, "
                  f"range: [{np.nanmin(ts_min):.1f}, {np.nanmax(ts_max):.1f}]")

            # ---- 1. Shade plot full ----
            p = _plot_shade(time_idx, ts_min, ts_max, centre_ts, lake_id,
                            f"Lake {lake_id} -- ST-DINEOF gap-filled LSWT",
                            os.path.join(output_dir, f"lake{lake_id}_shade_full.png"))
            saved.append(p)

            # ---- 2. Per-pixel plot full (LineCollection) — optional ----
            if self.perpixel_plots:
                p = _plot_perpixel(ds, var_name, time_idx, centre_ts, lake_iy, lake_ix,
                                   lake_id, f"Lake {lake_id} -- ST-DINEOF gap-filled LSWT",
                                   os.path.join(output_dir, f"lake{lake_id}_perpixel_full.png"))
                saved.append(p)

            # ---- 3 & 4. Per-year plots ----
            years = sorted(set(time_idx.year))
            for year in years:
                mask_yr = time_idx.year == year
                idx_yr = np.where(mask_yr)[0]
                if len(idx_yr) == 0:
                    continue
                time_yr = time_idx[mask_yr]
                min_yr = ts_min[mask_yr]
                max_yr = ts_max[mask_yr]
                centre_yr = centre_ts[mask_yr]

                p = _plot_shade_year(time_yr, min_yr, max_yr, centre_yr, lake_id, year,
                                     os.path.join(output_dir, f"lake{lake_id}_shade_{year}.png"))
                saved.append(p)

                if self.perpixel_plots:
                    p = _plot_perpixel_year(ds, var_name, time_yr, idx_yr, centre_yr,
                                            lake_iy, lake_ix, lake_id, year,
                                            os.path.join(output_dir, f"lake{lake_id}_perpixel_{year}.png"))
                    saved.append(p)

        finally:
            ds.close()

        return saved


# ---------------------------------------------------------------------------
# Plot functions (module-level, no class state needed)
# ---------------------------------------------------------------------------

def _plot_shade(time_idx, ts_min, ts_max, centre_ts, lake_id, title, fname):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    valid = np.isfinite(ts_min) & np.isfinite(ts_max)
    ax.fill_between(time_idx, ts_min, ts_max, where=valid,
                    color="grey", alpha=0.35, linewidth=0, label="pixel range")
    vc = np.isfinite(centre_ts)
    ax.plot(time_idx[vc], centre_ts[vc], color="red", lw=0.5, label="centre pixel")
    ax.set_xlim(pd.Timestamp(2000, 1, 1), pd.Timestamp(2023, 1, 1))
    _set_year_ticks(ax, time_idx)
    ax.set_ylabel("LSWT (\u00b0C)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(fname)
    plt.close(fig)
    print(f"[LSWTPlots] Saved {os.path.basename(fname)}")
    return fname


def _plot_perpixel(ds, var_name, time_idx, centre_ts, lake_iy, lake_ix, lake_id, title, fname):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    time_num = mdates.date2num(time_idx.to_pydatetime())

    segments = []
    for k in range(len(lake_iy)):
        ts = _to_celsius(_read_var(ds.variables[var_name], (slice(None), int(lake_iy[k]), int(lake_ix[k]))))
        valid = np.isfinite(ts)
        if valid.sum() < 2:
            continue
        x = time_num[valid]
        y = ts[valid]
        pts = np.column_stack([x, y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        segments.append(segs)

    if segments:
        all_segs = np.concatenate(segments, axis=0)
        lc = LineCollection(all_segs, colors="grey", alpha=0.05, linewidths=0.3, zorder=1)
        ax.add_collection(lc)

    vc = np.isfinite(centre_ts)
    ax.plot(time_idx[vc], centre_ts[vc], color="red", lw=0.5, label="centre pixel", zorder=2)
    ax.set_xlim(pd.Timestamp(2000, 1, 1), pd.Timestamp(2023, 1, 1))
    ax.autoscale_view(scaley=True)
    _set_year_ticks(ax, time_idx)
    ax.set_ylabel("LSWT (\u00b0C)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(fname)
    plt.close(fig)
    print(f"[LSWTPlots] Saved {os.path.basename(fname)}")
    return fname


def _plot_shade_year(time_yr, min_yr, max_yr, centre_yr, lake_id, year, fname):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    valid = np.isfinite(min_yr) & np.isfinite(max_yr)
    ax.fill_between(time_yr, min_yr, max_yr, where=valid,
                    color="grey", alpha=0.35, linewidth=0, label="pixel range")
    vc = np.isfinite(centre_yr)
    ax.plot(time_yr[vc], centre_yr[vc], color="red", lw=0.5, label="centre pixel")
    ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
    month_ticks = [pd.Timestamp(year, m, 1) for m in range(1, 13)]
    ax.set_xticks(month_ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("LSWT (\u00b0C)")
    ax.set_title(f"Lake {lake_id} -- {year}")
    ax.legend(loc="upper right")
    fig.savefig(fname)
    plt.close(fig)
    print(f"[LSWTPlots] Saved {os.path.basename(fname)}")
    return fname


def _plot_perpixel_year(ds, var_name, time_yr, idx_yr, centre_yr, lake_iy, lake_ix,
                        lake_id, year, fname):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    time_num = mdates.date2num(time_yr.to_pydatetime())
    t0 = int(idx_yr[0])
    t1 = int(idx_yr[-1]) + 1

    segments = []
    for k in range(len(lake_iy)):
        raw = _read_var(ds.variables[var_name], (slice(t0, t1), int(lake_iy[k]), int(lake_ix[k])))
        ts = _to_celsius(raw[idx_yr - t0])
        valid = np.isfinite(ts)
        if valid.sum() < 2:
            continue
        x = time_num[valid]
        y = ts[valid]
        pts = np.column_stack([x, y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        segments.append(segs)

    if segments:
        all_segs = np.concatenate(segments, axis=0)
        lc = LineCollection(all_segs, colors="grey", alpha=0.05, linewidths=0.3, zorder=1)
        ax.add_collection(lc)

    vc = np.isfinite(centre_yr)
    ax.plot(time_yr[vc], centre_yr[vc], color="red", lw=0.5, label="centre pixel", zorder=2)
    ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
    ax.autoscale_view(scaley=True)
    month_ticks = [pd.Timestamp(year, m, 1) for m in range(1, 13)]
    ax.set_xticks(month_ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("LSWT (\u00b0C)")
    ax.set_title(f"Lake {lake_id} -- {year}")
    ax.legend(loc="upper right")
    fig.savefig(fname)
    plt.close(fig)
    print(f"[LSWTPlots] Saved {os.path.basename(fname)}")
    return fname


def _set_year_ticks(ax, time_idx):
    years = sorted(set(time_idx.year))
    ticks = [pd.Timestamp(y, 1, 1) for y in years]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
