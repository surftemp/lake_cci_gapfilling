"""
LSWT Plotting Step - Generates time series PNG plots at the end of postprocessing.
"""

from __future__ import annotations
import os
import numpy as np
import xarray as xr
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Handle imports for both pipeline and standalone use
try:
    from .base import PostProcessingStep, PostContext
except ImportError:
    # Standalone mode - define minimal classes
    @dataclass
    class PostContext:
        """Minimal PostContext for standalone use."""
        lake_id: Optional[int] = None
        output_path: Optional[str] = None
        experiment_config_path: Optional[str] = None
        lake_path: Optional[str] = None
        dineof_input_path: Optional[str] = None
        dineof_output_path: Optional[str] = None
        output_html_folder: Optional[str] = None
        climatology_path: Optional[str] = None
    
    class PostProcessingStep:
        """Minimal base class for standalone use."""
        def __init__(self):
            pass
        
        @property
        def name(self) -> str:
            return self.__class__.__name__
        
        def should_apply(self, ctx, ds) -> bool:
            return True
        
        def apply(self, ctx, ds):
            raise NotImplementedError


# ==============================================================================
# Data Containers
# ==============================================================================

@dataclass
class LakeSeries:
    """Container for lake time series data"""
    time: pd.DatetimeIndex
    all_pixels: np.ndarray  # (time, n_pixels)
    center_pixel: np.ndarray  # (time,)


@dataclass 
class ClimatologySeries:
    """Container for climatology data (365 or 366 days)"""
    day_of_year: np.ndarray
    all_pixels: np.ndarray
    center_pixel: np.ndarray


# ==============================================================================
# Helper Functions
# ==============================================================================

def to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert DataArray to Celsius."""
    units = (da.attrs.get("units", "") or "").lower()
    
    # Explicit Celsius - never convert
    if "celsius" in units or units == "c" or "degree_c" in units:
        return da
    
    # Explicit Kelvin - always convert
    if "kelvin" in units or units == "k":
        return da - 273.15
    
    # Ambiguous units - use heuristic
    mean_val = float(da.mean(skipna=True))
    if mean_val > 100:
        return da - 273.15
    
    return da


def get_lake_mask(ds: xr.Dataset) -> np.ndarray:
    """Get lake mask, handling both [0,1] and [lake_id, nan] conventions."""
    if "lakeid" not in ds:
        # Fallback: use any finite values
        for var in ["temp_filled", "lake_surface_water_temperature", "lswt_mean_trimmed"]:
            if var in ds:
                return np.any(np.isfinite(ds[var].values), axis=0)
        raise ValueError("No lakeid or data variable found")
    
    lakeid_vals = ds["lakeid"].values
    # Auto-detect convention: [0,1] vs [lake_id, nan]
    if np.nanmax(lakeid_vals) == 1:
        return lakeid_vals == 1
    else:
        return np.isfinite(lakeid_vals) & (lakeid_vals != 0)


def extract_lake_series(ds: xr.Dataset, var_name: str, 
                        quality_threshold: Optional[int] = None) -> Optional[LakeSeries]:
    """
    Extract time series from a dataset.
    
    Args:
        ds: xarray Dataset
        var_name: variable name to extract
        quality_threshold: if provided, mask out pixels where quality_level < threshold
                          (only applies when quality_level variable exists in ds)
    """
    if var_name not in ds:
        return None
    
    lat = ds["lat"].values
    lon = ds["lon"].values
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon
    
    da = to_celsius(ds[var_name])
    
    # Apply quality filter if requested
    if quality_threshold is not None and "quality_level" in ds:
        quality_mask = ds["quality_level"] >= quality_threshold
        da = da.where(quality_mask)
        print(f"[LSWTPlots] Applied quality_level >= {quality_threshold} filter to {var_name}")
    
    mask = get_lake_mask(ds)
    
    if not mask.any():
        return None
    
    # Find center pixel
    lat_c = np.nanmean(lat2d[mask])
    lon_c = np.nanmean(lon2d[mask])
    iy_all, ix_all = np.where(mask)
    d2 = (lat2d[iy_all, ix_all] - lat_c)**2 + (lon2d[iy_all, ix_all] - lon_c)**2
    center_idx = np.argmin(d2)
    iy_c, ix_c = int(iy_all[center_idx]), int(ix_all[center_idx])
    
    time = pd.to_datetime(ds["time"].values)
    all_pixels = da.where(mask).stack(space=("lat", "lon")).dropna(dim="space", how="all").values
    center_pixel = da.isel(lat=iy_c, lon=ix_c).values
    
    return LakeSeries(time=time, all_pixels=all_pixels, center_pixel=center_pixel)


def extract_climatology(ds: xr.Dataset) -> Optional[ClimatologySeries]:
    """Extract climatology series from dataset."""
    var_name = None
    for name in ["lswt_mean_trimmed", "lswt_mean_trimmed_345", "lswt_mean", "climatology"]:
        if name in ds:
            var_name = name
            break
    
    if var_name is None:
        return None
    
    lat = ds["lat"].values
    lon = ds["lon"].values
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon
    
    da = to_celsius(ds[var_name])
    mask = get_lake_mask(ds)
    
    if not mask.any():
        return None
    
    # Find center pixel
    lat_c = np.nanmean(lat2d[mask])
    lon_c = np.nanmean(lon2d[mask])
    iy_all, ix_all = np.where(mask)
    d2 = (lat2d[iy_all, ix_all] - lat_c)**2 + (lon2d[iy_all, ix_all] - lon_c)**2
    center_idx = np.argmin(d2)
    iy_c, ix_c = int(iy_all[center_idx]), int(ix_all[center_idx])
    
    n_days = da.shape[0]
    day_of_year = np.arange(1, n_days + 1)
    
    all_pixels = da.where(mask).stack(space=("lat", "lon")).dropna(dim="space", how="all").values
    center_pixel = da.isel(lat=iy_c, lon=ix_c).values
    
    return ClimatologySeries(
        day_of_year=day_of_year,
        all_pixels=all_pixels,
        center_pixel=center_pixel
    )


# ==============================================================================
# Plotting Helpers
# ==============================================================================

def compute_year_ticks(time: pd.DatetimeIndex) -> tuple:
    """Compute yearly tick positions and labels."""
    years = np.unique(time.year)
    y_start = 1995 if 1995 in years else years.min()
    y_end = 2022 if 2022 in years else years.max()
    
    positions, labels = [], []
    for y in range(y_start, y_end + 1):
        if y in years:
            target = pd.Timestamp(y, 1, 1, 12)
            idx = int(np.argmin(np.abs(time - target)))
            positions.append(time[idx])
            labels.append(str(y))
    
    return positions, labels


def split_timeline(time: pd.DatetimeIndex) -> tuple:
    """Split timeline into first and second half."""
    years = np.unique(time.year)
    y_start = 1995 if 1995 in years else years.min()
    y_end = 2022 if 2022 in years else years.max()
    mid_year = y_start + (y_end - y_start) // 2
    
    i0 = int(np.searchsorted(time.values, np.datetime64(f"{y_start}-01-01")))
    i1 = int(np.searchsorted(time.values, np.datetime64(f"{mid_year + 1}-01-01")))
    
    return i0, i1, mid_year


def compute_ylim(all_pixels: np.ndarray, center_pixel: np.ndarray, padding: float = 0.02) -> tuple:
    """Compute y-axis limits."""
    all_vals = []
    finite_all = all_pixels[np.isfinite(all_pixels)]
    finite_center = center_pixel[np.isfinite(center_pixel)]
    if len(finite_all) > 0:
        all_vals.extend(finite_all)
    if len(finite_center) > 0:
        all_vals.extend(finite_center)
    
    if not all_vals:
        return (0, 30)
    
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    pad = padding * max(1.0, y_max - y_min)
    return (y_min - pad, y_max + pad)


def plot_panel(ax, x_vals, all_pixels, center_pixel, seg_slice, y_lim, 
               tick_pos, tick_lab, title, show_legend=False):
    """Plot a single panel."""
    ax.yaxis.grid(True, linestyle='--', linewidth=0.2)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.2)
    ax.set_ylim(*y_lim)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    
    x_seg = x_vals[seg_slice] if hasattr(x_vals, '__getitem__') else x_vals
    arr = all_pixels[seg_slice, :]
    center = center_pixel[seg_slice]
    
    # All pixels (faint)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        valid = np.isfinite(col)
        if valid.any():
            ax.plot(x_seg[valid], col[valid], '-', color='black', alpha=0.15, lw=0.2)
    
    # Center pixel (bold)
    valid_c = np.isfinite(center)
    if valid_c.any():
        ax.plot(x_seg[valid_c], center[valid_c], '-', color='red', lw=0.6, label='center pixel')
    
    if show_legend:
        ax.legend(fontsize=8)


# ==============================================================================
# Main Plotting Functions
# ==============================================================================

def plot_single_series(series: LakeSeries, lake_id: int, label: str, save_dir: str) -> Optional[str]:
    """Plot time series (2-panel split). Returns filepath."""
    if series is None or len(series.time) == 0:
        return None
    
    time = series.time
    i0, i1, mid_year = split_timeline(time)
    
    tt0, tt1 = time[i0:i1], time[i1:]
    if len(tt0) == 0 or len(tt1) == 0:
        # Not enough data to split
        return None
    
    pos0, lab0 = compute_year_ticks(tt0)
    pos1, lab1 = compute_year_ticks(tt1)
    
    y_lim = compute_ylim(series.all_pixels, series.center_pixel)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    fig.suptitle(f"LSWT Lake {lake_id} - {label}", fontsize=13)
    
    plot_panel(axes[0], time, series.all_pixels, series.center_pixel, 
               slice(i0, i1), y_lim, pos0, lab0,
               f"First half ({time[i0].year}-{mid_year})", show_legend=True)
    plot_panel(axes[1], time, series.all_pixels, series.center_pixel,
               slice(i1, None), y_lim, pos1, lab1,
               f"Second half ({mid_year+1}-{time[-1].year})")
    
    axes[1].set_xlabel("Time", fontsize=11)
    for ax in axes:
        ax.set_ylabel("LSWT (°C)", fontsize=11)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"LAKE{lake_id:09d}_{label}.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return filepath


def plot_climatology(clim: ClimatologySeries, lake_id: int, save_dir: str) -> Optional[str]:
    """Plot climatology (365/366 days on x-axis). Returns filepath."""
    if clim is None:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"LSWT Climatology - Lake {lake_id}", fontsize=13)
    
    doy = clim.day_of_year
    y_lim = compute_ylim(clim.all_pixels, clim.center_pixel)
    
    ax.set_ylim(*y_lim)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.2)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.2)
    
    # All pixels (faint)
    for j in range(clim.all_pixels.shape[1]):
        col = clim.all_pixels[:, j]
        valid = np.isfinite(col)
        if valid.any():
            ax.plot(doy[valid], col[valid], '-', color='black', alpha=0.15, lw=0.3)
    
    # Center pixel (bold)
    valid_c = np.isfinite(clim.center_pixel)
    if valid_c.any():
        ax.plot(doy[valid_c], clim.center_pixel[valid_c], '-', color='red', lw=0.8, 
                label='center pixel')
    
    ax.set_xlabel("Day of Year", fontsize=11)
    ax.set_ylabel("LSWT (°C)", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(1, len(doy))
    
    # Month ticks
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"LAKE{lake_id:09d}_Climatology.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return filepath


def plot_comparison(s_left: Optional[LakeSeries], s_right: Optional[LakeSeries],
                    lake_id: int, label_left: str, label_right: str, 
                    save_dir: str) -> Optional[str]:
    """Side-by-side comparison (2x2 grid). Returns filepath or None."""
    if s_left is None and s_right is None:
        return None
    
    # Find common time
    if s_left is not None and s_right is not None:
        common_time = s_left.time.intersection(s_right.time)
        if len(common_time) == 0:
            return None
        
        # Align both to common time
        def align(s, ref):
            aligned_center = np.full(len(ref), np.nan)
            aligned_all = np.full((len(ref), s.all_pixels.shape[1]), np.nan)
            pos = pd.Series(np.arange(len(s.time)), index=s.time)
            pos_al = pos.reindex(ref)
            valid = pos_al.notna().values
            idx = pos_al[valid].astype(int).values
            aligned_center[valid] = s.center_pixel[idx]
            aligned_all[valid, :] = s.all_pixels[idx, :]
            return LakeSeries(time=ref, all_pixels=aligned_all, center_pixel=aligned_center)
        
        left = align(s_left, common_time)
        right = align(s_right, common_time)
    else:
        if s_left is not None:
            common_time = s_left.time
            left, right = s_left, None
        else:
            common_time = s_right.time
            left, right = None, s_right
    
    i0, i1, mid_year = split_timeline(common_time)
    tt0, tt1 = common_time[i0:i1], common_time[i1:]
    
    if len(tt0) == 0 or len(tt1) == 0:
        return None
    
    pos0, lab0 = compute_year_ticks(tt0)
    pos1, lab1 = compute_year_ticks(tt1)
    
    # Common y-lim
    all_vals = []
    for s in [left, right]:
        if s is not None:
            all_vals.extend(s.all_pixels[np.isfinite(s.all_pixels)])
            all_vals.extend(s.center_pixel[np.isfinite(s.center_pixel)])
    if all_vals:
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        pad = 0.02 * max(1.0, y_max - y_min)
        y_lim = (y_min - pad, y_max + pad)
    else:
        y_lim = (0, 30)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    fig.suptitle(f"LSWT Lake {lake_id} - {label_left} vs {label_right}", fontsize=13)
    
    def plot_side(s, ax0, ax1, title_base):
        if s is None:
            for ax in [ax0, ax1]:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=11)
                ax.set_title(f"{title_base}", fontsize=10)
            return
        plot_panel(ax0, common_time, s.all_pixels, s.center_pixel,
                   slice(i0, i1), y_lim, pos0, lab0, 
                   f"{title_base} (first half)", show_legend=True)
        plot_panel(ax1, common_time, s.all_pixels, s.center_pixel,
                   slice(i1, None), y_lim, pos1, lab1,
                   f"{title_base} (second half)")
    
    plot_side(left, axes[0, 0], axes[1, 0], label_left)
    plot_side(right, axes[0, 1], axes[1, 1], label_right)
    
    axes[1, 0].set_xlabel("Time", fontsize=11)
    axes[1, 1].set_xlabel("Time", fontsize=11)
    axes[0, 0].set_ylabel("LSWT (°C)", fontsize=11)
    axes[1, 0].set_ylabel("LSWT (°C)", fontsize=11)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"LAKE{lake_id:09d}_{label_left}_vs_{label_right}.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return filepath


# ==============================================================================
# Yearly Plotting Functions
# ==============================================================================

def plot_yearly_series(series: LakeSeries, lake_id: int, label: str, save_dir: str) -> Optional[str]:
    """
    Plot time series with one row per year.
    
    Creates a single file with all years stacked vertically.
    Returns filepath or None if no data.
    """
    if series is None or len(series.time) == 0:
        return None
    
    time = series.time
    years = sorted(np.unique(time.year))
    
    if len(years) == 0:
        return None
    
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(14, 2.5 * n_years), squeeze=False)
    axes = axes.flatten()
    
    fig.suptitle(f"LSWT Lake {lake_id} - {label} (Yearly)", fontsize=13, y=1.0)
    
    # Compute global y-limits across all years for consistency
    y_lim = compute_ylim(series.all_pixels, series.center_pixel)
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        
        # Get indices for this year
        year_mask = time.year == year
        year_indices = np.where(year_mask)[0]
        
        if len(year_indices) == 0:
            ax.text(0.5, 0.5, f"{year}: No data", ha='center', va='center', fontsize=11)
            ax.set_title(f"{year}", fontsize=10)
            continue
        
        year_time = time[year_mask]
        year_all_pixels = series.all_pixels[year_mask, :]
        year_center = series.center_pixel[year_mask]
        
        # Setup axes
        ax.set_ylim(*y_lim)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.2)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.2)
        ax.set_title(f"{year}", fontsize=10)
        
        # Plot all pixels (faint)
        for j in range(year_all_pixels.shape[1]):
            col = year_all_pixels[:, j]
            valid = np.isfinite(col)
            if valid.any():
                ax.plot(year_time[valid], col[valid], '-', color='black', alpha=0.15, lw=0.2)
        
        # Plot center pixel (bold red)
        valid_c = np.isfinite(year_center)
        if valid_c.any():
            ax.plot(year_time[valid_c], year_center[valid_c], '-', color='red', lw=0.6, 
                   label='center pixel' if idx == 0 else None)
        
        ax.set_ylabel("LSWT (°C)", fontsize=9)
        
        # Set x-axis to show months
        ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
        
        # Month ticks
        month_positions = [pd.Timestamp(year, m, 1) for m in range(1, 13)]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(month_positions)
        ax.set_xticklabels(month_labels, fontsize=8)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    axes[-1].set_xlabel("Month", fontsize=11)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"LAKE{lake_id:09d}_{label}_yearly.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return filepath


def plot_yearly_comparison(s_left: Optional[LakeSeries], s_right: Optional[LakeSeries],
                           lake_id: int, label_left: str, label_right: str, 
                           save_dir: str) -> Optional[str]:
    """
    Side-by-side comparison with one row per year.
    
    Creates a single file with all years stacked vertically, 2 columns (left vs right).
    Returns filepath or None if no data.
    """
    if s_left is None and s_right is None:
        return None
    
    # Determine which series to use for years
    if s_left is not None and s_right is not None:
        all_years = sorted(set(s_left.time.year) | set(s_right.time.year))
    elif s_left is not None:
        all_years = sorted(np.unique(s_left.time.year))
    else:
        all_years = sorted(np.unique(s_right.time.year))
    
    if len(all_years) == 0:
        return None
    
    n_years = len(all_years)
    fig, axes = plt.subplots(n_years, 2, figsize=(14, 2.5 * n_years), squeeze=False, sharey=True)
    
    fig.suptitle(f"LSWT Lake {lake_id} - {label_left} vs {label_right} (Yearly)", fontsize=13, y=1.0)
    
    # Compute global y-limits
    all_vals = []
    for s in [s_left, s_right]:
        if s is not None:
            all_vals.extend(s.all_pixels[np.isfinite(s.all_pixels)])
            all_vals.extend(s.center_pixel[np.isfinite(s.center_pixel)])
    if all_vals:
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        pad = 0.02 * max(1.0, y_max - y_min)
        y_lim = (y_min - pad, y_max + pad)
    else:
        y_lim = (0, 30)
    
    def plot_year_panel(ax, series, year, label, show_legend=False):
        """Plot a single year panel for one series."""
        ax.set_ylim(*y_lim)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.2)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.2)
        
        if series is None:
            ax.text(0.5, 0.5, f"No data", ha='center', va='center', fontsize=10)
            ax.set_title(f"{year} - {label}", fontsize=9)
            return
        
        # Get indices for this year
        year_mask = series.time.year == year
        year_indices = np.where(year_mask)[0]
        
        if len(year_indices) == 0:
            ax.text(0.5, 0.5, f"No data", ha='center', va='center', fontsize=10)
            ax.set_title(f"{year} - {label}", fontsize=9)
            return
        
        year_time = series.time[year_mask]
        year_all_pixels = series.all_pixels[year_mask, :]
        year_center = series.center_pixel[year_mask]
        
        ax.set_title(f"{year} - {label}", fontsize=9)
        
        # Plot all pixels (faint)
        for j in range(year_all_pixels.shape[1]):
            col = year_all_pixels[:, j]
            valid = np.isfinite(col)
            if valid.any():
                ax.plot(year_time[valid], col[valid], '-', color='black', alpha=0.15, lw=0.2)
        
        # Plot center pixel (bold red)
        valid_c = np.isfinite(year_center)
        if valid_c.any():
            ax.plot(year_time[valid_c], year_center[valid_c], '-', color='red', lw=0.6,
                   label='center pixel' if show_legend else None)
        
        # Set x-axis to show months
        ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
        
        # Month ticks
        month_positions = [pd.Timestamp(year, m, 1) for m in range(1, 13)]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(month_positions)
        ax.set_xticklabels(month_labels, fontsize=7)
        
        if show_legend:
            ax.legend(fontsize=7, loc='upper right')
    
    for idx, year in enumerate(all_years):
        # Left column
        plot_year_panel(axes[idx, 0], s_left, year, label_left, show_legend=(idx == 0))
        # Right column
        plot_year_panel(axes[idx, 1], s_right, year, label_right, show_legend=False)
        
        # Y-axis label only on left
        axes[idx, 0].set_ylabel("LSWT (°C)", fontsize=8)
    
    axes[-1, 0].set_xlabel("Month", fontsize=10)
    axes[-1, 1].set_xlabel("Month", fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"LAKE{lake_id:09d}_{label_left}_vs_{label_right}_yearly.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return filepath


# ==============================================================================
# Post Processing Step
# ==============================================================================

class LSWTPlotsStep(PostProcessingStep):
    """Generate LSWT time series plots at end of postprocessing."""
    
    def __init__(self, original_ts_path: Optional[str] = None, 
                 quality_threshold: int = 3):
        super().__init__()
        self.original_ts_path = original_ts_path
        self.quality_threshold = quality_threshold
    
    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        # Always returns True when called directly; we handle missing files gracefully
        return True
    
    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        lake_id = ctx.lake_id or 0
        post_dir = os.path.dirname(ctx.output_path)
        plot_dir = os.path.join(post_dir, "plots")
        
        print(f"[LSWTPlots] Generating plots for lake {lake_id} in {plot_dir}")
        print(f"[LSWTPlots] Using quality_threshold >= {self.quality_threshold} for observations")
        
        results = []
        
        # Initialize all series as None
        observation = None
        climatology = None
        dineof = None              # raw, sparse
        dineof_filtered = None     # filtered, sparse
        dineof_interp = None       # raw, full daily
        dineof_filtered_interp = None  # filtered, full daily
        dincae = None
        dincae_interp = None       # DINCAE, full daily
        
        # Build paths explicitly from post_dir and lake_id to avoid issues when
        # ctx.output_path doesn't end with "_dineof.nc" (e.g., when running DINCAE postprocessor)
        lake_id_str = f"LAKE{lake_id:09d}"
        
        # Pattern: LAKE{id}-*_dineof.nc, LAKE{id}-*_dincae.nc, etc.
        # Try to find files matching expected patterns
        def find_output_file(suffix: str) -> Optional[str]:
            """Find output file with given suffix in post_dir."""
            import glob
            pattern = os.path.join(post_dir, f"{lake_id_str}-*{suffix}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
            # Also try without the dash (older naming convention)
            pattern = os.path.join(post_dir, f"{lake_id_str}*{suffix}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
            return None
        
        # Load main DINEOF output (raw, sparse)
        dineof_path = find_output_file("_dineof.nc")
        if dineof_path and os.path.exists(dineof_path):
            try:
                with xr.open_dataset(dineof_path) as ds_file:
                    dineof = extract_lake_series(ds_file, "temp_filled")
                print(f"[LSWTPlots] Loaded DINEOF from: {os.path.basename(dineof_path)}")
            except Exception as e:
                print(f"[LSWTPlots] Could not load DINEOF: {e}")
        
        # Load DINEOF filtered (sparse)
        filtered_path = find_output_file("_dineof_eof_filtered.nc")
        if filtered_path and os.path.exists(filtered_path):
            try:
                with xr.open_dataset(filtered_path) as ds_file:
                    dineof_filtered = extract_lake_series(ds_file, "temp_filled")
                print(f"[LSWTPlots] Loaded DINEOF filtered from: {os.path.basename(filtered_path)}")
            except Exception as e:
                print(f"[LSWTPlots] Could not load DINEOF filtered: {e}")
        
        # Load DINEOF interpolated (raw, full daily)
        interp_path = find_output_file("_dineof_eof_interp_full.nc")
        if interp_path and os.path.exists(interp_path):
            try:
                with xr.open_dataset(interp_path) as ds_file:
                    dineof_interp = extract_lake_series(ds_file, "temp_filled")
                print(f"[LSWTPlots] Loaded DINEOF interp from: {os.path.basename(interp_path)}")
            except Exception as e:
                print(f"[LSWTPlots] Could not load DINEOF interp: {e}")
        
        # Load DINEOF filtered interpolated (filtered, full daily)
        filtered_interp_path = find_output_file("_dineof_eof_filtered_interp_full.nc")
        if filtered_interp_path and os.path.exists(filtered_interp_path):
            try:
                with xr.open_dataset(filtered_interp_path) as ds_file:
                    dineof_filtered_interp = extract_lake_series(ds_file, "temp_filled")
                print(f"[LSWTPlots] Loaded DINEOF filtered interp from: {os.path.basename(filtered_interp_path)}")
            except Exception as e:
                print(f"[LSWTPlots] Could not load DINEOF filtered interp: {e}")
        
        # Load DINCAE output
        dincae_path = find_output_file("_dincae.nc")
        if dincae_path and os.path.exists(dincae_path):
            try:
                with xr.open_dataset(dincae_path) as ds_file:
                    dincae = extract_lake_series(ds_file, "temp_filled")
                print(f"[LSWTPlots] Loaded DINCAE from: {os.path.basename(dincae_path)}")
            except Exception as e:
                print(f"[LSWTPlots] Could not load DINCAE: {e}")
        
        # Load DINCAE interpolated (full daily)
        dincae_interp_path = find_output_file("_dincae_interp_full.nc")
        if dincae_interp_path and os.path.exists(dincae_interp_path):
            try:
                with xr.open_dataset(dincae_interp_path) as ds_file:
                    dincae_interp = extract_lake_series(ds_file, "temp_filled")
                print(f"[LSWTPlots] Loaded DINCAE interp from: {os.path.basename(dincae_interp_path)}")
            except Exception as e:
                print(f"[LSWTPlots] Could not load DINCAE interp: {e}")
        
        # Load observation time series (with quality filter)
        observation = None
        if self.original_ts_path and os.path.exists(self.original_ts_path):
            try:
                with xr.open_dataset(self.original_ts_path) as ds_file:
                    observation = extract_lake_series(
                        ds_file, "lake_surface_water_temperature",
                        quality_threshold=self.quality_threshold
                    )
                print(f"[LSWTPlots] Loaded Observation from: {os.path.basename(self.original_ts_path)}")
            except Exception as e:
                print(f"[LSWTPlots] Could not load observation: {e}")
        
        # Load climatology
        if ctx.climatology_path and os.path.exists(ctx.climatology_path):
            try:
                with xr.open_dataset(ctx.climatology_path) as ds_file:
                    climatology = extract_climatology(ds_file)
            except Exception as e:
                print(f"[LSWTPlots] Could not load climatology: {e}")
        
        # ==================== Individual Plots ====================
        individual_series = [
            (observation, "Observation"),
            (dineof, "DINEOF"),
            (dineof_filtered, "DINEOF_filtered"),
            (dineof_interp, "DINEOF_interp"),
            (dineof_filtered_interp, "DINEOF_filtered_interp"),
            (dincae, "DINCAE"),
            (dincae_interp, "DINCAE_interp"),
        ]
        
        for series, label in individual_series:
            if series:
                path = plot_single_series(series, lake_id, label, plot_dir)
                if path:
                    results.append(path)
                    print(f"[LSWTPlots] Saved: {os.path.basename(path)}")
        
        if climatology:
            path = plot_climatology(climatology, lake_id, plot_dir)
            if path:
                results.append(path)
                print(f"[LSWTPlots] Saved: {os.path.basename(path)}")
        
        # ==================== Comparison Plots ====================
        # Sparse comparisons: raw vs filtered
        comparisons = [
            (dineof, dineof_filtered, "DINEOF", "DINEOF_filtered"),
        ]
        
        # Full daily comparisons: raw_interp vs filtered_interp
        comparisons.append(
            (dineof_interp, dineof_filtered_interp, "DINEOF_interp", "DINEOF_filtered_interp")
        )
        
        # Observation vs outputs
        comparisons.extend([
            (observation, dineof, "Observation", "DINEOF"),
            (observation, dincae, "Observation", "DINCAE"),
        ])
        
        # DINEOF vs DINCAE (sparse)
        if dincae is not None:
            comparisons.append((dineof, dincae, "DINEOF", "DINCAE"))
            comparisons.append((dineof_filtered, dincae, "DINEOF_filtered", "DINCAE"))
        
        # DINCAE sparse vs DINCAE interp
        if dincae_interp is not None:
            comparisons.append((dincae, dincae_interp, "DINCAE", "DINCAE_interp"))
            # DINEOF interp vs DINCAE interp (daily-scale method comparison)
            comparisons.append((dineof_interp, dincae_interp, "DINEOF_interp", "DINCAE_interp"))
            comparisons.append((dineof_filtered_interp, dincae_interp, "DINEOF_filtered_interp", "DINCAE_interp"))
        
        for s1, s2, l1, l2 in comparisons:
            path = plot_comparison(s1, s2, lake_id, l1, l2, plot_dir)
            if path:
                results.append(path)
                print(f"[LSWTPlots] Saved: {os.path.basename(path)}")
        
        # ==================== Yearly Individual Plots ====================
        print(f"[LSWTPlots] Generating yearly plots...")
        
        for series, label in individual_series:
            if series:
                path = plot_yearly_series(series, lake_id, label, plot_dir)
                if path:
                    results.append(path)
                    print(f"[LSWTPlots] Saved: {os.path.basename(path)}")
        
        # ==================== Yearly Comparison Plots ====================
        for s1, s2, l1, l2 in comparisons:
            path = plot_yearly_comparison(s1, s2, lake_id, l1, l2, plot_dir)
            if path:
                results.append(path)
                print(f"[LSWTPlots] Saved: {os.path.basename(path)}")
        
        print(f"[LSWTPlots] Generated {len(results)} plots total")
        
        return ds if ds is not None else xr.Dataset()
    
    @property
    def name(self) -> str:
        return "LSWTPlots"