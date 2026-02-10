#!/usr/bin/env python3
"""
LSWT Reconstruction Blowup Detector (v2 — with local sparsity metrics)

Detects anomalous frames in DINEOF reconstruction output by reading:
  - dineof_results.nc  (anomaly-space reconstruction, no time coord)
  - eofs.nc            (time coordinate, eigenvalues)

Frame count between the two files is verified exactly before any analysis.

Detection methods (all applied in anomaly space):
  1. Frame-to-frame jump:  |lake_mean(t) - lake_mean(t-1)| outlier detection
  2. Within-frame spread:  frame_std(t) outlier detection
  3. Frame mean extreme:   |lake_mean(t)| outlier detection

All outlier detection uses robust MAD-based thresholding:
  flagged = |x - median(x)| > k * 1.4826 * MAD(x)

Local sparsity analysis (NEW in v2):
  For each frame, computes metrics characterising the local temporal
  observation density and regularity within windows of ±30, ±60, ±90 days.

  Metrics per (frame, radius):
    rms_gap:      √(mean(g²)) of gaps within window. The primary combined
                  metric: from interpolation theory, reconstruction error
                  scales as Σg², so rms_gap captures BOTH sparsity (large
                  mean gap) and irregularity (clustered obs → large gaps
                  in the sum). Single best predictor of DINEOF instability.
    n_obs:        Number of observations in window.
    obs_density:  n_obs / window_length (observations per day).
    max_gap:      Largest gap within window (worst unconstrained stretch).
    gap_cv:       Coefficient of variation = std(gaps)/mean(gaps).
                  Pure irregularity, independent of density. Zero for
                  a perfectly regular grid.
    gap_entropy:  Normalised Shannon entropy of gap proportions p = g/L.
                  1.0 for perfectly uniform gaps, lower when one gap
                  dominates (clustered observations).

  Gap sequences include boundary gaps (from window edge to nearest
  observation), ensuring frames at the edge of dense zones correctly
  show large gaps extending into voids.

Outputs:
  blowup_diagnostics/
  ├── blowup_events_all_lakes.csv    — one row per detected blowup frame
  ├── blowup_summary_all_lakes.csv   — one row per lake (now includes
  │                                     sparsity summary stats per radius)
  └── per_lake/{lake_id}/
      ├── blowup_events.csv
      └── frame_stats.csv            — per-frame stats for all frames,
                                       including sparsity columns

Usage:
  python blowup_detector.py --exp-dir /path/to/exp
  python blowup_detector.py --exp-dir /path/to/exp --lake-id 000000044
  python blowup_detector.py --exp-dir /path/to/exp --k 4.0

Author: Shaerdan / NCEO / University of Reading
Date: February 2026
"""

import argparse
import os
import sys
import csv
import glob
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray is required. pip install xarray netcdf4")
    sys.exit(1)


# ========== Constants ==========

EPOCH = np.datetime64("1981-01-01T12:00:00")


def day_to_datestr(d: int) -> str:
    dt = EPOCH + np.timedelta64(d, "D")
    return str(dt)[:10]


# ========== Data loading ==========

def load_time_from_eofs(eofs_path: str) -> np.ndarray:
    """Load physical time (days since epoch) from eofs.nc."""
    with xr.open_dataset(eofs_path) as ds:
        if "time" not in ds.coords and "time" not in ds.data_vars:
            raise ValueError(f"No 'time' variable in {eofs_path}")

        vals = ds["time"].values
        if np.issubdtype(vals.dtype, np.datetime64):
            base = np.datetime64("1981-01-01T12:00:00", "ns")
            days = ((vals.astype("datetime64[ns]") - base) / np.timedelta64(1, "D")).astype("int64")
        elif np.issubdtype(vals.dtype, np.integer) or np.issubdtype(vals.dtype, np.floating):
            days = vals.astype("int64")
        else:
            raise ValueError(f"Unexpected time dtype: {vals.dtype}")

        n_t = ds.sizes.get("t", ds.sizes.get("time", -1))

    return days, n_t


def load_reconstruction(dineof_results_path: str) -> np.ndarray:
    """
    Load temp_filled from dineof_results.nc.

    Returns:
        3D array (time, lat, lon) with DINEOF missing_value replaced by NaN.
    """
    with xr.open_dataset(dineof_results_path) as ds:
        if "temp_filled" not in ds:
            raise ValueError(f"No 'temp_filled' in {dineof_results_path}")

        tf = ds["temp_filled"].values.copy()  # (dim003=time, dim002=lat, dim001=lon)

        # Replace DINEOF missing value (9999) with NaN
        tf[tf > 9990] = np.nan
        tf[tf < -9990] = np.nan

    return tf


def load_lake_mask(eofs_path: str) -> Optional[np.ndarray]:
    """Try to extract lake mask from spatial EOFs (pixels where spatial_eof0 is not NaN)."""
    with xr.open_dataset(eofs_path) as ds:
        if "spatial_eof0" in ds:
            s0 = ds["spatial_eof0"].values
            return np.isfinite(s0)
    return None


def load_eigenvalues(eofs_path: str) -> Optional[np.ndarray]:
    """Load eigenvalues from eofs.nc."""
    with xr.open_dataset(eofs_path) as ds:
        if "eigenvalues" in ds:
            return ds["eigenvalues"].values.copy()
    return None


# ========== Spatial observation metrics ==========

def find_merged_file(alpha_dir: str) -> Optional[str]:
    """Find the *filled_fine_dineof.nc merged file in the parallel post/ directory."""
    post_dir = alpha_dir.replace("/dineof/", "/post/")
    candidates = glob.glob(os.path.join(post_dir, "*filled_fine_dineof.nc"))
    if candidates:
        return candidates[0]
    return None


def compute_shore_distance(lakeid: np.ndarray) -> np.ndarray:
    """
    Compute distance (in pixels) from each lake pixel to the nearest non-lake pixel.

    Uses scipy's Euclidean distance transform on the lake mask.
    Pixels at the shore (adjacent to land) have distance ~1.
    Pixels deep in the lake interior have large distances.

    Args:
        lakeid: 2D array (lat, lon), 1=lake, 0=land

    Returns:
        2D array same shape, distance to nearest land pixel.
        Land pixels have distance 0.
    """
    from scipy.ndimage import distance_transform_edt
    # distance_transform_edt computes distance from each 1-pixel to nearest 0-pixel
    # We want distance from each lake pixel to nearest land pixel
    return distance_transform_edt(lakeid.astype(bool))


def load_obs_fraction_and_shore(
    merged_path: str,
    n_t_expected: int,
    dineof_lake_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load per-frame observation fraction and shore distance stats from merged file.

    Reads data_source (0=gap, 1=observed, 2=cv, 255=not_recon) and lakeid.

    Uses dineof_lake_mask (from eofs.nc spatial_eof0) as the reference mask if provided,
    since the merged file's lakeid may include pixels that were removed during preprocessing
    (temporal coverage filter). obs_fraction = n_observed / n_dineof_lake_pixels.

    Returns:
        obs_fraction:          1D array (n_time_merged,) — fraction of DINEOF lake pixels observed per frame
        obs_shore_dist_median: 1D array (n_time_merged,) — median shore distance of observed pixels per frame
        shore_dist_map:        2D array (lat, lon) — pixel distance to shore (for reference)
        merged_days:           1D array (n_time_merged,) — days since epoch for each frame in merged file

    Returns (None, None, None, None) if merged file cannot be read.
    """
    try:
        with xr.open_dataset(merged_path) as ds:
            if "data_source" not in ds or "lakeid" not in ds:
                print(f"    WARNING: merged file missing data_source or lakeid: {merged_path}")
                return None, None, None, None

            lakeid = ds["lakeid"].values  # (lat, lon), 1=lake 0=land

            # Use DINEOF lake mask if provided (filtered subset of lakeid==1),
            # otherwise fall back to merged file's lakeid
            if dineof_lake_mask is not None:
                lake_mask = dineof_lake_mask
                n_lake = int(lake_mask.sum())
                n_lakeid = int((lakeid == 1).sum())
                if n_lake != n_lakeid:
                    print(f"    INFO: using DINEOF mask ({n_lake} px) vs merged lakeid ({n_lakeid} px)")
            else:
                lake_mask = lakeid == 1
                n_lake = int(lake_mask.sum())

            if n_lake == 0:
                print(f"    WARNING: no lake pixels in mask")
                return None, None, None, None

            # Load time axis from merged file
            if "time" not in ds.coords and "time" not in ds.data_vars:
                print(f"    WARNING: no time variable in merged file")
                return None, None, None, None

            merged_time_raw = ds["time"].values
            # Convert to days since epoch (same as eofs.nc convention)
            if np.issubdtype(merged_time_raw.dtype, np.datetime64):
                base = np.datetime64("1981-01-01T12:00:00", "ns")
                merged_days = ((merged_time_raw.astype("datetime64[ns]") - base)
                               / np.timedelta64(1, "D")).astype("int64")
            elif np.issubdtype(merged_time_raw.dtype, np.integer) or np.issubdtype(merged_time_raw.dtype, np.floating):
                merged_days = merged_time_raw.astype("int64")
            else:
                print(f"    WARNING: unexpected time dtype {merged_time_raw.dtype} in merged file")
                return None, None, None, None

            # Compute shore distance from the full lakeid (not the DINEOF subset)
            # so that shore distance reflects actual lake geometry
            shore_dist = compute_shore_distance(lakeid)

            n_t = len(merged_days)
            if n_t != n_t_expected:
                print(f"    INFO: merged file time dim ({n_t}) != DINEOF time dim ({n_t_expected}), "
                      f"will align by date")

            obs_frac = np.full(n_t, np.nan)
            obs_shore_med = np.full(n_t, np.nan)

            # Process frame by frame to avoid loading entire 3D array
            data_source_var = ds["data_source"]

            for t in range(n_t):
                ds_frame = data_source_var.isel(time=t).values  # (lat, lon)

                # Observed = data_source == 1 AND on DINEOF lake mask
                obs_mask = (ds_frame == 1) & lake_mask
                n_obs = int(obs_mask.sum())
                obs_frac[t] = n_obs / n_lake

                if n_obs > 0:
                    obs_shore_med[t] = np.median(shore_dist[obs_mask])
                # else: stays NaN (no observations this frame)

        return obs_frac, obs_shore_med, shore_dist, merged_days

    except Exception as e:
        print(f"    WARNING: Failed to read merged file: {e}")
        return None, None, None, None


# ========== Frame statistics ==========

def compute_frame_stats(
    tf: np.ndarray,
    lake_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame statistics from the reconstruction cube.

    Args:
        tf: (time, lat, lon) reconstruction array, NaN for non-lake / missing.
        lake_mask: (lat, lon) boolean mask of lake pixels. If None, uses all finite pixels.

    Returns:
        Dict of 1D arrays keyed by stat name.
    """
    n_t = tf.shape[0]

    # Apply lake mask if available — set non-lake pixels to NaN
    if lake_mask is not None:
        tf_masked = tf.copy()
        tf_masked[:, ~lake_mask] = np.nan
    else:
        tf_masked = tf

    # Reshape to (time, n_pixels) for vectorized stats
    tf_flat = tf_masked.reshape(n_t, -1)

    with np.errstate(all="ignore"):
        frame_mean = np.nanmean(tf_flat, axis=1)
        frame_std = np.nanstd(tf_flat, axis=1)
        frame_min = np.nanmin(tf_flat, axis=1)
        frame_max = np.nanmax(tf_flat, axis=1)
        frame_range = frame_max - frame_min
        frame_n_valid = np.sum(np.isfinite(tf_flat), axis=1)

    # Frame-to-frame jumps in lake mean
    mean_jump = np.full(n_t, np.nan)
    mean_jump[1:] = np.abs(np.diff(frame_mean))

    return {
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "frame_min": frame_min,
        "frame_max": frame_max,
        "frame_range": frame_range,
        "frame_n_valid": frame_n_valid,
        "mean_jump": mean_jump,
    }


# ========== Outlier detection ==========

def detect_outliers_mad(values: np.ndarray, k: float = 4.0) -> np.ndarray:
    """
    Detect outliers using MAD-based robust threshold.

    flagged = |x - median(x)| > k * 1.4826 * MAD(x)

    Only operates on finite values. NaN entries are not flagged.
    """
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.zeros(len(values), dtype=bool)

    med = np.nanmedian(values[valid])
    mad = np.nanmedian(np.abs(values[valid] - med))
    rsd = 1.4826 * mad

    if rsd == 0 or not np.isfinite(rsd):
        return np.zeros(len(values), dtype=bool)

    flagged = np.zeros(len(values), dtype=bool)
    flagged[valid] = np.abs(values[valid] - med) > (k * rsd)
    return flagged


def detect_blowups(
    stats: Dict[str, np.ndarray],
    k: float = 4.0,
) -> Dict[str, np.ndarray]:
    """
    Apply detection methods and return per-method boolean masks.

    Primary (most reliable, undampened signal):
      1. frame_range_flag: within-frame pixel range (max-min) is an outlier.
         On blowup frames, pixels fan out wildly (range 30-60°C vs normal 3-6°C).
         This signal is NOT dampened by averaging and adapts per lake via MAD.

    Secondary (supplementary, may have more false positives):
      2. frame_std_flag:   within-frame std is an outlier (correlated with range)
      3. mean_jump_flag:   frame-to-frame lake-mean jump is an outlier
      4. frame_mean_flag:  absolute lake-mean is an outlier

    A frame is flagged if the PRIMARY method flags it.
    Secondary flags are recorded for diagnostic purposes but do not
    contribute to the combined flag.
    """
    # PRIMARY: frame range — strongest, least dampened signal
    frame_range_flag = detect_outliers_mad(stats["frame_range"], k=k)

    # SECONDARY: supplementary diagnostics
    frame_std_flag = detect_outliers_mad(stats["frame_std"], k=k)
    mean_jump_flag = detect_outliers_mad(stats["mean_jump"], k=k)
    frame_mean_flag = detect_outliers_mad(stats["frame_mean"], k=k)

    # Combined = primary only (secondary recorded but not used for flagging)
    combined = frame_range_flag.copy()

    return {
        "frame_range_flag": frame_range_flag,
        "frame_std_flag": frame_std_flag,
        "mean_jump_flag": mean_jump_flag,
        "frame_mean_flag": frame_mean_flag,
        "combined": combined,
    }


# ========== Gap context ==========

def compute_gap_at_index(days: np.ndarray, idx: int) -> Tuple[int, int]:
    """Return (gap_before, gap_after) in days. -1 if at boundary."""
    gap_before = int(days[idx] - days[idx - 1]) if idx > 0 else -1
    gap_after = int(days[idx + 1] - days[idx]) if idx < len(days) - 1 else -1
    return gap_before, gap_after


# ========== Local sparsity metrics ==========

# Arc radii (in days) for windowed sparsity computation
SPARSITY_RADII = [30, 60, 90]


def _compute_gap_sequence(days: np.ndarray, center_idx: int, radius: int) -> np.ndarray:
    """
    Compute the full gap sequence within [center - radius, center + radius].

    The window [a, b] is partitioned into gaps by observations at times
    t_1, t_2, ..., t_k within it:

        g_0 = t_1 - a           (gap from window start to first obs)
        g_1 = t_2 - t_1         (internal gaps)
        ...
        g_k = b - t_k           (gap from last obs to window end)

    These k+1 gaps sum exactly to 2*radius and fully characterise the
    coverage of the window. Including boundary gaps is critical: a frame
    at the edge of a dense zone with a void extending beyond the window
    will correctly show a large boundary gap.

    If no observations fall in the window, returns [2*radius] (one gap
    spanning the entire window).
    """
    center_day = days[center_idx]
    a = center_day - radius
    b = center_day + radius

    # Find all observations within [a, b] (inclusive)
    mask = (days >= a) & (days <= b)
    obs_days = days[mask]  # already sorted since days array is sorted

    if len(obs_days) == 0:
        return np.array([2 * radius], dtype=float)

    # Build gap sequence: boundary + internal + boundary
    gaps = []
    gaps.append(float(obs_days[0] - a))        # start boundary gap
    if len(obs_days) > 1:
        gaps.extend(np.diff(obs_days).astype(float).tolist())  # internal gaps
    gaps.append(float(b - obs_days[-1]))        # end boundary gap

    return np.array(gaps, dtype=float)


def compute_local_sparsity(
    days: np.ndarray,
    radii: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute local sparsity metrics for each frame at multiple arc radii.

    For each frame i at time t_i, and each radius Δ, we examine the window
    [t_i - Δ, t_i + Δ] and compute metrics from the gap sequence within it.

    Metrics per (frame, radius):

      Primary combined metric (theoretically motivated):
        rms_gap:     √(mean(g_j²))  — root-mean-square of gaps.
                     From interpolation theory, reconstruction error across
                     multiple gaps scales as Σ g_j², so rms_gap captures both
                     density (fewer obs → larger mean gap) and irregularity
                     (clustered obs → one or more large gaps in the sum).

      Density measures:
        n_obs:       number of observations in the window.
        obs_density: n_obs / (2*Δ), observations per day.

      Regularity measures:
        max_gap:     largest gap in the window. The "weakest link" — the
                     longest stretch where DINEOF temporal EOFs are
                     unconstrained by data.
        gap_cv:      coefficient of variation = std(gaps) / mean(gaps).
                     Pure irregularity metric, independent of density when
                     normalised. Zero for a perfectly regular grid.
        gap_entropy: normalised Shannon entropy of gap proportions
                     p_j = g_j / (2Δ).  Equals 1.0 for uniform gaps, lower
                     when one gap dominates. Captures how "concentrated" the
                     void is.

    Returns:
        Dict mapping column names to 1D arrays of length n_frames.
        Column names follow the pattern: {metric}_{radius}d
        e.g. "rms_gap_60d", "n_obs_30d", "gap_cv_90d"
    """
    if radii is None:
        radii = SPARSITY_RADII

    n = len(days)
    result = {}

    for R in radii:
        suffix = f"_{R}d"

        arr_n_obs = np.zeros(n, dtype=int)
        arr_obs_density = np.full(n, np.nan)
        arr_rms_gap = np.full(n, np.nan)
        arr_max_gap = np.full(n, np.nan)
        arr_gap_cv = np.full(n, np.nan)
        arr_gap_entropy = np.full(n, np.nan)

        window_len = 2.0 * R

        for i in range(n):
            gaps = _compute_gap_sequence(days, i, R)

            # --- n_obs: count of observations in window ---
            # gaps has (n_obs_in_window + 1) elements (n_obs internal
            # boundaries + 2 boundary gaps - 1 because k obs make k+1 gaps).
            # So n_obs = len(gaps) - 1.
            n_obs_i = len(gaps) - 1
            arr_n_obs[i] = n_obs_i
            arr_obs_density[i] = n_obs_i / window_len

            # --- rms_gap: √(mean(g²)) ---
            arr_rms_gap[i] = np.sqrt(np.mean(gaps ** 2))

            # --- max_gap ---
            arr_max_gap[i] = np.max(gaps)

            # --- gap_cv: std/mean ---
            # Defined when there are ≥ 2 gaps (i.e. ≥ 1 observation)
            if len(gaps) >= 2:
                g_mean = np.mean(gaps)
                if g_mean > 0:
                    arr_gap_cv[i] = np.std(gaps, ddof=0) / g_mean
                else:
                    arr_gap_cv[i] = 0.0
            # else: stays NaN (0 observations → single gap → CV undefined)

            # --- gap_entropy: normalised Shannon entropy ---
            # H = -Σ p_j ln(p_j),  p_j = g_j / window_len
            # Normalised: H_norm = H / ln(n_gaps) ∈ [0, 1]
            #   1.0 = all gaps equal (perfectly regular)
            #   → 0  = one gap dominates (maximally irregular)
            if len(gaps) >= 2:
                props = gaps / window_len
                # Filter out zero-length gaps (obs exactly on boundary)
                props_nz = props[props > 0]
                if len(props_nz) >= 2:
                    H = -np.sum(props_nz * np.log(props_nz))
                    H_max = np.log(len(props_nz))
                    arr_gap_entropy[i] = H / H_max if H_max > 0 else 1.0
                elif len(props_nz) == 1:
                    arr_gap_entropy[i] = 0.0  # one gap = no entropy
            # else: stays NaN

        result[f"n_obs{suffix}"] = arr_n_obs
        result[f"obs_density{suffix}"] = arr_obs_density
        result[f"rms_gap{suffix}"] = arr_rms_gap
        result[f"max_gap{suffix}"] = arr_max_gap
        result[f"gap_cv{suffix}"] = arr_gap_cv
        result[f"gap_entropy{suffix}"] = arr_gap_entropy

    return result


def sparsity_column_names(radii: Optional[List[int]] = None) -> List[str]:
    """Return ordered list of sparsity column names for CSV output."""
    if radii is None:
        radii = SPARSITY_RADII
    cols = []
    for R in radii:
        s = f"_{R}d"
        cols.extend([
            f"n_obs{s}", f"obs_density{s}", f"rms_gap{s}",
            f"max_gap{s}", f"gap_cv{s}", f"gap_entropy{s}",
        ])
    return cols


# ========== Process one lake ==========

def process_lake(
    lake_id: str,
    dineof_results_path: str,
    eofs_path: str,
    k: float = 4.0,
    output_dir: Optional[str] = None,
    merged_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Detect blowup frames in one lake's DINEOF reconstruction.

    Returns:
        (blowup_events, lake_summary)
    """
    print(f"\n[{lake_id}] Loading ...")

    # --- Load time from eofs.nc ---
    try:
        days, n_t_eofs = load_time_from_eofs(eofs_path)
    except Exception as e:
        msg = f"Failed to load time from eofs.nc: {e}"
        print(f"  ERROR: {msg}")
        return [], {"lake_id": lake_id, "error": msg}

    # --- Load reconstruction ---
    try:
        tf = load_reconstruction(dineof_results_path)
    except Exception as e:
        msg = f"Failed to load dineof_results.nc: {e}"
        print(f"  ERROR: {msg}")
        return [], {"lake_id": lake_id, "error": msg}

    n_t_recon = tf.shape[0]

    # --- CRITICAL CHECK: frame count must match exactly ---
    if n_t_recon != len(days):
        msg = (f"FRAME COUNT MISMATCH: dineof_results.nc has {n_t_recon} frames, "
               f"eofs.nc time has {len(days)} entries. Cannot proceed.")
        print(f"  ERROR: {msg}")
        return [], {"lake_id": lake_id, "error": msg}

    if n_t_eofs != len(days):
        msg = (f"EOFS DIM MISMATCH: eofs.nc t dim = {n_t_eofs}, "
               f"time array length = {len(days)}.")
        print(f"  ERROR: {msg}")
        return [], {"lake_id": lake_id, "error": msg}

    print(f"  Frames: {n_t_recon} (verified: dineof_results == eofs.nc time)")

    # --- Load lake mask ---
    lake_mask = load_lake_mask(eofs_path)
    n_lake_pixels = int(lake_mask.sum()) if lake_mask is not None else -1
    print(f"  Lake pixels: {n_lake_pixels}")

    # --- Compute frame stats ---
    stats = compute_frame_stats(tf, lake_mask=lake_mask)

    # --- Detect blowups ---
    flags = detect_blowups(stats, k=k)
    n_flagged = int(flags["combined"].sum())

    print(f"  Flagged frames: {n_flagged} "
          f"(range={int(flags['frame_range_flag'].sum())}, "
          f"std={int(flags['frame_std_flag'].sum())}, "
          f"jump={int(flags['mean_jump_flag'].sum())}, "
          f"mean={int(flags['frame_mean_flag'].sum())})")

    # --- Compute local sparsity metrics ---
    print(f"  Computing local sparsity (radii={SPARSITY_RADII}) ...")
    sparsity = compute_local_sparsity(days, radii=SPARSITY_RADII)
    sparsity_cols = sparsity_column_names(radii=SPARSITY_RADII)

    # --- Compute spatial observation metrics (obs_fraction, shore distance) ---
    obs_frac = None
    obs_shore_med = None
    shore_dist_map = None
    has_spatial_obs = False

    if merged_path is not None:
        print(f"  Loading spatial obs metrics from merged file ...")
        obs_frac_raw, obs_shore_raw, shore_dist_map, merged_days = load_obs_fraction_and_shore(
            merged_path, n_t_expected=n_t_recon, dineof_lake_mask=lake_mask)
        if obs_frac_raw is not None and merged_days is not None:
            # Align by date: DINEOF days is a subset of merged_days
            # Build lookup from merged day → index
            merged_day_to_idx = {int(d): i for i, d in enumerate(merged_days)}

            obs_frac = np.full(n_t_recon, np.nan)
            obs_shore_med = np.full(n_t_recon, np.nan)
            n_matched = 0

            for i, d in enumerate(days):
                d_int = int(d)
                if d_int in merged_day_to_idx:
                    j = merged_day_to_idx[d_int]
                    obs_frac[i] = obs_frac_raw[j]
                    obs_shore_med[i] = obs_shore_raw[j]
                    n_matched += 1

            if n_matched == 0:
                print(f"    WARNING: no dates matched between DINEOF ({len(days)}) and merged ({len(merged_days)})")
                print(f"    DINEOF days range: {int(days[0])}–{int(days[-1])}, "
                      f"merged days range: {int(merged_days[0])}–{int(merged_days[-1])}")
                has_spatial_obs = False
            else:
                has_spatial_obs = True
                n_unmatched = n_t_recon - n_matched
                if n_unmatched > 0:
                    print(f"    Matched {n_matched}/{n_t_recon} frames by date "
                          f"({n_unmatched} DINEOF frames not in merged file)")
                else:
                    print(f"    Matched all {n_matched} frames by date")
                n_zero = int(np.nansum(obs_frac == 0))
                print(f"    obs_fraction: median={np.nanmedian(obs_frac):.3f}, "
                      f"min={np.nanmin(obs_frac[np.isfinite(obs_frac)]):.3f}, "
                      f"frames_with_0_obs={n_zero}")
                if shore_dist_map is not None:
                    lake_shore_dists = shore_dist_map[shore_dist_map > 0]
                    if len(lake_shore_dists) > 0:
                        print(f"    shore_dist (lake pixels): median={np.median(lake_shore_dists):.1f}px, "
                              f"max={np.max(lake_shore_dists):.1f}px")
    else:
        print(f"  No merged file found — skipping obs_fraction / shore metrics")

    # Spatial obs column names
    spatial_obs_cols = ["obs_fraction", "obs_shore_dist_median"] if has_spatial_obs else []

    # --- Write per-lake frame_stats.csv (all frames) ---
    if output_dir is not None:
        lake_out = os.path.join(output_dir, "per_lake", lake_id)
        os.makedirs(lake_out, exist_ok=True)

        # Column order: identifiers, gaps, frame stats, flags, sparsity metrics
        base_header = [
            "frame_index", "date", "days_since_epoch",
            "gap_before", "gap_after",
            "frame_mean", "frame_std", "frame_min", "frame_max",
            "frame_range", "frame_n_valid", "mean_jump",
            "flag_range", "flag_std", "flag_jump", "flag_mean", "flag_any",
        ]
        full_header = base_header + sparsity_cols + spatial_obs_cols

        frame_stats_path = os.path.join(lake_out, f"frame_stats_{lake_id}.csv")
        with open(frame_stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(full_header)
            for i in range(n_t_recon):
                gb, ga = compute_gap_at_index(days, i)

                # Base columns
                row = [
                    i,
                    day_to_datestr(int(days[i])),
                    int(days[i]),
                    gb if gb > 0 else "",
                    ga if ga > 0 else "",
                    f"{stats['frame_mean'][i]:.6f}" if np.isfinite(stats['frame_mean'][i]) else "",
                    f"{stats['frame_std'][i]:.6f}" if np.isfinite(stats['frame_std'][i]) else "",
                    f"{stats['frame_min'][i]:.6f}" if np.isfinite(stats['frame_min'][i]) else "",
                    f"{stats['frame_max'][i]:.6f}" if np.isfinite(stats['frame_max'][i]) else "",
                    f"{stats['frame_range'][i]:.6f}" if np.isfinite(stats['frame_range'][i]) else "",
                    int(stats['frame_n_valid'][i]),
                    f"{stats['mean_jump'][i]:.6f}" if np.isfinite(stats['mean_jump'][i]) else "",
                    int(flags['frame_range_flag'][i]),
                    int(flags['frame_std_flag'][i]),
                    int(flags['mean_jump_flag'][i]),
                    int(flags['frame_mean_flag'][i]),
                    int(flags['combined'][i]),
                ]

                # Sparsity columns
                for col in sparsity_cols:
                    val = sparsity[col][i]
                    if np.isfinite(val):
                        if isinstance(val, (int, np.integer)):
                            row.append(int(val))
                        else:
                            row.append(f"{val:.4f}")
                    else:
                        row.append("")

                # Spatial observation columns
                if has_spatial_obs:
                    # obs_fraction
                    if np.isfinite(obs_frac[i]):
                        row.append(f"{obs_frac[i]:.4f}")
                    else:
                        row.append("")
                    # obs_shore_dist_median
                    if np.isfinite(obs_shore_med[i]):
                        row.append(f"{obs_shore_med[i]:.2f}")
                    else:
                        row.append("")

                writer.writerow(row)
        print(f"  Wrote {frame_stats_path}")

    # --- Build blowup event records ---
    blowup_events = []
    flagged_indices = np.flatnonzero(flags["combined"])

    for idx in flagged_indices:
        gb, ga = compute_gap_at_index(days, idx)

        event = {
            "lake_id": lake_id,
            "frame_index": int(idx),
            "date": day_to_datestr(int(days[idx])),
            "days_since_epoch": int(days[idx]),
            "gap_before": gb,
            "gap_after": ga,
            "gap_max_either_side": max(gb if gb > 0 else 0, ga if ga > 0 else 0),
            "frame_mean": round(float(stats["frame_mean"][idx]), 6) if np.isfinite(stats["frame_mean"][idx]) else "",
            "frame_std": round(float(stats["frame_std"][idx]), 6) if np.isfinite(stats["frame_std"][idx]) else "",
            "frame_min": round(float(stats["frame_min"][idx]), 6) if np.isfinite(stats["frame_min"][idx]) else "",
            "frame_max": round(float(stats["frame_max"][idx]), 6) if np.isfinite(stats["frame_max"][idx]) else "",
            "frame_range": round(float(stats["frame_range"][idx]), 6) if np.isfinite(stats["frame_range"][idx]) else "",
            "frame_n_valid": int(stats["frame_n_valid"][idx]),
            "mean_jump": round(float(stats["mean_jump"][idx]), 6) if np.isfinite(stats["mean_jump"][idx]) else "",
            "flag_range": int(flags["frame_range_flag"][idx]),
            "flag_jump": int(flags["mean_jump_flag"][idx]),
            "flag_std": int(flags["frame_std_flag"][idx]),
            "flag_mean": int(flags["frame_mean_flag"][idx]),
            "n_lake_pixels": n_lake_pixels,
        }
        # Add spatial obs metrics if available
        if has_spatial_obs:
            event["obs_fraction"] = round(float(obs_frac[idx]), 4) if np.isfinite(obs_frac[idx]) else ""
            event["obs_shore_dist_median"] = round(float(obs_shore_med[idx]), 2) if np.isfinite(obs_shore_med[idx]) else ""
        blowup_events.append(event)

    # --- Write per-lake blowup events ---
    if output_dir is not None and blowup_events:
        lake_events_path = os.path.join(output_dir, "per_lake", lake_id, f"blowup_events_{lake_id}.csv")
        _write_events_csv(blowup_events, lake_events_path)

    # --- Lake summary ---
    # Robust stats on frame_mean for context
    valid_means = stats["frame_mean"][np.isfinite(stats["frame_mean"])]
    valid_stds = stats["frame_std"][np.isfinite(stats["frame_std"])]
    valid_jumps = stats["mean_jump"][np.isfinite(stats["mean_jump"])]
    valid_ranges = stats["frame_range"][np.isfinite(stats["frame_range"])]

    lake_summary = {
        "lake_id": lake_id,
        "n_frames": n_t_recon,
        "n_lake_pixels": n_lake_pixels,
        "n_blowup_frames": n_flagged,
        "n_flag_range": int(flags["frame_range_flag"].sum()),
        "n_flag_std": int(flags["frame_std_flag"].sum()),
        "n_flag_jump": int(flags["mean_jump_flag"].sum()),
        "n_flag_mean": int(flags["frame_mean_flag"].sum()),
        "fraction_flagged": round(n_flagged / n_t_recon, 6) if n_t_recon > 0 else 0,
        "has_blowups": n_flagged > 0,
        # Frame range stats (primary detection metric)
        "frame_range_median": round(float(np.median(valid_ranges)), 4) if len(valid_ranges) > 0 else "",
        "frame_range_rsd": round(float(1.4826 * np.median(np.abs(valid_ranges - np.median(valid_ranges)))), 4) if len(valid_ranges) > 0 else "",
        "frame_range_threshold": round(float(np.median(valid_ranges) + k * 1.4826 * np.median(np.abs(valid_ranges - np.median(valid_ranges)))), 4) if len(valid_ranges) > 0 else "",
        # Global frame stats
        "frame_mean_median": round(float(np.median(valid_means)), 4) if len(valid_means) > 0 else "",
        "frame_mean_rsd": round(float(1.4826 * np.median(np.abs(valid_means - np.median(valid_means)))), 4) if len(valid_means) > 0 else "",
        "frame_std_median": round(float(np.median(valid_stds)), 4) if len(valid_stds) > 0 else "",
        "frame_std_p99": round(float(np.percentile(valid_stds, 99)), 4) if len(valid_stds) > 0 else "",
        "frame_std_max": round(float(np.max(valid_stds)), 4) if len(valid_stds) > 0 else "",
        "mean_jump_median": round(float(np.median(valid_jumps)), 4) if len(valid_jumps) > 0 else "",
        "mean_jump_p99": round(float(np.percentile(valid_jumps, 99)), 4) if len(valid_jumps) > 0 else "",
        "mean_jump_max": round(float(np.max(valid_jumps)), 4) if len(valid_jumps) > 0 else "",
    }

    # Worst blowup info
    if blowup_events:
        # Worst by frame_range
        worst = max(blowup_events, key=lambda e: float(e["frame_range"]) if e["frame_range"] != "" else 0)
        lake_summary["worst_blowup_date"] = worst["date"]
        lake_summary["worst_blowup_range"] = worst["frame_range"]
        lake_summary["worst_blowup_gap_max"] = worst["gap_max_either_side"]

        # Severity ratio: worst range / median range
        if lake_summary["frame_range_median"] and float(lake_summary["frame_range_median"]) > 0:
            lake_summary["severity_ratio"] = round(
                float(worst["frame_range"]) / float(lake_summary["frame_range_median"]), 2)
        else:
            lake_summary["severity_ratio"] = ""

        # Gap stats at blowup locations
        blowup_gaps = []
        for e in blowup_events:
            if e["gap_before"] > 0:
                blowup_gaps.append(e["gap_before"])
            if e["gap_after"] > 0:
                blowup_gaps.append(e["gap_after"])
        if blowup_gaps:
            lake_summary["blowup_gap_mean"] = round(float(np.mean(blowup_gaps)), 2)
            lake_summary["blowup_gap_median"] = round(float(np.median(blowup_gaps)), 2)
            lake_summary["blowup_gap_max"] = int(np.max(blowup_gaps))
    else:
        lake_summary["worst_blowup_date"] = ""
        lake_summary["worst_blowup_range"] = ""
        lake_summary["worst_blowup_gap_max"] = ""
        lake_summary["severity_ratio"] = ""
        lake_summary["blowup_gap_mean"] = ""
        lake_summary["blowup_gap_median"] = ""
        lake_summary["blowup_gap_max"] = ""

    if n_flagged > 0:
        print(f"  Worst blowup: {lake_summary['worst_blowup_date']}, "
              f"range={lake_summary['worst_blowup_range']}, "
              f"gap_max={lake_summary['worst_blowup_gap_max']}")

    # --- Sparsity summary: lake-wide and at blowup frames ---
    # For each radius, report lake-wide median rms_gap and the rms_gap
    # at the worst blowup frame. This goes into the per-lake summary.
    for R in SPARSITY_RADII:
        s = f"_{R}d"
        rms_arr = sparsity[f"rms_gap{s}"]
        valid_rms = rms_arr[np.isfinite(rms_arr)]

        lake_summary[f"rms_gap_median{s}"] = (
            round(float(np.median(valid_rms)), 4) if len(valid_rms) > 0 else "")
        lake_summary[f"rms_gap_p99{s}"] = (
            round(float(np.percentile(valid_rms, 99)), 4) if len(valid_rms) > 0 else "")
        lake_summary[f"rms_gap_max{s}"] = (
            round(float(np.max(valid_rms)), 4) if len(valid_rms) > 0 else "")

        # Sparsity at worst blowup frame (if blowups exist)
        if blowup_events:
            worst_idx = int(worst["frame_index"])
            lake_summary[f"worst_blowup_rms_gap{s}"] = (
                round(float(rms_arr[worst_idx]), 4) if np.isfinite(rms_arr[worst_idx]) else "")
        else:
            lake_summary[f"worst_blowup_rms_gap{s}"] = ""

        # Median rms_gap at flagged vs unflagged frames
        flagged_mask = flags["combined"].astype(bool)
        if flagged_mask.sum() > 0:
            lake_summary[f"rms_gap_flagged_median{s}"] = (
                round(float(np.nanmedian(rms_arr[flagged_mask])), 4))
        else:
            lake_summary[f"rms_gap_flagged_median{s}"] = ""

        unflagged_mask = ~flagged_mask
        if unflagged_mask.sum() > 0:
            lake_summary[f"rms_gap_unflagged_median{s}"] = (
                round(float(np.nanmedian(rms_arr[unflagged_mask])), 4))
        else:
            lake_summary[f"rms_gap_unflagged_median{s}"] = ""

    # --- Spatial observation summary stats ---
    if has_spatial_obs:
        lake_summary["obs_fraction_median"] = round(float(np.nanmedian(obs_frac)), 4)
        lake_summary["obs_fraction_min"] = round(float(np.nanmin(obs_frac)), 4)
        lake_summary["obs_fraction_p10"] = round(float(np.nanpercentile(obs_frac, 10)), 4)

        if flagged_mask.sum() > 0:
            lake_summary["obs_fraction_flagged_median"] = (
                round(float(np.nanmedian(obs_frac[flagged_mask])), 4))
        else:
            lake_summary["obs_fraction_flagged_median"] = ""

        if unflagged_mask.sum() > 0:
            lake_summary["obs_fraction_unflagged_median"] = (
                round(float(np.nanmedian(obs_frac[unflagged_mask])), 4))
        else:
            lake_summary["obs_fraction_unflagged_median"] = ""

        # Shore distance summary (lake-level)
        if shore_dist_map is not None:
            lake_pixels_dist = shore_dist_map[shore_dist_map > 0]
            if len(lake_pixels_dist) > 0:
                lake_summary["shore_dist_median_px"] = round(float(np.median(lake_pixels_dist)), 2)
                lake_summary["shore_dist_max_px"] = round(float(np.max(lake_pixels_dist)), 2)
            else:
                lake_summary["shore_dist_median_px"] = ""
                lake_summary["shore_dist_max_px"] = ""
        else:
            lake_summary["shore_dist_median_px"] = ""
            lake_summary["shore_dist_max_px"] = ""

        # Obs shore distance at flagged frames
        if flagged_mask.sum() > 0:
            valid_shore = obs_shore_med[flagged_mask & np.isfinite(obs_shore_med)]
            lake_summary["obs_shore_dist_flagged_median"] = (
                round(float(np.median(valid_shore)), 2) if len(valid_shore) > 0 else "")
        else:
            lake_summary["obs_shore_dist_flagged_median"] = ""
    else:
        for col in ["obs_fraction_median", "obs_fraction_min", "obs_fraction_p10",
                     "obs_fraction_flagged_median", "obs_fraction_unflagged_median",
                     "shore_dist_median_px", "shore_dist_max_px",
                     "obs_shore_dist_flagged_median"]:
            lake_summary[col] = ""

    # --- Write per-lake JSON verdict (for SLURM parallel assembly) ---
    if output_dir is not None:
        import json
        lake_out = os.path.join(output_dir, "per_lake", lake_id)
        os.makedirs(lake_out, exist_ok=True)
        verdict_path = os.path.join(lake_out, f"verdict_{lake_id}.json")
        with open(verdict_path, "w") as f:
            json.dump(lake_summary, f, indent=2, default=str)
        print(f"  Wrote {verdict_path}")

    return blowup_events, lake_summary


# ========== CSV output ==========

EVENT_COLUMNS = [
    "lake_id", "frame_index", "date", "days_since_epoch",
    "gap_before", "gap_after", "gap_max_either_side",
    "frame_mean", "frame_std", "frame_min", "frame_max",
    "frame_range", "frame_n_valid", "mean_jump",
    "flag_range", "flag_jump", "flag_std", "flag_mean", "n_lake_pixels",
    "obs_fraction", "obs_shore_dist_median",
]

SUMMARY_COLUMNS = [
    "lake_id", "n_frames", "n_lake_pixels",
    "n_blowup_frames", "n_flag_range", "n_flag_std", "n_flag_jump", "n_flag_mean",
    "fraction_flagged", "has_blowups",
    "frame_range_median", "frame_range_rsd", "frame_range_threshold",
    "frame_mean_median", "frame_mean_rsd",
    "frame_std_median", "frame_std_p99", "frame_std_max",
    "mean_jump_median", "mean_jump_p99", "mean_jump_max",
    "worst_blowup_date", "worst_blowup_range", "worst_blowup_gap_max", "severity_ratio",
    "blowup_gap_mean", "blowup_gap_median", "blowup_gap_max",
]

# Append sparsity summary columns for each radius
for _R in SPARSITY_RADII:
    _s = f"_{_R}d"
    SUMMARY_COLUMNS.extend([
        f"rms_gap_median{_s}", f"rms_gap_p99{_s}", f"rms_gap_max{_s}",
        f"worst_blowup_rms_gap{_s}",
        f"rms_gap_flagged_median{_s}", f"rms_gap_unflagged_median{_s}",
    ])

# Spatial observation summary columns
SUMMARY_COLUMNS.extend([
    "obs_fraction_median", "obs_fraction_min", "obs_fraction_p10",
    "obs_fraction_flagged_median", "obs_fraction_unflagged_median",
    "shore_dist_median_px", "shore_dist_max_px",
    "obs_shore_dist_flagged_median",
])


def _write_events_csv(events: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    print(f"  Wrote {path} ({len(events)} events)")


def _write_summary_csv(summaries: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    summaries.sort(key=lambda x: x.get("lake_id", ""))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nWrote summary: {path} ({len(summaries)} lakes)")


# ========== Lake discovery ==========

def discover_lakes(exp_dir: str) -> List[Tuple[str, str, str, Optional[str]]]:
    """
    Find all lakes with both dineof_results.nc and eofs.nc.

    Returns:
        List of (lake_id, dineof_results_path, eofs_path, merged_path).
        merged_path is None if the *filled_fine_dineof.nc file is not found.
    """
    results = []
    dineof_dir = os.path.join(exp_dir, "dineof")

    if not os.path.isdir(dineof_dir):
        print(f"ERROR: {dineof_dir} does not exist")
        sys.exit(1)

    for lake_id in sorted(os.listdir(dineof_dir)):
        lake_dineof = os.path.join(dineof_dir, lake_id)
        if not os.path.isdir(lake_dineof):
            continue

        # Find alpha directories
        alpha_dirs = sorted(glob.glob(os.path.join(lake_dineof, "a*")))
        for adir in alpha_dirs:
            results_path = os.path.join(adir, "dineof_results.nc")
            eofs_path = os.path.join(adir, "eofs.nc")

            if os.path.isfile(results_path) and os.path.isfile(eofs_path):
                merged_path = find_merged_file(adir)
                results.append((lake_id, results_path, eofs_path, merged_path))
                break  # use first alpha dir found

    return results


# ========== CLI ==========

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LSWT Reconstruction Blowup Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detection methods (all in anomaly space from dineof_results.nc):
  1. Frame-to-frame jump in lake-mean (flag_jump)
  2. Within-frame pixel spread / std (flag_std)
  3. Absolute lake-mean outlier (flag_mean)

All use MAD-based robust thresholding with configurable k.

Examples:
  python blowup_detector.py --exp-dir /path/to/exp
  python blowup_detector.py --exp-dir /path/to/exp --lake-id 000000044
  python blowup_detector.py --exp-dir /path/to/exp --k 3.5
"""
    )
    p.add_argument("--exp-dir", required=True, help="Experiment root directory")
    p.add_argument("--lake-id", default=None, help="Process single lake")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: {exp-dir}/blowup_diagnostics/)")
    p.add_argument("--k", type=float, default=4.0,
                   help="Detection threshold in robust SDs (default: 4.0)")
    p.add_argument("--assemble-only", action="store_true",
                   help="Skip processing, just assemble per-lake verdict.json files into summary CSV")
    return p


def assemble_verdicts(output_dir: str) -> None:
    """Collect per-lake verdict.json files into a single summary CSV."""
    import json
    per_lake_dir = os.path.join(output_dir, "per_lake")
    if not os.path.isdir(per_lake_dir):
        print(f"ERROR: {per_lake_dir} does not exist")
        sys.exit(1)

    summaries = []
    for lake_id in sorted(os.listdir(per_lake_dir)):
        lake_dir = os.path.join(per_lake_dir, lake_id)
        if not os.path.isdir(lake_dir):
            continue
        # Try suffixed name first, fall back to old name for backward compatibility
        verdict_path = os.path.join(lake_dir, f"verdict_{lake_id}.json")
        if not os.path.isfile(verdict_path):
            verdict_path = os.path.join(lake_dir, "verdict.json")
        if os.path.isfile(verdict_path):
            with open(verdict_path) as f:
                summaries.append(json.load(f))

    if not summaries:
        print("No verdict JSON files found (looked for verdict_{lake_id}.json and verdict.json)")
        sys.exit(1)

    _write_summary_csv(summaries, os.path.join(output_dir, "blowup_summary_all_lakes.csv"))

    n_with = sum(1 for s in summaries if s.get("has_blowups"))
    print(f"\n{'='*60}")
    print(f"Assembled {len(summaries)} lakes. {n_with} have blowups.")

    if n_with > 0:
        print(f"\nLakes with blowups (sorted by severity_ratio):")
        for s in sorted(summaries,
                        key=lambda x: float(x.get("severity_ratio", 0)) if x.get("severity_ratio", "") != "" else 0,
                        reverse=True):
            if s.get("has_blowups"):
                print(f"  {s['lake_id']}: {s['n_blowup_frames']} frames, "
                      f"severity_ratio={s.get('severity_ratio', '?')}, "
                      f"worst={s.get('worst_blowup_date', '?')}, "
                      f"blowup_gap_max={s.get('blowup_gap_max', '?')}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    exp_dir = args.exp_dir.rstrip("/")
    output_dir = args.output_dir or os.path.join(exp_dir, "blowup_diagnostics")

    if args.assemble_only:
        print(f"Assembling verdicts from: {output_dir}")
        assemble_verdicts(output_dir)
        return

    print(f"Experiment dir: {exp_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Detection k:    {args.k}")

    # Discover lakes
    if args.lake_id:
        alpha_dirs = sorted(glob.glob(os.path.join(exp_dir, "dineof", args.lake_id, "a*")))
        found = False
        for adir in alpha_dirs:
            rp = os.path.join(adir, "dineof_results.nc")
            ep = os.path.join(adir, "eofs.nc")
            if os.path.isfile(rp) and os.path.isfile(ep):
                mp = find_merged_file(adir)
                lakes = [(args.lake_id, rp, ep, mp)]
                found = True
                break
        if not found:
            print(f"ERROR: No dineof_results.nc + eofs.nc found for lake {args.lake_id}")
            sys.exit(1)
    else:
        lakes = discover_lakes(exp_dir)

    print(f"Found {len(lakes)} lake(s)\n")

    all_events = []
    all_summaries = []

    for lake_id, results_path, eofs_path, merged_path in lakes:
        events, summary = process_lake(
            lake_id=lake_id,
            dineof_results_path=results_path,
            eofs_path=eofs_path,
            k=args.k,
            output_dir=output_dir,
            merged_path=merged_path,
        )
        all_events.extend(events)
        all_summaries.append(summary)

    # Write global outputs
    if all_events:
        _write_events_csv(all_events, os.path.join(output_dir, "blowup_events_all_lakes.csv"))

    if all_summaries:
        _write_summary_csv(all_summaries, os.path.join(output_dir, "blowup_summary_all_lakes.csv"))

    # Final report
    n_with = sum(1 for s in all_summaries if s.get("has_blowups"))
    n_total = len(all_summaries)
    print(f"\n{'='*60}")
    print(f"Done. {n_with}/{n_total} lakes have blowups ({len(all_events)} total events)")

    if n_with > 0:
        print(f"\nLakes with blowups (sorted by severity_ratio):")
        for s in sorted(all_summaries,
                        key=lambda x: float(x.get("severity_ratio", 0)) if x.get("severity_ratio", "") != "" else 0,
                        reverse=True):
            if s.get("has_blowups"):
                print(f"  {s['lake_id']}: {s['n_blowup_frames']} frames, "
                      f"severity_ratio={s.get('severity_ratio', '?')}, "
                      f"worst={s.get('worst_blowup_date', '?')}, "
                      f"blowup_gap_max={s.get('blowup_gap_max', '?')}")


if __name__ == "__main__":
    main()
