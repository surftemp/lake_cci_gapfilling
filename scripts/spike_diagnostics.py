#!/usr/bin/env python3
"""
Temporal EOF Spike Diagnostics

Auto-detect lakes with large temporal EOF spikes, and for each spike extract:
  - Which EOF mode(s) spiked
  - Spike magnitude (absolute value, ratio to neighbors, ratio to median)
  - Date of the spike
  - Temporal gap before and after the spike date
  - Spatial coverage fraction on the spike date (optional, requires loading prepared.nc)
  - Lake size (total lake pixels from lakeid mask)

Outputs:
  spike_diagnostics/
  ├── spike_events_all_lakes.csv    — one row per spike event across all lakes
  ├── spike_summary_all_lakes.csv   — one row per lake (has spikes / no spikes)
  └── per_lake/{lake_id}/
      └── spike_events.csv          — per-spike detail for this lake

Usage:
  # All lakes, fast mode (no spatial coverage — avoids reading multi-GB prepared.nc)
  python spike_diagnostics.py --exp-dir /path/to/exp

  # All lakes with spatial coverage (slower, reads prepared.nc per lake)
  python spike_diagnostics.py --exp-dir /path/to/exp --with-spatial-coverage

  # Single lake
  python spike_diagnostics.py --exp-dir /path/to/exp --lake-id 000000020

  # Custom detection threshold (default k=4, same as EOF filter step)
  python spike_diagnostics.py --exp-dir /path/to/exp --k 3.5

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
TEMPORAL_EOF_PREFIX = "temporal_eof"


def day_to_datestr(d: int) -> str:
    dt = EPOCH + np.timedelta64(d, "D")
    return str(dt)[:10]


# ========== Spike detection ==========

def detect_spikes_robust_sd(values: np.ndarray, k: float = 4.0) -> np.ndarray:
    """
    Detect spikes using robust standard deviation (same method as FilterTemporalEOFsStep).

    flagged = |x - median(x)| > k * 1.4826 * MAD(x)

    Args:
        values: 1D array of temporal EOF values.
        k: Threshold multiplier (default 4.0).

    Returns:
        Boolean mask, True = spike.
    """
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    rsd = 1.4826 * mad
    if not np.isfinite(rsd) or rsd == 0:
        return np.zeros(len(values), dtype=bool)
    return np.abs(values - med) > (k * rsd)


def get_spike_context(
    values: np.ndarray,
    spike_idx: int,
    window: int = 3
) -> Dict[str, float]:
    """
    Compute spike magnitude metrics relative to its local neighborhood.

    Args:
        values: Full temporal EOF array.
        spike_idx: Index of the spike.
        window: Number of neighbors on each side to consider.

    Returns:
        Dict with magnitude metrics.
    """
    n = len(values)
    spike_val = float(values[spike_idx])

    # Gather neighbor values (excluding the spike itself)
    lo = max(0, spike_idx - window)
    hi = min(n, spike_idx + window + 1)
    neighbors = np.concatenate([values[lo:spike_idx], values[spike_idx+1:hi]])
    neighbors = neighbors[np.isfinite(neighbors)]

    if len(neighbors) == 0:
        return {
            "spike_value": spike_val,
            "neighbor_mean": np.nan,
            "neighbor_median": np.nan,
            "abs_deviation_from_neighbor_mean": np.nan,
            "ratio_to_neighbor_abs_mean": np.nan,
        }

    neighbor_mean = float(np.mean(neighbors))
    neighbor_median = float(np.median(neighbors))
    neighbor_abs_mean = float(np.mean(np.abs(neighbors)))
    abs_dev = abs(spike_val - neighbor_mean)

    ratio = abs(spike_val) / neighbor_abs_mean if neighbor_abs_mean > 1e-12 else np.nan

    return {
        "spike_value": round(spike_val, 8),
        "neighbor_mean": round(neighbor_mean, 8),
        "neighbor_median": round(neighbor_median, 8),
        "abs_deviation_from_neighbor_mean": round(abs_dev, 8),
        "ratio_to_neighbor_abs_mean": round(ratio, 3),
    }


# ========== Time / gap helpers ==========

def load_time_days(prepared_nc_path: str) -> np.ndarray:
    """Load time as int64 days since epoch from prepared.nc (fast — only reads time var)."""
    with xr.open_dataset(prepared_nc_path) as ds:
        time_vals = ds["time"].values
        if np.issubdtype(time_vals.dtype, np.datetime64):
            base = np.datetime64("1981-01-01T12:00:00", "ns")
            days = ((time_vals.astype("datetime64[ns]") - base) / np.timedelta64(1, "D")).astype("int64")
        elif np.issubdtype(time_vals.dtype, np.integer):
            days = time_vals.astype("int64")
        else:
            raise ValueError(f"Unexpected time dtype: {time_vals.dtype}")
    return days


def load_time_from_eofs(eofs_ds: xr.Dataset) -> Optional[np.ndarray]:
    """Try to extract physical time from eofs.nc."""
    if "time" in eofs_ds.coords:
        vals = eofs_ds["time"].values
        if np.issubdtype(vals.dtype, np.datetime64):
            base = np.datetime64("1981-01-01T12:00:00", "ns")
            return ((vals.astype("datetime64[ns]") - base) / np.timedelta64(1, "D")).astype("int64")
        elif np.issubdtype(vals.dtype, np.integer) or np.issubdtype(vals.dtype, np.floating):
            return vals.astype("int64")
    return None


def compute_gap_at_index(days: np.ndarray, idx: int) -> Tuple[int, int]:
    """
    Return (gap_before, gap_after) for a given index in the days array.
    Returns -1 for unavailable (first/last element).
    """
    gap_before = int(days[idx] - days[idx - 1]) if idx > 0 else -1
    gap_after = int(days[idx + 1] - days[idx]) if idx < len(days) - 1 else -1
    return gap_before, gap_after


# ========== Spatial coverage (optional, slow) ==========

def compute_per_frame_coverage(prepared_nc_path: str, var_name: str = "lake_surface_water_temperature") -> np.ndarray:
    """
    Compute fraction of valid (non-NaN) lake pixels per timestep.

    Uses chunked reading to avoid loading the full cube into memory.

    Returns:
        1D array of shape (n_time,) with coverage fractions in [0, 1].
    """
    ds = xr.open_dataset(prepared_nc_path, chunks={"time": 100})

    # Get total lake pixels from lakeid mask
    if "lakeid" in ds:
        lakeid = ds["lakeid"].values
        n_lake_pixels = int(np.sum(np.isfinite(lakeid) & (lakeid != 0)))
    else:
        # Fallback: use first non-all-NaN frame
        n_lake_pixels = None

    lswt = ds[var_name]  # (time, lat, lon) — dask-backed
    n_time = lswt.shape[0]

    # Count valid pixels per frame
    coverage = np.zeros(n_time, dtype="float32")
    chunk_size = 100
    for t0 in range(0, n_time, chunk_size):
        t1 = min(t0 + chunk_size, n_time)
        chunk = lswt.isel(time=slice(t0, t1)).values  # (chunk, lat, lon)
        valid_per_frame = np.sum(np.isfinite(chunk), axis=(1, 2))

        if n_lake_pixels is not None and n_lake_pixels > 0:
            coverage[t0:t1] = valid_per_frame / n_lake_pixels
        else:
            # Use max observed as denominator
            coverage[t0:t1] = valid_per_frame / max(1, np.max(valid_per_frame))

    ds.close()
    return coverage


# ========== Lake size helpers ==========

def get_lake_n_pixels(prepared_nc_path: str) -> int:
    """Get total number of lake pixels from lakeid mask (fast — only reads 2D var)."""
    with xr.open_dataset(prepared_nc_path) as ds:
        if "lakeid" in ds:
            lakeid = ds["lakeid"].values
            return int(np.sum(np.isfinite(lakeid) & (lakeid != 0)))
    return -1


# ========== Process one lake ==========

def process_lake(
    lake_id: str,
    eofs_nc_path: str,
    prepared_nc_path: str,
    k: float = 4.0,
    with_spatial_coverage: bool = False,
    neighbor_window: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Detect spikes in one lake's temporal EOFs and build diagnostic records.

    Returns:
        (spike_events, lake_summary)
        spike_events: list of dicts, one per spike occurrence
        lake_summary: dict with lake-level stats
    """
    print(f"\n[{lake_id}] Loading eofs.nc ...")
    try:
        eofs_ds = xr.open_dataset(eofs_nc_path)
    except Exception as e:
        print(f"  ERROR opening {eofs_nc_path}: {e}")
        return [], {"lake_id": lake_id, "error": str(e)}

    # Get temporal EOF variable names
    temporal_vars = sorted(
        [v for v in eofs_ds.data_vars if v.startswith(TEMPORAL_EOF_PREFIX) and "t" in eofs_ds[v].dims],
        key=lambda x: int(x.replace(TEMPORAL_EOF_PREFIX, ""))
    )

    if not temporal_vars:
        print(f"  No temporal EOFs found")
        eofs_ds.close()
        return [], {"lake_id": lake_id, "n_temporal_eofs": 0, "n_spikes": 0}

    n_t = eofs_ds.dims["t"]
    print(f"  {len(temporal_vars)} temporal EOFs, {n_t} timesteps")

    # Get eigenvalues for variance explained
    eigenvalues = None
    if "eigenvalues" in eofs_ds:
        eigenvalues = eofs_ds["eigenvalues"].values
        total_var = np.sum(eigenvalues)

    # Get physical time
    days = load_time_from_eofs(eofs_ds)
    if days is None:
        print(f"  No time coord in eofs.nc, trying prepared.nc ...")
        try:
            days = load_time_days(prepared_nc_path)
        except Exception as e:
            print(f"  ERROR loading time from prepared.nc: {e}")
            eofs_ds.close()
            return [], {"lake_id": lake_id, "error": f"no time: {e}"}

    if len(days) != n_t:
        print(f"  WARNING: time length {len(days)} != eofs t dim {n_t}")
        eofs_ds.close()
        return [], {"lake_id": lake_id, "error": f"time mismatch: {len(days)} vs {n_t}"}

    # Get lake size
    n_lake_pixels = get_lake_n_pixels(prepared_nc_path)

    # Optional: per-frame spatial coverage
    frame_coverage = None
    if with_spatial_coverage:
        print(f"  Computing per-frame spatial coverage (this may be slow) ...")
        try:
            frame_coverage = compute_per_frame_coverage(prepared_nc_path)
        except Exception as e:
            print(f"  WARNING: Failed to compute spatial coverage: {e}")

    # Detect spikes per EOF mode
    spike_events = []
    per_mode_spike_count = {}
    combined_flagged = np.zeros(n_t, dtype=bool)

    for v in temporal_vars:
        vals = eofs_ds[v].values
        mode_idx = int(v.replace(TEMPORAL_EOF_PREFIX, ""))
        flagged = detect_spikes_robust_sd(vals, k=k)
        per_mode_spike_count[v] = int(flagged.sum())
        combined_flagged |= flagged

        # Variance explained by this mode
        var_explained = float(eigenvalues[mode_idx] / total_var) if eigenvalues is not None and mode_idx < len(eigenvalues) else np.nan

        # Global stats for this mode (for context)
        mode_median = float(np.nanmedian(vals))
        mode_mad = float(np.nanmedian(np.abs(vals - mode_median)))
        mode_rsd = 1.4826 * mode_mad

        for idx in np.flatnonzero(flagged):
            gap_before, gap_after = compute_gap_at_index(days, idx)
            ctx = get_spike_context(vals, idx, window=neighbor_window)

            event = {
                "lake_id": lake_id,
                "eof_mode": mode_idx,
                "eof_var_name": v,
                "variance_explained": round(var_explained, 6),
                "timestep_index": int(idx),
                "date": day_to_datestr(int(days[idx])),
                "days_since_epoch": int(days[idx]),
                "gap_before": gap_before,
                "gap_after": gap_after,
                "gap_max_either_side": max(gap_before if gap_before > 0 else 0,
                                           gap_after if gap_after > 0 else 0),
                "gap_sum_both_sides": (gap_before if gap_before > 0 else 0) +
                                      (gap_after if gap_after > 0 else 0),
                "spike_value": ctx["spike_value"],
                "neighbor_mean": ctx["neighbor_mean"],
                "neighbor_median": ctx["neighbor_median"],
                "abs_deviation_from_neighbor_mean": ctx["abs_deviation_from_neighbor_mean"],
                "ratio_to_neighbor_abs_mean": ctx["ratio_to_neighbor_abs_mean"],
                "mode_median": round(mode_median, 8),
                "mode_rsd": round(mode_rsd, 8),
                "n_rsd_from_median": round(abs(float(vals[idx]) - mode_median) / mode_rsd, 2) if mode_rsd > 0 else np.nan,
                "n_lake_pixels": n_lake_pixels,
            }

            # Spatial coverage on spike date (if available)
            if frame_coverage is not None and idx < len(frame_coverage):
                event["spatial_coverage_fraction"] = round(float(frame_coverage[idx]), 4)
            else:
                event["spatial_coverage_fraction"] = ""

            spike_events.append(event)

    n_total_spikes = len(spike_events)
    n_unique_dates = len(set(e["timestep_index"] for e in spike_events))

    # Lake-level summary
    lake_summary = {
        "lake_id": lake_id,
        "n_temporal_eofs": len(temporal_vars),
        "n_timesteps": n_t,
        "n_lake_pixels": n_lake_pixels,
        "n_spike_events": n_total_spikes,
        "n_unique_spike_dates": n_unique_dates,
        "fraction_dates_spiked": round(n_unique_dates / n_t, 6) if n_t > 0 else 0,
        "has_spikes": n_total_spikes > 0,
    }

    # Per-mode spike counts
    for v in temporal_vars:
        mode_idx = int(v.replace(TEMPORAL_EOF_PREFIX, ""))
        lake_summary[f"n_spikes_mode{mode_idx}"] = per_mode_spike_count[v]

    # Gap stats at spike locations (if any spikes)
    if spike_events:
        gap_befores = [e["gap_before"] for e in spike_events if e["gap_before"] > 0]
        gap_afters = [e["gap_after"] for e in spike_events if e["gap_after"] > 0]
        all_spike_gaps = gap_befores + gap_afters

        if all_spike_gaps:
            lake_summary["spike_gap_mean"] = round(float(np.mean(all_spike_gaps)), 2)
            lake_summary["spike_gap_median"] = round(float(np.median(all_spike_gaps)), 2)
            lake_summary["spike_gap_max"] = int(np.max(all_spike_gaps))
        else:
            lake_summary["spike_gap_mean"] = np.nan
            lake_summary["spike_gap_median"] = np.nan
            lake_summary["spike_gap_max"] = -1

        # Worst spike magnitude across all modes
        max_ratio = max((e["ratio_to_neighbor_abs_mean"] for e in spike_events
                         if np.isfinite(e["ratio_to_neighbor_abs_mean"])), default=np.nan)
        lake_summary["worst_spike_ratio"] = round(max_ratio, 2) if np.isfinite(max_ratio) else np.nan
    else:
        lake_summary["spike_gap_mean"] = np.nan
        lake_summary["spike_gap_median"] = np.nan
        lake_summary["spike_gap_max"] = -1
        lake_summary["worst_spike_ratio"] = np.nan

    # Print quick summary
    if n_total_spikes > 0:
        print(f"  SPIKES DETECTED: {n_total_spikes} events on {n_unique_dates} unique dates")
        for v in temporal_vars:
            mode_idx = int(v.replace(TEMPORAL_EOF_PREFIX, ""))
            cnt = per_mode_spike_count[v]
            if cnt > 0:
                print(f"    mode {mode_idx}: {cnt} spikes")
        if spike_events:
            worst = max(spike_events, key=lambda e: abs(e["abs_deviation_from_neighbor_mean"]))
            print(f"    worst spike: {worst['date']}, mode {worst['eof_mode']}, "
                  f"ratio={worst['ratio_to_neighbor_abs_mean']:.1f}x, "
                  f"gap_before={worst['gap_before']}, gap_after={worst['gap_after']}")
    else:
        print(f"  No spikes detected (k={k})")

    eofs_ds.close()
    return spike_events, lake_summary


# ========== Output ==========

SPIKE_EVENT_COLUMNS = [
    "lake_id", "eof_mode", "eof_var_name", "variance_explained",
    "timestep_index", "date", "days_since_epoch",
    "gap_before", "gap_after", "gap_max_either_side", "gap_sum_both_sides",
    "spike_value", "neighbor_mean", "neighbor_median",
    "abs_deviation_from_neighbor_mean", "ratio_to_neighbor_abs_mean",
    "mode_median", "mode_rsd", "n_rsd_from_median",
    "n_lake_pixels", "spatial_coverage_fraction",
]

LAKE_SUMMARY_COLUMNS = [
    "lake_id", "n_temporal_eofs", "n_timesteps", "n_lake_pixels",
    "n_spike_events", "n_unique_spike_dates", "fraction_dates_spiked", "has_spikes",
    "spike_gap_mean", "spike_gap_median", "spike_gap_max", "worst_spike_ratio",
]


def write_spike_events_csv(events: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPIKE_EVENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            writer.writerow(e)
    print(f"  Wrote {output_path} ({len(events)} events)")


def write_lake_summary_csv(summaries: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Collect all columns (base + dynamic per-mode columns)
    all_keys = list(LAKE_SUMMARY_COLUMNS)
    for s in summaries:
        for k in s:
            if k.startswith("n_spikes_mode") and k not in all_keys:
                all_keys.append(k)
    # Sort mode columns
    mode_cols = sorted([k for k in all_keys if k.startswith("n_spikes_mode")],
                       key=lambda x: int(x.replace("n_spikes_mode", "")))
    final_cols = [c for c in all_keys if not c.startswith("n_spikes_mode")] + mode_cols

    summaries.sort(key=lambda x: x.get("lake_id", ""))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nWrote summary: {output_path} ({len(summaries)} lakes)")


# ========== Lake discovery ==========

def discover_lakes(exp_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find all lakes with eofs.nc files.

    Returns:
        List of (lake_id, eofs_nc_path, prepared_nc_path) tuples.
    """
    results = []
    prepared_dir = os.path.join(exp_dir, "prepared")
    dineof_dir = os.path.join(exp_dir, "dineof")

    if not os.path.isdir(prepared_dir):
        print(f"ERROR: {prepared_dir} does not exist")
        sys.exit(1)

    for lake_id in sorted(os.listdir(prepared_dir)):
        prep_path = os.path.join(prepared_dir, lake_id, "prepared.nc")
        if not os.path.isfile(prep_path):
            continue

        # Find eofs.nc under dineof/{lake_id}/a*/eofs.nc
        eofs_candidates = glob.glob(os.path.join(dineof_dir, lake_id, "a*", "eofs.nc"))
        if not eofs_candidates:
            continue

        # Use the first alpha directory found
        eofs_path = sorted(eofs_candidates)[0]
        results.append((lake_id, eofs_path, prep_path))

    return results


# ========== CLI ==========

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Temporal EOF Spike Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All lakes, fast (no spatial coverage)
  python spike_diagnostics.py --exp-dir /path/to/exp

  # Single lake with spatial coverage
  python spike_diagnostics.py --exp-dir /path/to/exp --lake-id 000000044 --with-spatial-coverage

  # Stricter detection threshold
  python spike_diagnostics.py --exp-dir /path/to/exp --k 3.0
"""
    )

    p.add_argument("--exp-dir", required=True,
                   help="Experiment root directory")
    p.add_argument("--lake-id", default=None,
                   help="Process a single lake ID (e.g. 000000020)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Default: {exp-dir}/spike_diagnostics/")
    p.add_argument("--k", type=float, default=4.0,
                   help="Spike detection threshold in robust SDs (default: 4.0)")
    p.add_argument("--with-spatial-coverage", action="store_true",
                   help="Compute per-frame spatial coverage (slow — reads full prepared.nc)")
    p.add_argument("--neighbor-window", type=int, default=3,
                   help="Number of neighbors on each side for spike context (default: 3)")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    exp_dir = args.exp_dir.rstrip("/")
    output_dir = args.output_dir or os.path.join(exp_dir, "spike_diagnostics")

    print(f"Experiment dir: {exp_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Detection k:    {args.k}")
    print(f"Spatial cov:    {'yes' if args.with_spatial_coverage else 'no'}")

    # Discover lakes
    if args.lake_id:
        prep_path = os.path.join(exp_dir, "prepared", args.lake_id, "prepared.nc")
        eofs_candidates = glob.glob(os.path.join(exp_dir, "dineof", args.lake_id, "a*", "eofs.nc"))
        if not eofs_candidates:
            print(f"ERROR: No eofs.nc found for lake {args.lake_id}")
            sys.exit(1)
        lakes = [(args.lake_id, sorted(eofs_candidates)[0], prep_path)]
    else:
        lakes = discover_lakes(exp_dir)

    print(f"Found {len(lakes)} lake(s)")

    # Process
    all_events = []
    all_summaries = []

    for lake_id, eofs_path, prep_path in lakes:
        events, summary = process_lake(
            lake_id=lake_id,
            eofs_nc_path=eofs_path,
            prepared_nc_path=prep_path,
            k=args.k,
            with_spatial_coverage=args.with_spatial_coverage,
            neighbor_window=args.neighbor_window,
        )

        all_events.extend(events)
        all_summaries.append(summary)

        # Write per-lake spike events
        if events:
            per_lake_csv = os.path.join(output_dir, "per_lake", lake_id, "spike_events.csv")
            write_spike_events_csv(events, per_lake_csv)

    # Write global outputs
    if all_events:
        global_events_path = os.path.join(output_dir, "spike_events_all_lakes.csv")
        write_spike_events_csv(all_events, global_events_path)

    if all_summaries:
        global_summary_path = os.path.join(output_dir, "spike_summary_all_lakes.csv")
        write_lake_summary_csv(all_summaries, global_summary_path)

    # Print final stats
    n_with_spikes = sum(1 for s in all_summaries if s.get("has_spikes"))
    n_total = len(all_summaries)
    total_events = len(all_events)
    print(f"\n{'='*60}")
    print(f"Done. {n_with_spikes}/{n_total} lakes have spikes ({total_events} total spike events)")

    if n_with_spikes > 0:
        print(f"\nLakes with spikes:")
        for s in sorted(all_summaries, key=lambda x: x.get("n_spike_events", 0), reverse=True):
            if s.get("has_spikes"):
                print(f"  {s['lake_id']}: {s['n_spike_events']} events, "
                      f"{s['n_unique_spike_dates']} unique dates, "
                      f"worst_ratio={s.get('worst_spike_ratio', '?')}, "
                      f"spike_gap_max={s.get('spike_gap_max', '?')}")


if __name__ == "__main__":
    main()
