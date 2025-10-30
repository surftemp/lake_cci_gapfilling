# src/lake_dashboard/dineof_preprocessor/lswt_processing/stats.py
import os
import csv
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import xarray as xr

# --------- time helpers ---------
_REF = np.datetime64("1981-01-01T12:00:00")

def _to_datetime64_from_days(days: np.ndarray) -> np.ndarray:
    return _REF + days.astype("timedelta64[D]")

def _ymd_from_days(days_int: np.ndarray) -> np.ndarray:
    dts = _to_datetime64_from_days(days_int.astype("int64"))
    return dts.astype("datetime64[D]")

def _month(dtD: np.ndarray) -> np.ndarray:
    return dtD.astype("datetime64[M]")

def _season_name(month_num: int) -> str:
    # DJF, MAM, JJA, SON
    if month_num in (12, 1, 2):  return "DJF"
    if month_num in (3, 4, 5):   return "MAM"
    if month_num in (6, 7, 8):   return "JJA"
    return "SON"

# --------- data holders ---------
@dataclass
class _StepDaily:
    # maps: day(int days since 1981-01-01 12:00) -> count(int)
    pixels_removed: dict = field(default_factory=lambda: defaultdict(int))
    pixels_replaced: dict = field(default_factory=lambda: defaultdict(int))
    # list of removed time steps (as int "days since 1981-01-01 12:00")
    timesteps_removed_days: list = field(default_factory=list)

@dataclass
class StatsRecorder:
    """
    Global recorder used by pipeline steps.
    Fractions are computed against the ORIGINAL lake mask size for meaningful cross-lake comparisons.
    """
    # ORIGINAL denominator (set at the beginning, before any processing)
    lake_pixels_original: int | None = None
    # Final lake size (for metadata purposes)
    lake_pixels_final: int | None = None
    # for meaningful meta (optional)
    lake_id: int | None = None

    # per-step, per-day counts
    _daily: dict = field(default_factory=lambda: defaultdict(_StepDaily))
    # spatial-mask pruning totals (no per-day, just totals per step)
    _spatial_removed: dict = field(default_factory=lambda: defaultdict(int))

    # timing meta for % timesteps removed
    total_time_before: int | None = None     # count before frame removal (after date filtering)
    total_time_after: int | None = None      # count after all processing
    final_time_days: np.ndarray | None = None  # int64 array of final days (after all steps)
    # where to write CSVs
    output_dir: str | None = None

    # ---------- lifecycle ----------
    def reset(self):
        self.lake_pixels_original = None
        self.lake_pixels_final = None
        self.lake_id = None
        self._daily.clear()
        self._spatial_removed.clear()
        self.total_time_before = None
        self.total_time_after = None
        self.final_time_days = None
        self.output_dir = None

    # ---------- context setters ----------
    def set_lake_meta(self, ds: xr.Dataset):
        """Lightweight: gather lake id early if present (optional)."""
        if self.lake_id is None:
            lid = ds.attrs.get("_lake_id_value", ds.attrs.get("lake_id", None))
            if lid is not None:
                try:
                    self.lake_id = int(lid)
                except Exception:
                    pass

    def set_original_denominator_from_ds(self, ds: xr.Dataset):
        """
        Set the ORIGINAL denominator from the initial dataset BEFORE any processing.
        This should be called right after data loading, before any filtering steps.
        """
        if "lakeid" in ds:
            m = (ds["lakeid"].data > 0)
            self.lake_pixels_original = int(np.count_nonzero(m))
        elif "_mask" in ds.attrs:
            m = np.asarray(ds.attrs["_mask"])
            self.lake_pixels_original = int(np.count_nonzero(m))
        else:
            self.lake_pixels_original = 0  # safe default
        print(f"Stats: Set original lake size = {self.lake_pixels_original} pixels")

    def set_total_time_before(self, n: int):
        if self.total_time_before is None:
            self.total_time_before = int(n)

    def set_total_time_after(self, n: int):
        """Set the final time count after all processing."""
        self.total_time_after = int(n)

    def set_final_denominator_from_ds(self, ds: xr.Dataset):
        """
        Set the FINAL lake size from the final dataset after all processing.
        Used for metadata and validation purposes.
        """
        if "lakeid" in ds:
            m = (ds["lakeid"].data > 0)
            self.lake_pixels_final = int(np.count_nonzero(m))
        elif "_mask" in ds.attrs:
            m = np.asarray(ds.attrs["_mask"])
            self.lake_pixels_final = int(np.count_nonzero(m))
        else:
            self.lake_pixels_final = 0  # safe default

    def set_final_time_days(self, days_int64: np.ndarray):
        self.final_time_days = np.asarray(days_int64, dtype="int64")

    def set_output_dir_from_output_nc(self, output_nc_path: str):
        self.output_dir = os.path.dirname(os.path.abspath(output_nc_path))

    # ---------- recording APIs used by steps ----------
    def record_pixel_filter_daily(self, step_name: str, time_days: np.ndarray, removed_per_time: np.ndarray):
        """Record per-time pixel removals for a filtering step."""
        if removed_per_time is None or removed_per_time.size == 0:
            return
        daily = self._daily[step_name]
        for d, n in zip(time_days.astype("int64"), removed_per_time.astype("int64")):
            if n > 0:
                daily.pixels_removed[int(d)] += int(n)

    def record_pixel_replacement_daily(self, step_name: str, time_days: np.ndarray, replaced_per_time: np.ndarray):
        """Record per-time pixel replacements (e.g., IceMaskReplacementStep)."""
        if replaced_per_time is None or replaced_per_time.size == 0:
            return
        daily = self._daily[step_name]
        for d, n in zip(time_days.astype("int64"), replaced_per_time.astype("int64")):
            if n > 0:
                daily.pixels_replaced[int(d)] += int(n)

    def record_spatial_mask_prune(self, step_name: str, removed_pixels: int):
        """Record total lake-pixel removals due to spatial pruning (shore / availability)."""
        self._spatial_removed[step_name] += int(removed_pixels)

    def record_timesteps_removed(self, step_name: str, removed_time_days: list[int] | np.ndarray):
        """Record which time steps were removed by a frame filter."""
        daily = self._daily[step_name]
        for d in removed_time_days:
            daily.timesteps_removed_days.append(int(d))

    # ---------- CSV helpers ----------
    def _write_csv(self, path: str, header: list[str], rows: list[list]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def _denom(self) -> int:
        """Return the ORIGINAL lake size for fraction calculations."""
        return int(self.lake_pixels_original or 0)

    # ---------- aggregation logic ----------
    def _daily_rows_normalised(self) -> list[list]:
        """
        Build daily table with **fractions** computed against the ORIGINAL lake size.
        NOTE: Only processes SURVIVING days. Timestep removals are handled separately in summary.
        """
        denom = max(self._denom(), 1)  # Use ORIGINAL size, avoid div-by-zero
        rows = []
        
        # IMPORTANT: Only process days that survived all processing
        # Removed timesteps are not included here by design
        if self.final_time_days is not None and self.final_time_days.size > 0:
            all_days = np.sort(self.final_time_days)
        else:
            # fallback to only pixel removal/replacement days (not timestep removal days)
            seen = set()
            for d in self._daily.values():
                seen |= set(d.pixels_removed.keys()) | set(d.pixels_replaced.keys())
                # NOTE: We deliberately DON'T include timesteps_removed_days here
                # because those days are no longer in the dataset
            all_days = np.array(sorted(seen), dtype="int64") if seen else np.array([], dtype="int64")

        for step, S in self._daily.items():
            # build maps with default 0 for missing days
            rm_map = defaultdict(int, S.pixels_removed)
            rp_map = defaultdict(int, S.pixels_replaced)
            # NOTE: timesteps_removed_days are handled at summary level, not daily level
            for d in all_days:
                rm = rm_map[int(d)]
                rp = rp_map[int(d)]
                # Use ORIGINAL lake size as denominator
                frac_rm = float(rm) / float(denom)
                frac_rp = float(rp) / float(denom)
                dt = _ymd_from_days(np.array([d], dtype="int64"))[0].astype(object)
                # timestep_removed_flag is always 0 for surviving days
                tflag = "0"
                rows.append([step, str(dt), rm, frac_rm, rp, frac_rp, tflag])
        return rows

    def _aggregate_monthly(self, rows_daily: list[list]) -> list[list]:
        """
        Mean of daily fractions for each month (1-12), averaged across ALL years.
        Plus separate timestep removal counts per step per month.
        """
        agg = defaultdict(lambda: {"fr_rm_sum": 0.0, "fr_rp_sum": 0.0, "n": 0})
        for step, date_str, _, fr_rm, _, fr_rp, tflag in rows_daily:
            y, m, _ = (int(x) for x in date_str.split("-"))
            key = (step, m)  # Only group by step and month, not year
            A = agg[key]
            A["fr_rm_sum"] += float(fr_rm)
            A["fr_rp_sum"] += float(fr_rp)
            A["n"] += 1
        
        # Add timestep removal counts by analyzing removed days by month
        timestep_counts = defaultdict(int)
        for step, daily_data in self._daily.items():
            for removed_day in daily_data.timesteps_removed_days:
                dt = _ymd_from_days(np.array([removed_day], dtype="int64"))[0].astype(object)
                date_str = str(dt)
                try:
                    _, m, _ = (int(x) for x in date_str.split("-"))
                    timestep_counts[(step, m)] += 1
                except:
                    continue  # Skip malformed dates
        
        out = []
        # Combine pixel-based stats with timestep removal counts
        all_keys = set(agg.keys()) | set(timestep_counts.keys())
        for (step, m) in sorted(all_keys):
            pixel_stats = agg.get((step, m), {"fr_rm_sum": 0.0, "fr_rp_sum": 0.0, "n": 0})
            n = max(pixel_stats["n"], 1) if pixel_stats["n"] > 0 else 1
            mean_fr_rm = pixel_stats["fr_rm_sum"] / n if pixel_stats["n"] > 0 else 0.0
            mean_fr_rp = pixel_stats["fr_rp_sum"] / n if pixel_stats["n"] > 0 else 0.0
            t_removed = timestep_counts.get((step, m), 0)
            
            out.append([step, m, mean_fr_rm, mean_fr_rp, t_removed])
        return out

    def _aggregate_seasonal(self, rows_daily: list[list]) -> list[list]:
        """
        Mean of daily fractions for each season, averaged across ALL years.
        Plus separate timestep removal counts per step per season.
        """
        agg = defaultdict(lambda: {"fr_rm_sum": 0.0, "fr_rp_sum": 0.0, "n": 0})
        for step, date_str, _, fr_rm, _, fr_rp, tflag in rows_daily:
            y, m, _ = (int(x) for x in date_str.split("-"))
            key = (step, _season_name(m))  # Only group by step and season, not year
            A = agg[key]
            A["fr_rm_sum"] += float(fr_rm)
            A["fr_rp_sum"] += float(fr_rp)
            A["n"] += 1
        
        # Add timestep removal counts by analyzing removed days by season
        timestep_counts = defaultdict(int)
        for step, daily_data in self._daily.items():
            for removed_day in daily_data.timesteps_removed_days:
                dt = _ymd_from_days(np.array([removed_day], dtype="int64"))[0].astype(object)
                date_str = str(dt)
                try:
                    _, m, _ = (int(x) for x in date_str.split("-"))
                    season = _season_name(m)
                    timestep_counts[(step, season)] += 1
                except:
                    continue  # Skip malformed dates
        
        out = []
        # Combine pixel-based stats with timestep removal counts
        all_keys = set(agg.keys()) | set(timestep_counts.keys())
        for (step, season) in sorted(all_keys):
            pixel_stats = agg.get((step, season), {"fr_rm_sum": 0.0, "fr_rp_sum": 0.0, "n": 0})
            n = max(pixel_stats["n"], 1) if pixel_stats["n"] > 0 else 1
            mean_fr_rm = pixel_stats["fr_rm_sum"] / n if pixel_stats["n"] > 0 else 0.0
            mean_fr_rp = pixel_stats["fr_rp_sum"] / n if pixel_stats["n"] > 0 else 0.0
            t_removed = timestep_counts.get((step, season), 0)
            
            out.append([step, season, mean_fr_rm, mean_fr_rp, t_removed])
        return out

    def _summary_rows(self, rows_daily: list[list]) -> list[list]:
        """
        Per step summary with DIRECT timestep removal counting.
        """
        # initialise with zeros for all days per step
        fr_rm_sum = defaultdict(float)
        fr_rp_sum = defaultdict(float)
        fr_n      = defaultdict(int)

        # count days per step (only surviving days for pixel stats)
        for step, _date_str, _rm, fr_rm, _rp, fr_rp, tflag in rows_daily:
            fr_rm_sum[step] += float(fr_rm)
            fr_rp_sum[step] += float(fr_rp)
            fr_n[step] += 1

        # include spatial pruning as additional "removed pixels" fractions
        denom = max(self._denom(), 1)  # Use ORIGINAL size
        for step, nrem in self._spatial_removed.items():
            n_days = max(next(iter(fr_n.values()), 0), 0)
            if n_days == 0 and self.final_time_days is not None:
                n_days = int(self.final_time_days.size)
            if n_days == 0:
                n_days = 1
            fr_rm_sum[step] += float(nrem) / float(denom)
            fr_n[step] = max(fr_n[step], n_days)

        # DIRECT timestep removal counting
        timestep_removed_counts = defaultdict(int)
        for step, daily_data in self._daily.items():
            timestep_removed_counts[step] = len(daily_data.timesteps_removed_days)

        rows = []
        before_T = self.total_time_before
        after_T = self.total_time_after
        
        # Get all steps that have either pixel activities or timestep removals
        all_steps = set(fr_n.keys()) | set(self._spatial_removed.keys()) | set(timestep_removed_counts.keys())
        
        for step in sorted(all_steps):
            n = max(fr_n[step], 1)
            mean_fr_rm = fr_rm_sum[step] / n if fr_n[step] > 0 else 0.0
            mean_fr_rp = fr_rp_sum[step] / n if fr_n[step] > 0 else 0.0
            tcut = timestep_removed_counts.get(step, 0)  # DIRECT count
            pct_cut = (tcut / before_T) if (before_T and before_T > 0) else ""
            rows.append([
                step,
                mean_fr_rm,                        # mean_daily_fraction_removed (vs ORIGINAL)
                mean_fr_rp,                        # mean_daily_fraction_replaced (vs ORIGINAL)
                tcut,                              # timesteps_removed (DIRECT count)
                before_T if before_T is not None else "",  # time_steps_before
                after_T if after_T is not None else "",    # time_steps_after
                pct_cut                             # percent_timesteps_removed
            ])
        return rows

    # ---------- top-level writer ----------
    def write_all(self):
        if not self.output_dir:
            return

        # build daily rows (only surviving days)
        daily_rows = self._daily_rows_normalised()

        # DAILY
        self._write_csv(
            os.path.join(self.output_dir, "stats_daily.csv"),
            ["step", "date", "pixels_removed", "fraction_removed", "pixels_replaced", "fraction_replaced", "timestep_removed_flag"],
            daily_rows
        )

        # MONTHLY (mean of daily fractions across all years for each month + direct timestep counts)
        monthly = self._aggregate_monthly(daily_rows)
        self._write_csv(
            os.path.join(self.output_dir, "stats_monthly.csv"),
            ["step", "month", "mean_fraction_removed", "mean_fraction_replaced", "timesteps_removed"],
            monthly
        )

        # SEASONAL (mean of daily fractions across all years for each season + direct timestep counts)
        seasonal = self._aggregate_seasonal(daily_rows)
        self._write_csv(
            os.path.join(self.output_dir, "stats_seasonal.csv"),
            ["step", "season", "mean_fraction_removed", "mean_fraction_replaced", "timesteps_removed"],
            seasonal
        )

        # SUMMARY (mean of daily fractions over the final timeline + direct timestep counts)
        summary = self._summary_rows(daily_rows)
        self._write_csv(
            os.path.join(self.output_dir, "stats_summary.csv"),
            ["step", "mean_daily_fraction_removed", "mean_daily_fraction_replaced",
             "timesteps_removed", "time_steps_before", "time_steps_after", "percent_timesteps_removed"],
            summary
        )

# singleton accessor
_GLOBAL: StatsRecorder | None = None
def get_recorder() -> StatsRecorder:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = StatsRecorder()
    return _GLOBAL