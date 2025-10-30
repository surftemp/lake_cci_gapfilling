import xarray as xr
from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig
from .stats import get_recorder

class FinalStatsWriteStep(ProcessingStep):
    """Writes CSV stats next to prepared.nc using ORIGINAL lake size as denominator."""

    def should_apply(self, config: ProcessingConfig) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Final Stats Write"

    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            rec = get_recorder()
            
            # Ensure we have the final time information for proper aggregation
            if "time" in ds.coords:
                rec.set_final_time_days(ds.coords["time"].values)
            
            # Set final denominator for metadata purposes
            rec.set_final_denominator_from_ds(ds)
            
            # Set output directory and write all stats
            rec.set_output_dir_from_output_nc(config.output_file)
            
            # Validate we have the original denominator
            if rec.lake_pixels_original is None:
                print("Warning: Original lake size not captured. Stats fractions may be incorrect.")
            else:
                print(f"Stats: Using original lake size ({rec.lake_pixels_original} pixels) as denominator")
            
            rec.write_all()
            
            # Log useful summary
            if rec.lake_pixels_original and rec.lake_pixels_final:
                total_removed = rec.lake_pixels_original - rec.lake_pixels_final
                pct_removed = (total_removed / rec.lake_pixels_original) * 100
                print(f"Stats summary: {total_removed} pixels removed total "
                      f"({pct_removed:.1f}% of original {rec.lake_pixels_original} pixels)")
            
            print(f"Stats CSV written beside: {config.output_file}")
            return ds
            
        except Exception as e:
            raise ProcessingError(self.name, str(e))