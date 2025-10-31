from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PreparedNC:
    """Original pipeline prepared file (usually prepared.nc)."""
    path: Path

@dataclass
class DincaeArtifacts:
    """Manifest for DINCAE adaptor steps."""
    dincae_dir: Path  # e.g. {run_root}/dincae/{lake_id9}/{alpha_slug}
    prepared_datetime: Path                 # prepared_datetime.nc
    prepared_cropped: Path                  # prepared_datetime_cropped.nc
    prepared_cropped_cv: Path               # prepared_datetime_cropped_add_clouds.nc
    prepared_cropped_clean: Path            # prepared_datetime_cropped_add_clouds.clean.nc
    pred_path: Optional[Path] = None        # data-avg.nc (cropped)
    pred_full_path: Optional[Path] = None   # data-avg-full.nc (full grid/time)
