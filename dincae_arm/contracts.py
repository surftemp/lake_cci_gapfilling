from dataclasses import dataclass
from pathlib import Path

@dataclass
class PreparedNC:
    path: Path  # prepared.nc

@dataclass
class DincaeArtifacts:
    pred_path: Path | None = None  # raw DINCAE predictions
