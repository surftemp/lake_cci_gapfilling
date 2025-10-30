from pathlib import Path
from .contracts import PreparedNC, DincaeArtifacts

def write_dineof_shaped_outputs(arts: DincaeArtifacts, prepared: PreparedNC, out_dir: Path, cfg: dict) -> dict:
    """
    Produce files that the existing post-processor expects from DINEOF:
      - {out_dir}/output.nc  (required)
      - {out_dir}/merged.nc  (optional; create if your post uses it)
    Ensure var name, dims, and time coordinate match DINEOF conventions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # TODO: transform DINCAE predictions -> output.nc (DINEOF-like)
    output_nc = out_dir / "output.nc"
    merged_nc = None  # or out_dir / "merged.nc"
    return {"output_nc": output_nc, "merged_nc": merged_nc}
