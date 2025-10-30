from pathlib import Path
from .contracts import PreparedNC

def build_inputs(prepared: PreparedNC, workdir: Path, cfg: dict) -> dict:
    """
    Convert prepared.nc to DINCAE-ready tensors/windows/masks in workdir.
    Return an artifacts/manifests dict for the runner.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    # TODO: implement windowing/normalization per cfg["models"]["dincae"]["data"]
    return {"prepared_nc": prepared.path, "workdir": workdir}
