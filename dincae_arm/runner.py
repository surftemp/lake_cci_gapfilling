from pathlib import Path

def run(cfg: dict, inputs: dict) -> dict:
    """
    Train/infer DINCAE; return {'pred_path': Path(...)}.
    Replace with SLURM/local invocation using cfg['models']['dincae']['runner'].
    """
    workdir = Path(inputs["workdir"])
    # TODO: real call; placeholder output path:
    return {"pred_path": workdir / "dincae_raw_predictions.nc"}
