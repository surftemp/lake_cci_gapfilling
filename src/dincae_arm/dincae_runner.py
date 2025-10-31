from __future__ import annotations
import os, shlex, subprocess
from pathlib import Path
from typing import Dict
from .contracts import DincaeArtifacts

def _build_julia_cmd(cfg: Dict, arts: DincaeArtifacts) -> list[str]:
    julia = cfg.get("runner", {}).get("julia_exe", "julia")
    script = cfg.get("runner", {}).get("script", "run_dincae.jl")
    use_cv = bool(cfg.get("cv", {}).get("use_cv", True))
    in_path = arts.prepared_cropped_cv if use_cv else arts.prepared_cropped
    args = {
        "--in": str(in_path),
        "--outdir": str(arts.dincae_dir),
        "--epochs": cfg.get("train", {}).get("epochs", 300),
        "--batch": cfg.get("train", {}).get("batch_size", 32),
        "--ntime_win": cfg.get("train", {}).get("ntime_win", 0),
        "--lr": cfg.get("train", {}).get("learning_rate", 1e-4),
        "--enc_levels": cfg.get("train", {}).get("enc_levels", 3),
        "--obs_err_std": cfg.get("train", {}).get("obs_err_std", 0.2),
        "--save_interval": cfg.get("train", {}).get("save_epochs_interval", 10),
        "--use_gpu": int(bool(cfg.get("train", {}).get("use_gpu", True))),
    }
    cmd = [julia]
    if cfg.get("runner", {}).get("julia_project", True):
        cmd += ["--project"]
    cmd += [script]
    for k, v in args.items(): cmd += [k, str(v)]
    return cmd

def run_julia_local(arts: DincaeArtifacts, cfg: Dict) -> None:
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" in cfg.get("runner", {}):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["runner"]["CUDA_VISIBLE_DEVICES"])
    if "JULIA_PROJECT" in cfg.get("runner", {}):
        env["JULIA_PROJECT"] = str(cfg["runner"]["JULIA_PROJECT"])
    cmd = _build_julia_cmd(cfg, arts)
    subprocess.check_call(cmd, env=env, cwd=str(arts.dincae_dir))

def submit_slurm_job(arts: DincaeArtifacts, cfg: Dict) -> None:
    slurm = cfg.get("slurm", {})
    lake_id = cfg.get("lake_id", "lake")
    script_path = Path(arts.dincae_dir) / f"run_dincae_{lake_id}.slurm"
    log_out = slurm.get("log_out", f"logs_dincae_{lake_id}.out")
    log_err = slurm.get("log_err", f"logs_dincae_{lake_id}.err")
    cmd = " ".join(shlex.quote(p) for p in _build_julia_cmd(cfg, arts))
    script = f"""#!/bin/bash
#SBATCH -J dincae_{lake_id}
#SBATCH -o {log_out}
#SBATCH -e {log_err}
#SBATCH -p {slurm.get('partition','orchid')}
#SBATCH --gres=gpu:{slurm.get('gpus',1)}
#SBATCH -t {slurm.get('time','24:00:00')}
#SBATCH --mem={slurm.get('mem','128G')}
#SBATCH -c {slurm.get('cpus',4)}
{f"#SBATCH -A {slurm['account']}" if 'account' in slurm else ''}
{f"#SBATCH --qos={slurm['qos']}" if 'qos' in slurm else ''}
cd {arts.dincae_dir}
{cmd}
"""
    script_path.write_text(script)
    subprocess.check_call(["sbatch", str(script_path)])

def run(cfg: Dict, arts: DincaeArtifacts) -> DincaeArtifacts:
    mode = cfg.get("runner", {}).get("mode", "local")
    skip_existing = bool(cfg.get("runner", {}).get("skip_existing", True))
    pred = arts.dincae_dir / "data-avg.nc"
    if skip_existing and pred.exists():
        arts.pred_path = pred
        return arts
    if mode == "local":
        run_julia_local(arts, cfg)
    else:
        submit_slurm_job(arts, cfg)
    if pred.exists():
        arts.pred_path = pred
    return arts
