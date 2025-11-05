from __future__ import annotations
import os, shlex, subprocess
from pathlib import Path
from typing import Dict
from .contracts import DincaeArtifacts

def _generate_julia_script(arts: DincaeArtifacts, cfg: Dict) -> Path:
    """
    Write the Julia script we run (run_dincae.jl) into arts.dincae_dir.
    This closely follows your original DINCAERunner behavior.
    """
    jl = f"""
using Pkg
if !isempty(get(ENV, "JULIA_PROJECT", ""))  # optional external project
    Pkg.activate(ENV["JULIA_PROJECT"])
end
Pkg.instantiate()

using CUDA, cuDNN
using DINCAE, DINCAE_utils, Dates, NCDatasets, JSON, Printf, PyPlot

CUDA.allowscalar(false)
try
    CUDA.versioninfo()
    @info "cuDNN loaded; version = $(cuDNN.version())"
catch err
    error("cuDNN not available in this environment. Error: " * string(err))
end

varname   = "{cfg.get('var_name', 'lake_surface_water_temperature')}"
infile    = "{str(arts.prepared_cropped_cv)}"
outdir    = "{str(arts.dincae_dir)}"
mkpath(outdir)

# --- params from cfg ---
epochs      = {int(cfg.get('train', {}).get('epochs', 300))}
batch       = {int(cfg.get('train', {}).get('batch_size', 32))}
ntime_win   = {int(cfg.get('train', {}).get('ntime_win', 0))}
lr          = {float(cfg.get('train', {}).get('learning_rate', 1e-4))}
enc_levels  = {int(cfg.get('train', {}).get('enc_levels', 3))}
obs_err_std = {float(cfg.get('train', {}).get('obs_err_std', 0.2))}
save_int    = {int(cfg.get('train', {}).get('save_epochs_interval', 10))}

save_epochs = collect(save_int:save_int:epochs)
Atype = CuArray{{Float32}}

# load CV dataset, re-order to (lon,lat,time) and clean
println("Loading CV: ", infile); flush(stdout)
ds = NCDataset(infile, "r"; diskless=true, persist=false)
A  = ds[varname][:,:,:]
dimsA = dimnames(ds[varname])
target = ("lon","lat","time")
perm = map(d -> findfirst(==(d), dimsA), target)
if any(p->p===nothing, perm)
    close(ds)
    error("Var '" * string(varname) * "' missing one of " * string(target) * "; found " * string(dimsA))
end
# convert to Union{{Missing,Float32}} and permute
B = Array{{Union{{Missing,Float32}}}}(undef, size(A))
@inbounds for I in eachindex(A)
    x = A[I]
    if x isa Missing
        B[I] = missing
    else
        B[I] = isfinite(x) ? Float32(x) : missing
    end
end
B_llt = permutedims(B, Tuple(perm))
close(ds)

data = [(
    filename    = infile,
    varname     = varname,
    obs_err_std = obs_err_std,
    jitter_std  = 0.0005,
    isoutput    = true
)]
data_all = [data, data]
fname_rec = joinpath(outdir, "data-avg.nc")

println("="^60)
println("DINCAE reconstruction starting…")
println("epochs=",epochs, " batch=",batch, " ntime_win=",ntime_win, " lr=",lr, " enc_levels=",enc_levels)
flush(stdout)

loss = DINCAE.reconstruct(
    Atype, data_all, [fname_rec];
    epochs               = epochs,
    batch_size           = batch,
    enc_nfilter_internal = round.(Int, 32 * 2 .^ (0:enc_levels-1)),
    clip_grad            = 5.0,
    ntime_win            = ntime_win,
    upsampling_method    = :nearest,
    loss_weights_refine  = (0.3, 0.7),
    min_std_err          = obs_err_std,
    regularization_L2_beta = 1e-5,
    save_epochs          = collect(save_epochs),
    learning_rate        = lr,
    modeldir             = outdir
)

open(joinpath(outdir, "loss_history.json"), "w") do io
    JSON.print(io, Dict("loss"=>loss, "epochs"=>collect(1:length(loss))))
end
println("Wrote ", fname_rec); flush(stdout)
"""
    script_path = arts.dincae_dir / "run_dincae.jl"
    script_path.write_text(jl)
    return script_path

def submit_slurm_job(arts: DincaeArtifacts, cfg: Dict) -> None:
    slurm = cfg.get("slurm", {})
    lake_id = cfg.get("lake_id", "lake")
    script_path = _generate_julia_script(arts, cfg)
    jobfile = Path(arts.dincae_dir) / f"run_dincae_{lake_id}.slurm"
    log_out = slurm.get("log_out", f"logs_dincae_{lake_id}.out")
    log_err = slurm.get("log_err", f"logs_dincae_{lake_id}.err")
    julia   = cfg.get("runner", {}).get("julia_exe", "julia")

    # stage env from config
    env_lines = []
    # cfg["env"] is the whole env block from the top-level JSON; pick dincae section if present
    denv = cfg.get("env", {}).get("dincae", {})
    if denv.get("module_load"): env_lines.append(denv["module_load"])
    if denv.get("activate"):    env_lines.append(denv["activate"])
    # Julia environment
    if cfg.get("runner", {}).get("JULIA_PROJECT"):
        env_lines.append(f'export JULIA_PROJECT="{cfg["runner"]["JULIA_PROJECT"]}"')
    # CUDA visibility (optional)
    if "CUDA_VISIBLE_DEVICES" in cfg.get("runner", {}):
        env_lines.append(f'export CUDA_VISIBLE_DEVICES="{cfg["runner"]["CUDA_VISIBLE_DEVICES"]}"')
    # Depots per job to avoid contention
    depot_dir = Path(arts.dincae_dir) / ".julia_depot_${SLURM_JOB_ID}"
    env_lines.append("export JULIA_PKG_PRECOMPILE_AUTO=0")
    env_lines.append(f'export JULIA_DEPOT_PATH="{depot_dir}:$HOME/.julia"')
    env_lines.append(f"mkdir -p {depot_dir}")

    env_block = "\n".join(env_lines)

    sb = f"""#!/bin/bash
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
{env_block}

echo "Using JULIA_PROJECT=$JULIA_PROJECT"
echo "Using JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"

{julia} --project {script_path.name}
"""
    jobfile.write_text(sb)
    # Block until the GPU job finishes so 'dincae' stage completes only after training
    subprocess.check_call(["sbatch", "--wait", str(jobfile)])

def run_julia_local(arts: DincaeArtifacts, cfg: Dict) -> None:
    # local run (only if already on a GPU node)
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" in cfg.get("runner", {}):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["runner"]["CUDA_VISIBLE_DEVICES"])
    if "JULIA_PROJECT" in cfg.get("runner", {}):
        env["JULIA_PROJECT"] = str(cfg["runner"]["JULIA_PROJECT"])
    script = _generate_julia_script(arts, cfg)
    cmd = [cfg.get("runner", {}).get("julia_exe", "julia"), "--project", str(script)]
    subprocess.check_call(cmd, env=env, cwd=str(arts.dincae_dir))

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
