from __future__ import annotations
import os, shlex, subprocess
from pathlib import Path
from typing import Dict
from .contracts import DincaeArtifacts

def _generate_julia_script(arts: DincaeArtifacts, cfg: Dict) -> Path:
    """
    Write run_dincae.jl into arts.dincae_dir.
    """
    jl = f"""
using Pkg
if !isempty(get(ENV, "JULIA_PROJECT", ""))  # optional external project
    Pkg.activate(ENV["JULIA_PROJECT"])
end
Pkg.instantiate()

using CUDA, cuDNN
using DINCAE, DINCAE_utils, Dates, NCDatasets, JSON, Printf, PyPlot
using Logging

# make sure INFO-level logs show up in non-TTY jobs
global_logger(ConsoleLogger(stderr, Logging.Info))

CUDA.allowscalar(false)
try
    CUDA.versioninfo()
    @info "cuDNN loaded; version = $(cuDNN.version())"
catch err
    error("cuDNN not available in this environment. Error: " * string(err))
end

varname   = "{cfg.get('var_name', 'lake_surface_water_temperature')}"
fname_base = "{str(arts.prepared_cropped)}"  # Base file without CV
fname_cv   = "{str(arts.prepared_cropped_cv)}"  # CV file (uncleaned)
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

# --- STEP 1: Load & clean base file → lake_cleanup.nc ---
println("Loading base file: ", fname_base); flush(stdout)
ds_base = NCDataset(fname_base, "r"; diskless=true, persist=false)
A_base  = ds_base[varname][:,:,:]

# Convert NaN/Inf -> missing
A_base_clean = map(x -> (!ismissing(x) && isfinite(x)) ? x : missing, A_base)

# Get dimensions and permute
dimsA = dimnames(ds_base[varname])
target = ("lon","lat","time")
perm = map(d -> findfirst(==(d), dimsA), target)
if any(p->p===nothing, perm)
    close(ds_base)
    error("Variable '" * string(varname) * "' missing one of " * string(target) * "; found " * string(dimsA))
end
A_base_llt = permutedims(A_base_clean, Tuple(perm))

# Write to lake_cleanup.nc
fname_cleanup = joinpath(outdir, "lake_cleanup.nc")
println("Writing cleaned base to: ", fname_cleanup); flush(stdout)
ds_cleanup = NCDataset(fname_cleanup, "c", format = :netcdf4)
write(ds_cleanup, ds_base; exclude = [varname])
defVar(ds_cleanup, varname, A_base_llt, target)
close(ds_cleanup)
close(ds_base)

# --- STEP 2: Load & clean CV file → .clean.nc ---
println("Loading CV file: ", fname_cv); flush(stdout)
ds_cv = NCDataset(fname_cv, "r"; diskless=true, persist=false)
A_cv  = ds_cv[varname][:,:,:]
dimsCV = dimnames(ds_cv[varname])

# Ensure proper type and convert NaN/Inf -> missing
perm_cv = map(d -> findfirst(==(d), dimsCV), target)
if any(p->p===nothing, perm_cv)
    close(ds_cv)
    error("CV variable '" * string(varname) * "' missing one of " * string(target) * "; found " * string(dimsCV))
end

A_cv_typed = Array{{Union{{Missing,Float32}}}}(undef, size(A_cv))
@inbounds for I in eachindex(A_cv)
    x = A_cv[I]
    if x isa Missing
        A_cv_typed[I] = missing
    else
        A_cv_typed[I] = isfinite(x) ? Float32(x) : missing
    end
end
A_cv_llt = permutedims(A_cv_typed, Tuple(perm_cv))

# Write cleaned CV to .clean.nc
cv_dir, cv_name = splitdir(fname_cv)
cv_clean = joinpath(cv_dir, replace(cv_name, r"\\.nc$" => ".clean.nc"))
println("Writing cleaned CV to: ", cv_clean); flush(stdout)
ds_cv_clean = NCDataset(cv_clean, "c", format = :netcdf4)
write(ds_cv_clean, ds_cv; exclude = [varname])
defVar(ds_cv_clean, varname, A_cv_llt, target)
close(ds_cv_clean)
close(ds_cv)

# --- STEP 3: Use cleaned CV for DINCAE ---
data = [(
    filename    = cv_clean,  # Use the cleaned CV file we just created
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

    partition = slurm.get("partition", "orchid")
    account   = slurm.get("account",   "orchid")
    qos       = slurm.get("qos",       "orchid")
    gpus      = int(slurm.get("gpus",  1))
    cpus      = int(slurm.get("cpus",  4))
    mem       = slurm.get("mem",       "128G")
    wall      = slurm.get("time",      "24:00:00")
    julia     = cfg.get("runner", {}).get("julia_exe", "julia")

    env_lines = []
    denv = cfg.get("env", {}).get("dincae", {})
    if denv.get("module_load"): env_lines.append(denv["module_load"])
    if denv.get("activate"):    env_lines.append(denv["activate"])
    if cfg.get("runner", {}).get("JULIA_PROJECT"):
        env_lines.append(f'export JULIA_PROJECT="{cfg["runner"]["JULIA_PROJECT"]}"')
    
    depot_dir = Path(arts.dincae_dir) / ".julia_depot_${SLURM_JOB_ID}"
    env_lines += [
        "export JULIA_PKG_PRECOMPILE_AUTO=0",
        f'export JULIA_DEPOT_PATH="{depot_dir}:$HOME/.julia"',
        f"mkdir -p {depot_dir}",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
    ]
    env_block = "\n".join(env_lines)

    log_out = Path(arts.dincae_dir) / f"logs_dincae_{lake_id}.out"
    log_err = Path(arts.dincae_dir) / f"logs_dincae_{lake_id}.err"

    # Generate SLURM script
    sb = f"""#!/bin/bash
#SBATCH --job-name=dincae_{lake_id}
#SBATCH --exclude=gpuhost007,gpuhost012,gpuhost016
#SBATCH --time={wall}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --chdir={arts.dincae_dir}
#SBATCH -o {log_out}
#SBATCH -e {log_err}

cd {arts.dincae_dir}
{env_block}

set -euo pipefail

echo "=========================================="
echo "DINCAE Job Starting"
echo "=========================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "JULIA_PROJECT=$JULIA_PROJECT"
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# === Precompile OUTSIDE srun (no timeout, full output) ===
echo "=== Precompiling Julia packages ==="
echo "Start: $(date)"

{julia} -e 'using CUDA, cuDNN, DINCAE; println("All packages loaded successfully")'
PRECOMPILE_EXIT=$?

echo "End: $(date)"  
echo "Exit code: $PRECOMPILE_EXIT"
echo ""

if [ $PRECOMPILE_EXIT -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Package precompilation failed"
    echo "Exit code: $PRECOMPILE_EXIT"
    echo "=========================================="
    exit 1
fi

echo "Precompilation complete!"
echo ""

# === GPU verification (fast, already precompiled) ===
echo "=== Verifying GPU access ==="
srun --gres=gpu:{gpus} --ntasks=1 --cpus-per-task={cpus} nvidia-smi || {{
    echo "ERROR: No GPU visible"
    exit 1
}}
echo ""

echo "=== Verifying CUDA functional ==="
srun --gres=gpu:{gpus} --ntasks=1 --cpus-per-task={cpus} {julia} -e 'using CUDA; @assert CUDA.functional()' || {{
    echo "ERROR: CUDA not functional"  
    exit 1
}}
echo ""

# === Main execution ===
echo "=========================================="
echo "Starting DINCAE Training"
echo "Start time: $(date)"
echo "=========================================="
exec srun --gres=gpu:{gpus} --ntasks=1 --cpus-per-task={cpus} stdbuf -oL -eL {julia} --project {script_path.name}
"""
    jobfile.write_text(sb)
    subprocess.check_call(["sbatch", "--wait", str(jobfile)])

def run_julia_local(arts: DincaeArtifacts, cfg: Dict) -> None:
    # local run (only if already on a GPU node)
    env = os.environ.copy()
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
