from __future__ import annotations
import os, shlex, subprocess
from pathlib import Path
from typing import Dict
from .contracts import DincaeArtifacts

def _generate_julia_script(arts: DincaeArtifacts, cfg: Dict) -> Path:
    """
    Write run_dincae.jl into arts.dincae_dir.
    
    All DINCAE training parameters can be configured via the JSON config file
    under the "dincae.train" section.
    
    BACKWARD COMPATIBLE: All defaults match the original implementation.
    New parameters have sensible defaults that preserve original behavior.
    """
    
    # Extract train config with defaults
    train_cfg = cfg.get('train', {})
    
    # Core training parameters - ORIGINAL DEFAULTS PRESERVED
    epochs = int(train_cfg.get('epochs', 300))
    batch_size = int(train_cfg.get('batch_size', 32))
    ntime_win = int(train_cfg.get('ntime_win', 0))  # Original default: 0
    learning_rate = float(train_cfg.get('learning_rate', 1e-4))
    enc_levels = int(train_cfg.get('enc_levels', 3))  # Original default: 3
    save_epochs_interval = int(train_cfg.get('save_epochs_interval', 10))  # Original default: 10
    
    # Observation error - ORIGINAL DEFAULT PRESERVED
    obs_err_std = float(train_cfg.get('obs_err_std', 0.2))  # Original default: 0.2
    
    # NEW: min_std_err - separate from obs_err_std
    # If not specified, fall back to obs_err_std for backward compatibility with old behavior
    # (even though that behavior was problematic, we preserve it unless user specifies otherwise)
    min_std_err = float(train_cfg.get('min_std_err', obs_err_std))
    
    # Advanced training parameters - ORIGINAL HARDCODED VALUES AS DEFAULTS
    clip_grad = float(train_cfg.get('clip_grad', 5.0))
    upsampling_method = train_cfg.get('upsampling_method', 'nearest')  # Original: :nearest
    regularization_L2_beta = float(train_cfg.get('regularization_L2_beta', 1e-5))
    jitter_std = float(train_cfg.get('jitter_std', 0.0005))  # Original: 0.0005
    
    # Loss weights for refinement - ORIGINAL DEFAULT PRESERVED
    # Original hardcoded: (0.3, 0.7)
    loss_weights_refine = train_cfg.get('loss_weights_refine', [0.3, 0.7])
    if isinstance(loss_weights_refine, (list, tuple)):
        loss_weights_str = "(" + ", ".join(str(w) for w in loss_weights_refine) + ",)"
    else:
        loss_weights_str = f"({float(loss_weights_refine)},)"
    
    # Encoder filter configuration - ORIGINAL FORMULA PRESERVED
    enc_nfilter_base = int(train_cfg.get('enc_nfilter_base', 32))  # Original: 32
    
    # NEW: Laplacian penalty for spatial smoothness
    laplacian_penalty = float(train_cfg.get('laplacian_penalty', 0.0))  # Default: disabled
    
    # NEW: truth_uncertain - whether to account for observation measurement error in loss
    # When true, observations are treated as having uncertainty (obs_err_std)
    # When false (default), observations are treated as exact
    truth_uncertain = bool(train_cfg.get('truth_uncertain', False))
    
    # NEW: Skip connections configuration
    # Default: all levels except first (2:enc_levels+1 in Julia 1-indexing) - matches DINCAE default
    skip_connections_str = train_cfg.get('skipconnections', None)
    if skip_connections_str is None:
        skip_connections_julia = f"2:{enc_levels + 1}"
    else:
        # Allow explicit specification like "2:5" or [2,3,4,5]
        if isinstance(skip_connections_str, list):
            skip_connections_julia = "[" + ",".join(str(s) for s in skip_connections_str) + "]"
        else:
            skip_connections_julia = str(skip_connections_str)
    
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

# --- Training parameters from config ---
# Core parameters
epochs      = {epochs}
batch       = {batch_size}
ntime_win   = {ntime_win}
lr          = {learning_rate}
enc_levels  = {enc_levels}
save_int    = {save_epochs_interval}

# Error parameters
obs_err_std   = {obs_err_std}    # Observation uncertainty in cost function
min_std_err   = {min_std_err}    # Minimum reconstruction std error

# Advanced parameters (now configurable via JSON)
clip_grad              = {clip_grad}
upsampling_method      = :{upsampling_method}
regularization_L2_beta = {regularization_L2_beta}
jitter_std             = {jitter_std}
loss_weights_refine    = {loss_weights_str}
laplacian_penalty      = {laplacian_penalty}
truth_uncertain        = {str(truth_uncertain).lower()}
enc_nfilter_base       = {enc_nfilter_base}
skipconnections        = {skip_connections_julia}

# SLURM config for plotting job
slurm_partition = "{cfg.get('slurm', {}).get('partition', 'orchid')}"
slurm_account   = "{cfg.get('slurm', {}).get('account', 'orchid')}"
slurm_qos       = "{cfg.get('slurm', {}).get('qos', 'orchid')}"
slurm_gpus      = {int(cfg.get('slurm', {}).get('gpus', 1))}
slurm_cpus      = {int(cfg.get('slurm', {}).get('cpus', 4))}
slurm_mem       = "{cfg.get('slurm', {}).get('mem', '128G')}"
slurm_exclude   = "{cfg.get('slurm', {}).get('exclude', 'gpuhost007,gpuhost012,gpuhost016')}"
julia_exe       = "{cfg.get('runner', {}).get('julia_exe', 'julia')}"
conda_activate  = "{cfg.get('env', {}).get('dincae', {}).get('activate', '')}"
julia_depot     = "{cfg.get('runner', {}).get('julia_depot', '$HOME/.julia_cuda_depot')}"

save_epochs = collect(save_int:save_int:epochs)
Atype = CuArray{{Float32}}

# Log all parameters for reproducibility
@info "DINCAE Configuration" epochs=epochs batch=batch ntime_win=ntime_win lr=lr enc_levels=enc_levels
@info "Error parameters" obs_err_std=obs_err_std min_std_err=min_std_err truth_uncertain=truth_uncertain
@info "Advanced parameters" clip_grad=clip_grad upsampling_method=upsampling_method regularization_L2_beta=regularization_L2_beta
@info "Architecture" loss_weights_refine=loss_weights_refine laplacian_penalty=laplacian_penalty skipconnections=skipconnections

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
    jitter_std  = jitter_std,
    isoutput    = true
)]
data_all = [data, data]
fname_rec = joinpath(outdir, "data-avg.nc")

println("="^60)
println("DINCAE reconstruction starting…")
println("epochs=",epochs, " batch=",batch, " ntime_win=",ntime_win, " lr=",lr, " enc_levels=",enc_levels)
println("obs_err_std=",obs_err_std, " min_std_err=",min_std_err)
println("upsampling_method=",upsampling_method, " loss_weights_refine=",loss_weights_refine)
flush(stdout)

loss = DINCAE.reconstruct(
    Atype, data_all, [fname_rec];
    epochs               = epochs,
    batch_size           = batch,
    enc_nfilter_internal = round.(Int, enc_nfilter_base * 2 .^ (0:enc_levels-1)),
    clip_grad            = clip_grad,
    ntime_win            = ntime_win,
    upsampling_method    = upsampling_method,
    loss_weights_refine  = loss_weights_refine,
    min_std_err          = min_std_err,
    truth_uncertain      = truth_uncertain,
    regularization_L2_beta = regularization_L2_beta,
    laplacian_penalty    = laplacian_penalty,
    skipconnections      = skipconnections,
    save_epochs          = collect(save_epochs),
    learning_rate        = lr,
    modeldir             = outdir
)

open(joinpath(outdir, "loss_history.json"), "w") do io
    JSON.print(io, Dict("loss"=>loss, "epochs"=>collect(1:length(loss))))
end

# --- CV RMS computation ---
case = (
    fname_orig = fname_cleanup,  # cleaned base (no synthetic CV)
    fname_cv   = cv_clean,       # cleaned CV file (with synthetic CV clouds)
    varname    = varname,
)

cvrms = DINCAE_utils.cvrms(case, fname_rec)

open(joinpath(outdir, "cv_rms.txt"), "w") do io
    @printf(io, "CV_RMS %0.6f\\n", cvrms)
end

@info "CV_RMS: $cvrms"

# --- Optional: Spawn background plotting (non-blocking) ---
enable_plotting = {str(cfg.get('post', {}).get('enable_cv_plots', False)).lower()}
if enable_plotting
    figdir = joinpath(outdir, "Fig")
    mkpath(figdir)
    plot_log = joinpath(outdir, "plotting.log")
    plot_script = joinpath(outdir, "run_cv_plots.jl")
    
    # Write plotting script to file
    open(plot_script, "w") do io
        println(io, "using Pkg")
        jp = get(ENV, "JULIA_PROJECT", "")
        if !isempty(jp)
            println(io, "Pkg.activate(\\"$jp\\")")
        end
        println(io, "using DINCAE_utils, PyPlot")
        println(io, "case = (fname_orig=\\"$fname_cleanup\\", fname_cv=\\"$cv_clean\\", varname=\\"$varname\\")")
        println(io, "@info \\"Starting CV plots...\\"")
        println(io, "DINCAE_utils.plotres(case, \\"$fname_rec\\"; clim=nothing, figdir=\\"$figdir\\", which_plot=:cv)")
        println(io, "@info \\"Plotting complete. Figures saved to: $figdir\\"")
    end
    
    @info "Spawning background plotting"
    @info "  Script: $plot_script"
    @info "  Log: $plot_log"
    @info "  To run manually: $julia_exe $plot_script"
    
    # Submit as separate SLURM job (uses same config as main DINCAE job)
    slurm_script = joinpath(outdir, "run_cv_plots.slurm")
    open(slurm_script, "w") do io
        println(io, "#!/bin/bash")
        println(io, "#SBATCH --job-name=cv_plots")
        println(io, "#SBATCH --output=$plot_log")
        println(io, "#SBATCH --error=$plot_log")
        println(io, "#SBATCH --time=01:00:00")
        println(io, "#SBATCH --mem=$slurm_mem")
        println(io, "#SBATCH --partition=$slurm_partition")
        println(io, "#SBATCH --account=$slurm_account")
        println(io, "#SBATCH --qos=$slurm_qos")
        println(io, "#SBATCH --gres=gpu:$slurm_gpus")
        println(io, "#SBATCH --cpus-per-task=$slurm_cpus")
        if !isempty(slurm_exclude)
            println(io, "#SBATCH --exclude=$slurm_exclude")
        end
        println(io, "#SBATCH --chdir=$outdir")
        println(io, "")
        # Environment setup (same as main job)
        if !isempty(conda_activate)
            println(io, conda_activate)
        end
        println(io, "export JULIA_DEPOT_PATH=\\"$julia_depot\\"")
        println(io, "export CUDA_DEVICE_ORDER=PCI_BUS_ID")
        println(io, "export CUDA_PATH=\\"\\${{CONDA_PREFIX}}\\"")
        println(io, "")
        println(io, "$julia_exe $plot_script")
    end
    
    try
        run(`sbatch $slurm_script`)
        @info "  Submitted plotting as SLURM job"
    catch e
        @warn "Could not submit SLURM job for plotting" exception=e
        @info "  Run manually: $julia_exe $plot_script"
    end
else
    @info "CV plotting disabled (enable with dincae.post.enable_cv_plots: true)"
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
    
    # Use JULIA_PROJECT from config if specified
    julia_project = cfg.get("runner", {}).get("JULIA_PROJECT")
    if julia_project:
        env_lines.append(f'export JULIA_PROJECT="{julia_project}"')
    
    # Use a shared depot for julia-cuda environment, separate from $HOME/.julia
    # This avoids CUDA version conflicts with packages compiled against older CUDA
    # The depot can be configured via runner.julia_depot, defaults to ~/.julia_cuda_depot
    shared_depot = cfg.get("runner", {}).get("julia_depot", "$HOME/.julia_cuda_depot")
    env_lines += [
        "export JULIA_PKG_PRECOMPILE_AUTO=0",
        # Use shared depot for julia-cuda, do NOT fall back to $HOME/.julia
        f'export JULIA_DEPOT_PATH="{shared_depot}"',
        f'mkdir -p "{shared_depot}"',
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        # Ensure conda's CUDA is used
        'export CUDA_PATH="${CONDA_PREFIX}"',
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
echo "JULIA_PROJECT=${{JULIA_PROJECT:-not set}}"
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-not set}}"
echo "CONDA_PREFIX=${{CONDA_PREFIX:-not set}}"
echo "CUDA_PATH=${{CUDA_PATH:-not set}}"
echo ""

# === Check CUDA version from Julia ===
echo "=== Checking CUDA version ==="
{julia} -e 'using CUDA; CUDA.versioninfo()' || echo "Warning: CUDA check failed"
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