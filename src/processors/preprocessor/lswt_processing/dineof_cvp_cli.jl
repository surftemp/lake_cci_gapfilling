# dineof_cvp_cli.jl
using NCDatasets
using Missings
using Statistics
using Random
# PyPlot / Dates not actually needed for CV, but harmless if you want them:
# using PyPlot
# using Dates

# brings in nanmean, dineof_cvp, coverage methods
include("dineof_scripts.jl")

fname     = ARGS[1]           # "/path/prepared.nc#lake_surface_water_temperature"
maskfname = ARGS[2]           # "/path/prepared.nc#lakeid"
outdir    = ARGS[3]           # "/path/prepared/000003007/"
nbclean   = parse(Int, ARGS[4])

# Optional arguments with defaults
seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
min_cloud_frac = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.05
max_cloud_frac = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.70
min_obs_frac = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 0.05  # NEW PARAMETER

println("Calling dineof_cvp with:")
println("  fname=$fname")
println("  maskfname=$maskfname")
println("  outdir=$outdir")
println("  nbclean=$nbclean")
println("  seed=$seed")
println("  min_cloud_frac=$min_cloud_frac")
println("  max_cloud_frac=$max_cloud_frac")
println("  min_obs_frac=$min_obs_frac")

dineof_cvp(fname, maskfname, outdir, nbclean; 
           seed=seed, 
           min_cloud_frac=min_cloud_frac, 
           max_cloud_frac=max_cloud_frac,
           min_obs_frac=min_obs_frac)
