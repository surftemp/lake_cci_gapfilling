# dineof_cvp_cli.jl
using NCDatasets
using Missings
using Statistics
# PyPlot / Dates not actually needed for CV, but harmless if you want them:
# using PyPlot
# using Dates

# brings in nanmean, dineof_cvp, coverage methods
include("dineof_scripts.jl")

fname     = ARGS[1]           # "/path/prepared.nc#lake_surface_water_temperature"
maskfname = ARGS[2]           # "/path/prepared.nc#lakeid"
outdir    = ARGS[3]           # "/path/prepared/000003007/"
nbclean   = parse(Int, ARGS[4])

println("Calling dineof_cvp(fname=$fname, maskfname=$maskfname, outdir=$outdir, nbclean=$nbclean)")
dineof_cvp(fname, maskfname, outdir, nbclean)
