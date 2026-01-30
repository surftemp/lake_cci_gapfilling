"""

function dineof_cvp(fname,maskfname,outdir,nbclean; seed=nothing, min_cloud_frac=0.05, max_cloud_frac=0.70, min_obs_frac=0.05)
 Extracts clouds from file 'fname' and adds them to the 'nbclean'
   cleanest images from the data set, ensuring resulting frames maintain
   at least min_obs_frac observation coverage.

 Input variables:
       fname (string): file name in the disk (gher format)
       maskfname (string): mask file name in teh disk (gher format)
       outdir (string): directory where new file will be written
       nbclean (integer): number of cleanest images to be covered with clouds
       seed (integer, optional): random seed for reproducibility
       min_cloud_frac (float): minimum cloud fraction for donor candidates (default 0.05)
       max_cloud_frac (float): maximum cloud fraction for donor candidates (default 0.70)
       min_obs_frac (float): minimum observation fraction after CV withholding (default 0.05)
  
 File with added clouds will be written in a file with the same name
    as the initial file (and at the same location in the disk)
    but with the extension '.clouds' added. An additional file, 
    clouds_index is also written, which contains the indexes of 
    the added clouds (can be used as cross-validation points in 
    DINEOF)

 Reference:
 A. Alvera-Azcarate, A. Barth, M. Rixen, and J. M. Beckers. 
    Reconstruction of incomplete oceanographic data sets using 
    Empirical Orthogonal Functions. Application to the Adriatic 
    Sea Surface Temperature. Ocean Modelling, 9:325-346, 2005.

"""
function dineof_cvp(fname, maskfname, outdir, nbclean; 
                    seed=nothing, min_cloud_frac=0.05, max_cloud_frac=0.70, min_obs_frac=0.05)

    file, varname = split(fname, "#")
    @show file 
    ds = Dataset(String(file))
    tmp = ds[String(varname)][:,:,:]
    SST = nomissing(tmp, NaN)
    close(ds)

    mfile, mvarname = split(maskfname, "#")
    ds = Dataset(String(mfile))
    mask = ds[String(mvarname)][:,:]
    close(ds)

    # Apply land mask to SST
    for k = 1:size(SST, 3)
        tmp = SST[:,:,k]
        tmp[mask .== 0] .= NaN
        SST[:,:,k] = tmp
    end

    nbland = sum(mask[:] .== 0)
    mmax = sum(mask[:] .== 1)

    # Calculate cloud coverage for each timestep
    cloudcov = (sum(sum(isnan.(SST), dims=2), dims=1) .- nbland) / mmax
    cloudcov = cloudcov[:]

    # Set random seed for reproducibility
    if seed !== nothing
        Random.seed!(seed)
        println("Random seed set to: $seed")
    end

    println("========================================")
    println("CV Generation with Coverage Protection")
    println("========================================")
    println("min_obs_frac threshold: $min_obs_frac")
    println("Cloud coverage range in data: $(minimum(cloudcov)) to $(maximum(cloudcov))")

    # Filter candidate cloud sources by cloud fraction
    candidate_idx = findall((cloudcov .>= min_cloud_frac) .& (cloudcov .<= max_cloud_frac))
    println("Candidate timesteps with cloud frac in [$min_cloud_frac, $max_cloud_frac]: $(length(candidate_idx))")

    if length(candidate_idx) < nbclean
        println("WARNING: Not enough candidate timesteps ($(length(candidate_idx))) for nbclean=$nbclean")
        println("         Falling back to using all timesteps as candidates")
        candidate_idx = collect(1:length(cloudcov))
    end

    # Select the nbclean cleanest frames (lowest cloud coverage)
    clean = sortperm(cloudcov)
    clean = clean[1:nbclean]

    N = length(cloudcov)

    # Remove clean frames from candidate donors
    available_candidates = setdiff(candidate_idx, clean)

    println("Available donor candidates (excluding clean frames): $(length(available_candidates))")

    if length(available_candidates) == 0
        println("ERROR: No available donor candidates after excluding clean frames")
        println("       Creating empty clouds_index.nc")
        # Create empty output
        output = Dataset(joinpath(outdir, "clouds_index.nc"), "c")
        defDim(output, "nbpoints", 0)
        defDim(output, "index", 2)
        defDim(output, "indexcount", mmax)
        ncCloud = defVar(output, "clouds_index", Int64, ("nbpoints", "index"))
        
        # Still need iindex/jindex for spatial mapping
        imax = size(SST, 1)
        jmax = size(SST, 2)
        iindex = zeros(Int, mmax)
        jindex = zeros(Int, mmax)
        m = 0
        for i = 1:imax
            for j = 1:jmax
                if mask[i,j] == 1
                    m = m + 1
                    iindex[m] = i
                    jindex[m] = j
                end
            end
        end
        iCloud = defVar(output, "iindex", Int64, ("indexcount",))
        iCloud[:] = iindex
        jCloud = defVar(output, "jindex", Int64, ("indexcount",))
        jCloud[:] = jindex
        close(output)
        return
    end

    # Precompute lake mask for efficiency
    lake_mask_2d = (mask .== 1)

    # For each clean frame, find ALL valid donors, then randomly pick one
    accepted_clean = Int[]
    accepted_donor = Int[]
    donor_usage_count = Dict{Int, Int}()  # Track how many times each donor is used

    println("\nProcessing clean frames:")
    for i = 1:nbclean
        c = clean[i]
        clean_frame = SST[:,:,c]
        clean_obs_frac = 1.0 - cloudcov[c]  # observation fraction = 1 - cloud fraction
        clean_has_data = .!isnan.(clean_frame) .& lake_mask_2d
        current_valid = sum(clean_has_data)
        
        # Find ALL donors that maintain >= min_obs_frac
        valid_donors = Int[]
        valid_donors_resulting_frac = Float64[]
        
        for d in available_candidates
            if d == c
                continue  # skip if same as clean frame
            end
            
            donor_frame = SST[:,:,d]
            
            # Simulate applying this donor's cloud pattern
            # New NaNs = where clean has data AND donor has NaN AND within lake mask
            donor_has_cloud = isnan.(donor_frame) .& lake_mask_2d
            new_nans = clean_has_data .& donor_has_cloud
            
            # Calculate resulting observation count and fraction
            new_nan_count = sum(new_nans)
            resulting_valid = current_valid - new_nan_count
            resulting_frac = resulting_valid / mmax
            
            if resulting_frac >= min_obs_frac
                push!(valid_donors, d)
                push!(valid_donors_resulting_frac, resulting_frac)
            end
        end
        
        if length(valid_donors) > 0
            # Randomly select from valid donors
            chosen_idx = rand(1:length(valid_donors))
            chosen_donor = valid_donors[chosen_idx]
            chosen_resulting_frac = valid_donors_resulting_frac[chosen_idx]
            
            push!(accepted_clean, c)
            push!(accepted_donor, chosen_donor)
            
            # Track donor usage
            donor_usage_count[chosen_donor] = get(donor_usage_count, chosen_donor, 0) + 1
            
            println("  Frame $c (obs=$(round(clean_obs_frac, digits=3))): " *
                    "$(length(valid_donors)) valid donors, chose $chosen_donor " *
                    "(cloud=$(round(cloudcov[chosen_donor], digits=3))) → " *
                    "result obs=$(round(chosen_resulting_frac, digits=3)) ✓")
        else
            println("  Frame $c (obs=$(round(clean_obs_frac, digits=3))): " *
                    "NO valid donor found (all would drop below $min_obs_frac) ✗")
        end
    end

    n_accepted = length(accepted_clean)
    n_unique_donors = length(donor_usage_count)
    
    println("\n========================================")
    println("Accepted $n_accepted of $nbclean clean frames for CV")
    println("Used $n_unique_donors unique donor patterns")
    println("========================================")
    
    # Print donor usage distribution
    if n_unique_donors > 0
        println("\nDonor usage distribution:")
        usage_counts = sort(collect(values(donor_usage_count)), rev=true)
        println("  Max uses of single donor: $(maximum(usage_counts))")
        println("  Min uses of single donor: $(minimum(usage_counts))")
        println("  Mean uses per donor: $(round(mean(usage_counts), digits=2))")
    end

    if n_accepted == 0
        println("WARNING: No frames accepted for CV. Creating empty clouds_index.nc")
    end

    # Apply CV only to accepted pairs
    SST2 = copy(SST)
    for i = 1:n_accepted
        c = accepted_clean[i]
        d = accepted_donor[i]
        # Apply cloud pattern: divide by NOT(isnan) creates NaN where donor has clouds
        SST2[:,:,c] = SST[:,:,c] ./ .!isnan.(SST[:,:,d])
    end
    SST2[isinf.(SST2)] .= NaN

    # Build spatial index mapping
    imax = size(SST, 1)
    jmax = size(SST, 2)

    mindex = zeros(Int, imax, jmax)
    iindex = zeros(Int, mmax)
    jindex = zeros(Int, mmax)

    m = 0
    for i = 1:imax
        for j = 1:jmax
            if mask[i,j] == 1
                m = m + 1
                mindex[i,j] = m
                iindex[m] = i
                jindex[m] = j
            end
        end
    end

    # Find all CV points (where SST2 is NaN but SST was not)
    indexex = findall(isnan.(SST2) .& .!isnan.(SST))

    nbpoints = length(indexex)
    clouds_indexes = zeros(Int, nbpoints, 2)

    for l = 1:nbpoints
        iex = CartesianIndices(size(SST2))[indexex[l]]
        clouds_indexes[l, 1] = mindex[iex[1], iex[2]]
        clouds_indexes[l, 2] = iex[3]
    end

    # Calculate and report statistics
    nbgood = sum(.!isnan.(SST[:]))
    nbgood2 = sum(.!isnan.(SST2[:]))
    cv_percent = 100 * (nbgood - nbgood2) / nbgood

    println("\nCV Statistics:")
    println("  Total valid pixels before CV: $nbgood")
    println("  Total valid pixels after CV:  $nbgood2")
    println("  CV points created: $nbpoints")
    println("  Percentage of data withheld: $(round(cv_percent, digits=2))%")

    # Write output
    output = Dataset(joinpath(outdir, "clouds_index.nc"), "c")
    defDim(output, "nbpoints", size(clouds_indexes, 1))
    defDim(output, "index", size(clouds_indexes, 2))
    defDim(output, "indexcount", m)

    ncCloud = defVar(output, "clouds_index", Int64, ("nbpoints", "index"))
    ncCloud[:] = clouds_indexes

    iCloud = defVar(output, "iindex", Int64, ("indexcount",))
    iCloud[:] = iindex
    jCloud = defVar(output, "jindex", Int64, ("indexcount",))
    jCloud[:] = jindex

    # Add metadata attributes
    output.attrib["min_obs_frac"] = min_obs_frac
    output.attrib["min_cloud_frac"] = min_cloud_frac
    output.attrib["max_cloud_frac"] = max_cloud_frac
    output.attrib["nbclean_requested"] = nbclean
    output.attrib["nbclean_accepted"] = n_accepted
    output.attrib["n_unique_donors"] = n_unique_donors
    output.attrib["cv_fraction"] = cv_percent / 100.0

    close(output)

    println("\nOutput written to: $(joinpath(outdir, "clouds_index.nc"))")

end
