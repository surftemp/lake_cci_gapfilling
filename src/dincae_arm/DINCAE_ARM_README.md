# DINCAE Lake Temperature Gap-Filling Pipeline

## Overview

This pipeline processes lake surface water temperature data using DINCAE (Data-Interpolating Convolutional Auto-Encoder) for gap-filling satellite observations. The pipeline handles preprocessing, cross-validation setup, neural network training, and postprocessing.

---

## Architecture

```
Input: prepared.nc
    ↓
┌─────────────────────────────────────────────────────────┐
│  PREPROCESSING (dincae_adapter_in.py)                   │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 1: Time Conversion                           │ │
│  │  - Convert integer days → datetime64               │ │
│  │  Output: prepared_datetime.nc                      │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 2: Spatial Cropping                          │ │
│  │  - Find lake bounding box from lakeid mask         │ │
│  │  - Crop to bbox + buffer pixels                    │ │
│  │  - Store crop metadata for later uncropping        │ │
│  │  Output: prepared_datetime_cropped.nc              │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 3: Mask & CV Generation                      │ │
│  │  - Compute data availability mask (sea/land)       │ │
│  │  - Add mask & count_nomissing variables            │ │
│  │  - Generate cross-validation cloud patterns        │ │
│  │  Outputs:                                           │ │
│  │  - prepared_datetime_cropped_add_clouds.nc (CV)    │ │
│  │  - prepared_datetime_cropped_add_clouds.clean.nc   │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  DINCAE TRAINING (dincae_runner.py → Julia)             │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 4a: Data Cleaning (Julia)                    │ │
│  │  - Load base file → clean NaN/Inf → missing        │ │
│  │  - Save as lake_cleanup.nc                         │ │
│  │  - Load CV file → clean → .clean.nc                │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 4b: Neural Network Training                  │ │
│  │  - Load cleaned CV data                            │ │
│  │  - Train DINCAE encoder-decoder on GPU             │ │
│  │  - Save model checkpoints every N epochs           │ │
│  │  - Generate reconstructed data                     │ │
│  │  Output: data-avg.nc (gap-filled, cropped)         │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 4c: Cross-Validation                         │ │
│  │  - Compare predictions vs held-out observations    │ │
│  │  - Compute CV-RMS error                            │ │
│  │  Output: cv_rms.txt                                │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  POSTPROCESSING (dincae_adapter_out.py)                 │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 5: Spatial Uncropping                        │ │
│  │  - Restore to original grid dimensions             │ │
│  │  - Pad with NaN outside lake region                │ │
│  │  Output: data-avg-full.nc                          │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Processing Steps

### Step 1: Time Conversion (`convert_time`)

**Purpose**: Convert time coordinate from integer days to datetime64 format

**Input**: `prepared.nc`
- Time as integer/float days since epoch
- Example: `time = [8400, 8401, 8402, ...]`

**Process**:
```python
def convert_time(in_nc, out_nc, epoch="1981-01-01T12:00:00Z"):
    # Parse epoch to datetime64
    base = np.datetime64(epoch, "ns")
    
    # Convert integer days to timedelta
    time_vals = time_data.values.astype(float)
    datetime_vals = base + (time_vals * np.timedelta64(1, "D"))
    
    # Replace time coordinate
    ds = ds.assign_coords(time=datetime_vals)
```

**Output**: `prepared_datetime.nc`
- Time as datetime64: `time = [2004-01-01, 2004-01-02, ...]`

**Why needed**: DINCAE expects datetime64 format for temporal operations

---

### Step 2: Spatial Cropping (`crop_to_mask`)

**Purpose**: Reduce spatial dimensions to lake extent + buffer

**Input**: `prepared_datetime.nc`
- Full grid (e.g., 154 × 165 pixels)
- `lakeid` mask (1=lake, 0=land)

**Process**:
```python
def crop_to_mask(in_nc, out_nc, buffer=4):
    # Find bounding box of lake pixels
    ys, xs = np.where(lakeid == 1)
    i0, i1 = ys.min(), ys.max() + 1
    j0, j1 = xs.min(), xs.max() + 1
    
    # Add buffer
    i0 = max(0, i0 - buffer)
    i1 = min(grid_height, i1 + buffer)
    # ... same for j0, j1
    
    # Crop dataset
    ds_crop = ds.isel(lat=slice(i0, i1), lon=slice(j0, j1))
    
    # Store crop metadata
    ds_crop.attrs["crop_i0"] = i0
    # ... other metadata
```

**Output**: `prepared_datetime_cropped.nc`
- Reduced spatial dimensions (e.g., 31 × 43 pixels)
- ~95% size reduction
- Metadata for later uncropping

**Why needed**:
- Speeds up DINCAE training
- Reduces memory usage
- Focuses on relevant pixels

**Buffer pixels**: Extra pixels around lake to capture edge effects (default: 4)

---

### Step 3: Mask & CV Generation (`add_cv_clouds`)

**Purpose**: 
1. Create land/sea mask for DINCAE
2. Generate cross-validation dataset

**Input**: `prepared_datetime_cropped.nc`

#### Step 3a: Compute Data Availability Mask

```python
def add_cv_clouds(..., minseafrac=0.0):
    # Count valid observations per pixel
    var_data = ds["lake_surface_water_temperature"].values  # (time, lat, lon)
    count_nomissing = np.sum(~np.isnan(var_data), axis=0)  # (lat, lon)
    
    # Calculate fraction of valid data
    n_time = var_data.shape[0]
    frac_valid = count_nomissing / n_time
    
    # Create mask: 1 where sufficient data, 0 otherwise
    mask = (frac_valid > minseafrac).astype(np.int8)
    
    # Add to dataset
    ds['mask'] = xr.DataArray(mask, dims=('lat', 'lon'))
    ds['count_nomissing'] = xr.DataArray(count_nomissing, dims=('lat', 'lon'))
```

**Mask convention**:
- `mask = 1`: "sea" (actually lake) - pixels with sufficient data
- `mask = 0`: "land" - pixels with insufficient data
- `minseafrac`: minimum fraction of valid data (default: 0.0 = all pixels)

**Why needed**: 
- DINCAE needs to know which pixels to process
- Prevents training on pixels with no/little data
- Used in loss calculation

#### Step 3b: Generate Cross-Validation Points

```python
def add_cv_clouds(..., cv_fraction=0.1):
    # Select random time steps (10% by default)
    ntime = ds.sizes["time"]
    k = int(round(cv_fraction * ntime))
    sel_idx = rng.choice(ntime, size=k, replace=False)
    
    # Mask ALL lake pixels in selected timesteps
    var_mask = xr.zeros_like(var, dtype=bool)
    var_mask.loc[dict(time=ds["time"].isel(time=sel_idx))] = True
    
    # Apply mask (set to NaN)
    lake_mask = (ds["lakeid"] == 1)
    masked_var = var.where(~(var_mask & lake_mask), other=np.nan)
```

**Outputs**:
1. `prepared_datetime_cropped_add_clouds.clean.nc`
   - **WITH** mask & count_nomissing
   - **WITHOUT** CV masking
   - Used as reference

2. `prepared_datetime_cropped_add_clouds.nc`
   - **WITH** mask & count_nomissing
   - **WITH** CV masking (artificial clouds)
   - Passed to DINCAE for training

**Why needed**:
- Validates DINCAE's gap-filling ability
- Provides independent test set
- CV-RMS measures reconstruction quality

**Note on CV algorithm**:
- New pipeline: Simple random timestep selection
- Old pipeline: Complex pattern-based selection (more realistic)
- Trade-off: Simplicity vs realism

---

### Step 4a: Data Cleaning in Julia

**Purpose**: Convert NaN/Inf to proper Julia `missing` type

**Input**: 
- `prepared_datetime_cropped.nc` (base)
- `prepared_datetime_cropped_add_clouds.nc` (CV)

**Process**:
```julia
# Load base file
A = ds[varname][:,:,:]

# Convert NaN/Inf → missing
A_clean = map(x -> (!ismissing(x) && isfinite(x)) ? x : missing, A)

# Permute to (lon, lat, time)
A_llt = permutedims(A_clean, (lon_idx, lat_idx, time_idx))

# Write to lake_cleanup.nc
ds_cleanup = NCDataset("lake_cleanup.nc", "c")
write(ds_cleanup, ds_base; exclude=[varname])
defVar(ds_cleanup, varname, A_llt, ("lon", "lat", "time"))

# Repeat for CV file → .clean.nc
```

**Outputs**:
- `lake_cleanup.nc`: Cleaned base data
- `prepared_datetime_cropped_add_clouds.clean.nc`: Cleaned CV data

**Why needed**:
- Julia DINCAE requires proper `missing` type (not NaN)
- Ensures dimension ordering is correct
- Creates consistent data structure

---

### Step 4b: DINCAE Training

**Purpose**: Train neural network to reconstruct gaps

**Input**: `prepared_datetime_cropped_add_clouds.clean.nc`

**Architecture**:
```
Encoder (Downsampling)          Decoder (Upsampling)
┌──────────────────┐           ┌──────────────────┐
│  Conv + ReLU     │           │  Deconv + ReLU   │
│  32 filters      │◄─ skip ───┤  32 filters      │
├──────────────────┤           ├──────────────────┤
│  Conv + ReLU     │           │  Deconv + ReLU   │
│  64 filters      │◄─ skip ───┤  64 filters      │
├──────────────────┤           ├──────────────────┤
│  Conv + ReLU     │           │  Deconv + ReLU   │
│  128 filters     │◄─ skip ───┤  128 filters     │
└──────────────────┘           └──────────────────┘
        ↓                              ↑
   Latent Space                  Reconstruction
```

**Key Parameters**:
- `epochs`: Training iterations (default: 300)
- `batch_size`: Time steps per batch (default: 32)
- `ntime_win`: Temporal context window (default: 1)
- `learning_rate`: Adam optimizer rate (default: 1e-4)
- `obs_err_std`: Observation error (default: 0.2)
- `enc_levels`: Encoder depth (default: 3)

**Training Process**:
```julia
loss = DINCAE.reconstruct(
    CuArray{Float32},  # GPU arrays
    data_all,          # Training + validation data
    ["data-avg.nc"];   # Output filename
    epochs=300,
    batch_size=32,
    ntime_win=1,
    learning_rate=1e-4,
    min_std_err=0.2,
    # ... other params
)
```

**Loss Function**:
```
L = L_data + λ * L_reg

L_data = Σ (obs - pred)² / obs_err_std²  (only at observation points)
L_reg  = L2 regularization on weights
```

**Output**: `data-avg.nc`
- Gap-filled temperature field
- Reconstruction uncertainty
- Mean field (if computed)

**Training Progress**:
```
epoch:     1 loss -0.9943 time: 128.2s
epoch:     2 loss -1.7541 time: 2.2s
epoch:     3 loss -1.8097 time: 2.3s
...
epoch:   300 loss -1.4631 time: 2.3s
```

**GPU Requirements**:
- NVIDIA GPU with CUDA support
- Minimum 8GB VRAM (A100 recommended)
- cuDNN library installed

---

### Step 4c: Cross-Validation

**Purpose**: Assess reconstruction quality

**Process**:
```julia
cvrms = DINCAE_utils.cvrms(
    (fname_orig = "lake_cleanup.nc",      # Original data
     fname_cv   = "...add_clouds.nc",     # CV masked data
     varname    = varname),
    "data-avg.nc"  # DINCAE reconstruction
)
```

**Calculation**:
```
CV-RMS = sqrt(mean((obs_cv - pred_cv)²))

where:
- obs_cv  = held-out observations (CV points)
- pred_cv = DINCAE predictions at CV points
```

**Interpretation**:
- Lower CV-RMS = better reconstruction
- Typical values: 0.5 - 1.0 °C for good results
- Compare across different parameter settings

**Output**: `cv_rms.txt`
```
CV-RMS: 0.7033148
Lake ID: 1204
Epochs: 300
Batch size: 32
```

---

### Step 5: Spatial Uncropping

**Purpose**: Restore data to original grid dimensions

**Input**: 
- `data-avg.nc` (cropped, 31 × 43)
- `crop_metadata.npz` (crop indices)
- `prepared_datetime.nc` (original grid template)

**Process**:
```python
def uncrop_dataset(cropped_nc, template_nc, metadata):
    # Load crop indices
    i0, i1 = metadata["crop_i0"], metadata["crop_i1"]
    j0, j1 = metadata["crop_j0"], metadata["crop_j1"]
    
    # Create full grid filled with NaN
    full_grid = np.full(original_shape, np.nan)
    
    # Insert cropped data
    full_grid[:, i0:i1, j0:j1] = cropped_data
    
    # Restore coordinate arrays
    ds_full = xr.Dataset({
        "lake_surface_water_temperature": (["time", "lat", "lon"], full_grid),
        # ... other variables
    })
```

**Output**: `data-avg-full.nc`
- Original spatial dimensions (154 × 165)
- Gap-filled lake pixels
- NaN outside lake region

---

## Data Flow Summary

### File Dependencies

```
prepared.nc
    ↓ [convert_time]
prepared_datetime.nc
    ↓ [crop_to_mask]
prepared_datetime_cropped.nc
    ↓ [add_cv_clouds]
    ├─→ prepared_datetime_cropped_add_clouds.clean.nc (no CV mask)
    └─→ prepared_datetime_cropped_add_clouds.nc (with CV mask)
            ↓ [Julia cleaning]
            ├─→ lake_cleanup.nc
            └─→ prepared_datetime_cropped_add_clouds.clean.nc (cleaned)
                    ↓ [DINCAE.reconstruct]
                data-avg.nc
                    ↓ [uncrop_dataset]
                data-avg-full.nc
```

### Critical Variables in Files

#### prepared_datetime_cropped.nc
- `lake_surface_water_temperature(time, lat, lon)` - main variable
- `lakeid(lat, lon)` - lake mask
- `lat(lat)`, `lon(lon)`, `time(time)` - coordinates

#### prepared_datetime_cropped_add_clouds.clean.nc
- All of the above, PLUS:
- `mask(lat, lon)` ⚠️ **CRITICAL** - land/sea mask for DINCAE
- `count_nomissing(lat, lon)` - data availability count

#### data-avg.nc
- `lake_surface_water_temperature(time, lat, lon)` - gap-filled
- `lake_surface_water_temperature_error(time, lat, lon)` - uncertainty
- Same coordinates as cropped input

---

## Configuration Parameters

### Preprocessing (`dincae.crop`)
```json
{
  "epoch": "1981-01-01T12:00:00Z",  // Time origin
  "crop": {
    "buffer_pixels": 4  // Spatial buffer around lake
  }
}
```

### Cross-Validation (`dincae.cv`)
```json
{
  "cv": {
    "cv_fraction": 0.10,    // Fraction of data for CV (10%)
    "random_seed": 1234,    // For reproducibility
    "use_cv": true,         // Enable CV
    "minseafrac": 0.0       // Min data fraction for mask
  }
}
```

### Training (`dincae.train`)
```json
{
  "train": {
    "epochs": 300,              // Training iterations
    "batch_size": 32,           // Time steps per batch
    "ntime_win": 1,             // Temporal window
    "learning_rate": 0.0001,    // Adam LR
    "enc_levels": 3,            // Encoder depth
    "obs_err_std": 0.2,         // Observation error
    "save_epochs_interval": 10, // Checkpoint frequency
    "use_gpu": true             // Require GPU
  }
}
```

### Runtime (`dincae.runner`)
```json
{
  "runner": {
    "mode": "slurm",          // "local" or "slurm"
    "julia_exe": "julia",
    "skip_existing": true,
    "JULIA_PROJECT": "/path/to/julia/env"
  }
}
```

### SLURM (`dincae.slurm`)
```json
{
  "slurm": {
    "partition": "orchid",
    "gpus": 1,
    "cpus": 4,
    "time": "24:00:00",
    "mem": "128G",
    "account": "orchid",
    "qos": "orchid"
  }
}
```

---

## Common Issues & Solutions

### Issue 1: Dimension Mismatch Error
```
ERROR: DimensionMismatch: cannot create a BitMatrix from a 3-dimensional iterator
```

**Cause**: Missing `mask` variable in cleaned file

**Solution**: Ensure `add_cv_clouds()` creates mask before writing files

---

### Issue 2: B_llt Undefined
```
ERROR: UndefVarError: `B_llt` not defined
```

**Cause**: Julia script references variable before creating it

**Solution**: Use complete Julia script that loads and cleans data properly

---

### Issue 3: SLURM Container Error
```
container_p_join: open failed for /var/tmp/slurm_tmp/<jobid>/.ns
```

**Cause**: Broken compute node (e.g., gpuhost016)

**Solution**: Add `#SBATCH --exclude=gpuhost016` to SLURM script

---

### Issue 4: Out of GPU Memory
```
ERROR: CUDA.OutOfGPUMemoryError
```

**Causes**:
- Batch size too large
- Spatial dimensions too large
- Multiple jobs on same GPU

**Solutions**:
- Reduce `batch_size` (try 16 or 8)
- Increase spatial cropping (reduce buffer_pixels)
- Request exclusive GPU: `#SBATCH --exclusive`

---

### Issue 5: Poor CV-RMS
```
CV-RMS: 2.5 (very high)
```

**Possible causes**:
- Insufficient training (increase epochs)
- Learning rate too high/low
- Too much regularization
- Bad CV point selection

**Solutions**:
- Increase `epochs` to 500+
- Adjust `learning_rate` (try 5e-4 or 5e-5)
- Check `obs_err_std` (should match data uncertainty)
- Verify CV points are well-distributed

---

## Performance Benchmarks

### Lake 1204 (Medium lake, 215 pixels)

**Preprocessing**: ~30 seconds
- Time conversion: 5s
- Cropping: 2s
- CV generation: 3s

**DINCAE Training** (300 epochs, A100 GPU): ~15 minutes
- First epoch: 128s (compilation overhead)
- Subsequent epochs: 2-3s each
- Total: ~900s

**Memory Usage**:
- RAM: ~4 GB
- GPU VRAM: ~8 GB

**Output sizes**:
- Cropped data: ~50 MB
- Full data: ~200 MB
- Model checkpoints: ~500 MB total

### Scaling

For **N lakes** in parallel:
- Preprocessing: Negligible (fast)
- DINCAE: N × 15 min (if enough GPUs)
- Queue time: Depends on cluster load

**Recommended approach**:
- Process 10-20 lakes per GPU
- Use SLURM job arrays
- Monitor with `squeue` and log files

---

## Best Practices

### 1. Always Test on Single Lake First
```bash
# Test configuration
python your_pipeline.py --lake-id 1204 --test-mode
```

### 2. Monitor Training Progress
```bash
# Tail DINCAE output
tail -f /path/to/dincae/logs_dincae_1204.out

# Watch loss decrease
grep "epoch:" logs_dincae_1204.out | tail -20
```

### 3. Validate Outputs
```python
import xarray as xr
import numpy as np

# Check dimensions
ds = xr.open_dataset("data-avg.nc")
print(ds)

# Check for NaN propagation
lswt = ds["lake_surface_water_temperature"]
print(f"Missing: {lswt.isnull().sum().values}")

# Compare with input
ds_input = xr.open_dataset("prepared_datetime_cropped.nc")
reduction = (ds_input[var].isnull().sum() - lswt.isnull().sum()) / ds_input[var].size
print(f"Gap reduction: {reduction.values * 100:.1f}%")
```

### 4. Version Control Configurations
```bash
# Save configuration with timestamp
cp config.json configs/config_$(date +%Y%m%d_%H%M%S).json

# Log git commit in output
git rev-parse HEAD > output_dir/git_commit.txt
```

### 5. Document Parameter Choices
Create `experiment_notes.md` with:
- Why specific parameters chosen
- Expected behavior
- Comparison criteria
- Any deviations from standard settings

---

## Troubleshooting Checklist

Before asking for help, check:

- [ ] All input files exist and are readable
- [ ] `lakeid` variable is 2D (lat, lon)
- [ ] `mask` variable exists in .clean.nc files
- [ ] Time coordinate is datetime64 format
- [ ] GPU is accessible (`nvidia-smi` works)
- [ ] Julia environment is activated
- [ ] CUDA/cuDNN libraries are loaded
- [ ] Sufficient disk space (>10GB free)
- [ ] No jobs running on same GPU
- [ ] SLURM logs show actual error (not container error)

---

## References

### DINCAE Publications
- Barth, A., et al. (2020). "DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations"
- Barth, A., et al. (2022). "DINCAE 2.0: multivariate convolutional neural network with error estimates to reconstruct sea surface temperature satellite and altimetry observations"

### Implementation
- DINCAE.jl: https://github.com/gher-ulg/DINCAE.jl
- Documentation: https://gher-ulg.github.io/DINCAE.jl/

### JASMIN HPC
- ORCHID GPU cluster: https://help.jasmin.ac.uk/docs/interactive-computing/orchid/
- SLURM guide: https://help.jasmin.ac.uk/docs/batch-computing/slurm-scheduler/

---

## Contact & Support

For pipeline issues:
- Check this README first
- Review `NEW_PIPELINE_ISSUES.md` for known problems
- Check SLURM logs in `logs_dincae_*.err`

For DINCAE algorithm questions:
- See DINCAE.jl documentation
- Check DINCAE GitHub issues

For JASMIN/ORCHID issues:
- Email: support@jasmin.ac.uk
- Include job ID and error logs
