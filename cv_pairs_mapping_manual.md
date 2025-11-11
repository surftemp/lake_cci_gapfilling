# DINEOF CV Pairs Mapping Manual

## Overview

This document describes how cross-validation pairs stored in `cv_pairs.nc` map to 3D array coordinates `(time, lat, lon)` in the prepared dataset.

---

## File Structure

**cv_pairs.nc format:**
```
dimensions:
    index = 2
    nbpoints = N  (number of CV pairs)
    
variables:
    int cv_pairs(index, nbpoints)
```

**Data layout:**
- `cv_pairs(1, k)` = spatial index m
- `cv_pairs(2, k)` = temporal index t
- Where k = 1..N (each CV pair)

---

## DINEOF's Interpretation

### Step 1: Dimension Reversal

When DINEOF reads the NetCDF file using old Fortran NetCDF library:
- **NetCDF declares:** `cv_pairs(index=2, nbpoints=N)`
- **Fortran receives:** `cvp_indexes(N, 2)` ← **dimensions are reversed**

### Step 2: Reading CV Pairs

For each pair k (1..N):
```fortran
i = cvp_indexes(k, 1)  ! spatial index m
j = cvp_indexes(k, 2)  ! temporal index t
```

- Column 1: spatial indices (m)
- Column 2: temporal indices (t)

---

## Spatial Index Mapping

### Dataset Dimensions

**DINEOF's prepared.nc:**
- `imax = 147` → LON dimension
- `jmax = 152` → LAT dimension
- `nlines = 427` → TIME dimension

### Spatial Enumeration Order: LON-OUTER, LAT-INNER

DINEOF flattens the 2D spatial grid into 1D spatial index using:

```
m = 0
for i = 1 to 147 (LON):
    for j = 1 to 152 (LAT):
        if pixel is valid:
            m = m + 1
            # spatial index m now maps to (i, j)
```

**Key insight:** `i` iterates in the outer loop (LON), `j` in the inner loop (LAT).

### Example Mappings

From DINEOF debug output for this lake:
- m=1 → i=70, j=69
- m=2 → i=70, j=70
- m=8 → i=73, j=72
- m=10 → i=73, j=75

Where:
- i = LON coordinate
- j = LAT coordinate

---

## Complete 3D Coordinate Mapping

### Mapping Chain

```
cv_pairs.nc entry: (m, t)
                    ↓
DINEOF spatial map: m → (i, j)
                    ↓
3D coordinates: (i, j, t) = (LON, LAT, TIME)
```

### Indexing Conventions

- **Fortran (DINEOF):** 1-based indexing (indices start at 1)
- **Python:** 0-based indexing (indices start at 0)

### Conversion Formula

For a CV pair with spatial index m and temporal index t:

**1-based (Fortran/DINEOF):**
- LON = i (obtained from spatial mapping)
- LAT = j (obtained from spatial mapping)
- TIME = t

**0-based (Python):**
- lon_idx = i - 1
- lat_idx = j - 1
- time_idx = t - 1

---

## Complete Example: CV Pair 1

### From cv_pairs.nc
- Spatial index: m = 8
- Temporal index: t = 94

### DINEOF Spatial Mapping
- m=8 → i=73, j=72 (from debug output)

### 3D Coordinates (1-based Fortran)
- LON = 73
- LAT = 72
- TIME = 94

### 3D Coordinates (0-based Python)
- lon_idx = 72
- lat_idx = 71
- time_idx = 93

### Accessing in Python
```python
# Using numeric indices
value = dataset['lake_surface_water_temperature'][93, 71, 72]

# Using labeled coordinates (if available)
value = dataset['lake_surface_water_temperature'].isel(
    time=93, 
    lat=71, 
    lon=72
)
```

---

## Summary of Key Points

1. **Dimension Reversal:** NetCDF dimensions are reversed when read by Fortran NetCDF library
   - Python saves: `(index=2, nbpoints=N)`
   - Fortran reads: `(N, 2)`

2. **Spatial Enumeration:** DINEOF uses **LON-OUTER, LAT-INNER** ordering
   - First dimension (i) = LON
   - Second dimension (j) = LAT

3. **Index Convention:** All indices in cv_pairs.nc are **1-based** (Fortran convention)

4. **Temporal Indexing:** 
   - The temporal index t stored in cv_pairs.nc is the frame to be masked
   - DINEOF internally uses both frame t and t-1 during reconstruction
   - Your Python validation code must ensure both S[t] and S[t-1] are valid before including a CV pair

5. **Critical Requirement:** Python spatial indexing code **MUST** use the same LON-OUTER, LAT-INNER enumeration order to match DINEOF's expectations

---

## Implications for DINCAE

When creating independent CV masks for DINCAE that must match DINEOF's CV points:

1. Use the cv_pairs.nc file to read (m, t) pairs
2. Apply the same spatial mapping: m → (lon, lat) using LON-OUTER, LAT-INNER
3. Convert to 0-based indexing for Python/DINCAE
4. Mask the corresponding pixels at those exact coordinates

This ensures DINCAE and DINEOF mask **identical pixels** for fair comparison.

---

**Document Version:** 1.0  
**Date:** 2025-11-11  
**Lake ID Tested:** 3007
