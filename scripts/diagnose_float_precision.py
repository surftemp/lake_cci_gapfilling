#!/usr/bin/env python3
"""
Diagnostic script to understand the float32/float64 precision issue
in _find_nearest_grid_point.

This tests the EXACT same operations that happen in insitu_validation.py
to understand why backup gets correct results and new code doesn't.

Usage:
    python diagnose_float_precision.py
"""

import numpy as np
import xarray as xr
import pandas as pd

# Paths
PREPARED_NC = '/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/archive/anomaly-20251215-8ea02d-exp3/prepared/000000044/prepared.nc'
DINEOF_NC = '/gws/ssde/j25b/cds_c3s_lakes/users/SHAERDAN/archive/anomaly-20251215-8ea02d-exp3/post/000000044/a1000/LAKE000000044-CCI-L3S-LSWT-CDR-4.5-filled_fine_dineof.nc'
SELECTION_CSV = '/home/users/shaerdan/general_purposes/insitu_cv/L3S_QL_MDB_2010_selection.csv'

def test_find_nearest_grid_point():
    """Test the exact operations in _find_nearest_grid_point"""
    
    print("=" * 70)
    print("DIAGNOSTIC: Float precision in _find_nearest_grid_point")
    print("=" * 70)
    
    # Load target coordinates from CSV (same as both versions do)
    df = pd.read_csv(SELECTION_CSV)
    lake_df = df[df['lake_id_cci'] == 44]
    sites = lake_df[['latitude', 'longitude']].drop_duplicates().iloc[0]
    target_lat = sites['latitude']
    target_lon = sites['longitude']
    
    print(f"\n1. TARGET COORDINATES (from CSV)")
    print(f"   target_lat = {target_lat}")
    print(f"   target_lon = {target_lon}")
    print(f"   target_lat type = {type(target_lat)}")
    print(f"   (numpy type: {type(np.float64(target_lat))})")
    
    # Load lat/lon arrays from NetCDF
    ds = xr.open_dataset(PREPARED_NC, decode_times=False)
    lat_array = ds['lat'].values
    lon_array = ds['lon'].values
    ds.close()
    
    print(f"\n2. COORDINATE ARRAYS (from NetCDF)")
    print(f"   lat_array.dtype = {lat_array.dtype}")
    print(f"   lon_array.dtype = {lon_array.dtype}")
    print(f"   lat_array[163] = {lat_array[163]}")
    print(f"   lat_array[164] = {lat_array[164]}")
    print(f"   lon_array[155] = {lon_array[155]}")
    
    print(f"\n3. MESHGRID OPERATION")
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
    print(f"   lat_grid.dtype = {lat_grid.dtype}")
    print(f"   lon_grid.dtype = {lon_grid.dtype}")
    
    print(f"\n4. SUBTRACTION (lat_grid - target_lat)")
    diff_lat = lat_grid - target_lat
    print(f"   Result dtype = {diff_lat.dtype}")
    print(f"   diff_lat[163,155] = {diff_lat[163, 155]:.20f}")
    print(f"   diff_lat[164,155] = {diff_lat[164, 155]:.20f}")
    
    print(f"\n5. DISTANCE CALCULATION")
    distance = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    print(f"   distance.dtype = {distance.dtype}")
    print(f"   distance[163,155] = {distance[163, 155]:.15f}")
    print(f"   distance[164,155] = {distance[164, 155]:.15f}")
    print(f"   difference = {distance[163, 155] - distance[164, 155]:.20f}")
    
    print(f"\n6. ARGMIN RESULT")
    index = np.unravel_index(np.argmin(distance), distance.shape)
    print(f"   Selected pixel: {index}")
    
    # Now test with explicit float64 conversion
    print(f"\n" + "=" * 70)
    print("COMPARISON: With explicit float64 conversion")
    print("=" * 70)
    
    lat_array_64 = lat_array.astype(np.float64)
    lon_array_64 = lon_array.astype(np.float64)
    
    print(f"\n   lat_array_64.dtype = {lat_array_64.dtype}")
    
    lon_grid_64, lat_grid_64 = np.meshgrid(lon_array_64, lat_array_64)
    print(f"   lat_grid_64.dtype = {lat_grid_64.dtype}")
    
    diff_lat_64 = lat_grid_64 - target_lat
    print(f"   (lat_grid_64 - target_lat).dtype = {diff_lat_64.dtype}")
    
    distance_64 = np.sqrt((lat_grid_64 - target_lat)**2 + (lon_grid_64 - target_lon)**2)
    print(f"\n   distance_64.dtype = {distance_64.dtype}")
    print(f"   distance_64[163,155] = {distance_64[163, 155]:.15f}")
    print(f"   distance_64[164,155] = {distance_64[164, 155]:.15f}")
    print(f"   difference = {distance_64[163, 155] - distance_64[164, 155]:.20f}")
    
    index_64 = np.unravel_index(np.argmin(distance_64), distance_64.shape)
    print(f"\n   Selected pixel: {index_64}")
    
    # Check numpy version
    print(f"\n" + "=" * 70)
    print("ENVIRONMENT INFO")
    print("=" * 70)
    print(f"   numpy version: {np.__version__}")
    print(f"   xarray version: {xr.__version__}")
    print(f"   pandas version: {pd.__version__}")
    
    # The key question: why does float32 - float64 not upcast?
    print(f"\n" + "=" * 70)
    print("DTYPE PROMOTION TEST")
    print("=" * 70)
    
    f32 = np.float32(49.637500762939453)
    f64 = 49.64166666666665150842
    
    print(f"   np.float32 value: {f32}")
    print(f"   Python float value: {f64}")
    print(f"   f32 - f64 = {f32 - f64}")
    print(f"   type(f32 - f64) = {type(f32 - f64)}")
    print(f"   np.result_type(f32, f64) = {np.result_type(f32, f64)}")
    
    # Test with array
    arr32 = np.array([49.637500762939453, 49.645832061767578], dtype=np.float32)
    print(f"\n   arr32.dtype = {arr32.dtype}")
    print(f"   (arr32 - f64).dtype = {(arr32 - f64).dtype}")
    print(f"   arr32 - f64 = {arr32 - f64}")


def test_pandas_csv_types():
    """Check if pandas reads CSV coordinates as float32 or float64"""
    print(f"\n" + "=" * 70)
    print("PANDAS CSV DTYPE CHECK")
    print("=" * 70)
    
    df = pd.read_csv(SELECTION_CSV)
    lake_df = df[df['lake_id_cci'] == 44]
    
    print(f"   latitude column dtype: {lake_df['latitude'].dtype}")
    print(f"   longitude column dtype: {lake_df['longitude'].dtype}")
    
    lat_val = lake_df['latitude'].iloc[0]
    print(f"   lat_val = {lat_val}")
    print(f"   type(lat_val) = {type(lat_val)}")


if __name__ == "__main__":
    test_find_nearest_grid_point()
    test_pandas_csv_types()
