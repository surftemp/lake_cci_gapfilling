
import xarray as xr
import numpy as np
import argparse
import math
import os.path

class NewReconstructor(object):
    """
     Reconstruct and save eigenvalues, spatial and temporal EOFs to eofs.nc, reading the EOFs output from DINEOF output file dineof_output_eofs.nc
     (which is exported nby newer versions of DINEOF)
     """

    def __init__(self,mask_path,mask_variable,eof_folder,output_filename):
        self.eof_folder = eof_folder
        self.mask_path = mask_path
        self.mask_variable = mask_variable
        self.output_filename = output_filename

    def run(self):
        input_eofs = xr.open_dataset(os.path.join(self.eof_folder, "eof.nc"))
        ds = xr.open_dataset(self.mask_path)

        lat_values = ds.coords["lat"].data.tolist()
        lon_values = ds.coords["lon"].data.tolist()
        time_values = ds.coords["time"].data.tolist()
        lake_id = ds.attrs["lake_id"]

        eigenvalues = input_eofs["Sigma"].data
        spatial_eofs = input_eofs["Usst"].data
        temporal_eofs = input_eofs["V"].data

        variables = {}
        for eof_index in range(spatial_eofs.shape[0]):
            spatial_eof_data = spatial_eofs[eof_index,:,:]
            variables["spatial_eof%d"%eof_index] = xr.Variable(data=spatial_eof_data,dims=("y","x"))

        for eof_index in range(temporal_eofs.shape[0]):
            temporal_eof_data = temporal_eofs[eof_index,:]
            variables["temporal_eof%d"%eof_index] = xr.Variable(data=temporal_eof_data, dims=("t"))

        variables["eigenvalues"] =xr.Variable(data=eigenvalues, dims=("eofs"))

        out = xr.Dataset(data_vars=variables,coords={
            "lon":( ["x"], lon_values, ),
            "lat":( ["y"], lat_values, ),
            "time": (["t"], time_values, ),
            "eofs": (["eofs"], list(range(1,1+len(eigenvalues))), ),
        },attrs={"lake_id":lake_id})

        out.to_netcdf(os.path.join(self.eof_folder,self.output_filename))


class OldReconstructor(object):

    """
    Reconstruct and save eigenvalues, spatial and temporal EOFs to eofs.nc, reading the EOFs output from DINEOF to text files
    outputEof.lftvec, outputEof.rghvec, outputEof.vlsng.  These files are saved by older versions of DINEOF.
    """

    def __init__(self,mask_path,mask_variable,eof_folder,output_filename):
        self.eof_folder = eof_folder
        self.mask_path = mask_path
        self.mask_variable = mask_variable
        self.output_filename = output_filename
        self.spatial_eofs = []
        self.temporal_eofs = []
        self.eigenvalues = []

    def loadEofs(self):
        for (input_eofs,eof_filename) in [(self.spatial_eofs,"outputEof.lftvec"),(self.temporal_eofs,"outputEof.rghvec")]:
            f = open(os.path.join(self.eof_folder,eof_filename))
            count = 0
            for line in f.readlines():
                values = list(map(lambda x:float(x),line.strip().split()))
                if count == 0:
                    for idx in range(len(values)):
                        input_eofs.append([])
                count += 1
                for idx in range(len(values)):
                    input_eofs[idx].append(values[idx])

            print("Loaded %d eofs with length %d from %s"%(len(input_eofs),len(input_eofs[0]),eof_filename))

        f = open(os.path.join(self.eof_folder,"outputEof.vlsng"))
        for line in f.readlines():
            value = float(line.strip())
            self.eigenvalues.append(value)

        return self


    def run(self):

        ds = xr.open_dataset(self.mask_path)

        lat_values = ds.coords["lat"].data.tolist()
        lon_values = ds.coords["lon"].data.tolist()
        time_values = ds.coords["time"].data.tolist()
        lake_id = ds.attrs["lake_id"]

        masked = ds.variables[self.mask_variable][:,:]
        mshape = masked.shape
        count = 0
        (lats,lons) = mshape
        eofs = [np.array(np.zeros(mshape)) for f in self.spatial_eofs]
        for lon in range(lons):
            for lat in range(lats):
                mask = masked.data[lat,lon]
                for eof_index in range(len(self.spatial_eofs)):
                    if mask == 1:
                        eofs[eof_index][lat,lon] = self.spatial_eofs[eof_index][count]
                    else:
                        eofs[eof_index][lat,lon] = math.nan
                if mask == 1:
                    count += 1

        variables = {}
        for eof_index in range(len(self.spatial_eofs)):
            eof_data = eofs[eof_index]
            variables["spatial_eof%d"%eof_index] = xr.Variable(data=eof_data,dims=("y","x"))

        teofs = [np.array(t_eof) for t_eof in self.temporal_eofs]
        for eof_index in range(len(self.temporal_eofs)):
            eof_data = teofs[eof_index]
            variables["temporal_eof%d"%eof_index] = xr.Variable(data=eof_data, dims=("t"))

        variables["eigenvalues"] =xr.Variable(data=self.eigenvalues, dims=("eofs"))

        out = xr.Dataset(data_vars=variables,coords={
            "lon":( ["x"], lon_values, ),
            "lat":( ["y"], lat_values, ),
            "time": (["t"], time_values, ),
            "eofs": (["eofs"], list(range(1,1+len(self.eigenvalues))), ),
        },attrs={"lake_id":lake_id})

        out.to_netcdf(os.path.join(self.eof_folder,self.output_filename))



# matplotlib code for testing packing order of EOFs

# eofs = [1,7;2,8;3,9;4,10;5,11;6,12]
# mask = [true, false, true; false, true, false; true, true, true]
# r = dineof_unpack(eofs,mask)

# output from running:

# eofs =
#
#     1    7
#     2    8
#     3    9
#     4   10
#     5   11
#     6   12
#
# mask =
#
#   1  0  1
#   0  1  0
#   1  1  1
#
# r =
#
# ans(:,:,1) =
#
#      1   NaN     2
#    NaN     3   NaN
#      4     5     6
#
# ans(:,:,2) =
#
#      7   NaN     8
#    NaN     9   NaN
#     10    11    12
