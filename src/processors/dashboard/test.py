"""
Represent a test.  More functionality to be merged in...
"""

import xarray as xr
from .config import Config


class Test(object):

    def __init__(self, test_id, path, eofs_path=None):
        """
        Models a test run of dineof on a lake

        :param test_id (int): the id of the lake
        :param path(str): the path to the original LSWT file with the merged results
        :param eofs_path(str): the path to a netcdf4 file containing the EOFs
        """
        self.test_id = test_id
        self.path = path
        self.eofs_path = eofs_path
        ds = xr.open_dataset(self.path)
        self.description = ds.attrs.get(Config.test_info_attribute, "")

    def get_test_id(self):
        return self.test_id

    def get_path(self):
        return self.path

    def get_eofs_path(self):
        return self.eofs_path

    def get_description(self):
        return self.description

    def __repr__(self):
        return "Test(%s)" % (self.test_id)
