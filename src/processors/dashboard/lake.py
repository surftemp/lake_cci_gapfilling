"""
Represent a lake. More functionality to be merged in...
This class models a lake by storing its identifier, name, and any associated tests.
It also caches the x and y ranges from Bokeh figures for efficient re-use.
"""

class Lake(object):

    def __init__(self, lake_id, ice_data_path=None, fine_res_path=None):
        """
        Initialize a new Lake instance.

        :param lake_id (int): The unique identifier for the lake.
                             The lake name is derived from the lake id.
        """
        self.id = lake_id                # Store the lake's ID.
        self.ice_data_path = ice_data_path
        self.fine_res_path = fine_res_path

        self.name = str(lake_id)         # Use the lake ID (as a string) as the lake name.
        # Variables to cache the last used figure x and y ranges.
        self.x_range = None              # Cached x-axis range for Bokeh figures.
        self.y_range = None              # Cached y-axis range for Bokeh figures.
        self.tests = {}                  # Dictionary to store tests associated with this lake.

    def get_id(self):
        """
        Return the lake's unique identifier.
        """
        return self.id

    def get_name(self):
        """
        Return the lake's name.
        """
        return self.name

    def get_ice_data_path(self):
        return self.ice_data_path

    def get_fine_res_path(self):
        return self.fine_res_path

    def get_ranges(self):
        """
        Return the cached x and y ranges as a tuple.
        These ranges can be used to ensure consistent spatial plotting.
        """
        return (self.x_range, self.y_range)

    def set_ranges(self, x_range, y_range):
        """
        Set the x and y ranges for the lake's figures.

        :param x_range: The x-axis range (typically a Bokeh Range1d object).
        :param y_range: The y-axis range (typically a Bokeh Range1d object).
        """
        self.x_range = x_range
        self.y_range = y_range

    def add_test(self, test):
        """
        Add a test (data series) to the lake.

        :param test: A test object containing information about a particular dataset.
                     The test is stored in a dictionary keyed by its test ID.
        """
        self.tests[test.get_test_id()] = test

    def get_test_names(self):
        """
        Return a sorted list of test names (or IDs) associated with this lake.
        """
        return list(sorted(self.tests.keys()))

    def get_test(self, test_name):
        """
        Retrieve a test associated with the given test name.
        
        :param test_name: The identifier for the desired test.
        :return: The corresponding test object if it exists; otherwise, None.
        """
        if test_name in self.tests:
            return self.tests[test_name]
        else:
            return None

    def __repr__(self):
        """
        Return a string representation of the lake, including its id and name.
        """
        return "Lake(%d,%s)" % (self.id, self.name)
