"""
Represent a view of a lake and the reconstructed temperatures as part of a dashboard.
This class manages the creation and updating of various Bokeh figures that display the
quality flags, original temperatures, reconstructed temperatures, errors, and differences
between series for a given lake. It also handles user interactions like pixel selection.
"""

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, FixedTicker, ColorBar
from bokeh.models.formatters import CustomJSTickFormatter
from bokeh.models.tools import HoverTool
import numpy as np
import math
from .bokeh_utils import get_plot_dimensions, create_logger, create_hover_js
from .config import Config


logger = create_logger(__name__)

class AuxilaryView(object):

    def __init__(self, dashboard, variable, amin=0, amax=4, title=""):
        """
        Initialize the AuxilaryView with references to the dashboard and a variable name.
        All the data arrays, Bokeh data sources, and figure objects are initialized as None.
        """
        self.dashboard = dashboard         # Reference to the overall dashboard instance
        self.variable = variable
        self.ds = None

        self.lats = None  # Subset of latitude values corresponding to the valid region
        self.lons = None  # Subset of longitude values corresponding to the valid region
        self.amin = amin
        self.amax = amax
        self.title = title
        self.time_index = 0
        self.present_y_min = None
        self.present_y_max = None
        self.present_x_min = None
        self.present_x_max = None
        self.time_index = 0
        self.arr = None

        self.dashboard_times = None
        self.times = None

        self.p0 = None


    def bind(self, dashboard_times, ds):
        """
        Bind this view to a list of dashboard times and a dataset

        :param dashboard_times: a list of times (integer days since Jan 1st 1981) used in the main dashboard.  This may be different from the times in this view's dataset.
        :param ds: the Dataset object providing data for this view
        """
        self.ds = ds

        self.lats = self.ds.get_lats()
        self.lons = self.ds.get_lons()
        self.dashboard_times = dashboard_times  # main dataset, days since Jan 1, 1981
        self.times = self.dashboard.convert_times(self.ds.get_times())  # days since Jan 1, 1981

        # for auxilary views, simply define the display region as the entire data
        self.present_y_min = 0
        self.present_y_max = self.ds.xr_dataset.lat.shape[0]
        self.present_x_min = 0
        self.present_x_max = self.ds.xr_dataset.lon.shape[0]

    def build(self, plot_size):
        """
        Build the Bokeh figures for this view.
        For temperature datasets, builds p0–p4.
        For ice cover datasets, builds only p0 and p1.
        """
        height = 1 + self.present_y_max - self.present_y_min
        width = 1 + self.present_x_max - self.present_x_min
        (width_pixels, height_pixels) = get_plot_dimensions(plot_size)

        self.source = ColumnDataSource(data=dict(image=[self.arr], x=[0], y=[0], dw=[width], dh=[height]))

        if self.variable == "lake_ice_cover_class":
            self.color_mapper = LinearColorMapper(
                palette=["lightgray", "mediumblue", "cyan", "slategray","gold"],
                low=-0.5, high=4.5,
                nan_color="white"
            )
            ticker = FixedTicker(ticks=[0, 1, 2, 3, 4])

            formatter = CustomJSTickFormatter(code="""
                return {0: 'no lake', 1: 'water', 2: 'ice', 3: 'cloud', 4: 'bad'}[tick]
            """)
            color_bar = ColorBar(color_mapper=self.color_mapper,
                                 ticker=ticker,
                                 formatter=formatter,
                                 label_standoff=12,
                                 location=(0, 0))

        else:
            self.color_mapper = LinearColorMapper(
                palette=Config.temperature_color_map,
                low=math.floor(self.amin), low_color=Config.temperature_low_color,
                high=math.ceil(self.amax), high_color=Config.temperature_high_color,
                nan_color=Config.temperature_nan_color
            )
            color_bar = ColorBar(color_mapper=self.color_mapper,
                     label_standoff=12, border_line_color=None, location=(0, 0))

        self.p0 = figure(title=self.title, match_aspect=True, width=width_pixels,
                         height=height_pixels)
        self.p0.add_layout(color_bar, 'right')
        self.p0.axis.visible = False

        self.p0.image(image='image', x='x', y='y', dw='dw', dh='dh',
                      source=self.source,
                      color_mapper=self.color_mapper)

        customJS = create_hover_js(self.lats, self.lons)
        self.p0.add_tools(HoverTool(tooltips=[('lon/lat', '$x{custom}'), ("value", "@image")],
                                    formatters={"$x": customJS}))

        self.height = height
        self.width = width

    def rescale_to(self, minv, maxv, qa_filter):
        if minv is None:
            minv = self.amin
        if maxv is None:
            maxv = self.amax
        self.color_mapper.low = math.floor(minv)
        self.color_mapper.high = math.ceil(maxv)
        self.update_display()

    def redraw(self):
        self.update_display()

    def set_time_index(self, time_index):
        days_since_1981 = self.dashboard_times[time_index]
        # find the matching date in this data
        self.time_index = None
        for index in range(len(self.times)):
            if self.times[index] == days_since_1981:
                self.time_index = index
                break
        self.update_display()

    def update_display(self):
        if self.time_index is None:
            self.arr = np.zeros((self.present_y_max,self.present_x_max))
            self.arr[::] = np.nan
        else:
            self.arr = self.ds.get_data(self.variable, self.time_index,
                                    (self.present_y_min, self.present_y_max,
                                     self.present_x_min, self.present_x_max))
        self.source.data = dict(image=[self.arr], x=[0], y=[0], dw=[self.width], dh=[self.height])

    def get_figures(self):
        return [self.p0]

    def get_timeseries_at(self, x_index, y_index, quality_threshold=2):
        return None

    def get_local_timeseries_at(self, x_index, y_index, quality_threshold=2):
        return []

    def get_data(self):
        return self.ds

    def close(self):
        self.ds.close()
        self.p0 = None

    def __del__(self):
        try:
            logger.info("Deleting Dataset")
        except Exception:
            pass
