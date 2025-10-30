"""
Represent a view of a lake and the reconstructed temperatures as part of a dashboard.
This class manages the creation and updating of various Bokeh figures that display the
quality flags, original temperatures, reconstructed temperatures, errors, and differences
between series for a given lake. It also handles user interactions like pixel selection.
"""

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, LinearColorMapper, ColorBar, FixedTicker, NumeralTickFormatter
from bokeh.models.tools import CustomJSHover, HoverTool
from bokeh.events import Tap
from bokeh.palettes import diverging_palette, Blues9, Reds9, Viridis
import numpy as np
import math
from .bokeh_utils import get_plot_dimensions, create_logger, create_hover_js, create_timeseries_picker_js
from .config import Config
import json
import datetime

logger = create_logger(__name__)

class SpatialView(object):

    def __init__(self, dashboard, series):
        """
        Initialize the SpatialView with references to the dashboard and a series identifier.
        All the data arrays, Bokeh data sources, and figure objects are initialized as None.
        """
        self.dashboard = dashboard         # Reference to the overall dashboard instance
        self.series = series               # Identifier for which series (e.g., 1, 2, etc.) this view represents
        self.lake = None                   # The lake object (metadata, etc.)
        self.test_name = "?"               # Test/dataset identifier; will be set in bind()
        # New flags for independent behavior:
        self.interactive = True            # If False, do not attach event callbacks (for independent views)
        self.independent_ranges = False    # If True, use independent ranges (not shared with main dashboard)
        
        self.mask_arr = None               # Array for the data mask (areas where data is present)
        self.qaflag_arr = None             # Array for quality flags
        self.original_arr = None           # Array of original data (temperature or ice cover)
        self.reconstructed_arr = None      # Array of reconstructed temperature data (temperature datasets)
        self.error_arr = None              # Array of differences between reconstructed and original (temperature datasets)
        self.reconstruction_diff_arr = None# Array of differences between two reconstructions (temperature datasets)

        # Spatial extents where data is present
        self.present_x_min = None
        self.present_x_max = None
        self.present_y_min = None
        self.present_y_max = None

        self.lats = None                   # Subset of latitude values corresponding to the valid region
        self.lons = None                   # Subset of longitude values corresponding to the valid region

        # Bokeh data sources for different image layers
        self.source0 = None                # QA band image source
        self.source1 = None                # Original data image source (temperature or ice cover)
        self.source1m = None               # Mask image source
        self.source2 = None                # Reconstructed data image (temperature datasets)
        self.source3 = None                # Error image (temperature datasets)
        self.source4 = None                # Series difference image (temperature datasets)

        # Bokeh figure objects for displaying different layers
        self.p0 = None                     # Figure for QA Band display
        self.p1 = None                     # Figure for Original data display (temperature or ice cover)
        self.p2 = None                     # Figure for Reconstructed data (temperature datasets)
        self.p3 = None                     # Figure for Error (temperature datasets)
        self.p4 = None                     # Figure for Series Difference (temperature datasets)

        # glyphs that draw circles aroubnd selected pixels in the plots p0, p1, p2, p3, p4
        self.circle0 = None
        self.circle1 = None
        self.circle2 = None
        self.circle3 = None
        self.circle4 = None

        self.qa_threshold = 2              # Quality threshold
        self.ds = None                     # Primary dataset bound to this view
        self.ds_other = None               # Secondary dataset for comparisons
        self.times = None                  # List of time indices (local to this view)

        self.spatial_view1 = None
        self.spatial_view2 = None

    def get_test_name(self):
        """
        Return the test name/identifier associated with this view.
        """
        return self.test_name

    def bind(self, lake, ds, times, time_index, test_name):
        """
        Bind the lake and dataset to this view.
        Determines the valid region from the dataset's mask and subsets the latitude and longitude arrays.
        """
        self.lake = lake
        self.test_name = test_name
        self.ds = ds
        self.ds_other = None
        self.times = times

        mask = np.array(ds.get_mask())
        data_present = np.argwhere(mask > 0)
        if data_present.size == 0:
            raise Exception("No valid data found in the dataset mask.")
        present_x = [x for (y, x) in data_present]
        present_y = [y for (y, x) in data_present]
        self.present_x_min = min(present_x)
        self.present_x_max = max(present_x)
        self.present_y_min = min(present_y)
        self.present_y_max = max(present_y)

        lats = ds.get_lats()
        lons = ds.get_lons()
        self.lats = lats[self.present_y_min:self.present_y_max+1]
        self.lons = lons[self.present_x_min:self.present_x_max+1]

        self.set_time_index(time_index)

    def bind_other(self, ds_other):
        """
        Bind a secondary dataset for comparison.
        """
        self.ds_other = ds_other
        if ds_other is not None and hasattr(self, 'series_diff_color_mapper'):
            max_diff = self.ds.get_max_diff(self.dashboard.filled_temperature_variable, ds_other)
            self.series_diff_color_mapper.update(low=-max_diff, high=max_diff)
        self.set_time_index(self.time_index)

    def rescale_to(self, minv, maxv, qa_filter):
        if minv is None:
            minv = self.amin
        if maxv is None:
            maxv = self.amax
        self.color_mapper.low = math.floor(minv)
        self.color_mapper.high = math.ceil(maxv)
        if qa_filter == ">=1":
            self.qa_threshold = 1
        elif qa_filter == ">=2":
            self.qa_threshold = 2
        elif qa_filter == ">=3":
            self.qa_threshold = 3
        elif qa_filter == ">=4":
            self.qa_threshold = 4
        else:
            self.qa_threshold = 5
        self.redraw()

    def rescale_errors(self, min_error, max_error):
        # Update the error color mapper (used for p3)
        self.error_color_mapper.low = min_error
        self.error_color_mapper.high = max_error
        self.redraw()

    def rescale_diff(self, min_diff, max_diff):
        # Update the series diff color mapper (used for p4)
        self.series_diff_color_mapper.low = min_diff
        self.series_diff_color_mapper.high = max_diff
        self.redraw()

    def redraw(self):
        self.set_time_index(self.time_index)

    def build(self, plot_size):
        """
        Build the Bokeh figures for this view.
        For temperature datasets, builds p0–p4.
        For ice cover datasets, builds only p0 and p1.
        """
        height = 1 + self.present_y_max - self.present_y_min
        width = 1 + self.present_x_max - self.present_x_min
        (width_pixels, height_pixels) = get_plot_dimensions(plot_size)

        # Prepare a custom JS hover formatter using the local lat/lon arrays.
        customJS = create_hover_js(self.lats, self.lons)
        pick_timeseries = create_timeseries_picker_js(self.lats, self.lons)

        self.amin = self.ds.get_min_value(self.dashboard.temperature_variable)
        self.amax = self.ds.get_max_value(self.dashboard.temperature_variable)
        title_p1 = "Original LSWT (K)"

        self.qa_color_mapper = LinearColorMapper(palette=Config.qa_color_map, low=1, high=5,
                                                   nan_color=(255, 255, 255, 0))

        self.color_mapper = LinearColorMapper(
            palette=Config.temperature_color_map,
            low=math.floor(self.amin), low_color=Config.temperature_low_color,
            high=math.ceil(self.amax), high_color=Config.temperature_high_color,
            nan_color=Config.temperature_nan_color
        )

        error_palette = diverging_palette(Blues9, Reds9, 18, 0.5)
        self.error_color_mapper = LinearColorMapper(
            palette=error_palette,
            low=math.floor(-abs(self.amax)), high=math.ceil(abs(self.amax)),
            nan_color=(255, 255, 255, 0)
        )
        self.series_diff_color_mapper = LinearColorMapper(
            palette=error_palette, low=-4, high=4,
            nan_color=(255, 255, 255, 0)
        )

        # --- RANGE SETUP ---
        if not self.independent_ranges:
            (x_range, y_range) = self.lake.get_ranges()
            if not x_range or not y_range:
                # Create a temporary figure to obtain default ranges.
                temp_fig = figure(width=width_pixels, height=height_pixels)
                x_range = temp_fig.x_range
                y_range = temp_fig.y_range
                self.lake.set_ranges(x_range, y_range)
            range_arguments = {"x_range": x_range, "y_range": y_range}
        else:
            # Independent views do not use shared ranges.
            range_arguments = {}

        from bokeh.models import FixedTicker, ColorBar, NumeralTickFormatter
        custom_palette = ["black", "purple", "green", "cyan", "red"]
        ticker = FixedTicker(ticks=[1, 2, 3, 4, 5])
        qa_local_color_mapper = LinearColorMapper(palette=custom_palette, low=0.5, high=5.5)
        qa_color_bar = ColorBar(color_mapper=qa_local_color_mapper,
                                ticker=ticker,
                                formatter=NumeralTickFormatter(format="0"),
                                label_standoff=12,
                                location=(0, 0))

        # Build p0 and p1 (always present)
        self.p0 = figure(title="QA Band", match_aspect=True, width=width_pixels,
                         height=height_pixels, **range_arguments)
        self.p0.add_tools(HoverTool(tooltips=[('lon/lat', '$x{custom}'), ("value", "@image")],
                                     formatters={"$x": customJS}))
        self.p0.add_layout(qa_color_bar, 'right')
        self.p0.axis.visible = False

        self.p1 = figure(title=title_p1, match_aspect=True, width=width_pixels,
                         height=height_pixels, **range_arguments)
        self.p1.add_tools(HoverTool(tooltips=[('lon/lat', '$x{custom}'), ("value", "@image")],
                                     formatters={"$x": customJS}))
        self.p1.axis.visible = False


        self.p2 = figure(title="Series%d Reconstructed LSWT" % (self.series), match_aspect=True,
                         width=width_pixels, height=height_pixels, x_range=x_range, y_range=y_range)
        self.p2.add_tools(HoverTool(tooltips=[('lon/lat', '$x{custom}'), ("value", "@image")],
                                     formatters={"$x": customJS}))
        self.p2.axis.visible = False

        self.p3 = figure(title="Difference: Series%d - Original" % (self.series), match_aspect=True,
                         width=width_pixels, height=height_pixels, x_range=x_range, y_range=y_range)
        self.p3.add_tools(HoverTool(tooltips=[('lon/lat', '$x{custom}'), ("value", "@image")],
                                     formatters={"$x": customJS}))
        self.p3.axis.visible = False

        self.p4 = figure(title="Series%d - Series%d" % (self.series, 3 - self.series), match_aspect=True,
                         width=width_pixels, height=height_pixels, x_range=x_range, y_range=y_range)
        self.p4.add_tools(HoverTool(tooltips=[('lon/lat', '$x{custom}'), ("value", "@image")],
                                     formatters={"$x": customJS}))
        self.p4.axis.visible = False

        # Add images to p0 and p1.
        self.p0.image(image='image', x='x', y='y', dw='dw', dh='dh',
                      source=self.source1m,
                      color_mapper=LinearColorMapper(palette=["white", "#E0E0E0"]))
        self.p0.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source0,
                      color_mapper=qa_local_color_mapper)

        self.p1.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source1m,
                      color_mapper=LinearColorMapper(palette=["white", "#E0E0E0"]))
        self.p1.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source1,
                      color_mapper=self.color_mapper)


        self.p2.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source1m,
                      color_mapper=LinearColorMapper(palette=["white", "#E0E0E0"]))
        self.p2.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source2,
                      color_mapper=self.color_mapper)
        self.p3.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source1m,
                      color_mapper=LinearColorMapper(palette=["white", "#E0E0E0"]))
        self.p3.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source3,
                      color_mapper=self.error_color_mapper)
        self.p4.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source1m,
                      color_mapper=LinearColorMapper(palette=["white", "#E0E0E0"]))
        self.p4.image(image='image', x='x', y='y', dw='dw', dh='dh', source=self.source4,
                      color_mapper=self.series_diff_color_mapper)
        color_bar = ColorBar(color_mapper=self.color_mapper,
                             label_standoff=12, border_line_color=None, location=(0, 0))
        error_color_bar = ColorBar(color_mapper=self.error_color_mapper,
                                   label_standoff=12, border_line_color=None, location=(0, 0))
        series_diff_color_bar = ColorBar(color_mapper=self.series_diff_color_mapper,
                                         label_standoff=12, border_line_color=None, location=(0, 0))
        self.p1.add_layout(color_bar, 'right')
        self.p2.add_layout(color_bar, 'right')
        self.p3.add_layout(error_color_bar, 'right')
        self.p4.add_layout(series_diff_color_bar, 'right')

        # Attach event callbacks only if interactive is True.
        if self.interactive:
            for panel in self.get_figures():
                panel.js_on_event(Tap, pick_timeseries)
                panel.on_event(Tap, lambda e: self.dashboard.select_pixel(e.x, e.y))

        self.height = height
        self.width = width

    def redraw(self):
        self.set_time_index(self.time_index)

    def set_time_index(self, index):
        height = 1 + self.present_y_max - self.present_y_min
        width = 1 + self.present_x_max - self.present_x_min
        self.time_index = index

        self.mask_arr = self.ds.get_mask(slice=(self.present_y_min, self.present_y_max,
                                                  self.present_x_min, self.present_x_max))
        if "lake_ice_cover_class" in self.ds.xr_dataset.variables:
            self.original_arr = self.ds.get_data("lake_ice_cover_class", self.time_index,
                                                  (self.present_y_min, self.present_y_max,
                                                   self.present_x_min, self.present_x_max))
            self.qaflag_arr = self.original_arr
            self.reconstructed_arr = self.original_arr
            self.error_arr = np.zeros_like(self.original_arr)
            self.reconstruction_diff_arr = np.zeros_like(self.original_arr)
        else:
            self.original_arr = self.ds.get_data(self.dashboard.temperature_variable, self.time_index,
                                                  (self.present_y_min, self.present_y_max,
                                                   self.present_x_min, self.present_x_max))
            self.qaflag_arr = self.ds.get_data(self.dashboard.quality_level_variable, self.time_index,
                                                (self.present_y_min, self.present_y_max,
                                                 self.present_x_min, self.present_x_max))
            self.original_arr = np.where(self.qaflag_arr >= self.qa_threshold, self.original_arr, np.nan)
            self.reconstructed_arr = self.ds.get_data(self.dashboard.filled_temperature_variable, self.time_index,
                                                       (self.present_y_min, self.present_y_max,
                                                        self.present_x_min, self.present_x_max))
            self.error_arr = (np.array(self.reconstructed_arr) - np.array(self.original_arr))
            if self.ds_other:
                other_reconstructed_arr = self.ds_other.get_data(self.dashboard.filled_temperature_variable, self.time_index,
                                                                 (self.present_y_min, self.present_y_max,
                                                                  self.present_x_min, self.present_x_max))
                self.reconstruction_diff_arr = (np.array(self.reconstructed_arr) - np.array(other_reconstructed_arr))
            else:
                self.reconstruction_diff_arr = (np.array(self.reconstructed_arr) - np.array(self.reconstructed_arr))
    
        if self.source1 is None:
            self.source0 = ColumnDataSource(data=dict(image=[self.qaflag_arr], x=[0], y=[0], dw=[width], dh=[height]))
            self.source1m = ColumnDataSource(data=dict(image=[self.mask_arr], x=[0], y=[0], dw=[width], dh=[height]))
            self.source1 = ColumnDataSource(data=dict(image=[self.original_arr], x=[0], y=[0], dw=[width], dh=[height]))
            self.source2 = ColumnDataSource(data=dict(image=[self.reconstructed_arr], x=[0], y=[0], dw=[width], dh=[height]))
            self.source3 = ColumnDataSource(data=dict(image=[self.error_arr], x=[0], y=[0], dw=[width], dh=[height]))
            self.source4 = ColumnDataSource(data=dict(image=[self.reconstruction_diff_arr], x=[0], y=[0], dw=[width], dh=[height]))
        else:
            self.source0.data = dict(image=[self.qaflag_arr], x=[0], y=[0], dw=[width], dh=[height])
            self.source1.data = dict(image=[self.original_arr], x=[0], y=[0], dw=[width], dh=[height])
            self.source2.data = dict(image=[self.reconstructed_arr], x=[0], y=[0], dw=[width], dh=[height])
            self.source3.data = dict(image=[self.error_arr], x=[0], y=[0], dw=[width], dh=[height])
            self.source4.data = dict(image=[self.reconstruction_diff_arr], x=[0], y=[0], dw=[width], dh=[height])

    def get_figures(self):
        if "lake_ice_cover_class" in self.ds.xr_dataset.variables:
            return [self.p0, self.p1]
        else:
            return [self.p0, self.p1, self.p2, self.p3, self.p4]
        
    def select_pixel(self, x, y):
        # Fixme is there a way to remove any prevous selection - would be cleaner than making it invisible
        if self.circle0:
            self.circle0.visible = False
        if self.circle1:
            self.circle1.visible = False
        if self.circle2:
            self.circle2.visible = False
        if self.circle3:
            self.circle3.visible = False
        if self.circle4:
            self.circle4.visible = False

        if self.p0:
            self.circle0 = self.p0.circle([x], [y], radius=max(self.height, self.width)/15, line_color="red",
                           fill_color=(255, 255, 255, 0))
        if self.p1:
            self.circle1 = self.p1.circle([x], [y], radius=max(self.height, self.width)/15, line_color="red",
                           fill_color=(255, 255, 255, 0))
        if self.p2:
            self.circle2 = self.p2.circle([x], [y], radius=max(self.height, self.width)/15, line_color="red",
                           fill_color=(255, 255, 255, 0))
        if self.p3:
            self.circle3 = self.p3.circle([x], [y], radius=max(self.height, self.width)/15, line_color="red",
                           fill_color=(255, 255, 255, 0))
        if self.p4:
            self.circle4 = self.p4.circle([x], [y], radius=max(self.height, self.width)/15, line_color="red",
                           fill_color=(255, 255, 255, 0))
    
    def get_timeseries_at(self, x_index, y_index, quality_threshold=2):
        self.status = "Loading Time Series"
        ts1 = []
        ts2 = []
        quality_levels = []
        columns_temps = ["date", "original"]
        columns_diffs = ["date"]
        if self.spatial_view1:
            ts1 = self.spatial_view1.get_timeseries_at(x_index, y_index)
        columns_temps.append("series1")
        columns_diffs.append("series1_diff")
        columns_diffs.append("series1_minus_series2")
        if self.spatial_view2:
            ts2 = self.spatial_view2.get_timeseries_at(x_index, y_index)
        columns_temps.append("series2")
        columns_diffs.append("series2_diff")
        columns_diffs.append("series2_minus_series1")
        tsdata_temps = ",".join(columns_temps)
        tsdata_diffs = ",".join(columns_diffs)
        rowcount = max(len(ts1), len(ts2))
        eofdummy = "date,eof"
        for days in self.times:
            dt = datetime.datetime(1981, 1, 1) + datetime.timedelta(days=int(days))
            dts = dt.strftime("%Y%m%d")
            eof_value = ""
            if "temporal_eof0" in self.eof_mappings:
                eof_value = self.eof_mappings["temporal_eof0"].get(dts, "")
            eofdummy += "\n" + dts + "," + str(eof_value)
        for row in range(rowcount):
            tsdata_diffs += "\n"
            tsdata_temps += "\n"
            if row < len(ts1) and ts1:
                days = ts1[row][0]
                v = ts1[row][1]
                q = ts1[row][2]
            elif row < len(ts2) and ts2:
                days = ts2[row][0]
                v = ts2[row][1]
                q = ts2[row][2]
            else:
                continue
            quality_levels.append(q if not math.isnan(q) else -1)
            dt = datetime.datetime(1981, 1, 1) + datetime.timedelta(days=int(days))
            dts = dt.strftime("%Y%m%d")
            v_str = f"{v}" if not math.isnan(v) else ""
            tsdata_temps += f"{dts},{v_str}"
            tsdata_diffs += f"{dts}"
            if row < len(ts1) and ts1:
                s = ts1[row][3]
                e = ts1[row][4]
                d = ts1[row][5]
                tsdata_temps += f",{s if not math.isnan(s) else ''}"
                tsdata_diffs += f",{e if not math.isnan(e) else ''},{d if not math.isnan(d) else ''}"
            else:
                tsdata_temps += ","
                tsdata_diffs += ",,"
            if row < len(ts2) and ts2:
                s = ts2[row][3]
                e = ts2[row][4]
                d = ts2[row][5]
                tsdata_temps += f",{s if not math.isnan(s) else ''}"
                tsdata_diffs += f",{e if not math.isnan(e) else ''},{d if not math.isnan(d) else ''}"
            else:
                tsdata_temps += ","
                tsdata_diffs += ",,"
        self.status = ""
        return {
            "temps": tsdata_temps,
            "diffs": tsdata_diffs,
            "quality_levels": quality_levels,
            "eof": eofdummy
        }

    def get_local_timeseries_at(self, x_index, y_index, quality_threshold=2):
        """
        Internal method to extract time series data from this view’s dataset.
        It replicates the original CSV-building logic but does not attempt to call a subview.
        Returns a list of tuples: (date, original, quality, reconstructed, error, diff)
        (Note: diff here corresponds to the extra columns in your CSV.)
        """
        # Adjust pixel indices to the full dataset space
        true_x = x_index + self.present_x_min
        true_y = y_index + self.present_y_min

        # Extract time series data from the dataset
        original_ts = self.ds.get_timeseries(self.dashboard.temperature_variable, true_x, true_y)
        quality_ts  = self.ds.get_timeseries(self.dashboard.quality_level_variable, true_x, true_y)
        reconstructed_ts = self.ds.get_timeseries(self.dashboard.filled_temperature_variable, true_x, true_y)
        error_ts = [r - o for o, r in zip(original_ts, reconstructed_ts)]
        # For simplicity, we fill the "diff" columns with zeros (or you can add any logic here)
        diff_ts = [0 for _ in error_ts]

        # Build a list of tuples for each time step
        result = []
        for time_val, o, q, r, e, d in zip(self.dashboard.times, original_ts, quality_ts, reconstructed_ts, error_ts, diff_ts):
            # Apply quality filter: if quality below threshold, set original and error to NaN
            if q < quality_threshold:
                o = float('nan')
                e = float('nan')
            result.append((time_val, o, q, r, e, d))
        return result

    
    def get_data(self):
        return self.ds

    def get_lake_name(self):
        return self.lake.get_name()

    def close(self):
        self.ds.close()
        self.qaflag_arr = None
        self.mask_arr = None
        self.original_arr = None
        self.reconstructed_arr = None
        self.error_arr = None
        self.source0 = None
        self.source1 = None
        self.source1m = None
        self.source2 = None
        self.source3 = None
        self.source4 = None
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p4 = None

    def __del__(self):
        try:
            logger.info("Deleting Dataset")
        except Exception:
            pass
