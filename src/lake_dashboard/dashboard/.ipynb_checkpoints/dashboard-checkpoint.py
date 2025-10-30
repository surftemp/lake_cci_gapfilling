"""
This module sets up the Bokeh dashboard as a web page with various plots and interactive controls.
It integrates with lake data (temperature, quality flags, EOFs, etc.) and provides a GUI for users to interact with spatial views and time series.
"""

import math
import os
import gc
import datetime
import xarray as xr
import pandas as pd
import numpy as np
import glob

from lake_dashboard.data.metadata_reader import MetadataReader
from .dataset import Dataset
from .spatial_view import SpatialView
from .auxilary_view import AuxilaryView
from .lake import Lake
from .test import Test

from bokeh.plotting import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, Spinner, Select, CustomJS, Button, CheckboxGroup
from .bokeh_utils import create_bokeh_div, get_plot_dimensions, create_logger

from .config import Config

DEFAULT_MIN_TEMP = 273.15
DEFAULT_MAX_TEMP = 305

logger = create_logger(__name__)
mr = MetadataReader()
mr.load()


class Dashboard(object):
    def __init__(self):
        self.times = None
        self.time_index = 0
        self.lakes = {}
        self.lake = None
        self.selected_test1 = None
        self.selected_test2 = None
        self.spatial_view1 = None
        self.spatial_view2 = None
        # NEW: Additional views for high-res LSWT (p5) and ice cover (p6)
        self.aux_view1 = None
        self.eof_mappings = {}

        self.metadata_div = create_bokeh_div("", sz=16)
        self.output1_link = create_bokeh_div("", sz=16)
        self.output2_link = create_bokeh_div("", sz=16)
        # NEW: Output links for p5 and p6
        self.output3_link = create_bokeh_div("", sz=16)
        self.output4_link = create_bokeh_div("", sz=16)
        self.description1_txt = create_bokeh_div("Description 1", sz=16)
        self.description2_txt = create_bokeh_div("Description 2", sz=16)
        # NEW: Descriptions for p5 and p6
        self.description3_txt = create_bokeh_div("Fine Resolution LSWT", sz=16)
        self.description4_txt = create_bokeh_div("Lake Ice Cover", sz=16)
        self.plot_size = "medium"
        self.selected_x = None
        self.selected_y = None
        self.request_index = -1
        self.status = ""
        self.refresh_time_series = False
        self.play_btn = None
        self.playing = False
        self.temp_min_value = None
        self.temp_max_value = None
        self.qa_filter = ">=2"
        self.status = ""

    def load_eofs(self, eof_file):
        ds_eofs = xr.open_dataset(eof_file)
        reference_date = np.datetime64('1981-01-01')
        ds_times = reference_date + ds_eofs['time'].astype('timedelta64[D]').values
        ds_eofs = ds_eofs.assign_coords(time=ds_times)
        self.eof_mappings = {}
        for i in range(11):
            var_name = f"temporal_eof{i}"
            if var_name in ds_eofs.variables:
                values = ds_eofs[var_name].values
                dates = pd.to_datetime(ds_eofs['time'].values)
                formatted_dates = dates.strftime("%Y%m%d")
                mapping_for_this_var = dict(zip(formatted_dates, values))
                self.eof_mappings[var_name] = mapping_for_this_var
        ds_eofs.close()

    def get_status(self):
        return self.status

    def get_refresh_timeseries(self):
        refresh_time_series = self.refresh_time_series
        self.refresh_time_series = False
        return refresh_time_series

    def load_data(self, lake, test, series):
        if test is None:
            return None
        path = test.get_path()
        ds1 = Dataset(self, xr.open_dataset(path, cache=True))
        self.times = self.convert_times(ds1.get_times())
        view = SpatialView(self, series)
        view.bind(lake, ds1, self.times, self.time_index, test.get_test_id())
        return view

    def load_auxilary_view(self, path, variable, amin, amax, title):
        logger.info("Opening auxilary dataset from path: %s", path)
        ds = Dataset(self, xr.open_dataset(path, engine="netcdf4", cache=True))
        view = AuxilaryView(self, variable=variable, amin=amin, amax=amax, title=title)
        view.bind(ds=ds, dashboard_times=self.times)
        return view

    def convert_times(self, times):
        jan01_1981 = datetime.datetime(1981, 1, 1, 12, 0, 0)
        logger.info("Raw times (first 5): %s", times[:5])
        def convert_date(npdt):
            odt = datetime.datetime.utcfromtimestamp(npdt / 1e9)
            return (odt - jan01_1981).days
        converted_times = list(map(convert_date, times))
        logger.info("Converted times (first 5): %s", converted_times[:5])
        return converted_times

    def set_log_output_links(self):
        # For primary views (p0 and p1)
        if self.spatial_view1:
            output_path = self.selected_lake.get_test(self.spatial_view1.get_test_name()).get_path()
            output_netcdf4_name = os.path.split(output_path)[1]
            self.output1_link.text = output_netcdf4_name
        else:
            self.output1_link.text = ''
        if self.selected_test1:
            self.description1_txt.text = self.selected_test1.get_description()
        else:
            self.description1_txt.text = ''
        if self.spatial_view2:
            output_path = self.selected_lake.get_test(self.spatial_view2.get_test_name()).get_path()
            output_netcdf4_name = os.path.split(output_path)[1]
            self.output2_link.text = output_netcdf4_name
        else:
            self.output2_link.text = ''
        if self.selected_test2:
            self.description2_txt.text = self.selected_test2.get_description()
        else:
            self.description2_txt.text = ''
        # For independent views (p5 and p6)
        if self.aux_view0:
            self.output3_link.text = os.path.split(Config.fine_res_path)[1] if Config.fine_res_path else ""
            self.description3_txt.text = "Fine Resolution LSWT"
        else:
            self.output3_link.text = ""
            self.description3_txt.text = ""
        if self.aux_view1:
            self.output4_link.text = os.path.split(Config.ice_data_path)[1] if Config.ice_data_path else ""
            self.description4_txt.text = "Lake Ice Cover"
        else:
            self.output4_link.text = ""
            self.description4_txt.text = ""

    def close(self):
        logger.info("closing dashboard")
        if self.spatial_view1:
            self.spatial_view1.close()
            self.spatial_view1 = None
        if self.spatial_view2:
            self.spatial_view2.close()
            self.spatial_view2 = None
        if self.aux_view0:
            self.aux_view0.close()
            self.aux_view0 = None
        if self.aux_view1:
            self.aux_view1.close()
            self.aux_view1 = None

    # NEW: Add bind_other() method to support linking primary views.
    def bind_other(self):
        if self.spatial_view1:
            if self.spatial_view2:
                self.spatial_view1.bind_other(self.spatial_view2.get_data())
            else:
                self.spatial_view1.bind_other(None)
        if self.spatial_view2:
            if self.spatial_view1:
                self.spatial_view2.bind_other(self.spatial_view1.get_data())
            else:
                self.spatial_view2.bind_other(None)

    def init(self, temperature_variable="", filled_temperature_variable="", quality_level_variable="",
             lake_mask_variable="", data=[]):
        # read through all the data files
        self.status = "Initializing"
        first_lake_test = None
        self.temperature_variable = temperature_variable
        self.filled_temperature_variable = filled_temperature_variable
        self.quality_level_variable = quality_level_variable
        self.lake_mask_variable = lake_mask_variable

        self.lakes = {}
        for data_path in data:
            filename = os.path.split(data_path)[-1]
            folder = os.path.split(data_path)[0]

            ds = xr.open_dataset(data_path)
            lake_id = ds.attrs.get(Config.lake_id_attribute, filename)
            test_id = ds.attrs.get(Config.test_id_attribute, filename)
            lake_name = str(lake_id)

            # retrieve the paths of auxilary files
            def resolve_aux_path(path, folder):
                if not path:
                    return None
                elif os.path.isabs(path):
                    return path
                else:
                    relative_path = os.path.join(folder, path)
                    potential_matches = glob.glob(relative_path, recursive=True)
                    if len(potential_matches) != 1:
                        raise ValueError("Failed to find 1 match for {relative_path}")
                    return potential_matches[0]

            # see if a Lake object is already created, retrieve it or create one if needed
            if lake_name in self.lakes:
                lake = self.lakes[lake_name]
            else:
                ice_data_path = resolve_aux_path(Config.ice_data_path,folder)
                fine_res_path = resolve_aux_path(Config.fine_res_path,folder)

                lake = Lake(lake_id,ice_data_path=ice_data_path,fine_res_path=fine_res_path)
                self.lakes[lake_name] = lake

            # create a test
            eofs_path = resolve_aux_path(Config.eofs_path,folder)
            test = Test(test_id, data_path, eofs_path=eofs_path)

            lake.add_test(test)
            if not first_lake_test:
                first_lake_test = (lake, test)
            logger.info("Registering (%s,%s)" % (lake_id, test_id))
        if first_lake_test:
            self.selected_lake = first_lake_test[0]
            self.selected_test1 = first_lake_test[1]
        else:
            self.selected_lake = None
            self.selected_test1 = None
        self.selected_test2 = None
        self.update_metadata()

    def setup(self, doc):
        self.status = "Loading"
        self.spatial_view1 = self.load_data(self.selected_lake, self.selected_test1, 1)
        if self.spatial_view1:
            self.spatial_view1.build(self.plot_size)
            eofs_path = self.selected_test1.get_eofs_path()
            if eofs_path:
                self.load_eofs(eofs_path)
            else:
                self.eof_mappings = {}

        self.spatial_view2 = self.load_data(self.selected_lake, self.selected_test2, 2)
        if self.spatial_view2:
            self.spatial_view2.build(self.plot_size)

        self.aux_view0 = None

        ice_data_path = self.selected_lake.ice_data_path
        if ice_data_path:
            self.aux_view1 = self.load_auxilary_view(ice_data_path, "lake_ice_cover_class", 0, 4, "Ice Cover")
            self.aux_view1.build(self.plot_size)
            self.aux_view1.set_time_index(self.time_index)
        else:
            self.aux_view1 = None

        fine_res_path = self.selected_lake.fine_res_path
        if fine_res_path:
            self.aux_view0 = self.load_auxilary_view(fine_res_path, "lake_surface_water_temperature", self.spatial_view1.amin, self.spatial_view1.amax, "Fine res LSWT")
            self.aux_view0.build(self.plot_size)
            self.aux_view0.set_time_index(self.time_index)
        else:
            self.aux_view0 = None

        self.set_log_output_links()
        self.setup_spatial_plots()

        select_lake = Select(title="Select Lake", options=list(self.lakes.keys()), value=self.selected_lake.get_name())
        select_size = Select(title="Plot Sizes", options=["small", "medium", "large"], value=self.plot_size)
        select_qa_filter = Select(title="QA Band Filter", options=[">=1", ">=2", ">=3", ">=4", "=5"], value=self.qa_filter)

        def update_qa_filter(attr, old, new):
            self.qa_filter = new
            self.rescale()

        select_qa_filter.on_change('value', update_qa_filter)
        self.manual_scale = CheckboxGroup(labels=['Manual Temperature Scale'], active=[1])
        self.min_temp = Spinner(value=DEFAULT_MIN_TEMP, width=100, title="min")
        self.min_temp.visible = False
        self.max_temp = Spinner(value=DEFAULT_MAX_TEMP, width=100, title="max")
        self.max_temp.visible = False
        self.rescale_btn = Button(label='Rescale', width=100)
        self.rescale_btn.visible = False

        def update_manual_scale(attr, old, new):
            if not self.manual_scale.active[0]:
                self.min_temp.visible = True
                self.max_temp.visible = True
                self.rescale_btn.visible = True
                self.temp_min_value = self.min_temp.value
                self.temp_max_value = self.max_temp.value
            else:
                self.min_temp.visible = False
                self.max_temp.visible = False
                self.rescale_btn.visible = False
                self.temp_min_value = None
                self.temp_max_value = None
            self.rescale()
        self.manual_scale.on_change('active', update_manual_scale)

        def update_temp_min(attr, old, new):
            self.temp_min_value = self.min_temp.value
        def update_temp_max(attr, old, new):
            self.temp_max_value = self.max_temp.value
        self.rescale_btn.on_click(lambda: self.rescale())
        self.min_temp.on_change('value', update_temp_min)
        self.max_temp.on_change('value', update_temp_max)

        self.manual_error_scale = CheckboxGroup(labels=['Manual Diff Scale 1'], active=[])
        self.min_error = Spinner(value=0, width=100, title="Error Min")
        self.min_error.visible = False
        self.max_error = Spinner(value=10, width=100, title="Error Max")
        self.max_error.visible = False
        self.rescale_error_btn = Button(label='Rescale', width=100)
        self.rescale_error_btn.visible = False

        def update_manual_error_scale(attr, old, new):
            if self.manual_error_scale.active:
                self.min_error.visible = True
                self.max_error.visible = True
                self.rescale_error_btn.visible = True
                self.error_min_value = self.min_error.value
                self.error_max_value = self.max_error.value
            else:
                self.min_error.visible = False
                self.max_error.visible = False
                self.rescale_error_btn.visible = False
                self.error_min_value = None
                self.error_max_value = None
        self.manual_error_scale.on_change('active', update_manual_error_scale)

        def update_error_min(attr, old, new):
            self.error_min_value = self.min_error.value
        def update_error_max(attr, old, new):
            self.error_max_value = self.max_error.value

        self.min_error.on_change('value', update_error_min)
        self.max_error.on_change('value', update_error_max)
        self.rescale_error_btn.on_click(lambda: self.rescale_errors())

        self.manual_diff_scale = CheckboxGroup(labels=['Manual Diff Scale 2'], active=[])
        self.min_diff = Spinner(value=-5, width=100, title="Diff Min")
        self.min_diff.visible = False
        self.max_diff = Spinner(value=5, width=100, title="Diff Max")
        self.max_diff.visible = False
        self.rescale_diff_btn = Button(label='Rescale', width=100)
        self.rescale_diff_btn.visible = False

        def update_manual_diff_scale(attr, old, new):
            if self.manual_diff_scale.active:
                self.min_diff.visible = True
                self.max_diff.visible = True
                self.rescale_diff_btn.visible = True
                self.diff_min_value = self.min_diff.value
                self.diff_max_value = self.max_diff.value
            else:
                self.min_diff.visible = False
                self.max_diff.visible = False
                self.rescale_diff_btn.visible = False
                self.diff_min_value = None
                self.diff_max_value = None
        self.manual_diff_scale.on_change('active', update_manual_diff_scale)

        def update_diff_min(attr, old, new):
            self.diff_min_value = self.min_diff.value
        def update_diff_max(attr, old, new):
            self.diff_max_value = self.max_diff.value
        self.min_diff.on_change('value', update_diff_min)
        self.max_diff.on_change('value', update_diff_max)
        self.rescale_diff_btn.on_click(lambda: self.rescale_diff())

        select_test1 = Select(title="Select Series1", options=["NONE"] + self.selected_lake.get_test_names(),
                              value=self.selected_test1.get_test_id() if self.selected_test1 else "NONE")
        select_test2 = Select(title="Select Series2", options=["NONE"] + self.selected_lake.get_test_names(),
                              value=self.selected_test2.get_test_id() if self.selected_test2 else "NONE")
        def update_test1(attr, old, new):
            self.selected_test1 = self.selected_lake.get_test(new)
            self.test_changed()
        select_test1.on_change('value', update_test1)
        def update_test2(attr, old, new):
            self.selected_test2 = self.selected_lake.get_test(new)
            self.test_changed()
        select_test2.on_change('value', update_test2)
        def update_plot_size(attr, old, new):
            self.plot_size = new
            self.lake_changed()
        select_size.on_change('value', update_plot_size)
        def update_lake(attr, old, new):
            self.time_index = 0
            self.selected_lake = self.lakes[new]
            avail_tests = ["NONE"] + self.selected_lake.get_test_names()
            select_test1.options = avail_tests
            select_test2.options = avail_tests
            if self.selected_test1 is not None:
                self.selected_test1 = self.selected_lake.get_test(self.selected_test1.getTestId())
            if self.selected_test2 is not None:
                self.selected_test2 = self.selected_lake.get_test(self.selected_test2.getTestId())
            select_test1.value = self.selected_test1.get_test_id() if self.selected_test1 else "NONE"
            select_test2.value = self.selected_test2.get_test_id() if self.selected_test2 else "NONE"
            self.selected_x = None
            self.selected_y = None
            self.lake_changed()
            self.update_metadata()
        select_lake.on_change('value', update_lake)
        hide_timeseries = CustomJS(args=dict(), code="hide_timeseries();")
        select_lake.js_on_change('value', hide_timeseries)
        select_size.js_on_change('value', hide_timeseries)
        select_test1.js_on_change('value', hide_timeseries)
        select_test2.js_on_change('value', hide_timeseries)

        if self.spatial_view1:
            spatial_row1 = self.spatial_view1.get_figures()
        else:
            spatial_row1 = [create_bokeh_div("", sz=8)]
        if self.spatial_view2:
            spatial_row2 = self.spatial_view2.get_figures()
        else:
            spatial_row2 = [create_bokeh_div("", sz=8)]

        aux_row = []
        if self.aux_view0:
            aux_row += self.aux_view0.get_figures()
        else:
            aux_row.append(create_bokeh_div("", sz=8))

        if self.aux_view1:
            aux_row += self.aux_view1.get_figures()
        else:
            aux_row.append(create_bokeh_div("", sz=8))
        
        layout = column(
            row(
                column(
                    row(select_lake, select_size, select_qa_filter),
                    row(self.manual_scale, self.rescale_btn, self.min_temp, self.max_temp),
                    row(self.manual_error_scale, self.rescale_error_btn, self.min_error, self.max_error),
                    row(self.manual_diff_scale, self.rescale_diff_btn, self.min_diff, self.max_diff)
                ),
                row(*aux_row, name="aux") # NEW: Additional independent plots added as separate rows.
            ),
            row(select_test1, self.output1_link, self.description1_txt),
            row(*spatial_row1, name="spatial1"),
            row(select_test2, self.output2_link, self.description2_txt),
            row(*spatial_row2, name="spatial2"),
            row(self.play_btn, self.spinner, self.slider, name="slider"),
            name="mainLayout", sizing_mode='stretch_both'
        )

        doc.add_root(layout)
        self.bind_other()
        self.status = ""

    def rescale(self):
        for view in [self.spatial_view1, self.spatial_view2, self.aux_view0]:
            if view:
                view.rescale_to(self.temp_min_value, self.temp_max_value, self.qa_filter)

    def rescale_errors(self):
        for view in [self.spatial_view1, self.spatial_view2]:
            if view:
                view.rescale_errors(self.error_min_value, self.error_max_value)

    def rescale_diff(self):
        for view in [self.spatial_view1, self.spatial_view2]:
            if view:
                view.rescale_diff(self.diff_min_value, self.diff_max_value)

    def setup_spatial_plots(self):
        button_width = 100
        spinner_width = 200
        (width_pixels, height_pixels) = get_plot_dimensions(self.plot_size)
        slider_width = (3 * width_pixels) - (button_width + spinner_width)
        self.slider = Slider(start=0, end=(len(self.times) - 1), value=self.time_index, step=1, title="",
                             width=slider_width, show_value=False)
        self.spinner = Spinner(value=self.time_index, width=spinner_width, low=0, high=len(self.times) - 1)
        select_time_slider = CustomJS(args=dict(slider=self.slider), code="""
            if (dg_value) {
                dg_value.setSelection(slider.value);
            }
            if (dg_diff) {
                dg_diff.setSelection(slider.value);
            }
        """)
        select_time_spinner = CustomJS(args=dict(spinner=self.spinner), code="""
            /* if (dg_value) {
                dg_value.setSelection(spinner.value);
            }
            if (dg_diff) {
                dg_diff.setSelection(spinner.value);
            } */
        """)
        self.play_btn = Button(label='Play', width=button_width)
        def play_handler():
            self.playing = not self.playing
            self.play_btn.label = "Stop" if self.playing else "Play"
        self.play_btn.on_click(play_handler)
        self.update_slider_title()
        def update(attr, old, new):
            self.slider.value = new
            self.spinner.value = new
            self.time_index = new
            update_time()
        def update_time():
            self.update_time()
        self.slider.js_on_change('value', select_time_slider)
        self.spinner.js_on_change('value', select_time_spinner)
        self.slider.on_change('value_throttled', update)
        self.spinner.on_change('value', update)

    def update_time(self):
        if self.spatial_view1:
            time = self.spatial_view1.set_time_index(self.time_index)
        if self.spatial_view2:
            time = self.spatial_view2.set_time_index(self.time_index)
        if self.aux_view0:
            self.aux_view0.set_time_index(self.time_index)
        if self.aux_view1:
            self.aux_view1.set_time_index(self.time_index)

        self.update_slider_title()

    def update_slider_title(self):
        days = self.times[self.time_index]
        dt = datetime.datetime(1981, 1, 1) + datetime.timedelta(days=days)
        dts = dt.strftime("%Y-%m-%d")
        self.slider.title = dts

    def update_metadata(self):
        md = mr.getMetadata(self.selected_lake.get_id())
        self.metadata_div.text = create_bokeh_div("%d / %s / %s / %0.2fN / %0.2fE" % (
            self.selected_lake.get_id(), md["name"], md["country"], md["lat"], md["lon"]),
            return_html_only=True, sz=16)

    def bind_other(self):
        # Bind primary views to each other (p0 and p1)
        if self.spatial_view1:
            if self.spatial_view2:
                self.spatial_view1.bind_other(self.spatial_view2.get_data())
            else:
                self.spatial_view1.bind_other(None)
        if self.spatial_view2:
            if self.spatial_view1:
                self.spatial_view2.bind_other(self.spatial_view1.get_data())
            else:
                self.spatial_view2.bind_other(None)

    def lake_changed(self):
        # Clear existing layouts
        curdoc().get_model_by_name('spatial1').children.clear()
        curdoc().get_model_by_name('spatial2').children.clear()
        curdoc().get_model_by_name('slider').children.clear()
        self.status = "Loading"
        
        # Process primary views (spatial_view1 and spatial_view2 remain unchanged)
        if self.spatial_view1:
            if self.spatial_view1.get_lake_name() != self.selected_lake.get_name() or \
               ((not self.selected_test1) or (self.spatial_view1.get_test_name() != self.selected_test1.get_test_id())):
                self.spatial_view1.close()
                self.spatial_view1 = self.load_data(self.selected_lake, self.selected_test1, 1)
        else:
            self.spatial_view1 = self.load_data(self.selected_lake, self.selected_test1, 1)
        if self.spatial_view1:
            self.spatial_view1.build(self.plot_size)
        
        if self.spatial_view2:
            if self.spatial_view2.get_lake_name() != self.selected_lake.get_name() or \
               ((not self.selected_test2) or (self.spatial_view2.get_test_name() != self.selected_test2.get_test_id())):
                self.spatial_view2.close()
                self.spatial_view2 = self.load_data(self.selected_lake, self.selected_test2, 2)
        else:
            self.spatial_view2 = self.load_data(self.selected_lake, self.selected_test2, 2)
        if self.spatial_view2:
            self.spatial_view2.build(self.plot_size)
        
        self.set_log_output_links()
        self.setup_spatial_plots()
        
        # Update layout for primary spatial views.
        for (tag, view) in [("spatial1", self.spatial_view1), ("spatial2", self.spatial_view2)]:
            r = curdoc().get_model_by_name(tag)
            if view:
                plots = view.get_figures()
                for plot in plots:
                    r.children.append(plot)
            else:
                r.children.append(create_bokeh_div("", sz=8))
        
        # Update layout for auxiliary views

        r = curdoc().get_model_by_name("aux")
        if r is None:
            r = row(name="aux")
            curdoc().add_root(r)
        r.children.clear()
        for view in [self.aux_view0, self.aux_view1]:
            if view:
                plots = view.get_figures()
                for plot in plots:
                    r.children.append(plot)
            else:
                r.children.append(create_bokeh_div("", sz=8))
                

        # # --- NEW: Align independent high-res views (p5 and p6) ---
        # if self.spatial_view3 and self.spatial_view4:
        #     import numpy as np
        #     # Convert lat/lon arrays to numpy arrays.
        #     lat3 = np.array(self.spatial_view3.lats)
        #     lon3 = np.array(self.spatial_view3.lons)
        #     lat4 = np.array(self.spatial_view4.lats)
        #     lon4 = np.array(self.spatial_view4.lons)
        #     # Determine the overlapping region.
        #     lat_min_common = max(lat3.min(), lat4.min())
        #     lat_max_common = min(lat3.max(), lat4.max())
        #     lon_min_common = max(lon3.min(), lon4.min())
        #     lon_max_common = min(lon3.max(), lon4.max())
        #     logger.info("Common lat range: %.2f to %.2f; common lon range: %.2f to %.2f",
        #                 lat_min_common, lat_max_common, lon_min_common, lon_max_common)
        #     # For spatial_view3 (p5): find indices in its grid within the common region.
        #     idx_lat3 = np.where((lat3 >= lat_min_common) & (lat3 <= lat_max_common))[0]
        #     idx_lon3 = np.where((lon3 >= lon_min_common) & (lon3 <= lon_max_common))[0]
        #     if idx_lat3.size > 0 and idx_lon3.size > 0:
        #         self.spatial_view3.present_y_min = int(idx_lat3.min())
        #         self.spatial_view3.present_y_max = int(idx_lat3.max())
        #         self.spatial_view3.present_x_min = int(idx_lon3.min())
        #         self.spatial_view3.present_x_max = int(idx_lon3.max())
        #         self.spatial_view3.lats = lat3[idx_lat3].tolist()
        #         self.spatial_view3.lons = lon3[idx_lon3].tolist()
        #     # For spatial_view4 (p6): do the same.
        #     idx_lat4 = np.where((lat4 >= lat_min_common) & (lat4 <= lat_max_common))[0]
        #     idx_lon4 = np.where((lon4 >= lon_min_common) & (lon4 <= lon_max_common))[0]
        #     if idx_lat4.size > 0 and idx_lon4.size > 0:
        #         self.spatial_view4.present_y_min = int(idx_lat4.min())
        #         self.spatial_view4.present_y_max = int(idx_lat4.max())
        #         self.spatial_view4.present_x_min = int(idx_lon4.min())
        #         self.spatial_view4.present_x_max = int(idx_lon4.max())
        #         self.spatial_view4.lats = lat4[idx_lat4].tolist()
        #         self.spatial_view4.lons = lon4[idx_lon4].tolist()
        #     # Check grid shapes and trim one extra row/column if needed.
        #     shape3 = (len(self.spatial_view3.lats), len(self.spatial_view3.lons))
        #     shape4 = (len(self.spatial_view4.lats), len(self.spatial_view4.lons))
        #     if shape4[0] > shape3[0]:
        #         self.spatial_view4.lats = self.spatial_view4.lats[:shape3[0]]
        #         self.spatial_view4.present_y_max = self.spatial_view4.present_y_min + shape3[0] - 1
        #     if shape4[1] > shape3[1]:
        #         self.spatial_view4.lons = self.spatial_view4.lons[:shape3[1]]
        #         self.spatial_view4.present_x_max = self.spatial_view4.present_x_min + shape3[1] - 1
        #     if shape3[0] > shape4[0]:
        #         self.spatial_view3.lats = self.spatial_view3.lats[:shape4[0]]
        #         self.spatial_view3.present_y_max = self.spatial_view3.present_y_min + shape4[0] - 1
        #     if shape3[1] > shape4[1]:
        #         self.spatial_view3.lons = self.spatial_view3.lons[:shape4[1]]
        #         self.spatial_view3.present_x_max = self.spatial_view3.present_x_min + shape4[1] - 1
        #     # Rebuild both independent views (p5 and p6) with the new aligned grid.
        #     self.spatial_view3.build("small")
        #     self.spatial_view4.build("small")
        # --- End of NEW block ---
        
        # Rebuild the slider container.
        slider_container = curdoc().get_model_by_name('slider')
        slider_container.children.clear()
        slider_container.children.append(self.spinner)
        slider_container.children.append(self.play_btn)
        slider_container.children.append(self.slider)
        self.status = ""
        gc.collect()
        self.bind_other()



    def test_changed(self):
        self.lake_changed()
        if self.selected_y is not None and self.selected_x is not None:
            if self.spatial_view1 is not None:
                self.spatial_view1.select_pixel(self.selected_x, self.selected_y)
            if self.spatial_view2 is not None:
                self.spatial_view2.select_pixel(self.selected_x, self.selected_y)
            self.refresh_time_series = True

    def update(self):
        if self.request_index >= 0:
            self.slider.value = self.request_index
            self.spinner.value = self.request_index
            self.time_index = self.request_index
            if self.spatial_view1:
                self.spatial_view1.set_time_index(self.time_index)
            if self.spatial_view2:
                self.spatial_view2.set_time_index(self.time_index)
            self.update_slider_title()
            self.request_index = -1
        elif self.playing:
            if self.time_index + 1 < len(self.times):
                self.time_index += 1
                self.slider.value = self.time_index
                self.spinner.value = self.time_index
                self.update_time()
            else:
                self.playing = False
                self.play_btn.label = "Play"

    def pick_index(self, index):
        self.request_index = index

    def select_pixel(self, x, y):
        if self.spatial_view1:
            self.spatial_view1.select_pixel(x, y)
        if self.spatial_view2:
            self.spatial_view2.select_pixel(x, y)
        self.selected_x = x
        self.selected_y = y

    def get_timeseries_at(self, x_index, y_index, eof_var="temporal_eof0"):
        self.status = "Loading Time Series"
        ts1 = []
        ts2 = []
        quality_levels = []
        columns_temps = ["date", "original"]
        columns_diffs = ["date"]
        if self.spatial_view1:
            ts1 = self.spatial_view1.get_local_timeseries_at(x_index, y_index)
        columns_temps.append("series1")
        columns_diffs.append("series1_diff")
        columns_diffs.append("series1_minus_series2")
        if self.spatial_view2:
            ts2 = self.spatial_view2.get_local_timeseries_at(x_index, y_index)
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
            if eof_var in self.eof_mappings:
                eof_value = self.eof_mappings[eof_var].get(dts, "")
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

    def get_data(self):
        return self.ds

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
