
import argparse
import xarray as xr
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, BasicTicker, BasicTickFormatter
from bokeh.io import export_svgs
import numpy as np
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.palettes import Category10_10, Category20_20
from bokeh.models import Legend
import os
import csv
import base64

from processors.data.metadata_reader import MetadataReader

dygraph_template = """
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dineof Lake Surface Water Temperature Results</title>
    <script src="//cdnjs.cloudflare.com/ajax/libs/dygraph/2.1.0/dygraph.min.js"></script>
    <link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/dygraph/2.1.0/dygraph.min.css" />

    <script>
        function boot() {
            var data = "{{CSV}}";
            var dg = new Dygraph(
                  document.getElementById("dg_div"),
                  data,
                  {
                    title: "Temporal EOFs",
                    ylabel: 'Value',
                    legend: 'always',
                    showRangeSelector: true,
                    rangeSelectorHeight: 30,
                    rangeSelectorPlotStrokeColor: 'blue',
                    rangeSelectorPlotFillColor: 'lightyellow',
                    pointSize: 4,
                    plugins: [],
                    labelsKMB: true
                 });


          toggleSeries = function(nr) {
              var checked = document.getElementById("eof"+nr).checked;
              dg.setVisibility(nr-1, checked);
          }

          for(var i=2; i<12; i+=1) {
              dg.setVisibility(i, false);
          }
       }
    </script>
    <style>
        table, td, th {
            border: 1px solid black;
        }

        table {
            border-collapse: collapse;
        }
    </style>

</head>

<body onload="boot();" style="font-family:Helvetica, Arial, sans-serif;">
  <div>
      <h3>{{TITLE}}</h3>
  </div>
  <h2>Metadata</h2>
  <table>
    {{METADATA-TABLE}}
  </table>
  <h2>Spatial EOFs</h1>
  <table>
    {{SPATIAL-EOFS-TABLE}}
  </table>
  <h2>Temporal EOFs</h1>
  <table>
        <tr>
            <td colspan="12">Select Temporal EOFs to display</td>
        </tr>
        <tr>
            <td>1</td>
            <td>2</td>
            <td>3</td>
            <td>4</td>
            <td>5</td>
            <td>6</td>
            <td>7</td>
            <td>8</td>
            <td>9</td>
            <td>10</td>
            <td>11</td>
            <td>12</td>
        </tr>
        <tr>
            <td><input id="eof1" type="checkbox" checked onclick="toggleSeries(1)"/></td>
            <td><input id="eof2" type="checkbox" checked onclick="toggleSeries(2)"/></td>
            <td><input id="eof3" type="checkbox" onclick="toggleSeries(3)"/></td>
            <td><input id="eof4" type="checkbox" onclick="toggleSeries(4)"/></td>
            <td><input id="eof5" type="checkbox" onclick="toggleSeries(5)"/></td>
            <td><input id="eof6" type="checkbox" onclick="toggleSeries(6)"/></td>
            <td><input id="eof7" type="checkbox" onclick="toggleSeries(7)"/></td>
            <td><input id="eof8" type="checkbox" onclick="toggleSeries(8)"/></td>
            <td><input id="eof9" type="checkbox" onclick="toggleSeries(9)"/></td>
            <td><input id="eof10" type="checkbox" onclick="toggleSeries(10)"/></td>
            <td><input id="eof11" type="checkbox" onclick="toggleSeries(11)"/></td>
            <td><input id="eof12" type="checkbox" onclick="toggleSeries(12)"/></td>
        </tr>
    </table>
  <div id="dg_div" style="width:100%; height:300px;">
  </div>
</body>
</html>
"""

style = """
table { border: 2px solid black;  border-collapse: collapse; }
th { padding: 5px; border: 1px solid black; }
td { padding: 5px; border: 1px solid black; }
"""


def getDataURI(path):
    data = open(path,"rb").read()
    encoded = base64.b64encode(data).decode("ascii")
    return "data:image/png;base64,"+encoded

mr = MetadataReader()
mr.load()

class MetadataTable(object):

    def __init__(self,lake_id, test_id, prep_options):
        self.lake_id = lake_id
        self.test_id = test_id
        self.prep_options = prep_options

    def get_html(self):
        html = "<thead><tr>"
        for fixed_column_name in ["Lake Id","Test Id", "Name","Country","Data Prep Options"]:
            html += "<th>%s</th>" % (fixed_column_name)
        html += "</tr></thead>\n"
        html += "<tbody>"
        metadata = mr.getMetadata(self.lake_id)
        html += "<tr>"
        html += "<td>%s</td>" % (self.lake_id)
        html += "<td>%s</td>" % (self.test_id)
        html += "<td>%s</td>" % (metadata["name"])
        html += "<td>%s</td>" % (metadata["country"])
        html += "<td>%s</td>" % (self.prep_options)
        html += "</tr>\n"
        html += "</tbody>"
        return html


class SpatialEOFSTable(object):

    def __init__(self):
        self.rows = []

    def add_eof(self, name, image_path):
        self.rows.append((name,image_path))

    def get_html(self):
        html = "<thead><tr>"
        for fixed_column_name in ["Spatial EOF", "Plot"]:
            html += "<th>%s</th>" % (fixed_column_name)
        html += "</tr></thead>\n"
        html += "<tbody>"
        for (name,image_path) in self.rows:
            html += "<tr>"
            html += "<td>%s</td>" % name
            html += "<td><img src=\"%s\"></td>" % image_path
            html += "</tr>\n"
        html += "</tbody>"
        return html

class PlotExporter(object):

    def __init__(self,output_folder,eof_folder,eof_file_name, lake_id, test_id, prep_options):
        self.output_folder = output_folder
        self.eof_folder = eof_folder
        self.eof_file_name = eof_file_name
        self.lake_id = lake_id
        self.test_id = test_id
        self.prep_options = prep_options

    def run(self):

        os.makedirs(self.output_folder,exist_ok=True)

        ds = xr.open_dataset(os.path.join(self.eof_folder,self.eof_file_name))

        keys = ds.variables.keys()

        key = "eigenvalues"
        data = ds.variables[key].data[:].tolist()
        times = ds.coords["eofs"].data[:].tolist()
        p = figure(title=key, y_range=(0, max(data)), width=768, height=256)
        p.line(times, data, line_width=4)
        p.circle(times, data, size=8, fill_color="red")
        p.xaxis.visible = False
        p.toolbar.logo = None
        p.toolbar_location = None
        p.output_backend = "svg"

        output_path = os.path.join(self.output_folder, key + ".svg")
        export_svgs(p, filename=output_path)


        data_min = 0.0
        data_max = 0.0
        for key in keys:
            if key.startswith("temporal_eof"):
                data = ds.variables[key].data[:].tolist()
                data_min = min(data_min,min(data))
                data_max = max(data_max, max(data))
        p = figure(title=key, y_range=(data_min, data_max), width=768, height=256)
        p.output_backend = "svg"

        palette = Category10_10

        legend_items = []
        csvdata = {}
        eofnames = []
        for key in keys:
            if key.startswith("temporal_eof"):
                eof = int(key[len("temporal_eof"):])
                data = ds.variables[key].data[:].tolist()
                times = ds.coords["time"].data[:].tolist()
                if "times" not in csvdata:
                    csvdata["times"] = times
                eofname = "eof%02d"%(eof+1)
                eofnames.append(eofname)
                csvdata[eofname] = data
                if eof < 2:
                    colour = palette[eof]
                    i = p.line(times, data, line_width=2, color=colour)
                    legend_items.append((eofname,[i]))

        legend = Legend(items=legend_items, location="center")
        p.add_layout(legend, 'right')

        p.toolbar.logo = None
        p.toolbar_location = None
        p.output_backend = "svg"
        output_path = os.path.join(self.output_folder, "temporal_eofs.svg")

        print("Exporting temporal EOF (first 2 EOFs) plot to %s" % (output_path))

        export_svgs(p, filename=output_path)

        eofnames = sorted(eofnames)
        csv = ",".join(["t"]+eofnames)
        for idx in range(len(csvdata["times"])):
            csv += "\\n"
            csv += str(csvdata["times"][idx])
            for eofname in eofnames:
                csv += ","
                csv += str(csvdata[eofname][idx])

        metadata_table = MetadataTable(self.lake_id, self.test_id, self.prep_options)
        spatial_eofs_table = SpatialEOFSTable()
        title = "All EOFS"

        data_min = 0.0
        data_max = 0.0
        for key in keys:
            if key.startswith("spatial_eof"):
                data = ds.variables[key].data[:,:]
                minv = np.nanmin(data)
                maxv = np.nanmax(data)
                data_min = min(data_min,minv)
                data_max = max(data_max,maxv)
        color_mapper = LinearColorMapper(palette="Viridis256", low=data_min, high=data_max, nan_color="lightgrey")

        for key in keys:
            if key.startswith("spatial_eof"):
                data = ds.variables[key].data[:,:]
                (height,width) = data.shape
                cds = ColumnDataSource(data=dict(image=[data]))
                p1 = figure(title=key, match_aspect=True, width=512)
                p1.image(image="image", x=0, y=0, dw=width, dh=height,source=cds,color_mapper=color_mapper)
                p1.axis.visible = False
                p1.toolbar.logo = None
                p1.toolbar_location = None
                p1.output_backend = "svg"
                color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(desired_num_ticks=10), formatter=BasicTickFormatter(),
                             label_standoff=12, border_line_color=None,location=(0,0))

                p1.add_layout(color_bar, 'right')
                image_filename = key+".svg"
                output_path = os.path.join(self.output_folder,image_filename)
                print("Exporting plot to %s"%(output_path))
                export_svgs(p1,filename=output_path)
                spatial_eofs_table.add_eof(key,image_filename)


        output_html_path = os.path.join(self.output_folder, "index.html")
        print("Exporting temporal EOF / spatial EOF plot (all EOFs) to %s" % (output_html_path))
        open(output_html_path, "w").write(dygraph_template
                                          .replace("{{CSV}}", csv)
                                          .replace("{{TITLE}}", title)
                                          .replace("{{METADATA-TABLE}}", metadata_table.get_html())
                                          .replace("{{SPATIAL-EOFS-TABLE}}", spatial_eofs_table.get_html()))




