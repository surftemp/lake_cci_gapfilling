// Global variables to hold Dygraph objects, current coordinates, and chosen EOF variable
var dg_value = null;          // Dygraph object for temperature time series
var dg_diff = null;           // Dygraph object for temperature difference time series
var current_x = 0;            // Current x-coordinate (pixel index) for time series extraction
var current_y = 0;            // Current y-coordinate (pixel index) for time series extraction
var chosenEofVar = "temporal_eof0";  // Default EOF (Empirical Orthogonal Function) variable

// Arrays to hold current longitude and latitude values used for tooltips and computations
var current_lons = [];
var current_lats = [];

// Colour definitions for quality levels and data series display
var low_quality_colour = "red";
var medium_quality_colour = "orange";
var high_quality_colour = "green";

var original_data_colour = "lightgray";
var series1_colour = "pink";
var series2_colour = "blue";

// ------------------------------------------------------------
// Helper function to plot a circle with an outline based on data quality.
// ctx: canvas drawing context, (cx,cy): center, quality: data quality, color: fill color, radius: circle radius.
var point_plot = function(ctx, cx, cy, quality, color, radius) {
    ctx.lineWidth = 4;            // Set line width for the circle outline
    ctx.fillStyle = color;        // Set fill color as passed
    // Determine stroke color based on quality thresholds
    if (quality <= 1) {
        ctx.strokeStyle = low_quality_colour;
    } else if (quality == 2) {
        ctx.strokeStyle = medium_quality_colour;
    } else {
        ctx.strokeStyle = high_quality_colour;
    }
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2, false);  // Draw circle at (cx, cy)
    ctx.closePath();
    ctx.stroke();                 // Draw the circle outline
    ctx.fill();                   // Fill the circle with the set color
}

// ------------------------------------------------------------
// Helper function to plot a small dot without an outline.
// ctx: canvas context, (cx,cy): center, color: fill color, radius: dot radius.
var dot_plot = function(ctx, cx, cy, color, radius) {
    ctx.lineWidth = 0;            // No outline
    ctx.fillStyle = color;        // Set fill color
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2, false);
    ctx.closePath();
    ctx.fill();                   // Draw the filled dot
}

// ------------------------------------------------------------
// Compute the latitude or longitude from a pixel index using an array of values.
// Returns the value formatted to three decimal places, or "?" if out-of-bounds.
function compute_latlon(pixel, values) {
    var i = Math.floor(pixel);
    if (i < 0 || i >= values.length) {
        return "?";
    } else {
        return values[i].toFixed(3);
    }
}

// ------------------------------------------------------------
// Retrieve the session ID from the first Bokeh document's title.
// This is used to construct URLs for server endpoints.
function getSessionId() {
    try {
        return Bokeh.documents[0]._title;
    } catch(e) {
        return "no_session";
    }
}

// ------------------------------------------------------------
// Update the status display on the web page.
// Changes the opacity of the main content if a status message is present,
// and triggers a time series refresh if requested.
function display_status(obj) {
    var status = document.getElementById("status");
    var main = document.getElementById("main");
    var msg = ("status" in obj) ? obj["status"] : "Unable to reach server";
    if (msg != "") {
        msg = " - " + msg;
        main.setAttribute("style", "opacity:0.3;");
    } else {
        main.setAttribute("style", "opacity:1.0;");
    }
    status.innerHTML = "";
    status.appendChild(document.createTextNode(msg));
    if (obj["refresh_time_series"]) {
        refresh_timeseries();
    }
}

// ------------------------------------------------------------
// Fetch the current status from the server and display it.
function refresh_status() {
    fetch("/sessions/" + getSessionId() + "/status")
        .then(r => r.json(), e => display_status({"status": "Unable to connect to server"}))
        .then(obj => display_status(obj));
}

// ------------------------------------------------------------
// Start an interval loop to refresh status every 2 seconds.
// Also draws example points on canvases for the quality legend.
function start_status_loop() {
    window.setInterval(refresh_status, 2000);

    // Draw example points for medium and high quality in the legend canvases
    point_plot(document.getElementById("medium_quality_canvas_legend").getContext("2d"), 10, 10, 2, "white", 4);
    point_plot(document.getElementById("high_quality_canvas_legend").getContext("2d"), 10, 10, 3, "white", 4);
}

// ------------------------------------------------------------
// Hide the time series panels and quality legend elements.
function hide_timeseries() {
    document.getElementById("quality_legend").setAttribute("style", "display:none;");
    document.getElementById("ts_value").innerHTML = "";
    document.getElementById("ts_diff").innerHTML = "";
    document.getElementById("ts_value_controls").setAttribute("style", "display:none;");
    document.getElementById("ts_diff_controls").setAttribute("style", "display:none;");
}

// ------------------------------------------------------------
// Called when the EOF selection dropdown changes.
// Updates the chosen EOF variable and refreshes the EOF graph.
function selectEofChanged() {
    console.log("selectEofChanged triggered");
    var selectElem = document.getElementById("eof_select");
    chosenEofVar = selectElem.value; 
    console.log("User picked EOF var =", chosenEofVar);    
    refresh_eof();
}

// ------------------------------------------------------------
// Refresh the time series graphs using the current pixel coordinates and chosen EOF variable.
function refresh_timeseries() {
    if (!chosenEofVar) {
        chosenEofVar = "temporal_eof0";
    }
    console.log("current_lats.length =", current_lats.length, ", current_lons.length =", current_lons.length);
    console.log("refresh_timeseries() - chosenEofVar =", chosenEofVar);
    console.log("Calling fetch_timeseries...");
    fetch_timeseries(current_x, current_y, current_lons, current_lats, chosenEofVar);
}

// ------------------------------------------------------------
// Request the server to change the current time index (e.g., when a point is clicked).
function pick_time(index) {
    var params = new URLSearchParams({
         "index": "" + index
    });
    fetch("/sessions/" + getSessionId() + "/pick_time?" + params.toString()).then(r => {});
}

// ------------------------------------------------------------
// Custom plotter function for Dygraphs to render histogram bars.
// Calculates bar width from the spacing of data points and draws a rectangle for each.
function barChartPlotter(e) {
    var ctx = e.drawingContext;
    ctx.save();
    var points = e.points;
    var yZero = e.dygraph.toDomYCoord(0); // y-coordinate for zero value
    var barWidth = points.length > 1 ? Math.max(4, points[1].canvasx - points[0].canvasx - 2) : 10;
  
    for (var i = 0; i < points.length; i++) {
        var p = points[i];
        var left = p.canvasx - barWidth / 2;
        var top = p.canvasy;
        var height = yZero - p.canvasy;
        ctx.fillStyle = e.color;
        ctx.fillRect(left, top, barWidth, height);
    }
    ctx.restore();
}

// ------------------------------------------------------------
// Build a histogram from EOF CSV data and render it using Dygraphs.
// Splits CSV data into bins and creates a data array for the histogram.
function createHistogram(eofCSV) {
    var lines = eofCSV.split("\n");  // Split CSV into lines
    var values = [];
    for (var i = 1; i < lines.length; i++) {
        if (lines[i].trim() === "") continue;
        var parts = lines[i].split(",");
        if (parts.length < 2) continue;
        var val = parseFloat(parts[1]);
        if (!isNaN(val)) {
            values.push(val);
        }
    }
    if (values.length === 0) return;
  
    var numBins = 20;  // Number of bins for histogram
    var minVal = Math.min(...values);
    var maxVal = Math.max(...values);
    var binSize = (maxVal - minVal) / numBins;
    var bins = new Array(numBins).fill(0);
  
    values.forEach(function(val) {
        var binIndex = Math.floor((val - minVal) / binSize);
        if (binIndex === numBins) binIndex = numBins - 1;
        bins[binIndex]++;
    });
  
    var histData = [];
    for (var i = 0; i < numBins; i++) {
        var binCenter = minVal + binSize * (i + 0.5);
        histData.push([binCenter, bins[i]]);
    }
  
    new Dygraph(
        document.getElementById("eof_hist"),
        histData,
        {
            title: "EOF Histogram",
            xlabel: "EOF Value",
            ylabel: "Frequency",
            strokeWidth: 0,
            drawPoints: false,
            plotter: barChartPlotter,
            height: 300
        }
    );
}

// ------------------------------------------------------------
// Fetch timeseries data for a given pixel (x, y) from the server,
// update UI elements, and create Dygraphs for temperatures, differences, and EOF.
function fetch_timeseries(x, y, lons, lats, eof_var) {
    console.log("fetch_timeseries() called. Stack trace:\n", new Error().stack);
    console.log("fetch_timeseries() - x=", x, "y=", y, "eof_var=", eof_var);

    var params = new URLSearchParams({
        "x": x,
        "y": y
    });

    // Convert pixel indices to lat/lon values for display in the graph title.
    var lat = lats[Math.floor(y)];
    var lon = lons[Math.floor(x)];
    var lonlat = "(lon,lat) = (" + lat.toFixed(3) + "," + lon.toFixed(3) + ")";

    // Show the time series control panels and quality legend.
    document.getElementById("ts_value_controls").style.display = "block";
    document.getElementById("ts_diff_controls").style.display = "block";
    document.getElementById("quality_legend").style.display = "block";

    // Fetch timeseries data from the server endpoint.
    fetch("/sessions/" + getSessionId() + "/timeseries?" + params.toString())
        .then(response => response.json())
        .then(data => {

            // Callback function to draw individual points on Dygraphs.
            var point_draw_cb = function(g, seriesName, ctx, cx, cy, color, radius, idx) {
                var ql = data["quality_levels"][idx];
                if (seriesName === "original") {
                    point_plot(ctx, cx, cy, ql, color, radius);
                } else {
                    dot_plot(ctx, cx, cy, color, 3);
                }
            };

            // Create a Dygraph for the temperature time series.
            dg_value = new Dygraph(
                document.getElementById("ts_value"),
                data["temps"],  // CSV string with temperature data
                {
                    title: 'Lake Surface Water Temperatures - Time Series for ' + lonlat,
                    ylabel: 'Water Temperature (K)',
                    legend: 'always',
                    showRangeSelector: true,
                    rangeSelectorHeight: 30,
                    rangeSelectorPlotStrokeColor: 'blue',
                    rangeSelectorPlotFillColor: 'lightyellow',
                    colors: [original_data_colour, series1_colour, series2_colour],
                    pointSize: 4,
                    plugins: [
                        new Dygraph.Plugins.Crosshair({ direction: "vertical" })
                    ],
                    series: {
                        'original': { strokeWidth: 0 },
                        'series1':   {},
                        'series2':   {}
                    },
                    labelsKMB: true,
                    clickCallback: function(e, x, points) {
                        pick_time(points[0]["idx"]);
                    },
                    drawPointCallback: point_draw_cb
                }
            );

            // Create a Dygraph for the difference time series.
            dg_diff = new Dygraph(
                document.getElementById("ts_diff"),
                data["diffs"],  // CSV string with difference data
                {
                    title: 'Lake Surface Water Temperature Differences - Time Series for ' + lonlat,
                    drawAxesAtZero: true,
                    ylabel: 'Reconstruction Diff (K)',
                    legend: 'always',
                    showRangeSelector: true,
                    rangeSelectorHeight: 30,
                    rangeSelectorPlotStrokeColor: 'blue',
                    rangeSelectorPlotFillColor: 'lightyellow',
                    colors: [series1_colour, series1_colour, series2_colour, series2_colour],
                    pointSize: 4,
                    plugins: [
                        new Dygraph.Plugins.Crosshair({ direction: "vertical" })
                    ],
                    series: {
                        'series1_diff':          { strokeWidth: 0 },
                        'series1_minus_series2': {},
                        'series2_diff':          { strokeWidth: 0 },
                        'series2_minus_series1': {}
                    },
                    labelsKMB: true,
                    clickCallback: function(e, x, points) {
                        pick_time(points[0]["idx"]);
                    },
                    drawPointCallback: point_draw_cb
                }
            );

            // Create a Dygraph for the EOF time series.
            eof = new Dygraph(
                document.getElementById("eof"),
                data["eof"],  // CSV string with EOF data
                {
                    title: "Temporal Eof (" + eof_var + ")",
                    drawAxesAtZero: true,
                    ylabel: 'EOF Value',
                    legend: 'always',
                    showRangeSelector: true,
                    rangeSelectorHeight: 30,
                    rangeSelectorPlotStrokeColor: 'blue',
                    rangeSelectorPlotFillColor: 'lightyellow',
                    colors: ["red"],
                    pointSize: 4,
                    plugins: [
                        new Dygraph.Plugins.Crosshair({ direction: "vertical" })
                    ],
                    series: {
                        'eof': { strokeWidth: 2 }
                    },
                    labelsKMB: true,
                    clickCallback: function(e, x, points) {
                        pick_time(points[0]["idx"]);
                    },
                    drawPointCallback: point_draw_cb
                }
            );
            
            // Create a histogram for the EOF values.
            createHistogram(data["eof"]);

            // Hide specific series in the difference graph by default.
            dg_diff.setVisibility(1, false); // Hide series1_minus_series2
            dg_diff.setVisibility(3, false); // Hide series2_minus_series1

            // Synchronize the range of the three Dygraphs.
            Dygraph.synchronize(dg_value, dg_diff, eof, { "range": false });

            // Update series visibility based on user toggle settings.
            toggleSeries();
        });
}

// ------------------------------------------------------------
// Refresh the EOF graph when the EOF variable changes.
// Fetches updated timeseries data with the new EOF and updates the Dygraph accordingly.
function refresh_eof() {
    var params = new URLSearchParams({
        "eof_var": chosenEofVar
    });
    
    fetch("/sessions/" + getSessionId() + "/timeseries?" + params.toString())
        .then(response => response.json())
        .then(data => {
            if (typeof eof !== "undefined" && eof) {
                eof.updateOptions({
                    file: data["eof"],
                    title: "Temporal EOF (" + chosenEofVar + ")"
                });
            } else {
                eof = new Dygraph(
                    document.getElementById("eof"),
                    data["eof"],
                    {
                        title: "Temporal EOF (" + chosenEofVar + ")",
                        drawAxesAtZero: true,
                        ylabel: 'EOF Value',
                        legend: 'always',
                        showRangeSelector: true,
                        rangeSelectorHeight: 30,
                        rangeSelectorPlotStrokeColor: 'blue',
                        rangeSelectorPlotFillColor: 'lightyellow',
                        colors: ["red"],
                        pointSize: 4,
                        plugins: [ new Dygraph.Plugins.Crosshair({ direction: "vertical" }) ],
                        series: { 'eof': { strokeWidth: 2 } },
                        labelsKMB: true,
                        clickCallback: function(e, x, points) {
                            pick_time(points[0]["idx"]);
                        }
                    }
                );
            }
            createHistogram(data["eof"]);
        })
        .catch(error => {
            console.error("Error in refresh_eof:", error);
        });
}

// ------------------------------------------------------------
// Toggle visibility of various time series in the Dygraphs based on user control toggles.
// Ensures that certain series (e.g. series differences) are mutually exclusive.
function toggleSeries(toggle_id) {
    var original_toggle = document.getElementById("o_toggle");
    var series1_toggle = document.getElementById("s1_toggle");
    var series1e_toggle = document.getElementById("s1e_toggle");
    var series1d_toggle = document.getElementById("s1d_toggle");
    var series2_toggle = document.getElementById("s2_toggle");
    var series2e_toggle = document.getElementById("s2e_toggle");
    var series2d_toggle = document.getElementById("s2d_toggle");

    // Ensure that series1d and series2d toggles are mutually exclusive.
    if (toggle_id == "s1d_toggle") {
        if (series1d_toggle.checked) {
            series2d_toggle.checked = false;
        }
    } else if (toggle_id == "s2d_toggle") {
        if (series2d_toggle.checked) {
            series1d_toggle.checked = false;
        }
    }
    // Set visibility for the temperature time series.
    dg_value.setVisibility(0, original_toggle.checked);
    dg_value.setVisibility(1, series1_toggle.checked);
    dg_value.setVisibility(2, series2_toggle.checked);

    // Set visibility for the difference time series.
    dg_diff.setVisibility(0, series1e_toggle.checked);
    dg_diff.setVisibility(1, series1d_toggle.checked);
    dg_diff.setVisibility(2, series2e_toggle.checked);
    dg_diff.setVisibility(3, series2d_toggle.checked);
}
