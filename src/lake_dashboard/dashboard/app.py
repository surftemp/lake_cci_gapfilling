"""
Flask-based app that provides web services for the application, delegating to the Bokeh server when appropriate.
This module sets up a Flask web server, defines endpoints for embedding the Bokeh dashboard, fetching timeseries data,
status updates, and time index changes. It also parses command-line arguments to configure the dashboard.
"""

import argparse
import os
import glob
from threading import Thread
from .config import Config
from .dashboard import Dashboard

from bokeh.embed import server_document
from flask import Flask, render_template_string, jsonify, request

# Create the Flask application instance
app = Flask(__name__)

# Import additional utilities: logger creation, session management, and Bokeh worker functions.
from .bokeh_utils import create_logger, SessionManager, bk_worker, configure_bokeh_logging

# ------------------------------------------------------------
# Helper function to load file content relative to this file's location.
def load_relative(relpath):
    basedir = os.path.split(__file__)[0]
    loadpath = os.path.join(basedir, relpath)
    with open(loadpath) as f:
        return f.read()

# Load the HTML template and JavaScript file used for the dashboard.
template = load_relative("dashboard.html")
templatejs = load_relative("dashboard.js")

# Create a logger for this module.
logger = create_logger(__name__)

# ------------------------------------------------------------
# Route for the dashboard page.
@app.route('/', methods=['GET'])
def bkapp_page():
    # Generate a script tag for embedding the Bokeh server document.
    script = server_document('http://%s:%d/bkapp' % (Config.external_address, Config.bokeh_port))
    # Render the HTML template with the Bokeh script embedded.
    return render_template_string(template, script=script, template="Flask")

# ------------------------------------------------------------
# Route to serve the dashboard JavaScript.
@app.route('/dashboard.js', methods=['GET'])
def sendjs():
    return render_template_string(templatejs)

# ------------------------------------------------------------
# Route to retrieve time series data for a specific session and pixel location.
@app.route("/sessions/<session_id>/timeseries", methods=['GET'])
def get_timeseries(session_id):
    # Retrieve the Dashboard instance for the given session.
    dashb = SessionManager.get_session(session_id)
    # Try to get the x and y pixel indices from the request arguments;
    # default to 0 if not provided.
    x_param = request.args.get("x")
    y_param = request.args.get("y")
    if x_param is None or y_param is None:
        x_index = 0
        y_index = 0
    else:
        x_index = int(float(x_param))
        y_index = int(float(y_param))
    
    # Get the EOF variable from the query parameters, defaulting to "temporal_eof0".
    eof_var = request.args.get("eof_var", default="temporal_eof0")

    # Get the time series data from the dashboard object.
    tsdata = dashb.get_timeseries_at(x_index, y_index, eof_var)
    return jsonify(tsdata)

# ------------------------------------------------------------
# Route to get the current status of a session.
@app.route("/sessions/<session_id>/status", methods=['GET'])
def get_status(session_id):
    # If there is no session, return an empty status.
    if session_id == "no_session":
        status = {"status": ""}
    else:
        # Retrieve the dashboard instance.
        dashb = SessionManager.get_session(session_id)
        if dashb:
            # Return the status and whether the timeseries should be refreshed.
            status = {"status": dashb.get_status(), "refresh_time_series": dashb.get_refresh_timeseries()}
        else:
            # If the session is disconnected, inform the client.
            status = {"status": "Session disconnected (refresh page to start new session)"}
    return jsonify(status)

# ------------------------------------------------------------
# Route to handle time index change requests.
@app.route("/sessions/<session_id>/pick_time", methods=['GET'])
def pick_time(session_id):
    # Retrieve the dashboard session.
    dashb = SessionManager.get_session(session_id)
    # Get the time index from the request parameters.
    index = int(request.args.get("index"))
    # Set the requested time index in the dashboard.
    dashb.pick_index(index)
    return jsonify({"index": index})

# ------------------------------------------------------------
# Add headers to each response to disable caching.
@app.after_request
def add_header(r):
    # Prevent browsers from caching responses.
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# ------------------------------------------------------------
# Main entry point for the Flask app.
def main():
    # Configure logging for Bokeh.
    configure_bokeh_logging()
    
    # Set up argument parsing for command-line configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="The host name for the dashboard service", default="localhost")
    parser.add_argument("--port", help="The port number for the dashboard service", type=int, default=8999)
    parser.add_argument("--data", nargs="+", required=True, help="provide paths to one or more datasets to compare")

    # Optional arguments for EOF file and variable names.
    parser.add_argument("--eof-file", required=False, default=None,
                        help="Path to the EOF netCDF file (eofs.nc).")
    parser.add_argument("--temperature-variable", default="lake_surface_water_temperature",
                        help="the variable containing gappy LSWT observations")
    parser.add_argument("--filled-temperature-variable", default="temp_filled",
                        help="the variable containing reconstructed LSWT values")
    parser.add_argument("--quality-level-variable", default="quality_level",
                        help="the variable containing quality flags")

    parser.add_argument("--lake-mask-variable", default="lakeid_GloboLakes",
                        help="the name of the mask variable which defines the lake extent")
    parser.add_argument("--lake-id-attribute", default="lake_id",
                        help="the dataset attribute which records the lake id")
    parser.add_argument("--test-id-attribute", default="test_id",
                        help="the dataset attribute which records the test id")
    parser.add_argument("--test-info-attribute", default="lake_dashboard_prepare_data",
                        help="the dataset attribute which records experimental settings for the test")

    parser.add_argument("--fine-res-path", required=False, default=None,
                        help="Path to the additional dataset for plot p5")
    parser.add_argument("--ice-data-path", required=False, default=None,
                        help="Path to the additional dataset for plot p6")

    # Parse the command-line arguments.
    args = parser.parse_args()
    
    # Set Flask host and port in the configuration.
    Config.flask_port = args.port
    Config.flask_host = args.host

    # locate the data files, expanding any wildcard characters
    Config.data = []
    for data_path in args.data:
        Config.data += glob.glob(data_path,recursive=True)

    # Set dataset and variable configurations.
    Config.eof_file = args.eof_file
    Config.temperature_variable = args.temperature_variable
    Config.filled_temperature_variable = args.filled_temperature_variable
    Config.quality_level_variable = args.quality_level_variable
    Config.lake_mask_variable = args.lake_mask_variable

    Config.lake_id_attribute = args.lake_id_attribute
    Config.test_id_attribute = args.test_id_attribute
    Config.test_info_attribute = args.test_info_attribute

    Config.fine_res_path = args.fine_res_path
    Config.ice_data_path = args.ice_data_path
    Config.eofs_path = args.eof_file

    # Start the Bokeh worker thread, passing a function that creates a new Dashboard instance.
    Thread(target=bk_worker, args=[lambda: Dashboard()]).start()
    # Optionally, the worker can be started without EOF file:
    # Thread(target=bk_worker, args=[lambda: Dashboard()]).start()

    print(f"Running lake dashboard at URL:  http://{args.host}:{args.port}")
    # Run the Flask app using the provided host and port.
    app.run(host=args.host, port=Config.flask_port)

# ------------------------------------------------------------
# If this script is run directly, call the main() function.
if __name__ == '__main__':
    main()
