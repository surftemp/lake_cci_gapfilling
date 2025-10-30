"""
Mixed bag of common utilities to help interfacing with Bokeh.
This module provides logging configuration, session management, standardized plot sizing,
and functions to create Bokeh Div elements and launch a Bokeh server worker.
"""

import sys
import logging
import json

from logging import Logger, INFO, StreamHandler, Formatter, basicConfig
from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.models import Div

from bokeh.models.tools import CustomJSHover
from bokeh.models import CustomJS


from .config import Config


class SessionManager(object):
    """
    Manage multiple sessions within a single process.
    
    This class keeps a global mapping of Bokeh session IDs to Dashboard objects.
    It provides static methods to add, remove, clear, and retrieve sessions.
    """
    sessions = {}  # mapping from Bokeh session id to a Dashboard object

    @staticmethod
    def clear_sessions():
        # Close each dashboard session and reset the sessions mapping.
        for sid in SessionManager.sessions:
            SessionManager.sessions[sid].close()
        SessionManager.sessions = {}

    @staticmethod
    def remove_session(sid):
        # If the session exists, close it and remove it from the mapping.
        if sid in SessionManager.sessions:
            SessionManager.sessions[sid].close()
            del SessionManager.sessions[sid]

    @staticmethod
    def add_session(sid, dashboard):
        # Add a new session with the given session id and dashboard object.
        SessionManager.sessions[sid] = dashboard

    @staticmethod
    def get_session(sid):
        # Retrieve the dashboard associated with a given session id.
        if sid in SessionManager.sessions:
            return SessionManager.sessions[sid]
        else:
            return None


def create_logger(forName):
    """
    Create and return a logger given a descriptive name.
    
    The logger is configured to output INFO-level messages to stdout,
    using a standardized message format.
    
    :param forName: Descriptive name for this logger.
    :return: A configured Logger instance.
    """
    logger = Logger(forName)
    logger.setLevel(INFO)
    sh = StreamHandler(sys.stdout)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# Create a logger for this module using the create_logger function.
logger = create_logger(__name__)


def configure_bokeh_logging():
    """
    Turn off detailed logging for Bokeh and Tornado.
    
    This function adjusts the logging levels for Bokeh's internal logging
    and for Tornado's access logs to reduce verbosity.
    """
    from bokeh.util.logconfig import basicConfig as bokehBasicConfig
    bokehBasicConfig(level=logging.INFO)

    # Set logging level for 'werkzeug' and 'tornado.access' to WARNING.
    for logger_name in ["werkzeug", "tornado.access"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)


def get_plot_dimensions(plot_size):
    """
    Get standardized plot sizes based on a descriptive plot size.
    
    :param plot_size: A string indicating the desired plot size ("large", "medium", or "small").
    :return: A tuple (width_pixels, height_pixels) in pixels.
    """
    # Default dimensions for "large" plots.
    width_pixels = 512
    height_pixels = 512
    # Adjust dimensions for "medium" plots.
    if plot_size == "medium":
        width_pixels = 384
        height_pixels = 384
    # Adjust dimensions for "small" plots.
    if plot_size == "small":
        width_pixels = 256
        height_pixels = 256
    return (width_pixels, height_pixels)


def create_bokeh_div(txt, return_html_only=False, sz=32):
    """
    Helper function for creating a Bokeh Div containing some text.
    
    This function creates an HTML snippet styled with a specified font size,
    and either returns the raw HTML or wraps it in a Bokeh Div object.
    
    :param txt: The text to display.
    :param return_html_only: If True, return the HTML string; otherwise, return a Bokeh Div.
    :param sz: The font size in pixels.
    :return: A string of HTML or a Bokeh Div object.
    """
    html = "<div style=\"font-size: %dpx;\">%s</div>" % (sz, txt)
    if return_html_only:
        return html
    return Div(text=html, width=800)


def bk_worker(dashboard_factory):
    """
    A Bokeh worker function that starts a Bokeh server.
    
    The worker sets up a Bokeh server, and for each new session, it creates a dashboard 
    by calling the provided dashboard_factory. It also registers session teardown and 
    periodic update callbacks.
    
    :param dashboard_factory: A callable that returns a new Dashboard instance.
    """

    def teardown(sid):
        # Log session destruction and remove the session from SessionManager.
        logger.info("session %s destroyed" % (sid))
        SessionManager.remove_session(sid)

    def launch(d):
        """
        Launch a new Bokeh server session when a new user connects.
        
        :param d: A Bokeh document to be populated.
        """
        # Create a new dashboard instance using the factory.
        dashboard = dashboard_factory()
        sid = d.session_context.id
        d.title = sid
        logger.info("session %s created" % (sid))
        # Register a callback to clean up when the session is destroyed.
        d.on_session_destroyed(lambda sc: teardown(sc.id))
        # Set a periodic callback (every 1000ms) to update the dashboard.
        d.add_periodic_callback(lambda: dashboard.update(), 1000)
        # Initialize the dashboard with configuration and dataset paths.
        dashboard.init(temperature_variable=Config.temperature_variable,
                       filled_temperature_variable=Config.filled_temperature_variable,
                       quality_level_variable=Config.quality_level_variable,
                       lake_mask_variable=Config.lake_mask_variable, data=Config.data)
        # Setup the dashboard on the current Bokeh document.
        dashboard.setup(d)
        # Add the dashboard instance to the session manager.
        SessionManager.add_session(sid, dashboard)

    # Create a Bokeh server with the '/bkapp' endpoint, using the launch function.
    server = Server({'/bkapp': launch},
                    host=Config.flask_host,
                    port=Config.bokeh_port,
                    io_loop=IOLoop(),
                    allow_websocket_origin=["%s:%d" % (Config.external_address, Config.flask_port)])
    server.start()
    # Start the server's IOLoop so it begins processing connections.
    server.io_loop.start()

# Prepare a custom JS hover formatter using the local lat/lon arrays.
def create_hover_js(lats, lons):
    return CustomJSHover(code="""
        var lats = compute_latlon(special_vars.y, %s);
        var lons = compute_latlon(special_vars.x, %s);
        return "(" + lons + ":" + lats + ")[" + Math.floor(special_vars.x) + "," + Math.floor(special_vars.y) + "]";
    """ % (json.dumps(lats), json.dumps(lons)))

def create_timeseries_picker_js(lats, lons):
    return CustomJS(args=dict(), code="""
        fetch_timeseries(cb_obj.x, cb_obj.y, %s, %s, chosenEofVar);
    """ % (json.dumps(lons), json.dumps(lats)))