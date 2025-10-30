class Config(object):
    """
    Provides one single place to configure what is configurable.
    This class holds various settings used throughout the application,
    such as host/port information, variable names, and color mapping configurations.
    """

    # Flask server configuration: host and port where the Flask app will run.
    flask_host = "localhost"
    flask_port = 8080

    # External address for the Bokeh server to be reached from clients.
    external_address = "localhost"

    # Variable names for data processing:
    # 'temperature_variable' is the name of the variable for observed lake surface water temperature.
    temperature_variable = ""
    # 'filled_temperature_variable' is the name of the variable containing reconstructed temperatures.
    filled_temperature_variable = ""
    # 'quality_level_variable' is the name of the variable holding quality flags.
    quality_level_variable = ""
    # 'lake_mask_variable' defines the lake extent in the dataset.
    lake_mask_variable = ""

    # Attribute names in the dataset for identifying the lake and test:
    # 'lake_id_attribute' holds the lake identifier.
    lake_id_attribute = ""
    # 'test_id_attribute' holds the test or experiment identifier.
    test_id_attribute = ""
    # 'test_info_attribute' may contain additional information about the test.
    test_info_attribute = ""

    # List of dataset paths to be processed.
    data = []  # list of data_path
    
    # New attributes for the additional datasets
    fine_res_path = ""
    ice_data_path = ""
    eofs_path = ""
    
    # Root folder for the application (default is current directory).
    root_folder = "."

    # Bokeh server configuration: port where the Bokeh app will run.
    bokeh_port = 5006

    # Color mapping configurations used in the dashboard:
    # QA color map to visualize quality flag values.
    qa_color_map = "Viridis256"
    # Color map for displaying temperature values.
    temperature_color_map = "Plasma256"
    # Color for low temperature values.
    temperature_low_color = "black"
    # Color for high temperature values.
    temperature_high_color = "red"
    # Color for missing (NaN) temperature values (using an RGBA tuple).
    temperature_nan_color = (255, 255, 255, 0)


