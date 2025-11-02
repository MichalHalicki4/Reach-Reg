import pandas as pd
from model.river_utils import prepare_river_object
from model.gauge_data_processing import download_in_situ_data
from model.dahiti_data_processing import prepare_vs_stations_for_river
from model.densification_processing import densify_wl_with_gdata, densify_wl_no_gdata
from model.data_mapping import all_river_data


results_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python_clean/results/'
data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python_clean/data/'
rs_dir = results_dir + 'rs_stations/'
ts_dir = results_dir + 'timeseries/'
g_dir = data_dir + 'g_data/'
vs_dir = data_dir + 'vs_data/'
riv_dir = data_dir + 'rivers/'


def densify_wl_at_river_with_gdata(riv, t_1, t_2, dir_rs, dir_g, dir_vs, dir_riv, dir_ts):
    """
    Orchestrates the entire densification workflow for a specified river, including the
    integration and use of in-situ gauge station data for validation.

    The process covers river object preparation, in-situ data download, filtering and
    preparation of Virtual Stations (VS), and then running the densification algorithm
    for each VS.

    :param riv: River key identifier (e.g., 'amazon').
    :param t_1: Start date (time_1) for data retrieval.
    :param t_2: End date (time_2) for data retrieval.
    :param dir_rs: Directory path for saving regression results.
    :param dir_g: Directory path for saving gauge data.
    :param dir_vs: Directory path for saving VS data.
    :param dir_riv: Directory path for river geometry data.
    :param dir_ts: Directory path for saving the final densified time series.
    :returns: None (The main output is saved files in the specified directories).
    """

    # 1. Initialization and metadata retrieval
    riv_nm, riv_nms = all_river_data[riv]['river'], all_river_data[riv]['river_names']
    basin_nm, country = all_river_data[riv]['basin'], all_river_data[riv]['country'],
    riv_path = all_river_data[riv]['sword_river_file']

    # 2. Prepare River Geometry Object (e.g., SWORD file processing)
    cur_river = prepare_river_object(riv_path, riv, dir_riv)

    # 3. Download and filter Gauge Data (In-situ data)
    gauges = download_in_situ_data(cur_river, riv_nm, country, t_1, dir_g)

    # 4. Prepare Virtual Stations (VS) - Downloading SWOT WSE data and filtering adjacent VS
    vs_stations = prepare_vs_stations_for_river(cur_river, riv_nm, riv_nms, basin_nm, t_1, t_2, dir_vs, gauges)

    # 5. Run Densification for each VS
    for vs_id in [x.id for x in vs_stations]:
        # densify_wl_with_gdata is a function that creates a DensificationStation object,
        # executes all its methods (regression calculation, pathfinding, smoothing),
        # and saves the results.
        densify_wl_with_gdata(vs_id, riv, cur_river, vs_stations, gauges, dir_rs, dir_ts)


def densify_wl_at_river_no_gdata(riv, t_1, t_2, dir_rs, dir_vs, dir_riv, dir_ts):
    """
    Orchestrates the entire densification workflow for a specified river in a scenario
    where no in-situ gauge station data is available or used.

    The process omits the gauge data download, and therefore it does not perform the accuracy analysis.

    :param riv: River key identifier (e.g., 'amazon').
    :param t_1: Start date (time_1) for data retrieval.
    :param t_2: End date (time_2) for data retrieval.
    :param dir_rs: Directory path for saving regression results.
    :param dir_vs: Directory path for saving VS data.
    :param dir_riv: Directory path for river geometry data.
    :param dir_ts: Directory path for saving the final densified time series.
    :returns: None (The main output is saved files in the specified directories).
    """

    # 1. Initialization and metadata retrieval
    riv_nm, riv_nms = all_river_data[riv]['river'], all_river_data[riv]['river_names']
    basin_nm, country = all_river_data[riv]['basin'], all_river_data[riv]['country'],
    riv_path = all_river_data[riv]['sword_river_file']

    # 2. Prepare River Geometry Object
    cur_river = prepare_river_object(riv_path, riv, dir_riv)

    # 3. Prepare Virtual Stations (VS) - Downloading SWOT WSE data and filtering adjacent VS.
    vs_stations = prepare_vs_stations_for_river(cur_river, riv_nm, riv_nms, basin_nm, t_1, t_2, dir_vs)

    # 4. Run Densification for each VS (using the no-gdata function)
    for vs_id in [x.id for x in vs_stations]:
        # densify_wl_no_gdata is a function that creates a DensificationStation object
        # and runs the process without gauge-based validation.
        densify_wl_no_gdata(vs_id, riv, cur_river, vs_stations, dir_rs, dir_ts)

river = 'Elbe'
t1, t2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2025-10-01 00:00')
densify_wl_at_river_with_gdata(river, t1, t2, rs_dir, g_dir, vs_dir, riv_dir, ts_dir)
# densify_wl_at_river_no_gdata(river, t1, t2, rs_dir, vs_dir, riv_dir)
