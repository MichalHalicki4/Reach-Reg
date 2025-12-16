# ==============================================================================
#                      REACH-REG METHOD: CONFIGURATION GUIDE
# ==============================================================================
# This is the central configuration file for running the Reach-Reg densification
# method. Follow the steps below to configure and run the analysis.

# STEP 1: Dependencies and Authorization
# -------------------------------------
# A. Install all required Python packages (see requirements.txt).
# B. If utilizing data from DAHITI, ensure you have set up your credentials
#    and authorized access in the relevant module (e.g., model/dahiti_data_processing.py).
#    Data providers like RiverSP (HydroCon) typically do not require external authorization.

# STEP 2: River and Configuration Dictionaries
# --------------------------------------------
# A. Define the river to be processed by adding its dictionary entry to the
#    'all_river_data' dictionary in model/data_mapping.py.
# B. Review the 'configs' dictionary in model/data_mapping.py:
#    configs = {
#        'amp_thres': 1,            # WSE amplitude threshold [m]
#        'rmse_thres': 10,          # RMSE threshold for filtering vs_reaches [m]
#        'single_rmse_thres': 0.2,  # Single RMSE threshold for a reach pair [m]
#        'itpd_method': 'akima',    # Interpolation method (e.g., 'akima', 'linear')
#        'buffer': 300,             # River buffer size [km]
#        'corr_thres': 0.75,        # Correlation threshold for reach pairs
#        'bottom': 0.1              # River bed level offset [m]
#    }
#    These are the recommended settings tested on 8 rivers. To experiment
#    (e.g., changing the interpolation method to 'linear' when using HydroCon data),
#    adjust values here.

# STEP 3: Define Input/Output Directories
# ---------------------------------------
# Adjust the directory paths below to point to your local storage locations.
# All intermediate and final results will be saved here.

# STEP 4: In-situ Data Customization
# -------------------------------------------------------------
# If you plan to run the method with in-situ data (densify_wl_at_river_with_gdata),
# you must ensure your in-situ data is correctly loaded.
# Review the functions in model/gauge_data_processing.py and either adapt your
# data format to match existing functions or create a new function for your data source.

# STEP 5: Define Temporal Range
# -----------------------------
# Set the analysis start (t1) and end (t2) dates below.

# STEP 6: Run Densification
# -------------------------
# Uncomment and call the appropriate function in the main loop:
# - densify_wl_at_river_with_gdata: To run the method and perform accuracy validation.
# - densify_wl_at_river_no_gdata: To run the method without validation.
# ==============================================================================


import pandas as pd
from model.river_utils import prepare_river_object
from model.gauge_data_processing import download_in_situ_data
from model.dahiti_data_processing import prepare_vs_stations_for_river
# from model.hydrochron_data_processing import prepare_vs_stations_for_river

from model.densification_processing import densify_wl_with_gdata, densify_wl_no_gdata
from model.data_mapping import all_river_data


results_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python_clean/results/'
data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python_clean/data/'
rs_dir = results_dir + 'rs_stations/'
ts_dir = results_dir + 'timeseries/'
g_dir = data_dir + 'g_data/'
vs_dir = data_dir + 'vs_data/'
riv_dir = data_dir + 'rivers/'


def densify_wl_at_river_with_gdata(riv, t_1, t_2, dir_rs, dir_g, dir_vs, dir_riv, dir_ts, target_rs_id=None):
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
    :param target_rs_id: Optional. If specified, only this single RS (by its ID)
                         will be processed instead of all available RSs on the river.
    :returns: None (The main output is saved files in the specified directories).
    """

    # 1. Initialization and metadata retrieval
    country = all_river_data[riv]['country'],
    riv_path = all_river_data[riv]['sword_river_file']

    # 2. Prepare River Geometry Object (e.g., SWORD file processing)
    cur_river = prepare_river_object(riv_path, riv, dir_riv)

    # 3. Download and filter Gauge Data (In-situ data)
    gauges = download_in_situ_data(cur_river, country, t_1, dir_g)

    # 4. Prepare Virtual Stations (VS) - Downloading SWOT WSE data and filtering adjacent VS
    vs_stations = prepare_vs_stations_for_river(cur_river, t_1, t_2, dir_vs, gauges)

    # 5. Run Densification for each VS
    rs_ids_to_process = [x.id for x in vs_stations if target_rs_id is None or x.id == target_rs_id]
    if not rs_ids_to_process:
        print(f"No RS found to process for river {riv} (or target_rs_id {target_rs_id} not found).")
        return
    for vs_id in rs_ids_to_process:
        densify_wl_with_gdata(vs_id, riv, cur_river, vs_stations, gauges, dir_rs, dir_ts)


def densify_wl_at_river_no_gdata(riv, t_1, t_2, dir_rs, dir_vs, dir_riv, dir_ts, target_rs_id=None):
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
    :param target_rs_id: Optional. If specified, only this single RS (by its ID)
                         will be processed instead of all available RSs on the river.
    :returns: None (The main output is saved files in the specified directories).
    """

    # 1. Initialization and metadata retrieval
    country = all_river_data[riv]['country']
    riv_path = all_river_data[riv]['sword_river_file']

    # 2. Prepare River Geometry Object
    cur_river = prepare_river_object(riv_path, riv, dir_riv)

    # 3. Prepare Virtual Stations (VS) - Downloading SWOT WSE data and filtering adjacent VS.
    vs_stations = prepare_vs_stations_for_river(cur_river, t_1, t_2, dir_vs)

    # 4. Run Densification for each VS (using the no-gdata function)
    rs_ids_to_process = [x.id for x in vs_stations if target_rs_id is None or x.id == target_rs_id]
    if not rs_ids_to_process:
        print(f"No RS found to process for river {riv} (or target_rs_id {target_rs_id} not found).")
        return

    for vs_id in rs_ids_to_process:
        densify_wl_no_gdata(vs_id, riv, cur_river, vs_stations, dir_rs, dir_ts)


t1, t2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2025-10-01 00:00')
# densify_wl_at_river_no_gdata(river, t1, t2, rs_dir, vs_dir, riv_dir, ts_dir)
# densify_wl_at_river_with_gdata(river, t1, t2, rs_dir, g_dir, vs_dir, riv_dir, ts_dir)
