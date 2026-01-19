import os
import sys
import json
import pandas as pd

from model.config import ReachRegConfig
from model.river_utils import prepare_river_object
from model.gauge_data_processing import download_in_situ_data
from model.dahiti_data_processing import prepare_vs_stations_for_river
from model.densification_processing import densify_wl_with_gdata, densify_wl_no_gdata


def load_config(json_path):
    """
    Loads configuration from a JSON file and initializes the ReachRegConfig object.
    """
    if not os.path.exists(json_path):
        print(f"Error: Configuration file '{json_path}' not found.")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validation for mandatory keys
    required = ['river_name', 'river_metadata', 'model_configs', 'temporal_range']
    for key in required:
        if key not in data:
            print(f"Error: Missing mandatory key '{key}' in JSON config.")
            sys.exit(1)

    # Initialize with the full data dictionary to support Recursive Namespace
    cfg = ReachRegConfig(data)

    temporal_range = data['temporal_range']
    validate = data.get('validate_with_gauge', False)
    target_id = data.get('target_rs_id')

    return cfg, temporal_range, validate, target_id


def densify_wl_at_river_with_gdata(cfg, t_1, t_2, dirs, target_rs_id=None):
    """
    Orchestrates the workflow using in-situ gauge station data for validation.
    """
    # 1. Prepare River Geometry Object
    cur_river = prepare_river_object(cfg, dirs['riv'])

    # 2. Download and filter Gauge Data
    gauges = download_in_situ_data(cfg, cur_river, t_1, dirs['g'])

    # 3. Prepare Virtual Stations (VS)
    # Added 'cfg' as first argument to match the new sterile signature
    vs_stations = prepare_vs_stations_for_river(cfg, cur_river, t_1, t_2, dirs['vs'], gauges)

    # 4. Run Densification for each VS
    rs_ids_to_process = [x.id for x in vs_stations if target_rs_id is None or x.id == target_rs_id]
    if not rs_ids_to_process:
        print(f"No RS found to process for river {cfg.river_name}.")
        return

    for vs_id in rs_ids_to_process:
        densify_wl_with_gdata(vs_id, cfg, cur_river, vs_stations, gauges, dirs['rs'], dirs['ts'])


def densify_wl_at_river_no_gdata(cfg, t_1, t_2, dirs, target_rs_id=None):
    """
    Orchestrates the workflow without in-situ gauge station data.
    """
    # 1. Prepare River Geometry Object
    cur_river = prepare_river_object(cfg, dirs['riv'])

    # 2. Prepare Virtual Stations (VS)
    # Added 'cfg' as first argument to match the new sterile signature
    vs_stations = prepare_vs_stations_for_river(cfg, cur_river, t_1, t_2, dirs['vs'])

    # 3. Run Densification for each VS
    rs_ids_to_process = [x.id for x in vs_stations if target_rs_id is None or x.id == target_rs_id]
    if not rs_ids_to_process:
        print(f"No RS found to process for river {cfg.river_name}.")
        return

    for vs_id in rs_ids_to_process:
        densify_wl_no_gdata(vs_id, cfg, cur_river, vs_stations, dirs['rs'], dirs['ts'])


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py path/to/config.json")
        return

    # Load configuration and settings from JSON
    cfg, temp_range, validate, target_id = load_config(sys.argv[1])
    t1 = pd.to_datetime(temp_range['t1'])
    t2 = pd.to_datetime(temp_range['t2'])

    print(f"--- Reach-Reg Processing: {cfg.river_full_name} ---")

    # Define workspace directories
    base_results = './results/'
    base_data = './data/'

    dirs = {
        'rs': os.path.join(base_results, 'rs_stations/'),
        'ts': os.path.join(base_results, 'timeseries/'),
        'g': os.path.join(base_data, 'g_data/'),
        'vs': os.path.join(base_data, 'vs_data/'),
        'riv': os.path.join(base_data, 'rivers/')
    }

    # Ensure all directories exist
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Execute selected workflow
    if validate:
        densify_wl_at_river_with_gdata(cfg, t1, t2, dirs, target_id)
    else:
        densify_wl_at_river_no_gdata(cfg, t1, t2, dirs, target_id)

    print("--- Processing Complete ---")


if __name__ == "__main__":
    main()