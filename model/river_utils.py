import geopandas as gpd
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from .River_class import River


def find_main_stream_recursively(df, current_id, source_id, path=None):
    """
    Finds the list of IDs of the main stream reaches from the current ID to the source
    using a recursive approach to handle SWORD topology.

    Args:
        df (pd.DataFrame): DataFrame with river reach information (SWORD data).
        current_id (int): ID of the currently checked reach (usually downstream).
        source_id (int): ID of the target source reach (upstream).
        path (list, optional): List of reaches visited so far.

    Returns:
        list: Ordered list of reach IDs forming the main stream, or None if not found.
    """
    if path is None:
        path = [current_id]

    if current_id == source_id:
        return path

    # Find the current reach row
    reach_rows = df[df['reach_id'] == current_id]
    if reach_rows.empty:
        return None

    reach = reach_rows.iloc[0]
    rch_id_up_str = reach['rch_id_up']

    if pd.isna(rch_id_up_str):
        return None

    # SWORD stores multiple upstream IDs as a space-separated string
    next_ids = [int(id_str.strip()) for id_str in str(rch_id_up_str).split()]

    for next_id in next_ids:
        if next_id not in path:
            result = find_main_stream_recursively(df, next_id, source_id, path + [next_id])
            if result is not None and result[-1] == source_id:
                return result

    return None


def prepare_river_object(cfg, dir_riv_data):
    """
    Orchestrates the creation of a River object. It identifies the main stream,
    orders segments, simplifies geometry, and handles tributaries/dams.

    Args:
        cfg (ReachRegConfig): The configuration object containing river metadata.
        dir_riv_data (str): Directory where the river object pickle file is stored/saved.

    Returns:
        River: Initialized and processed instance of the River class.
    """
    filepath = os.path.join(dir_riv_data, f'{cfg.river_name}_object.pkl')

    # Check if object was already processed to save time
    if os.path.exists(filepath):
        print(f"Loading existing river object from: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    print(f"Creating new river object for: {cfg.river_name}")

    # Load SWORD spatial data using path from config
    sword_rivers = gpd.read_file(cfg.sword_river_file)

    # Find main stream reaches using IDs provided in cfg
    river_reaches_li = find_main_stream_recursively(
        sword_rivers,
        cfg.dn_reach,
        cfg.up_reach
    )

    if river_reaches_li is None:
        raise ValueError(f"Could not find a continuous path between upstream {cfg.up_reach} "
                         f"and downstream {cfg.dn_reach} for river {cfg.river_name}.")

    # Filter and sort the GeoDataFrame
    selected_river = sword_rivers[sword_rivers['reach_id'].isin(river_reaches_li)].copy()
    selected_river = selected_river.sort_values('dist_out')
    selected_river['order'] = range(len(selected_river))

    # Initialize River class using config parameters
    current_river = River(selected_river, cfg.metrical_crs, cfg.river_name)
    current_river.get_simplified_geometry()

    # Upload dams and tributaries using list from config
    current_river.upload_dam_and_tributary_chains(cfg.river_tributary_reaches)

    # Save for future use
    with open(filepath, "wb") as f:
        pickle.dump(current_river, f)

    return current_river


def plot_river_profile(river_obj):
    """
    Generates a longitudinal profile plot (WSE vs Chainage).

    Args:
        river_obj (River): Instance of the River class.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(river_obj.gdf['dist_out'] / 1000, river_obj.gdf['wse'], label='Main Stream Profile')
    ax.grid(linestyle='dashed', alpha=0.6)
    ax.set_ylabel('Water Surface Elevation (WSE) [m]')
    ax.set_xlabel('Chainage (Distance from outlet) [km]')
    ax.set_title(f"Longitudinal Profile: {river_obj.name}")
    plt.legend()
    plt.show(block=True)