import geopandas as gpd
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from .data_mapping import all_river_data
from .River_class import River


def find_main_stream_recursively(df, current_id, source_id, path=None):
    """
    Finds the list of ID's of the main stream reaches from the current ID to the source
    (recursive version handling multiple IDs in 'rch_id_up').

    Args:
        df (pd.DataFrame): DataFrame with river reach information,
                           containing columns 'reach_id' (reach ID) and
                           'rch_id_up' (string with upstream reach IDs, space-separated).
        current_id (int): ID of the currently checked reach.
        source_id (int): ID of the source reach.
        path (list, optional): List of reaches visited so far. Defaults to None.

    Returns:
        list: List of reach IDs of the main stream from the current ID to the source,
              or None if no path to the source is found.
    """
    if path is None:
        path = [current_id]

    if current_id == source_id:
        return path

    reach = df[df['reach_id'] == current_id].iloc[0]
    rch_id_up_str = reach['rch_id_up']

    if pd.isna(rch_id_up_str):
        return None

    next_ids = [int(id_str.strip()) for id_str in rch_id_up_str.split()]

    for next_id in next_ids:
        if next_id not in path:
            result = find_main_stream_recursively(df, next_id, source_id, path + [next_id])
            if result is not None and result[-1] == source_id:
                return result

    return None


def prepare_river_object(riv_path, riv, dir_riv_data):
    """
    Reads spatial data for a river, identifies the main stream, orders its segments, and initializes a River object.
    It uses predefined information to extract river names and basin details, then calls methods to simplify the river's
        geometry. Later it uploads dams and tributaries and finally saves the object to a pickle file.

    Args:
        dir_riv_data: The directory, where the river object pickle file should be saved.
        riv_path (str): The file path to the GeoJSON or shapefile containing the river segments (SWORD data).
        riv (str): A key used to look up specific river details in the 'dahiti_river_names_and_basins' dictionary.

    Returns:
        River: An instance of the River class with the selected main stream segments and simplified geometry.
        """
    filepath = f'{dir_riv_data}{riv}_object.pkl'
    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the River object creation.")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    sword_rivers = gpd.read_file(riv_path)
    river_reaches_li = find_main_stream_recursively(sword_rivers,
                                                    all_river_data[riv]['dn_reach'], all_river_data[riv]['up_reach'])
    selected_river = sword_rivers[sword_rivers['reach_id'].isin(river_reaches_li)]
    selected_river = selected_river.sort_values('dist_out')
    selected_river['order'] = range(len(selected_river))
    current_river = River(selected_river, all_river_data[riv]['metrical_crs'], riv)
    current_river.get_simplified_geometry()
    current_river.upload_dam_and_tributary_chains(all_river_data[riv]['river_tributary_reaches'])
    with open(filepath, "wb") as f:
        pickle.dump(current_river, f)
    return current_river


def plot_river_profile(riv):
    """
    Generates and displays a longitudinal profile plot of the river, showing the Water Surface Elevation (WSE) against
        the chainage.
    It plots the 'wse' column versus the 'dist_out' column (converted to kilometers) from the River object's GDF.

    Args:
        riv (River): An instance of the River class containing the river segments' GeoDataFrame (self.gdf) with
            'dist_out' and 'wse' columns.
        """
    fig, ax = plt.subplots()
    ax.plot(riv.gdf['dist_out'] / 1000, riv.gdf['wse'])
    ax.grid(linestyle='dashed', alpha=0.6)
    ax.set_ylabel('WSE [m]')
    ax.set_xlabel('Chainage [km]')
    plt.show(block=True)
