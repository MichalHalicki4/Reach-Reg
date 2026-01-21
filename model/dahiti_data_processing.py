import pandas as pd
import pickle
import os
from dahitiapi.DAHITI import DAHITI
from model.Station_class import VirtualStation
import geopandas as gpd
from shapely.geometry import Point


def prepare_vs_stations_for_river(cfg, riv_obj, t1, t2, res_dir, loaded_gauges={}):
    """
    Prepares a list of Virtual Stations (VS) for a specific river by downloading data
    from the DAHITI platform.
    """
    # Use cfg.river_name as the standard naming convention for files
    river_name = cfg.river_name

    # Corrected filepath using river_name from config
    filepath = os.path.join(res_dir,
                            f'vs_at_{river_name}_dahiti.pkl' if loaded_gauges else f'vs_at_{river_name}_no_gdata.pkl')

    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the DAHITI downloading.")
        with open(filepath, "rb") as f:
            vs_stations = pickle.load(f)
            return sorted(vs_stations, key=lambda x: x.chainage)

    # Initialize DAHITI and handle gauges
    if loaded_gauges is None:
        loaded_gauges = {}

    dahiti = DAHITI()
    all_targets = dahiti.list_targets(args={})

    # Filter by SWORD reach IDs found in our river geometry
    all_reach_ids = riv_obj.gdf['reach_id'].unique()
    data = [x for x in all_targets if x.get('SWORD_reach_id') in all_reach_ids]

    vs_objects = []
    # Get velocity from config if available, otherwise default for juxtaposition
    vel = getattr(cfg, 'velocity', 1)

    for vs_set in data:
        vs_id, vs_x, vs_y = vs_set['dahiti_id'], vs_set['longitude'], vs_set['latitude']
        vs = VirtualStation(vs_id, vs_x, vs_y)
        vs.get_sword_reach(riv_obj.gdf)

        # Spatial filter: 5km buffer from river center line
        if vs.is_away_from_river(riv_obj, 5000):
            continue

        vs.upload_chainage(riv_obj.get_chainage_of_point(vs.x, vs.y))

        if len(loaded_gauges.keys()) > 0:
            vs.find_closest_gauge_and_chain(loaded_gauges)

        vs.get_water_levels(dahiti)
        vs.river = river_name

        if not isinstance(vs.wl, pd.DataFrame) or len(vs.wl) == 0:
            continue

        vs.time_filter(t1, t2)
        if len(vs.wl) == 0:
            continue

        # SWOT correction if applicable
        if 'mission' in vs.wl.columns:
            vs.wl.loc[vs.wl['mission'] == 'swot', 'wse_u'] += 0.2

        # Juxtaposition logic (Linking VS with Gauge data)
        if len(loaded_gauges.keys()) == 0:
            vs.get_juxtaposed_vs_and_gauge_meas(None, None, None)
            vs_objects.append(vs)
            continue

        # Match VS with neighboring gauges
        if vs.neigh_g_up and vs.neigh_g_dn:
            vs.get_juxtaposed_vs_and_gauge_meas(
                loaded_gauges[vs.neigh_g_up].wl_df,
                loaded_gauges[vs.neigh_g_dn].wl_df,
                loaded_gauges[vs.neigh_g_dn].sampling,
                vel
            )
        elif vs.neigh_g_up:
            vs.get_juxtaposed_vs_and_gauge_meas(
                loaded_gauges[vs.neigh_g_up].wl_df,
                None,
                loaded_gauges[vs.neigh_g_up].sampling,
                vel
            )
        elif vs.neigh_g_dn:
            vs.get_juxtaposed_vs_and_gauge_meas(
                None,
                loaded_gauges[vs.neigh_g_dn].wl_df,
                loaded_gauges[vs.neigh_g_dn].sampling,
                vel
            )
        vs_objects.append(vs)

    # Final saving and sorting
    with open(filepath, "wb") as f:
        pickle.dump(vs_objects, f)

    return sorted(vs_objects, key=lambda x: x.chainage)