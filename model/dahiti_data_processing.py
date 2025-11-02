import pandas as pd
import pickle
import os
from dahitiapi.DAHITI import DAHITI
from model.Station_class import VirtualStation


def prepare_vs_stations_for_river(riv_obj, riv_nm, riv_nms, basin_nm, t1, t2, res_dir, loaded_gauges=None, vel=None):
    """
        Prepares a list of Virtual Stations (VS) for a specific river by downloading data
        from the DAHITI platform, processing their geometry, filtering time series, and
        linking them to nearby Gauge Stations (GS) if available.

        The function first checks for an existing processed pickle file. If found, it loads
        and returns the saved VS objects, sorted by chainage. Otherwise, it connects to DAHITI,
        filters the VS based on the basin and river names, checks if they are close to the
        river line, calculates their chainage, downloads water level data, and filters the
        time series within the specified period (t1 to t2). If Gauge Stations are provided,
        it identifies the closest upstream and downstream GS and prepares juxtaposed water
        level measurements for validation or comparison.

        :param riv_obj: River object containing geometrical data, used for chainage calculation.
        :param riv_nm: Main river name (e.g., 'Po, Italy'), used for file naming.
        :param riv_nms: A list of specific river/target names (from DAHITI) to filter the VS.
        :param basin_nm: Name of the hydrological basin for initial DAHITI filtering.
        :param t1: Start date/time for filtering the VS time series.
        :param t2: End date/time for filtering the VS time series.
        :param res_dir: Directory path where the resulting pickle file will be saved/loaded from.
        :param loaded_gauges: A dictionary of loaded GaugeStation objects (ID: object) for linking VS to GS. Defaults to None.
        :param vel: Velocity parameter used in the calculation of juxtaposed measurements, particularly in time-shifting. Defaults to None.
        :returns: A sorted list of processed VirtualStation objects.
        """
    riv = riv_nm.split(",")[0]
    filepath = f'{res_dir}vs_at_{riv}.pkl' if loaded_gauges != {} else f'{res_dir}vs_at_{riv}_no_gdata.pkl'
    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the DAHITI downloading.")
        with open(filepath, "rb") as f:
            vs_stations = pickle.load(f)
            return sorted(vs_stations, key=lambda x: x.chainage)

    if loaded_gauges is None:
        loaded_gauges = {}
    dahiti = DAHITI()
    data = dahiti.list_targets(args={'basin': basin_nm})
    vs_stations = [(vs['dahiti_id'], vs['longitude'], vs['latitude']) for vs in data if vs['target_name'] in riv_nms]
    vs_objects = []
    for vs_set in vs_stations:
        vs_id, vs_x, vs_y = vs_set[0], vs_set[1], vs_set[2]
        vs = VirtualStation(vs_id, vs_x, vs_y)
        if vs.is_away_from_river(riv_obj, 5000):
            continue
        vs.upload_chainage(riv_obj.get_chainage_of_point(vs.x, vs.y))
        if len(loaded_gauges.keys()) > 0:
            vs.find_closest_gauge_and_chain(loaded_gauges)
        vs.get_water_levels(dahiti)
        if type(vs.wl) != pd.DataFrame:
            continue
        vs.time_filter(t1, t2)
        if len(vs.wl) == 0:
            continue
        if len(loaded_gauges.keys()) == 0:
            vs.get_juxtaposed_vs_and_gauge_meas(None, None, None)
            vs_objects.append(vs)
            continue
        if vs.neigh_g_up and vs.neigh_g_dn:
            vs.get_juxtaposed_vs_and_gauge_meas(loaded_gauges[vs.neigh_g_up].wl_df, loaded_gauges[vs.neigh_g_dn].wl_df,
                                                loaded_gauges[vs.neigh_g_dn].sampling, vel)
        elif vs.neigh_g_up:
            vs.get_juxtaposed_vs_and_gauge_meas(loaded_gauges[vs.neigh_g_up].wl_df, None,
                                                loaded_gauges[vs.neigh_g_up].sampling, vel)
        elif vs.neigh_g_dn:
            vs.get_juxtaposed_vs_and_gauge_meas(None, loaded_gauges[vs.neigh_g_dn].wl_df,
                                                loaded_gauges[vs.neigh_g_dn].sampling, vel)
        vs_objects.append(vs)
    with open(filepath, "wb") as f:
        pickle.dump(vs_objects, f)
    return sorted(vs_objects, key=lambda x: x.chainage)
