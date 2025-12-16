import pandas as pd
import pickle
import os
from model.Station_class import VirtualStation

import requests
from tqdm.auto import tqdm
import warnings
import pathlib


def prepare_vs_stations_for_river(riv_obj, t1, t2, res_dir, loaded_gauges=None, vel=None,
                                  v17_translator_dir=None, RiverSP_version=None):
    """
        Prepares a list of Virtual Stations (VS) for a specific river by downloading data
        from the Hydrochron platform, processing their geometry, filtering time series, and
        linking them to nearby Gauge Stations (GS) if available.

        The function first checks for an existing processed pickle file. If found, it loads
        and returns the saved VS objects, sorted by chainage. Otherwise, it connects to Hydrochron,
        filters the VS based on the basin and river names, checks if they are close to the
        river line, calculates their chainage, downloads water level data, and filters the
        time series within the specified period (t1 to t2). If Gauge Stations are provided,
        it identifies the closest upstream and downstream GS and prepares juxtaposed water
        level measurements for validation or comparison.

        :param riv_obj: River object containing geometrical data, used for chainage calculation.
        :param t1: Start date/time for filtering the VS time series.
        :param t2: End date/time for filtering the VS time series.
        :param res_dir: Directory path where the resulting pickle file will be saved/loaded from.
        :param loaded_gauges: A dictionary of loaded GaugeStation objects (ID: object) for linking VS to GS. Defaults to None.
        :param vel: Velocity parameter used in the calculation of juxtaposed measurements, particularly in time-shifting. Defaults to None.
        :returns: A sorted list of processed VirtualStation objects.
        """
    filepath = f'{res_dir}vs_at_{riv_obj.name}_hydrochron.pkl' if loaded_gauges != {} else f'{res_dir}vs_at_{riv_obj.name}_no_gdata_hydrochron.pkl'
    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the Hydrochron downloading.")
        with open(filepath, "rb") as f:
            vs_stations_list = pickle.load(f)
            return sorted(vs_stations_list, key=lambda x: x.chainage)

    if loaded_gauges is None:
        loaded_gauges = {}
    # dahiti = DAHITI()
    # data = dahiti.list_targets(args={'basin': basin_nm})
    # vs_stations_list = [(vs['dahiti_id'], vs['longitude'], vs['latitude']) for vs in data if vs['target_name'] in riv_nms]

    feature = "Reach"
    start_time = "2023-01-01T00:00:00Z"
    end_time = "2030-01-01T00:00:00Z"
    output = "geojson"
    # fields="sword_version,reach_id,time_str,wse,width,dark_frac,node_q,node_q_b,xovr_cal_q,ice_clim_f,area_wse"
    fields = "wse_u,sword_version,reach_id,time_str,wse,width,dark_frac,reach_q,reach_q_b,xovr_cal_q,ice_clim_f,area_wse,obs_frac_n,p_lat,p_lon"
    vs_data_sets = []

    v17_lookup = None
    for i, v17_feature_id in enumerate(riv_obj.gdf.reach_id.unique()):
        dfs = []
        for _version in ['C', 'D']:
            if RiverSP_version is not None:
                if _version != RiverSP_version:
                    continue
            if _version == 'C' and v17_lookup is None:
                if v17_translator_dir is None:
                    v17_translator_dir = pathlib.Path(res_dir).parent / "SWORD_v17b_rename/"
                else:
                    v17_translator_dir = pathlib.Path(v17_translator_dir)
                if not v17_translator_dir.exists():
                    raise FileNotFoundError(f"Directory {v17_translator_dir} does not exist.")
                for fn in v17_translator_dir.iterdir():
                    if fn.is_file() and (fn.suffix == '.csv') and "Reach" in fn.stem:
                        _lookup = pd.read_csv(fn, sep=',', index_col=None)
                        if str(v17_feature_id)[:2] in _lookup.v17_reach_id.astype(str).str[:2].values:
                            v17_lookup = _lookup
                            break
            if _version == 'C':
                if v17_lookup is None:
                    raise ValueError(
                        f"Could not find v17 lookup for reach_id {v17_feature_id} in translators directory.")
                v16_feature_id = v17_lookup.set_index('v17_reach_id').loc[int(v17_feature_id), 'v16_reach_id']
                if v16_feature_id == 0:
                    print(f"Skipping reach_id {v17_feature_id} as it has no v16 equivalent.")
                    continue
                _feature_id = str(v16_feature_id)
                # collection_name = "SWOT_L2_HR_RiverSP_reach_2.0"
                collection_name = ""

            else:
                _feature_id = str(v17_feature_id)
                collection_name = "SWOT_L2_HR_RiverSP_D"

            print(
                f"{i} of {len(riv_obj.gdf.reach_id.unique())}: Getting RiverSP version: {_version} for reach_id: {v17_feature_id} ({_feature_id})")
            url = f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?&feature={feature}&feature_id={_feature_id}&start_time={start_time}&end_time={end_time}&output={output}&fields={fields}"
            if _version == 'D':
                url += f"&collection_name={collection_name}"
            hydrocron_response = requests.get(url).json()

            if 'error' in hydrocron_response.keys():
                print(f"Hydrochron request failed for reach_id {_feature_id}: {hydrocron_response['error']}")
                continue
            df = pd.DataFrame([x['properties'] for x in hydrocron_response['results']['geojson']['features']])
            if df.empty:
                continue
            for field in ['wse', 'wse_u', 'width', 'dark_frac', 'reach_q', 'reach_q_b', 'xovr_cal_q', 'ice_clim_f',
                          'area_wse', 'obs_frac_n']:
                df[field] = pd.to_numeric(df[field])
            df = df.query(
                'time_str != "no_data" & dark_frac < 0.4 & reach_q <= 2 & reach_q_b <= 2097152 & xovr_cal_q <= 1 & area_wse > 0 & obs_frac_n >= 0.5')
            if df.empty:
                continue
            dfs.append(df)
        if len(dfs) == 0:
            continue
        df = pd.concat(dfs, ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['datetime'] = pd.to_datetime(df['time_str'], utc=True).dt.tz_localize(None)
            df['mission'] = 'SWOT'
            df['river'] = riv_obj.name
        df.drop_duplicates(subset=['datetime'], inplace=True, keep='first')
        df = df.set_index('datetime', drop=False)
        df.sort_index(inplace=True)
        vs_data_sets.append(
            [v17_feature_id, df['p_lon'].iloc[0], df['p_lat'].iloc[0], df]
        )

    vs_objects = []
    for vs_set in vs_data_sets:
        vs_id, vs_x, vs_y = vs_set[0], vs_set[1], vs_set[2]
        print(vs_id)
        vs = VirtualStation(vs_id, vs_x, vs_y)
        if vs.is_away_from_river(riv_obj, 5000):
            print('away')
            continue
        vs.upload_chainage(riv_obj.get_chainage_of_point(vs.x, vs.y))
        if len(loaded_gauges.keys()) > 0:
            vs.find_closest_gauge_and_chain(loaded_gauges)
        # vs.get_water_levels(dahiti)
        vs.wl = vs_set[3]
        vs.swot_wl = vs_set[3]
        if type(vs.wl) != pd.DataFrame:
            print('DF error')
            continue
        t1 = pd.to_datetime(t1)
        t2 = pd.to_datetime(t2)
        vs.time_filter(t1, t2)
        vs.river = riv_obj.name
        if len(vs.wl) == 0:
            print('len 0')
            continue
        if len(loaded_gauges.keys()) == 0:
            print('No loaded gauges')
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