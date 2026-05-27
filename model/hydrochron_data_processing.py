import pandas as pd
import pickle
import os
from model.Station_class import VirtualStation

import requests
from tqdm.auto import tqdm
import warnings
import pathlib


def prepare_vs_stations_for_river(cfg, riv_obj, t1, t2, res_dir, loaded_gauges={}):
    """
    Prepares a list of Virtual Stations (VS) for a specific river by downloading data
    from the Hydrochron platform, based on configurations from the cfg object.
    """
    # Use cfg.river_name as the standard naming convention for files
    river_name = cfg.river_name

    # Corrected filepath using river_name from config (consistent with DAHITI)
    filepath = os.path.join(res_dir,
                            f'vs_at_{river_name}_hydrochron.pkl' if loaded_gauges else f'vs_at_{river_name}_no_gdata_hydrochron.pkl')

    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the Hydrochron downloading.")
        with open(filepath, "rb") as f:
            vs_stations = pickle.load(f)
            return sorted(vs_stations, key=lambda x: x.chainage)

    if loaded_gauges is None:
        loaded_gauges = {}

    # Extract optional translator settings from config if available
    v17_translator_dir = getattr(cfg, 'v17_translator_dir', None)
    RiverSP_version = getattr(cfg, 'RiverSP_version', None)

    feature = "Reach"
    start_time = "2023-01-01T00:00:00Z"
    end_time = "2030-01-01T00:00:00Z"
    output = "geojson"
    fields = "wse_u,sword_version,reach_id,time_str,wse,width,dark_frac,reach_q,reach_q_b,xovr_cal_q,ice_clim_f,area_wse,obs_frac_n,p_lat,p_lon"
    vs_data_sets = []

    v17_lookup = None
    all_reach_ids = riv_obj.gdf.reach_id.unique()

    for i, v17_feature_id in enumerate(all_reach_ids):
        dfs = []
        for _version in ['C', 'D']:
            if RiverSP_version is not None and _version != RiverSP_version:
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
                collection_name = ""
            else:
                _feature_id = str(v17_feature_id)
                collection_name = "SWOT_L2_HR_RiverSP_D"

            print(
                f"{i} of {len(all_reach_ids)}: Getting RiverSP version: {_version} for reach_id: {v17_feature_id} ({_feature_id})")
            url = f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?&feature={feature}&feature_id={_feature_id}&start_time={start_time}&end_time={end_time}&output={output}&fields={fields}"
            if _version == 'D':
                url += f"&collection_name={collection_name}"

            try:
                hydrocron_response = requests.get(url).json()
            except Exception as e:
                print(f"Request failed for reach_id {_feature_id}: {e}")
                continue

            if 'error' in hydrocron_response.keys():
                print(f"Hydrochron request failed for reach_id {_feature_id}: {hydrocron_response['error']}")
                continue
            try:
                df = pd.DataFrame([x['properties'] for x in hydrocron_response['results']['geojson']['features']])
                if df.empty:
                    continue
            except KeyError:
                if 'error' in hydrocron_response['message']:
                    print(f'Error for {_feature_id}')
                    continue
                else:
                    print(f'Error: {hydrocron_response}')
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
            df['river'] = river_name

        df.drop_duplicates(subset=['datetime'], inplace=True, keep='first')
        df = df.set_index('datetime', drop=False)
        df.sort_index(inplace=True)
        vs_data_sets.append([v17_feature_id, df['p_lon'].iloc[0], df['p_lat'].iloc[0], df])

    vs_objects = []
    vel = getattr(cfg, 'velocity', 1)

    for vs_set in vs_data_sets:
        vs_id, vs_x, vs_y = vs_set[0], vs_set[1], vs_set[2]
        vs = VirtualStation(vs_id, vs_x, vs_y)

        # Spatial filter: 5km buffer from river center line
        if vs.is_away_from_river(riv_obj, 5000):
            continue

        vs.upload_chainage(riv_obj.get_chainage_of_point(vs.x, vs.y))

        if len(loaded_gauges.keys()) > 0:
            vs.find_closest_gauge_and_chain(loaded_gauges)

        vs.wl = vs_set[3]
        vs.swot_wl = vs_set[3]

        if not isinstance(vs.wl, pd.DataFrame) or len(vs.wl) == 0:
            continue

        vs.time_filter(pd.to_datetime(t1), pd.to_datetime(t2))
        if len(vs.wl) == 0:
            continue

        vs.river = river_name

        # Juxtaposition logic (Linking VS with Gauge data)
        if len(loaded_gauges.keys()) == 0:
            vs.get_juxtaposed_vs_and_gauge_meas(None, None, None)
            vs_objects.append(vs)
            continue

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

    with open(filepath, "wb") as f:
        pickle.dump(vs_objects, f)

    return sorted(vs_objects, key=lambda x: x.chainage)