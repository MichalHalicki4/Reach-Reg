import pandas as pd
import pickle
import geopandas as gpd
import os
import numpy as np
from insituapi.InSitu import InSitu
from model import station_utils as sc
from model.Station_class import GaugeStation


def download_in_situ_data_dahiti(cfg, riv_obj, t1, res_dir):
    """
    Downloads in-situ water level data from DAHITI based on the config object.
    """
    filepath = os.path.join(res_dir, f'gauge_at_{cfg.river_name}.pkl')
    if os.path.exists(filepath):
        print(f"Loading cached DAHITI gauges for {cfg.river_name}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    insitu = InSitu()
    stations_data = sc.get_list_of_stations_from_config(cfg, insitu)
    print(f"DEBUG: Found {len(stations_data)} raw stations in DAHITI collections.")

    insitu_df = pd.DataFrame(stations_data)
    insitu_gdf = gpd.GeoDataFrame(
        insitu_df,
        geometry=gpd.points_from_xy(insitu_df.longitude, insitu_df.latitude),
        crs="EPSG:4326"
    )
    insitu_gdf = insitu_gdf.to_crs(riv_obj.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'longitude': 'X', 'latitude': 'Y'})

    # 1. Spatial filter
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj)
    riv_name_query = cfg.river_name
    try:
        if hasattr(cfg, 'river_metadata'):
            # Jeśli river_metadata to słownik
            if isinstance(cfg.river_metadata, dict):
                riv_name_query = cfg.river_metadata.get('insitu_query_name', cfg.river_name)
            # Jeśli river_metadata to obiekt (przypadek Recursive Namespace)
            else:
                riv_name_query = getattr(cfg.river_metadata, 'insitu_query_name', cfg.river_name)
        elif hasattr(cfg, 'insitu_query_name'):
            riv_name_query = cfg.insitu_query_name
    except Exception:
        riv_name_query = cfg.river_name

    # 3. Apply Name and Metadata filters
    selected_gauges = sc.filter_gauges_by_target_name(gauges_on_river_metadata, insitu, riv_name_query)

    selected_gauges = sc.filter_gauges_by_dt_freq_target(selected_gauges, t1)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)

    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        gauges_on_river_dict[row['id']] = GaugeStation(
            row['X'], row['Y'], row['id'], cfg.river_name, row['chainage'],
            row['unit'], row['data_sampling']
        )
        row_data = insitu.download(int(row['id']))
        row_df = pd.DataFrame(row_data['data']).rename(columns={'value': 'stage'})
        gauges_on_river_dict[row['id']].upload_wl(row_df)

        # Specific outlier handling for the Oder station
        if row['id'] == 19140:
            gauges_on_river_dict[row['id']].wl_df.loc[
                gauges_on_river_dict[row['id']].wl_df['stage'] > 22.5, 'stage'
            ] = np.nan

    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_usgs_insitu_data(cfg, riv_obj, t1, res_dir):
    """
    Processes USGS data from local files. Path must be defined in cfg.river_metadata.
    """
    filepath = os.path.join(res_dir, f'gauge_at_{cfg.river_name}.pkl')
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # Get path from JSON config, fallback to a default data folder
    usgs_dir = cfg.river_metadata.get('local_data_path', './data/external/usgs/')
    metadata_path = os.path.join(usgs_dir, f'{cfg.river_name}_gauge_metadata.csv')
    data_path = os.path.join(usgs_dir, f'{cfg.river_name}_gauge_data.csv')

    df_data = pd.read_csv(data_path, sep=';')
    insitu_df = pd.read_csv(metadata_path)
    insitu_gdf = gpd.GeoDataFrame(
        insitu_df,
        geometry=gpd.points_from_xy(insitu_df.dec_long_va, insitu_df.dec_lat_va),
        crs=4326
    )
    insitu_gdf = insitu_gdf.to_crs(riv_obj.gdf.crs).rename(columns={'dec_long_va': 'X', 'dec_lat_va': 'Y'})

    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj)

    for index, row in gauges_on_river_metadata.iterrows():
        curr_data = df_data.loc[df_data['id'] == row['site_no']].copy()
        if not curr_data.empty:
            curr_data['date'] = pd.to_datetime(curr_data['date']).dt.tz_convert('UTC').dt.tz_localize(None)
            gauges_on_river_metadata.at[index, 'min_date'] = curr_data['date'].min()
            gauges_on_river_metadata.at[index, 'max_date'] = curr_data['date'].max()

    gauges_on_river_metadata = gauges_on_river_metadata.dropna(subset=['min_date', 'max_date'])
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)

    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_data = df_data.loc[df_data['id'] == row['site_no']].copy()
        curr_data['date'] = pd.to_datetime(curr_data['date']).dt.tz_convert('UTC').dt.tz_localize(None)
        sampling = curr_data['date'].diff().mode()[0].resolution_string

        gauges_on_river_dict[row['site_no']] = GaugeStation(
            row['X'], row['Y'], row['site_no'], cfg.river_name,
            row['chainage'], 'm', sampling
        )
        gauges_on_river_dict[row['site_no']].upload_wl(curr_data)

    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_ana_insitu_data(cfg, riv_obj, t1, res_dir):
    """
    Processes ANA (Brazil) data from local CSV files.
    """
    filepath = os.path.join(res_dir, f'gauge_at_{cfg.river_name}.pkl')
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    gauge_id = 13150003  # Specific hardcoded gauge for Solimoes/Amazon
    ana_dir = cfg.river_metadata.get('local_data_path', './data/external/ana/')
    metadata_path = os.path.join(ana_dir, 'RIO SOLIMÕES-AMAZONAS_gauge_metadata.csv')
    data_path = os.path.join(ana_dir, 'RIO SOLIMÕES-AMAZONAS_gauge_data.csv')

    df_data = pd.read_csv(data_path, sep=';')
    insitu_df = pd.read_csv(metadata_path, sep=';')

    curr_metadata = insitu_df.loc[insitu_df['StationCode'] == gauge_id]
    insitu_gdf = gpd.GeoDataFrame(
        curr_metadata,
        geometry=gpd.points_from_xy(curr_metadata['Longitude'], curr_metadata['Latitude']),
        crs=4326
    )
    insitu_gdf = insitu_gdf.to_crs(riv_obj.gdf.crs).rename(columns={'Longitude': 'X', 'Latitude': 'Y'})

    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj, 5000)

    # Filter and process dates
    for index, row in gauges_on_river_metadata.iterrows():
        curr_df = df_data.loc[df_data['id'].astype(int) == row['StationCode']].copy()
        if not curr_df.empty:
            curr_df['date'] = pd.to_datetime(curr_df['date'])
            gauges_on_river_metadata.at[index, 'min_date'] = curr_df['date'].min()
            gauges_on_river_metadata.at[index, 'max_date'] = curr_df['date'].max()

    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)

    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_df = df_data.loc[df_data['id'] == row['StationCode']].copy()
        curr_df.index = pd.to_datetime(curr_df['date'])
        curr_df = curr_df[t1:]
        curr_df['stage'] = curr_df['stage'] / 100

        sampling = curr_df['date'].diff().mode()[0].resolution_string
        gauges_on_river_dict[row['StationCode']] = GaugeStation(
            row['X'], row['Y'], row['StationCode'], cfg.river_name,
            row['chainage'], 'm', sampling
        )
        gauges_on_river_dict[row['StationCode']].upload_wl(curr_df)

    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_ganges_insitu_data(cfg, riv_obj, t1, res_dir):
    """
    Processes Ganges in-situ data from local CSV files.
    """
    filepath = os.path.join(res_dir, f'gauge_at_{cfg.river_name}.pkl')
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    ganges_dir = cfg.river_metadata.get('local_data_path', './data/external/ganges/')
    metadata_path = os.path.join(ganges_dir, f'{cfg.river_name}_gauge_metadata.csv')
    data_path = os.path.join(ganges_dir, f'{cfg.river_name}_gauge_data.csv')

    df_data = pd.read_csv(data_path, sep=';', decimal=',')
    insitu_df = pd.read_csv(metadata_path, sep=';', decimal=',')

    insitu_gdf = gpd.GeoDataFrame(
        insitu_df,
        geometry=gpd.points_from_xy(insitu_df.y, insitu_df.x),
        crs=4326
    ).to_crs(riv_obj.gdf.crs).rename(columns={'y': 'X', 'x': 'Y'})

    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj, 2000)

    for index, row in gauges_on_river_metadata.iterrows():
        curr_data = df_data.loc[df_data['name'] == row['name']].copy()
        if not curr_data.empty:
            curr_data['date'] = pd.to_datetime(curr_data['dt']).dt.tz_localize('Asia/Kolkata').dt.tz_convert(
                'UTC').dt.tz_localize(None)
            gauges_on_river_metadata.at[index, 'min_date'] = curr_data['date'].min()
            gauges_on_river_metadata.at[index, 'max_date'] = curr_data['date'].max()

    gauges_on_river_metadata = gauges_on_river_metadata.dropna(subset=['min_date', 'max_date'])
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)

    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_data = df_data.loc[df_data['name'] == row['name']].copy()
        curr_data['date'] = pd.to_datetime(curr_data['dt']).dt.tz_localize('Asia/Kolkata').dt.tz_convert(
            'UTC').dt.tz_localize(None)
        curr_data = curr_data[['date', 'WSE']].rename(columns={'WSE': 'stage'})

        sampling = curr_data['date'].diff().mode()[0].resolution_string
        gauges_on_river_dict[index] = GaugeStation(
            row['X'], row['Y'], index, cfg.river_name,
            row['chainage'], 'm', sampling
        )
        gauges_on_river_dict[index].upload_wl(curr_data)

    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_in_situ_data(cfg, riv_obj, t_1, dir_g_data):
    """
    Main router function for in-situ data retrieval.
    """
    if cfg.river_name in ['Po', 'Oder', 'Rhine', 'Elbe']:
        return download_in_situ_data_dahiti(cfg, riv_obj, t_1, dir_g_data)
    elif cfg.river_name in ['Mississippi', 'Missouri']:
        return download_usgs_insitu_data(cfg, riv_obj, t_1, dir_g_data)
    elif cfg.river_name == 'Ganges':
        return download_ganges_insitu_data(cfg, riv_obj, t_1, dir_g_data)
    elif cfg.river_name.replace('õ', 'o') in ['Solimoes', 'Amazon']:
        return download_ana_insitu_data(cfg, riv_obj, t_1, dir_g_data)
    return None
