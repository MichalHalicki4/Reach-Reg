import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from insituapi.InSitu import InSitu
from dahitiapi.DAHITI import DAHITI
import River_class as rv
import Station_class as sc
import pickle
import numpy as np
import copy
from datetime import timezone

data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/'
results_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/article/results/'
# t1, t2 = pd.to_datetime('2023-07-10 00:00'), pd.to_datetime('2025-01-30 00:00')
t1, t2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2025-04-29 23:59')
# t1, t2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2025-10-01 00:00')

# riv, metric_crs = 'Missouri', 'ESRI:102010'
# riv, metric_crs = 'Solimoes', 'ESRI:102033'
riv, metric_crs = 'Ganges', 'ESRI:102025'

river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp'  # POLAND
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb23_v17b.shp'  # ELBE, RHINE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb22_v17b.shp'  # DANUBE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb21_v17b.shp'  # ITALY
riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/NA/na_sword_reaches_hb74_v17b.shp'  # NA


save_river_to_file = False
if save_river_to_file:
    current_river = rv.prepare_river_object(riv_path, riv, metric_crs)
    current_river.upload_dam_and_tributary_chains(rv.river_tributary_reaches[riv])
    with open(f'{results_dir}{river_name.split(",")[0]}_object.pkl', "wb") as f:
        pickle.dump(current_river, f)
else:
    with open(f'{results_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
        current_river = pickle.load(f)


def download_in_situ_data(riv_object, riv_name, cntry, t1, t2):
    insitu = InSitu()
    stations_data = sc.get_list_of_stations_from_country(cntry, insitu)
    insitu_df = pd.DataFrame(stations_data)
    insitu_gdf = gpd.GeoDataFrame(insitu_df, geometry=gpd.points_from_xy(insitu_df.longitude, insitu_df.latitude),
                                  crs="EPSG:4326")
    insitu_gdf = insitu_gdf.to_crs(riv_object.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'longitude': 'X', 'latitude': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_object)
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, t2)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_object)
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        gauges_on_river_dict[row['id']] = sc.GaugeStation(row['X'], row['Y'], row['id'], riv_name, row['chainage'], row['unit'], row['data_sampling'])
        row_data = insitu.download(int(row['id']))
        row_df = pd.DataFrame(row_data['data']).rename(columns={'value': 'stage'})
        gauges_on_river_dict[row['id']].upload_wl(row_df)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(f'{results_dir}gauge_at_{riv_name.split(",")[0]}.pkl', "wb") as f:
        pickle.dump(gauges_on_river_dict, f)
    print(1)


def download_usgs_insitu_data(riv_object, riv, t1, t2):
    riv = riv.replace(',', '')
    usgs_dir = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_USGS/'
    metadata_path = f'{usgs_dir}{riv}_gauge_metadata.csv'
    data_path = f'{usgs_dir}{riv}_gauge_data.csv'
    # t1_utc, t2_utc = t1.tz_localize('UTC'), t2.tz_localize('UTC')
    df_data = pd.read_csv(data_path, sep=';')
    insitu_df = pd.read_csv(metadata_path)
    insitu_gdf = gpd.GeoDataFrame(insitu_df, geometry=gpd.points_from_xy(insitu_df.dec_long_va, insitu_df.dec_lat_va),
                                  crs=4326)
    insitu_gdf = insitu_gdf.to_crs(riv_object.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'dec_long_va': 'X', 'dec_lat_va': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_object)
    min_dates, max_dates = [], []
    for index, row in gauges_on_river_metadata.iterrows():
        curr_data = df_data.loc[df_data['id'] == row['site_no']]
        if len(curr_data) > 0:
            curr_data['date'] = pd.to_datetime(curr_data['date'])
            # if curr_data['date'].dt.tz != timezone.utc:
            curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
            curr_data['date'] = curr_data['date'].dt.tz_localize(None)

            min_dates.append(curr_data['date'].min())
            max_dates.append(curr_data['date'].max())
        else:
            min_dates.append(None)
            max_dates.append(None)
    gauges_on_river_metadata['min_date'] = min_dates
    gauges_on_river_metadata['max_date'] = max_dates

    gauges_on_river_metadata = gauges_on_river_metadata.dropna(subset=['min_date', 'max_date'])
    gauges_on_river_metadata['type'] = 'water_level'
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, t2, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_object)
    selected_gauges['unit'] = 'm'
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_data = df_data.loc[df_data['id'] == row['site_no']]
        curr_data['date'] = pd.to_datetime(curr_data['date'])
        curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
        curr_data['date'] = curr_data['date'].dt.tz_localize(None)

        data_sampling = pd.to_datetime(curr_data['date']).diff().mode()[0].resolution_string
        gauges_on_river_dict[row['site_no']] = sc.GaugeStation(row['X'], row['Y'], row['site_no'], riv, row['chainage'],
                                                               row['unit'], data_sampling)
        gauges_on_river_dict[row['site_no']].upload_wl(curr_data)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(f'{results_dir}gauge_at_{riv.split(" ")[0]}.pkl', "wb") as f:
        pickle.dump(gauges_on_river_dict, f)


def download_ana_insitu_data(riv_object, riv, t1, t2):
    riv = riv.replace(',', '')
    gauge_id = 13150003
    ana_dir = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_ANA/'
    metadata_path = f'{ana_dir}RIO SOLIMÕES-AMAZONAS_gauge_metadata.csv'
    data_path = f'{ana_dir}RIO SOLIMÕES-AMAZONAS_gauge_data.csv'
    # t1_utc, t2_utc = t1.tz_localize('UTC'), t2.tz_localize('UTC')
    df_data = pd.read_csv(data_path, sep=';')
    insitu_df = pd.read_csv(metadata_path, sep=';')
    curr_metadata = insitu_df.loc[insitu_df['StationCode'] == gauge_id]
    # x, y = curr_metadata['Longitude'], curr_metadata['Latitude']
    insitu_gdf = gpd.GeoDataFrame(curr_metadata, geometry=gpd.points_from_xy(curr_metadata['Longitude'],
                                                                             curr_metadata['Latitude']), crs=4326)
    insitu_gdf = insitu_gdf.to_crs(riv_object.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'Longitude': 'X', 'Latitude': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_object, 5000)
    min_dates, max_dates = [], []
    for index, row in gauges_on_river_metadata.iterrows():
        curr_df = df_data.loc[df_data['id'].astype(int) == row['StationCode']]
        if len(curr_df) > 0:
            curr_df.index = pd.to_datetime(curr_df['date'])
            curr_df = curr_df[t1:t2]
            curr_df['stage'] = curr_df['stage'] / 100
            curr_df[['stage']].to_csv(f'{ana_dir}gauge{gauge_id}.csv', sep=';', decimal=',')

            min_dates.append(curr_df['date'].min())
            max_dates.append(curr_df['date'].max())
        else:
            min_dates.append(None)
            max_dates.append(None)
    gauges_on_river_metadata['min_date'] = min_dates
    gauges_on_river_metadata['max_date'] = max_dates

    gauges_on_river_metadata['type'] = 'water_level'
    curr_t1, curr_t2 = t1+pd.to_timedelta('2 days'), t2-pd.to_timedelta('2 days')
    # selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1+pd.to_timedelta('50 days'), t2-pd.to_timedelta('150 days'), False)
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, curr_t1, curr_t2, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_object)
    selected_gauges['unit'] = 'm'
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_df = df_data.loc[df_data['id'] == row['StationCode']]
        curr_df.index = pd.to_datetime(curr_df['date'])
        curr_df = curr_df[t1:t2]
        curr_df['stage'] = curr_df['stage'] / 100
        curr_df[['stage']].to_csv(f'{ana_dir}gauge{gauge_id}.csv', sep=';', decimal=',')

        data_sampling = pd.to_datetime(curr_df['date']).diff().mode()[0].resolution_string
        gauges_on_river_dict[row['StationCode']] = sc.GaugeStation(row['X'], row['Y'], row['StationCode'], riv,
                                                                   row['chainage'], row['unit'], data_sampling)
        gauges_on_river_dict[row['StationCode']].upload_wl(curr_df)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(f'{results_dir}gauge_at_{riv.split(" ")[0]}.pkl', "wb") as f:
        pickle.dump(gauges_on_river_dict, f)


def download_ganges_insitu_data(riv_object, riv, t1, t2):
    riv = riv.replace(',', '')
    ganges_dir = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_GANGES/'
    metadata_path = f'{ganges_dir}{riv}_gauge_metadata.csv'
    data_path = f'{ganges_dir}{riv}_gauge_data.csv'
    # t1_utc, t2_utc = t1.tz_localize('UTC'), t2.tz_localize('UTC')
    df_data = pd.read_csv(data_path, sep=';', decimal=',')
    insitu_df = pd.read_csv(metadata_path, sep=';', decimal=',')
    insitu_gdf = gpd.GeoDataFrame(insitu_df, geometry=gpd.points_from_xy(insitu_df.y, insitu_df.x),
                                  crs=4326)
    insitu_gdf = insitu_gdf.to_crs(riv_object.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'y': 'X', 'x': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_object, 2000)
    min_dates, max_dates = [], []
    for index, row in gauges_on_river_metadata.iterrows():
        curr_data = df_data.loc[df_data['name'] == row['name']]
        if len(curr_data) > 0:
            curr_data['date'] = pd.to_datetime(curr_data['dt'])
            curr_data['date'] = curr_data['date'].dt.tz_localize('Asia/Kolkata')
            curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
            curr_data['date'] = curr_data['date'].dt.tz_localize(None)

            min_dates.append(curr_data['date'].min())
            max_dates.append(curr_data['date'].max())
        else:
            min_dates.append(None)
            max_dates.append(None)
    gauges_on_river_metadata['min_date'] = min_dates
    gauges_on_river_metadata['max_date'] = max_dates

    gauges_on_river_metadata = gauges_on_river_metadata.dropna(subset=['min_date', 'max_date'])
    gauges_on_river_metadata['type'] = 'water_level'
    curr_t1, curr_t2 = t1+pd.to_timedelta('2 days'), t2-pd.to_timedelta('2 days')
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, curr_t1, curr_t2, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_object)
    selected_gauges['unit'] = 'm'
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_data = df_data.loc[df_data['name'] == row['name']]
        curr_data['date'] = pd.to_datetime(curr_data['dt'])
        curr_data['date'] = curr_data['date'].dt.tz_localize('Asia/Kolkata')
        curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
        curr_data['date'] = curr_data['date'].dt.tz_localize(None)
        curr_data = curr_data[['date', 'WSE']].rename(columns={'WSE': 'stage'})
        data_sampling = pd.to_datetime(curr_data['date']).diff().mode()[0].resolution_string
        gauges_on_river_dict[index] = sc.GaugeStation(row['X'], row['Y'], index, riv, row['chainage'],
                                                      row['unit'], data_sampling)
        gauges_on_river_dict[index].upload_wl(curr_data)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(f'{results_dir}gauge_at_{riv.split(" ")[0]}.pkl', "wb") as f:
        pickle.dump(gauges_on_river_dict, f)


# download_ana_insitu_data(current_river, riv, t1, t2)
# download_ganges_insitu_data(current_river, riv, t1, t2)


def download_and_juxtapose_dahiti_and_gauge_oop(riv_object, loaded_gauges, riv_name, riv_names, basin_nm, vel = None):
    dahiti = DAHITI()
    data = dahiti.list_targets(args={'basin': basin_nm})
    vs_stations = [(vs['dahiti_id'], vs['longitude'], vs['latitude']) for vs in data if vs['target_name'] in riv_names]
    vs_objects = []
    # filtered_gauges = {}
    # for x in [loaded_gauges[x] for x in loaded_gauges.keys() if
    #           loaded_gauges[x].wl_df.index.min() < t1 and loaded_gauges[x].wl_df.index.max() > t2]:
    #     filtered_gauges[x.id] = x
    for vs_set in vs_stations:
        vs_id, vs_x, vs_y = vs_set[0], vs_set[1], vs_set[2]
        VS = sc.VirtualStation(vs_id, vs_x, vs_y)
        if VS.is_away_from_river(riv_object, 5000):
            continue
        VS.upload_chainage(riv_object.get_chainage_of_point(VS.x, VS.y))
        VS.find_closest_gauge_and_chain(loaded_gauges)
        VS.get_water_levels(dahiti)
        if type(VS.wl) != pd.DataFrame:
            continue
        VS.time_filter(t1, t2)
        if len(VS.wl) == 0:
            continue
        if VS.neigh_g_up and VS.neigh_g_dn:
            VS.get_juxtaposed_vs_and_gauge_meas(loaded_gauges[VS.neigh_g_up].wl_df, loaded_gauges[VS.neigh_g_dn].wl_df,
                                                loaded_gauges[VS.neigh_g_dn].sampling, vel)
        elif VS.neigh_g_up:
            VS.get_juxtaposed_vs_and_gauge_meas(loaded_gauges[VS.neigh_g_up].wl_df, None,
                                                loaded_gauges[VS.neigh_g_up].sampling, vel)
        elif VS.neigh_g_dn:
            VS.get_juxtaposed_vs_and_gauge_meas(None, loaded_gauges[VS.neigh_g_dn].wl_df,
                                                loaded_gauges[VS.neigh_g_dn].sampling, vel)
            # continue
        vs_objects.append(VS)

    with open(f'{results_dir}up_and_dn_gauges/vs_updt_riv_at_{riv_name.split(",")[0]}.pkl', "wb") as f:
        pickle.dump(vs_objects, f)


def densify_wl(vs_id, riv_name, riv_object, loaded_stations, loaded_gauges):
    riv = riv_name.split(",")[0]
    tributary_chains = riv_object.tributary_chains if riv in ['Missouri', 'Oder'] else []
    amp_thres, rmse_thres, single_rmse_thres, itpd_method = 1, 10, 0.2, 'akima'
    buffer, corr_thres, bottom = 300, 0.75, 0.1

    VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
    gauge_dist_thres = 5 if riv not in ['Mississippi', 'Ganges'] else 20
    gauge_chain = VS.neigh_g_up_chain if VS.closest_gauge == 'up' else VS.neigh_g_dn_chain
    gauge_id = VS.neigh_g_up if VS.closest_gauge == 'up' else VS.neigh_g_dn
    vs_gauge_dist = abs(VS.chainage - gauge_chain) / 1000
    if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in rv.vs_with_neight_dams[riv]:
        return []
    if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
        return []
    DS = sc.DensificationStation(VS, buffer, None, itpd_method)
    DS.get_upstream_adjacent_vs(loaded_stations)
    df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample('D').mean().dropna()
    DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, riv_object.dams, tributary_chains,
                                                    rv.vs_with_neight_dams[riv], False)
    DS.filter_stations_only_with_swot()
    if DS.is_ds_empty_or_at_edge():
        return []
    DS.get_slope_of_all_vs()
    DS.get_single_vs_interpolated_ts()
    DS.get_densified_wl_by_regressions(rmse_thres=rmse_thres, single_rmse_thres=single_rmse_thres)
    DS.calibrate_mannings_c()
    DS.densified_ts = DS.calculate_shifted_time_by_simplified_mannig(DS.densified_ts, bottom)
    rms_thr = DS.get_rmse_agg_threshold(df_true)
    DS.densified_ts = DS.densified_ts.loc[DS.densified_ts['rmse_sum'] < rms_thr]
    DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
    DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
    DS.densified_itpd = DS.interpolate(DS.densified_daily)

    densified_ts_cval = DS.densified_ts.loc[DS.densified_ts['id_vs'] != DS.id]
    densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
    densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)

    adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[gauge_id].wl_df, gauge_chain)
    DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, DS.densified_ts)
    DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

    DS.get_svr_smoothed_data()
    DS.svr_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, DS.svr_ts)


    rmse_reg, nse_reg = DS.get_rmse_nse_values(DS.densified_itpd, DS.densified_itpd.index.min(),
                                               DS.densified_itpd.index.max(), 'REGRESSIONS')
    rmse_sv, nse_sv = DS.get_rmse_nse_values(DS.svr_itpd, DS.svr_itpd.index.min(),
                                               DS.svr_itpd.index.max(), 'SVR')
    rmse_raw, nse_raw = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(),
                                               DS.single_VS_itpd.index.max(), 'SINGLE VS')
    rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd, densified_ts_cval_itpd.index.min(),
                                                 densified_ts_cval_itpd.index.max(), 'CrossVal', df_true)
    rmse_daily, nse_daily = DS.get_rmse_nse_values(densified_ts_cval_daily, densified_ts_cval_daily.index.min(),
                                                   densified_ts_cval_daily.index.max(), 'Daily')
    rmse_srd, nse_srd = DS.get_rmse_nse_values(DS.wl['wse'].resample('D').mean(), DS.wl.index.min(), DS.wl.index.max(),
                                               'VS ACCURACY')
    mean_bias = DS.densified_ts['bias'].mean()
    mean_uncrt = DS.densified_ts['uncertainty'].mean()
    mean_rmse_sum = DS.densified_ts['rmse_sum'].mean()

    res_path = f'{results_dir}/rs_stations/up_and_dn_gauges/{riv}_RS{DS.id}.pkl'
    with open(res_path, "wb") as f:
        pickle.dump(DS, f)

    return [vs_id, DS.chainage, riv, gauge_chain, DS.speed_ms, DS.c, len(DS.densified_ts), len(DS.densified_ts['id_vs'].unique()),
            mean_bias, mean_uncrt, mean_rmse_sum, rmse_reg, rmse_raw, rmse_cval, rmse_daily, rmse_srd, rmse_sv,
            nse_reg, nse_raw, nse_cval, nse_daily, nse_srd, nse_sv]


calculate_all_data = True
if calculate_all_data:
    results_list = []
    for riv, metric_crs in [('Elbe', '4839'), ('Oder', '2180'), ('Rhine', '4839'), ('Ganges', 'ESRI:102025'), ('Mississippi', 'ESRI:102010'), ('Missouri', 'ESRI:102010'), ('Po', '3035')]:
    # for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Oder', '2180')]:
        river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
        with open(f'{results_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
            current_river = pickle.load(f)
        if riv not in ['Ganges', 'Po', 'Mississippi', 'Missouri', 'Oder', 'Rhine', 'Elbe']:
            t1_1, t1_2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2025-04-29 23:59')
            t2_1, t2_2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2025-10-01 00:00')
            t1 = t1_1 if riv == 'Oder' else t2_1
            t2 = t1_2 if riv == 'Oder' else t2_2
            download_in_situ_data(current_river, river_name, country, t1, t2)
            # download_usgs_insitu_data(current_river, river_name, t1, t2)
        with open(f'{results_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_gauges = pickle.load(f)
        if riv not in ['Ganges', 'Po', 'Mississippi', 'Missouri', 'Oder', 'Rhine']:
            download_and_juxtapose_dahiti_and_gauge_oop(current_river, loaded_gauges, river_name, riv_names, basin_name)

        with open(f'{results_dir}up_and_dn_gauges/vs_updt_riv_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_stations = pickle.load(f)
            loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)

        loaded_stns = copy.deepcopy(loaded_stations)
        # for vs in loaded_stns:
        #     t1, t2 = pd.to_datetime('2023-07-10'), pd.to_datetime('2024-12-31')
        #     vs.wl = vs.wl.loc[(vs.wl['datetime'] > t1) & (vs.wl['datetime'] < t2)]
        #     vs.swot_wl = vs.swot_wl.loc[
        #         (pd.to_datetime(vs.swot_wl['datetime']) > t1) & (pd.to_datetime(vs.swot_wl['datetime']) < t2)]
        #     vs.juxtaposed_wl = vs.juxtaposed_wl.loc[
        #         (pd.to_datetime(vs.juxtaposed_wl['dt']) > t1) & (pd.to_datetime(vs.juxtaposed_wl['dt']) < t2)]

        # if riv not in ['Po', 'Rhine', 'Elbe', 'Odra', 'Mississippi', 'Missouri']:
        # if riv not in ['Po', 'Rhine', 'Elbe', 'Odra', 'Missouri']:
        for vs in [x.id for x in loaded_stns]:
            curr_res = densify_wl(vs, river_name, current_river, loaded_stns, loaded_gauges)
            if len(curr_res) > 0:
                results_list.append(curr_res)
    res_cols = ['id', 'DS.chainage', 'river', 'g_chain', 'velocity', 'c', 'num_of_all_meas',
                'num_of_vs', 'mean_bias', 'mean_uncrt', 'mean_rmse_sum', 'rmse_reg', 'rmse_raw',
                'rmse_cval', 'rmse_daily', 'rmse_srd', 'rmse_sv', 'nse_reg', 'nse_raw', 'nse_cval', 'nse_daily', 'nse_srd', 'nse_sv']
    res_df = pd.DataFrame(results_list, columns=res_cols)

    # filestr = f'{results_dir}up_and_dn_gauges/all_accuracies_v3_svr_akima.csv'
    # old_res = pd.read_csv(filestr, decimal=',', sep=';')
    # old_res = old_res.loc[old_res['river'] != riv]
    # new_res = pd.concat([old_res, res_df])
    # new_res = new_res.reset_index()[res_cols]
    # print(new_res)
    # new_res.to_csv(filestr, decimal=',', sep=';')

    print(res_df)

    """ ACCURACIES PLOT """
    # metric, metric2 = 'RMSE [m]', 'NSE'
    # cols1 = ['rmse_dst', 'rmse_rms']
    # cols2 = ['nse_dst', 'nse_rms']
    # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7, 5))
    # # ax.boxplot(res_df[res_df.columns[[5,7,8,9,11,12]]].dropna(axis=0))  # RMSE
    # selected_data1 = res_df[cols1].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
    # selected_data2 = res_df[cols2].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
    # ax.boxplot(selected_data1)
    # ax.set_xlabel('Methods')
    # ax.set_ylabel(metric)
    # cols1_labels = [f'{lab}\nmean: {round(res_df[col].mean(), 3)}' for lab, col in
    #                 [('RMSE densified RS', cols1[0]), ('RMSE just RS', cols1[1])]]
    # print(cols1_labels)
    # ax.set_xticklabels(cols1_labels)
    #
    # ax2.boxplot(selected_data2)
    # ax2.set_xlabel('Methods')
    # ax2.set_ylabel(metric2)
    # cols2_labels = [f'{lab}\nmean: {round(res_df[col].mean(), 3)}' for lab, col in
    #                 [('NSE densified RS', cols2[0]), ('NSE just RS', cols2[1])]]
    # ax2.set_xticklabels(cols2_labels)
    #
    # ax.grid(True, linestyle='--', alpha=0.6)
    # ax2.grid(True, linestyle='--', alpha=0.6)
    # fig.suptitle(
    #     f'Accuracy of daily water level time series based on {len(selected_data1)} RS at the {riv} River')
    # plt.tight_layout()
    # plt.show(block=True)

    """ ANALYSE CORRELATIONS BETWEEN VS (DAM IMPACT ANALYSIS) """
    # def znajdz_najblizsze_num_bez_duplikatow(dane):
    #     wynik = []
    #     dodane_polaczenia = set()
    #     for id1, num1 in dane:
    #         najblizszy_id = None
    #         najblizszy_num = None
    #         min_roznica = float('inf')
    #         for id2, num2 in dane:
    #             if id1 == id2:
    #                 continue
    #             roznica = abs(num1 - num2)
    #             if roznica < min_roznica:
    #                 min_roznica = roznica
    #                 najblizszy_id = id2
    #                 najblizszy_num = num2
    #             elif roznica == min_roznica:
    #                 if (najblizszy_id is None or
    #                         (id2, num2) < (najblizszy_id, najblizszy_num)):
    #                     najblizszy_id = id2
    #                     najblizszy_num = num2
    #         para1 = (id1, num1)
    #         para2 = (najblizszy_id, najblizszy_num)
    #         unikalne_polaczenie = tuple(sorted([para1, para2]))
    #         if unikalne_polaczenie not in dodane_polaczenia:
    #             dodane_polaczenia.add(unikalne_polaczenie)
    #             if para1 < para2:
    #                 wynik.append([id1, num1, najblizszy_id, najblizszy_num])
    #             else:
    #                 wynik.append([najblizszy_id, najblizszy_num, id1, num1])
    #     return wynik
    #
    #
    # pairs = znajdz_najblizsze_num_bez_duplikatow([(x.id, x.chainage) for x in loaded_stations])
    # for pair in pairs:
    #     vs1_id, vs2_id, vs1_chain, vs2_chain = pair[0], pair[2], pair[1], pair[3]
    #     VS1 = [x for x in loaded_stations if x.id == vs1_id][0]
    #     VS2 = [x for x in loaded_stations if x.id == vs2_id][0]
    #     VS1.wl.index = VS1.wl.index.round('D')
    #     VS2.wl.index = VS2.wl.index.round('D')
    #     df_merged = VS1.wl.join(VS2.wl, how='inner', lsuffix='_df1', rsuffix='_df2')
    #     if df_merged['wse_df1'].corr(df_merged['wse_df2']) < 0.6:
    #         print(vs1_id, vs2_id, vs1_chain, vs2_chain, vs1_chain - vs2_chain,
    #               df_merged['wse_df1'].corr(df_merged['wse_df2']))
    #         fig, ax = plt.subplots()
    #         ax.scatter(df_merged['wse_df1'], df_merged['wse_df2'])
    #         ax.set_title(f'{vs1_id} vs. {vs2_id}')
    #         plt.show(block=True)

    """" DAMS BETWEEN """
    # current_river.get_dams_chainages()
    # for pair in pairs:
    #     vs1_id, vs2_id, vs1_chain, vs2_chain = pair[0], pair[2], pair[1], pair[3]
    #     if sc.is_dam_between(vs1_chain, vs2_chain, current_river.dams):
    #         VS1 = [x for x in loaded_stations if x.id == vs1_id][0]
    #         VS2 = [x for x in loaded_stations if x.id == vs2_id][0]
    #         print(VS1.chainage - VS2.chainage)
    #         fig, ax = plt.subplots()
    #         custom_colors = {1: 'blue', 2: 'steelblue', 3: 'aqua', 4: 'red', 5: 'gray', 6: 'black'}
    #         current_river.gdf['colors'] = current_river.gdf['type'].map(custom_colors)
    #         current_river.gdf.plot(color=current_river.gdf['colors'], ax=ax)
    #         gpd.GeoSeries.from_xy(x=[VS1.x], y=[VS1.y], crs=4326).plot(ax=ax)
    #         gpd.GeoSeries.from_xy(x=[VS2.x], y=[VS2.y], crs=4326).plot(ax=ax)
    #         ax.set_xlim(min([VS1.x, VS2.x]) - .25, max([VS1.x, VS2.x]) + .25)
    #         ax.set_ylim(min([VS1.y, VS2.y]) - .25, max([VS1.y, VS2.y]) + .25)
    #         ax.annotate(str(vs1_chain), (VS1.x, VS1.y), textcoords="offset points", xytext=(5, 5), ha='center',
    #                     fontsize=8, color='black')
    #         ax.annotate(str(vs2_chain), (VS2.x, VS2.y), textcoords="offset points", xytext=(5, 5), ha='center',
    #                     fontsize=8, color='black')
    #         plt.show(block=True)


    """ HOVMULLER DIAGRAM """
    # import matplotlib.dates as mdates
    # from scipy.interpolate import griddata
    # from datetime import datetime, timedelta
    # all_times = []
    # all_chainages = []
    # all_water_levels = []
    #
    # for vs in loaded_stations:
    #     # Upewnij się, że 'dt' jest datetime, a 'vs_anom' numeryczne
    #     times = vs.juxtaposed_wl['dt'].values
    #     water_levels = vs.juxtaposed_wl['vs_anom'].values
    #     chainage = vs.chainage/1000
    #
    #     # Konwersja datetime na timestampy dla interpolacji
    #     # Używamy astype('datetime64[ns]').astype(int) // 10**9, aby uzyskać timestamp w sekundach
    #     # Pandas datetime64 jest w nanosekundach, więc dzielimy przez 10^9
    #     all_times.extend([t.astype('datetime64[ns]').astype(int) // 10**9 for t in times])
    #     all_chainages.extend([chainage] * len(times))
    #     all_water_levels.extend(water_levels)
    #
    # # Konwersja na tablice NumPy
    # all_times = np.array(all_times)
    # all_chainages = np.array(all_chainages)
    # all_water_levels = np.array(all_water_levels)
    #
    # # --- 3. Określ granice siatki i stwórz regularną siatkę ---
    # # Określ zakresy dla osi czasu i kilometrażu
    # min_time_numeric = np.min(all_times)
    # max_time_numeric = np.max(all_times)
    # min_chainage = np.min(all_chainages)
    # max_chainage = np.max(all_chainages)
    #
    # # Liczba punktów na siatce (możesz dostosować dla lepszej rozdzielczości/wydajności)
    # num_time_grid_points = 200 # np. 200 punktów czasowych
    # num_chainage_grid_points = 100 # np. 100 punktów kilometrażowych
    #
    # grid_time_numeric = np.linspace(min_time_numeric, max_time_numeric, num_time_grid_points)
    # grid_chainage = np.linspace(min_chainage, max_chainage, num_chainage_grid_points)
    #
    # # Tworzenie siatki 2D dla interpolacji
    # XI, YI = np.meshgrid(grid_time_numeric, grid_chainage)
    #
    # # --- 4. Interpolacja danych na regularną siatkę ---
    # # 'linear', 'nearest', 'cubic' - wybierz metodę interpolacji
    # # 'linear' jest dobrym punktem wyjścia, 'cubic' może dać gładsze rezultaty,
    # # ale może być wrażliwa na braki danych.
    # water_level_grid = griddata(
    #     (all_times, all_chainages), # punkty źródłowe (czas, kilometraż)
    #     all_water_levels,          # wartości (stan wody)
    #     (XI, YI),                  # punkty docelowe siatki
    #     method='linear'
    # )
    #
    # # Konwersja numerycznej osi czasu z powrotem na obiekty datetime dla matplotlib
    # grid_time_datetime = [datetime.fromtimestamp(t) for t in grid_time_numeric]
    #
    # # --- 5. Wizualizacja diagramu Hovmullera ---
    # plt.figure(figsize=(14, 8)) # Zwiększ rozmiar figury dla lepszej czytelności
    #
    # # Użyj pcolormesh do stworzenia mapy kolorów
    # # Transponowanie water_level_grid jest ważne, ponieważ
    # # pcolormesh oczekuje, że pierwszy wymiar macierzy Z będzie odpowiadał Y (chainage),
    # # a drugi wymiar Z będzie odpowiadał X (czas).
    # plt.pcolormesh(grid_time_datetime, grid_chainage, water_level_grid[:-1, :-1], shading='flat', cmap='viridis')
    #
    # # Dodanie paska kolorów (legendy)
    # cbar = plt.colorbar()
    # cbar.set_label('Anomalia stanu wody [m]') # Zmieniamy etykietę na anomalię
    #
    # # Ustawienia osi i tytułu
    # plt.xlabel('Czas')
    # plt.ylabel('Kilometraż [km]')
    # plt.title('Diagram Hovmullera: Anomalia stanu wody w funkcji czasu i kilometrażu (VS)')
    #
    # # Formatowanie osi czasu
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    # plt.gcf().autofmt_xdate() # Automatyczne formatowanie dat
    #
    # plt.tight_layout() # Dopasowanie elementów, aby nie nachodziły na siebie
    # plt.show(block=True)
