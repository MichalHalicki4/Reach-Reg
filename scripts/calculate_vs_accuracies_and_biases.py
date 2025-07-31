# import pandas as pd
# from datetime import timedelta
# from sklearn.metrics import mean_squared_error
# import matplotlib.cm as cm
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
# import requests
# import json
# import geopandas as gpd
# from shapely.geometry import Point
# from shapely.ops import linemerge, nearest_points
# import warnings
# warnings.filterwarnings('ignore')
#
# geoid_vals = {1: 38.39, 2: 39.03, 3: 39.27, 4: 39.22, 5: 39.26, 6: 39.47, 7: 40.02, 8: 39.68}
#
#
# data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/'
# imgw_file = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/imgw_h_all_gauges_2022_to_2024.csv'
# figs_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/figs_tabs/'
# imgw_df = pd.read_csv(imgw_file, sep=';')
# # typical_freq = pd.to_timedelta(np.diff(imgw_df.index)).value_counts().index[0]
# # imgw_df = imgw_df.resample(typical_freq).fillna(method=None)
#
#
# def get_optimum_lag(ts1, ts2, n):
#     max_corr, best_lag = 0, 0
#     for lag in range(n):
#         ts2_shifted = ts2.copy()
#         ts2_shifted.index = [t + timedelta(hours=lag) for t in ts2_shifted.index]
#         corr = ts1.corr(ts2_shifted)
#         if corr > max_corr:
#             max_corr = corr
#             best_lag = lag
#     return best_lag, round(max_corr, 3)
#
#
# download_and_juxtapose_dahiti_and_gauge = False
# if download_and_juxtapose_dahiti_and_gauge:
#
#     juxtaposed_columns = ['id_vs', 'vs_chain', 'dt', 'gauge_up', 'dist_up', 'gauge_down', 'dist_down', 'lag', 'vs_wl', 'up_wl',
#                           'uncertainty', 'g_anom', 'vs_anom', 'bias']
#     results_df = pd.DataFrame(columns=juxtaposed_columns)
#     list_url = 'https://dahiti.dgfi.tum.de/api/v2/list-targets/'
#     # river_name = 'Wisła, River'
#     # basin_name = 'Wisla'
#     # rzeka = gpd.read_file(f'{data_dir}wisla.shp', crs=2180)
#
#     river_name = 'Oder, River'
#     basin_name = 'Oder'
#     rzeka = gpd.read_file(f'{data_dir}odra.shp', crs=2180)
#
#     """ VS """
#     target_id, station_for_densification = 18750, None
#     # target_id, station_for_densification = 13661, None
#     # target_id, station_for_densification = 19763, None
#     # target_id, station_for_densification = 38861, None
#
#     t1, t2 = pd.to_datetime('2022-01-01 00:00'), pd.to_datetime('2025-01-01 00:00')
#
#     args = {'api_key': '59B654E28B331DF19DFD0E252F4627EB723A281AC163AC648A82C841636131D6', 'basin': basin_name}
#     response = requests.post(list_url, json=args)
#     vs_metadata = []
#     if response.status_code == 200:
#         data = json.loads(response.text)
#         for vs in data['data']:
#             if vs['target_name'] == river_name:
#                 print(vs)
#                 vs_metadata.append(vs)
#
#
#     def get_vs_xy_and_chainage(river, vs_list, vs_num):
#         vs_x, vs_y = vs_list[vs_num]['longitude'], vs_list[vs_num]['latitude']
#         point = gpd.GeoDataFrame(geometry=[Point(vs_x, vs_y)], crs=4326)
#         point = point.to_crs(river.crs)
#         line = river.geometry[0]  # Zakładamy, że mamy tylko jedną linię rzeki
#         merged_line = linemerge(line)
#         reverse_line = merged_line.reverse()
#         naj_punkt = nearest_points(reverse_line, point.geometry[0])[0]
#         odleglosc = reverse_line.project(naj_punkt)
#         return vs_x, vs_y, odleglosc
#
#
#     def get_dahiti_water_levels(vs_code):
#         args = {'api_key': '59B654E28B331DF19DFD0E252F4627EB723A281AC163AC648A82C841636131D6', 'format': 'json'}
#         url = "https://dahiti.dgfi.tum.de/api/v2/download-water-level/"
#         args['dahiti_id'] = vs_code
#         response = requests.post(url, json=args)
#         df = pd.DataFrame(json.loads(response.text)['data'])
#         df = df.set_index(pd.to_datetime(df['datetime']))
#         df['id'] = vs_code
#         return df
#
#
#     def get_imgw_chainage(row):
#         point = gpd.GeoDataFrame(geometry=[Point(row['X'], row['Y'])], crs=4326)
#         point = point.to_crs(rzeka.crs)
#         line = rzeka.geometry[0]  # Zakładamy, że mamy tylko jedną linię rzeki
#         merged_line = linemerge(line)
#         reverse_line = merged_line.reverse()
#         naj_punkt = nearest_points(reverse_line, point.geometry[0])[0]
#         odleglosc = reverse_line.project(naj_punkt)
#         return odleglosc
#
#
#     def find_closest_gauges_and_chains(row_res, curr_gauges_df):
#         chainage_res = row_res['chainage']
#         # Oblicz różnicę w chainage
#         curr_gauges_df['roznica'] = (curr_gauges_df['chainage'] - chainage_res).abs()
#         # 1. Absolutnie najbliższy
#         najblizszy = curr_gauges_df.sort_values(by='roznica').iloc[0]['id']
#         najblizszy_chain = najblizszy = curr_gauges_df.sort_values(by='roznica').iloc[0]
#         # 2. Najbliższy powyżej
#         powyzej = curr_gauges_df[curr_gauges_df['chainage'] > chainage_res].sort_values(by='chainage').iloc[0][
#             'id'] if any(curr_gauges_df['chainage'] > chainage_res) else None
#         powyzej_chain = curr_gauges_df[curr_gauges_df['chainage'] > chainage_res].sort_values(by='chainage').iloc[0][
#             'chainage'] if any(curr_gauges_df['chainage'] > chainage_res) else None
#
#         # 3. Najbliższy poniżej
#         ponizej = \
#             curr_gauges_df[curr_gauges_df['chainage'] < chainage_res].sort_values(by='chainage', ascending=False).iloc[
#                 0][
#                 'id'] if any(curr_gauges_df['chainage'] < chainage_res) else None
#         ponizej_chain = \
#             curr_gauges_df[curr_gauges_df['chainage'] < chainage_res].sort_values(by='chainage', ascending=False).iloc[
#                 0]['chainage'] if any(curr_gauges_df['chainage'] < chainage_res) else None
#         return pd.Series(
#             {'closest_id': najblizszy, 'closest_chain': najblizszy_chain, 'closest_id_up': powyzej,
#              'closest_chain_up': powyzej_chain, 'closest_id_down': ponizej, 'closest_chain_down': ponizej_chain})
#
#
#     coords_and_mean_wls = []
#     coords = []
#     for i in range(len(vs_metadata)):
#         vs_code = vs_metadata[i]['dahiti_id']
#         df = get_dahiti_water_levels(vs_code)
#         wl_mean = df['water_level'].mean()
#         wl_max = df['water_level'].max()
#         wl_min = df['water_level'].min()
#         x, y, chainage = get_vs_xy_and_chainage(rzeka, vs_metadata, i)
#         # df.to_csv(f'dahiti_data/{vs_code}.csv')  # SŁUŻY DO ZAPISYWANIA PLIKÓW
#         coords_and_mean_wls.append([vs_code, x, y, chainage, wl_mean, wl_max, wl_min])
#         coords_and_mean_wls_df = pd.DataFrame(coords_and_mean_wls, columns=['vs_code', 'x', 'y', 'chainage', 'wl_mean', 'wl_max', 'wl_min'])
#         coords_and_mean_wls_df.to_csv(f'{basin_name}_dahiti_coords_and_mean_wls.csv', sep=';')
#         coords.append([vs_code, x, y])
#
#     coords_df = pd.DataFrame(coords, columns=['vs_code', 'x', 'y'])
#     # coords_df.to_csv('dahiti_coords.csv')
#
#     vs_data_df = pd.DataFrame(coords_and_mean_wls, columns=['vs_code', 'x', 'y', 'chainage', 'wl_mean', 'wl_max', 'wl_min'])
#
#     for station in vs_metadata:
#         if station['dahiti_id'] == target_id:
#             station_for_densification = station
#             break
#
#     central_vs_chng = vs_data_df[vs_data_df['vs_code'] == station_for_densification['dahiti_id']]['chainage'].values[0]
#     central_vs_id = station_for_densification['dahiti_id']
#     central_vs_gpd = gpd.GeoDataFrame(
#         geometry=[Point(station_for_densification['longitude'], station_for_densification['latitude'])], crs=4326)
#     central_vs_gpd = central_vs_gpd.to_crs(rzeka.crs)
#
#     vs_data_gdf = gpd.GeoDataFrame(vs_data_df, geometry=[Point(xy) for xy in zip(vs_data_df['x'], vs_data_df['y'])], crs=4326)
#     vs_data_gdf = vs_data_gdf.to_crs(rzeka.crs)
#
#     # bufor = central_vs_gpd.buffer(100000)
#     # vs_gdf_buffer = vs_data_gdf[vs_data_gdf.within(bufor.iloc[0])]
#
#     # vs_gdf_buffer = vs_gdf_buffer[vs_gdf_buffer['chainage'] >= central_vs_chng]
#     # vs_gdf_buffer = pd.merge(vs_gdf_buffer, vs_data_df[['vs_code', 'chainage']], left_on='vs_code', right_on='vs_code',
#     #                           how='left')
#
#     # central_vs_mean_wl = vs_gdf_buffer['wl_mean'][vs_gdf_buffer['vs_code'] == central_vs_id].values[0]
#
#     # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 9))
#     # ax.scatter(vs_data_gdf.x, vs_data_gdf.y, marker='o', color='purple', label=f'all VS on the {river_name}')
#     # ax.scatter(vs_gdf_buffer.x, vs_gdf_buffer.y, marker='o', color='blue', label='vs_location')
#     # ax.scatter(station_for_densification['longitude'], station_for_densification['latitude'], marker='o', color='red',
#     #            label='vs_densification')
#     # rzeka.to_crs(4326).plot(ax=ax, color='black')
#     # ax.legend()
#     #
#     # ax2.scatter(vs_gdf_buffer.x, vs_gdf_buffer.y, marker='o', label='vs_location')
#     # ax2.scatter(station_for_densification['longitude'], station_for_densification['latitude'], marker='o', color='red',
#     #             label='vs_densification')
#     # rzeka.to_crs(4326).plot(ax=ax2, color='black')
#     # for index, row in vs_gdf_buffer.iterrows():
#     #     ax2.annotate(row['vs_code'], (row.x, row.y))
#     #
#     # minx, miny, maxx, maxy = vs_gdf_buffer.to_crs(4326).total_bounds
#     # buffer = 0.1
#     # ax2.set_xlim(minx - buffer, maxx + buffer)
#     # ax2.set_ylim(miny - buffer, maxy + buffer)
#     #
#     # plt.show()
#
#     args = {'api_key': '59B654E28B331DF19DFD0E252F4627EB723A281AC163AC648A82C841636131D6', 'format': 'json'}
#     url = "https://dahiti.dgfi.tum.de/api/v2/download-water-level/"
#     all_dahiti_meas = pd.DataFrame()
#     for vs in vs_data_gdf['vs_code']:
#         args['dahiti_id'] = vs
#         response = requests.post(url, json=args)
#         # print(response.text)
#         df = pd.DataFrame(json.loads(response.text)['data'])
#         df = df.set_index(pd.to_datetime(df['datetime']))
#         df['id'] = vs
#         all_dahiti_meas = pd.concat([all_dahiti_meas, df])
#
#     all_dahiti_meas = pd.merge(all_dahiti_meas, vs_data_df[['vs_code', 'chainage']], left_on='id', right_on='vs_code',
#                                how='left')
#     all_dahiti_meas['datetime'] = pd.to_datetime(all_dahiti_meas['datetime'])
#     curr_dahiti = all_dahiti_meas.loc[(all_dahiti_meas['datetime'] > t1) & (all_dahiti_meas['datetime'] < t2)]
#     print(all_dahiti_meas)
#
#     imgw_coords = pd.read_csv(f'{data_dir}imgw_coords.csv', sep=',')
#     imgw_gdf = gpd.GeoDataFrame(imgw_coords, geometry=[Point(xy) for xy in zip(imgw_coords['X'], imgw_coords['Y'])],
#                                 crs=4326)
#     imgw_gdf = imgw_gdf.to_crs(rzeka.crs)
#     curr_gauges = imgw_gdf[imgw_gdf.within(rzeka.buffer(1000).iloc[0])]
#     curr_gauges['chainage'] = curr_gauges.apply(get_imgw_chainage, axis=1)
#     imgw_on_river = imgw_df.loc[imgw_df['id'].isin(curr_gauges['id'].to_list())]
#
#     vs_data_gdf[['closest_id', 'closest_chain', 'closest_id_up', 'closest_chain_up', 'closest_id_down',
#                     'closest_chain_down']] = vs_data_gdf.apply(
#         lambda row: find_closest_gauges_and_chains(row, curr_gauges), axis=1)
#     curr_gauges = curr_gauges.drop(columns=['roznica'])
#
#     # vs_id = 38367
#     for vs_id in vs_data_gdf['vs_code']:
#         g_id, g_chain = vs_data_gdf[['closest_id', 'closest_chain']][vs_data_gdf['vs_code'] == vs_id].values[0]
#         g_up_id, g_up_chain = \
#         vs_data_gdf[['closest_id_up', 'closest_chain_up']][vs_data_gdf['vs_code'] == vs_id].values[0]
#         g_down_id, g_down_chain = \
#         vs_data_gdf[['closest_id_down', 'closest_chain_down']][vs_data_gdf['vs_code'] == vs_id].values[0]
#         vs_meas = curr_dahiti.loc[curr_dahiti['id'] == vs_id]
#         gauge_meas_up = imgw_on_river.loc[imgw_on_river['id'] == g_up_id]
#         gauge_meas_up = gauge_meas_up.set_index(pd.to_datetime(gauge_meas_up['date']))
#         gauge_meas_up = gauge_meas_up.sort_index()
#         gauge_meas_down = imgw_on_river.loc[imgw_on_river['id'] == g_down_id]
#         gauge_meas_down = gauge_meas_down.set_index(pd.to_datetime(gauge_meas_down['date']))
#         gauge_meas_down = gauge_meas_down.sort_index()
#
#         juxtaposed_data = []
#         for index, row in vs_meas.iterrows():
#             vs_wl, vs_dt = row[['water_level', 'datetime']]
#             vs_dt_prev = vs_dt - pd.to_timedelta('5 days')
#             vs_chainage = row['chainage']
#
#             ts_down = gauge_meas_down['stage'].loc[(gauge_meas_down.index > vs_dt_prev) & (gauge_meas_down.index < vs_dt)]
#             ts_up = gauge_meas_up['stage'].loc[(gauge_meas_up.index > vs_dt_prev) & (gauge_meas_up.index < vs_dt)]
#
#             lag, corr = get_optimum_lag(ts_down, ts_up, 50)
#             ratio = g_up_chain / (g_up_chain + g_down_chain)
#             final_lag = lag * ratio
#
#             gauge_up_time = (vs_dt - pd.to_timedelta(f'{final_lag} hours')).round('H')
#             gauge_wl = np.nan
#             for i in range(6):
#                 try:
#                     gauge_wl = \
#                     gauge_meas_up['stage'][gauge_meas_up.index == gauge_up_time + pd.to_timedelta(f'{i} hours')].values[0]
#                     break
#                 except IndexError:
#                     try:
#                         gauge_wl = gauge_meas_up['stage'][
#                             gauge_meas_up.index == gauge_up_time + pd.to_timedelta(f'-{i} hours')].values[0]
#                         break
#                     except IndexError:
#                         continue
#             print(vs_id, vs_dt, vs_wl, gauge_wl, final_lag)
#             juxtaposed_data.append([vs_id, vs_chainage, vs_dt, g_up_id, g_up_chain, g_down_id, g_down_chain, final_lag, vs_wl, gauge_wl, row['error']])
#         curr_results = pd.DataFrame(juxtaposed_data, columns=juxtaposed_columns[:-3])
#         mean_g, mean_vs = curr_results['up_wl'].mean(), curr_results['vs_wl'].mean()
#         curr_results['g_anom'] = curr_results['up_wl'] - mean_g
#         curr_results['vs_anom'] = curr_results['vs_wl'] - mean_vs
#         curr_results['bias'] = abs(curr_results['vs_anom'] - curr_results['g_anom'])
#         results_df = pd.concat([results_df, curr_results])
#     results_df.to_csv(f'{basin_name}_juxtaposed_wl.csv', sep=';')
#     print(1)
# # vs_metadata['lag_pair'] = ['1-2', '1-2', '1-2', '24-3', '3-4', '3-4', '4-5', '4-5']
#
#
# densify_wl = False
#
# river_name = 'Oder, River'
# basin_name = 'Oder'
# coords_and_mean_wls_df = pd.read_csv(f'{basin_name}_dahiti_coords_and_mean_wls.csv', sep=';')
# vs_ordered = coords_and_mean_wls_df.sort_values(by='chainage')['vs_code'].to_list()
#
# speeds, speeds_acc = np.arange(.1, 5.1, 0.1).tolist(), []
# buffers, buffers_acc = np.arange(50, 250, 10).tolist(), []
# stations, stations_acc = vs_ordered[1::5], []
# stations = [13661]
# if densify_wl:
#     rzeka = gpd.read_file(f'{data_dir}odra.shp', crs=2180)
#     # target_id = 13661
#     for target_id in stations:
#         # target_id = 18750
#         speed_kmh = 2
#         buffer = 100
#         all_river_dahiti_meas = pd.read_csv(f'{basin_name}_juxtaposed_wl.csv', sep=';')
#         all_river_dahiti_meas = all_river_dahiti_meas.set_index(pd.to_datetime(all_river_dahiti_meas['dt']))
#         central_vs_chng = all_river_dahiti_meas['vs_chain'].loc[all_river_dahiti_meas['id_vs'] == target_id].values[0]
#         for speed_kmh in speeds:
#         # for buffer in buffers:
#             speed_kmh = 2
#             all_dahiti_meas = all_river_dahiti_meas.loc[(all_river_dahiti_meas['vs_chain'] >= central_vs_chng) &
#                                                         (all_river_dahiti_meas['vs_chain'] < central_vs_chng + buffer*1000)]
#             t1, t2 = pd.to_datetime('2023-01-01'), pd.to_datetime('2024-12-31')
#             # speed_kmh = 3  # km/h
#             slope = 0.275  # m/km
#             speed_ms = speed_kmh * 1000 / 3600  # Convert to meters per second
#
#             def get_shifted_wl(row):
#                 mean_wl = curr_dahiti['vs_wl'].loc[curr_dahiti['id_vs'] == row['id_vs']].mean()
#                 central_vs_curr_mean_wl = curr_dahiti['vs_wl'].loc[curr_dahiti['id_vs'] == target_id].mean()
#                 return row['vs_wl'] - (mean_wl - central_vs_curr_mean_wl)
#
#
#             def get_corresponding_gauge_wl(row):
#                 try:
#                     return curr_gauge['shifted_wl'].loc[
#                         curr_gauge['shifted_time'].dt.round('H') == row['shifted_time'].round('H')].values[0]
#                 except IndexError:
#                     return np.nan
#
#
#             curr_dahiti = all_dahiti_meas.loc[(all_dahiti_meas.index > t1) & (all_dahiti_meas.index < t2)]
#             # curr_dahiti = pd.merge(curr_dahiti, res_df[['vs_code', 'chainage']], left_on='id', right_on='vs_code', how='left')
#             curr_dahiti = curr_dahiti.sort_index()
#             # fig, ax = plt.subplots(figsize=(11,4))
#             # ax.plot(curr_dahiti['dt'], curr_dahiti['water_level'], marker='o', color='blue', label='water level')
#
#             # Calculate time and vertical differences
#             curr_dahiti['time_diff_seconds'] = (curr_dahiti['vs_chain'] - central_vs_chng) / speed_ms
#             # curr_dahiti['shifted_wl'] = curr_dahiti['water_level'] - (curr_dahiti['chainage'] - central_vs_chng) * slope/1000
#             # Convert time difference to timedelta
#             curr_dahiti['time_diff'] = pd.to_timedelta(curr_dahiti['time_diff_seconds'], unit='s')
#
#             # Shift the time
#             curr_dahiti['shifted_time'] = pd.to_datetime(curr_dahiti['dt']) + curr_dahiti['time_diff']
#             reference_station_data = curr_dahiti.loc[curr_dahiti['id_vs'] == target_id]
#             curr_dahiti = curr_dahiti.sort_values(by='shifted_time')
#
#             curr_dahiti['shifted_wl'] = curr_dahiti.apply(get_shifted_wl, axis=1)
#             reference_station_data = curr_dahiti.loc[curr_dahiti['id_vs'] == target_id]
#
#             # fig, ax = plt.subplots(figsize=(11, 4))
#             # ax.plot(curr_dahiti['shifted_time'], curr_dahiti['shifted_wl'], marker='o', color='blue', label='water level',
#             #         zorder=2)
#             # ax.scatter(reference_station_data['shifted_time'], reference_station_data['shifted_wl'], marker='d', s=50,
#             #            color='red', label='water level from reference station', zorder=11)
#             # ax.legend()
#             # plt.show()
#
#             imgw_coords = pd.read_csv(f'{data_dir}imgw_coords.csv', sep=',')
#             imgw_gdf = gpd.GeoDataFrame(imgw_coords, geometry=[Point(xy) for xy in zip(imgw_coords['X'], imgw_coords['Y'])],
#                                         crs=4326)
#             imgw_gdf = imgw_gdf.to_crs(rzeka.crs)
#             curr_gauges = imgw_gdf[imgw_gdf.within(rzeka.buffer(1000).iloc[0])]
#             imgw_on_river = imgw_df.loc[imgw_df['id'].isin(curr_gauges['id'].to_list())]
#             imgw_chainage = reference_station_data['dist_up'].values[0]
#             gauge_meas_up = imgw_on_river.loc[imgw_on_river['id'] == reference_station_data['gauge_up'].values[0]]
#             gauge_meas_up = gauge_meas_up.set_index(pd.to_datetime(gauge_meas_up['date']))
#             gauge_meas_up = gauge_meas_up.sort_index()
#
#             curr_gauge = gauge_meas_up.loc[(gauge_meas_up.index > t1) & (gauge_meas_up.index < t2)]
#             mean_gauge = curr_gauge['stage'].mean()
#             curr_gauge['shifted_wl'] = curr_gauge['stage'] / 100 + reference_station_data[
#                 'shifted_wl'].mean() - mean_gauge / 100
#
#             time_diff = pd.to_timedelta((imgw_chainage - central_vs_chng) / speed_ms, unit='s')
#             # Shift the time
#             curr_gauge['shifted_time'] = pd.to_datetime(curr_gauge.index) + time_diff
#
#             curr_dahiti['shifted_wl_gauge'] = curr_dahiti.apply(get_corresponding_gauge_wl, axis=1)
#             mean_g, mean_vs = curr_dahiti['shifted_wl_gauge'].mean(), curr_dahiti['shifted_wl'].mean()
#             curr_dahiti['shifted_wl_gauge_anom'] = curr_dahiti['shifted_wl_gauge'] - mean_g
#             curr_dahiti['shifted_wl_anom'] = curr_dahiti['shifted_wl'] - mean_vs
#             curr_dahiti['shifted_wl_bias'] = abs(curr_dahiti['shifted_wl_anom'] - curr_dahiti['shifted_wl_gauge_anom'])
#
#             # fig, axs = plt.subplot_mosaic(
#             #     [['upper_left', 'upper_right'],
#             #      ['bottom', 'bottom']],
#             #     figsize=(18, 9),  # Możesz dostosować rozmiar figury
#             #     height_ratios=[1, 2]  # Proporcje wysokości wierszy: górny ma 1 jednostkę, dolny ma 2
#             # )
#             #
#             # ax1 = axs['upper_left']
#             # ax2 = axs['upper_right']
#             # ax = axs['bottom']
#             #
#             # curr_dahiti['chainage_diff'] = (curr_dahiti['vs_chain'] - central_vs_chng) / 1000
#             # n_classes = 10
#             #
#             # # Tworzymy granice klas na podstawie min i max wartości chainage_diff
#             # min_val = curr_dahiti['chainage_diff'].min()
#             # max_val = curr_dahiti['chainage_diff'].max()
#             # bins = np.linspace(min_val, max_val, n_classes + 1)
#             #
#             # # Przypisujemy każdemu punktowi indeks klasy
#             # digitized = np.digitize(curr_dahiti['chainage_diff'], bins) - 1  # Indeksy od 0 do n_classes - 1
#             #
#             # # Tworzymy niestandardową mapę kolorów od zielonego do czerwonego
#             # colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Zielony -> Żółty -> Czerwony
#             # cmap = LinearSegmentedColormap.from_list("my_gyr", colors, N=n_classes)
#             #
#             # # Tworzymy BoundaryNorm na podstawie naszych przedziałów
#             # norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
#             #
#             # # ax.plot(curr_dahiti['shifted_time'], curr_dahiti['shifted_wl'], marker='.', color='blue', label='water level',
#             # #         zorder=2)
#             # scatter = ax.scatter(curr_dahiti['shifted_time'], curr_dahiti['shifted_wl'],
#             #                      c=curr_dahiti['chainage_diff'], cmap=cmap, marker='o', label='water level colored by distance',
#             #                      zorder=2, norm=norm)
#             # ax.errorbar(curr_dahiti['shifted_time'], curr_dahiti['shifted_wl'], yerr=curr_dahiti['bias'], ecolor='red',
#             #             label='VS measurement bias')
#             # ax.scatter(reference_station_data['shifted_time'], reference_station_data['shifted_wl'], marker='d', s=50,
#             #            color='purple', label='water level from reference station', zorder=11)
#             # ax.plot(curr_gauge['shifted_time'], curr_gauge['shifted_wl'], color='black', label='shifted gauge WL')
#             # ax.set_xlabel('Time')
#             # ax.set_ylabel('Water level [m]')
#             # # ax.plot(curr_gauge['shifted_wl'], color='grey', label='gauge data_no_shift')
#             # # Tworzymy pasek kolorów powiązany z scatter i naszą mapą kolorów i normalizacją
#             # cbar = plt.colorbar(scatter, ticks=bins)
#             # # Ustawiamy etykiety ticków na wartości granic przedziałów
#             # cbar.set_label('Along-river distance from ref. station')
#             # cbar.ax.set_yticklabels([f'{b:.0f}' for b in bins])
#             # plt.grid(True)
#             # ax.legend()
#             #
#             # ax1.scatter(curr_dahiti['chainage_diff'], curr_dahiti['shifted_wl_bias'])
#             # ax1.set_xlabel('Along-river distance from ref. station')
#             # ax1.set_ylabel('Bias to shifted gauge WL')
#             #
#             # ax2.scatter(curr_dahiti['bias'], curr_dahiti['shifted_wl_bias'])
#             # ax2.set_xlabel('Bias of VS measurement')
#             # ax2.set_ylabel('Bias to shifted gauge WL')
#             # fig.tight_layout()
#             # plt.show(block=True)
#
#             """ SECOND PLOT - INTERPOLATED AND DAILY MEANS """
#             import hydroeval as he
#             fig, ax = plt.subplots(figsize=(18, 9))
#             resampled = pd.Series(curr_dahiti['shifted_wl'].values, index=curr_dahiti['shifted_time'])
#             resampled_gauge = pd.Series(curr_gauge['shifted_wl'].values, index=curr_gauge['shifted_time'])
#             resampled_gauge_mean = resampled_gauge.resample('D').mean().dropna()
#             resampled_mean = resampled.resample('D').mean().interpolate(method='akima')
#             df_combined = pd.concat([resampled_gauge_mean, resampled_mean], axis=1)
#             df_combined.columns = ['gauge_mean', 'model_mean']
#             df_cleaned = df_combined.dropna()
#             y_true = df_cleaned['gauge_mean']
#             y_predicted = df_cleaned['model_mean']
#             rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
#             nse = he.evaluator(he.nse, y_predicted, y_true)
#             # speeds_acc.append([speed_kmh, rmse, nse])
#             # print(speed_kmh, rmse, nse)
#             # buffers_acc.append([buffer, rmse, nse])
#             # print(buffer, rmse, nse)
#             # stations_acc.append([target_id, buffer, rmse, nse])
#             # print(target_id, buffer, rmse, nse)
#             stations_acc.append([target_id, speed_kmh, rmse, nse])
#             print(target_id, speed_kmh, rmse, nse)
#
#             ax.plot(resampled.resample('D').mean().dropna(), marker='.', color='blue', label='water level',
#                     zorder=2)
#             ax.plot(resampled.resample('D').mean().interpolate(method='akima'), marker='o', color='orange',
#                     label='interpolated')
#             ax.scatter(reference_station_data['shifted_time'], reference_station_data['shifted_wl'], marker='d', s=50,
#                        color='purple', label='water level from reference station', zorder=11)
#             ax.plot(resampled_gauge.resample('D').mean(), color='black', label='1-day shifted gauge WL')
#             ax.set_xlabel('Time')
#             ax.set_ylabel('Water level [m]')
#             ax.set_title(f'VS {target_id}, RMSE: {round(rmse, 2)} m, NSE: {round(nse[0], 3)}')
#             plt.grid(True)
#             ax.legend()
#             plt.show(block=True)
#             print(f"RMSE: {rmse}, NSE: {nse}")
#
#             """ THIRD PLOT: UNCERTAINTY VS BIAS"""
#             # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#             # ax.hist(curr_dahiti['uncertainty'])
#             # ax.set_xlabel('Altimetric measurement uncertainty')
#             # ax.set_ylabel('Ocurrences')
#             # ax2.scatter(curr_dahiti['uncertainty'], curr_dahiti['shifted_wl_bias'])
#             # ax2.set_xlabel('Altimetric measurement uncertainty')
#             # ax2.set_ylabel('Shifted WL bias')
#             # plt.show(block='True')
#
#     # buffers_df = pd.DataFrame(buffers_acc, columns=['buffer', 'rmse', 'nse'])
#     # fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 9))
#     # ax.plot(buffers_df['buffer'], buffers_df['rmse'])
#     # ax.set_ylabel('RMSE [m]')
#     # ax2.plot(buffers_df['buffer'], buffers_df['nse'])
#     # ax2.set_xlabel('Along-river buffer [km]')
#     # ax2.set_ylabel('NSE')
#     # fig.tight_layout()
#     # plt.savefig(f'{figs_dir}{target_id}_buffer_accuracies.png')
#
#     # speed_df = pd.DataFrame(speeds_acc, columns=['speed', 'rmse', 'nse'])
#     # fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 9))
#     # ax.plot(speed_df['speed'], speed_df['rmse'])
#     # # ax.set_xlabel('flow velocity [km/h]')
#     # ax.set_ylabel('RMSE [m]')
#     # ax2.plot(speed_df['speed'], speed_df['nse'])
#     # ax2.set_xlabel('Flow velocity [km/h]')
#     # ax2.set_ylabel('NSE')
#     # fig.tight_layout()
#     # plt.savefig(f'{figs_dir}{target_id}_speed_accuracies.png')
#
#     # targets_df = pd.DataFrame(stations_acc, columns=['target', 'buffer', 'rmse', 'nse'])
#     # fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 9))
#     # num_targets = len(stations)
#     # viridis_colors = cm.viridis(np.linspace(0, 1, num_targets))  # Generuj kolory z palety viridis
#     #
#     # for i, target_id in enumerate(stations):
#     #     buffers_df = targets_df.loc[targets_df['target'] == target_id]
#     #     target_chain = all_river_dahiti_meas['vs_chain'].loc[all_river_dahiti_meas['id_vs'] == target_id].values[0]
#     #     color = viridis_colors[i]
#     #
#     #     ax.plot(buffers_df['buffer'], buffers_df['rmse'], label=f'{target_id} [{target_chain} km]', color=color)
#     #     ax2.plot(buffers_df['buffer'], buffers_df['nse'], color=color)
#     # ax.set_ylabel('RMSE [m]')
#     # # ax2.set_xlabel('Along-river buffer [km]')
#     # ax2.set_ylabel('NSE')
#     # ax2.legend()
#     # fig.tight_layout()
#     # plt.show()
#         # plt.savefig(f'{figs_dir}{target_id}_buffer_accuracies.png')
#
#     targets_df = pd.DataFrame(stations_acc, columns=['target', 'speed', 'rmse', 'nse'])
#     fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 9))
#     num_targets = len(stations)
#     viridis_colors = cm.viridis(np.linspace(0, 1, num_targets))  # Generuj kolory z palety viridis
#
#     for i, target_id in enumerate(stations):
#         speeds_df = targets_df.loc[targets_df['target'] == target_id]
#         target_chain = all_river_dahiti_meas['vs_chain'].loc[all_river_dahiti_meas['id_vs'] == target_id].values[0]
#         color = viridis_colors[i]
#
#         ax.plot(speeds_df['speed'], speeds_df['rmse'], label=f'{target_id} [{round(target_chain/1000, 2)} km]', color=color)
#         ax2.plot(speeds_df['speed'], speeds_df['nse'], label=f'{target_id} [{round(target_chain/1000, 2)} km]', color=color)
#     ax.set_ylabel('RMSE [m]')
#     ax2.set_xlabel('Flow velocity [km/h]')
#     ax2.set_ylabel('NSE')
#     ax2.legend()
#     fig.tight_layout()
#     plt.show()
#
#
#
#     print(1)
#
#
# sword_topology_playground = False
# if sword_topology_playground:
#     from River_class import River, dahiti_river_names_and_basins, find_main_stream
#     from dahitiapi.DAHITI import DAHITI
#
#     river_name, basin_name, up_reach, dn_reach = dahiti_river_names_and_basins['odra'].values()
#     riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp'
#     sword_rivers = gpd.read_file(riv_path)
#     river_reaches_li = find_main_stream(sword_rivers, dn_reach, up_reach)
#     selected_river = sword_rivers[sword_rivers['reach_id'].isin(river_reaches_li)]
#     current_river = River(selected_river, '2180', river_name)
#     current_river.get_simplified_geometry()
#
#     dahiti = DAHITI()
#     dahiti_id = 13661
#     ret = dahiti.download_water_level(dahiti_id)
#     print(ret)
#
#     list_url = 'https://dahiti.dgfi.tum.de/api/v2/list-targets/'
#     args = {'api_key': '59B654E28B331DF19DFD0E252F4627EB723A281AC163AC648A82C841636131D6', 'basin': basin_name}
#     response = requests.post(list_url, json=args)
#     vs_metadata = []
#     if response.status_code == 200:
#         data = json.loads(response.text)
#         for vs in data['data']:
#             if vs['target_name'] == river_name:
#                 print(vs)
#                 vs_metadata.append(vs)
#
#     stations_xy = []
#     for i in range(len(vs_metadata)):
#         vs_x, vs_y = vs_metadata[i]['longitude'], vs_metadata[i]['latitude']
#         chainage = current_river.get_chainage_of_point(vs_x, vs_y)
#         stations_xy.append([vs_x, vs_y, chainage])
#     stations_xy_df = pd.DataFrame(stations_xy, columns=['x', 'y', 'dist'])
#     stations_xy_gdf = gpd.GeoDataFrame(stations_xy_df, geometry=gpd.points_from_xy(stations_xy_df.x, stations_xy_df.y),
#                                        crs=4326)
#
#     fig, ax = plt.subplots()
#     current_river.gdf.plot(ax=ax)
#     stations_xy_df.plot(ax=ax, column='dist', cmap='viridis', legend=True)
#     ax.set_title(f'{river_name}')
#     plt.show(block='True')
#     print(1)
#
#
# new_dahiti_download = False
# if new_dahiti_download:
#     from dahitiapi.DAHITI import DAHITI
#     # Initialize Class
#     dahiti = DAHITI()
#     dahiti_id = 13661
#     ret = dahiti.download_water_level(dahiti_id)
#     print(ret)


""" ------------------- OOP APPROACH  ------------------- """
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

data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/'
t1, t2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2024-12-31 23:59')
velocity, buffer = 30 / 36, 100
riv, metric_crs = 'Rhine', '4839'
# riv, metric_crs = 'Wisla', '2180'
# riv, metric_crs = 'Odra', '2180'
# riv, metric_crs = 'Elbe', '4839'
# riv, metric_crs = 'Danube', '3035'
river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp'  # POLAND
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb23_v17b.shp'  # ELBE, RHINE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb22_v17b.shp'  # DANUBE

save_river_to_file = False
if save_river_to_file:
    current_river = rv.prepare_river_object(riv_path, riv, metric_crs)
    current_river.upload_dam_and_tributary_chains(rv.river_tributary_reaches[riv])
    with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "wb") as f:
        pickle.dump(current_river, f)
else:
    with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
        current_river = pickle.load(f)

download_dahiti_insitu_data = False
if download_dahiti_insitu_data:
    insitu = InSitu()
    stations_data = sc.get_list_of_stations_from_country(country, insitu)
    insitu_df = pd.DataFrame(stations_data)
    insitu_gdf = gpd.GeoDataFrame(insitu_df, geometry=gpd.points_from_xy(insitu_df.longitude, insitu_df.latitude),
                                  crs="EPSG:4326")
    insitu_gdf = insitu_gdf.to_crs(current_river.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'longitude': 'X', 'latitude': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, current_river)
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, t2)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, current_river)
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        gauges_on_river_dict[row['id']] = sc.GaugeStation(row['X'], row['Y'], row['id'], river_name, row['chainage'], row['unit'], row['data_sampling'])
        row_data = insitu.download(int(row['id']))
        row_df = pd.DataFrame(row_data['data']).rename(columns={'value': 'stage'})
        gauges_on_river_dict[row['id']].upload_wl(row_df)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "wb") as f:
        pickle.dump(gauges_on_river_dict, f)
    print(1)

download_and_juxtapose_dahiti_and_gauge_oop = False
if download_and_juxtapose_dahiti_and_gauge_oop:
    dahiti = DAHITI()
    data = dahiti.list_targets(args={'basin': basin_name})
    vs_stations = [vs['dahiti_id'] for vs in data if vs['target_name'] in riv_names]
    vs_objects = []
    # t1, t2 = pd.to_datetime('2022-01-01 00:00'), pd.to_datetime('2025-01-01 00:00')
    with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
        loaded_gauges = pickle.load(f)
    # filtered_gauges = {}
    # for x in [loaded_gauges[x] for x in loaded_gauges.keys() if
    #           loaded_gauges[x].wl_df.index.min() < t1 and loaded_gauges[x].wl_df.index.max() > t2]:
    #     filtered_gauges[x.id] = x
    for vs_id in vs_stations:
        VS = sc.VirtualStation(vs_id)
        VS.get_water_levels(dahiti)
        if type(VS.wl) != pd.DataFrame:
            continue
        VS.time_filter(t1, t2)
        VS.upload_chainage(current_river.get_chainage_of_point(VS.x, VS.y))
        VS.find_closest_gauge_and_chain(loaded_gauges)
        if VS.neigh_g_up:
            VS.get_juxtaposed_vs_and_gauge_meas(loaded_gauges[VS.neigh_g_up].wl_df, loaded_gauges[VS.neigh_g_dn].wl_df, loaded_gauges[VS.neigh_g_dn].sampling, velocity)
            vs_objects.append(VS)

    with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "wb") as f:
        pickle.dump(vs_objects, f)


densify_wl_dahiti = True
if densify_wl_dahiti:
    with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
        loaded_stations = pickle.load(f)
        loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
    with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
        loaded_gauges = pickle.load(f)

    test_different_approaches = False
    if test_different_approaches:
        # vs_id, buffer = 41895, 100  # RHINE
        # vs_id, buffer = 41905, 100  # RHINE, CLOSE TO MOUTH
        # vs_id, buffer = 41925, 100  # ELBE
        # vs_id, buffer = 13651, 100  # ELBE
        vs_id, buffer = 38476, 100  # ELBE
        # vs_id, buffer = 41801, 100  # DANUBE
        # for velocity in [x/36 for x in range(10, 140, 10)]:
        velocity = 30 / 36
        VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
        DS = sc.DensificationStation(VS)
        DS.get_upstream_adjacent_vs(loaded_stations, buffer)
        DS.get_densified_wl(velocity)
        # DS.plot_densified_wl()
        DS.juxtapose_gauge_data_to_vs(loaded_gauges[DS.neigh_g_up].wl_df)
        DS.get_closest_in_situ_daily_wl(loaded_gauges[VS.neigh_g_up].wl_df, DS.juxtaposed_wl['dt'].min(),
                                        DS.juxtaposed_wl['dt'].max())
        # DS.plot_densified_vs_gauge_color_by_chainage()
        DS.juxtaposed_wl = DS.juxtaposed_wl.loc[DS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]
        DS.get_daily_interpolated_wl_ts()
        DS.get_single_vs_interpolated_ts()
        DS.dist_weighted_daily_wl = DS.get_dist_weighted_wl(DS.densified_wl)
        DS.dist_weighted_daily_wl_itpd = DS.interpolate(DS.dist_weighted_daily_wl)
        DS.get_densified_wl_by_norm(DS.juxtaposed_wl, velocity)
        DS.normalized_ts_daily = DS.get_dist_weighted_wl(DS.normalized_ts)
        DS.normalized_ts_itpd = DS.interpolate(DS.normalized_ts_daily)
        DS.get_spline_interpolated_ts(DS.dist_weighted_daily_wl, 'wl_weighted', 5)
        DS.get_rmse_nse_values(DS.interpolated_wl, DS.juxtaposed_wl['dt'].min(), DS.juxtaposed_wl['dt'].max(),
                               'CLASSIC METHOD:')
        DS.get_rmse_nse_values(DS.spline_itpd_wl, DS.juxtaposed_wl['dt'].min(), DS.juxtaposed_wl['dt'].max(),
                               'SPLINE AND DIST WEIGHT:')
        DS.get_rmse_nse_values(DS.single_VS_itpd, DS.juxtaposed_wl['dt'].min(), DS.juxtaposed_wl['dt'].max(),
                               'RAW INTERPOLATION')
        DS.get_rmse_nse_values(DS.normalized_ts_itpd, DS.juxtaposed_wl['dt'].min(), DS.juxtaposed_wl['dt'].max(),
                               'NORMALIZED')
        print('______________________________')

    analyse_accuracies = True
    if analyse_accuracies:
        velocity, buffer = 30/36, 150
        accuracies = []
        corr_thres, amp_thres = 0.8, 1
        itpd_method, norm_method = 'akima', 'standard'
        gauge_dist_thres = 20
        # tributary_chains = current_river.tributary_chains[:-1]
        tributary_chains = []
        neigh_dam_vs = rv.vs_with_neight_dams[riv]
        res_cols_all_vs = ['id', 'chain', 'velocity', 'data_len', 'vs_len', 'mean_bias', 'mean_uncrt',
                    'swot_mean_bias', 'swot_mean_uncrt', 'rmse_c', 'rmse_s', 'rmse_n', 'rmse_r', 'rmse_sc',
                    'rmse_ss', 'rmse_sn', 'rmse_sr', 'nse_c', 'nse_s', 'nse_n', 'nse_r', 'nse_sc', 'nse_ss',
                    'nse_sn', 'nse_sr', 'x', 'y']
        res_cols_swot = ['id', 'chain', 'velocity', 'data_len', 'vs_len', 'swot_mean_bias', 'swot_mean_uncrt',
                         'rmse_sc', 'rmse_ss', 'rmse_sn', 'rmse_srg', 'rmse_sr', 'nse_sc', 'nse_ss', 'nse_sn',
                         'nse_srg', 'nse_sr', 'x', 'y']
        # t1, t2 = pd.to_datetime('2023-07-11'), pd.to_datetime('2025-05-01')
        for vs_id in [x.id for x in loaded_stations]:
        # chainage_limit = [x for x in loaded_stations if x.id == 15352][0].chainage  # WISŁA > WŁOCŁAWKA
        # chainage_limit = [x for x in loaded_stations if x.id == 23410][0].chainage  # ODRA > Ścinawy
        # for vs_id in [x.id for x in [x for x in loaded_stations if x.chainage < chainage_limit]]:
            VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
            vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
            if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
                continue
                """ ALL VS: """
            all_vs = False
            if all_vs:
                DS = sc.DensificationStation(VS)
                DS.get_upstream_adjacent_vs(loaded_stations, buffer)
                DS.filter_upstream_stations_by_correlation(corr_thres, False)
                DS.filter_upstream_stations_by_wl_amplitude(amp_thres)
                DS.filter_upstream_stations_by_dams_and_tributaries(current_river.dams, tributary_chains)
                if len(DS.upstream_adjacent_vs) == 0:
                    print(f'NO UPSTREAM VS MEETING THE CORR AND AMP CRITERIA AT DS {DS.id}')
                    continue
                DS.get_densified_wl(velocity)
                DS.densified_wl = DS.densified_wl.loc[(DS.densified_wl['dt'] > t1) & (DS.densified_wl['dt'] < t2)]
                adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_mean_diff(loaded_gauges[DS.neigh_g_up].wl_df)
                DS.juxtapose_gauge_data_to_vs(adjusted_gauge_data)
                DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

                DS.get_daily_interpolated_wl_ts(method=itpd_method)
                DS.get_single_vs_interpolated_ts()
                DS.dist_weighted_daily_wl = DS.get_dist_weighted_wl(DS.densified_wl)
                DS.dist_weighted_daily_wl_itpd = DS.interpolate(DS.dist_weighted_daily_wl)
                DS.get_densified_wl_by_norm(DS.juxtaposed_wl, velocity, norm_method)
                DS.normalized_ts_daily = DS.get_dist_weighted_wl(DS.normalized_ts)
                DS.normalized_ts_itpd = DS.interpolate(DS.normalized_ts_daily)

                DS.get_spline_interpolated_ts(DS.dist_weighted_daily_wl, 'wl_weighted', 5)
                rmse_c, nse_c = DS.get_rmse_nse_values(DS.interpolated_wl, DS.interpolated_wl.index.min(), DS.interpolated_wl.index.max(), 'CLASSIC METHOD:')
                rmse_s, nse_s = DS.get_rmse_nse_values(DS.spline_itpd_wl, DS.spline_itpd_wl.index.min(), DS.spline_itpd_wl.index.max(), 'SPLINE AND DIST WEIGHT:')
                rmse_n, nse_n = DS.get_rmse_nse_values(DS.normalized_ts_itpd, DS.normalized_ts_itpd.index.min(), DS.normalized_ts_itpd.index.max(), 'NORMALIZED')
                rmse_r, nse_r = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(), DS.single_VS_itpd.index.max(), 'RAW INTERPOLATION')
                mean_bias, mean_uncrt = DS.densified_wl['bias'].mean(), DS.densified_wl['uncertainty'].mean()

            """ SWOT: """
            DS = sc.DensificationStation(VS)
            DS.get_upstream_adjacent_vs(loaded_stations, buffer)
            DS.filter_upstream_stations_by_correlation(corr_thres, True)
            DS.filter_upstream_stations_by_wl_amplitude(amp_thres)
            DS.filter_upstream_stations_by_dams_and_tributaries(current_river.dams, tributary_chains)
            if len(DS.upstream_adjacent_vs) == 0:
                    print(f'NO UPSTREAM VS MEETING THE CORR AND AMP CRITERIA AT DS {DS.id}')
                    continue
            DS.get_densified_wl(velocity)
            DS.densified_wl = DS.densified_wl.loc[(DS.densified_wl['dt'] > t1) & (DS.densified_wl['dt'] < t2)]
            adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_mean_diff(loaded_gauges[DS.neigh_g_up].wl_df)
            DS.juxtapose_gauge_data_to_vs(adjusted_gauge_data)
            DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

            DS.juxtaposed_wl = DS.juxtaposed_wl.loc[DS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]
            if len(DS.juxtaposed_wl) == 0:
                # rmse_sc, rmse_ss, rmse_sn, rmse_sr, nse_sc, nse_ss, nse_sn, nse_sr = [np.nan for x in range(8)]
                # accuracies.append(
                #     [DS.id, DS.chainage, velocity, len(DS.densified_wl), len(DS.upstream_adjacent_vs), mean_bias,
                #      mean_uncrt, np.nan, np.nan, rmse_c, rmse_s, rmse_n, rmse_r, rmse_sc, rmse_ss, rmse_sn, rmse_sr,
                #      nse_c, nse_s, nse_n, nse_r, nse_sc, nse_ss, nse_sn, nse_sr, DS.x, DS.y])
                continue
            for VS in DS.upstream_adjacent_vs:
                VS.juxtaposed_wl = VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]
            DS.densified_wl = DS.densified_wl.loc[DS.densified_wl['mission'].str.contains('SWOT', na=False)]
            DS.get_daily_interpolated_wl_ts(method=itpd_method)  # TUTAJ DODAJE METODE LINIOWA
            DS.get_single_vs_interpolated_ts()
            DS.dist_weighted_daily_wl = DS.get_dist_weighted_wl(DS.densified_wl)
            DS.dist_weighted_daily_wl_itpd = DS.interpolate(DS.dist_weighted_daily_wl)
            DS.get_densified_wl_by_norm(DS.juxtaposed_wl, velocity, norm_method)
            DS.normalized_ts_daily = DS.get_dist_weighted_wl(DS.normalized_ts)
            DS.normalized_ts_itpd = DS.interpolate(DS.normalized_ts_daily)
            DS.get_densified_wl_by_regressions(velocity)
            DS.densified_wl_by_regr_daily = DS.get_dist_weighted_wl(DS.densified_wl_by_regr_ts)
            DS.densified_wl_by_regr_itpd = DS.interpolate(DS.densified_wl_by_regr_daily)
            DS.get_spline_interpolated_ts(DS.dist_weighted_daily_wl, 'wl_weighted', 5)
            rmse_sc, nse_sc = DS.get_rmse_nse_values(DS.interpolated_wl, DS.wl.index.min(), DS.wl.index.max(), 'SWOT CLASSIC METHOD:')
            rmse_ss, nse_ss = DS.get_rmse_nse_values(DS.spline_itpd_wl, DS.wl.index.min(), DS.wl.index.max(), 'SWOT SPLINE AND DIST WEIGHT:')
            rmse_sn, nse_sn = DS.get_rmse_nse_values(DS.normalized_ts_itpd, DS.wl.index.min(), DS.wl.index.max(), 'SWOT NORMALIZED')
            rmse_srg, nse_srg = DS.get_rmse_nse_values(DS.densified_wl_by_regr_itpd, DS.densified_wl_by_regr_itpd.index.min(),DS.densified_wl_by_regr_itpd.index.max(), 'REGRESSIONS')
            rmse_sr, nse_sr = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.wl.index.min(), DS.wl.index.max(), 'SWOT RAW INTERPOLATION')
            swot_mean_bias, swot_mean_uncrt = DS.densified_wl['bias'].mean(), DS.densified_wl['uncertainty'].mean()
            if all_vs:
                accuracies.append(
                    [DS.id, DS.chainage, velocity, len(DS.densified_wl), len(DS.upstream_adjacent_vs), mean_bias,
                     mean_uncrt, swot_mean_bias, swot_mean_uncrt, rmse_c, rmse_s, rmse_n, rmse_r, rmse_sc, rmse_ss,
                     rmse_sn, rmse_sr, nse_c, nse_s, nse_n, nse_r, nse_sc, nse_ss, nse_sn, nse_sr, DS.x, DS.y])
            else:
                accuracies.append(
                    [DS.id, DS.chainage, velocity, len(DS.densified_wl), len(DS.upstream_adjacent_vs), swot_mean_bias,
                     swot_mean_uncrt, rmse_sc, rmse_ss, rmse_sn, rmse_srg, rmse_sr, nse_sc, nse_ss, nse_sn, nse_srg,
                     nse_sr, DS.x, DS.y])
            print(f'Regressions based on {DS.regressions_df["data_len"].min()} to {DS.regressions_df["data_len"].max()} concurrent measurements______________________________')
        if all_vs:
            res_df = pd.DataFrame(accuracies, columns=res_cols_all_vs)
        else:
            res_df = pd.DataFrame(accuracies, columns=res_cols_swot)
        res_df.to_csv(f'{data_dir}accuracies_at_{river_name.split(",")[0]}_{str(t1)[:10]}_to_{str(t2)[:10]}_'
                      f'corr{str(corr_thres).replace(".", "")}_amp{str(amp_thres).replace(".", "")}_minmax.csv', sep=';')
        print(res_df)
        print(1)

        """ CHAINAGE - ACCURACY SCATTERPLOT"""
        # data_col, col_label = 'rmse_sn', 'RMSE SWOT norm [m]'
        # filtered_df = res_df.dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8.5))
        # axes_cols = [(ax1, 'chain'), (ax2, 'data_len'), (ax3, 'swot_mean_bias'), (ax4, 'swot_mean_uncrt')]
        # # for num, col in enumerate(res_df.columns[5:13]):
        # for ax, col in axes_cols:
        #     curr_df = filtered_df[[col, data_col]]
        #     ax.scatter(curr_df[col], curr_df[data_col])
        #     ax.set_ylabel(col_label)
        #     ax.set_xlabel(col)
        #     ax.grid(True, linestyle='--', alpha=0.6)
        #     plt.tight_layout()
        #     fig.suptitle(
        #         f'{len(filtered_df)} VS at the {current_river.name} River')
        # # plt.show()
        # plt.savefig(f'{data_dir}accuracies_scatterplot_at_{river_name.split(",")[0]}.png', dpi=300)
        #
        """ BOXPLOT OF ACCURACIES """
        # metric, metric2 = 'RMSE [m]', 'NSE'
        # cols1 = ['rmse_c', 'rmse_n', 'rmse_r', 'rmse_sc', 'rmse_sn', 'rmse_sr']
        # cols2 = ['nse_c', 'nse_n', 'nse_r', 'nse_sc', 'nse_sn', 'nse_sr']
        # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        # # ax.boxplot(res_df[res_df.columns[[5,7,8,9,11,12]]].dropna(axis=0))  # RMSE
        # selected_data1 = res_df[cols1].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
        # selected_data2 = res_df[cols2].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
        # ax.boxplot(selected_data1)
        # ax.set_xlabel('Methods')
        # ax.set_ylabel(metric)
        # ax.set_xticklabels(
        #     ['classic', 'normalised', 'single VS', 'SWOT\nclassic', 'SWOT\nnormalised', 'SWOT\nsingle VS'])
        #
        # ax2.boxplot(selected_data2)
        # ax2.set_xlabel('Methods')
        # ax2.set_ylabel(metric2)
        # ax2.set_xticklabels(
        #     ['classic', 'normalised', 'single VS', 'SWOT\nclassic', 'SWOT\nnormalised', 'SWOT\nsingle VS'])
        #
        # ax.grid(True, linestyle='--', alpha=0.6)
        # ax2.grid(True, linestyle='--', alpha=0.6)
        # fig.suptitle(
        #     f'Accuracy of different densification methods at {len(selected_data1)} VS at the {riv} River \n corr. thres.: {corr_thres}, velocity: {round(velocity, 2)} m/s, buffer: {buffer} km, up to {gauge_dist_thres} km from gauge, norm: {norm_method}, interpolation: {itpd_method}')
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig(f'{data_dir}accuracies_at_{river_name.split(",")[0]}_v2.png', dpi=300)

        """ BOXPLOT OF ACCURACIES (SWOT ONLY) """
        # metric, metric2 = 'RMSE [m]', 'NSE'
        # cols1 = ['rmse_sc', 'rmse_sn', 'rmse_srg', 'rmse_sr']
        # cols2 = ['nse_sc', 'nse_sn', 'nse_srg', 'nse_sr']
        # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        # # ax.boxplot(res_df[res_df.columns[[5,7,8,9,11,12]]].dropna(axis=0))  # RMSE
        # selected_data1 = res_df[cols1].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
        # selected_data2 = res_df[cols2].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
        # ax.boxplot(selected_data1)
        # ax.set_xlabel('Methods')
        # ax.set_ylabel(metric)
        # cols1_labels = [f'{col}\n{round(res_df[col].mean(), 3)}' for col in cols1]
        # ax.set_xticklabels(cols1_labels)
        #
        # ax2.boxplot(selected_data2)
        # ax2.set_xlabel('Methods')
        # ax2.set_ylabel(metric2)
        # cols2_labels = [f'{col}\n{round(res_df[col].mean(), 3)}' for col in cols2]
        # ax2.set_xticklabels(cols2_labels)
        #
        # ax.grid(True, linestyle='--', alpha=0.6)
        # ax2.grid(True, linestyle='--', alpha=0.6)
        # fig.suptitle(
        #     f'Accuracy of different densification methods at {len(selected_data1)} VS at the {riv} River \n corr. thres.: {corr_thres}, velocity: {round(velocity, 2)} m/s, buffer: {buffer} km, up to {gauge_dist_thres} km from gauge, norm: {norm_method}, interpolation: {itpd_method}')
        # plt.tight_layout()
        # plt.show()
        print(1)

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
    #         plt.show()

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
    #         plt.show()


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
    # plt.show()
