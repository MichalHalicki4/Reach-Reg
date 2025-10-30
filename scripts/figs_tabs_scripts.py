import pandas as pd
import geopandas as gpd
import numpy as np
from insituapi.InSitu import InSitu
from dahitiapi.DAHITI import DAHITI
import River_class as rv
import Station_class as sc
import pickle
import copy
from shapely.geometry.point import Point
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from scipy import stats
import time


data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/'
fig_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/altirunde25/'
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp'  # POLAND
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb23_v17b.shp'  # ELBE, RHINE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb21_v17b.shp'  # ELBE, RHINE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb22_v17b.shp'  # DANUBE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/NA/na_sword_reaches_hb74_v17b.shp'  # USA
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/SA/sa_sword_reaches_hb62_v17b.shp'  # Amazon
riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/AS/as_sword_reaches_hb45_v17b.shp'  # GANGES

t1, t2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2024-12-31 23:59')

# riv, metric_crs = 'Odra', '2180'
# riv, metric_crs = 'Rhine', '4839'
# riv, metric_crs = 'Elbe', '4839'
# riv, metric_crs = 'Po', '3035'
# riv, metric_crs = 'Missouri', 'ESRI:102010'
# riv, metric_crs = 'Mississippi', 'ESRI:102010'
# riv, metric_crs = 'Solimoes', 'ESRI:102033'
riv, metric_crs = 'Ganges', 'ESRI:102025'

river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
save_river_to_file = False
if save_river_to_file:
    current_river = rv.prepare_river_object(riv_path, riv, metric_crs)
    current_river.upload_dam_and_tributary_chains(rv.river_tributary_reaches[riv])
    with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "wb") as f:
        pickle.dump(current_river, f)
else:
    with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
        current_river = pickle.load(f)

# with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
#     loaded_stations = pickle.load(f)
#     loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
# with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
#     loaded_gauges = pickle.load(f)


def plot_accuracies(res_df, riv, cols1=[]):
    if cols1 == []:
        cols1 = res_df.columns[-10:-5]
    metric, metric2 = 'RMSE [m]', 'NSE'
    cols2 = [x.replace('rmse', 'nse') for x in cols1]
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # ax.boxplot(res_df[res_df.columns[[5,7,8,9,11,12]]].dropna(axis=0))  # RMSE
    selected_data1 = res_df[cols1].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
    selected_data2 = res_df[cols2].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
    ax.boxplot(selected_data1)
    ax.set_xlabel('Methods')
    ax.set_ylabel(metric)
    cols1_labels = [f'{col}\nmean: {round(res_df[col].mean(), 3)}' for col in cols1]
    ax.set_xticklabels(cols1_labels)

    ax2.boxplot(selected_data2)
    ax2.set_xlabel('Methods')
    ax2.set_ylabel(metric2)
    cols2_labels = [f'{col}\nmean: {round(res_df[col].mean(), 3)}' for col in cols2]
    ax2.set_xticklabels(cols2_labels)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)
    fig.suptitle(
        f'Accuracy of daily water level time series based on {len(selected_data1)} RS at the {riv} River')
    plt.tight_layout()
    plt.show(block=True)
    # plt.savefig(f'{fig_dir}RS_accuracy_at_{riv}.png', dpi=300)


densify_wl_at_station = False
if densify_wl_at_station:
    # vs_id = 42255
    # vs_id = 42305  # 42224
    # vs_id = 42257  # 13655
    # vs_id = 41905
    # vs_id = 41861
    # vs_id = 41900
    # vs_id = 46217
    vs_id = 41931
    velocity, buffer = 30 / 36, 500
    # corr_thres, amp_thres = 0.8, 1
    corr_thres, amp_thres = 0.001, 20
    rmse_thres, single_rmse_thres = 0.5, 0.2
    itpd_method, norm_method = 'akima', 'standard'
    gauge_dist_thres = 5
    # tributary_chains = current_river.tributary_chains
    tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]

    VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
    DS = sc.DensificationStation(VS, buffer, velocity, itpd_method)
    DS.get_upstream_adjacent_vs(loaded_stations)
    DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                    rv.vs_with_neight_dams[riv], False)
    DS.get_densified_wl()
    DS.densified_wl = DS.densified_wl.loc[(DS.densified_wl['dt'] > t1) & (DS.densified_wl['dt'] < t2)]
    DS.filter_stations_only_with_swot()
    DS.get_daily_interpolated_wl_ts(method=itpd_method)  # TUTAJ DODAJE METODE LINIOWA
    DS.get_single_vs_interpolated_ts()
    DS.dist_weighted_daily_wl = sc.get_rmse_weighted_wl(DS.densified_wl)
    DS.dist_weighted_daily_wl_itpd = DS.interpolate(DS.dist_weighted_daily_wl)
    DS.get_densified_wl_by_norm(DS.juxtaposed_wl, norm_method)
    DS.normalized_ts = sc.filter_outliers_by_tstudent_test(DS.normalized_ts)
    DS.normalized_ts_daily = sc.get_rmse_weighted_wl(DS.normalized_ts)
    DS.normalized_ts_itpd = DS.interpolate(DS.normalized_ts_daily)
    DS.get_densified_wl_by_regressions(rmse_thres=rmse_thres, single_rmse_thres=single_rmse_thres)
    DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
    DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
    DS.densified_itpd = DS.interpolate(DS.densified_daily)
    adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_mean_diff(loaded_gauges[DS.neigh_g_up].wl_df)
    DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, DS.densified_ts)
    DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())
    # DS.get_spline_interpolated_ts(DS.dist_weighted_daily_wl, 'wl_weighted', 5)
    rmse_sc, nse_sc = DS.get_rmse_nse_values(DS.interpolated_wl, DS.wl.index.min(), DS.wl.index.max(),
                                             'SWOT CLASSIC METHOD:')
    rmse_sn, nse_sn = DS.get_rmse_nse_values(DS.normalized_ts_itpd, DS.wl.index.min(), DS.wl.index.max(),
                                             'SWOT NORMALIZED')
    rmse_srg, nse_srg = DS.get_rmse_nse_values(DS.densified_itpd, DS.densified_itpd.index.min(),
                                               DS.densified_itpd.index.max(), 'REGRESSIONS')
    rmse_sr, nse_sr = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.wl.index.min(), DS.wl.index.max(),
                                             'SWOT RAW INTERPOLATION')
    swot_mean_bias, swot_mean_uncrt = DS.densified_wl['bias'].mean(), DS.densified_wl['uncertainty'].mean()
    DS.plot_vs_setting_with_regressions_rmse(current_river)


def analyse_buffers_with_cross_validation(vs_id, river_name, res_li):
    velocity, buffer, corr_thres, amp_thres, rmse_thres, single_rmse_thres, itpd_method = rv.configs[riv].values()
    if riv == 'Odra':
        tributary_chains = current_river.tributary_chains
    else:
        tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    gauge_dist_thres = 5
    buffers = [x*25 for x in range(4, 21)]
    for buff in buffers:
        VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
        vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
        if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
            continue
        if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
            continue
        DS = sc.DensificationStation(VS,  buff, velocity, itpd_method)
        DS.get_upstream_adjacent_vs(loaded_stations)
        DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                        rv.vs_with_neight_dams[riv], False)
        DS.filter_stations_only_with_swot()
        if len(DS.upstream_adjacent_vs) == 0:
            continue

        DS.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
        DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
        densified_ts_cval = DS.densified_ts.loc[DS.densified_ts['id_vs'] != DS.id]

        DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
        DS.densified_itpd = DS.interpolate(DS.densified_daily)

        densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
        densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample('D').mean().dropna()

        adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[DS.neigh_g_up].wl_df)
        DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, DS.densified_ts)
        DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

        rms_thr = (DS.wl['wse'].max() - DS.wl['wse'].min()) / 10
        mean_bias = DS.densified_ts['bias'].mean()
        mean_uncrt = DS.densified_ts['uncertainty'].mean()
        max_rmse_sum = DS.densified_ts['rmse_sum'].max()

        stations_num = len(DS.densified_ts["id_vs"].unique())
        if len(res_li) > 2:
            if stations_num == res_li[-1][-3] and stations_num == res_li[-2][-3]:
                break

        rmse, nse = DS.get_rmse_nse_values(DS.densified_itpd, DS.densified_itpd.index.min(),
                                                   DS.densified_itpd.index.max(), f'BUFFER {buff}')
        rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd, densified_ts_cval_itpd.index.min(),
                                                   densified_ts_cval_itpd.index.max(), f'CVAL, BUFFER {buff}', df_true)
        print(f'Max rmse_sum: {round(max_rmse_sum, 2)}, amp-based thres: {round(rms_thr, 2)}, stations: {stations_num}')
        res_li.append([river_name.split(",")[0], vs_id, buff, mean_bias, mean_uncrt, rmse_thres, rms_thr, max_rmse_sum, stations_num, rmse, rmse_cval])
        print('---------------------')
    return res_li


# results_list = []
# for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Odra', '2180')]:
#     river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
#     with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
#         current_river = pickle.load(f)
#     with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
#         loaded_stations = pickle.load(f)
#         loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
#     with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
#         loaded_gauges = pickle.load(f)
#     for vs_id in [x.id for x in loaded_stations]:
#         results_list = analyse_buffers_with_cross_validation(vs_id, river_name, results_list)
#
# res_df = pd.DataFrame(results_list, columns=['river', 'vs_id', 'buffer', 'mean_bias', 'mean_uncrt', 'rmse_thres',
#                                              'amp_rms_thr', 'max_rmse_sum', 'stations_num', 'rmse', 'rmse_cval'])
# res_df.to_csv('buffers_analysis.csv', sep=';', decimal='.')
# print(results_list)
# print(1)

    # fig, ax = plt.subplots()
    # ax.plot(buffers, rmses)
    # plt.show(block=True)
    # print(1)


def analyse_accuracies(vs_stations, res_li):
    velocity, buffer, corr_thres, amp_thres, rmse_thres, single_rmse_thres, itpd_method = rv.configs[riv].values()
    if riv == 'Odra':
        tributary_chains = current_river.tributary_chains
    else:
        tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    gauge_dist_thres = 5
    for vs_id in [x.id for x in vs_stations]:
        for velocity in [x / 3.6 for x in range(1, 8)]:
            # vs_id = 41903  # 46217 23410 18853
            VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
            vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
            if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
                continue
            if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
                continue
            buffer, corr_thres = 300, 0.75
            DS = sc.DensificationStation(VS,  buffer, velocity, itpd_method)
            DS.get_upstream_adjacent_vs(loaded_stations)
            df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
                'D').mean().dropna()

            DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                            rv.vs_with_neight_dams[riv], False)
            DS.filter_stations_only_with_swot()
            if len(DS.upstream_adjacent_vs) == 0:
                continue

            DS.get_single_vs_interpolated_ts()
            DS.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
            cval_rmse = DS.get_rmse_of_cval_ts(DS.densified_ts, df_true)
            amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
            wl_amplitude = DS.densified_ts['shifted_wl'].max() - DS.densified_ts[
                'shifted_wl'].min()
            rms_thr = wl_amplitude * amp_thres_final
            DS.densified_ts = DS.densified_ts.loc[DS.densified_ts['rmse_sum'] < rms_thr]

            DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
            DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
            # DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
            DS.densified_itpd = DS.interpolate(DS.densified_daily)
            df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample('D').mean().dropna()

            densified_ts_cval = DS.densified_ts.loc[DS.densified_ts['id_vs'] != DS.id]
            # densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
            densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
            densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)

            adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[DS.neigh_g_up].wl_df)
            DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, DS.densified_ts)
            DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

            rmse_reg, nse_reg = DS.get_rmse_nse_values(DS.densified_itpd,
                                                       DS.densified_itpd.index.min(),
                                                       DS.densified_itpd.index.max(), 'REGRESSIONS')
            rmse_raw, nse_raw = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(),
                                                       DS.single_VS_itpd.index.max(), 'SINGLE VS')
            rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd,
                                                         densified_ts_cval_itpd.index.min(),
                                                         densified_ts_cval_itpd.index.max(), 'CrossVal', df_true)
            rmse_daily, nse_daily = DS.get_rmse_nse_values(densified_ts_cval_daily,
                                                           densified_ts_cval_daily.index.min(),
                                                           densified_ts_cval_daily.index.max(), 'Daily')
            rmse_srd, nse_srd = DS.get_rmse_nse_values(DS.wl['wse'].resample('D').mean(), DS.wl.index.min(),
                                                       DS.wl.index.max(), 'VS ACCURACY')
            mean_bias = DS.densified_ts['bias'].mean()
            mean_uncrt = DS.densified_ts['uncertainty'].mean()
            mean_rmse_sum = DS.densified_ts['rmse_sum'].mean()
            # DS.plot_vs_setting_with_regressions_rmse(current_river)
            res_li.append(
                [vs_id, DS.chainage, velocity, len(DS.densified_itpd), len(DS.upstream_adjacent_vs),
                 mean_bias, mean_uncrt, mean_rmse_sum, rmse_reg, rmse_raw, rmse_cval, rmse_daily, rmse_srd,
                 nse_reg, nse_raw, nse_cval, nse_daily, nse_srd])

    res_cols = ['id', 'DS.chainage', 'velocity', 'len(DS.densified_itpd)',
                'len(DS.upstream_adjacent_vs)', 'mean_bias', 'mean_uncrt', 'mean_rmse_sum', 'rmse_reg', 'rmse_raw',
                'rmse_cval', 'rmse_daily', 'rmse_srd', 'nse_reg', 'nse_raw', 'nse_cval', 'nse_daily', 'nse_srd']
    res_df = pd.DataFrame(res_li, columns=res_cols)
    print('---------------------')
    return res_df


do_analyse_accuracies = False
if do_analyse_accuracies:
    res_li = []
    for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Odra', '2180')]:
        river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
        with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
            current_river = pickle.load(f)
        with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_stations = pickle.load(f)
            loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
        with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_gauges = pickle.load(f)

        res_df = analyse_accuracies(loaded_stations, res_li)
    print(1)


def calibrate_velocity_by_crosscal(vs_stations, res_li):
    velocity, buffer, corr_thres, amp_thres, rmse_thres, single_rmse_thres, itpd_method = rv.configs[riv].values()
    if riv == 'Odra':
        tributary_chains = current_river.tributary_chains
    else:
        tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    gauge_dist_thres = 5
    for vs_id in [x.id for x in vs_stations]:
        VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
        vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
        if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
            continue
        if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
            continue
        buffer, corr_thres = 300, 0.75
        DS = sc.DensificationStation(VS, buffer, velocity, itpd_method)
        DS.get_upstream_adjacent_vs(loaded_stations)
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
            'D').mean().dropna()

        DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                        rv.vs_with_neight_dams[riv], False)
        DS.filter_stations_only_with_swot()
        if len(DS.upstream_adjacent_vs) == 0:
            continue

        DS.get_single_vs_interpolated_ts()
        DS.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
        for velocity in [x/25 for x in range(5, 50)]:
            DS.speed_ms = velocity
            DS.densified_ts = DS.calculate_shifted_time(DS.densified_ts)
            cval_rmse = DS.get_rmse_of_cval_ts(DS.densified_ts, df_true)
            amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
            wl_amplitude = DS.densified_ts['shifted_wl'].max() - DS.densified_ts[
                'shifted_wl'].min()
            rms_thr = wl_amplitude * amp_thres_final
            DS.densified_ts = DS.densified_ts.loc[
                DS.densified_ts['rmse_sum'] < rms_thr]
            DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
            DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
            DS.densified_itpd = DS.interpolate(DS.densified_daily)
            df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
                'D').mean().dropna()

            densified_ts_cval = DS.densified_ts.loc[DS.densified_ts['id_vs'] != DS.id]
            densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
            densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)

            adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[DS.neigh_g_up].wl_df)
            DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data,
                                                                            DS.densified_ts)
            DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

            rmse_reg, nse_reg = DS.get_rmse_nse_values(DS.densified_itpd,
                                                       DS.densified_itpd.index.min(),
                                                       DS.densified_itpd.index.max(), 'REGRESSIONS')
            rmse_raw, nse_raw = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(),
                                                       DS.single_VS_itpd.index.max(), 'SINGLE VS')
            rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd,
                                                         densified_ts_cval_itpd.index.min(),
                                                         densified_ts_cval_itpd.index.max(), 'CrossVal',
                                                         df_true)
            rmse_daily, nse_daily = DS.get_rmse_nse_values(densified_ts_cval_daily,
                                                           densified_ts_cval_daily.index.min(),
                                                           densified_ts_cval_daily.index.max(), 'Daily')
            rmse_srd, nse_srd = DS.get_rmse_nse_values(DS.wl['wse'].resample('D').mean(), DS.wl.index.min(),
                                                       DS.wl.index.max(), 'VS ACCURACY')
            mean_bias = DS.densified_ts['bias'].mean()
            mean_uncrt = DS.densified_ts['uncertainty'].mean()
            mean_rmse_sum = DS.densified_ts['rmse_sum'].mean()
            # DS.plot_vs_setting_with_regressions_rmse(current_river)
            res_li.append(
                [vs_id, DS.chainage, velocity, len(DS.densified_itpd), len(DS.upstream_adjacent_vs),
                 mean_bias, mean_uncrt, mean_rmse_sum, rmse_reg, rmse_raw, rmse_cval, rmse_daily, rmse_srd,
                 nse_reg, nse_raw, nse_cval, nse_daily, nse_srd])

    res_cols = ['id', 'DS.chainage', 'velocity', 'len(DS.densified_itpd)',
                'len(DS.upstream_adjacent_vs)', 'mean_bias', 'mean_uncrt', 'mean_rmse_sum', 'rmse_reg', 'rmse_raw',
                'rmse_cval', 'rmse_daily', 'rmse_srd', 'nse_reg', 'nse_raw', 'nse_cval', 'nse_daily', 'nse_srd']
    res_df = pd.DataFrame(res_li, columns=res_cols)
    print('---------------------')
    return res_df


do_calibrate_by_crossval = False
if do_calibrate_by_crossval:
    res_li = []
    for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Odra', '2180')]:
        river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
        with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
            current_river = pickle.load(f)
        with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_stations = pickle.load(f)
            loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
        with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_gauges = pickle.load(f)

        res_df = calibrate_velocity_by_crosscal(loaded_stations, res_li)
    print(1)


def densify_wl_with_vel_curves(vs_stations, res_li, df_vels):
    velocity, buffer, corr_thres, amp_thres, rmse_thres, single_rmse_thres, itpd_method = rv.configs[riv].values()
    if riv == 'Odra':
        tributary_chains = current_river.tributary_chains
    else:
        tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    gauge_dist_thres = 5
    for vs_id in [x.id for x in vs_stations]:
        VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
        vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
        if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
            continue
        if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
            continue
        buffer, corr_thres = 300, 0.75
        DS = sc.DensificationStation(VS, buffer, None, itpd_method)
        DS.get_upstream_adjacent_vs(loaded_stations)
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
            'D').mean().dropna()

        DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                        rv.vs_with_neight_dams[riv], False)
        DS.filter_stations_only_with_swot()
        DS.get_slope_of_all_vs()
        if len(DS.upstream_adjacent_vs) == 0:
            continue

        DS.get_single_vs_interpolated_ts()
        DS.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
        DS.densified_ts = DS.calculate_shifted_time_by_curve(DS.densified_ts, df_vels)
        cval_rmse = DS.get_rmse_of_cval_ts(DS.densified_ts, df_true)
        amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
        wl_amplitude = DS.densified_ts['shifted_wl'].max() - DS.densified_ts[
            'shifted_wl'].min()
        rms_thr = wl_amplitude * amp_thres_final
        DS.densified_ts = DS.densified_ts.loc[
            DS.densified_ts['rmse_sum'] < rms_thr]
        DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
        DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
        DS.densified_itpd = DS.interpolate(DS.densified_daily)
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
            'D').mean().dropna()

        densified_ts_cval = DS.densified_ts.loc[
            DS.densified_ts['id_vs'] != DS.id]
        densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
        densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)

        adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[DS.neigh_g_up].wl_df, df_vels)
        DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data,
                                                                        DS.densified_ts)
        DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

        rmse_reg, nse_reg = DS.get_rmse_nse_values(DS.densified_itpd,
                                                   DS.densified_itpd.index.min(),
                                                   DS.densified_itpd.index.max(), 'REGRESSIONS')
        rmse_raw, nse_raw = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(),
                                                   DS.single_VS_itpd.index.max(), 'SINGLE VS')
        rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd,
                                                     densified_ts_cval_itpd.index.min(),
                                                     densified_ts_cval_itpd.index.max(), 'CrossVal',
                                                     df_true)
        rmse_daily, nse_daily = DS.get_rmse_nse_values(densified_ts_cval_daily,
                                                       densified_ts_cval_daily.index.min(),
                                                       densified_ts_cval_daily.index.max(), 'Daily')
        rmse_srd, nse_srd = DS.get_rmse_nse_values(DS.wl['wse'].resample('D').mean(), DS.wl.index.min(),
                                                   DS.wl.index.max(), 'VS ACCURACY')
        mean_bias = DS.densified_ts['bias'].mean()
        mean_uncrt = DS.densified_ts['uncertainty'].mean()
        mean_rmse_sum = DS.densified_ts['rmse_sum'].mean()
        # DS.plot_vs_setting_with_regressions_rmse(current_river)
        res_li.append(
            [vs_id, DS.chainage, riv, velocity, len(DS.densified_itpd), len(DS.upstream_adjacent_vs),
             mean_bias, mean_uncrt, mean_rmse_sum, rmse_reg, rmse_raw, rmse_cval, rmse_daily, rmse_srd,
             nse_reg, nse_raw, nse_cval, nse_daily, nse_srd])

    res_cols = ['id', 'DS.chainage', 'river', 'velocity', 'len(DS.densified_itpd)',
                'len(DS.upstream_adjacent_vs)', 'mean_bias', 'mean_uncrt', 'mean_rmse_sum', 'rmse_reg', 'rmse_raw',
                'rmse_cval', 'rmse_daily', 'rmse_srd', 'nse_reg', 'nse_raw', 'nse_cval', 'nse_daily', 'nse_srd']
    res_df = pd.DataFrame(res_li, columns=res_cols)
    print('---------------------')
    return res_df


do_densify_with_vel_curves = False
if do_densify_with_vel_curves:
    res_li = []
    df_v_po = pd.read_csv(
        '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/velocity_rating_curve_Po_11.6419_44.8956.csv')
    df_v_oder = pd.read_csv(
        '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/velocity_rating_curve_Oder_14.424585_52.694887.csv')
    for riv, metric_crs, df_v in [('Odra', '2180', df_v_oder), ('Po', '3035', df_v_po)]:
        river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
        with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
            current_river = pickle.load(f)
        with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_stations = pickle.load(f)
            loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
        with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_gauges = pickle.load(f)

        res_df = densify_wl_with_vel_curves(loaded_stations, res_li, df_v)
    print(1)


def densify_with_calib_vel_analysis(vs_stations, res_li):
    velocity, buffer, corr_thres, amp_thres, rmse_thres, single_rmse_thres, itpd_method = rv.configs[riv].values()
    if riv == 'Odra':
        tributary_chains = current_river.tributary_chains
    else:
        tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    gauge_dist_thres = 5
    for vs_id in [x.id for x in vs_stations]:
        VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
        vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
        if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
            continue
        if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
            continue
        buffer, corr_thres = 300, 0.75
        DS = sc.DensificationStation(VS, buffer, None, itpd_method)
        DS.get_upstream_adjacent_vs(loaded_stations)
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
            'D').mean().dropna()

        DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                        rv.vs_with_neight_dams[riv], False)
        DS.filter_stations_only_with_swot()
        DS.get_slope_of_all_vs()
        if len(DS.upstream_adjacent_vs) == 0:
            continue

        DS.get_single_vs_interpolated_ts()
        DS.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
        for c in [x/500 for x in range(10, 51)]:
            for bottom in [x/10 for x in range(1, 11)]:
                DS.densified_ts = DS.calculate_shifted_time_by_simplified_mannig(DS.densified_ts, c, bottom)
                cval_rmse = DS.get_rmse_of_cval_ts(DS.densified_ts, df_true)
                amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
                wl_amplitude = DS.densified_ts['shifted_wl'].max() - DS.densified_ts[
                    'shifted_wl'].min()
                rms_thr = wl_amplitude * amp_thres_final
                DS.densified_ts = DS.densified_ts.loc[
                    DS.densified_ts['rmse_sum'] < rms_thr]
                DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
                DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
                DS.densified_itpd = DS.interpolate(DS.densified_daily)
                df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
                    'D').mean().dropna()

                densified_ts_cval = DS.densified_ts.loc[
                    DS.densified_ts['id_vs'] != DS.id]
                densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
                densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)

                adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[DS.neigh_g_up].wl_df)
                DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data,
                                                                                DS.densified_ts)
                DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

                rmse_reg, nse_reg = DS.get_rmse_nse_values(DS.densified_itpd,
                                                           DS.densified_itpd.index.min(),
                                                           DS.densified_itpd.index.max(), 'REGRESSIONS')
                rmse_raw, nse_raw = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(),
                                                           DS.single_VS_itpd.index.max(), 'SINGLE VS')
                rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd,
                                                             densified_ts_cval_itpd.index.min(),
                                                             densified_ts_cval_itpd.index.max(), 'CrossVal',
                                                             df_true)
                rmse_daily, nse_daily = DS.get_rmse_nse_values(densified_ts_cval_daily,
                                                               densified_ts_cval_daily.index.min(),
                                                               densified_ts_cval_daily.index.max(), 'Daily')
                rmse_srd, nse_srd = DS.get_rmse_nse_values(DS.wl['wse'].resample('D').mean(), DS.wl.index.min(),
                                                           DS.wl.index.max(), 'VS ACCURACY')
                mean_bias = DS.densified_ts['bias'].mean()
                mean_uncrt = DS.densified_ts['uncertainty'].mean()
                mean_rmse_sum = DS.densified_ts['rmse_sum'].mean()
                # DS.plot_vs_setting_with_regressions_rmse(current_river)
                res_li.append(
                    [vs_id, DS.chainage, riv, DS.speed_ms, c, bottom, wl_amplitude, len(DS.densified_itpd), len(DS.upstream_adjacent_vs),
                     mean_bias, mean_uncrt, mean_rmse_sum, rmse_reg, rmse_raw, rmse_cval, rmse_daily, rmse_srd,
                     nse_reg, nse_raw, nse_cval, nse_daily, nse_srd])

    res_cols = ['id', 'DS.chainage', 'river', 'velocity', 'c', 'bottom', 'wl_amp', 'len(DS.densified_itpd)',
                'len(DS.upstream_adjacent_vs)', 'mean_bias', 'mean_uncrt', 'mean_rmse_sum', 'rmse_reg', 'rmse_raw',
                'rmse_cval', 'rmse_daily', 'rmse_srd', 'nse_reg', 'nse_raw', 'nse_cval', 'nse_daily', 'nse_srd']
    res_df = pd.DataFrame(res_li, columns=res_cols)
    print('---------------------')
    return res_df


do_densify_with_calib_vel_analysis = False
if do_densify_with_calib_vel_analysis:
    res_li = []
    # df_v_po = pd.read_csv(
    #     '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/velocity_rating_curve_Po_11.6419_44.8956.csv')
    # df_v_oder = pd.read_csv(
    #     '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/velocity_rating_curve_Oder_14.424585_52.694887.csv')
    for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Odra', '2180')]:
        river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
        with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
            current_river = pickle.load(f)
        with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_stations = pickle.load(f)
            loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
        with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_gauges = pickle.load(f)

        res_df = densify_with_calib_vel_analysis(loaded_stations, res_li)
    print(1)


def densify_with_calib_vel(vs_stations, res_li):
    velocity, buffer, corr_thres, amp_thres, rmse_thres, single_rmse_thres, itpd_method = rv.configs[riv].values()
    if riv == 'Odra':
        tributary_chains = current_river.tributary_chains
    else:
        tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    gauge_dist_thres = 5
    for vs_id in [x.id for x in vs_stations]:
        VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
        vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
        if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
            continue
        if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
            continue
        buffer, corr_thres = 300, 0.75
        DS = sc.DensificationStation(VS, buffer, None, itpd_method)
        DS.get_upstream_adjacent_vs(loaded_stations)
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
            'D').mean().dropna()

        DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                        rv.vs_with_neight_dams[riv], False)
        DS.filter_stations_only_with_swot()
        DS.get_slope_of_all_vs()
        if len(DS.upstream_adjacent_vs) == 0:
            continue

        DS.get_single_vs_interpolated_ts()
        DS.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
        c, vel_uncrt = DS.calibrate_mannings_c()
        DS.densified_ts = DS.calculate_shifted_time_by_simplified_mannig(DS.densified_ts, c)
        cval_rmse = DS.get_rmse_of_cval_ts(DS.densified_ts, df_true)
        amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
        wl_amplitude = DS.densified_ts['shifted_wl'].max() - DS.densified_ts[
            'shifted_wl'].min()
        rms_thr = wl_amplitude * amp_thres_final
        DS.densified_ts = DS.densified_ts.loc[
            DS.densified_ts['rmse_sum'] < rms_thr]
        DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
        DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
        DS.densified_itpd = DS.interpolate(DS.densified_daily)
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
            'D').mean().dropna()

        densified_ts_cval = DS.densified_ts.loc[
            DS.densified_ts['id_vs'] != DS.id]
        densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
        densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)

        adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[DS.neigh_g_up].wl_df)
        DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data,
                                                                        DS.densified_ts)
        DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

        rmse_reg, nse_reg = DS.get_rmse_nse_values(DS.densified_itpd,
                                                   DS.densified_itpd.index.min(),
                                                   DS.densified_itpd.index.max(), 'REGRESSIONS')
        rmse_raw, nse_raw = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(),
                                                   DS.single_VS_itpd.index.max(), 'SINGLE VS')
        rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd,
                                                     densified_ts_cval_itpd.index.min(),
                                                     densified_ts_cval_itpd.index.max(), 'CrossVal',
                                                     df_true)
        rmse_daily, nse_daily = DS.get_rmse_nse_values(densified_ts_cval_daily,
                                                       densified_ts_cval_daily.index.min(),
                                                       densified_ts_cval_daily.index.max(), 'Daily')
        rmse_srd, nse_srd = DS.get_rmse_nse_values(DS.wl['wse'].resample('D').mean(), DS.wl.index.min(),
                                                   DS.wl.index.max(), 'VS ACCURACY')
        mean_bias = DS.densified_ts['bias'].mean()
        mean_uncrt = DS.densified_ts['uncertainty'].mean()
        mean_rmse_sum = DS.densified_ts['rmse_sum'].mean()
        # DS.plot_vs_setting_with_regressions_rmse(current_river)
        res_li.append(
            [vs_id, DS.chainage, riv, DS.speed_ms, c, len(DS.densified_itpd), len(DS.upstream_adjacent_vs),
             mean_bias, mean_uncrt, mean_rmse_sum, rmse_reg, rmse_raw, rmse_cval, rmse_daily, rmse_srd,
             nse_reg, nse_raw, nse_cval, nse_daily, nse_srd])

    res_cols = ['id', 'DS.chainage', 'river', 'velocity', 'c', 'len(DS.densified_itpd)',
                'len(DS.upstream_adjacent_vs)', 'mean_bias', 'mean_uncrt', 'mean_rmse_sum', 'rmse_reg', 'rmse_raw',
                'rmse_cval', 'rmse_daily', 'rmse_srd', 'nse_reg', 'nse_raw', 'nse_cval', 'nse_daily', 'nse_srd']
    res_df = pd.DataFrame(res_li, columns=res_cols)
    print('---------------------')
    return res_df


do_densify_with_calib_vel = False
if do_densify_with_calib_vel:
    res_li = []
    df_v_po = pd.read_csv(
        '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/velocity_rating_curve_Po_11.6419_44.8956.csv')
    df_v_oder = pd.read_csv(
        '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/velocity_rating_curve_Oder_14.424585_52.694887.csv')
    for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Odra', '2180')]:
        river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
        with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
            current_river = pickle.load(f)
        with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_stations = pickle.load(f)
            loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
        with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_gauges = pickle.load(f)

        res_df = densify_with_calib_vel(loaded_stations, res_li)
    print(1)



def create_rs_pickle_files(vs_id, riv):
    velocity, buffer, corr_thres, amp_thres, rmse_thres, single_rmse_thres, itpd_method = rv.configs[riv].values()
    # corr_thres, amp_thres = 0.5, 5
    corr_thres, amp_thres = 0.001, 10
    if riv == 'Odra':
        tributary_chains = current_river.tributary_chains
    else:
        tributary_chains = []
    neigh_dam_vs = rv.vs_with_neight_dams[riv]
    gauge_dist_thres = 5
    buffer, rmse_thres = 300, 10

    VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
    vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
    if len(VS.juxtaposed_wl) == 0 or vs_gauge_dist > gauge_dist_thres or VS.id in neigh_dam_vs:
        return None
    if len(VS.juxtaposed_wl.loc[VS.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
        return None
    DS = sc.DensificationStation(VS, buffer, velocity, itpd_method)
    DS.get_upstream_adjacent_vs(loaded_stations)
    DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                    rv.vs_with_neight_dams[riv], False)
    DS.filter_stations_only_with_swot()
    if len(DS.upstream_adjacent_vs) == 0:
        return None
    DS.get_single_vs_interpolated_ts()
    DS.get_densified_wl_by_regressions(rmse_thres=rmse_thres, single_rmse_thres=single_rmse_thres)
    df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['dt'])).resample('D').mean().dropna()

    DS.densified_ts = sc.filter_outliers_by_tstudent_test(DS.densified_ts)
    DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
    # DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
    DS.densified_itpd = DS.interpolate(DS.densified_daily)

    densified_ts_cval = DS.densified_ts.loc[DS.densified_ts['id_vs'] != DS.id]
    # densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
    densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
    densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)

    adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[DS.neigh_g_up].wl_df)
    DS.densified_ts = sc.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, DS.densified_ts)
    DS.get_closest_in_situ_daily_wl(adjusted_gauge_data, DS.wl.index.min(), DS.wl.index.max())

    res_path = f'{data_dir}/rs_stations_v02/{riv}_RS{DS.id}.pkl'
    with open(res_path, "wb") as f:
        pickle.dump(DS, f)
    print(res_path)


save_rs_to_pickle = False
if save_rs_to_pickle:
    for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Odra', '2180')]:
        river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
        with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
            current_river = pickle.load(f)
        with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_stations = pickle.load(f)
            loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
        with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
            loaded_gauges = pickle.load(f)
        for vs_id in [x.id for x in loaded_stations]:
            create_rs_pickle_files(vs_id, riv)
    print(1)


def analyse_accuracies_from_ready_files(vs_stations, riv, res_li, ampltd_thres, unwanted_stations):
    for vs_id in [x.id for x in vs_stations]:
        if vs_id in unwanted_stations:
            continue
        try:
            with open(f'{data_dir}rs_stations_v02/{riv}_RS{vs_id}.pkl', "rb") as f:
                DS = pickle.load(f)
        except FileNotFoundError:
            continue

        # ampltd_thres = 0.1 if riv == 'Po' else 0.05
        df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample('D').mean().dropna()
        cval_rmse = DS.get_rmse_of_cval_ts(DS.densified_ts, df_true)
        amp_thres_final = ampltd_thres if cval_rmse < 0.1 else ampltd_thres * 2
        wl_amplitude = DS.densified_ts['shifted_wl'].max() - DS.densified_ts['shifted_wl'].min()
        rms_thr = wl_amplitude * amp_thres_final

        # rms_thr = 1.5 if riv == 'Po' else 0.5
        DS.densified_ts = DS.densified_ts.loc[DS.densified_ts['rmse_sum'] < rms_thr]

        # corr_thres = 0.7 if riv == 'Po' else 0.8
        corr_thres = 0.75
        DS.filter_upstream_stations_by_correlation(corr_thres)

        # buffer = 150 if riv == 'Po' else 300
        # DS.upstream_adjacent_vs = [x for x in DS.upstream_adjacent_vs if abs(x.chainage) - DS.chainage < buffer * 1000]
        stations = [DS] + DS.upstream_adjacent_vs
        DS.densified_ts = DS.densified_ts.loc[DS.densified_ts['id_vs'].isin([x.id for x in stations])]

        DS.densified_daily = sc.get_rmse_weighted_wl(DS.densified_ts)
        DS.densified_itpd = DS.interpolate(DS.densified_daily)

        densified_ts_cval = DS.densified_ts.loc[DS.densified_ts['id_vs'] != DS.id]
        # densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
        densified_ts_cval_daily = sc.get_rmse_weighted_wl(densified_ts_cval)
        densified_ts_cval_itpd = DS.interpolate(densified_ts_cval_daily)
        print_res = False
        rmse_reg, nse_reg = DS.get_rmse_nse_values(DS.densified_itpd,
                                                   DS.densified_itpd.index.min(),
                                                   DS.densified_itpd.index.max(), 'REGRESSIONS', print_res=print_res)
        rmse_raw, nse_raw = DS.get_rmse_nse_values(DS.single_VS_itpd, DS.single_VS_itpd.index.min(),
                                                   DS.single_VS_itpd.index.max(), 'SINGLE VS', print_res=print_res)
        rmse_cval, nse_cval = DS.get_rmse_nse_values(densified_ts_cval_itpd,
                                                     densified_ts_cval_itpd.index.min(),
                                                     densified_ts_cval_itpd.index.max(), 'CrossVal', df_true, print_res=print_res)
        rmse_daily, nse_daily = DS.get_rmse_nse_values(densified_ts_cval_daily,
                                                       densified_ts_cval_daily.index.min(),
                                                       densified_ts_cval_daily.index.max(), 'Daily', print_res=print_res)
        rmse_srd, nse_srd = DS.get_rmse_nse_values(DS.wl['wse'].resample('D').mean(), DS.wl.index.min(),
                                                   DS.wl.index.max(), 'VS ACCURACY', print_res=print_res)
        print(f'{riv}, {DS.id}, cval: {round(cval_rmse, 3)}, amp_thres: {round(amp_thres_final, 3)}, RMSE_THRES: {round(rms_thr, 3)}, RMSE: {rmse_reg}, NRMSE: {rmse_reg/wl_amplitude}')
        mean_bias = DS.densified_ts['bias'].mean()
        mean_uncrt = DS.densified_ts['uncertainty'].mean()
        mean_rmse_sum = DS.densified_ts['rmse_sum'].mean()
        # DS.plot_vs_setting_with_regressions_rmse(current_river)
        res_li.append(
            [riv, vs_id, DS.chainage, DS.speed_ms, len(DS.densified_itpd), len(DS.upstream_adjacent_vs),
             mean_bias, mean_uncrt, mean_rmse_sum, rmse_reg, rmse_raw, rmse_cval, rmse_daily, rmse_srd,
             nse_reg, nse_raw, nse_cval, nse_daily, nse_srd])
        # print('---------------------')
    return res_li


# for thres in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
#     thres = 0.10
#     unwanted_stations_list = [14039, 18853, 23410]
#     results_list = []
#     for riv, metric_crs in [('Po', '3035'), ('Rhine', '4839'), ('Elbe', '4839'), ('Odra', '2180')]:
#         river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
#         with open(f'{data_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
#             current_river = pickle.load(f)
#         with open(f'{data_dir}vs_at_{river_name.split(",")[0]}.pkl', "rb") as f:
#             loaded_stations = pickle.load(f)
#             loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
#         with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
#             loaded_gauges = pickle.load(f)
#         results_list = analyse_accuracies_from_ready_files(loaded_stations, riv, results_list, thres, unwanted_stations_list)
#
#     res_cols = ['riv', 'id', 'DS.chainage', 'velocity', 'len(DS.densified_itpd)',
#                 'len(DS.upstream_adjacent_vs)', 'mean_bias', 'mean_uncrt', 'mean_rmse_sum', 'rmse_reg', 'rmse_raw',
#                 'rmse_cval', 'rmse_daily', 'rmse_srd', 'nse_reg', 'nse_raw', 'nse_cval', 'nse_daily', 'nse_srd']
#     res_df = pd.DataFrame(results_list, columns=res_cols)
#     print(thres, round(res_df['rmse_reg'].mean(), 3), round(res_df['rmse_cval'].mean(), 3))
#     for riv in ['Po', 'Rhine', 'Elbe', 'Odra']:
#         curr_res = res_df.loc[res_df['riv'] == riv]
#         print(riv, round(curr_res['rmse_reg'].mean(), 3),  round(curr_res['nse_reg'].mean(), 3), round(curr_res['rmse_cval'].mean(), 3))
#     print('-----------------')
# print(res_df)


def plot_swot_obs_against_gauge_data(ds):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.errorbar(x=ds.wl.index, y=ds.wl['wse'], yerr=ds.wl['wse_u'] / 2, fmt='o', capsize=3,
                label='SWOT observations with uncertainty', color='red')
    ax.plot(ds.closest_in_situ_daily_wl, label='Gauge water level', color='blue')
    ax.legend(loc='upper left')
    ax.set_title(f'Water levels at the {riv} River')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylabel('Water level [m]')
    ax.set_xlabel('Time')
    plt.tight_layout()
    # plt.show(block=True)
    plt.savefig(f'{fig_dir}swot_wl_at_{river_name.split(",")[0]}.png', dpi=300)


def plot_map_with_vs_setting_and_vs_water_levels(DS):
    swot_tiles_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/SWORD_v17b_shp/SWOT_tiles/swot_passes_oder.shp'
    swot_gdf = gpd.read_file(swot_tiles_file)
    x_list, y_list = map(list, zip(*[(a.x, a.y) for a in DS.upstream_adjacent_vs]))
    plot_buffer_up, plot_buffer_down, plot_buffer_left, plot_buffer_right = 0.1, 0.5, 0.3, 0.3
    x_max, x_min, y_max, y_min = max(x_list) + plot_buffer_right, min(x_list) - plot_buffer_left, max(
        y_list) + plot_buffer_up, min(
        y_list) - plot_buffer_down
    # points = gpd.GeoSeries(
    #     [Point(x_min, y_min), Point(x_max, y_max)], crs=4326
    # )
    # points = points.to_crs(2180)
    # distance_meters = points[0].distance(points[1])
    # print(distance_meters / 1000)

    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 2])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = plt.cm.get_cmap('viridis')
    num_vs = len(DS.upstream_adjacent_vs)
    colors = cmap(np.linspace(0, 1, num_vs))
    # Rysowanie elementów na mapie
    swot_gdf.plot(
        ax=ax,
        color='gray',  # Ustawia kolor wypełnienia na szary
        alpha=0.4,  # Ustawia przezroczystość na 50% (0.0 to w pełni przezroczysty, 1.0 to w pełni kryjący)
        edgecolor='black',  # Ustawia kolor obwódki na czarny
        linewidth=0.5,  # Ustawia grubość obwódki (możesz dostosować tę wartość, aby uzyskać "cienki" efekt)
        zorder=1
    )

    ax.scatter(DS.x, DS.y, color='red', linewidth=1, edgecolor='black', label='reference station', zorder=2)
    for i, curr_vs in enumerate(DS.upstream_adjacent_vs):
        ax.scatter(curr_vs.x, curr_vs.y, color=colors[i], linewidth=1, edgecolor='black', zorder=2)
    # Użycie osobnego wywołania scatter dla legendy, aby uniknąć wielu wpisów
    ax.scatter([], [], color='purple', linewidth=1, edgecolor='black', label='VS within buffer', zorder=2)
    current_river.gdf.to_crs(4326).plot(ax=ax, label='Odra river', zorder=1)
    ax.annotate(
        '',
        xy=(0.65, 0.49),  # Czubek strzałki
        xytext=(0.78, 0.46),  # Ogon strzałki
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            facecolor='steelblue',
            width=4
        ),
        transform=ax.transAxes,
        zorder=10
    )

    x_north, y_north, arrow_length = 0.5, 0.97, 0.1
    ax.annotate('N', xy=(x_north, y_north), xytext=(x_north, y_north - arrow_length),
                arrowprops=dict(facecolor='black', width=3, headwidth=10),
                ha='center', va='center', fontsize=12,
                xycoords=ax.transAxes)

    # Dodanie innych elementów
    # ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # Zmieniono, aby używać `ax` zamiast `plt` i usunięto drugą adnotację flow direction
    ax.text(0.082, 0.98, '(a)', transform=ax.transAxes, ha='right', va='top', fontsize=11)
    ax2.text(0.034, 0.98, '(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=11)

    # --- Poprawiona sekcja legendy ---
    handles, labels = ax.get_legend_handles_labels()
    flow_direction_handle = mlines.Line2D(
        [], [],
        color='steelblue',
        marker='$\u2192$',
        linestyle='None',
        markersize=15,
        label='Flow direction'
    )
    handles.append(flow_direction_handle)
    labels.append('Flow direction')

    swot_patch = mpatches.Patch(
        facecolor='gray',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.4,
        label='SWOT tiles'
    )
    handles.append(swot_patch)
    labels.append('SWOT tiles')
    ax.legend(
        handles=handles,
        labels=labels,
        loc='lower left'
    )

    first_vs = DS.upstream_adjacent_vs[0]
    ax2.errorbar(x=DS.chainage / 1000, y=DS.wl['wse'].mean(), yerr=(DS.wl['wse'].max() - DS.wl['wse'].min()) / 2,
                 fmt='o', capsize=3, label='Reference station water level', color='red')
    ax2.errorbar(x=first_vs.chainage / 1000, y=first_vs.wl['wse'].mean(),
                 yerr=(first_vs.wl['wse'].max() - first_vs.wl['wse'].min()) / 2, fmt='o', capsize=3,
                 label='VS water level', color=colors[0])
    for i, curr_vs in enumerate(DS.upstream_adjacent_vs[1:]):
        ax2.errorbar(x=curr_vs.chainage / 1000, y=curr_vs.wl['wse'].mean(),
                     yerr=(curr_vs.wl['wse'].max() - curr_vs.wl['wse'].min()) / 2, fmt='o', capsize=3,
                     color=colors[i + 1])
    ax2.set_xlabel('Chainage [km]')
    ax2.set_ylabel('Water level [m]')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')

    # --- ZMIANA: Ręczne dostosowanie marginesów ---
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.12, top=0.95, wspace=0.05)

    # plt.savefig(f'{fig_dir}vs_setting_at_{river_name.split(",")[0]}_with_SWOT.png', dpi=300)
    plt.show(block=True)


def plot_map_with_vs_setting_and_vs_water_levels_v2(DS):
    swot_tiles_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/SWORD_v17b_shp/SWOT_tiles/swot_passes_oder.shp'
    swot_gdf = gpd.read_file(swot_tiles_file)
    x_list, y_list = map(list, zip(*[(a.x, a.y) for a in DS.upstream_adjacent_vs]))
    plot_buffer_up, plot_buffer_down, plot_buffer_left, plot_buffer_right = 0.1, 0.1, 0.1, 0.1
    x_max, x_min, y_max, y_min = max(x_list) + plot_buffer_right, min(x_list) - plot_buffer_left, max(
        y_list) + plot_buffer_up, min(
        y_list) - plot_buffer_down

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = plt.cm.get_cmap('viridis')
    num_vs = len(DS.upstream_adjacent_vs)
    colors = cmap(np.linspace(0, 1, num_vs))
    # Rysowanie elementów na mapie
    swot_gdf.plot(
        ax=ax,
        color='gray',  # Ustawia kolor wypełnienia na szary
        alpha=0.4,  # Ustawia przezroczystość na 50% (0.0 to w pełni przezroczysty, 1.0 to w pełni kryjący)
        edgecolor='black',  # Ustawia kolor obwódki na czarny
        linewidth=0.5,  # Ustawia grubość obwódki (możesz dostosować tę wartość, aby uzyskać "cienki" efekt)
        zorder=1
    )

    ax.scatter(DS.x, DS.y, color='red', linewidth=1, edgecolor='black', s=75, label='reference station', zorder=2)
    for i, curr_vs in enumerate(DS.upstream_adjacent_vs):
        ax.scatter(curr_vs.x, curr_vs.y, color=colors[i], linewidth=1, s=75, edgecolor='black', zorder=2)
    # Użycie osobnego wywołania scatter dla legendy, aby uniknąć wielu wpisów
    ax.scatter([], [], color=colors[10], linewidth=1, edgecolor='black', s=75, label='VS within buffer', zorder=2)
    current_river.gdf.to_crs(4326).plot(ax=ax, label='Odra river', linewidth=3, zorder=1)
    ax.annotate(
        '',
        xy=(0.66, 0.36),  # Czubek strzałki
        xytext=(0.77, 0.33),  # Ogon strzałki
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            facecolor='steelblue',
            width=4
        ),
        transform=ax.transAxes,
        zorder=10
    )

    x_north, y_north, arrow_length = 0.95, 0.97, 0.08
    ax.annotate('N', xy=(x_north, y_north), xytext=(x_north, y_north - arrow_length),
                arrowprops=dict(facecolor='black', width=3, headwidth=10),
                ha='center', va='center', fontsize=12,
                xycoords=ax.transAxes)

    # Dodanie innych elementów
    # ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # Zmieniono, aby używać `ax` zamiast `plt` i usunięto drugą adnotację flow direction
    ax.text(0.082, 0.98, '(a)', transform=ax.transAxes, ha='right', va='top', fontsize=11)
    ax2.text(0.075, 0.98, '(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=11)

    # --- Poprawiona sekcja legendy ---
    handles, labels = ax.get_legend_handles_labels()
    flow_direction_handle = mlines.Line2D(
        [], [],
        color='steelblue',
        marker='$\u2192$',
        linestyle='None',
        markersize=15,
        label='Flow direction'
    )
    handles.append(flow_direction_handle)
    labels.append('flow direction')

    swot_patch = mpatches.Patch(
        facecolor='gray',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.4,
        label='SWOT tiles'
    )
    handles.append(swot_patch)
    labels.append('examplary SWOT tiles')
    ax.legend(
        handles=handles,
        labels=labels,
        loc='lower left'
    )

    first_vs = DS.upstream_adjacent_vs[0]
    ax2.errorbar(x=DS.chainage / 1000, y=DS.wl['wse'].mean(), yerr=(DS.wl['wse'].max() - DS.wl['wse'].min()) / 2,
                 fmt='o', capsize=3, label='Reference station water level', color='red')
    ax2.errorbar(x=first_vs.chainage / 1000, y=first_vs.wl['wse'].mean(),
                 yerr=(first_vs.wl['wse'].max() - first_vs.wl['wse'].min()) / 2, fmt='o', capsize=3,
                 label='VS water level', color=colors[0])
    for i, curr_vs in enumerate(DS.upstream_adjacent_vs[1:]):
        ax2.errorbar(x=curr_vs.chainage / 1000, y=curr_vs.wl['wse'].mean(),
                     yerr=(curr_vs.wl['wse'].max() - curr_vs.wl['wse'].min()) / 2, fmt='o', capsize=3,
                     color=colors[i + 1])
    ax2.set_xlabel('Chainage [km]')
    ax2.set_ylabel('Water level [m]')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')

    # --- ZMIANA: Ręczne dostosowanie marginesów ---
    # fig.subplots_adjust(left=0.04, right=0.98, bottom=0.12, top=0.95, wspace=0.05)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}vs_setting_at_{river_name.split(",")[0]}_with_SWOT2.png', dpi=300)
    # plt.show(block=True)


def plot_all_vs_timeseries_with_correlations(self, just_swot=False):
    fig, ax = plt.subplots(figsize=(11, 6))

    # Collect all chainages to normalize them for colormap
    all_chainages = [self.chainage] + [vs.chainage for vs in self.upstream_adjacent_vs]
    min_chainage = min(all_chainages)
    max_chainage = max(all_chainages)

    # Get the viridis colormap
    cmap = cm.get_cmap('viridis')
    # List to store plot data before plotting, for sorting
    plot_data = []

    # Function to get color based on chainage
    def get_color_from_chainage(chainage, min_c, max_c, colormap):
        if min_c == max_c:  # Handle case with only one chainage
            norm_chainage = 0.5
        else:
            norm_chainage = (chainage - min_c) / (max_c - min_c)
        return colormap(norm_chainage)

    # Process the current VS
    vs_to_corr1 = self.get_daily_linear_interpolated_wl_of_single_vs(just_swot)
    color_curr_vs = get_color_from_chainage(self.chainage, min_chainage, max_chainage, cmap)
    plot_data.append({
        'series': vs_to_corr1,
        'label': f'RS {self.id} ({self.chainage / 1000:.1f}km)',
        'color': 'red',
        'chainage': self.chainage,
        'linewidth': 2  # Make the current VS line thicker
    })

    # Process upstream adjacent VSs
    for vs in self.upstream_adjacent_vs:
        vs_to_corr2 = vs.get_daily_linear_interpolated_wl_of_single_vs(just_swot)
        if len(vs_to_corr2) > 0 and vs.id != self.id:
            correlation = vs_to_corr1.corr(vs_to_corr2)
            color_vs = get_color_from_chainage(vs.chainage, min_chainage, max_chainage, cmap)
            plot_data.append({
                'series': vs_to_corr2,
                'label': f'VS {vs.id} ({vs.chainage / 1000:.1f}km, Corr: {correlation:.3f})',
                'color': color_vs,
                'chainage': vs.chainage,
                'linewidth': 1  # Default linewidth
            })

    # Sort the plot_data list by chainage in descending order
    plot_data_sorted = sorted(plot_data, key=lambda x: x['chainage'], reverse=True)

    # Plot the data in the sorted order
    for data_item in plot_data_sorted:
        ax.plot(data_item['series'],
                label=data_item['label'],
                color=data_item['color'],
                linewidth=data_item['linewidth'])

    ax.set_xlabel('Time')
    ax.set_ylabel('Water Level [m]')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='center', bbox_to_anchor=(-0.35, 0.5))
    plt.tight_layout()
    plt.show(block=True)
    # plt.savefig(f'{fig_dir}RS_wl_corrs_at_{river_name.split(",")[0]}.png', dpi=300)


def plot_all_vs_timeseries(self):
    viridis_cmap = plt.colormaps['viridis']
    colors = [viridis_cmap(i) for i in np.linspace(0, 1, len(self.upstream_adjacent_vs))]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(self.closest_in_situ_daily_wl, label=f'In situ {round((self.neigh_g_up_chain - self.chainage) / 1000)}'
                                                 f' km from DS', color='red', linewidth=4)
    mean_ds = self.wl.wse.mean()
    curr_wl = self.wl.loc[(self.wl.index > self.wl.index.min()) & (self.wl.index < self.wl.index.max())]
    ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color='magenta', linewidth=3, label=f'DS {self.id}')
    for i, vs in enumerate(sorted(self.upstream_adjacent_vs, key=lambda x: x.chainage)):
        # print(vs.wl.index)
        curr_wl = vs.wl.loc[(vs.wl.index > self.wl.index.min()) & (vs.wl.index < self.wl.index.max())]
        ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color=colors[i],
                label=f'{vs.id}, {round((vs.chainage - self.chainage) / 1000)} km from DS')
    ax.legend(loc='center', bbox_to_anchor=(-0.35, 0.5))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show(block=True)


def plot_regressions_btwn_stations(vs_id_cofl, vs_id_main, res_str, color):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    dahiti = DAHITI()
    VS_cofl = sc.VirtualStation(vs_id_cofl)
    VS_cofl.get_water_levels(dahiti)
    VS_cofl.time_filter(t1, t2)

    VS_main = sc.VirtualStation(vs_id_main)
    VS_main.get_water_levels(dahiti)
    VS_main.time_filter(t1, t2)

    vs_cofl_set = set(pd.to_datetime(VS_cofl.wl['datetime']).dt.round(res_str))
    vs_main_set = set(pd.to_datetime(VS_main.wl['datetime']).dt.round(res_str))
    common_indices = vs_cofl_set.intersection(vs_main_set)
    vs_cofl_swot_ts = VS_cofl.wl.set_index(pd.to_datetime(VS_cofl.wl['datetime']).dt.round(res_str))['wse'].loc[
        list(common_indices)]
    vs_main_swot_ts = VS_main.wl.set_index(pd.to_datetime(VS_main.wl['datetime']).dt.round(res_str))['wse'].loc[
        list(common_indices)]
    regr_df = pd.DataFrame(index=list(common_indices))
    regr_df['vs_cofl_wse'] = vs_cofl_swot_ts
    regr_df['vs_main_wse'] = vs_main_swot_ts
    linr_model = LinearRegression().fit(regr_df[['vs_cofl_wse']], regr_df[['vs_main_wse']])
    r_squared = r2_score(y_true=regr_df[['vs_main_wse']], y_pred=linr_model.predict(regr_df[['vs_cofl_wse']]))
    a, b, r2, len_common = round(linr_model.coef_[0][0], 3), round(linr_model.intercept_[0], 3), round(r_squared,
                                                                                                       3), len(
        common_indices)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(regr_df['vs_cofl_wse'], regr_df['vs_main_wse'], color=color)
    ax.plot(regr_df['vs_cofl_wse'], linr_model.predict(regr_df['vs_cofl_wse'].values.reshape(-1, 1)), color=color)
    # ax.set_xlabel(f'{vs_id_cofl} water levels [m]')
    # ax.set_ylabel(f'{vs_id_main} water levels [m]')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    print(f'y = {a}x + {b}, r$^2$ = {r2}')
    # ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    ax.axis('off')
    plt.savefig(f'{fig_dir}regression_{vs_id_main}_{vs_id_cofl}.png', dpi=300, transparent=True)
    # plt.show(block=True)


plot_RS_regressions_background = False
if plot_RS_regressions_background:
    cmap = plt.cm.get_cmap('viridis')
    num_vs = len(DS.upstream_adjacent_vs)
    colors = cmap(np.linspace(0, 1, num_vs))
    # colors = cmap(np.linspace(0, 1, 7))
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(x=DS.chainage / 1000, y=DS.wl['wse'].mean(), yerr=(DS.wl['wse'].max() - DS.wl['wse'].min()) / 2,
                fmt='o', capsize=3, label='RS water levels', color='red')
    chains, wls = [DS.chainage / 1000], [DS.wl['wse'].mean()]
    for i, curr_vs in enumerate(DS.upstream_adjacent_vs[16:20]):
        if i == 0:
            ax.errorbar(x=curr_vs.chainage / 1000, y=curr_vs.wl['wse'].mean(),
                        yerr=(curr_vs.wl['wse'].max() - curr_vs.wl['wse'].min()) / 2,
                        fmt='o', capsize=3, label=f'VS water levels', color=colors[i + 1])
        else:
            ax.errorbar(x=curr_vs.chainage / 1000, y=curr_vs.wl['wse'].mean(),
                        yerr=(curr_vs.wl['wse'].max() - curr_vs.wl['wse'].min()) / 2,
                        fmt='o', capsize=3, color=colors[i + 1])
        chains.append(curr_vs.chainage / 1000)
        wls.append(curr_vs.wl['wse'].mean())
        # ax.text(curr_vs.chainage/1000, curr_vs.wl['wse'].min() - 1, f'VS {curr_vs.id}')
        # ax.scatter(curr_vs.chainage/1000, curr_vs.wl['wse'].mean(), color=colors[i + 1])
    ax.plot(chains, wls)
    ax.set_xlabel('Chainage [km]')
    ax.set_ylabel('Water level [m]')
    # ax.set_xlim(min(chains) - 20, max(chains) + 20)
    # ax.set_ylim(min(wls) - 3, max(wls) + 5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    plt.tight_layout()
    # plt.show(block=True)
    plt.savefig(f'{fig_dir}RS_regressions.png', dpi=300)


def filter_outliers_by_tstudent_test(df, window_days=3, min_periods=3, confidence_level=0.99, plot_outliers=True):
    df = df.copy(deep=True)
    alpha = 1 - confidence_level  # Poziom istotności (0.01)
    z_critical = stats.norm.ppf(1 - alpha / 2)  # Dla 99% to ok. 2.576
    df['Rolling_Mean'] = df['shifted_wl'].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).mean()
    df['Rolling_Std'] = df['shifted_wl'].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).std()
    df['Lower_Bound'] = df['Rolling_Mean'] - (z_critical * df['Rolling_Std'])
    df['Upper_Bound'] = df['Rolling_Mean'] + (z_critical * df['Rolling_Std'])
    df['Is_Outlier'] = (df['shifted_wl'] < df['Lower_Bound']) | \
                       (df['shifted_wl'] > df['Upper_Bound'])

    print(f"\nŁączna liczba zidentyfikowanych outlierów: {df['Is_Outlier'].sum()}")
    if plot_outliers:
        fig, ax = plt.subplots(figsize=(10*0.72, 7*0.72))
        ax.plot(df.index, df['shifted_wl'], label='Water level', alpha=0.8, marker='.')
        ax.plot(df.index, df['Rolling_Mean'], label='Water level average (moving window)', color='orange', linestyle='--')
        ax.plot(df.index, df['Lower_Bound'], label='Lower confidence interval (99%)', color='red', linestyle=':')
        ax.plot(df.index, df['Upper_Bound'], label='Upper confidence interval (99%)', color='red', linestyle=':')

        # Zaznaczanie outlierów
        outliers = df[df['Is_Outlier']]
        ax.scatter(outliers.index, outliers['shifted_wl'], color='purple', marker='o', s=50, zorder=5,
                    label='Identified outliers')

        # plt.title('Wykrywanie Outlierów w Znormalizowanym Szeregu Czasowym')
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level')
        ax.legend()
        curr_data = df.loc[pd.to_datetime('2024-10-16'): pd.to_datetime('2024-11-07')]
        ax.set_xlim(curr_data.index.min(), curr_data.index.max())
        ax.set_ylim(curr_data['Lower_Bound'].min() - .1, curr_data['Upper_Bound'].max() + .4)
        ax.grid(True, linestyle='--', alpha=0.7)
        # plt.show(block=True)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}RS_wl_outliers.png', dpi=300)

    # return df.loc[(-df['Is_Outlier'])]


def plot_approach(self, ts_juxtp, ts_itpd):
    fig, ax = plt.subplots(figsize=(12 * 0.72, 7 * 0.72))
    ax.plot(self.closest_in_situ_daily_wl, label='Gauge water level', color='black', linewidth=4, zorder=1)
    ax.plot(self.single_VS_itpd, label='RS water levels interpolated', color='blue', linewidth=2, zorder=4)
    ax.plot(ts_itpd, label='RS densified water levels interpolated', color='red', linewidth=2, zorder=4)
    ax.scatter(ts_juxtp['shifted_time'], ts_juxtp['shifted_wl'], marker='.', s=75, color='red', edgecolor='grey',
               linewidth=.45, label='WL measurements juxtaposed at RS', zorder=3)

    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Water level [m]')
    curr_df = ts_itpd.loc[pd.to_datetime('2024-01-01'): pd.to_datetime('2024-06-01')]
    # curr_df = ts_itpd.loc[pd.to_datetime('2024-06-01'): pd.to_datetime('2024-12-01')]
    ax.set_xlim(curr_df.index.min(), curr_df.index.max())
    ax.set_ylim(curr_df.values.min() - .5, curr_df.values.max() + .5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.show(block=True)
    plt.savefig(f'{fig_dir}RS_wl_resulting_ts_{riv}.png', dpi=300)


def plot_accuracies(res_df, riv, cols1=res_df.columns[-10:-5]):
    metric, metric2 = 'RMSE [m]', 'NSE'
    cols2 = [x.replace('rmse', 'nse') for x in cols1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # ax.boxplot(res_df[res_df.columns[[5,7,8,9,11,12]]].dropna(axis=0))  # RMSE
    selected_data1 = res_df[cols1].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
    selected_data2 = res_df[cols2].dropna(axis=0).loc[~res_df['id'].isin(neigh_dam_vs)]
    ax.boxplot(selected_data1)
    ax.set_xlabel('Methods')
    ax.set_ylabel(metric)
    cols1_labels = [f'{col}\nmean: {round(res_df[col].mean(), 3)}' for col in cols1]
    ax.set_xticklabels(cols1_labels)

    ax2.boxplot(selected_data2)
    ax2.set_xlabel('Methods')
    ax2.set_ylabel(metric2)
    cols2_labels = [f'{col}\nmean: {round(res_df[col].mean(), 3)}' for col in cols2]
    ax2.set_xticklabels(cols2_labels)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)
    fig.suptitle(
        f'Accuracy of daily water level time series based on {len(selected_data1)} RS at the {riv} River')
    plt.tight_layout()
    plt.show(block=True)
    # plt.savefig(f'{fig_dir}RS_accuracy_at_{riv}.png', dpi=300)


def get_n_colors_from_cmap(cmap_name, n):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]

# plot_swot_obs_against_gauge_data(DS)
plot_map_with_vs_setting_and_vs_water_levels(DS)
print(1)


# plot_regressions_btwn_stations(23404, DS.id, 'h', colors[1])
# plot_regressions_btwn_stations(42305, 23404, 'h', colors[2])
# plot_regressions_btwn_stations(19763, 42305, 'h', colors[3])
# plot_regressions_btwn_stations(23406, 19763, 'h', colors[4])