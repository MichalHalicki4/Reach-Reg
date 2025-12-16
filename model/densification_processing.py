import pandas as pd
from model import station_utils as s_utils
from model.Station_class import ReferenceStation
from model.data_mapping import all_river_data, configs
import pickle
import copy
import os


def _run_core_densification(vs, riv_object, loaded_stations, confg, river_data):
    """
    Performs the core densification process for a single Virtual Station (VS).

    This includes filtering neighboring VS, calculating slopes, interpolating,
    performing regression-based densification, calibrating Manning's 'c',
    time-shifting, filtering outliers, and performing cross-validation.

    :param vs: The (copied) target VirtualStation object.
    :param riv_object: River object containing dams and tributary chains.
    :param loaded_stations: A list of all VirtualStation objects.
    :param confg: Global configuration dictionary (buffer, thresholds, etc.).
    :param river_data: Global dictionary containing river-specific data (e.g., vs_with_neight_dams).
    :returns: The fully processed ReferenceStation object, and two cross-validation metrics (rmse_cval, nse_cval).
    """

    rs = ReferenceStation(vs, confg['buffer'], confg['itpd_method'])
    rs.get_upstream_adjacent_vs(loaded_stations)
    df_true = rs.swot_wl[['wse']].set_index(pd.to_datetime(rs.swot_wl['datetime'])).resample('D').mean().dropna()

    # 1. Spatial and Statistical Filtering
    rs.filter_stations_by_corr_amp_dams_tribs_other(confg['corr_thres'], confg['amp_thres'], riv_object.dams,
                                                    riv_object.tributary_chains,
                                                    river_data[vs.river.split(',')[0]]['vs_with_neight_dams'], False)
    rs.filter_stations_only_with_swot()
    if rs.is_rs_empty_or_at_edge():
        return None, None, None, None, None

    # 2. Densification and Hydraulic Adjustment
    rs.get_slope_of_all_vs()
    rs.get_single_vs_interpolated_ts()
    rs.get_densified_wl_by_regressions(rmse_thres=confg['rmse_thres'], single_rmse_thres=confg['single_rmse_thres'])
    rs.calibrate_mannings_c()
    rs.densified_ts = rs.calculate_shifted_time_by_simplified_mannig(rs.densified_ts, confg['bottom'])

    # 3. Filtering and Smoothing
    rms_thr = rs.get_rmse_agg_threshold(df_true)
    rs.densified_ts = rs.densified_ts.loc[rs.densified_ts['rmse_sum'] < rms_thr]
    rs.densified_ts = s_utils.filter_outliers_by_tstudent_test(rs.densified_ts)
    rs.densified_ts, rs.densified_daily, rs.densified_itpd = rs.get_svr_smoothed_data(rs.densified_ts)

    rs.add_uncertainty_column()
    rs.merge_regr_and_itp_uncertainty()

    # 4. Cross-Validation (CVAL)
    densified_ts_cval = rs.densified_ts.loc[rs.densified_ts['id_vs'] != rs.id]
    densified_ts_cval_daily = s_utils.get_final_weighted_wl(densified_ts_cval)
    densified_ts_cval_itpd = rs.interpolate(densified_ts_cval_daily, 'daily_wse')

    rmse_cval, nse_cval = rs.get_rmse_nse_values(densified_ts_cval_itpd['daily_wse'], 'CrossVal', df_true)

    return rs, rmse_cval, nse_cval, densified_ts_cval_daily, densified_ts_cval_itpd


def densify_wl_no_gdata(vs_id, riv, riv_object, loaded_stations, dir_rs, dir_ts):
    """
    Densifies the water level time series for a single Virtual Station (VS) using data
    from adjacent upstream VS, without relying on any ground truth Gauge Stations.

    :param vs_id: The unique identifier of the target Virtual Station (VS) to be densified.
    :param riv: The name of the river, used for file naming and accessing global data structures.
    :param riv_object: River object containing dams and tributary chains.
    :param loaded_stations: A list of all VirtualStation objects available in the area.
    :param dir_rs: Directory path for saving the resulting pickled ReferenceStation object.
    :param dir_ts: Directory path for saving the resulting densified time series (CSV) and metadata (CSV).
    :returns: None, as the function primarily saves results to disk.
    """
    # Setup and Pre-checks
    vs = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
    if vs.id in all_river_data[riv]['vs_with_neight_dams']:
        return None

    # Core Densification Logic
    rs, rmse_cval, nse_cval, _, _ = _run_core_densification(vs, riv_object, loaded_stations, configs, all_river_data)
    if rs is None:
        return None
    # Metrics Calculation
    mean_uncrt = rs.densified_ts['uncertainty'].mean()
    mean_rmse_sum = rs.densified_ts['rmse_sum'].mean()

    # File Paths and Metadata
    res_path_pkl = f'{dir_rs}{riv}_RS{rs.id}_no_gdata.pkl'
    res_path_csv = f'{dir_ts}{riv}_RS{rs.id}.csv'
    metadata_path = f'{dir_ts}{riv}_metadata.csv'

    metadata_list = [vs_id, vs.x, vs.y, round(rs.chainage / 1000, 3), riv, rs.speed_ms, rs.c, rs.v_uncrt_range,
                     len(rs.densified_ts), len(rs.densified_ts['id_vs'].unique()), round(mean_uncrt, 3),
                     round(mean_rmse_sum, 3), rmse_cval, nse_cval]

    # Saving Files
    with open(res_path_pkl, "wb") as f:
        pickle.dump(rs, f)
    rs.densified_itpd.to_csv(res_path_csv, sep=';')

    # Metadata Update Logic
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path, sep=';')
        metadata_df.loc[len(metadata_df)] = metadata_list
    else:
        res_cols_no_gdata = ['id', 'x', 'y', 'chain', 'river', 'velocity', 'c', 'v_uncrt_range', 'num_of_all_meas',
                             'num_of_vs', 'mean_uncrt', 'mean_rmse_sum', 'rmse_cval', 'nse_cval']
        metadata_df = pd.DataFrame([metadata_list], columns=res_cols_no_gdata)
    metadata_df.to_csv(metadata_path, sep=';', index=False)


def densify_wl_with_gdata(vs_id, riv, riv_object, loaded_stations, loaded_gauges, dir_rs, dir_ts):
    """
    Densifies the water level time series for a single Virtual Station (VS) and
    validates/calibrates the result against ground truth data from the closest
    Gauge Station (GS).

    :param vs_id: The unique identifier of the target Virtual Station (VS) to be densified.
    :param riv: The name of the river, used for file naming and accessing global data structures.
    :param riv_object: River object containing dams and tributary chains.
    :param loaded_stations: A list of all VirtualStation objects available in the area.
    :param loaded_gauges: A dictionary of loaded GaugeStation objects (ID: object) for ground truth.
    :param dir_rs: Directory path for saving the resulting pickled ReferenceStation object.
    :param dir_ts: Directory path for saving the resulting densified time series (CSV) and metadata (CSV).
    :returns: None, as the function primarily saves results to disk.
    """
    # Setup and Pre-checks
    vs = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
    gauge_chain = vs.neigh_g_up_chain if vs.closest_gauge == 'up' else vs.neigh_g_dn_chain
    gauge_id = vs.neigh_g_up if vs.closest_gauge == 'up' else vs.neigh_g_dn
    vs_gauge_dist = abs(vs.chainage - gauge_chain) / 1000

    if len(vs.juxtaposed_wl) == 0 or vs_gauge_dist > all_river_data[riv]['gauge_dist_threshold'] \
            or vs.id in all_river_data[riv]['vs_with_neight_dams'] or \
            len(vs.juxtaposed_wl.loc[vs.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0:
        return None

    # Core Densification Logic
    rs, rmse_cval, nse_cval, densified_ts_cval_daily, densified_ts_cval_itpd = \
        _run_core_densification(vs, riv_object, loaded_stations, configs, all_river_data)

    if rs is None:
        return None

    # Integration with Gauge Data (Specific to 'with_gdata')
    adjusted_gauge_data = rs.adjust_gauge_data_to_vs_by_regr(loaded_gauges[gauge_id].wl_df, gauge_chain)
    rs.densified_ts = s_utils.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, rs.densified_ts)
    rs.get_closest_in_situ_daily_wl(adjusted_gauge_data, rs.wl.index.min(), rs.wl.index.max())

    # Metrics Calculation
    rmse_rr, nse_rr = rs.get_rmse_nse_values(rs.densified_itpd['wse'], 'REACH-REG')
    rmse_raw, nse_raw = rs.get_rmse_nse_values(rs.single_VS_itpd, 'SINGLE VS')
    rmse_daily, nse_daily = rs.get_rmse_nse_values(densified_ts_cval_daily['daily_wse'], 'Daily')
    prct_in_unct = rs.get_percentage_within_uncrt()

    mean_bias = rs.densified_ts['bias'].mean()
    mean_rmse_sum = rs.densified_ts['rmse_sum'].mean()

    # File Paths and Metadata
    res_path_pkl = f'{dir_rs}{riv}_RS{rs.id}.pkl'
    res_path_csv = f'{dir_ts}{riv}_RS{rs.id}.csv'
    metadata_path = f'{dir_ts}{riv}_metadata.csv'

    metadata_list = [vs_id, vs.x, vs.y, round(rs.chainage / 1000, 3), riv, round(gauge_chain / 1000, 3),
                     round(rs.speed_ms, 3), rs.c, rs.v_uncrt_range, len(rs.densified_ts),
                     len(rs.densified_ts['id_vs'].unique()), round(mean_bias, 3), round(prct_in_unct, 3),
                     round(mean_rmse_sum, 3), rmse_rr, rmse_raw, rmse_cval, rmse_daily, nse_rr, nse_raw, nse_cval,
                     nse_daily]

    # Saving Files
    with open(res_path_pkl, "wb") as f:
        pickle.dump(rs, f)
    rs.densified_itpd.to_csv(res_path_csv, sep=';')

    # Metadata Update Logic
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path, sep=';')
        metadata_df.loc[len(metadata_df)] = metadata_list
    else:
        res_cols_with_gdata = ['id', 'x', 'y', 'chain', 'river', 'g_chain', 'velocity', 'c', 'v_uncrt_range',
                               'num_of_all_meas', 'num_of_vs', 'mean_bias', 'prct_in_unct', 'mean_rmse_sum', 'rmse_rr',
                               'rmse_raw', 'rmse_cval', 'rmse_daily', 'nse_rr', 'nse_raw', 'nse_cval', 'nse_daily']
        metadata_df = pd.DataFrame([metadata_list], columns=res_cols_with_gdata)
    metadata_df.to_csv(metadata_path, sep=';', index=False)
