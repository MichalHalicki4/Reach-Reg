import pandas as pd
from model import station_utils as s_utils
from model.Station_class import ReferenceStation
import pickle
import copy
import os
import datetime


def _run_core_densification(vs, riv_object, loaded_stations, cfg):
    """
    Performs the core densification process for a single Virtual Station (VS).

    :param vs: The (copied) target VirtualStation object.
    :param riv_object: River object containing dams and tributary chains.
    :param loaded_stations: A list of all VirtualStation objects.
    :param cfg: ReachRegConfig object containing all hyperparameters and metadata.
    :returns: The processed ReferenceStation object and cross-validation metrics.
    """
    start = datetime.datetime.now()
    # Use cfg object for initialization
    rs = ReferenceStation(vs, cfg.buffer, cfg.itpd_method)
    rs.get_upstream_adjacent_vs(loaded_stations)

    df_true = rs.swot_wl[['wse']].set_index(pd.to_datetime(rs.swot_wl['datetime'])).resample('D').mean().dropna()

    # 1. Spatial and Statistical Filtering
    # Using thresholds and dam-related lists directly from the cfg object
    rs.filter_stations_by_corr_amp_dams_tribs_other(
        cfg.corr_thres,
        cfg.amp_thres,
        riv_object.dams,
        riv_object.tributary_chains,
        cfg.vs_with_neight_dams,
        False
    )

    rs.filter_stations_only_with_swot()
    if rs.is_rs_empty_or_at_edge():
        return None, None, None, None, None

    # 2. Densification and Hydraulic Adjustment
    rs.get_slope_of_all_vs()
    rs.get_single_vs_interpolated_ts()
    #print(f'RS before densification: {datetime.datetime.now() - start}')
    rs.get_densified_wl_by_regressions(
        rmse_thres=cfg.rmse_thres,
        single_rmse_thres=cfg.single_rmse_thres
    )
    #print(f'RS after initial densification: {datetime.datetime.now() - start}')
    # rs.calibrate_mannings_c()
    rs.calibrate_mannings_c_parallel()
    rs.densified_ts = rs.calculate_shifted_time_by_simplified_mannig(rs.densified_ts, cfg.bottom)

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
    #print(f'RS after densification: {start - datetime.datetime.now()}')

    return rs, rmse_cval, nse_cval, densified_ts_cval_daily, densified_ts_cval_itpd


def densify_wl_no_gdata(vs_id, cfg, riv_object, loaded_stations, dir_rs, dir_ts):
    """
    Densifies time series without ground truth validation.
    """
    vs = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])

    # Check against excluded VS list from cfg
    if vs.id in cfg.vs_with_neight_dams:
        return None

    # Core Logic
    rs, rmse_cval, nse_cval, _, _ = _run_core_densification(vs, riv_object, loaded_stations, cfg)

    if rs is None:
        return None

    mean_uncrt = rs.densified_ts['uncertainty'].mean()
    mean_rmse_sum = rs.densified_ts['rmse_sum'].mean()

    # Paths using cfg.river_name
    res_path_pkl = f"{dir_rs}{cfg.river_name}_RS{rs.id}_no_gdata.pkl"
    res_path_csv = f"{dir_ts}{cfg.river_name}_RS{rs.id}.csv"
    metadata_path = f"{dir_ts}{cfg.river_name}_metadata_no_gdata.csv"
    lstm_metadata_path = f"{dir_ts}{cfg.river_name}_lstm_metadata_no_gdata.csv"

    metadata_list = [vs_id, vs.x, vs.y, round(rs.chainage / 1000, 3), cfg.river_name, rs.speed_ms, rs.c,
                     rs.v_uncrt_range,
                     len(rs.densified_ts), len(rs.densified_ts['id_vs'].unique()), round(mean_uncrt, 3),
                     round(mean_rmse_sum, 3), rmse_cval, nse_cval]
    lstm_metadata_list = [vs_id, vs.x, vs.y, vs.sword_reach['wse'], vs.sword_reach['width'], vs.sword_reach['facc'],
                          rs.slope]

    # Saving
    with open(res_path_pkl, "wb") as f:
        pickle.dump(rs, f)
    rs.densified_itpd.to_csv(res_path_csv, sep=';')

    _update_metadata_file(metadata_path, metadata_list, 'no-gauge')
    _update_metadata_file(lstm_metadata_path, lstm_metadata_list, 'lstm')


def densify_wl_with_gdata(vs_id, cfg, riv_object, loaded_stations, loaded_gauges, dir_rs, dir_ts):
    """
    Densifies time series and validates against Gauge Station data.
    """
    vs = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
    gauge_chain = vs.neigh_g_up_chain if vs.closest_gauge == 'up' else vs.neigh_g_dn_chain
    gauge_id = vs.neigh_g_up if vs.closest_gauge == 'up' else vs.neigh_g_dn
    vs_gauge_dist = abs(vs.chainage - gauge_chain) / 1000

    # Filtering using cfg thresholds
    if (len(vs.juxtaposed_wl) == 0 or
            vs_gauge_dist > cfg.gauge_dist_threshold or
            vs.id in cfg.vs_with_neight_dams or
            len(vs.juxtaposed_wl.loc[vs.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) == 0):
        return None

    # Core Logic
    rs, rmse_cval, nse_cval, densified_ts_cval_daily, densified_ts_cval_itpd = \
        _run_core_densification(vs, riv_object, loaded_stations, cfg)

    if rs is None:
        return None

    # Gauge Integration
    adjusted_gauge_data = rs.adjust_gauge_data_to_vs_by_regr(loaded_gauges[gauge_id].wl_df, gauge_chain)
    rs.densified_ts = s_utils.juxtapose_gauge_to_densified_wl(adjusted_gauge_data, rs.densified_ts)
    rs.get_closest_in_situ_daily_wl(adjusted_gauge_data, rs.wl.index.min(), rs.wl.index.max())

    # Metrics
    rmse_rr, nse_rr = rs.get_rmse_nse_values(rs.densified_itpd['wse'], 'REACH-REG')
    rmse_raw, nse_raw = rs.get_rmse_nse_values(rs.single_VS_itpd, 'SINGLE VS')
    rmse_daily, nse_daily = rs.get_rmse_nse_values(densified_ts_cval_daily['daily_wse'], 'Daily')
    prct_in_unct = rs.get_percentage_within_uncrt()

    mean_bias = rs.densified_ts['bias'].mean()
    mean_rmse_sum = rs.densified_ts['rmse_sum'].mean()

    # Paths
    res_path_pkl = f"{dir_rs}{cfg.river_name}_RS{rs.id}.pkl"
    res_path_csv = f"{dir_ts}{cfg.river_name}_RS{rs.id}.csv"
    metadata_path = f"{dir_ts}{cfg.river_name}_metadata_.csv"
    lstm_metadata_path = f"{dir_ts}{cfg.river_name}_lstm_metadata.csv"
    metadata_list = [vs_id, vs.x, vs.y, round(rs.chainage / 1000, 3), cfg.river_name, round(gauge_chain / 1000, 3),
                     round(rs.speed_ms, 3), rs.c, rs.v_uncrt_range, len(rs.densified_ts),
                     len(rs.densified_ts['id_vs'].unique()), round(mean_bias, 3), round(prct_in_unct, 3),
                     round(mean_rmse_sum, 3), rmse_rr, rmse_raw, rmse_cval, rmse_daily, nse_rr, nse_raw, nse_cval,
                     nse_daily]
    lstm_metadata_list = [vs_id, vs.x, vs.y, vs.sword_reach['wse'], vs.sword_reach['width'], vs.sword_reach['facc'],
                          rs.slope]

    # Saving
    with open(res_path_pkl, "wb") as f:
        pickle.dump(rs, f)
    rs.densified_itpd.to_csv(res_path_csv, sep=';')
    _update_metadata_file(metadata_path, metadata_list, 'gauge')
    _update_metadata_file(lstm_metadata_path, lstm_metadata_list, 'lstm')


def _update_metadata_file(path, data_list, dtype):
    """
    Helper function to handle CSV metadata updates.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, sep=';')
        df.loc[len(df)] = data_list
    else:
        if dtype == 'gauge':
            cols = ['id', 'x', 'y', 'chain', 'river', 'g_chain', 'velocity', 'c', 'v_uncrt_range',
                    'num_of_all_meas', 'num_of_vs', 'mean_bias', 'prct_in_unct', 'mean_rmse_sum', 'rmse_rr',
                    'rmse_raw', 'rmse_cval', 'rmse_daily', 'nse_rr', 'nse_raw', 'nse_cval', 'nse_daily']
        elif dtype == 'lstm':
            cols = ['id', 'x', 'y', 'wse', 'width', 'facc', 'slope']
        else:
            cols = ['id', 'x', 'y', 'chain', 'river', 'velocity', 'c', 'v_uncrt_range', 'num_of_all_meas',
                    'num_of_vs', 'mean_uncrt', 'mean_rmse_sum', 'rmse_cval', 'nse_cval']
        df = pd.DataFrame([data_list], columns=cols)
    df.to_csv(path, sep=';', index=False)
