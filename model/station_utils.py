from datetime import timedelta
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from shapely import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import copy
from .data_mapping import dahiti_in_situ_collections


def get_optimum_lag(ts1, ts2, n):
    """
    Calculates the optimum time lag (in hours) that maximizes the correlation
    coefficient between two time series (ts1 and ts2).

    The search is performed for lags from 0 up to n hours. This is typically used
    to estimate the travel time of a wave between two adjacent gauge stations.

    :param ts1: The first time series (Pandas Series).
    :param ts2: The second time series (Pandas Series), which is shifted in time.
    :param n: The maximum number of hours to check for the lag.
    :returns: A tuple (best_lag, max_corr) containing the lag in hours and the
              maximum correlation coefficient achieved.
    """
    max_corr, best_lag = 0, 0
    for lag in range(n):
        ts2_shifted = ts2.copy()
        ts2_shifted.index = [t + timedelta(hours=lag) for t in ts2_shifted.index]
        corr = ts1.corr(ts2_shifted)
        if corr > max_corr:
            max_corr = corr
            best_lag = lag
    return best_lag, round(max_corr, 3)


def select_gauges_from_river(gdata_gdf, river, buff=1000):
    """
    Filters a GeoDataFrame of all gauge stations to select only those that fall
    within a specified buffer distance (buff) of the river centerline geometry.

    :param gdata_gdf: GeoDataFrame containing all gauge stations (points).
    :param river: River object containing the river geometry and CRS information.
    :param buff: The buffer distance (in meters) around the river centerline.
    :returns: A filtered GeoDataFrame of gauge stations located within the river buffer.
    """
    gdata_gdf = gdata_gdf.to_crs(river.gdf.crs)
    gdata_gdf_metric = gdata_gdf.to_crs(river.metrical_crs)
    river_buffer_metric = gpd.GeoSeries(river.simplified_river).buffer(buff)
    return gdata_gdf_metric[gdata_gdf.to_crs(river.metrical_crs).within(river_buffer_metric.iloc[0])]


def get_chainages_for_all_gauges(curr_gauges, river):
    """
    Calculates the chainage (distance along the river centerline from a fixed origin)
    for each gauge station in the provided GeoDataFrame.

    :param curr_gauges: GeoDataFrame of gauge stations.
    :param river: River object with a method for calculating chainage.
    :returns: The input GeoDataFrame, updated with a 'chainage' column.
    """
    chainages = []
    for index, row in curr_gauges.iterrows():
        chainages.append(river.get_chainage_of_point(row['X'], row['Y']))
    curr_gauges['chainage'] = chainages
    return curr_gauges


def get_list_of_stations_from_country(country, insitu):
    """
    Retrieves a list of all available in-situ stations (from DAHITI's collections)
    for a specified country/region.

    :param country: Identifier for the country/region (used to access a global dict
                    'dahiti_in_situ_collections').
    :param insitu: Client object for the in-situ data provider (DAHITI).
    :returns: A flattened list of all station metadata objects/dictionaries.
    """
    stations_data = []
    for insitu_id in dahiti_in_situ_collections[country]:
        curr_stations = insitu.list_collection(insitu_id)
        for station in curr_stations:
            stations_data.append(station)
    return stations_data


def filter_gauges_by_dt_freq_target(gauges_metadata, min_dt, dahiti=True):
    """
    Filters gauge station metadata based on several criteria:
    - Data type must be 'water_level'.
    - Station must have started measuring before a minimum date ('min_dt').
    - (DAHITI specific): Excludes stations whose 'target_name' contains 'see' (e.g., lakes).

    :param gauges_metadata: DataFrame containing metadata for all gauge stations.
    :param min_dt: The minimum required start date for the time series.
    :param dahiti: Boolean flag to apply DAHITI-specific filters.
    :returns: A filtered DataFrame of valid gauge station metadata.
    """
    if dahiti:
        return gauges_metadata.loc[(gauges_metadata['type'] == 'water_level') &
                                   (pd.to_datetime(gauges_metadata['max_date']) > pd.to_datetime(min_dt)) &
                                   # (pd.to_datetime(gauges_metadata['min_date']) < pd.to_datetime(min_dt)) &
                                   # (pd.to_datetime(gauges_metadata['max_date']) > pd.to_datetime(max_dt)) &
                                   # (gauges_metadata['data_sampling'] != 'daily') &
                                   (~gauges_metadata['target_name'].str.contains('see', na=False))]
    else:
        return gauges_metadata.loc[(gauges_metadata['type'] == 'water_level') &
                                   (pd.to_datetime(gauges_metadata['max_date']) > pd.to_datetime(min_dt))
                                   # (pd.to_datetime(gauges_metadata['max_date']) > pd.to_datetime(max_dt))
                                   ]


def juxtapose_gauge_to_densified_wl(gauge_meas_up, timeseries):
    """
    Compares the densified VS time series ('timeseries') with the time-shifted
    and WSE-adjusted gauge measurements ('gauge_meas_up').

    The comparison is done by rounding both timestamps to the nearest hour or day,
    averaging gauge data, and then merging. It calculates water level anomalies and
    the absolute bias between them.

    :param gauge_meas_up: DataFrame of gauge measurements, already shifted and adjusted.
    :param timeseries: The densified VS time series (self.densified_ts).
    :returns: The input 'timeseries' DataFrame, updated with gauge comparison data
              ('shifted_wl_gauge', anomalies, bias).
    """
    gauge_meas_up = copy.deepcopy(gauge_meas_up)
    if str(gauge_meas_up.index.freq) == '<Day>':
        timeseries['rounded_time'] = timeseries['shifted_time'].dt.round('d')
        gauge_meas_up['rounded_time'] = gauge_meas_up['shifted_time'].dt.round('d')
    else:
        timeseries['rounded_time'] = timeseries['shifted_time'].dt.round('h')
        gauge_meas_up['rounded_time'] = gauge_meas_up['shifted_time'].dt.round('h')

    gauge_meas_agg = gauge_meas_up.groupby('rounded_time')['shifted_wl'].mean().reset_index()
    timeseries_merged = pd.merge(
        timeseries,
        gauge_meas_agg,
        on='rounded_time',
        how='left',
        suffixes=('_x', '_y')
    )
    timeseries['shifted_wl_gauge'] = timeseries_merged['shifted_wl_y'].values
    timeseries.drop(columns=['rounded_time'], inplace=True)
    gauge_meas_up.drop(columns=['rounded_time'], inplace=True)
    mean_dens_g, mean_dens_vs = timeseries['shifted_wl_gauge'].mean(), timeseries['shifted_wl'].mean()
    timeseries['shifted_wl_gauge_anom'] = timeseries['shifted_wl_gauge'] - mean_dens_g
    timeseries['shifted_wl_anom'] = timeseries['shifted_wl'] - mean_dens_vs
    timeseries['shifted_wl_bias'] = abs(timeseries['shifted_wl_anom'] - timeseries['shifted_wl_gauge_anom'])
    return timeseries


def create_gdf_from_metadata(data_objects, is_gauge=False):
    """
    Creates a GeoDataFrame (gdf) from a list of VirtualStation objects or a dictionary
    of GaugeStation objects, including their coordinates, ID, chainage, and length
    of available data.

    :param data_objects: List or dictionary of station objects.
    :param is_gauge: Boolean flag indicating if the input is a dictionary of GaugeStation objects.
    :returns: A GeoDataFrame (CRS 4326) containing the key metadata for the stations.
    """
    list_x, list_y, list_id, list_lengths, list_chains, metadata = [], [], [], [], [], []

    if is_gauge:
        for key in data_objects.keys():
            obj = data_objects[key]
            list_x.append(obj.x)
            list_y.append(obj.y)
            list_id.append(obj.id)
            list_chains.append(obj.chainage)
            list_lengths.append(len(obj.wl_df))
    else:
        for obj in data_objects:
            list_x.append(obj.x)
            list_y.append(obj.y)
            list_id.append(obj.id)
            list_chains.append(obj.chainage)
            list_lengths.append(len(obj.wl))

    for i in range(len(list_x)):
        metadata.append({'id': list_id[i], 'x': list_x[i], 'y': list_y[i], 'length': list_lengths[i],
                         'chainage': list_chains[i]})

    df = pd.DataFrame(metadata)
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['x'], df['y'])], crs=4326)


def is_dam_between(st1_chain, st2_chain, dams):
    """
    Checks if there is a dam located between two given chainage points (st1_chain and st2_chain).

    :param st1_chain: Chainage of the first station.
    :param st2_chain: Chainage of the second station.
    :param dams: A list of tuples/lists where each element represents a dam location
                 (e.g., [dam_start_chain, dam_end_chain] or just [dam_chain, dam_chain]).
    :returns: True if a dam is found between the two chainages, False otherwise.
    """
    if st1_chain < st2_chain:
        chain1, chain2 = st2_chain, st1_chain
    else:
        chain1, chain2 = st1_chain, st2_chain
    for dam in dams:
        if chain1 > dam[0] > chain2 or chain1 > dam[1] > chain2:
            return True
    return False


def is_tributary_between(st1_chain, st2_chain, tributary_chains):
    """
    Checks if a tributary junction is located between two given chainage points.

    :param st1_chain: Chainage of the first station.
    :param st2_chain: Chainage of the second station.
    :param tributary_chains: A list of chainages where tributaries meet the main river.
    :returns: True if a tributary junction is found between the two chainages, False otherwise.
    """
    if st1_chain < st2_chain:
        chain1, chain2 = st2_chain, st1_chain
    else:
        chain1, chain2 = st1_chain, st2_chain
    for trib_chain in tributary_chains:
        if chain1 > trib_chain > chain2:
            return True
    return False


def get_rmse_between_two_ts(ts_true, ts_model):
    """
    Calculates the Root Mean Square Error (RMSE) between two time series.

    The time series are first combined, non-overlapping or NaN values are removed,
    and then the RMSE is computed only on the mutually available data points.

    :param ts_true: The reference (true) time series.
    :param ts_model: The model/predicted time series.
    :returns: The RMSE value, rounded to 4 decimal places.
    """
    ts_true = ts_true.loc[~ts_true.index.duplicated(keep='first')]
    ts_model = ts_model.loc[~ts_model.index.duplicated(keep='first')]
    df_combined = pd.concat([ts_true, ts_model], axis=1)
    df_combined.columns = ['gauge_mean', 'model_mean']
    df_cleaned = df_combined.dropna()
    y_true = df_cleaned['gauge_mean']
    y_predicted = df_cleaned['model_mean']
    rmse = round(np.sqrt(mean_squared_error(y_true, y_predicted)), 4)
    return rmse


def get_rmse_weighted_wl(ts):
    """
    Aggregates the individual WSE measurements in the densified time series ('ts')
    to a daily average, using a weighting scheme based on the cumulative regression
    error ('rmse_sum').

    Points with lower 'rmse_sum' receive a higher weight in the daily average calculation.

    :param ts: The densified time series DataFrame ('self.densified_ts').
    :returns: A Pandas Series of daily, RMSE-weighted water level averages.
    """
    curr_densified = ts.copy()
    curr_densified['rmse_weight'] = 1 / (curr_densified['rmse_sum'] + 1e-6)  # Mała wartość, żeby uniknąć 1/0
    curr_densified['weight_x_wl'] = curr_densified['shifted_wl'] * curr_densified['rmse_weight']
    rmse_weighted_daily = curr_densified.resample('D').agg(
        wl_x_weight_sum=('weight_x_wl', 'sum'),
        weight_sum=('rmse_weight', 'sum')
    )
    rmse_weighted_daily['wl_weighted'] = rmse_weighted_daily['wl_x_weight_sum'] / rmse_weighted_daily['weight_sum']
    return rmse_weighted_daily['wl_weighted']


def get_vs_neighbors(vs_station, vs_list):
    """
    Finds the immediate upstream ('neighbor_before') and downstream ('neighbor_after')
    Virtual Stations (VS) relative to a given 'vs_station' within a list of candidates.

    If no station exists in one direction, the 'vs_station' itself is returned as the neighbor
    in that direction (handling edge cases).

    :param vs_station: The target Virtual Station object.
    :param vs_list: A list of all candidate VS objects (should be the filtered adjacent VS).
    :returns: A tuple (neighbor_before, neighbor_after) of the two nearest VS objects.
    """
    try:
        neighbor_before = max((p for p in vs_list if p.chainage < vs_station.chainage), key=lambda p: p.chainage)
    except ValueError:
        neighbor_before = vs_station

    try:
        neighbor_after = min((p for p in vs_list if p.chainage > vs_station.chainage), key=lambda p: p.chainage)
    except ValueError:
        neighbor_after = vs_station

    return neighbor_before, neighbor_after


def get_slope(neighbor_before, neighbor_after):
    """
    Calculates the mean water surface slope (gradient) between two neighboring stations.

    The slope is calculated as the difference in mean WSE divided by the difference in chainage
    (converted to kilometers).

    :param neighbor_before: The upstream VS object.
    :param neighbor_after: The downstream VS object.
    :returns: The calculated mean slope in meters per kilometer (m/km), rounded to 3 decimal places.
    """
    chain_diff = (neighbor_after.chainage - neighbor_before.chainage)/1000
    wl_diff = neighbor_after.wl['wse'].mean() - neighbor_before.wl['wse'].mean()
    return round(wl_diff/chain_diff, 3)


def get_regression_coeffs_from_df(regr_df, str1, str2):
    """
    Performs a linear regression between two variables (str1 and str2) within a DataFrame.

    The model is fitted to predict str1 (Y) from str2 (X): Y = a*X + b.

    :param regr_df: DataFrame containing the two variables to be correlated.
    :param str1: Name of the dependent variable (Y, to be predicted).
    :param str2: Name of the independent variable (X, the predictor).
    :returns: A tuple (a, b, r2, commons) containing the slope (a), intercept (b),
              R-squared (r2), and the number of data points used (commons).
    """
    lin_model = LinearRegression().fit(regr_df[[str2]], regr_df[[str1]])
    r2 = r2_score(y_true=regr_df[[str1]], y_pred=lin_model.predict(regr_df[[str2]]))
    a, b, r2, commons = round(lin_model.coef_[0][0], 3), round(lin_model.intercept_[0], 3), round(r2, 3), len(regr_df)
    return a, b, r2, commons


def get_linear_regression_coeffs_btwn_stations(st_low_chain, st_high_chain, res_str='h'):
    """
    Calculates the linear regression coefficients, R-squared, and Root Mean Square Error (RMSE)
    between the water level time series of two adjacent stations.

    This function first identifies and aligns common timestamps (rounded to the specified
    resolution) between the two stations before fitting the regression model.

    :param st_low_chain: The station with the lower chainage.
    :param st_high_chain: The station with the higher chainage.
    :param res_str: The resampling resolution for the time index ('h' for hourly is default).
    :returns: A tuple (a, b, r2, rmse, data_len) or NaNs if fewer than 5 common points are found.
    """
    vs_set = set(pd.to_datetime(st_low_chain.wl['datetime']).dt.round(res_str).drop_duplicates(keep=False))
    vs2_set = set(pd.to_datetime(st_high_chain.wl['datetime']).dt.round(res_str).drop_duplicates(keep=False))
    common_indices = vs_set.intersection(vs2_set)

    if len(common_indices) < 5:
        print(st_low_chain.id, st_high_chain.id, len(common_indices))
        return np.nan, np.nan, np.nan, np.nan, len(common_indices)

    st_low_ts = st_low_chain.wl.set_index(pd.to_datetime(st_low_chain.wl['datetime']).dt.round(res_str))['wse'].loc[
        list(common_indices)]
    st_high_ts = st_high_chain.wl.set_index(pd.to_datetime(st_high_chain.wl['datetime']).dt.round(res_str))['wse'].loc[
        list(common_indices)]

    regr_df = pd.DataFrame(index=list(common_indices))
    regr_df['x_wse'] = st_low_ts
    regr_df['y_wse'] = st_high_ts

    a, b, r2, len_common = get_regression_coeffs_from_df(regr_df, 'x_wse', 'y_wse')

    ts1, ts2 = a * st_high_chain.wl['wse'] + b, st_low_chain.wl['wse']
    ts1.index = ts1.index.round('h')
    ts2.index = ts2.index.round('h')

    return a, b, r2, get_rmse_between_two_ts(ts1, ts2), len(common_indices)


def _apply_regression(known_wse: float, known_station_id: int, reg_row: Dict[str, Any]) -> float:
    """
    Applies the linear regression equation to calculate the Water Surface Elevation (WSE)
    at the second station.

    Args:
        known_wse: The known water level.
        known_station_id: The ID of the station for which the water level is known.
        reg_row: The regression parameters row from the DataFrame as a dictionary.

    Returns:
        The calculated water level at the target station.
    """
    a, b = reg_row['a'], reg_row['b']
    st1, st2 = reg_row['st1'], reg_row['st2']

    # The regression is defined as: WSE_st1 = a * WSE_st2 + b.

    # Case 1: We know WSE for st1, calculating for st2 (X = (Y - b) / a)
    if known_station_id == st1:
        # Avoids division by zero, although 'a' is rarely zero in river WSE regressions.
        return (known_wse - b) / a if a != 0 else np.nan

    # Case 2: We know WSE for st2, calculating for st1 (Y = a*X + b)
    elif known_station_id == st2:
        return a * known_wse + b

    else:
        # This should ideally not happen if regression lookup is correct.
        raise ValueError("known_station_id does not match either station in the regression.")


def find_regression(st_id1: int, st_id2: int, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Searches for the regression parameters between two stations in the regression DataFrame,
    accounting for the order of stations (st1-st2 or st2-st1).

    :param st_id1: ID of the first station.
    :param st_id2: ID of the second station.
    :param df: DataFrame containing all pre-calculated regression parameters.
    :returns: A dictionary containing the regression parameters, or None if no regression is found.
    """
    condition = ((df['st1'] == st_id1) & (df['st2'] == st_id2)) | \
                ((df['st1'] == st_id2) & (df['st2'] == st_id1))
    result = df[condition]
    if not result.empty:
        return result.iloc[0].to_dict()
    return None


def get_wl_by_regression_v_pro(
        start_wse: float,
        start_station_id: int,
        target_station_id: int,
        regressions_df: pd.DataFrame,
        all_stations_list: List,
        single_rmse_thres: float,
        total_rmse_thres: float
) -> Tuple[float, float, str]:
    """
    Propagates the water state (WSE) from the starting station to the target station
    by iteratively using inter-station regressions, implementing a pathfinding algorithm
    that selects detours (2-step jumps) for weak (high-RMSE) direct connections.

    Args:
        start_wse: The initial water state.
        start_station_id: ID of the starting station (where WSE is known).
        target_station_id: ID of the target station (RS).
        regressions_df: DataFrame with regression parameters (1-step and 2-step).
        all_stations_list: An ordered list of all stations along the river.
        single_rmse_thres: The RMSE threshold for a single connection, classifying it as "weak" if exceeded.
        total_rmse_thres: The maximum allowable sum of RMSE for the entire propagation path.

    Returns:
        A tuple (final_wse, total_rmse, path_string) or (NaN, NaN, "Error message") upon failure.
    """
    if start_station_id == target_station_id:
        return start_wse, 0, str(start_station_id)

    # Data structure preparation
    station_pos_map = {st.id: i for i, st in enumerate(all_stations_list)}

    if start_station_id not in station_pos_map or target_station_id not in station_pos_map:
        return np.nan, np.nan, "Station not found in all_stations_list"

    current_station_id = start_station_id
    path = [current_station_id]
    wse_at_station = {current_station_id: start_wse}
    total_rmse = 0.0

    while current_station_id != target_station_id:
        current_pos = station_pos_map[current_station_id]
        target_pos = station_pos_map[target_station_id]

        # Determine direction: 1 for downstream, -1 for upstream propagation
        direction = 1 if target_pos > current_pos else -1

        # 1. Identify the direct neighbor in the direction of the target
        direct_neighbor_pos = current_pos + direction
        if not (0 <= direct_neighbor_pos < len(all_stations_list)):
            return np.nan, np.nan, "No path available (river boundary reached)"
        direct_neighbor_id = all_stations_list[direct_neighbor_pos].id

        direct_reg = find_regression(current_station_id, direct_neighbor_id, regressions_df)
        if not direct_reg:
            return np.nan, np.nan, f"No regression found for neighbor: {current_station_id}-{direct_neighbor_id}"

        # 2. Decision: Go direct or seek a detour
        if direct_reg['rmse'] < single_rmse_thres:
            # Direct connection is reliable, choose it
            best_move = {
                'start_id': current_station_id,
                'target_id': direct_neighbor_id,
                'reg': direct_reg,
                'type': 'direct'
            }
        else:
            # Connection is weak (high RMSE), search for alternatives (detours)
            detour_options = []

            # Option 1: Forward Jump (e.g., jump from station 2 to 4, skipping 3)
            forward_jump_pos = current_pos + 2 * direction
            # Check range and ensure the jump does not overshoot the target
            if (0 <= forward_jump_pos < len(all_stations_list)) and \
                    (direction * (forward_jump_pos - target_pos) <= 0):
                forward_jump_target_id = all_stations_list[forward_jump_pos].id
                forward_reg = find_regression(current_station_id, forward_jump_target_id, regressions_df)
                if forward_reg:
                    detour_options.append({
                        'start_id': current_station_id,
                        'target_id': forward_jump_target_id,
                        'reg': forward_reg,
                        'type': 'forward_jump'
                    })

            # Option 2: Backward Jump (e.g., from st. 1 to 3, having just arrived at 2)
            # This is only possible if we are not at the start of the entire path
            if len(path) > 1:
                previous_station_id = path[-2]
                # The target of this jump is the same neighbor that had the weak connection
                backward_jump_target_id = direct_neighbor_id
                backward_reg = find_regression(previous_station_id, backward_jump_target_id, regressions_df)
                if backward_reg:
                    detour_options.append({
                        'start_id': previous_station_id,
                        'target_id': backward_jump_target_id,
                        'reg': backward_reg,
                        'type': 'backward_jump'
                    })

            # Include the original, weak connection as a fallback option
            detour_options.append({
                'start_id': current_station_id,
                'target_id': direct_neighbor_id,
                'reg': direct_reg,
                'type': 'weak_direct'
            })

            # Select the best option from all available (lowest RMSE)
            best_move = min(detour_options, key=lambda x: x['reg']['rmse'])

        # 3. Apply the selected move (best_move)
        move_reg = best_move['reg']
        move_start_id = best_move['start_id']
        move_target_id = best_move['target_id']

        # Calculate new WSE
        wse_for_calc = wse_at_station[move_start_id]
        new_wse = _apply_regression(wse_for_calc, move_start_id, move_reg)

        # Update total RMSE
        total_rmse += move_reg['rmse']
        if total_rmse > total_rmse_thres:
            return np.nan, np.nan, "Exceeded total RMSE threshold"

        # Update loop state
        wse_at_station[move_target_id] = new_wse

        # Path update depends on the move type
        if best_move['type'] == 'backward_jump':
            # Remove the last station because we "stepped back" (effectively st1 -> st3 skip st2)
            path.pop()
            path.append(move_target_id)
            current_station_id = move_target_id

        else:  # direct, weak_direct, forward_jump
            path.append(move_target_id)
            current_station_id = move_target_id

    # 4. Finalization and result return
    final_wse = wse_at_station[target_station_id]
    path_string = '->'.join(map(str, path))

    return final_wse, total_rmse, path_string


def calculate_path_for_row(row, target_station_id, regressions_df, all_stations_list, single_rmse_thres,
                           total_rmse_thres):
    """
    An adapter function for use with pandas.DataFrame.apply().
    It extracts row-specific values and calls the main computational function
    to find the optimal regression path and WSE propagation for a single measurement.

    :param row: A single measurement row from the juxtaposed time series.
    :param target_station_id: ID of the target station (RS).
    :param regressions_df: DataFrame of regression parameters.
    :param all_stations_list: Ordered list of all stations.
    :param single_rmse_thres: Threshold for a single regression RMSE.
    :param total_rmse_thres: Threshold for the cumulative path RMSE.
    :returns: The tuple (final_wse, total_rmse, path_string) from get_wl_by_regression_v_pro.
    """
    # Extract row-specific values
    start_wse = row['vs_wl']
    start_station_id = row['id_vs']

    # Call the main function with the full set of arguments
    return get_wl_by_regression_v_pro(
        start_wse=start_wse,
        start_station_id=start_station_id,
        target_station_id=target_station_id,
        regressions_df=regressions_df,
        all_stations_list=all_stations_list,
        single_rmse_thres=single_rmse_thres,
        total_rmse_thres=total_rmse_thres
    )


def filter_outliers_by_tstudent_test(df, window_days=3, min_periods=3, confidence_level=0.99, plot_outliers=False):
    """
    Filters outliers from the densified time series ('shifted_wl') using a rolling
    mean and standard deviation approach, typically based on a confidence interval
    (e.g., 99% using the Z-score/t-score approximation).

    :param df: The time series DataFrame (e.g., self.densified_ts) with 'shifted_wl'.
    :param window_days: The time window (in days) for the rolling calculation.
    :param min_periods: Minimum number of observations required to calculate a rolling value.
    :param confidence_level: The confidence level (e.g., 0.99) to define the bounds.
    :param plot_outliers: Boolean to display a diagnostic plot of the detected outliers.
    :returns: The DataFrame containing only non-outlier data points.
    """
    df = df.copy(deep=True)
    alpha = 1 - confidence_level  # Significance level (0.01)

    # Using Z-critical (from a standard normal distribution) as an approximation
    # for large sample size or when the t-distribution is complex to use on rolling data.
    z_critical = stats.norm.ppf(1 - alpha / 2)  # For 99% this is approx. 2.576

    # Calculate rolling statistics
    df['Rolling_Mean'] = df['shifted_wl'].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).mean()
    df['Rolling_Std'] = df['shifted_wl'].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).std()

    # Calculate confidence bounds (Mean +/- Z_critical * Std)
    df['Lower_Bound'] = df['Rolling_Mean'] - (z_critical * df['Rolling_Std'])
    df['Upper_Bound'] = df['Rolling_Mean'] + (z_critical * df['Rolling_Std'])

    # Identify outliers
    df['Is_Outlier'] = (df['shifted_wl'] < df['Lower_Bound']) | \
                       (df['shifted_wl'] > df['Upper_Bound'])

    if plot_outliers:
        # Diagnostic plot implementation
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['shifted_wl'], label='Water Level', alpha=0.8, marker='.')
        plt.plot(df.index, df['Rolling_Mean'], label='Rolling Mean', color='orange', linestyle='--')
        plt.plot(df.index, df['Lower_Bound'], label=f'Lower Confidence Limit ({confidence_level * 100:.0f}%)',
                 color='red', linestyle=':')
        plt.plot(df.index, df['Upper_Bound'], label=f'Upper Confidence Limit ({confidence_level * 100:.0f}%)',
                 color='red', linestyle=':')

        # Highlight outliers
        outliers = df[df['Is_Outlier']]
        plt.scatter(outliers.index, outliers['shifted_wl'], color='purple', marker='o', s=50, zorder=5,
                    label='Identified Outliers')

        plt.title('Outlier Detection in Normalized Time Series')
        plt.xlabel('Date')
        plt.ylabel('Normalized Water Level')
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
        #  # Example of a valuable image tag

    return df.loc[(-df['Is_Outlier'])]
