from datetime import timedelta
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.odr import ODR, Model, Data, RealData
import pandas as pd
from shapely import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import copy


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


def filter_gauges_by_target_name(gauges_df, insitu, riv_name):
    """
    Filters a GeoDataFrame of all gauge stations to select only those from a given river.

    :param gauges_df: GeoDataFrame containing all gauge stations (points).
    :param insitu: The InSitu object from Dahiti.
    :param riv_name: The name of the river.
    :returns: A filtered GeoDataFrame of gauge stations from a river.
    """
    valid_gauges = []
    for x in gauges_df['id'].unique():
        target_info = insitu.get_target_info(int(x))
        if target_info['target_name'] == riv_name or target_info['target_name'] == f'{riv_name}, River':
            valid_gauges.append(x)
    return gauges_df[gauges_df['id'].isin(valid_gauges)]


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


def get_list_of_stations_from_config(cfg, insitu):
    """
    Retrieves a list of all available in-situ stations from DAHITI's collections
    based on the IDs provided in the configuration object.

    :param cfg: ReachRegConfig object containing the 'dahiti_collections' list.
    :param insitu: Client object for the in-situ data provider (DAHITI).
    :returns: A flattened list of all station metadata objects/dictionaries.
    """
    stations_data = []

    # We no longer look up by country name in a global dict.
    # The cfg object already holds the correct list of IDs for this specific run.
    for insitu_id in cfg.dahiti_collections:
        try:
            curr_stations = insitu.list_collection(insitu_id)
            for station in curr_stations:
                stations_data.append(station)
        except Exception as e:
            print(f"Warning: Could not retrieve DAHITI collection {insitu_id}: {e}")

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


def get_final_weighted_wl(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza dobowe WSE oraz dobową Niepewność (U_daily) używając Ważonej Średniej.

    Zakłada, że df ma kolumny 'shifted_wl' (WSE po SVR) i
    'regr_svr_u' (u_final z SVR-P.U.).
    """
    eps = 1e-6
    df = input_df.copy(deep=True)

    # 1. Obliczenie wag: w_j = 1 / u_final^2
    # Zapewnienie, że wariancja jest niezerowa
    try:
        df['u_final_safe'] = np.maximum(eps, df['regr_svr_u'])
    except KeyError:
        df['u_final_safe'] = np.maximum(eps, df['regr_u'])

    weights_var = df['u_final_safe'] ** 2
    df['weight'] = 1.0 / weights_var

    # 2. Obliczenie Ważonej Średniej dla WSE
    df['weighted_wl'] = df['shifted_wl'] * df['weight']

    # Agregacja dzienna
    daily_aggregated = df.groupby(df.index.date).agg(
        sum_weighted_wl=('weighted_wl', 'sum'),
        sum_weights=('weight', 'sum'),
        # Zliczanie obserwacji dla kontroli
        N=('shifted_wl', 'count')
    )

    # 3. Obliczenie finalnej Średniej Dobowej WSE (L_bar)
    # L_bar = sum(w*L) / sum(w)
    daily_aggregated['daily_wse'] = daily_aggregated['sum_weighted_wl'] / daily_aggregated['sum_weights']

    # 4. Obliczenie Finalnej Niepewności Dobowej (U_daily)
    # u_daily^2 = 1 / sum(w)
    daily_aggregated['daily_uncertainty'] = np.sqrt(1.0 / daily_aggregated['sum_weights'])

    # Czyszczenie i przygotowanie wyników
    results_df = daily_aggregated[['daily_wse', 'daily_uncertainty', 'N']]
    results_df.index = pd.to_datetime(results_df.index)

    return results_df[['daily_wse', 'daily_uncertainty', 'N']]  # Zwracamy WSE, U_daily i U


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


def linear_func(params, x_data):
    return params[0] * x_data + params[1]


def get_odr_regression_coeffs(regr_df, str1, str2):
    """
    Performs an Orthogonal Distance Regression (ODR) (Errors-in-Variables)
    between two variables, including known uncertainties for both.
    ...
    """
    sigma_system_min = 0.01
    x = regr_df[str2].values
    y = regr_df[str1].values

    x_u = np.maximum(regr_df[str2 + '_u'].values, sigma_system_min)
    y_u = np.maximum(regr_df[str1 + '_u'].values, sigma_system_min)

    linear_model = Model(linear_func)

    data = RealData(
        x=x,
        y=y,
        sx=x_u,
        sy=y_u
    )

    ols_model = LinearRegression().fit(x.reshape(-1, 1), y)
    beta0_ols = [ols_model.coef_[0], ols_model.intercept_]
    odr_obj = ODR(data, linear_model, beta0=beta0_ols)
    output = odr_obj.run()
    a = output.beta[0]
    b = output.beta[1]
    a_std_err = output.sd_beta[0]
    b_std_err = output.sd_beta[1]

    y_pred = linear_func(output.beta, x)  # <--- POPRAWNE OBLICZANIE y_pred
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r2 = 1.0 if ss_total == 0 else 1 - (ss_residual / ss_total)
    reduced_chi_sq = output.res_var
    results = {
        'a': round(a, 3),
        'b': round(b, 3),
        'a_std_err': round(a_std_err, 3),
        'b_std_err': round(b_std_err, 3),
        'r2': round(r2, 3),
        'reduced_chi_sq': round(reduced_chi_sq, 3),
        'commons': len(regr_df)
    }
    return results['a'], results['b'], results['r2'], results['commons']


def fit_odr(x, y, x_err=None, y_err=None):  # DANIEL'S SCRIPT
    """
    Perform a linear Orthogonal Distance Regression (Total Least Squares).

    Parameters
    ----------
    x, y : array-like
        Input data arrays.
    x_err, y_err : array-like or None
        Standard deviations (uncertainties) of x and y values.

    Returns
    -------
    result : dict
        Dictionary containing fit parameters and uncertainties:
        - 't', 'c': slope and intercept
        - 't_err', 'c_err': standard errors of parameters
        - 'cov_tc': covariance between t and c
        - 'model': the linear model function
        - 'out': full ODR output object
    """
    # Linear model: y = t*x + c
    def f(B, x):
        return B[0] * x + B[1]

    # Prepare data for ODR
    data = RealData(x, y, sx=x_err, sy=y_err)
    model = Model(f)
    ols_model = LinearRegression().fit(x.reshape(-1, 1), y)
    beta0_ols = [ols_model.coef_[0], ols_model.intercept_]
    odr = ODR(data, model, beta0=beta0_ols)
    out = odr.run()

    # Extract results
    t, c = out.beta
    cov_beta = out.cov_beta
    t_err, c_err = np.sqrt(np.diag(cov_beta))
    cov_tc = cov_beta[0, 1]

    result = {
        "t": t,
        "c": c,
        "t_err": t_err,
        "c_err": c_err,
        "cov_tc": cov_tc,
        "model": f,
        "out": out
    }
    return result


def predict_odr(x_new, x_err, fit_result):  # DANIEL'S SCRIPT
    """
    Compute predicted y values and propagated uncertainties from an ODR fit.

    Parameters
    ----------
    x_new : array-like
        New x values for prediction.
    x_err : array-like
        Standard deviations (uncertainties) of x values.
    fit_result : dict
        Output dictionary from fit_odr().

    Returns
    -------
    y_pred, y_err : np.ndarray
        Predicted y values and propagated uncertainties.
    """
    t = fit_result["t"]
    c = fit_result["c"]
    t_err = fit_result["t_err"]
    c_err = fit_result["c_err"]
    cov_tc = fit_result["cov_tc"]

    x_new = np.asarray(x_new)
    x_err = np.asarray(x_err)

    # Predicted y
    y_pred = t * x_new + c

    # Error propagation:
    # σ_y² = (x * σ_t)² + σ_c² + (t * σ_x)² + 2 * x * cov_tc
    y_err = np.sqrt((x_new * t_err)**2 + c_err**2 + (t * x_err)**2 + 2 * x_new * cov_tc)

    return y_pred, y_err


def get_linear_regression_coeffs_btwn_stations(st_low_chain, st_high_chain, res_str='h'):
    """
    Calculates ODR parameters, R-squared, and RMSE between two time series,
    including uncertainty parameters (t_err, c_err, cov_tc) for propagation.
    """
    # ... (Fragmenty kodu do uzgadniania indeksów czasowych są zachowane) ...
    vs_set = set(pd.to_datetime(st_low_chain.wl['datetime']).dt.round(res_str).drop_duplicates(keep=False))
    vs2_set = set(pd.to_datetime(st_high_chain.wl['datetime']).dt.round(res_str).drop_duplicates(keep=False))
    common_indices = vs_set.intersection(vs2_set)

    if len(common_indices) < 5:
        # print(st_low_chain.id, st_high_chain.id, len(common_indices))
        # Zwraca 8 wartości: a, b, r2, rmse, commons + a_err, b_err, cov_ab
        return np.nan, np.nan, np.nan, np.nan, len(common_indices), np.nan, np.nan, np.nan

        # 1. Wyrównanie danych (zakładamy, że 'wl' zawiera 'wse_u' - niepewność WSE)
    st_low_ts = st_low_chain.wl.set_index(pd.to_datetime(st_low_chain.wl['datetime']).dt.round(res_str)).loc[
        list(common_indices)].rename(columns={'wse': 'x_wse', 'wse_u': 'x_u'})
    st_high_ts = st_high_chain.wl.set_index(pd.to_datetime(st_high_chain.wl['datetime']).dt.round(res_str)).loc[
        list(common_indices)].rename(columns={'wse': 'y_wse', 'wse_u': 'y_u'})

    regr_df = pd.concat([st_low_ts[['x_wse', 'x_u']], st_high_ts[['y_wse', 'y_u']]], axis=1)

    # Konwencja: st2 (y) = t * st1 (x) + c.
    # Używamy st_low_chain.id jako st1 i st_high_chain.id jako st2.
    # Choć to jest niezgodne z hydro-logiką, jest zgodne z Twoją starą logiką.

    # 2. Fit ODR: WSE_st_high (Y) = t * WSE_st_low (X) + c
    # Musimy zdecydować, które idzie na X, a które na Y. Utrzymajmy konwencję:
    # X = st_low_chain, Y = st_high_chain
    # Poprawienie konwencji z Twojego starego kodu: st_low_chain (X) -> st_high_chain (Y)

    fit_result = fit_odr(
        regr_df['x_wse'].values, regr_df['y_wse'].values,
        x_err=regr_df['x_u'].values, y_err=regr_df['y_u'].values
    )

    a, b, a_err, b_err, cov_tc = fit_result["t"], fit_result["c"], fit_result["t_err"], fit_result["c_err"], fit_result[
        "cov_tc"]

    # 3. Obliczenie R2 i RMSE
    # RMSE i R2 są liczone z prognoz Y = t*X + c
    y_pred = a * regr_df['x_wse'] + b
    r2 = r2_score(y_true=regr_df['y_wse'], y_pred=y_pred)
    rmse = np.sqrt(mean_squared_error(regr_df['y_wse'], y_pred))

    len_common = len(common_indices)

    # 4. Zwracanie wyników (zgodnie z poprzednią strukturą + nowe parametry)
    return round(a, 3), round(b, 3), round(r2, 3), round(rmse, 4), len_common, round(a_err, 4), round(b_err, 4), round(
        cov_tc, 6)


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


def propagate_wse_and_uncertainty(
        known_wse: float,
        known_u: float,
        reg_row: Dict[str, Any],
        known_id: int
) -> Tuple[float, float]:
    """
    Propagates WSE and its uncertainty (U) through a single ODR regression segment,
    applying the correct P.U. formula based on the propagation direction.

    Args:
        known_wse: The known WSE value (start point).
        known_u: The propagated uncertainty (U) of the known WSE.
        reg_row: The regression parameters row (must contain a, b, a_err, b_err, cov_ab, st1, st2).
        known_id: The ID of the station where WSE is currently known.

    Returns:
        Tuple (calculated_wse, calculated_u).
    """
    a, b = reg_row["a"], reg_row["b"]
    st1, st2 = reg_row["st1"], reg_row["st2"]

    # Parametry ODR/P.U. (Zapisane w reg_row jako a_err, b_err, cov_ab)
    sigma_a, sigma_b, cov_tc = reg_row["a_err"], reg_row["b_err"], reg_row["cov_ab"]

    # SCENARIUSZ 1: PROPAGACJA ZGODNA Z TRENOWANIEM (st1 -> st2), y = t*x + c
    if known_id == st1:
        x, sigma_x = known_wse, known_u
        y_pred = a * x + b

        # Propagacja Niepewności (Wzór Liniowy): σ_y² = (x * σ_t)² + σ_c² + (t * σ_x)² + 2 * x * cov_tc
        sigma_y_sq = (x * sigma_a) ** 2 + sigma_b ** 2 + (a * sigma_x) ** 2 + 2 * x * cov_tc
        sigma_y_sq_safe = np.maximum(0, sigma_y_sq)

        return y_pred, np.sqrt(sigma_y_sq_safe)

    # SCENARIUSZ 2: PROPAGACJA PRZECIWNA (st2 -> st1), x = (y-c)/t
    elif known_id == st2:
        y, sigma_y = known_wse, known_u

        if a == 0:
            return np.nan, np.nan

        x_pred = (y - b) / a

        # Wzór Różniczki Totalnej: df/dy = 1/t, df/dt = (-y + c) / t², df/dc = -1/t
        df_dy = 1 / a
        df_dt = (-y + b) / (a ** 2)
        df_dc = -1 / a

        # Wariancja σ_x²
        sigma_x_sq = (df_dy ** 2 * sigma_y ** 2) + \
                     (df_dt ** 2 * sigma_a ** 2) + \
                     (df_dc ** 2 * sigma_b ** 2) + \
                     2 * df_dt * df_dc * cov_tc

        sigma_x_sq_safe = np.maximum(0, sigma_x_sq)
        return x_pred, np.sqrt(sigma_x_sq_safe)

    else:
        raise ValueError("known_id nie pasuje do żadnej stacji w regresji.")


def get_wl_by_regression_v_pro(
        start_wse: float,
        start_u: float, # NOWY ARGUMENT: Niepewność początkowa (np. z Dahiti/SWOT)
        start_station_id: int,
        target_station_id: int,
        regressions_df: pd.DataFrame,
        all_stations_list: List,
        single_rmse_thres: float,
        total_rmse_thres: float
) -> Tuple[float, float, float, str]: # ZWRACA: wse, uncrt_prop, total_rmse, path_string
    """
    Przenosi WSE i jego niepewność ze stacji startowej na docelową, używając
    propagacji niepewności ODR. Wybór ścieżki jest oparty na minimalizacji sumy RMSE ODR.

    Returns:
        Krotka (final_wse, final_u, total_odr_rmse_sum, path_string) lub (NaN, NaN, NaN, Komunikat błędu).
    """
    if start_station_id == target_station_id:
        return start_wse, start_u, 0, str(start_station_id)

    # Data structure preparation
    station_pos_map = {st.id: i for i, st in enumerate(all_stations_list)}

    if start_station_id not in station_pos_map or target_station_id not in station_pos_map:
        return np.nan, np.nan, np.nan, "Station not found in all_stations_list"

    current_station_id = start_station_id
    path = [current_station_id]
    wse_at_station = {current_station_id: start_wse}
    u_at_station = {current_station_id: start_u} # NOWY SŁOWNIK: Zapisywanie propagowanej niepewności
    total_odr_rmse = 0.0 # Nadal służy do thresholdingu i wyboru ścieżki

    while current_station_id != target_station_id:
        current_pos = station_pos_map[current_station_id]
        target_pos = station_pos_map[target_station_id]

        # Determine direction: 1 for downstream, -1 for upstream propagation
        direction = 1 if target_pos > current_pos else -1

        # 1. Identify the direct neighbor in the direction of the target
        direct_neighbor_pos = current_pos + direction
        if not (0 <= direct_neighbor_pos < len(all_stations_list)):
            return np.nan, np.nan, np.nan, "No path available (river boundary reached)"
        direct_neighbor_id = all_stations_list[direct_neighbor_pos].id

        direct_reg = find_regression(current_station_id, direct_neighbor_id, regressions_df)
        if not direct_reg:
            return np.nan, np.nan, np.nan, f"No regression found for neighbor: {current_station_id}-{direct_neighbor_id}"

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
        move_reg = best_move['reg']
        move_start_id = best_move['start_id']
        move_target_id = best_move['target_id']

        # Obliczenie nowego WSE I NIEPEWNOŚCI za pomocą ODR-P.U.
        wse_for_calc = wse_at_station[move_start_id]
        u_for_calc = u_at_station[move_start_id]  # POBRANIE WCZEŚNIEJ PROPAGOWANEJ NIEPEWNOŚCI

        # Użycie nowej funkcji ODR-P.U.:
        new_wse, new_u = propagate_wse_and_uncertainty(
            known_wse=wse_for_calc,
            known_u=u_for_calc,
            reg_row=move_reg,
            known_id=move_start_id
        )

        if np.isnan(new_wse):
            return np.nan, np.nan, np.nan, f"Błąd propagacji ODR: t=0 lub błąd matematyczny."

        # Aktualizacja total_odr_rmse (tylko do filtrowania ścieżki)
        total_odr_rmse += move_reg['rmse']
        if total_odr_rmse > total_rmse_thres:
            return np.nan, np.nan, np.nan, "Przekroczono całkowity próg RMSE ODR"

        # Aktualizacja stanu pętli
        wse_at_station[move_target_id] = new_wse
        u_at_station[move_target_id] = new_u  # ZAPIS NOWEJ PROPAGOWANEJ NIEPEWNOŚCI
        current_station_id = move_target_id

        # Path update depends on the move type
        if best_move['type'] == 'backward_jump':
            # Remove the last station because we "stepped back" (effectively st1 -> st3 skip st2)
            path.pop()
            path.append(move_target_id)
            current_station_id = move_target_id

        else:  # direct, weak_direct, forward_jump
            path.append(move_target_id)
            current_station_id = move_target_id

    # 4. Zakończenie i zwrot wyników
    final_wse = wse_at_station[target_station_id]
    final_u = u_at_station[target_station_id]  # ZWROT FINALNEJ NIEPEWNOŚCI
    path_string = '->'.join(map(str, path))

    return final_wse, final_u, total_odr_rmse, path_string


def calculate_path_for_row(row, target_station_id, regressions_df, all_stations_list, single_rmse_thres,
                           total_rmse_thres):
    """
    Funkcja-adapter do użycia z pandas.DataFrame.apply().
    Teraz przekazuje WSE i niepewność (U) początkową.
    """
    # Wyodrębnij wartości specyficzne dla wiersza
    start_wse = row['vs_wl']
    start_u = row['uncertainty']  # NOWY POBIERANY ARGUMENT
    start_station_id = row['id_vs']

    # Wywołaj główną funkcję z pełnym zestawem argumentów
    return get_wl_by_regression_v_pro(
        start_wse=start_wse,
        start_u=start_u,  # NOWY ARGUMENT
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


def reindex_series_to_daily(ts):
    start_date = ts.index.min()
    end_date = ts.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return ts.reindex(full_date_range)


def run_single_calibration_step(c, slf_base, original_base_ts, bottom, df_true):
    """
    Wykonuje pojedynczą iterację kalibracji dla podanego c.
    To musi być funkcja samodzielna (top-level), aby multiprocessing mógł ją spakować (pickle).
    """
    # 1. Shift
    current_ts = slf_base.calculate_shifted_time_by_simplified_mannig(original_base_ts, bottom, c)

    # 2. Szybkie RMSE do progu (Twoja nowa funkcja)
    raw_rmse = slf_base.get_raw_rmse_fast(current_ts, df_true)

    # 3. Progi i filtracja
    amp_thres_final = 0.1 if raw_rmse < 0.15 else 0.2
    wl_amplitude = current_ts['shifted_wl'].max() - current_ts['shifted_wl'].min()
    rms_thr = wl_amplitude * amp_thres_final

    filtered_ts = current_ts.loc[current_ts['rmse_sum'] < rms_thr].copy()
    filtered_ts = filter_outliers_by_tstudent_test(filtered_ts)

    # 4. SVR
    densified_ts_cval = filtered_ts.loc[filtered_ts['id_vs'] != slf_base.id]
    ds_cval, ds_cval_daily, ds_cval_itpd = slf_base.get_svr_smoothed_data(densified_ts_cval)

    # 5. Ewaluacja końcowa
    rmse_cval, nse_cval = slf_base.get_rmse_nse_values(ds_cval_itpd['daily_wse'], 'CrossVal', df_true, False)

    # Zwracamy triplet: c, prędkość, rmse
    return [c, slf_base.speed_ms, rmse_cval]
