import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from shapely.ops import substring
from datetime import timedelta
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import folium
import copy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import hydroeval as he
from typing import List, Dict, Any, Tuple, Optional
import statsmodels.api as sm
import sklearn.svm as svm

dahiti_in_situ_collections = {
    'germany': [5, 44, 46, 48, 49, 50, 51, 53],
    # 'poland': [35]
    'poland': [57],
    'italy': [29]

}


def get_optimum_lag(ts1, ts2, n):
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
    gdata_gdf = gdata_gdf.to_crs(river.gdf.crs)
    gdata_gdf_metric = gdata_gdf.to_crs(river.metrical_crs)
    river_buffer_metric = gpd.GeoSeries(river.simplified_river).buffer(buff)
    return gdata_gdf_metric[gdata_gdf.to_crs(river.metrical_crs).within(river_buffer_metric.iloc[0])]


def get_chainages_for_all_gauges(curr_gauges, river):
    chainages = []
    for index, row in curr_gauges.iterrows():
        chainages.append(river.get_chainage_of_point(row['X'], row['Y']))
    curr_gauges['chainage'] = chainages
    return curr_gauges


def get_list_of_stations_from_country(country, insitu):
    stations_data = []
    for insitu_id in dahiti_in_situ_collections[country]:
        curr_stations = insitu.list_collection(insitu_id)
        for station in curr_stations:
            stations_data.append(station)
    return stations_data


def filter_gauges_by_dt_freq_target(gauges_metadata, min_dt, max_dt, dahiti=True):
    if dahiti:
        return gauges_metadata.loc[(gauges_metadata['type'] == 'water_level') &
                                   (pd.to_datetime(gauges_metadata['min_date']) < pd.to_datetime(min_dt)) &
                                   (pd.to_datetime(gauges_metadata['max_date']) > pd.to_datetime(max_dt)) &
                                   # (gauges_metadata['data_sampling'] != 'daily') &
                                   (~gauges_metadata['target_name'].str.contains('see', na=False))]
    else:
        return gauges_metadata.loc[(gauges_metadata['type'] == 'water_level') &
                                   (pd.to_datetime(gauges_metadata['min_date']) < pd.to_datetime(min_dt)) &
                                   (pd.to_datetime(gauges_metadata['max_date']) > pd.to_datetime(max_dt))]


def juxtapose_gauge_to_densified_wl(gauge_meas_up, timeseries):
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
    if st1_chain < st2_chain:
        chain1, chain2 = st2_chain, st1_chain
    else:
        chain1, chain2 = st1_chain, st2_chain
    for dam in dams:
        if chain1 > dam[0] > chain2 or chain1 > dam[1] > chain2:
            return True
    return False


def is_tributary_between(st1_chain, st2_chain, tributary_chains):
    if st1_chain < st2_chain:
        chain1, chain2 = st2_chain, st1_chain
    else:
        chain1, chain2 = st1_chain, st2_chain
    for trib_chain in tributary_chains:
        if chain1 > trib_chain > chain2:
            return True
    return False


def get_rmse_between_two_ts(ts_true, ts_model):
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
    chain_diff = (neighbor_after.chainage - neighbor_before.chainage)/1000
    wl_diff = neighbor_after.wl['wse'].mean() - neighbor_before.wl['wse'].mean()
    return round(wl_diff/chain_diff, 3)


def get_regression_coeffs_from_df(regr_df, str1, str2):
    lin_model = LinearRegression().fit(regr_df[[str2]], regr_df[[str1]])
    r2 = r2_score(y_true=regr_df[[str1]], y_pred=lin_model.predict(regr_df[[str2]]))
    a, b, r2, commons = round(lin_model.coef_[0][0], 3), round(lin_model.intercept_[0], 3), round(r2, 3), len(regr_df)
    return a, b, r2, commons


def get_linear_regression_coeffs_btwn_stations(st_low_chain, st_high_chain, res_str='h'):
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
    Stosuje równanie regresji do obliczenia WSE na drugiej stacji.

    Args:
        known_wse: Znany poziom wody.
        known_station_id: ID stacji, dla której poziom wody jest znany.
        reg_row: Wiersz z ramki danych regresji jako słownik.

    Returns:
        Obliczony poziom wody na stacji docelowej.
    """
    a, b = reg_row['a'], reg_row['b']
    st1, st2 = reg_row['st1'], reg_row['st2']

    # Mamy WSE dla st1, liczymy dla st2 (y = ax + b)
    if known_station_id == st1:
        return (known_wse - b) / a if a != 0 else np.nan
    # Mamy WSE dla st2, liczymy dla st1 (x = (y - b) / a)
    elif known_station_id == st2:
        # Unikamy dzielenia przez zero, chociaż w regresji liniowej 'a' rzadko będzie zerem
        return a * known_wse + b
    else:
        raise ValueError("known_station_id nie pasuje do żadnej stacji w regresji.")


def find_regression(st_id1: int, st_id2: int, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Wyszukuje regresję między dwiema stacjami w DataFrame."""
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
    Przenosi stan wody (WSE) ze stacji startowej na docelową, używając regresji między
    sąsiadami i szukając obejść dla słabych połączeń.

    Args:
        start_wse: Początkowy stan wody.
        start_station_id: ID stacji początkowej.
        target_station_id: ID stacji docelowej (RS).
        regressions_df: DataFrame z parametrami regresji.
        all_stations_list: Uporządkowana lista wszystkich stacji na rzece.
        single_rmse_thres: Próg RMSE dla pojedynczego połączenia, powyżej którego jest ono "słabe".
        total_rmse_thres: Maksymalny dopuszczalny suma RMSE dla całej ścieżki.

    Returns:
        Krotka (final_wse, total_rmse, path_string) lub (NaN, NaN, "Komunikat błędu").
    """
    if start_station_id == target_station_id:
        return start_wse, 0, str(start_station_id)

    # Przygotowanie struktur danych
    station_pos_map = {st.id: i for i, st in enumerate(all_stations_list)}

    if start_station_id not in station_pos_map or target_station_id not in station_pos_map:
        return np.nan, np.nan, "Brak stacji w all_stations_list"

    current_station_id = start_station_id
    path = [current_station_id]
    wse_at_station = {current_station_id: start_wse}
    total_rmse = 0.0

    while current_station_id != target_station_id:
        current_pos = station_pos_map[current_station_id]
        target_pos = station_pos_map[target_station_id]

        direction = 1 if target_pos > current_pos else -1

        # 1. Identyfikacja bezpośredniego sąsiada w kierunku celu
        direct_neighbor_pos = current_pos + direction
        if not (0 <= direct_neighbor_pos < len(all_stations_list)):
            return np.nan, np.nan, "Brak ścieżki (koniec rzeki)"
        direct_neighbor_id = all_stations_list[direct_neighbor_pos].id

        direct_reg = find_regression(current_station_id, direct_neighbor_id, regressions_df)
        if not direct_reg:
            return np.nan, np.nan, f"Brak regresji dla sąsiada: {current_station_id}-{direct_neighbor_id}"

        # 2. Decyzja: czy iść bezpośrednio, czy szukać obejścia
        # Jeśli bezpośrednie połączenie jest dobre, wybieramy je
        if direct_reg['rmse'] < single_rmse_thres:
            best_move = {
                'start_id': current_station_id,
                'target_id': direct_neighbor_id,
                'reg': direct_reg,
                'type': 'direct'
            }
        else:
            # Połączenie jest słabe, szukamy alternatyw
            detour_options = []

            # Opcja 1: Obejście "w przód" (np. 2 -> 4)
            forward_jump_pos = current_pos + 2 * direction
            # Sprawdzenie, czy nie wyjdziemy poza zakres i czy nie przeskoczymy celu
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

            # Opcja 2: Obejście "z powrotem" (np. z 1 -> 3, będąc na 2)
            # Sprawdzenie, czy to nie jest pierwszy krok w całej ścieżce (warunek 2)
            if len(path) > 1:
                previous_station_id = path[-2]
                # Celem tego skoku jest ten sam sąsiad, do którego było słabe połączenie
                backward_jump_target_id = direct_neighbor_id
                backward_reg = find_regression(previous_station_id, backward_jump_target_id, regressions_df)
                if backward_reg:
                    detour_options.append({
                        'start_id': previous_station_id,
                        'target_id': backward_jump_target_id,
                        'reg': backward_reg,
                        'type': 'backward_jump'
                    })

            # Dodajemy oryginalne, słabe połączenie jako opcję rezerwową
            detour_options.append({
                'start_id': current_station_id,
                'target_id': direct_neighbor_id,
                'reg': direct_reg,
                'type': 'weak_direct'
            })

            # Wybieramy najlepszą opcję z dostępnych (najniższe RMSE)
            best_move = min(detour_options, key=lambda x: x['reg']['rmse'])

        # 3. Zastosowanie wybranego ruchu (best_move)
        move_reg = best_move['reg']
        move_start_id = best_move['start_id']
        move_target_id = best_move['target_id']

        # Obliczenie nowego WSE
        wse_for_calc = wse_at_station[move_start_id]
        new_wse = _apply_regression(wse_for_calc, move_start_id, move_reg)

        # Aktualizacja sumy RMSE
        total_rmse += move_reg['rmse']
        if total_rmse > total_rmse_thres:
            return np.nan, np.nan, "Przekroczono całkowity próg RMSE"

        # Aktualizacja stanu pętli
        wse_at_station[move_target_id] = new_wse
        current_station_id = move_target_id

        # Aktualizacja ścieżki jest zależna od typu ruchu
        if best_move['type'] == 'backward_jump':
            path.pop()  # Usuwamy ostatnią stację, bo "cofnęliśmy się"
            path.append(move_target_id)
        else:  # direct, weak_direct, forward_jump
            path.append(move_target_id)

    # 4. Zakończenie i zwrot wyników
    final_wse = wse_at_station[target_station_id]
    path_string = '->'.join(map(str, path))

    return final_wse, total_rmse, path_string


def calculate_path_for_row(row, target_station_id, regressions_df, all_stations_list, single_rmse_thres,
                           total_rmse_thres):
    """
    Funkcja-adapter do użycia z pandas.DataFrame.apply().
    Pobiera wiersz i stałe argumenty, a następnie wywołuje główną funkcję obliczeniową.
    """
    # Wyodrębnij wartości specyficzne dla wiersza
    start_wse = row['vs_wl']
    start_station_id = row['id_vs']

    # Wywołaj główną funkcję z pełnym zestawem argumentów
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
    df = df.copy(deep=True)
    alpha = 1 - confidence_level  # Poziom istotności (0.01)
    z_critical = stats.norm.ppf(1 - alpha / 2)  # Dla 99% to ok. 2.576
    df['Rolling_Mean'] = df['shifted_wl'].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).mean()
    df['Rolling_Std'] = df['shifted_wl'].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).std()
    df['Lower_Bound'] = df['Rolling_Mean'] - (z_critical * df['Rolling_Std'])
    df['Upper_Bound'] = df['Rolling_Mean'] + (z_critical * df['Rolling_Std'])
    df['Is_Outlier'] = (df['shifted_wl'] < df['Lower_Bound']) | \
                       (df['shifted_wl'] > df['Upper_Bound'])

    # print(f"\nŁączna liczba zidentyfikowanych outlierów: {df['Is_Outlier'].sum()}")
    if plot_outliers:
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['shifted_wl'], label='Stan wody', alpha=0.8, marker='.')
        plt.plot(df.index, df['Rolling_Mean'], label='Średnia Krocząca', color='orange', linestyle='--')
        plt.plot(df.index, df['Lower_Bound'], label='Dolny Limit Ufności (99%)', color='red', linestyle=':')
        plt.plot(df.index, df['Upper_Bound'], label='Górny Limit Ufności (99%)', color='red', linestyle=':')

        # Zaznaczanie outlierów
        outliers = df[df['Is_Outlier']]
        plt.scatter(outliers.index, outliers['shifted_wl'], color='purple', marker='o', s=50, zorder=5,
                    label='Zidentyfikowane Outliery')

        plt.title('Wykrywanie Outlierów w Znormalizowanym Szeregu Czasowym')
        plt.xlabel('Data')
        plt.ylabel('Znormalizowany Poziom Wody')
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
    return df.loc[(-df['Is_Outlier'])]


def plot_measurements_along_river(river, column_name, zoom_x, zoom_y, station_gdf=None, gauge_gdf=None, zoom_level=1):
    fig, ax = plt.subplots(figsize=(10, 8))
    if station_gdf is not None:
        station_gdf.plot(
            column_name,
            cmap='viridis',
            legend_kwds={'label': column_name, 'shrink': 0.7},
            ax=ax,
            legend=True,
            markersize=50
        )
        for idx, row in station_gdf.iterrows():
            x, y = row.geometry.x, row.geometry.y
            ax.annotate(str(row['id']), (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8,
                        color='black')

    if gauge_gdf is not None:
        gauge_gdf.plot(
            column_name,
            marker='D',  # Zmieniony marker dla odróżnienia od stacji
            cmap='viridis',
            legend_kwds={'label': column_name, 'shrink': 0.7},
            ax=ax,
            legend=True,
            markersize=50
        )
        for idx, row in gauge_gdf.iterrows():
            x, y = row.geometry.x, row.geometry.y
            ax.annotate(str(row['id']), (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8,
                        color='black')

    custom_colors = {1: 'blue', 2: 'steelblue', 3: 'aqua', 4: 'red', 5: 'gray', 6: 'black'}
    river.gdf['colors'] = river.gdf['type'].map(custom_colors)
    river.gdf.plot(color=river.gdf['colors'], ax=ax)

    ax.set_title(f"VS and gauges along {river.name}")
    ax.set_ylabel('Latitude (Y)')
    ax.set_xlabel('Longitude (X)')
    ax.set_xlim(zoom_x - zoom_level, zoom_x + zoom_level)
    ax.set_ylim(zoom_y - zoom_level, zoom_y + zoom_level)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(block=True)


def plot_interactive_measurements_along_river(river, column_name='length', vs_stations=None, gauge_stations=None):
    if river.gdf.crs is None:
        print("Ostrzeżenie: CRS rzeki nie jest ustawiony. Ustawiam na EPSG:4326 (WGS84).")
        river.gdf = river.gdf.set_crs("EPSG:4326", allow_override=True)
    custom_colors = {
        1: 'blue',
        2: 'steelblue',
        3: 'aqua',
        4: 'red',
        5: 'gray',
        6: 'black'
    }

    river_centroids = river.gdf.geometry.centroid
    center_lat = river_centroids.y.mean()
    center_lon = river_centroids.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    for r_type, color in custom_colors.items():
        type_segment = river.gdf[river.gdf['type'] == r_type]
        if not type_segment.empty:
            type_segment.explore(
                m=m,
                color=color,  # Użyj przypisanego koloru
                tooltip="type",  # Możesz wyświetlić typ w dymku
                popup="river_name",  # Lub nazwę rzeki
                style_kwds={"weight": 3, "opacity": 0.7},
                name=f"Rzeka - Typ {r_type}",  # Nazwa warstwy w kontroli warstw
                highlight=True  # Podświetlanie po najechaniu myszką
            )

    if vs_stations is not None:
        station_gdf = create_gdf_from_metadata(vs_stations)
        if station_gdf.crs is None:
            station_gdf = station_gdf.set_crs("EPSG:4326", allow_override=True)
        station_gdf.explore(
            m=m,  # Dodaj do istniejącej mapy
            column=column_name,  # Kolumna z wartościami do kolorowania markerów
            cmap='viridis',
            marker_type="circle_marker",  # Domyślny marker to koło
            style_kwds={"fillOpacity": 0.8, "radius": 8},  # Styl markerów
            tooltip=[column_name, 'id'],  # Co ma się wyświetlać w tooltipie
            popup=True,  # Włączanie pop-upów
            name="virtual station",  # Nazwa warstwy na mapie
            legend=True,  # Wyświetlanie legendy dla tej warstwy
            legend_kwds={'caption': column_name}
        )

    if gauge_stations is not None:
        gauge_gdf = create_gdf_from_metadata(gauge_stations, True)
        if gauge_gdf.crs is None:
            gauge_gdf = gauge_gdf.set_crs("EPSG:4326", allow_override=True)

        gauge_gdf.explore(
            m=m,
            marker_type="marker",  # Wodowskazy jako piny
            marker_kwds={
                "radius": 10,  # Rozmiar pina (np. większy niż domyślny)
                "color": "darkgrey",  # Kolor ramki pina (opcjonalnie)
                "fill": True,  # Wypełnienie pina
                "fillOpacity": 0.8,  # Przezroczystość wypełnienia
            },
            tooltip=[column_name, 'id'],
            popup=True,
            name="gauge station",
            legend=True,
            legend_kwds={'caption': column_name}
        )
    folium.LayerControl().add_to(m)
    m.save(f"mapa_{river.name}.html")
    print(f'Saved map to: {os.getcwd()}mapa_{river.name}.html')


def plot_dams_with_vs_chains(loaded_stations, current_river):
    vs_gdf = create_gdf_from_metadata(loaded_stations, is_gauge=False)
    chains = []
    for vs in loaded_stations:
        chains.append(current_river.get_chainage_of_point(vs.x, vs.y))
    vs_gdf['new_chains'] = chains
    fig, ax = plt.subplots()
    custom_colors = {1: 'blue', 2: 'steelblue', 3: 'aqua', 4: 'red', 5: 'gray', 6: 'black'}
    current_river.gdf['colors'] = current_river.gdf['type'].map(custom_colors)
    current_river.gdf.plot(color=current_river.gdf['colors'], ax=ax)
    vs_gdf.plot('new_chains', cmap='viridis', legend_kwds={'label': 'new_chains', 'shrink': 0.7}, ax=ax, legend=True,
                markersize=50)
    for idx, row in vs_gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        ax.annotate(str(row['id']), (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8,
                    color='black')

    ax.set_ylabel('Latitude (Y)')
    ax.set_xlabel('Longitude (X)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    ax.legend()
    plt.show(block=True)


def plot_river_profile_with_stations(loaded_stations, loaded_gauges, river):
    fig, ax = plt.subplots()
    chains, mean_wls = [x.chainage / 1000 for x in loaded_stations], [x.wl.wse.mean() for x in loaded_stations]
    gauge_chains, gauge_mean_wls = [x.chainage / 1000 for x in loaded_gauges.values()], [x.wl_df.stage.mean() for x in
                                                                                         loaded_gauges.values()]
    ymin, ymax = min([x for x in mean_wls if x > 0]), max([x for x in mean_wls if x > 0])
    ax.scatter(chains, mean_wls, label='VS')
    ax.scatter(gauge_chains, gauge_mean_wls, label='Gauge')
    ax.vlines([x / 1000 for group in river.dams for x in group], ymin=ymin, ymax=ymax, color='red')
    for i in range(len(chains)):
        x, y = chains[i], mean_wls[i]
        ax.annotate(loaded_stations[i].id, (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8,
                    color='black')
    ax.set_ylabel('Water level [m]')
    ax.set_xlabel('River chainage [km]')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    ax.legend()
    plt.show(block=True)


class GaugeStation:
    def __init__(self, x, y, g_id, riv, chain, unit, sampling):
        self.x = x
        self.y = y
        self.id = g_id
        self.river = riv
        self.chainage = chain
        self.unit = unit
        self.sampling = sampling
        self.wl_df = pd.DataFrame()

    def __repr__(self):
        return f'Gauge {self.id} on {self.chainage} km of the {self.river} with {len(self.wl_df)} measurements'

    def upload_wl(self, wl_df):
        self.wl_df = wl_df
        self.wl_df = self.wl_df.set_index(pd.to_datetime(self.wl_df['date']))
        self.wl_df = self.wl_df.sort_index()
        if self.unit == 'cm':
            self.wl_df['stage'] = self.wl_df['stage']
            self.unit = 'm'
        if self.sampling in ['hourly', 'h', '15-minute', '10-minute']:
            resample_str = 'H'
        else:
            resample_str = 'D'
        self.wl_df = self.wl_df[['stage']].resample(resample_str).mean()


class VirtualStation:
    def __init__(self, vs_id, x, y):
        self.id = vs_id
        self.x, self.y = x, y
        self.river, self.chainage = None, None
        self.wl, self.swot_wl, self.geoid, self.swot_mmxo_rbias, self.slope_correction = None, None, None, None, None
        self.neigh_g_up, self.neigh_g_up_chain, self.neigh_g_dn, self.neigh_g_dn_chain = None, None, None, None
        self.closest_gauge = None
        self.juxtaposed_wl = None
        self.mean_g_wl, self.mean_vs_wl = None, None
        self.slope = None

    def __repr__(self):
        return f'Virtual station ID: {self.id}, chainage: {self.chainage} with {len(self.wl)} measurements'

    def is_away_from_river(self, riv_object, distance):
        riv_series = gpd.GeoSeries(riv_object.simplified_river, crs=riv_object.metrical_crs)
        vs_series = gpd.GeoSeries(Point(self.x, self.y), crs=4326)
        dist = vs_series.to_crs(riv_series.crs).distance(riv_series)
        return dist.values[0] > distance

    def get_water_levels(self, dahiti):
        try:
            try:
                wl_data = dahiti.download_water_level(self.id, parameters=['mission'])
            except:
                return None
            if len(wl_data['data']) == 0:
                return None
        except Exception as e:
            print(e)
            self.wl = 'error'
            return None
        self.river = wl_data['target_name']
        self.geoid = wl_data['geoid']
        self.swot_mmxo_rbias = wl_data['SWOT_MMXO_rbias']
        self.slope_correction = wl_data['WSS_correction_applied'].replace('yes', 'True').replace('no', 'False')
        self.wl = pd.DataFrame(wl_data['data'])
        if self.wl['mission'].str.contains('SWOT').any():
            self.swot_wl = self.wl[self.wl['mission'].str.contains('SWOT', na=False)]
        else:
            self.swot_wl = pd.DataFrame(columns=self.wl.columns)
        self.wl['datetime'] = pd.to_datetime(self.wl['datetime'])
        self.wl = self.wl.set_index(self.wl['datetime'])

    def time_filter(self, t1, t2):
        self.wl = self.wl.loc[(self.wl['datetime'] > t1) & (self.wl['datetime'] < t2)]
        self.swot_wl = self.swot_wl.loc[
            (pd.to_datetime(self.swot_wl['datetime']) > t1) & (pd.to_datetime(self.swot_wl['datetime']) < t2)]

    def upload_chainage(self, chainage):
        self.chainage = chainage

    def find_closest_gauge_and_chain(self, gauges_chains):
        gauges_chains = pd.DataFrame([[g.id, g.chainage] for g in gauges_chains.values()], columns=['id', 'chainage'])
        if self.chainage >= gauges_chains['chainage'].max():
            up_id, up_chain = None, None
            dn_id, dn_chain = gauges_chains[gauges_chains['chainage'] < self.chainage].sort_values(
                by='chainage', ascending=False).iloc[0][['id', 'chainage']]
        elif self.chainage <= gauges_chains['chainage'].min():
            up_id, up_chain = gauges_chains[gauges_chains['chainage'] > self.chainage].sort_values(
                by='chainage').iloc[0][['id', 'chainage']]
            dn_id, dn_chain = None, None
        else:
            up_id, up_chain = gauges_chains[gauges_chains['chainage'] > self.chainage].sort_values(
                by='chainage').iloc[0][['id', 'chainage']]
            dn_id, dn_chain = gauges_chains[gauges_chains['chainage'] < self.chainage].sort_values(
                by='chainage', ascending=False).iloc[0][['id', 'chainage']]

        self.neigh_g_up, self.neigh_g_up_chain = up_id, up_chain
        self.neigh_g_dn, self.neigh_g_dn_chain = dn_id, dn_chain

    def get_juxtaposed_vs_and_gauge_meas(self, gauge_meas_up, gauge_meas_down, gdata_sampling, velocity=None):
        hours_to_juxtapose = 12
        juxtaposed_columns = ['id_vs', 'vs_chain', 'dt', 'mission', 'gauge_up', 'dist_up', 'gauge_down', 'dist_down',
                              'lag',
                              'vs_wl', 'g_wl', 'uncertainty', 'g_anom', 'vs_anom', 'bias']
        juxtaposed_data = []
        for index, row in self.wl.iterrows():
            vs_wl, vs_dt = row[['wse', 'datetime']]
            vs_dt_prev = vs_dt - pd.to_timedelta('5 days')
            try:
                dist_up, dist_dn = abs(self.neigh_g_up_chain - self.chainage), abs(self.neigh_g_dn_chain - self.chainage)
                self.closest_gauge = 'up' if dist_up < dist_dn else 'dn'
            except TypeError:
                self.closest_gauge = 'up' if type(gauge_meas_up) == pd.DataFrame else 'dn'
            closest_gdata = gauge_meas_up if self.closest_gauge == 'up' else gauge_meas_down
            closest_chain = self.neigh_g_up_chain if self.closest_gauge == 'up' else self.neigh_g_dn_chain
            if gdata_sampling == 'daily' or velocity is not None or gauge_meas_down is None or gauge_meas_up is None:
                vel = 1 if velocity is None else velocity
                shift = pd.to_timedelta((closest_chain - self.chainage) / vel, unit='s')
                gauge_time = (vs_dt - shift).round('H')
                final_lag = round(shift.total_seconds() / 3600)
            else:
                ts_up = gauge_meas_up['stage'].loc[(gauge_meas_up.index > vs_dt_prev) & (gauge_meas_up.index < vs_dt)]
                ts_dn = gauge_meas_down['stage'].loc[
                    (gauge_meas_down.index > vs_dt_prev) & (gauge_meas_down.index < vs_dt)]

                lag, corr = get_optimum_lag(ts_dn, ts_up, 50)
                ratio = (closest_chain - self.chainage) / (self.neigh_g_up_chain - self.neigh_g_dn_chain)
                final_lag = lag * ratio

                gauge_time = (vs_dt - pd.to_timedelta(f'{final_lag} hours')).round('H')
            gauge_wl = np.nan
            for i in range(hours_to_juxtapose):
                try:
                    gauge_wl = \
                        closest_gdata['stage'][
                            closest_gdata.index == gauge_time + pd.to_timedelta(f'{i} hours')].values[0]
                    if not np.isnan(gauge_wl):
                        break
                except IndexError:
                    try:
                        gauge_wl = closest_gdata['stage'][
                            closest_gdata.index == gauge_time + pd.to_timedelta(f'-{i} hours')].values[0]
                        if not np.isnan(gauge_wl):
                            break
                    except IndexError:
                        continue
            print(self.id, vs_dt, vs_wl, gauge_wl, final_lag)
            juxtaposed_data.append(
                [self.id, self.chainage, vs_dt, row['mission'], self.neigh_g_up, self.neigh_g_up_chain, self.neigh_g_dn,
                 self.neigh_g_dn_chain, final_lag, vs_wl, gauge_wl, row['wse_u']])
        curr_results = pd.DataFrame(juxtaposed_data, columns=juxtaposed_columns[:-3])
        mean_g, mean_vs = curr_results['g_wl'].mean(), curr_results['vs_wl'].mean()
        curr_results['g_anom'] = curr_results['g_wl'] - mean_g
        curr_results['vs_anom'] = curr_results['vs_wl'] - mean_vs
        curr_results['bias'] = abs(curr_results['vs_anom'] - curr_results['g_anom'])
        self.mean_g_wl, self.mean_vs_wl = mean_g, mean_vs
        self.juxtaposed_wl = curr_results

    def plot_anomalies(self):
        fig, ax = plt.subplots()
        ax.plot(self.juxtaposed_wl['dt'], self.juxtaposed_wl['vs_anom'], label=f'VS {self.id} anom')
        ax.plot(self.juxtaposed_wl['dt'], self.juxtaposed_wl['g_anom'], label='Gauge anom')
        ax.set_ylabel('Water level anomaly [m]')
        ax.set_xlabel('Time')
        plt.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.show(block=True)

    def get_daily_linear_interpolated_wl_of_single_vs(self, just_swot=False):
        if just_swot:
            vs_wl_to_corr = self.wl.loc[self.wl['mission'].str.contains('SWOT', na=False)].copy()
        else:
            vs_wl_to_corr = self.wl.copy()
        return vs_wl_to_corr['wse'].resample('D').mean().interpolate(method='linear')


class DensificationStation(VirtualStation):
    def __init__(self, vs_object, buffer, speed_ms, itpd_method, **kwargs):
        super().__init__(vs_object.id, vs_object.x, vs_object.y)
        self.cval_buff = 0.01
        self.__dict__.update(vs_object.__dict__)
        self.buffer, self.upstream_adjacent_vs = buffer, None
        self.regressions_df = None
        self.itpd_method = itpd_method
        self.densified_wl = None
        self.speed_ms = speed_ms
        self.closest_in_situ_daily_wl = None
        self.daily_wl, self.interpolated_wl, self.daily_gauge_wl, self.single_VS_itpd = None, None, None, None
        self.dist_weighted_daily_wl, self.dist_weighted_daily_wl_itpd, self.spline_itpd_wl = None, None, None
        self.normalized_ts, self.normalized_ts_daily, self.normalized_ts_itpd = None, None, None
        self.densified_ts, self.densified_daily, self.densified_itpd = None, None, None
        self.loess_filtered_ts, self.svr_ts, self.svr_itpd = None, None, None
        self.slopes_dict, self.c = None, None
        self.rmse_thres, self.single_rmse_thres = None, None
        self.rmse, self.nse = None, None

    def __repr__(self):
        return f'Densification station ID: {self.id}, chainage: {self.chainage} with' \
               f' {len(self.upstream_adjacent_vs)} VS within buffer'

    def get_upstream_adjacent_vs(self, vs_list):
        selected_list = []
        for vs in vs_list:
            if self.chainage - self.buffer * 1000 <= vs.chainage < self.chainage + self.buffer * 1000:
                selected_list.append(copy.deepcopy(vs))
        self.upstream_adjacent_vs = sorted(selected_list, key=lambda k: k.chainage)

    def filter_stations_by_corr_amp_dams_tribs_other(self, corr_thres, amp_thres, dams, tributary_reaches,
                                                     other_reaches, just_swot):
        self.filter_upstream_stations_by_correlation(corr_thres, just_swot)
        self.filter_upstream_stations_by_wl_amplitude(amp_thres)
        self.filter_upstream_stations_by_dams_and_tributaries(dams, tributary_reaches)
        self.upstream_adjacent_vs = [x for x in self.upstream_adjacent_vs if x.id not in other_reaches]

    def filter_upstream_stations_by_correlation(self, corr_thres, just_swot=False):
        stations = []
        vs_to_corr1 = self.get_daily_linear_interpolated_wl_of_single_vs(just_swot)
        for vs in self.upstream_adjacent_vs:
            vs_to_corr2 = vs.get_daily_linear_interpolated_wl_of_single_vs(just_swot)
            if len(vs_to_corr2) > 0 and vs.id != self.id:
                correlation = vs_to_corr1.corr(vs_to_corr2)
                if correlation > corr_thres:
                    stations.append(vs)
        self.upstream_adjacent_vs = stations

    def filter_upstream_stations_by_wl_amplitude(self, thres):
        stations = []
        ds_amp = self.wl['wse'].max() - self.wl['wse'].min()
        for vs in self.upstream_adjacent_vs:
            vs_amp = vs.wl['wse'].max() - vs.wl['wse'].min()
            if ds_amp + thres * ds_amp > vs_amp > ds_amp - thres * ds_amp:
                stations.append(vs)
        self.upstream_adjacent_vs = stations

    def filter_upstream_stations_by_dams_and_tributaries(self, dams, tributary_reaches):
        stations = []
        for vs in self.upstream_adjacent_vs:
            cond1 = not is_dam_between(self.chainage, vs.chainage, dams)
            cond2 = not is_tributary_between(self.chainage, vs.chainage, tributary_reaches)
            if cond1 and cond2:
                stations.append(vs)
        self.upstream_adjacent_vs = stations

    def filter_stations_only_with_swot(self):
        vs_with_swot_data = []
        for vs in self.upstream_adjacent_vs:
            if len(vs.juxtaposed_wl.loc[vs.juxtaposed_wl['mission'].str.contains('SWOT', na=False)]) > 0:
                vs_with_swot_data.append(vs)
        self.upstream_adjacent_vs = vs_with_swot_data

    def is_ds_empty_or_at_edge(self):
        cond1 = len(self.upstream_adjacent_vs) == 0
        cond2 = len([x for x in self.upstream_adjacent_vs if x.chainage > self.chainage]) == 0
        cond3 = len([x for x in self.upstream_adjacent_vs if x.chainage < self.chainage]) == 0
        if cond1 or cond2 or cond3:
            return True

    def get_slope_of_all_vs(self):
        for vs in [self] + self.upstream_adjacent_vs:
            neigh_bef, neigh_aft = get_vs_neighbors(vs, self.upstream_adjacent_vs)
            vs.slope = get_slope(neigh_bef, neigh_aft)

    def get_depths_of_all_vs(self, bottom_thres=0.1):
        for vs in [self] + self.upstream_adjacent_vs:
            curr_amplitude = vs.wl['wse'].max() - vs.wl['wse'].min()
            bottom_height = vs.wl['wse'].min() - bottom_thres * curr_amplitude
            vs.waterdepths = vs.wl['wse'] - bottom_height

    def get_mean_slope_to_vs(self, vs_id):
        curr_vs = [x for x in self.upstream_adjacent_vs if x.id == vs_id][0]
        curr_vs_wl, curr_vs_chain = curr_vs.wl['wse'].mean(), curr_vs.chainage
        ds_wl = self.wl['wse'].mean()
        chain_diff = abs(self.chainage - curr_vs_chain) / 1000
        return round(abs(curr_vs_wl - ds_wl) / chain_diff, 3)

    def get_mean_slopes_dict(self):
        slopes_dict = {}
        for vs in self.upstream_adjacent_vs:
            slopes_dict[vs.id] = self.get_mean_slope_to_vs(vs.id)
        return slopes_dict

    def get_bottom_heights_dict(self, bottom_thres=0.1):
        bottom_heights = {}
        for vs in [self] + self.upstream_adjacent_vs:
            curr_amplitude = vs.wl['wse'].max() - vs.wl['wse'].min()
            bottom_heights[vs.id] = vs.wl['wse'].min() - bottom_thres * curr_amplitude
        return bottom_heights

    def get_vs_regressions_df_extended(self, res_str='h'):
        """
        Tworzy ramkę danych regresji dla bezpośrednich sąsiadów i obejść (2 kroki).
        """
        regressions = []

        # Sortujemy wszystkie stacje od najniższego do najwyższego chainage
        all_stations = sorted([self] + self.upstream_adjacent_vs, key=lambda x: x.chainage)

        # # Przechowujemy mapowanie id na obiekt stacji, dla ułatwienia dostępu
        # station_obj_map = {st.id: st for st in all_stations}
        # station_pos_map = {st.id: i for i, st in enumerate(all_stations)}

        # Pętla dla sąsiadów (1 krok)
        for i in range(len(all_stations) - 1):
            vs1 = all_stations[i]
            vs2 = all_stations[i + 1]

            # Ustalenie, która stacja jest w dół rzeki, a która w górę
            if vs1.chainage < vs2.chainage:
                st_low_chain = vs1
                st_high_chain = vs2
            else:
                st_low_chain = vs2
                st_high_chain = vs1

            a, b, r2, rmse, data_len = get_linear_regression_coeffs_btwn_stations(st_low_chain, st_high_chain, res_str)

            if a is not None:
                regressions.append({
                    'st1': st_low_chain.id, 'st2': st_high_chain.id,
                    'st1_chain': st_low_chain.chainage, 'st2_chain': st_high_chain.chainage,
                    'a': a, 'b': b, 'r2': r2, 'rmse': rmse, 'data_len': data_len,
                    'steps': 1
                })

        # Pętla dla obejść (2 kroki)
        for i in range(len(all_stations) - 2):
            vs1 = all_stations[i]
            vs3 = all_stations[i + 2]

            # Ustalenie, która stacja jest w dół rzeki, a która w górę
            if vs1.chainage < vs3.chainage:
                st_low_chain = vs1
                st_high_chain = vs3
            else:
                st_low_chain = vs3
                st_high_chain = vs1

            a, b, r2, rmse, data_len = get_linear_regression_coeffs_btwn_stations(st_low_chain, st_high_chain, res_str)

            if a is not None:
                regressions.append({
                    'st1': st_low_chain.id, 'st2': st_high_chain.id,
                    'st1_chain': st_low_chain.chainage, 'st2_chain': st_high_chain.chainage,
                    'a': a, 'b': b, 'r2': r2, 'rmse': rmse, 'data_len': data_len,
                    'steps': 2
                })

        # Tworzymy ramkę danych i usuwamy duplikaty
        self.regressions_df = pd.DataFrame(regressions).drop_duplicates(subset=['st1', 'st2'])
        return self.regressions_df

    def get_used_regressions(self):
        used_regressions = []
        for path in self.densified_ts['regr_path'].unique():
            if type(path) != str:
                continue
            iself = [int(round(float(x))) for x in path.split('->')]
            for i, vs_id in enumerate(iself[:-1]):
                vs2_id = iself[i + 1]
                found = False
                try:
                    x = self.regressions_df.loc[
                        (self.regressions_df['st1'] == vs_id) & (self.regressions_df['st2'] == vs2_id)
                        ].iloc[0].values
                    used_regressions.append([vs_id, vs2_id, x[-3], x[-1]])
                    found = True
                except IndexError:
                    pass  # Błąd, spróbuj następnej opcji

                if not found:
                    try:
                        x = self.regressions_df.loc[
                            (self.regressions_df['st2'] == vs_id) & (self.regressions_df['st1'] == vs2_id)
                            ].iloc[0].values
                        used_regressions.append([vs_id, vs2_id, x[-3], x[-1]])
                        found = True
                    except IndexError:
                        pass  # Błąd, spróbuj ostatniej opcji

                if not found:
                    print(vs_id, vs2_id, '!!!!!!!!!!!')
                    vs1 = [x for x in [self] + self.upstream_adjacent_vs if x.id == vs_id][0]
                    vs2 = [x for x in [self] + self.upstream_adjacent_vs if x.id == vs2_id][0]
                    a, b, r2, rmse, data_len = get_linear_regression_coeffs_btwn_stations(vs1, vs2, 'h')
                    used_regressions.append([vs_id, vs2_id, rmse, 2])
        df_used_regressions = pd.DataFrame(used_regressions, columns=['st1', 'st2', 'rmse', 'steps'])
        df_used_regressions = df_used_regressions.drop_duplicates()

        merged_df = self.regressions_df.loc[self.regressions_df['steps'] == 1].merge(
            df_used_regressions,
            on=['rmse'],
            how='left',
            indicator=True
        )
        list_stations = set(list(df_used_regressions['st1'].values) + list(df_used_regressions['st2'].values))
        result_df = merged_df[(merged_df['_merge'] == 'left_only') &
                              (merged_df['st1_x'].isin(list_stations)) &
                              (merged_df['st2_x'].isin(list_stations))
                              ].drop(columns='_merge')
        result_df = result_df.rename(columns={'st1_x': 'st1', 'st2_x': 'st2', 'steps_x': 'steps'})
        result_df = result_df[df_used_regressions.columns]
        result_df['steps'] = 3
        return pd.concat([df_used_regressions, result_df])

    def clip_river_to_vs_section(self, current_river):
        river_line = gpd.GeoSeries(current_river.simplified_river, crs=current_river.metrical_crs)
        vs_upstr, vs_dwnstr = self.upstream_adjacent_vs[0], self.upstream_adjacent_vs[-1]
        points = gpd.GeoSeries([Point(vs_upstr.x, vs_upstr.y), Point(vs_dwnstr.x, vs_dwnstr.y)], crs=4326)
        line_crs_y = river_line.to_crs(points.crs)
        point1_proj = line_crs_y.iloc[0].project(points.iloc[0])
        point2_proj = line_crs_y.iloc[0].project(points.iloc[1])
        start = min(point1_proj, point2_proj)
        end = max(point1_proj, point2_proj)
        clipped_line = substring(line_crs_y.iloc[0], start, end)
        return gpd.GeoSeries(clipped_line)

    def interpolate(self, ts):
        return ts.interpolate(method=self.itpd_method)

    # def get_densified_wl(self):
    #     multi_vs_wl_df = self.juxtaposed_wl.copy()
    #     multi_vs_wl_df['time_diff'] = pd.to_timedelta(0, unit='s')
    #     multi_vs_wl_df['shifted_time'] = pd.to_datetime(multi_vs_wl_df['dt']) + multi_vs_wl_df['time_diff']
    #     multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['vs_wl']
    #     multi_vs_wl_df['vs_chain'] = self.chainage
    #     for vs in self.upstream_adjacent_vs:
    #         cond1 = len(vs.juxtaposed_wl) == 0
    #         cond2 = vs.id == self.id
    #         if cond1 or cond2:
    #             continue
    #         curr_vs_wl_df = vs.juxtaposed_wl.copy()
    #         curr_vs_wl_df['time_diff'] = pd.to_timedelta((vs.chainage - self.chainage) / self.speed_ms, unit='s')
    #         curr_vs_wl_df['shifted_time'] = pd.to_datetime(curr_vs_wl_df['dt']) + curr_vs_wl_df['time_diff']
    #         curr_vs_wl_df = curr_vs_wl_df.sort_values(by='shifted_time')
    #         vs_bias = curr_vs_wl_df['vs_wl'].loc[(self.juxtaposed_wl['dt'].min() < curr_vs_wl_df['dt']) & (
    #                 curr_vs_wl_df['dt'] < self.juxtaposed_wl['dt'].max())].mean() - self.juxtaposed_wl[
    #                       'vs_wl'].mean()
    #         curr_vs_wl_df['shifted_wl'] = curr_vs_wl_df['vs_wl'] - vs_bias
    #         curr_vs_wl_df['vs_chain'] = vs.chainage
    #         multi_vs_wl_df = pd.concat(
    #             [multi_vs_wl_df.dropna(axis=1, how='all'), curr_vs_wl_df.dropna(axis=1, how='all')])
    #     multi_vs_wl_df = multi_vs_wl_df.set_index(pd.to_datetime(multi_vs_wl_df['shifted_time']))
    #     self.densified_wl = multi_vs_wl_df.sort_index()

    # def get_densified_wl_by_norm(self, ts, norm_method):
    #     multi_vs_wl_df = ts[self.upstream_adjacent_vs[0].juxtaposed_wl.columns].copy()
    #     multi_vs_wl_df['time_diff'] = pd.to_timedelta(0, unit='s')
    #     multi_vs_wl_df['shifted_time'] = pd.to_datetime(multi_vs_wl_df['dt']) + multi_vs_wl_df['time_diff']
    #
    #     perc1, perc2 = .03, .85  # minmax or percentile norm
    #     vs_median = multi_vs_wl_df['vs_wl'].median()
    #     vs_iqr = multi_vs_wl_df['vs_wl'].quantile(0.75) - multi_vs_wl_df['vs_wl'].quantile(0.25)
    #     vs_min, vs_max = multi_vs_wl_df['vs_wl'].quantile(perc1), multi_vs_wl_df['vs_wl'].quantile(perc2)
    #     scaler_standard = StandardScaler()
    #
    #     if norm_method == 'standard':
    #         multi_vs_wl_df['normalized_wl'] = \
    #             pd.DataFrame(scaler_standard.fit_transform(multi_vs_wl_df[['vs_wl']]), index=multi_vs_wl_df.index,
    #                          columns=['vs_wl'])['vs_wl']
    #     elif norm_method in ['minmax', 'minmax_percentile']:
    #         multi_vs_wl_df['normalized_wl'] = (multi_vs_wl_df['vs_wl'] - vs_min) / (vs_max - vs_min)
    #     elif norm_method == 'median':
    #         multi_vs_wl_df['normalized_wl'] = (multi_vs_wl_df['vs_wl'] - vs_median) / vs_iqr
    #
    #     multi_vs_wl_df['vs_chain'] = self.chainage
    #     for vs in self.upstream_adjacent_vs:
    #         cond1 = len(vs.juxtaposed_wl) == 0
    #         cond2 = vs.id == self.id
    #         if cond1 or cond2:
    #             continue
    #         curr_vs_wl_df = vs.juxtaposed_wl.copy()
    #         curr_vs_wl_df['time_diff'] = pd.to_timedelta((vs.chainage - self.chainage) / self.speed_ms, unit='s')
    #         curr_vs_wl_df['shifted_time'] = pd.to_datetime(curr_vs_wl_df['dt']) + curr_vs_wl_df['time_diff']
    #         curr_vs_wl_df = curr_vs_wl_df.sort_values(by='shifted_time')
    #         if norm_method == 'standard':
    #             curr_scaler = StandardScaler()
    #             curr_vs_wl_df['normalized_wl'] = \
    #                 pd.DataFrame(curr_scaler.fit_transform(curr_vs_wl_df[['vs_wl']]), index=curr_vs_wl_df.index,
    #                              columns=['vs_wl'])['vs_wl']
    #         elif norm_method in ['minmax', 'minmax_percentile']:
    #             # curr_min, curr_max = curr_vs_wl_df['vs_wl'].min(), curr_vs_wl_df['vs_wl'].max()
    #             curr_min, curr_max = curr_vs_wl_df['vs_wl'].quantile(perc1), curr_vs_wl_df['vs_wl'].quantile(perc2)
    #             curr_vs_wl_df['normalized_wl'] = (curr_vs_wl_df['vs_wl'] - curr_min) / (curr_max - curr_min)
    #         elif norm_method == 'median':
    #             curr_median = curr_vs_wl_df['vs_wl'].median()
    #             curr_iqr = curr_vs_wl_df['vs_wl'].quantile(0.75) - curr_vs_wl_df['vs_wl'].quantile(0.25)
    #             curr_vs_wl_df['normalized_wl'] = (curr_vs_wl_df['vs_wl'] - curr_median) / curr_iqr
    #
    #         curr_vs_wl_df['vs_chain'] = vs.chainage
    #         multi_vs_wl_df = pd.concat(
    #             [multi_vs_wl_df.dropna(axis=1, how='all'), curr_vs_wl_df.dropna(axis=1, how='all')])
    #     multi_vs_wl_df = multi_vs_wl_df.set_index(pd.to_datetime(multi_vs_wl_df['shifted_time']))
    #     if norm_method == 'standard':
    #         multi_vs_wl_df['shifted_wl'] = \
    #             pd.DataFrame(scaler_standard.inverse_transform(multi_vs_wl_df[['normalized_wl']]), columns=['vs_wl'],
    #                          index=multi_vs_wl_df.index)['vs_wl']
    #     elif norm_method in ['minmax', 'minmax_percentile']:
    #         multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['normalized_wl'] * (vs_max - vs_min) + vs_min
    #     elif norm_method == 'median':
    #         multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['normalized_wl'] * vs_iqr + vs_median
    #     self.normalized_ts = multi_vs_wl_df.sort_index()

    def get_densified_wl_by_regressions(self, without_rs=False, rmse_thres=0.5, single_rmse_thres=0.2,
                                        res_str='h'):
        self.get_vs_regressions_df_extended(res_str)
        self.rmse_thres = rmse_thres
        self.single_rmse_thres = single_rmse_thres
        if without_rs:
            # print(without_RS, '12345!')
            multi_vs_wl_df = pd.DataFrame(columns=self.juxtaposed_wl.columns)
        else:
            multi_vs_wl_df = self.juxtaposed_wl.copy()
            # multi_vs_wl_df['time_diff'] = pd.to_timedelta(0, unit='s')
            # multi_vs_wl_df['shifted_time'] = pd.to_datetime(multi_vs_wl_df['dt']) + multi_vs_wl_df['time_diff']
            multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['vs_wl']
            multi_vs_wl_df['rmse_sum'] = 0
            multi_vs_wl_df['vs_chain'] = self.chainage
        for vs in self.upstream_adjacent_vs:
            cond1 = len(vs.juxtaposed_wl) == 0
            cond2 = vs.id == self.id
            if cond1 or cond2:
                continue
            curr_vs_wl_df = vs.juxtaposed_wl.copy()
            # curr_vs_wl_df['time_diff'] = pd.to_timedelta((vs.chainage - self.chainage) / self.speed_ms, unit='s')
            # curr_vs_wl_df['shifted_time'] = pd.to_datetime(curr_vs_wl_df['dt']) + curr_vs_wl_df['time_diff']
            # curr_vs_wl_df = curr_vs_wl_df.sort_values(by='shifted_time')
            vs_list = sorted([self] + self.upstream_adjacent_vs, key=lambda x: x.chainage)
            curr_vs_wl_df[['shifted_wl', 'rmse_sum', 'regr_path']] = curr_vs_wl_df.apply(calculate_path_for_row, axis=1,
                                                                                         args=(
                                                                                             self.id,
                                                                                             self.regressions_df,
                                                                                             vs_list,
                                                                                             self.single_rmse_thres,
                                                                                             self.rmse_thres),
                                                                                         result_type='expand')
            # curr_vs_wl_df['vs_chain'] = vs.chainage
            curr_vs_wl_df = curr_vs_wl_df.dropna(subset=['shifted_wl'])
            multi_vs_wl_df = pd.concat(
                [multi_vs_wl_df.dropna(axis=1, how='all'), curr_vs_wl_df.dropna(axis=1, how='all')])
        # multi_vs_wl_df = multi_vs_wl_df.set_index(pd.to_datetime(multi_vs_wl_df['shifted_time']))
        multi_vs_wl_df
        self.densified_ts = multi_vs_wl_df

    def calculate_shifted_time(self, ts):

        ts['time_diff'] = pd.to_timedelta((ts['vs_chain'] - self.chainage) / self.speed_ms, unit='s')
        ts['shifted_time'] = pd.to_datetime(ts['dt']) + ts['time_diff']
        ts = ts.set_index(pd.to_datetime(ts['shifted_time']))
        return ts.sort_index()

    def calculate_shifted_time_by_curve(self, ts, vel_df):
        scaler_stg_vel, scaler_ds = MinMaxScaler(), MinMaxScaler()
        vel_df['waterlevel_norm'] = scaler_stg_vel.fit_transform(vel_df[['waterlevel']])
        ts['shifted_wl_norm'] = scaler_ds.fit_transform(ts[['shifted_wl']])
        vel_df_sorted = vel_df.sort_values('waterlevel_norm')
        ts_sorted = ts.sort_values('shifted_wl_norm').dropna()
        ts_sorted = pd.merge_asof(
            ts_sorted,
            vel_df_sorted[['waterlevel_norm', 'velocity']],
            left_on='shifted_wl_norm',
            right_on='waterlevel_norm',
            direction='nearest'
        )

        ts_sorted['time_diff'] = pd.to_timedelta((ts_sorted['vs_chain'] - self.chainage) / ts_sorted['velocity'],
                                                 unit='s')
        ts_sorted['shifted_time'] = pd.to_datetime(ts_sorted['dt']) + ts_sorted['time_diff']
        ts_sorted = ts_sorted.set_index(pd.to_datetime(ts_sorted['shifted_time']))
        return ts_sorted.sort_index()

    def calculate_shifted_time_by_simplified_mannig(self, ts, bottom, c=None):
        ts = ts.copy(deep=True)
        c = self.c if c is None else c
        slopes_dict = self.get_mean_slopes_dict()
        ts['slope_to_ds'] = ts['id_vs'].map(slopes_dict)
        bottom_heights = self.get_bottom_heights_dict(bottom)
        ts['bottom_height'] = ts['id_vs'].map(bottom_heights)
        ts['waterdepth'] = ts['vs_wl'] - ts['bottom_height']
        ts['velocity'] = (1 / c * ts['waterdepth'] ** (2 / 3) * (ts['slope_to_ds'] / 1000) ** (1 / 2)) * 5/3
        ts['time_diff'] = pd.to_timedelta((ts['vs_chain'] - self.chainage) / ts['velocity'], unit='s')
        ts['shifted_time'] = pd.to_datetime(ts['dt']) + ts['time_diff']
        ts['shifted_time'] = ts['shifted_time'].fillna(ts['dt'])
        ts = ts.set_index(pd.to_datetime(ts['shifted_time']))
        self.speed_ms = ts['velocity'].mean()
        return ts.sort_index()

    def calibrate_mannings_c(self, bottom=0.1):
        slf = copy.deepcopy(self)
        calibration_accuracies = []
        for c in [x / 500 for x in range(10, 51)]:
            slf.densified_ts = slf.calculate_shifted_time_by_simplified_mannig(slf.densified_ts,
                                                                                          bottom, c)
            df_true = slf.swot_wl[['wse']].set_index(pd.to_datetime(slf.swot_wl['datetime'])).resample(
                'D').mean().dropna()
            cval_rmse = slf.get_rmse_of_cval_ts(slf.densified_ts, df_true)
            amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
            wl_amplitude = slf.densified_ts['shifted_wl'].max() - slf.densified_ts[
                'shifted_wl'].min()
            rms_thr = wl_amplitude * amp_thres_final
            slf.densified_ts = slf.densified_ts.loc[
                slf.densified_ts['rmse_sum'] < rms_thr]
            slf.densified_ts = filter_outliers_by_tstudent_test(slf.densified_ts)

            densified_ts_cval = slf.densified_ts.loc[
                slf.densified_ts['id_vs'] != slf.id]
            densified_ts_cval_daily = get_rmse_weighted_wl(densified_ts_cval)
            densified_ts_cval_itpd = slf.interpolate(densified_ts_cval_daily)
            rmse_cval, nse_cval = slf.get_rmse_nse_values(densified_ts_cval_itpd,
                                                          densified_ts_cval_itpd.index.min(),
                                                          densified_ts_cval_itpd.index.max(), 'CrossVal',
                                                          df_true, False)
            calibration_accuracies.append([c, slf.speed_ms, rmse_cval])

        df_calib = pd.DataFrame(calibration_accuracies, columns=['c', 'velocity', 'rmse_cval'])
        rmse_cval, vels, c_cvals = df_calib['rmse_cval'], df_calib['velocity'], df_calib['c']
        min_index, min_cval = rmse_cval.idxmin(), rmse_cval.min()
        cval_range = rmse_cval[rmse_cval < min_cval + self.cval_buff]
        vels_at_cval_range = vels[cval_range.index]
        mean_vel = (vels_at_cval_range.max() + vels_at_cval_range.min()) / 2
        # vel_uncrt = (vels_at_cval_range.max() - vels_at_cval_range.min()) / 2
        mean_vel_idx = (vels_at_cval_range - mean_vel).abs().idxmin()
        c_cval = c_cvals[mean_vel_idx]
        self.c = c_cval
        # return c_cval, vel_uncrt

    def get_rmse_of_cval_ts(self, timeseries, val_ts):
        ts_cval = timeseries.loc[timeseries['id_vs'] != self.id]
        ts_daily = get_rmse_weighted_wl(ts_cval)
        ts_itpd = self.interpolate(ts_daily)
        r, n = self.get_rmse_nse_values(ts_itpd, ts_itpd.index.min(), ts_itpd.index.max(), '', val_ts, False)
        return r

    def get_rmse_agg_threshold(self, df_true):
        cval_rmse = self.get_rmse_of_cval_ts(self.densified_ts, df_true)
        amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
        wl_amplitude = self.densified_ts['shifted_wl'].max() - self.densified_ts[
            'shifted_wl'].min()
        return wl_amplitude * amp_thres_final

    def get_closest_in_situ_daily_wl(self, gauge_meas, t1, t2):
        gauge_meas = copy.deepcopy(gauge_meas)
        gauge_meas.index = pd.to_datetime(gauge_meas['shifted_time'])
        ts = gauge_meas['shifted_wl'].loc[(gauge_meas.index > t1) & (gauge_meas.index < t2)]
        self.closest_in_situ_daily_wl = ts.resample('D').mean()

    # def adjust_gauge_data_to_vs_by_mean_diff(self, gauge_meas_up):
    #     gauge_meas_up = copy.deepcopy(gauge_meas_up)
    #     mean_g = self.juxtaposed_wl['up_wl'].loc[~self.juxtaposed_wl['vs_wl'].isna()].mean()
    #     mean_vs = self.juxtaposed_wl['vs_wl'].loc[~self.juxtaposed_wl['up_wl'].isna()].mean()
    #     gauge_meas_up['shifted_wl'] = gauge_meas_up['stage'] + mean_vs - mean_g
    #     time_diff = pd.to_timedelta((self.neigh_g_up_chain - self.chainage) / self.speed_ms, unit='s')
    #     gauge_meas_up['shifted_time'] = pd.to_datetime(gauge_meas_up.index) + time_diff
    #     return gauge_meas_up

    def adjust_gauge_data_to_vs_by_regr(self, gauge_meas, g_chain, bottom_thres=0.1, vel_df=pd.DataFrame()):
        gauge_meas = copy.deepcopy(gauge_meas)
        # time_diff = pd.to_timedelta((self.neigh_g_up_chain - self.chainage) / self.speed_ms, unit='s')
        gauge_meas = gauge_meas.loc[self.juxtaposed_wl['dt'].min(): self.juxtaposed_wl['dt'].max()]
        a, b, r2, num_of_meas = get_regression_coeffs_from_df(
            self.juxtaposed_wl.dropna(subset=['vs_wl', 'g_wl'], how='any'), 'vs_wl', 'g_wl')
        gauge_meas['shifted_wl'] = a * gauge_meas['stage'] + b
        gauge_meas.loc[:, 'vs_chain'] = g_chain
        gauge_meas.loc[:, 'dt'] = pd.to_datetime(gauge_meas.index)
        if len(vel_df) > 0:
            gauge_meas = self.calculate_shifted_time_by_curve(gauge_meas, vel_df)
        else:
        #     time_diff = pd.to_timedelta((self.neigh_g_up_chain - self.chainage) / self.speed_ms, unit='s')
        #     gauge_meas_up['shifted_time'] = pd.to_datetime(gauge_meas_up.index) + time_diff

            slopes_dict = self.get_mean_slopes_dict()
            closest_vs = sorted([(x.id, abs(g_chain - x.chainage)) for x in self.upstream_adjacent_vs],
                                key=lambda xx: xx[1])[0][0]
            curr_slope = slopes_dict[closest_vs]
            gauge_meas['slope_to_ds'] = curr_slope

            curr_amplitude = gauge_meas['shifted_wl'].max() - gauge_meas['shifted_wl'].min()
            bottom_height = gauge_meas['shifted_wl'].min() - bottom_thres * curr_amplitude
            gauge_meas['bottom_height'] = bottom_height
            gauge_meas['waterdepth'] = gauge_meas['shifted_wl'] - gauge_meas['bottom_height']
            gauge_meas['velocity'] = (1 / self.c * gauge_meas['waterdepth'] ** (2 / 3) * (
                        gauge_meas['slope_to_ds'] / 1000) ** (1 / 2)) * 5 / 3
            gauge_meas['time_diff'] = pd.to_timedelta(
                (g_chain - self.chainage) / gauge_meas['velocity'], unit='s')
            gauge_meas['shifted_time'] = pd.to_datetime(gauge_meas['dt']) + gauge_meas['time_diff']

        return gauge_meas

    def get_daily_interpolated_wl_ts(self, method='akima'):
        self.itpd_method = method
        resampled = pd.Series(self.densified_wl['shifted_wl'].values, index=self.densified_wl['shifted_time'])
        # resampled_gauge = pd.Series(self.densified_wl['shifted_wl_gauge'].values,
        #                             index=self.densified_wl['shifted_time'])
        self.daily_wl = resampled.resample('D').mean()
        self.interpolated_wl = self.interpolate(self.daily_wl)
        # self.daily_gauge_wl = resampled_gauge.resample('D').mean().dropna()

    def get_single_vs_interpolated_ts(self):
        if self.wl['mission'].str.contains('SWOT').any():
            filtered_wl = self.wl[self.wl['mission'].str.contains('SWOT', na=False)]
        else:
            filtered_wl = self.wl
        resampled = pd.Series(filtered_wl.wse, index=filtered_wl.index)
        daily_wl = resampled.resample('D').mean()
        self.single_VS_itpd = self.interpolate(daily_wl)

    # def get_dist_weighted_wl(self, ts):
    #     curr_densified = ts.copy()
    #     curr_densified['dist_weight'] = round(self.buffer - (curr_densified['vs_chain'] - self.chainage) / 1000)
    #     curr_densified['weight_x_wl'] = curr_densified['shifted_wl'] * curr_densified['dist_weight']
    #     dist_weighted_daily = curr_densified.resample('D').agg(wl_x_weight_sum=('weight_x_wl', 'sum'),
    #                                                            weight_sum=('dist_weight', 'sum'))
    #     dist_weighted_daily['wl_weighted'] = dist_weighted_daily['wl_x_weight_sum'] / dist_weighted_daily['weight_sum']
    #     return dist_weighted_daily['wl_weighted']
    #
    # def get_spline_interpolated_ts(self, ts_to_interpolate, s):
    #     df_clean = ts_to_interpolate.dropna()
    #     x_data = df_clean.index.astype(np.int64) // 10 ** 9
    #     y_data = df_clean.values
    #     spl = UnivariateSpline(x_data, y_data, s=s)
    #     full_dt_range = pd.date_range(start=ts_to_interpolate.index.min(), end=ts_to_interpolate.index.max(), freq='D')
    #     x_full = full_dt_range.astype(np.int64) // 10 ** 9
    #     df_full = pd.DataFrame(index=full_dt_range)
    #     df_full['spline_itpd_wl'] = spl(x_full)
    #     self.spline_itpd_wl = df_full
    #
    # def get_densified_wl_filtered_by_bias(self, bias=1):
    #     return self.interpolate(
    #         self.densified_wl['shifted_wl'].loc[self.densified_wl['bias'] < bias].resample('D').mean())

    def get_svr_smoothed_data(self, c=100, gamma=0.0001, epsilon=0.1, do_plot=False):
        input_df = self.densified_ts.copy()
        eps = 1e-6  # Bardzo mała stała, by uniknąć dzielenia przez 0 dla RS
        max_weight_factor = 30
        base_weights = 1 / (input_df['rmse_sum'] + eps)
        min_weight = base_weights.min()
        weights = base_weights / min_weight
        weights = weights.round().astype(int)
        weights = weights.clip(upper=max_weight_factor)
        svr_rbf = svm.SVR(kernel='rbf', C=c, gamma=gamma, epsilon=epsilon)

        index_dates = input_df.index
        start_time = index_dates.min()
        end_time = input_df.index.max()
        time_deltas = index_dates - start_time
        X_hours = time_deltas.total_seconds() / 3600
        X_train = X_hours.values.reshape(-1, 1)
        svr_rbf.fit(X_train, input_df['shifted_wl'], sample_weight=weights)
        # svr_rbf.fit(X_train, input_df['shifted_wl'])

        # HOURLY PREDICT AND RESAMPLE DAILY
        # start_date = index_dates.min().floor(freq='H')
        # end_date = index_dates.max().ceil(freq='H')
        # hourly_predict_index = pd.date_range(start=start_date, end=end_date, freq='H')
        # predict_deltas = hourly_predict_index - start_time
        # x_hourly_predict = predict_deltas.total_seconds() / 3600
        # x_hourly_predict_2D = x_hourly_predict.values.reshape(-1, 1)
        # y_hourly_numpy = svr_rbf.predict(x_hourly_predict_2D)
        # hourly_svr_series = pd.Series(y_hourly_numpy, index=hourly_predict_index)
        # daily_aggregated_svr = hourly_svr_series.resample('D').mean()

        y_res = svr_rbf.predict(X_train)
        input_df['shifted_wl'] = y_res
        y_res_series = pd.Series(y_res, index=input_df.index).resample('D').mean().interpolate(method=self.itpd_method)

        if do_plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(self.densified_ts.index, self.densified_ts['shifted_wl'], marker='.', color='purple',
                       label='meas')
            ax.plot(self.densified_itpd, linewidth=2, color='purple', label='DS REGR ITPD')
            # ax.plot(self.single_VS_itpd, linewidth=2, color='green', label='SINGLE VS ITPD')
            ax.plot(y_res_series, linewidth=2, color='orange', label='SVR v2', marker='.')
            # ax.plot(daily_aggregated_svr, linewidth=2, color='pink', label='SVR v3', marker='.')
            ax.plot(self.closest_in_situ_daily_wl, linewidth=4, alpha=0.5, color='black', label='in situ')
            ax.legend()
            ax.grid(alpha=0.6, linestyle='dashed')
            ax.set_xlabel('Time')
            ax.set_ylabel('Water level [m]')
            plt.show(block=True)
        self.svr_ts, self.svr_itpd = input_df, y_res_series

    # def get_loess_filtered_ts(self, smoothing_window_days=50, max_weight_factor=10, do_plot=False):
    #     input_df = self.densified_ts.copy()
    #     epsilon = 1e-6  # Bardzo mała stała, by uniknąć dzielenia przez 0 dla RS
    #
    #     # 1. Konwersja błędu na wagę
    #     base_weights = 1 / (input_df['rmse_sum'] + epsilon)
    #
    #     # 2. Normalizacja wag, aby najgorszy punkt miał wagę ~1
    #     min_weight = base_weights.min()
    #     repetition_factors = base_weights / min_weight
    #
    #     # 3. Zaokrąglenie do liczb całkowitych i ograniczenie
    #     repetition_factors = repetition_factors.round().astype(int)
    #     repetition_factors = repetition_factors.clip(upper=max_weight_factor)
    #
    #     # print("Przykładowe współczynniki powtórzeń:")
    #     # print(repetition_factors.value_counts().sort_index())
    #
    #     # Bardziej wydajny sposób na upsampling z użyciem .repeat()
    #     upsampled_df = input_df.loc[input_df.index.repeat(repetition_factors)]
    #
    #     # print(f"\nRozmiar oryginalnego zbioru danych: {len(input_df)}")
    #     # print(f"Rozmiar zbioru danych po upsamplingu: {len(upsampled_df)}")
    #
    #     # Obliczamy medianę liczby punktów w oknie czasowym na ORYGINALNYCH danych
    #     points_in_window = input_df.rolling(f'{smoothing_window_days}D').count()['shifted_wl']
    #     median_points_in_window = int(points_in_window.median())
    #
    #     # Zabezpieczenie, jeśli mediana jest zbyt mała
    #     if median_points_in_window < 5:
    #         median_points_in_window = 5
    #
    #     # Obliczamy 'frac' dla ZBIORU PRZESAMPLOWANEGO
    #     # To jest kluczowy krok: frac adaptuje się do gęstości danych i wag
    #     dynamic_frac = median_points_in_window / len(upsampled_df)
    #
    #     # Zabezpieczenie przed zbyt małym frac, który powoduje niestabilność
    #     min_safe_frac = 10 / len(upsampled_df)  # Zapewnia, że w oknie jest co najmniej 10 punktów
    #     if dynamic_frac < min_safe_frac:
    #         dynamic_frac = min_safe_frac
    #
    #     # print(f"Zdefiniowano okno: {smoothing_window_days} dni.")
    #     # print(f"Mediana punktów w takim oknie (dane oryginalne): {median_points_in_window}")
    #     # print(f"Rozmiar danych po upsamplingu: {len(upsampled_df)}")
    #     # print(f"Dynamicznie obliczony 'frac' do użycia w LOESS: {dynamic_frac:.4f}")
    #
    #     # --- Krok 3: Uruchomienie LOESS na upsamplowanych danych ---
    #     y = upsampled_df['shifted_wl']
    #     x_numeric = upsampled_df.index.astype(np.int64)
    #
    #     # Uruchomienie LOESS
    #     smoothed_loess = sm.nonparametric.lowess(endog=y, exog=x_numeric, frac=dynamic_frac)
    #     # Wynik LOESS jako seria z poprawnym indeksem czasu
    #     smoothed_irregular_series = pd.Series(smoothed_loess[:, 1], index=upsampled_df.index)
    #
    #     moothed_irregular_series = pd.Series(smoothed_loess[:, 1], index=pd.to_datetime(smoothed_loess[:, 0]))
    #     smoothed_irregular_series = smoothed_irregular_series.groupby(smoothed_irregular_series.index).mean()
    #     smoothed_irregular_series = filter_outliers_by_tstudent_test(
    #         pd.DataFrame(smoothed_irregular_series.values, index=smoothed_irregular_series.index,
    #                      columns=['shifted_wl']))
    #     smoothed_irregular_series = smoothed_irregular_series['shifted_wl']
    #     daily_index = pd.date_range(start=input_df.index.min(), end=input_df.index.max(), freq='D').normalize()
    #     final_daily_series = smoothed_irregular_series.reindex(
    #         daily_index.union(smoothed_irregular_series.index)).interpolate(
    #         method=self.itpd_method).reindex(daily_index)
    #
    #     if do_plot:
    #         fig, ax = plt.subplots(figsize=(15, 8))
    #         ax.scatter(input_df[input_df['id_vs'] != self.id].index, input_df[input_df['id_vs'] != self.id]['shifted_wl'],
    #                    s=10, c='purple', alpha=0.4, label='Pomiary VS')
    #         ax.scatter(input_df[input_df['id_vs'] == self.id].index, input_df[input_df['id_vs'] == self.id]['shifted_wl'],
    #                    s=60, c='red', marker='*', zorder=5, label='Pomiary RS')
    #         ax.plot(final_daily_series, color='dodgerblue', linewidth=2.5,
    #                 label=f"Finalny szereg czasowy (codzienny, dynamiczny LOESS + {self.itpd_method})")
    #         ax.plot(self.closest_in_situ_daily_wl, color='black', linewidth=2, label='in situ WL')
    #         ax.plot(self.wl['wse'].resample('D').mean().interpolate(method='akima'), color='green', linewidth=2,
    #                 label='just itpd')
    #         ax.plot(self.densified_itpd, color='purple', linewidth=2, label='previous approach')
    #         ax.set_title(f"Finalny proces: Dynamiczny Upsampling (z rmse_sum) -> LOESS -> Interpolacja {self.itpd_method}",
    #                      fontsize=16)
    #         ax.set_xlabel('Czas', fontsize=12)
    #         ax.set_ylabel('Poziom wody [m]', fontsize=12)
    #         ax.legend(loc='upper right', fontsize=10)
    #         ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    #         plt.tight_layout()
    #         plt.show(block=True)
    #
    #     self.loess_filtered_ts = final_daily_series

    def get_rmse_nse_values(self, interpolated, t1, t2, text, other_validation_ts=pd.Series(), print_res = True):
        if len(other_validation_ts) > 0:
            df_combined = pd.concat([other_validation_ts, interpolated], axis=1)
        else:
            df_combined = pd.concat([self.closest_in_situ_daily_wl, interpolated], axis=1)
        df_combined.columns = ['gauge_mean', 'model_mean']
        df_cleaned = df_combined.dropna()
        if len(df_cleaned) < 5:
            return np.nan, np.nan
        df_cleaned = df_cleaned[t1:t2]
        if len(df_cleaned) < 5:
            return np.nan, np.nan
        y_true = df_cleaned['gauge_mean']
        y_predicted = df_cleaned['model_mean']
        rmse = round(np.sqrt(mean_squared_error(y_true, y_predicted)), 4)
        nse = round(he.evaluator(he.nse, y_predicted, y_true)[0], 4)
        if print_res:
            # print(f'{text} {self.id}, velocity {round(self.speed_ms, 3)} m/s, RMSE: {rmse}m, NSE: {nse}')
            print(f'{text} {self.id}, velocity {round(self.densified_ts["velocity"].mean(), 3)} m/s, RMSE: {rmse}m, NSE: {nse}')
        return rmse, nse

    def plot_daily_wl(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.interpolated_wl, marker='o', label=f'{self.id} interpolated WL')
        ax.plot(self.closest_in_situ_daily_wl, marker='o', label='daily gauge WL')
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', label='VS data', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.show(block=True)

    def plot_densified_wl(self):
        fig, ax = plt.subplots()
        ax.plot(self.densified_wl.index, self.densified_wl['shifted_wl'], label='densified_wl')
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', label='VS data', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show(block=True)

    def plot_densified_vs_gauge_color_by_chainage(self, timeseries):
        fig, axs = plt.subplot_mosaic([['upper_left', 'upper_right'], ['bottom', 'bottom']],
                                      figsize=(18, 9), height_ratios=[1, 2])
        ax, ax1, ax2 = axs['bottom'], axs['upper_left'], axs['upper_right']
        timeseries['chainage_diff'] = (timeseries['vs_chain'] - self.chainage) / 1000
        n_classes = 10
        min_val = timeseries['chainage_diff'].min()
        max_val = timeseries['chainage_diff'].max()
        bins = np.linspace(min_val, max_val, n_classes + 1)
        cmap = cm.get_cmap('RdYlGn')
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)

        scatter = ax.scatter(timeseries['shifted_time'], timeseries['shifted_wl'],
                             c=timeseries['chainage_diff'], cmap=cmap, marker='o',
                             label='WL colored by distance', zorder=2, norm=norm, edgecolors='black', linewidths=0.5)
        ax.errorbar(timeseries['shifted_time'], timeseries['shifted_wl'], yerr=timeseries['bias'],
                    ecolor='red', label='VS measurement bias', zorder=1)
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', s=50, color='purple',
                   label='water level from reference station', zorder=11, edgecolors='black', linewidths=0.5)
        # ax.plot(self.densified_wl['shifted_time'], self.densified_wl['shifted_wl_gauge'], color='black',
        #         label='shifted gauge WL')
        ax.plot(self.closest_in_situ_daily_wl, color='black', label='shifted gauge WL')

        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        cbar = plt.colorbar(scatter, ticks=bins)
        cbar.set_label('Along-river distance from ref. station')
        cbar.ax.set_yticklabels([f'{b:.0f}' for b in bins])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        ax1.scatter(timeseries['chainage_diff'], timeseries['shifted_wl_bias'])
        ax1.set_xlabel('Along-river distance from ref. station')
        ax1.set_ylabel('Bias to shifted gauge WL')

        ax2.scatter(timeseries['bias'], timeseries['shifted_wl_bias'])
        ax2.set_xlabel('Bias of VS measurement')
        ax2.set_ylabel('Bias to shifted gauge WL')

        fig.tight_layout()
        plt.show(block=True)

    def plot_all_approaches(self, not_plotting: set = 'spline'):
        fig, ax = plt.subplots(figsize=(12, 8))
        if self.closest_in_situ_daily_wl is not None:
            ax.plot(self.closest_in_situ_daily_wl, label='in situ', color='black', linewidth=3)
        if self.single_VS_itpd is not None and 'single' not in not_plotting:
            ax.plot(self.single_VS_itpd, label='just itp', color='blue')
        if self.interpolated_wl is not None and 'classic' not in not_plotting:
            ax.plot(self.interpolated_wl, label='densified', color='orange')
            ax.scatter(self.densified_wl['shifted_time'], self.densified_wl['shifted_wl'], marker='.', color='orange')
        if self.spline_itpd_wl is not None and 'spline' not in not_plotting:
            ax.plot(self.spline_itpd_wl, label='spline_itpd', color='magenta')
        if self.normalized_ts_itpd is not None and 'norm' not in not_plotting:
            ax.plot(self.normalized_ts_itpd, label='normalized', color='red')
            ax.scatter(self.normalized_ts['shifted_time'], self.normalized_ts['shifted_wl'], marker='.', color='red')
        if self.densified_ts is not None and 'regr' not in not_plotting:
            ax.plot(self.densified_itpd, label='regressions', color='purple')
            ax.scatter(self.densified_ts['shifted_time'], self.densified_ts['shifted_wl'],
                       marker='.', color='purple')

        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show(block=True)

    def plot_all_vs_timeseries(self):
        viridis_cmap = plt.colormaps['viridis']
        colors = [viridis_cmap(i) for i in np.linspace(0, 1, len(self.upstream_adjacent_vs))]
        fig, ax = plt.subplots()
        g_chain = self.neigh_g_up_chain if self.closest_gauge == 'up' else self.neigh_g_dn_chain
        ax.plot(self.closest_in_situ_daily_wl, label=f'In situ {round((g_chain - self.chainage) / 1000)}'
                                                     f' km from DS', color='red', linewidth=4)
        mean_ds = self.wl.wse.mean()
        curr_wl = self.wl.loc[(self.wl.index > self.wl.index.min()) & (self.wl.index < self.wl.index.max())]
        ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color='magenta', linewidth=3, label=f'DS {self.id}')
        for i, vs in enumerate(sorted(self.upstream_adjacent_vs, key=lambda x: x.chainage)):
            # print(vs.wl.index)
            curr_wl = vs.wl.loc[(vs.wl.index > self.wl.index.min()) & (vs.wl.index < self.wl.index.max())]
            ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color=colors[i],
                    label=f'{vs.id}, {round((vs.chainage - self.chainage) / 1000)} km from DS')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show(block=True)

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

    def plot_all_vs_timeseries_with_cofluent_gauge(self, df_gauge, loaded_stations):
        viridis_cmap = plt.colormaps['viridis']
        colors = [viridis_cmap(i) for i in np.linspace(0, 1, len(self.upstream_adjacent_vs))]
        df_gauge = df_gauge.set_index(pd.to_datetime(df_gauge['date']))
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax.plot(self.closest_in_situ_daily_wl, label='In situ', color='red', linewidth=4)
        mean_ds = self.wl.wse.mean()
        for i, vs in enumerate(sorted(self.upstream_adjacent_vs, key=lambda x: x.chainage)):
            curr_wl = vs.wl.loc[(vs.wl.index > self.wl.index.min()) & (vs.wl.index < self.wl.index.max())]
            if vs.chainage >= [x for x in loaded_stations if x.id == 959][0].chainage:
                linestyle = '--'
            else:
                linestyle = '-'
            ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color=colors[i],
                    label=f'{vs.id}, {round((vs.chainage - self.chainage) / 1000)} km from DS', linestyle=linestyle)
        ax.set_ylim(5, 8.5)
        ax.legend(ncol=2)
        x_min, x_max = pd.to_datetime('2023-07-01'), pd.to_datetime('2024-07-01')
        ax2.plot(df_gauge.loc[x_min:]['stage'], label='Gauge water levels', linestyle='--')
        # ax3=ax2.twinx()
        ax2.legend(loc='upper left')
        ax2.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        ax2.set_ylabel('Water level [m]')
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.7)
        plt.show(block=True)

    def plot_vs_setting_with_regressions_rmse(self, current_river, plot_swot_tiles=True):
        used_regressions = self.get_used_regressions()
        curr_riv_sect = self.clip_river_to_vs_section(current_river)
        cmap = plt.get_cmap('viridis')
        vmin = used_regressions['rmse'].min()
        vmax = used_regressions['rmse'].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        text_bbox_props = dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.5)
        all_x = []
        all_y = []
        for index, row in used_regressions.iterrows():
            vs1 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st1']][0]
            vs2 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st2']][0]
            all_x.extend([vs1.x, vs2.x])
            all_y.extend([vs1.y, vs2.y])
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)

        base_width = 7
        min_width, max_width = 6, 12
        min_height, max_height = 5, 10
        # Obliczenie wstępnej szerokości i wysokości
        aspect_ratio_data = y_range / x_range if x_range > 0 else 1

        # Zastosowanie limitów
        desired_width = np.clip(base_width, min_width, max_width)
        desired_height = np.clip(desired_width * aspect_ratio_data, min_height, max_height)
        margin_factor = 1.05  # Dodaj 20% marginesu
        fig_width = desired_width * margin_factor
        fig_height = desired_height * margin_factor

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        curr_riv_sect.plot(ax=ax, linewidth=2, color='grey', zorder=1)
        if plot_swot_tiles:
            swot_tiles_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/dane_SWOT/SWOT_tiles/swot_science_hr_Aug2021-v05_shapefile_swath/swot_science_hr_2.0s_4.0s_Aug2021-v5_swath.shp'
            swot_gdf = gpd.read_file(swot_tiles_file)
            swot_gdf.plot(
                ax=ax,
                color='gray',  # Ustawia kolor wypełnienia na szary
                alpha=0.4,  # Ustawia przezroczystość na 50% (0.0 to w pełni przezroczysty, 1.0 to w pełni kryjący)
                edgecolor='black',  # Ustawia kolor obwódki na czarny
                linewidth=0.5,  # Ustawia grubość obwódki (możesz dostosować tę wartość, aby uzyskać "cienki" efekt)
                zorder=1
            )
        for index, row in used_regressions.iterrows():
            vs1 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st1']][0]
            vs2 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st2']][0]
            line_color = cmap(norm(row['rmse']))
            if row['steps'] == 1:
                ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=3, zorder=2, marker='o')
            elif row['steps'] == 2:
                ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=3, linestyle='dashed', zorder=2,
                        marker='o')
            elif row['steps'] == 3:
                ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=3, linestyle='dotted', zorder=2,
                        marker='o')

            ax.text(vs1.x + 0.01, vs1.y + 0.0002, str(vs1.id), ha='left', fontsize=9, bbox=text_bbox_props)
            ax.text(vs2.x + 0.01, vs2.y + 0.0002, str(vs2.id), ha='left', fontsize=9, bbox=text_bbox_props)

        ax.scatter(self.x, self.y, label='RS', color='red', zorder=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(
            [])  # Ustawiamy pustą tablicę, ponieważ pasek kolorów nie jest bezpośrednio powiązany z wykresem plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(sm, cax=cax, label='RMSE [m]')

        solid_line = mlines.Line2D([], [], color='darkblue', linewidth=3, linestyle='-', label='direct regressions')
        dashed_line = mlines.Line2D([], [], color='darkblue', linewidth=3, linestyle='dashed',
                                    label='bypass regressions')
        dotted_line = mlines.Line2D([], [], color='yellow', linewidth=3, linestyle='dotted',
                                    label='unused regressions')
        river_line = mlines.Line2D([], [], color='grey', linewidth=2, label='river course')
        rs_point = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='reference station')
        vs_point = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None', label='virtual station')

        # ax.legend(handles=[solid_line, dashed_line, river_line, ax.get_children()[-1]])
        ax.legend(handles=[rs_point, vs_point, solid_line, dashed_line, dotted_line, river_line])

        aspect_ratio_fig = fig_height / fig_width
        aspect_ratio_data = y_range / x_range
        if aspect_ratio_data < aspect_ratio_fig:
            # Rysunek jest "wyższy" niż dane. Dostosuj Y-margines.
            new_y_range = x_range * aspect_ratio_fig
            y_center = (max(all_y) + min(all_y)) / 2
            ax.set_ylim((y_center - new_y_range / 2) - 0.075, (y_center + new_y_range / 2) + 0.075)
            ax.set_xlim(min(all_x) - 0.075, max(all_x) + 0.075)
        else:
            # Rysunek jest "szerszy" niż dane. Dostosuj X-margines.
            new_x_range = y_range / aspect_ratio_fig
            x_center = (max(all_x) + min(all_x)) / 2
            ax.set_xlim((x_center - new_x_range / 2) - 0.075, (x_center + new_x_range / 2) + 0.075)
            ax.set_ylim(min(all_y) - 0.075, max(all_y) + 0.075)

        ax.set_title(f'RS{self.id} regressions')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        plt.show(block=True)
        # plt.savefig(f'RS{self.id}_at_{current_river.name}_regressions_map.png', dpi=300, bbox_inches='tight')

    def plot_all_gauge_timeseries_within_ds(self, loaded_gauges):
        min_chain = min([x.chainage for x in self.upstream_adjacent_vs])
        max_chain = max([x.chainage for x in self.upstream_adjacent_vs])
        gauges = [x for x in loaded_gauges.values() if min_chain < x.chainage < max_chain]
        first_gauge = \
            [x for x in gauges if
             abs(x.chainage - self.chainage) == min([abs(x.chainage - self.chainage) for x in gauges])][0]
        first_gauge_mean_wl = first_gauge.wl_df['stage'].loc[self.closest_in_situ_daily_wl.index].mean()

        bins = sorted(list(set([g.chainage for g in gauges])))
        bins.append(bins[-1] + 1e-6)  # Dodaj małą wartość do ostatniego binu
        bins = np.array(bins)

        cmap = cm.get_cmap('viridis')
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)

        fig, ax = plt.subplots()
        for gauge in gauges:
            color = cmap(norm(gauge.chainage))
            ts = gauge.wl_df.loc[self.closest_in_situ_daily_wl.index]
            ts['stage'] = ts['stage'] - ts['stage'].mean() + first_gauge_mean_wl
            ax.plot(ts['stage'], label=f'gauge {gauge.id} at {round(gauge.chainage / 1000)} km', color=color)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show(block=True)
