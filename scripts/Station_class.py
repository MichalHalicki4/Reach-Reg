import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.ops import linemerge, nearest_points, unary_union
from shapely.geometry import Point
import json
import requests
from datetime import timedelta
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
import os
import folium
import copy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


dahiti_in_situ_collections = {
    'germany': [5, 44, 46, 48, 49, 50, 51, 53],
    # 'poland': [35]
    'poland': [57]

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


def get_corresponding_gauge_wl(row, gauge_data):
    try:
        return gauge_data['shifted_wl'].loc[
            gauge_data['shifted_time'].dt.round('H') == row['shifted_time'].round('H')].values[0]
    except IndexError:
        return np.nan


def select_gauges_from_river(gdata_gdf, river):
    gdata_gdf = gdata_gdf.to_crs(river.gdf.crs)
    gdata_gdf_metric = gdata_gdf.to_crs(river.metrical_crs)
    river_buffer_metric = gpd.GeoSeries(river.simplified_river).buffer(1000)
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


def filter_gauges_by_dt_freq_target(gauges_metadata, min_dt, max_dt):
    return gauges_metadata.loc[(gauges_metadata['type'] == 'water_level') &
                               (pd.to_datetime(gauges_metadata['min_date']) < pd.to_datetime(min_dt)) &
                               (pd.to_datetime(gauges_metadata['max_date']) > pd.to_datetime(max_dt)) &
                               # (gauges_metadata['data_sampling'] != 'daily') &
                               (~gauges_metadata['target_name'].str.contains('see'))]


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
        metadata.append({'id': list_id[i], 'x': list_x[i], 'y': list_y[i], 'length': list_lengths[i], 'chainage':list_chains[i]})

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


def get_linear_regression_coeffs_btwn_stations(vs, vs2):
    vs_set = set(pd.to_datetime(vs.swot_wl['datetime']).round('H'))
    vs2_set = set(pd.to_datetime(vs2.swot_wl['datetime']).round('H'))
    common_indices = vs_set.intersection(vs2_set)
    vs_swot_ts = vs.swot_wl.set_index(pd.to_datetime(vs.swot_wl['datetime']).round('H'))['wse'].loc[
        list(common_indices)]
    vs2_swot_ts = vs2.swot_wl.set_index(pd.to_datetime(vs2.swot_wl['datetime']).round('H'))['wse'].loc[
        list(common_indices)]
    regr_df = pd.DataFrame(index=list(common_indices))
    regr_df['vs_wse'] = vs_swot_ts
    regr_df['vs2_wse'] = vs2_swot_ts
    linr_model = LinearRegression().fit(regr_df[['vs2_wse']], regr_df[['vs_wse']])
    r_squared = r2_score(y_true=regr_df[['vs_wse']], y_pred=linr_model.predict(regr_df[['vs2_wse']]))
    return round(linr_model.coef_[0][0], 3), round(linr_model.intercept_[0], 3), round(r_squared, 3), len(common_indices)


def get_wl_by_multiple_regressions(row, st_d_id, regressions_df, st_u_id):
    next_vs = 0
    u_wse = row['vs_wl']
    while next_vs != st_d_id:
        next_vs, st1, a, b, r2, data_len = regressions_df.loc[regressions_df['st2'] == st_u_id].values[0]
        st1, next_vs = int(st1), int(next_vs)
        u_wse = a * u_wse + b
        st_u_id = next_vs
    return u_wse


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
    plt.show()


def plot_interactive_measurements_along_river(river, column_name, vs_stations=None, gauge_stations=None):
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
                color=color, # Użyj przypisanego koloru
                tooltip="type", # Możesz wyświetlić typ w dymku
                popup="river_name", # Lub nazwę rzeki
                style_kwds={"weight": 3, "opacity": 0.7},
                name=f"Rzeka - Typ {r_type}", # Nazwa warstwy w kontroli warstw
                highlight=True # Podświetlanie po najechaniu myszką
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
    plt.show()


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
    plt.show()


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
        if self.sampling in ['hourly', '15-minute', '10-minute']:
            resample_str = 'H'
        else:
            resample_str = 'D'
        self.wl_df = self.wl_df[['stage']].resample(resample_str).mean()


class VirtualStation:
    def __init__(self, vs_id):
        self.id = vs_id
        self.x, self.y, self.river, self.chainage = None, None, None, None
        self.wl, self.swot_wl, self.geoid, self.swot_mmxo_rbias, self.slope_correction = None, None, None, None, None
        self.neigh_g_up, self.neigh_g_up_chain, self.neigh_g_dn, self.neigh_g_dn_chain = None, None, None, None
        self.juxtaposed_wl = None
        self.mean_g_wl, self.mean_vs_wl = None, None

    def get_water_levels(self, dahiti):
        try:
            wl_data = dahiti.download_water_level(self.id, parameters=['mission'])
        except Exception as e:
            print(e)
            self.wl = 'error'
            return None
        self.x = wl_data['longitude']
        self.y = wl_data['latitude']
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
        self.swot_wl = self.swot_wl.loc[(pd.to_datetime(self.swot_wl['datetime']) > t1) & (pd.to_datetime(self.swot_wl['datetime']) < t2)]

    def upload_chainage(self, chainage):
        self.chainage = chainage

    def find_closest_gauge_and_chain(self, gauges_chains):
        gauges_chains = pd.DataFrame([[g.id, g.chainage] for g in gauges_chains.values()], columns=['id', 'chainage'])
        if self.chainage > gauges_chains['chainage'].max() or self.chainage < gauges_chains['chainage'].min():  # VS at end of gauges - either no upstream, or no downstream from VS
            self.neigh_g_dn, self.neigh_g_up, self.neigh_g_dn_chain, self.neigh_g_up_chain = False, False, False, False
            return None
        up_id, up_chain = gauges_chains[gauges_chains['chainage'] > self.chainage].sort_values(by='chainage').iloc[0][[
            'id', 'chainage']]
        dn_id, dn_chain = \
            gauges_chains[gauges_chains['chainage'] < self.chainage].sort_values(by='chainage', ascending=False).iloc[
                0][[
                'id', 'chainage']]

        self.neigh_g_up, self.neigh_g_up_chain = up_id, up_chain
        self.neigh_g_dn, self.neigh_g_dn_chain = dn_id, dn_chain

    def get_juxtaposed_vs_and_gauge_meas(self, gauge_meas_up, gauge_meas_down, gdata_sampling, velocity=None):
        hours_to_juxtapose = 12
        juxtaposed_columns = ['id_vs', 'vs_chain', 'dt', 'mission', 'gauge_up', 'dist_up', 'gauge_down', 'dist_down', 'lag',
                              'vs_wl', 'up_wl', 'uncertainty', 'g_anom', 'vs_anom', 'bias']
        juxtaposed_data = []
        for index, row in self.wl.iterrows():
            vs_wl, vs_dt = row[['wse', 'datetime']]
            vs_dt_prev = vs_dt - pd.to_timedelta('5 days')

            if gdata_sampling == 'daily':
                shift = pd.to_timedelta((self.neigh_g_up_chain - self.chainage) / velocity, unit='s')
                gauge_up_time = (vs_dt - shift).round('H')
                final_lag = round(shift.total_seconds()/3600)
            else:
                ts_up = gauge_meas_up['stage'].loc[(gauge_meas_up.index > vs_dt_prev) & (gauge_meas_up.index < vs_dt)]
                ts_dn = gauge_meas_down['stage'].loc[(gauge_meas_down.index > vs_dt_prev) & (gauge_meas_down.index < vs_dt)]

                lag, corr = get_optimum_lag(ts_dn, ts_up, 50)
                ratio = self.neigh_g_up_chain / (self.neigh_g_up_chain + self.neigh_g_dn_chain)
                final_lag = lag * ratio

                gauge_up_time = (vs_dt - pd.to_timedelta(f'{final_lag} hours')).round('H')
            gauge_wl = np.nan
            for i in range(hours_to_juxtapose):
                try:
                    gauge_wl = \
                        gauge_meas_up['stage'][
                            gauge_meas_up.index == gauge_up_time + pd.to_timedelta(f'{i} hours')].values[0]
                    break
                except IndexError:
                    try:
                        gauge_wl = gauge_meas_up['stage'][
                            gauge_meas_up.index == gauge_up_time + pd.to_timedelta(f'-{i} hours')].values[0]
                        break
                    except IndexError:
                        continue
            print(self.id, vs_dt, vs_wl, gauge_wl, final_lag)
            juxtaposed_data.append(
                [self.id, self.chainage, vs_dt, row['mission'], self.neigh_g_up, self.neigh_g_up_chain, self.neigh_g_dn,
                 self.neigh_g_dn_chain, final_lag, vs_wl, gauge_wl, row['wse_u']])
        curr_results = pd.DataFrame(juxtaposed_data, columns=juxtaposed_columns[:-3])
        mean_g, mean_vs = curr_results['up_wl'].mean(), curr_results['vs_wl'].mean()
        curr_results['g_anom'] = curr_results['up_wl'] - mean_g
        curr_results['vs_anom'] = curr_results['vs_wl'] - mean_vs
        curr_results['bias'] = abs(curr_results['vs_anom'] - curr_results['g_anom'])
        self.mean_g_wl, self.mean_vs_wl = mean_g, mean_vs
        self.juxtaposed_wl = curr_results

    def plot_anomalies(self):
        fig, ax = plt.subplots()
        ax.plot(self.juxtaposed_wl['dt'], self.juxtaposed_wl['vs_anom'], label='VS anom')
        ax.plot(self.juxtaposed_wl['dt'], self.juxtaposed_wl['g_anom'], label='Gauge anom')
        ax.set_ylabel('Water level anomaly [m]')
        ax.set_xlabel('Time')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def get_daily_linear_interpolated_wl_of_single_vs(self, just_swot=False):
        if just_swot:
            vs_wl_to_corr = self.wl.loc[self.wl['mission'].str.contains('SWOT', na=False)].copy()
        else:
            vs_wl_to_corr = self.wl.copy()
        return vs_wl_to_corr['wse'].resample('D').mean().interpolate(method='linear')


class DensificationStation(VirtualStation):
    def __init__(self, vs_object):
        super().__init__(vs_object.id)
        self.__dict__.update(vs_object.__dict__)
        self.buffer, self.upstream_adjacent_vs = None, None
        self.regressions_df = None
        self.itpd_method = None
        self.densified_wl = None
        self.speed_ms = None
        self.closest_in_situ_daily_wl = None
        self.daily_wl, self.interpolated_wl, self.daily_gauge_wl, self.single_VS_itpd = None, None, None, None
        self.dist_weighted_daily_wl, self.dist_weighted_daily_wl_itpd, self.spline_itpd_wl = None, None, None
        self.normalized_ts, self.normalized_ts_daily, self.normalized_ts_itpd = None, None, None
        self.densified_wl_by_regr_ts, self.densified_wl_by_regr_daily, self.densified_wl_by_regr_itpd = None, None, None
        self.rmse, self.nse = None, None

    def __repr__(self):
        return f'Densification station ID: {self.id}, chainage: {self.chainage} with {len(self.upstream_adjacent_vs)} VS within buffer'

    def get_upstream_adjacent_vs(self, vs_list, buffer):
        self.buffer = buffer
        selected_list = []
        for vs in vs_list:
            if self.chainage <= vs.chainage < self.chainage + self.buffer * 1000:
                selected_list.append(copy.deepcopy(vs))
        self.upstream_adjacent_vs = sorted(selected_list, key=lambda k: k.chainage)

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

    def get_vs_regressions_df(self):
        regressions = []
        vs = self.upstream_adjacent_vs[0]
        a, b, r2, data_len = get_linear_regression_coeffs_btwn_stations(self, vs)
        regressions.append({'st1': self.id, 'st2': vs.id, 'a': a, 'b': b, 'r2': r2, 'data_len': data_len})
        for i, vs in enumerate(self.upstream_adjacent_vs[:-1]):
            vs2 = self.upstream_adjacent_vs[i + 1]
            a, b, r2, data_len = get_linear_regression_coeffs_btwn_stations(vs, vs2)
            regressions.append({'st1': vs.id, 'st2': vs2.id, 'a': a, 'b': b, 'r2': r2, 'data_len': data_len})
        self.regressions_df = pd.DataFrame(regressions)

    def interpolate(self, ts):
        return ts.interpolate(method=self.itpd_method)

    def get_densified_wl(self, speed_ms):
        self.speed_ms = speed_ms
        multi_vs_wl_df = self.juxtaposed_wl.copy()
        multi_vs_wl_df['time_diff'] = pd.to_timedelta(0, unit='s')
        multi_vs_wl_df['shifted_time'] = pd.to_datetime(multi_vs_wl_df['dt']) + multi_vs_wl_df['time_diff']
        multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['vs_wl']
        multi_vs_wl_df['vs_chain'] = self.chainage
        for vs in self.upstream_adjacent_vs:
            cond1 = len(vs.juxtaposed_wl) == 0
            cond2 = vs.id == self.id
            if cond1 or cond2:
                continue
            curr_vs_wl_df = vs.juxtaposed_wl.copy()
            curr_vs_wl_df['time_diff'] = pd.to_timedelta((vs.chainage - self.chainage) / self.speed_ms, unit='s')
            curr_vs_wl_df['shifted_time'] = pd.to_datetime(curr_vs_wl_df['dt']) + curr_vs_wl_df['time_diff']
            curr_vs_wl_df = curr_vs_wl_df.sort_values(by='shifted_time')
            vs_bias = curr_vs_wl_df['vs_wl'].loc[(self.juxtaposed_wl['dt'].min() < curr_vs_wl_df['dt']) & (
                        curr_vs_wl_df['dt'] < self.juxtaposed_wl['dt'].max())].mean() - self.juxtaposed_wl[
                          'vs_wl'].mean()
            curr_vs_wl_df['shifted_wl'] = curr_vs_wl_df['vs_wl'] - vs_bias
            curr_vs_wl_df['vs_chain'] = vs.chainage
            multi_vs_wl_df = pd.concat([multi_vs_wl_df, curr_vs_wl_df])
        multi_vs_wl_df = multi_vs_wl_df.set_index(pd.to_datetime(multi_vs_wl_df['shifted_time']))
        self.densified_wl = multi_vs_wl_df.sort_index()

    def get_densified_wl_by_norm(self, ts, speed_ms, norm_method):
        self.speed_ms = speed_ms
        multi_vs_wl_df = ts[self.upstream_adjacent_vs[0].juxtaposed_wl.columns].copy()
        multi_vs_wl_df['time_diff'] = pd.to_timedelta(0, unit='s')
        multi_vs_wl_df['shifted_time'] = pd.to_datetime(multi_vs_wl_df['dt']) + multi_vs_wl_df['time_diff']
        if norm_method == 'standard':
            scaler_standard = StandardScaler()
            multi_vs_wl_df['normalized_wl'] = pd.DataFrame(scaler_standard.fit_transform(multi_vs_wl_df[['vs_wl']]), index=multi_vs_wl_df.index, columns=['vs_wl'])['vs_wl']
        elif norm_method in ['minmax', 'minmax_percentile']:
            perc1, perc2 = .03, .85
            # vs_min, vs_max = multi_vs_wl_df['vs_wl'].min(), multi_vs_wl_df['vs_wl'].max()
            vs_min, vs_max = multi_vs_wl_df['vs_wl'].quantile(perc1), multi_vs_wl_df['vs_wl'].quantile(perc2)
            multi_vs_wl_df['normalized_wl'] = (multi_vs_wl_df['vs_wl'] - vs_min) / (vs_max - vs_min)
        elif norm_method == 'median':
            vs_median = multi_vs_wl_df['vs_wl'].median()
            vs_iqr = multi_vs_wl_df['vs_wl'].quantile(0.75) - multi_vs_wl_df['vs_wl'].quantile(0.25)
            multi_vs_wl_df['normalized_wl'] = (multi_vs_wl_df['vs_wl'] - vs_median) / vs_iqr

        multi_vs_wl_df['vs_chain'] = self.chainage
        for vs in self.upstream_adjacent_vs:
            cond1 = len(vs.juxtaposed_wl) == 0
            cond2 = vs.id == self.id
            if cond1 or cond2:
                continue
            curr_vs_wl_df = vs.juxtaposed_wl.copy()
            curr_vs_wl_df['time_diff'] = pd.to_timedelta((vs.chainage - self.chainage) / self.speed_ms, unit='s')
            curr_vs_wl_df['shifted_time'] = pd.to_datetime(curr_vs_wl_df['dt']) + curr_vs_wl_df['time_diff']
            curr_vs_wl_df = curr_vs_wl_df.sort_values(by='shifted_time')
            if norm_method == 'standard':
                curr_scaler = StandardScaler()
                curr_vs_wl_df['normalized_wl'] = pd.DataFrame(curr_scaler.fit_transform(curr_vs_wl_df[['vs_wl']]), index=curr_vs_wl_df.index, columns=['vs_wl'])['vs_wl']
            elif norm_method in ['minmax', 'minmax_percentile']:
                # curr_min, curr_max = curr_vs_wl_df['vs_wl'].min(), curr_vs_wl_df['vs_wl'].max()
                curr_min, curr_max = curr_vs_wl_df['vs_wl'].quantile(perc1), curr_vs_wl_df['vs_wl'].quantile(perc2)
                curr_vs_wl_df['normalized_wl'] = (curr_vs_wl_df['vs_wl'] - curr_min) / (curr_max - curr_min)
            elif norm_method == 'median':
                curr_median = curr_vs_wl_df['vs_wl'].median()
                curr_iqr = curr_vs_wl_df['vs_wl'].quantile(0.75) - curr_vs_wl_df['vs_wl'].quantile(0.25)
                curr_vs_wl_df['normalized_wl'] = (curr_vs_wl_df['vs_wl'] - curr_median) / curr_iqr

            curr_vs_wl_df['vs_chain'] = vs.chainage
            multi_vs_wl_df = pd.concat([multi_vs_wl_df, curr_vs_wl_df])
        multi_vs_wl_df = multi_vs_wl_df.set_index(pd.to_datetime(multi_vs_wl_df['shifted_time']))
        if norm_method == 'standard':
            multi_vs_wl_df['shifted_wl'] = pd.DataFrame(scaler_standard.inverse_transform(multi_vs_wl_df[['normalized_wl']]), columns=['vs_wl'], index=multi_vs_wl_df.index)['vs_wl']
        elif norm_method in ['minmax', 'minmax_percentile']:
            multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['normalized_wl'] * (vs_max - vs_min) + vs_min
        elif norm_method == 'median':
            multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['normalized_wl'] * (vs_iqr) + vs_median

        self.normalized_ts = multi_vs_wl_df.sort_index()

    def get_densified_wl_by_regressions(self, speed_ms):
        self.speed_ms = speed_ms
        self.get_vs_regressions_df()
        multi_vs_wl_df = self.juxtaposed_wl.copy()
        multi_vs_wl_df['time_diff'] = pd.to_timedelta(0, unit='s')
        multi_vs_wl_df['shifted_time'] = pd.to_datetime(multi_vs_wl_df['dt']) + multi_vs_wl_df['time_diff']
        multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['vs_wl']
        multi_vs_wl_df['vs_chain'] = self.chainage
        for vs in self.upstream_adjacent_vs:
            cond1 = len(vs.juxtaposed_wl) == 0
            cond2 = vs.id == self.id
            if cond1 or cond2:
                continue
            curr_vs_wl_df = vs.juxtaposed_wl.copy()
            curr_vs_wl_df['time_diff'] = pd.to_timedelta((vs.chainage - self.chainage) / self.speed_ms, unit='s')
            curr_vs_wl_df['shifted_time'] = pd.to_datetime(curr_vs_wl_df['dt']) + curr_vs_wl_df['time_diff']
            curr_vs_wl_df = curr_vs_wl_df.sort_values(by='shifted_time')
            curr_vs_wl_df['shifted_wl'] = curr_vs_wl_df.apply(get_wl_by_multiple_regressions, axis=1, args=(self.id, self.regressions_df, vs.id,))
            curr_vs_wl_df['vs_chain'] = vs.chainage
            multi_vs_wl_df = pd.concat([multi_vs_wl_df, curr_vs_wl_df])
        multi_vs_wl_df = multi_vs_wl_df.set_index(pd.to_datetime(multi_vs_wl_df['shifted_time']))
        self.densified_wl_by_regr_ts = multi_vs_wl_df.sort_index()

    def get_closest_in_situ_daily_wl(self, gauge_meas_up, t1, t2):
        gauge_meas_up = copy.deepcopy(gauge_meas_up)
        gauge_meas_up.index = pd.to_datetime(gauge_meas_up['shifted_time'])
        ts_up = gauge_meas_up['shifted_wl'].loc[(gauge_meas_up.index > t1) & (gauge_meas_up.index < t2)]
        self.closest_in_situ_daily_wl = ts_up.resample('D').mean()
        # self.closest_in_situ_daily_wl = ts_up.resample('D').mean() - self.mean_g_wl + self.mean_vs_wl
        # print(ts_up.resample('D').mean() - self.mean_g_wl + self.mean_vs_wl)
        # print(self.closest_in_situ_daily_wl)

    def adjust_gauge_data_to_vs_by_mean_diff(self, gauge_meas_up):
        gauge_meas_up = copy.deepcopy(gauge_meas_up)
        mean_g = self.juxtaposed_wl['up_wl'].loc[~self.juxtaposed_wl['vs_wl'].isna()].mean()
        mean_vs = self.juxtaposed_wl['vs_wl'].loc[~self.juxtaposed_wl['up_wl'].isna()].mean()
        gauge_meas_up['shifted_wl'] = gauge_meas_up['stage'] + mean_vs - mean_g
        time_diff = pd.to_timedelta((self.neigh_g_up_chain - self.chainage) / self.speed_ms, unit='s')
        gauge_meas_up['shifted_time'] = pd.to_datetime(gauge_meas_up.index) + time_diff
        return gauge_meas_up

    def juxtapose_gauge_data_to_vs(self, gauge_meas_up):
        # gauge_meas_up = gauge_data.loc[gauge_data['id'] == self.neigh_g_up]
        # gauge_meas_up = gauge_meas_up.set_index(pd.to_datetime(gauge_meas_up['date']))
        # gauge_meas_up = gauge_meas_up.sort_index()
        gauge_meas_up = copy.deepcopy(gauge_meas_up)
        self.densified_wl['shifted_wl_gauge'] = self.densified_wl.apply(get_corresponding_gauge_wl,
                                                                        args=(gauge_meas_up,), axis=1)
        mean_dens_g, mean_dens_vs = self.densified_wl['shifted_wl_gauge'].mean(), self.densified_wl['shifted_wl'].mean()
        self.densified_wl['shifted_wl_gauge_anom'] = self.densified_wl['shifted_wl_gauge'] - mean_dens_g
        self.densified_wl['shifted_wl_anom'] = self.densified_wl['shifted_wl'] - mean_dens_vs
        self.densified_wl['shifted_wl_bias'] = abs(self.densified_wl['shifted_wl_anom'] -
                                                   self.densified_wl['shifted_wl_gauge_anom'])

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

    def get_dist_weighted_wl(self, ts):
        curr_densified = ts.copy()
        curr_densified['dist_weight'] = round(self.buffer - (curr_densified['vs_chain'] - self.chainage) / 1000)
        curr_densified['weight_x_wl'] = curr_densified['shifted_wl'] * curr_densified['dist_weight']
        dist_weighted_daily = curr_densified.resample('D').agg(wl_x_weight_sum=('weight_x_wl', 'sum'),
                                                               weight_sum=('dist_weight', 'sum'))
        dist_weighted_daily['wl_weighted'] = dist_weighted_daily['wl_x_weight_sum'] / dist_weighted_daily['weight_sum']
        return dist_weighted_daily['wl_weighted']

    def get_spline_interpolated_ts(self, ts_to_interpolate, column_str, s):
        df_clean = ts_to_interpolate.dropna()
        x_data = df_clean.index.astype(np.int64) // 10 ** 9
        y_data = df_clean.values
        spl = UnivariateSpline(x_data, y_data, s=s)
        full_dt_range = pd.date_range(start=ts_to_interpolate.index.min(), end=ts_to_interpolate.index.max(), freq='D')
        x_full = full_dt_range.astype(np.int64) // 10 ** 9
        df_full = pd.DataFrame(index=full_dt_range)
        df_full['spline_itpd_wl'] = spl(x_full)
        self.spline_itpd_wl = df_full

    def get_densified_wl_filtered_by_bias(self, bias=1):
        return self.interpolate(self.densified_wl['shifted_wl'].loc[self.densified_wl['bias'] < bias].resample('D').mean())

    def get_rmse_nse_values(self, interpolated, t1, t2, text):
        from sklearn.metrics import mean_squared_error
        import hydroeval as he

        df_combined = pd.concat([self.closest_in_situ_daily_wl, interpolated], axis=1)
        df_combined.columns = ['gauge_mean', 'model_mean']
        df_cleaned = df_combined.dropna()
        df_cleaned = df_cleaned[t1:t2]
        if len(df_cleaned) < 5:
            return np.nan, np.nan
        y_true = df_cleaned['gauge_mean']
        y_predicted = df_cleaned['model_mean']
        rmse = round(np.sqrt(mean_squared_error(y_true, y_predicted)), 3)
        nse = round(he.evaluator(he.nse, y_predicted, y_true)[0], 3)
        print(f'{text} {self.id}, velocity {round(self.speed_ms, 3)} m/s, RMSE: {rmse}m, NSE: {nse}')
        return rmse, nse

    def plot_daily_wl(self):
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(self.interpolated_wl, marker='o', label=f'{self.id} interpolated WL')
        ax.plot(self.closest_in_situ_daily_wl, marker='o', label='daily gauge WL')
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', label='VS data', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_densified_wl(self):
        fig, ax = plt.subplots()
        ax.plot(self.densified_wl.index, self.densified_wl['shifted_wl'], label='densified_wl')
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', label='VS data', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_densified_vs_gauge_color_by_chainage(self):
        from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
        fig, axs = plt.subplot_mosaic([['upper_left', 'upper_right'], ['bottom', 'bottom']],
                                      figsize=(18, 9), height_ratios=[1, 2])
        ax, ax1, ax2 = axs['bottom'], axs['upper_left'], axs['upper_right']
        self.densified_wl['chainage_diff'] = (self.densified_wl['vs_chain'] - self.chainage) / 1000
        n_classes = 10
        min_val = self.densified_wl['chainage_diff'].min()
        max_val = self.densified_wl['chainage_diff'].max()
        bins = np.linspace(min_val, max_val, n_classes + 1)
        colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Green -> Yellow -> Red
        cmap = LinearSegmentedColormap.from_list("my_gyr", colors, N=n_classes)
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)

        scatter = ax.scatter(self.densified_wl['shifted_time'], self.densified_wl['shifted_wl'],
                             c=self.densified_wl['chainage_diff'], cmap=cmap, marker='o',
                             label='water level colored by distance', zorder=2, norm=norm)
        ax.errorbar(self.densified_wl['shifted_time'], self.densified_wl['shifted_wl'], yerr=self.densified_wl['bias'],
                    ecolor='red', label='VS measurement bias')
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', s=50,
                   color='purple', label='water level from reference station', zorder=11)
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

        ax1.scatter(self.densified_wl['chainage_diff'], self.densified_wl['shifted_wl_bias'])
        ax1.set_xlabel('Along-river distance from ref. station')
        ax1.set_ylabel('Bias to shifted gauge WL')

        ax2.scatter(self.densified_wl['bias'], self.densified_wl['shifted_wl_bias'])
        ax2.set_xlabel('Bias of VS measurement')
        ax2.set_ylabel('Bias to shifted gauge WL')

        fig.tight_layout()
        plt.show(block=True)

    def plot_all_approaches(self, not_plotting=['spline']):
        fig, ax = plt.subplots(figsize=(12, 8))
        if self.closest_in_situ_daily_wl is not None:
            ax.plot(self.closest_in_situ_daily_wl, label='in situ', color='black', linewidth=3)
        if self.single_VS_itpd is not None and 'single' not in not_plotting:
            ax.plot(self.single_VS_itpd, label='just itp', color='blue')
            # ax.scatter(self.swot_wl['datetime'], self.swot_wl['wse'], marker='.', color='blue')  # tylko SWOT, bez S3, S6
            # ax.scatter(self.wl.index, self.wl['wse'], marker='.', color='blue') # WSZYSTKIE DANE Z VS, CZYLI SWOT, S3, S6
        if self.interpolated_wl is not None and 'classic' not in not_plotting:
            ax.plot(self.interpolated_wl, label='densified', color='orange')
            ax.scatter(self.densified_wl['shifted_time'], self.densified_wl['shifted_wl'], marker='.', color='orange')
        if self.spline_itpd_wl is not None and 'spline' not in not_plotting:
            ax.plot(self.spline_itpd_wl, label='spline_itpd', color='magenta')
        if self.normalized_ts_itpd is not None and 'norm' not in not_plotting:
            ax.plot(self.normalized_ts_itpd, label='normalized', color='red')
            ax.scatter(self.normalized_ts['shifted_time'], self.normalized_ts['shifted_wl'], marker='.', color='red')
        if self.densified_wl_by_regr_ts is not None and 'regr' not in not_plotting:
            ax.plot(self.densified_wl_by_regr_itpd, label='regressions', color='purple')
            ax.scatter(self.densified_wl_by_regr_ts['shifted_time'], self.densified_wl_by_regr_ts['shifted_wl'],
                       marker='.', color='purple')

        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_all_vs_timeseries(self):
        viridis_cmap = plt.colormaps['viridis']
        colors = [viridis_cmap(i) for i in np.linspace(0, 1, len(self.upstream_adjacent_vs))]
        fig, ax = plt.subplots()
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
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_all_vs_timeseries_with_correlations(self, just_swot=True):
        fig, ax = plt.subplots(figsize=(10, 6))

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
            'label': f'VS {self.id} (Chainage: {self.chainage / 1000:.1f}km)',
            'color': color_curr_vs,
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
                    'label': f'VS {vs.id} (Chainage: {vs.chainage / 1000:.1f}km, Corr: {correlation:.3f})',
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

        ax.set_title('Water Levels vs. Time, Colored by River Chainage')
        ax.set_xlabel('Date')
        ax.set_ylabel('Water Level')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
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
        plt.show()

    def plot_all_gauge_timeseries_within_DS(self, loaded_gauges):
        from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
        import matplotlib.cm as cm  # Import modułu cm, aby uzyskać dostęp do wbudowanych map kolorów
        min_chain = self.chainage
        max_chain = max([x.chainage for x in self.upstream_adjacent_vs])
        gauges = [x for x in loaded_gauges.values() if min_chain < x.chainage < max_chain]
        first_gauge = [x for x in gauges if x.chainage == min([x.chainage for x in gauges])][0]
        first_gauge_mean_wl = first_gauge.wl_df['stage'].loc[self.closest_in_situ_daily_wl.index].mean()

        n_classes = len(gauges)
        bins = sorted(list(set([g.chainage for g in gauges])))
        bins.append(bins[-1] + 1e-6)  # Dodaj małą wartość do ostatniego binu
        bins = np.array(bins)

        cmap = cm.viridis
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
        plt.show()
