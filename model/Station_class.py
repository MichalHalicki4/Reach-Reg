import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from shapely.ops import substring
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
from sklearn.metrics import mean_squared_error
import hydroeval as he
import sklearn.svm as svm
from . import station_utils as s_utils


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
        """
        Uploads water level data (wl_df) to the station object, processes it by setting the date
        as the index, sorting, and resampling the time series  to a standardized frequency (hourly or daily).

        :param wl_df: A pandas DataFrame containing raw water level measurements,
                      expected to have a 'date' column and a 'stage' column.
        :returns: None. The processed DataFrame replaces the existing 'self.wl_df'.
        """
        self.wl_df = wl_df
        self.wl_df = self.wl_df.set_index(pd.to_datetime(self.wl_df['date']))
        self.wl_df = self.wl_df.sort_index()
        if self.sampling in ['hourly', 'h', '15-minute', '10-minute']:
            resample_str = 'h'
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
        self.juxtaposed_wl = pd.DataFrame()
        self.mean_g_wl, self.mean_vs_wl = None, None
        self.slope = None
        self.single_VS_itpd = None

    def __repr__(self):
        return f'Virtual station ID: {self.id}, chainage: {self.chainage} with {len(self.wl)} measurements'

    def is_away_from_river(self, riv_object, distance):
        """
        Checks if the virtual station is farther than a specified distance from the river centerline.

        The method converts the river's simplified geometry and the VS coordinates to a common
        metrical Coordinate Reference System (CRS) and calculates the shortest distance.

        :param riv_object: River object containing the river's geometry ('simplified_river')
                           and metrical CRS ('metrical_crs').
        :param distance: Maximum allowed distance (in meters) from the river.
        :returns: True if the distance is greater than the specified limit, False otherwise.
        """
        riv_series = gpd.GeoSeries(riv_object.simplified_river, crs=riv_object.metrical_crs)
        vs_series = gpd.GeoSeries(Point(self.x, self.y), crs=4326)
        dist = vs_series.to_crs(riv_series.crs).distance(riv_series)
        return dist.values[0] > distance

    def get_water_levels(self, dahiti):
        """
        Downloads water level data and related metadata (geoid, bias, slope correction)
        for the Virtual Station from the DAHITI API.

        The method handles potential download errors (PermissionError, general exceptions),
        stores key metadata attributes, and organizes the time series data (wl) into a
        pandas DataFrame, separately isolating SWOT mission data (swot_wl).

        :param dahiti: An instance of the DAHITI API client class, used for downloading data.
        :returns: None, but updates several attributes of the station object (self.river,
                  self.wl, self.swot_wl, etc.).
        """
        try:
            try:
                wl_data = dahiti.download_water_level(self.id, parameters=['mission'])
            except PermissionError:
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
        """
        Filters the main water level time series (self.wl) and the separate SWOT time
        series (self.swot_wl) to include only measurements between the specified start
        time (t1) and end time (t2).

        :param t1: Start datetime for filtering (exclusive).
        :param t2: End datetime for filtering (exclusive).
        :returns: None. The DataFrames are filtered in place.
        """
        self.wl = self.wl.loc[(self.wl['datetime'] > t1) & (self.wl['datetime'] < t2)]
        self.swot_wl = self.swot_wl.loc[
            (pd.to_datetime(self.swot_wl['datetime']) > t1) & (pd.to_datetime(self.swot_wl['datetime']) < t2)]

    def upload_chainage(self, chainage):
        """
        Sets the chainage (distance along the river centerline) for the Virtual Station.

        :param chainage: The calculated distance along the river (in meters).
        :returns: None.
        """
        self.chainage = chainage

    def find_closest_gauge_and_chain(self, gauges_chains):
        """
        Identifies the immediate upstream and downstream Gauge Stations (GS) relative
        to the Virtual Station's chainage.

        This logic handles edge cases where the VS is at the very upstream or very
        downstream end of the available GS data.

        :param gauges_chains: A dictionary of GaugeStation objects (ID: object), used to
                              extract their IDs and chainages.
        :returns: None, but sets 'self.neigh_g_up', 'self.neigh_g_dn', and their respective chainages.
        """
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
        """
        Compares the VS water level measurements with data from the closest upstream
        and/or downstream Gauge Stations (GS) after applying a time-shift correction.

        The method calculates the required time shift based either on a simple velocity
        model or on cross-correlation between upstream and downstream GS data, then
        searches for the closest GS measurement around the estimated time. It computes
        water level anomalies and bias for comparison.

        :param gauge_meas_up: Time series DataFrame for the upstream GS.
        :param gauge_meas_down: Time series DataFrame for the downstream GS.
        :param gdata_sampling: Sampling frequency of the gauge data (e.g., 'daily').
        :param velocity: Assumed wave propagation velocity (in m/s) for time shifting. Optional.
        :returns: None, but populates 'self.juxtaposed_wl' with the comparison results.
        """
        hours_to_juxtapose = 12
        juxtaposed_columns = ['id_vs', 'vs_chain', 'dt', 'mission', 'gauge_up', 'dist_up', 'gauge_down', 'dist_down',
                              'lag',
                              'vs_wl', 'g_wl', 'uncertainty', 'g_anom', 'vs_anom', 'bias']
        juxtaposed_data = []
        for index, row in self.wl.iterrows():
            if gauge_meas_up is None and gauge_meas_down is None:
                juxtaposed_data.append(
                    [self.id, self.chainage, row['datetime'], row['mission'], self.neigh_g_up, self.neigh_g_up_chain, self.neigh_g_dn,
                     self.neigh_g_dn_chain, None, row['wse'], None, row['wse_u']])
                continue
            vs_wl, vs_dt = row[['wse', 'datetime']]
            vs_dt_prev = vs_dt - pd.to_timedelta('5 days')
            try:
                dist_up = abs(self.neigh_g_up_chain - self.chainage)
                dist_dn = abs(self.neigh_g_dn_chain - self.chainage)
                self.closest_gauge = 'up' if dist_up < dist_dn else 'dn'
            except TypeError:
                self.closest_gauge = 'up' if type(gauge_meas_up) == pd.DataFrame else 'dn'
            closest_gdata = gauge_meas_up if self.closest_gauge == 'up' else gauge_meas_down
            closest_chain = self.neigh_g_up_chain if self.closest_gauge == 'up' else self.neigh_g_dn_chain
            if gdata_sampling == 'daily' or velocity is not None or gauge_meas_down is None or gauge_meas_up is None:
                vel = 1 if velocity is None else velocity
                shift = pd.to_timedelta((closest_chain - self.chainage) / vel, unit='s')
                gauge_time = (vs_dt - shift).round('h')
                final_lag = round(shift.total_seconds() / 3600)
            else:
                ts_up = gauge_meas_up['stage'].loc[(gauge_meas_up.index > vs_dt_prev) & (gauge_meas_up.index < vs_dt)]
                ts_dn = gauge_meas_down['stage'].loc[
                    (gauge_meas_down.index > vs_dt_prev) & (gauge_meas_down.index < vs_dt)]

                lag, corr = s_utils.get_optimum_lag(ts_dn, ts_up, 50)
                ratio = (closest_chain - self.chainage) / (self.neigh_g_up_chain - self.neigh_g_dn_chain)
                final_lag = lag * ratio

                gauge_time = (vs_dt - pd.to_timedelta(f'{final_lag} hours')).round('h')
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
        """
        Generates a plot comparing the water level anomalies of the Virtual Station
        with the anomalies of the juxtaposed Gauge Station.

        Raises a ValueError if no gauge data is available for plotting.

        :returns: None, but displays a plot.
        """
        if self.juxtaposed_wl['g_anom'].isnull().all():
            raise ValueError('No gauge data to plot')
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
        """
        Resamples the VS water level time series (either all data or just SWOT data)
        to daily frequency, then linearly interpolates any resulting gaps.

        :param just_swot: If True, uses only SWOT measurements; otherwise, uses all available data.
        :returns: A pandas Series containing the daily, linearly interpolated water levels.
        """
        if just_swot:
            vs_wl_to_corr = self.wl.loc[self.wl['mission'].str.contains('SWOT', na=False)].copy()
        else:
            vs_wl_to_corr = self.wl.copy()
        return vs_wl_to_corr['wse'].resample('D').mean().interpolate(method='linear')


class DensificationStation(VirtualStation):
    def __init__(self, vs_object, buffer, itpd_method, speed_ms=None):
        super().__init__(vs_object.id, vs_object.x, vs_object.y)
        self.cval_buff = 0.01
        self.__dict__.update(vs_object.__dict__)
        self.buffer, self.upstream_adjacent_vs = buffer, None
        self.regressions_df = None
        self.itpd_method = itpd_method
        self.speed_ms = speed_ms
        self.closest_in_situ_daily_wl = pd.DataFrame()
        self.densified_ts, self.densified_daily, self.densified_itpd = None, None, None
        self.slopes_dict, self.c = None, None
        self.rmse_thres, self.single_rmse_thres = None, None
        self.rmse, self.nse = None, None

    def __repr__(self):
        return f'Densification station ID: {self.id}, chainage: {self.chainage} with' \
               f' {len(self.upstream_adjacent_vs)} VS within buffer'

    def get_upstream_adjacent_vs(self, vs_list):
        """
        Selects all Virtual Stations (VS) that fall within a defined chainage buffer
        around the current station (self).

        The buffer distance is symmetrical (both upstream and downstream). The selected
        stations are then deep-copied and sorted by chainage.

        :param vs_list: A list of all available VirtualStation objects.
        :returns: None, but updates 'self.upstream_adjacent_vs'.
        """
        selected_list = []
        for vs in vs_list:
            if self.chainage - self.buffer * 1000 <= vs.chainage < self.chainage + self.buffer * 1000:
                selected_list.append(copy.deepcopy(vs))
        self.upstream_adjacent_vs = sorted(selected_list, key=lambda k: k.chainage)

    def filter_stations_by_corr_amp_dams_tribs_other(self, corr_thres, amp_thres, dams, tributary_reaches,
                                                     other_reaches, just_swot):
        """
        Applies a sequence of filters to the initially selected adjacent VS list based on:
        1. Correlation threshold (spatial and temporal similarity).
        2. Water level amplitude similarity.
        3. Exclusion of VS affected by dams or tributary junctions.
        4. Exclusion of VS specified in a general exclusion list ('other_reaches').

        :param corr_thres: Minimum acceptable correlation coefficient.
        :param amp_thres: Maximum acceptable fractional difference in water level amplitude.
        :param dams: List of dam locations.
        :param tributary_reaches: List of tributary reach definitions.
        :param other_reaches: General list of VS IDs to exclude (e.g., those near dams).
        :param just_swot: Boolean to use only SWOT data for correlation calculation.
        :returns: None, updates 'self.upstream_adjacent_vs' in place.
        """
        self.filter_upstream_stations_by_correlation(corr_thres, just_swot)
        self.filter_upstream_stations_by_wl_amplitude(amp_thres)
        self.filter_upstream_stations_by_dams_and_tributaries(dams, tributary_reaches)
        self.upstream_adjacent_vs = [x for x in self.upstream_adjacent_vs if x.id not in other_reaches]

    def filter_upstream_stations_by_correlation(self, corr_thres, just_swot=False):
        """
        Filters adjacent VS based on a minimum correlation coefficient with the target
        station (self). Correlation is calculated on daily, linearly interpolated time series.

        :param corr_thres: Minimum acceptable Pearson correlation coefficient.
        :param just_swot: If True, uses only SWOT data for correlation calculation.
        :returns: None, updates 'self.upstream_adjacent_vs'.
        """
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
        """
        Filters adjacent VS whose water level amplitude falls outside a defined fractional
        range of the target station's amplitude.

        :param thres: Maximum fractional deviation from the target station's amplitude (e.g., 0.1 for 10%).
        :returns: None, updates 'self.upstream_adjacent_vs'.
        """
        stations = []
        ds_amp = self.wl['wse'].max() - self.wl['wse'].min()
        for vs in self.upstream_adjacent_vs:
            vs_amp = vs.wl['wse'].max() - vs.wl['wse'].min()
            if ds_amp + thres * ds_amp > vs_amp > ds_amp - thres * ds_amp:
                stations.append(vs)
        self.upstream_adjacent_vs = stations

    def filter_upstream_stations_by_dams_and_tributaries(self, dams, tributary_reaches):
        """
        Filters adjacent VS if a dam or a major tributary junction is located between
        the VS and the target station (self). This is a hydraulic consistency filter.

        :param dams: List of dam locations.
        :param tributary_reaches: List of tributary reach definitions.
        :returns: None, updates 'self.upstream_adjacent_vs'.
        """
        stations = []
        for vs in self.upstream_adjacent_vs:
            cond1 = not s_utils.is_dam_between(self.chainage, vs.chainage, dams)
            cond2 = not s_utils.is_tributary_between(self.chainage, vs.chainage, tributary_reaches)
            if cond1 and cond2:
                stations.append(vs)
        self.upstream_adjacent_vs = stations

    def filter_stations_only_with_swot(self):
        """
        Filters adjacent VS, keeping only those that contain at least one SWOT mission measurement.

        :returns: None, updates 'self.upstream_adjacent_vs'.
        """
        vs_with_swot_data = []
        for vs in self.upstream_adjacent_vs:
            if len(vs.wl.loc[vs.wl['mission'].str.contains('SWOT', na=False)]) > 0:
                vs_with_swot_data.append(vs)
        self.upstream_adjacent_vs = vs_with_swot_data

    def is_ds_empty_or_at_edge(self):
        """
        Checks for various conditions that would make densification impossible or unreliable:
        1. No adjacent VS after filtering.
        2. No SWOT data for the target station itself.
        3. No VS upstream or no VS downstream (meaning the section is too short or is at an edge).

        :returns: True if any condition is met, False otherwise.
        """
        cond1 = len(self.upstream_adjacent_vs) == 0
        cond2 = len(self.swot_wl) == 0
        cond3 = len([x for x in self.upstream_adjacent_vs if x.chainage > self.chainage]) == 0
        cond4 = len([x for x in self.upstream_adjacent_vs if x.chainage < self.chainage]) == 0
        if cond1 or cond2 or cond3 or cond4:
            return True

    def get_slope_of_all_vs(self):
        """
        Calculates the slope at the location of each VS (including self)
        using its nearest upstream and downstream neighbors within the filtered set.

        :returns: None, but updates the 'slope' attribute of each VS object.
        """
        for vs in [self] + self.upstream_adjacent_vs:
            neigh_bef, neigh_aft = s_utils.get_vs_neighbors(vs, self.upstream_adjacent_vs)
            vs.slope = s_utils.get_slope(neigh_bef, neigh_aft)

    def get_depths_of_all_vs(self, bottom_thres=0.1):
        """
        Estimates the water depth time series for all involved VS (including self)
        based on a simplified assumption for the river bottom height.

        The bottom height is set as the minimum observed WSE minus a fractional threshold
        (bottom_thres) of the water level amplitude (range).

        :param bottom_thres: Fractional factor (e.g., 0.1) used to estimate the river
                             bottom below the minimum observed water level.
        :returns: None, but updates the 'waterdepths' attribute of each VS object.
        """
        for vs in [self] + self.upstream_adjacent_vs:
            curr_amplitude = vs.wl['wse'].max() - vs.wl['wse'].min()
            bottom_height = vs.wl['wse'].min() - bottom_thres * curr_amplitude
            vs.waterdepths = vs.wl['wse'] - bottom_height

    def get_mean_slope_to_vs(self, vs_id):
        """
        Calculates the mean water surface slope (in m/km) between the target station
        (self) and a specified adjacent VS.

        It uses the difference in mean WSE divided by the chainage difference.

        :param vs_id: ID of the adjacent Virtual Station.
        :returns: The mean slope in meters per kilometer (m/km), rounded to 3 decimal places.
        """
        curr_vs = [x for x in self.upstream_adjacent_vs if x.id == vs_id][0]
        curr_vs_wl, curr_vs_chain = curr_vs.wl['wse'].mean(), curr_vs.chainage
        ds_wl = self.wl['wse'].mean()
        chain_diff = abs(self.chainage - curr_vs_chain) / 1000
        return round(abs(curr_vs_wl - ds_wl) / chain_diff, 3)

    def get_mean_slopes_dict(self):
        """
        Calculates the mean water surface slope from the target station (self) to every
        adjacent VS and stores the results in a dictionary (VS ID: mean slope).

        :returns: A dictionary mapping VS IDs to their calculated mean slope to the DS.
        """
        slopes_dict = {}
        for vs in self.upstream_adjacent_vs:
            slopes_dict[vs.id] = self.get_mean_slope_to_vs(vs.id)
        return slopes_dict

    def get_bottom_heights_dict(self, bottom_thres=0.1):
        """
        Calculates the estimated river bottom height for all involved VS (including self)
        based on the minimum observed WSE and the given threshold.

        :param bottom_thres: Fractional factor used in the bottom estimation.
        :returns: A dictionary mapping VS IDs to their estimated bottom height (in meters).
        """
        bottom_heights = {}
        for vs in [self] + self.upstream_adjacent_vs:
            curr_amplitude = vs.wl['wse'].max() - vs.wl['wse'].min()
            bottom_heights[vs.id] = vs.wl['wse'].min() - bottom_thres * curr_amplitude
        return bottom_heights

    def get_vs_regressions_df_extended(self, res_str='h'):
        """
        Calculates linear regression coefficients (y = ax + b) for WsE between:
        1. All direct neighbors (1 step).
        2. Stations separated by one intermediate station (2 steps/skips).

        These regressions form the basis for the densification pathfinding algorithm.

        :param res_str: Resampling string (e.g., 'h' for hourly) for time series used in regression.
        :returns: The DataFrame containing all calculated regression coefficients.
        """
        regressions = []
        all_stations = sorted([self] + self.upstream_adjacent_vs, key=lambda x: x.chainage)

        for i in range(len(all_stations) - 1):
            vs1 = all_stations[i]
            vs2 = all_stations[i + 1]

            if vs1.chainage < vs2.chainage:
                st_low_chain = vs1
                st_high_chain = vs2
            else:
                st_low_chain = vs2
                st_high_chain = vs1

            a, b, r2, rmse, data_len = s_utils.get_linear_regression_coeffs_btwn_stations(st_low_chain, st_high_chain,
                                                                                          res_str)

            if a is not None:
                regressions.append({
                    'st1': st_low_chain.id, 'st2': st_high_chain.id,
                    'st1_chain': st_low_chain.chainage, 'st2_chain': st_high_chain.chainage,
                    'a': a, 'b': b, 'r2': r2, 'rmse': rmse, 'data_len': data_len,
                    'steps': 1
                })

        for i in range(len(all_stations) - 2):
            vs1 = all_stations[i]
            vs3 = all_stations[i + 2]

            if vs1.chainage < vs3.chainage:
                st_low_chain = vs1
                st_high_chain = vs3
            else:
                st_low_chain = vs3
                st_high_chain = vs1

            a, b, r2, rmse, data_len = s_utils.get_linear_regression_coeffs_btwn_stations(st_low_chain, st_high_chain,
                                                                                          res_str)

            if a is not None:
                regressions.append({
                    'st1': st_low_chain.id, 'st2': st_high_chain.id,
                    'st1_chain': st_low_chain.chainage, 'st2_chain': st_high_chain.chainage,
                    'a': a, 'b': b, 'r2': r2, 'rmse': rmse, 'data_len': data_len,
                    'steps': 2
                })

        self.regressions_df = pd.DataFrame(regressions).drop_duplicates(subset=['st1', 'st2'])
        return self.regressions_df

    def get_used_regressions(self):
        """
        Analyzes the densified time series ('self.densified_ts') to identify all unique
        inter-station regression paths that were actually used by the pathfinding algorithm.

        It then compares these used paths with the full set of pre-calculated
        regressions ('self.regressions_df') to track which regressions contributed to
        the final result. It also includes "fallback" regressions if a path was not
        found in the pre-calculated set.

        :returns: A DataFrame detailing the regressions actually used, including their
                  RMSE and number of steps (1, 2, or 3 for fallback).
        """
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
                    pass

                if not found:
                    try:
                        x = self.regressions_df.loc[
                            (self.regressions_df['st2'] == vs_id) & (self.regressions_df['st1'] == vs2_id)
                            ].iloc[0].values
                        used_regressions.append([vs_id, vs2_id, x[-3], x[-1]])
                        found = True
                    except IndexError:
                        pass

                if not found:
                    print(vs_id, vs2_id, 'Not used!')
                    vs1 = [x for x in [self] + self.upstream_adjacent_vs if x.id == vs_id][0]
                    vs2 = [x for x in [self] + self.upstream_adjacent_vs if x.id == vs2_id][0]
                    a, b, r2, rmse, data_len = s_utils.get_linear_regression_coeffs_btwn_stations(vs1, vs2, 'h')
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
        """
        Clips the river centerline geometry to the section spanned by the most upstream
        and most downstream adjacent VS in the filtered list.

        This is typically used for visualization purposes (plotting the working reach).

        :param current_river: River object containing the full river geometry.
        :returns: A GeoSeries object containing the clipped river centerline segment.
        """
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

    def get_densified_wl_by_regressions(self, without_rs=False, rmse_thres=0.5, single_rmse_thres=0.2,
                                        res_str='h'):
        """
        Performs the main densification process by:
        1. Calculating all necessary inter-station regressions.
        2. Iterating through all measurements in adjacent VS.
        3. Applying the pathfinding algorithm (via s_utils.calculate_path_for_row) to find
           the optimal regression path from each measurement's origin VS to the target
           station (self).
        4. Storing the resulting shifted water level ('shifted_wl') and the cumulative
           regression error ('rmse_sum').

        :returns: None, but populates 'self.densified_ts'.
        """
        self.get_vs_regressions_df_extended(res_str)
        self.rmse_thres = rmse_thres
        self.single_rmse_thres = single_rmse_thres
        if without_rs:
            multi_vs_wl_df = pd.DataFrame(columns=self.juxtaposed_wl.columns)
        else:
            multi_vs_wl_df = self.juxtaposed_wl.copy()
            multi_vs_wl_df['shifted_wl'] = multi_vs_wl_df['vs_wl']
            multi_vs_wl_df['rmse_sum'] = 0
            multi_vs_wl_df['vs_chain'] = self.chainage
        for vs in self.upstream_adjacent_vs:
            cond1 = len(vs.juxtaposed_wl) == 0
            cond2 = vs.id == self.id
            if cond1 or cond2:
                continue
            curr_vs_wl_df = vs.juxtaposed_wl.copy()
            vs_list = sorted([self] + self.upstream_adjacent_vs, key=lambda x: x.chainage)
            curr_vs_wl_df[['shifted_wl', 'rmse_sum', 'regr_path']] = curr_vs_wl_df.apply(s_utils.calculate_path_for_row,
                                                                                         axis=1,
                                                                                         args=(
                                                                                             self.id,
                                                                                             self.regressions_df,
                                                                                             vs_list,
                                                                                             self.single_rmse_thres,
                                                                                             self.rmse_thres),
                                                                                         result_type='expand')
            curr_vs_wl_df = curr_vs_wl_df.dropna(subset=['shifted_wl'])
            multi_vs_wl_df = pd.concat(
                [multi_vs_wl_df.dropna(axis=1, how='all'), curr_vs_wl_df.dropna(axis=1, how='all')])
        self.densified_ts = multi_vs_wl_df

    def calculate_shifted_time_by_curve(self, ts, vel_df):
        """
        Calculates the required time shift based on a rating curve that relates
        water level to wave propagation velocity.

        It normalizes water levels and uses 'merge_asof' to match the current
        water level to the closest velocity from the curve, then calculates the shift.

        :param ts: The time series DataFrame (e.g., densified or adjusted gauge data).
        :param vel_df: DataFrame containing the velocity rating curve (waterlevel, velocity).
        :returns: The time series DataFrame with calculated time shift and updated index.
        """
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
        """
        Calculates the time shift (celerity) required for each measurement to travel from its origin
        VS to the target station (self) using a simplified form of Manning's equation.

        It calculates water depth (H) based on an estimated river bottom and then
        estimates velocity (V) using Manning's V = (1/c) * H^(2/3) * S^(1/2).

        :param ts: The densified time series DataFrame ('self.densified_ts').
        :param bottom: Factor for estimating the river bottom height (e.g., 0.1 of amplitude).
        :param c: Manning's roughness coefficient. If None, uses 'self.c'.
        :returns: The time series DataFrame updated with 'velocity', 'time_diff', and a new 'shifted_time' index.
        """
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
        """
        Optimizes the Manning's roughness coefficient ('c') by iterating over a range of 'c' values,
        applying time-shifting, and evaluating the resulting Cross-Validation (CVAL) RMSE against
        the target station's true data.

        The final 'c' value is chosen to minimize CVAL RMSE while also aiming for a
        mean velocity close to the middle of the acceptable velocity range (within the CVAL buffer).

        :param bottom: Factor for estimating the river bottom height.
        :returns: None, but sets 'self.c' to the optimal Manning's coefficient.
        """
        slf = copy.deepcopy(self)
        calibration_accuracies = []
        for c in [x / 500 for x in range(10, 51)]:
            slf.densified_ts = slf.calculate_shifted_time_by_simplified_mannig(slf.densified_ts, bottom, c)
            df_true = slf.swot_wl[['wse']].set_index(pd.to_datetime(slf.swot_wl['datetime'])).resample(
                'D').mean().dropna()
            cval_rmse = slf.get_rmse_of_cval_ts(slf.densified_ts, df_true)
            amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
            wl_amplitude = slf.densified_ts['shifted_wl'].max() - slf.densified_ts[
                'shifted_wl'].min()
            rms_thr = wl_amplitude * amp_thres_final
            slf.densified_ts = slf.densified_ts.loc[
                slf.densified_ts['rmse_sum'] < rms_thr]
            slf.densified_ts = s_utils.filter_outliers_by_tstudent_test(slf.densified_ts)

            densified_ts_cval = slf.densified_ts.loc[
                slf.densified_ts['id_vs'] != slf.id]
            densified_ts_cval_daily = s_utils.get_rmse_weighted_wl(densified_ts_cval)
            densified_ts_cval_itpd = slf.interpolate(densified_ts_cval_daily)
            rmse_cval, nse_cval = slf.get_rmse_nse_values(densified_ts_cval_itpd, 'CrossVal', df_true, False)
            calibration_accuracies.append([c, slf.speed_ms, rmse_cval])

        df_calib = pd.DataFrame(calibration_accuracies, columns=['c', 'velocity', 'rmse_cval'])
        rmse_cval, vels, c_cvals = df_calib['rmse_cval'], df_calib['velocity'], df_calib['c']
        min_index, min_cval = rmse_cval.idxmin(), rmse_cval.min()
        cval_range = rmse_cval[rmse_cval < min_cval + self.cval_buff]
        vels_at_cval_range = vels[cval_range.index]
        mean_vel = (vels_at_cval_range.max() + vels_at_cval_range.min()) / 2
        mean_vel_idx = (vels_at_cval_range - mean_vel).abs().idxmin()
        c_cval = c_cvals[mean_vel_idx]
        self.c = c_cval

    def get_rmse_of_cval_ts(self, timeseries, val_ts):
        """
        Calculates the Cross-Validation (CVAL) RMSE of the densified time series against
        a reference time series (usually the target station's SWOT data).

        CVAL uses only measurements from adjacent VS (excluding the target station's own data).

        :param timeseries: The full densified time series ('self.densified_ts').
        :param val_ts: The ground truth time series (e.g., daily SWOT data).
        :returns: The calculated CVAL RMSE (Root Mean Square Error).
        """
        ts_cval = timeseries.loc[timeseries['id_vs'] != self.id]
        ts_daily = s_utils.get_rmse_weighted_wl(ts_cval)
        ts_itpd = self.interpolate(ts_daily)
        r, n = self.get_rmse_nse_values(ts_itpd, '', val_ts, False)
        return r

    def get_rmse_agg_threshold(self, df_true):
        """
        Calculates a threshold value used to filter out densified measurements with
        an excessively high cumulative regression error ('rmse_sum').

        The threshold is based on the water level amplitude, scaled by a factor (0.1 or 0.2)
        that depends on the CVAL RMSE.

        :param df_true: The ground truth time series (daily SWOT data).
        :returns: The calculated RMSE aggregation threshold.
        """
        cval_rmse = self.get_rmse_of_cval_ts(self.densified_ts, df_true)
        amp_thres_final = 0.1 if cval_rmse < 0.1 else 0.2
        wl_amplitude = self.densified_ts['shifted_wl'].max() - self.densified_ts[
            'shifted_wl'].min()
        return wl_amplitude * amp_thres_final

    def get_closest_in_situ_daily_wl(self, gauge_meas, t1, t2):
        """
        Extracts and resamples the shifted and adjusted Gauge Station (GS) data to
        a daily average time series, effectively creating the daily ground truth WSE
        at the VS location.

        :param gauge_meas: The DataFrame of adjusted GS measurements.
        :param t1: Start time for filtering.
        :param t2: End time for filtering.
        :returns: None, but sets 'self.closest_in_situ_daily_wl'.
        """
        gauge_meas = copy.deepcopy(gauge_meas)
        gauge_meas.index = pd.to_datetime(gauge_meas['shifted_time'])
        ts = gauge_meas['shifted_wl'].loc[(gauge_meas.index > t1) & (gauge_meas.index < t2)]
        self.closest_in_situ_daily_wl = ts.resample('D').mean()

    def adjust_gauge_data_to_vs_by_regr(self, gauge_meas, g_chain, bottom_thres=0.1, vel_df=pd.DataFrame()):
        """
        Adjusts the WSE values from the Gauge Station (GS) to the height of the Virtual
        Station (VS) using the regression coefficients derived from their juxtaposed
        measurements. It also applies a time shift.

        The time shift is calculated either using a velocity curve ('vel_df') or the
        simplified Manning's equation (using 'self.c' and mean slope).

        :param gauge_meas: Raw GS time series data.
        :param g_chain: Chainage of the Gauge Station.
        :param bottom_thres: Factor for bottom estimation (used if Manning's is applied).
        :param vel_df: Optional velocity rating curve.
        :returns: The DataFrame of GS measurements, adjusted for WSE and time-shifted to the VS location.
        """
        gauge_meas = copy.deepcopy(gauge_meas)
        # time_diff = pd.to_timedelta((self.neigh_g_up_chain - self.chainage) / self.speed_ms, unit='s')
        gauge_meas = gauge_meas.loc[self.juxtaposed_wl['dt'].min(): self.juxtaposed_wl['dt'].max()]
        a, b, r2, num_of_meas = s_utils.get_regression_coeffs_from_df(
            self.juxtaposed_wl.dropna(subset=['vs_wl', 'g_wl'], how='any'), 'vs_wl', 'g_wl')
        gauge_meas['shifted_wl'] = a * gauge_meas['stage'] + b
        gauge_meas.loc[:, 'vs_chain'] = g_chain
        gauge_meas.loc[:, 'dt'] = pd.to_datetime(gauge_meas.index)
        if len(vel_df) > 0:
            gauge_meas = self.calculate_shifted_time_by_curve(gauge_meas, vel_df)
        else:
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

    def get_single_vs_interpolated_ts(self):
        """
        Creates a daily, linearly interpolated time series using only the raw
        measurements from the target Virtual Station (self).

        :returns: None, but sets 'self.single_VS_itpd'.
        """
        if self.wl['mission'].str.contains('SWOT').any():
            filtered_wl = self.wl[self.wl['mission'].str.contains('SWOT', na=False)]
        else:
            filtered_wl = self.wl
        resampled = pd.Series(filtered_wl.wse, index=filtered_wl.index)
        daily_wl = resampled.resample('D').mean()
        self.single_VS_itpd = self.interpolate(daily_wl)

    def get_svr_smoothed_data(self, c=100, gamma=0.0001, epsilon=0.1):
        """
        Applies Support Vector Regression (SVR) to the densified time series
        ('self.densified_ts') for final smoothing.

        Measurements are weighted inversely proportional to their cumulative regression
        error ('rmse_sum'), giving higher weights to more reliable points.

        :param c, gamma, epsilon: Hyperparameters for the SVR model.
        :returns: None, but updates 'self.densified_ts', 'self.densified_daily', and 'self.densified_itpd'.
        """
        input_df = self.densified_ts.copy()
        eps = 1e-6
        max_weight_factor = 30
        base_weights = 1 / (input_df['rmse_sum'] + eps)
        min_weight = base_weights.min()
        weights = base_weights / min_weight
        weights = weights.round().astype(int)
        weights = weights.clip(upper=max_weight_factor)
        svr_rbf = svm.SVR(kernel='rbf', C=c, gamma=gamma, epsilon=epsilon)

        index_dates = input_df.index
        start_time = index_dates.min()
        time_deltas = index_dates - start_time
        x_hours = time_deltas.total_seconds() / 3600
        x_train = x_hours.values.reshape(-1, 1)
        svr_rbf.fit(x_train, input_df['shifted_wl'], sample_weight=weights)
        y_res = svr_rbf.predict(x_train)
        input_df['shifted_wl'] = y_res
        y_res_series_daily = s_utils.get_rmse_weighted_wl(input_df)
        y_res_series_itpd = y_res_series_daily.interpolate(method=self.itpd_method)
        self.densified_ts, self.densified_daily, self.densified_itpd = input_df, y_res_series_daily, y_res_series_itpd

    def get_rmse_nse_values(self, interpolated, text, other_validation_ts=pd.Series(), print_res=True):
        """
        Calculates the Root Mean Square Error (RMSE) and Nash-Sutcliffe Efficiency (NSE)
        between the interpolated model output and a reference time series (either the closest
        daily in-situ data or another validation time series provided).

        :param interpolated: The interpolated time series from the model (e.g., self.densified_itpd).
        :param text: Label for printing results.
        :param other_validation_ts: Optional, external time series for validation (e.g., raw VS data).
        :param print_res: If True, prints the results to console.
        :returns: Tuple (rmse, nse) or (np.nan, np.nan) if insufficient data points are available.
        """
        if len(other_validation_ts) > 0:
            df_combined = pd.concat([other_validation_ts, interpolated], axis=1)
        else:
            df_combined = pd.concat([self.closest_in_situ_daily_wl, interpolated], axis=1)
        df_combined.columns = ['gauge_mean', 'model_mean']
        df_cleaned = df_combined.dropna()
        if len(df_cleaned) < 5:
            return np.nan, np.nan
        y_true = df_cleaned['gauge_mean']
        y_predicted = df_cleaned['model_mean']
        rmse = round(np.sqrt(mean_squared_error(y_true, y_predicted)), 4)
        nse = round(he.evaluator(he.nse, y_predicted, y_true)[0], 4)
        if print_res:
            print(f'{text} {self.id} V: {round(self.densified_ts["velocity"].mean(), 3)}m/s, RMSE: {rmse}m, NSE: {nse}')
        return rmse, nse

    def plot_daily_wl(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.single_VS_itpd, marker='o', label=f'{self.id} interpolated WL')
        ax.plot(self.closest_in_situ_daily_wl, marker='o', label='daily gauge WL')
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', label='VS data', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
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
        norm = mcolors.BoundaryNorm(bins, ncolors=cmap.N, clip=True)

        scatter = ax.scatter(timeseries['shifted_time'], timeseries['shifted_wl'],
                             c=timeseries['chainage_diff'], cmap=cmap, marker='o',
                             label='WL colored by distance', zorder=2, norm=norm, edgecolors='black', linewidths=0.5)
        try:
            ax.errorbar(timeseries['shifted_time'], timeseries['shifted_wl'], yerr=timeseries['bias'],
                        ecolor='red', label='VS measurement bias', zorder=1)
        except KeyError:
            pass
        ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', s=50, color='purple',
                   label='water level from reference station', zorder=11, edgecolors='black', linewidths=0.5)
        if len(self.closest_in_situ_daily_wl) > 0:
            ax.plot(self.closest_in_situ_daily_wl, color='black', label='shifted gauge WL')

        ax.set_xlabel('Time')
        ax.set_ylabel('Water level [m]')
        cbar = plt.colorbar(scatter, ticks=bins)
        cbar.set_label('Along-river distance from ref. station')
        cbar.ax.set_yticklabels([f'{b:.0f}' for b in bins])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        if len(self.closest_in_situ_daily_wl) > 0:
            ax1.scatter(timeseries['chainage_diff'], timeseries['shifted_wl_bias'])
            ax2.scatter(timeseries['bias'], timeseries['shifted_wl_bias'])

        ax1.set_xlabel('Along-river distance from ref. station')
        ax1.set_ylabel('Bias to shifted gauge WL')

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
        if len(self.closest_in_situ_daily_wl) > 0:
            ax.plot(self.closest_in_situ_daily_wl, label=f'In situ {round((g_chain - self.chainage) / 1000)}'
                                                         f' km from DS', color='red', linewidth=4)
        mean_ds = self.wl.wse.mean()
        curr_wl = self.wl.loc[(self.wl.index > self.wl.index.min()) & (self.wl.index < self.wl.index.max())]
        ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color='magenta', linewidth=3, label=f'DS {self.id}')
        for i, vs in enumerate(sorted(self.upstream_adjacent_vs, key=lambda x: x.chainage)):
            curr_wl = vs.wl.loc[(vs.wl.index > self.wl.index.min()) & (vs.wl.index < self.wl.index.max())]
            ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color=colors[i],
                    label=f'{vs.id}, {round((vs.chainage - self.chainage) / 1000)} km from DS')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show(block=True)

    def plot_all_vs_timeseries_with_correlations(self, just_swot=False):
        fig, ax = plt.subplots(figsize=(11, 6))

        all_chainages = [self.chainage] + [vs.chainage for vs in self.upstream_adjacent_vs]
        min_chainage = min(all_chainages)
        max_chainage = max(all_chainages)
        cmap = cm.get_cmap('viridis')
        plot_data = []

        def get_color_from_chainage(chainage, min_c, max_c, colormap):
            if min_c == max_c:
                norm_chainage = 0.5
            else:
                norm_chainage = (chainage - min_c) / (max_c - min_c)
            return colormap(norm_chainage)

        vs_to_corr1 = self.get_daily_linear_interpolated_wl_of_single_vs(just_swot)
        plot_data.append({
            'series': vs_to_corr1,
            'label': f'RS {self.id} ({self.chainage / 1000:.1f}km)',
            'color': 'red',
            'chainage': self.chainage,
            'linewidth': 2
        })

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

        plot_data_sorted = sorted(plot_data, key=lambda x: x['chainage'], reverse=True)

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
        aspect_ratio_data = y_range / x_range if x_range > 0 else 1

        desired_width = np.clip(base_width, min_width, max_width)
        desired_height = np.clip(desired_width * aspect_ratio_data, min_height, max_height)
        margin_factor = 1.05
        fig_width = desired_width * margin_factor
        fig_height = desired_height * margin_factor

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        curr_riv_sect.plot(ax=ax, linewidth=2, color='grey', zorder=1)
        if plot_swot_tiles:
            swot_tiles_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/' \
                              'dane_SWOT/SWOT_tiles/swot_science_hr_Aug2021-v05_shapefile_swath/' \
                              'swot_science_hr_2.0s_4.0s_Aug2021-v5_swath.shp'
            swot_gdf = gpd.read_file(swot_tiles_file)
            swot_gdf.plot(
                ax=ax,
                color='gray',
                alpha=0.4,
                edgecolor='black',
                linewidth=0.5,
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
        s_m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        s_m.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(s_m, cax=cax, label='RMSE [m]')

        solid_line = mlines.Line2D([], [], color='darkblue', linewidth=3, linestyle='-', label='direct regressions')
        dashed_line = mlines.Line2D([], [], color='darkblue', linewidth=3, linestyle='dashed',
                                    label='bypass regressions')
        dotted_line = mlines.Line2D([], [], color='yellow', linewidth=3, linestyle='dotted',
                                    label='unused regressions')
        river_line = mlines.Line2D([], [], color='grey', linewidth=2, label='river course')
        rs_point = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='reference station')
        vs_point = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None', label='virtual station')
        ax.legend(handles=[rs_point, vs_point, solid_line, dashed_line, dotted_line, river_line])

        aspect_ratio_fig = fig_height / fig_width
        aspect_ratio_data = y_range / x_range
        if aspect_ratio_data < aspect_ratio_fig:
            new_y_range = x_range * aspect_ratio_fig
            y_center = (max(all_y) + min(all_y)) / 2
            ax.set_ylim((y_center - new_y_range / 2) - 0.075, (y_center + new_y_range / 2) + 0.075)
            ax.set_xlim(min(all_x) - 0.075, max(all_x) + 0.075)
        else:
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

    def plot_all_gauge_timeseries_within_ds(self, loaded_gauges):
        min_chain = min([x.chainage for x in self.upstream_adjacent_vs])
        max_chain = max([x.chainage for x in self.upstream_adjacent_vs])
        gauges = [x for x in loaded_gauges.values() if min_chain < x.chainage < max_chain]
        first_gauge = \
            [x for x in gauges if
             abs(x.chainage - self.chainage) == min([abs(x.chainage - self.chainage) for x in gauges])][0]
        first_gauge_mean_wl = first_gauge.wl_df['stage'].loc[self.closest_in_situ_daily_wl.index].mean()

        bins = sorted(list(set([g.chainage for g in gauges])))
        bins.append(bins[-1] + 1e-6)
        bins = np.array(bins)

        cmap = cm.get_cmap('viridis')
        norm = mcolors.BoundaryNorm(bins, ncolors=cmap.N, clip=True)

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
