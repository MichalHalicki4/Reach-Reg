import pandas as pd
import geopandas as gpd
import numpy as np
from insituapi.InSitu import InSitu
from dahitiapi.DAHITI import DAHITI
import River_class as rv
import Station_class as sc
import pickle
import copy
from shapely.geometry import Point, Polygon
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.colors as mcolors
from scipy import stats
from matplotlib.colors import BoundaryNorm
from scipy.stats import gaussian_kde  # Potrzebne do obliczenia gęstości
import time


data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/'
fig_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/altirunde25/'
results_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/article/results/'
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp'  # POLAND
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb23_v17b.shp'  # ELBE, RHINE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb21_v17b.shp'  # ELBE, RHINE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb22_v17b.shp'  # DANUBE
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/NA/na_sword_reaches_hb74_v17b.shp'  # USA
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/SA/sa_sword_reaches_hb62_v17b.shp'  # Amazon
riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/AS/as_sword_reaches_hb45_v17b.shp'  # GANGES

t1, t2 = pd.to_datetime('2023-07-11 00:00'), pd.to_datetime('2024-12-31 23:59')

bbox_props = dict(
    boxstyle="round,pad=0.25",  # Styl ramki: "round" (zaokrąglone), "square", itd. oraz wewnętrzny padding (odstęp)
    fc="white",  # Face color: kolor tła ramki
    alpha=0.6,  # Przezroczystość tła (0.0 to przezroczyste, 1.0 to pełne krycie)
    ec="black",  # Edge color: kolor ramki
    lw=.5  # Line width: grubość ramki
)

riv, metric_crs = 'Oder', '2180'
# riv, metric_crs = 'Rhine', '4839'
# riv, metric_crs = 'Elbe', '4839'
# riv, metric_crs = 'Po', '3035'
# riv, metric_crs = 'Missouri', 'ESRI:102010'
# riv, metric_crs = 'Mississippi', 'ESRI:102010'
# riv, metric_crs = 'Solimoes', 'ESRI:102033'
# riv, metric_crs = 'Ganges', 'ESRI:102025'

river_name, riv_names, basin_name, up_reach, dn_reach, country = rv.dahiti_river_names_and_basins[riv].values()
with open(f'{results_dir}{river_name.split(",")[0]}_object.pkl', "rb") as f:
    current_river = pickle.load(f)
with open(f'{results_dir}up_and_dn_gauges/vs_updt_riv_at_{river_name.split(",")[0]}.pkl', "rb") as f:
    loaded_stations = pickle.load(f)
    loaded_stations = sorted(loaded_stations, key=lambda x: x.chainage)
with open(f'{data_dir}gauge_at_{river_name.split(",")[0]}.pkl', "rb") as f:
    loaded_gauges = pickle.load(f)


# vs_id = 42255
# vs_id = 42305  # 42224
# vs_id = 42257  # 13655
# vs_id = 41905
# vs_id = 41861
# vs_id = 41900
# vs_id = 46217
# vs_id = 41931
vs_id = 42224
if riv == 'Oder':
    tributary_chains = current_river.tributary_chains
else:
    tributary_chains = []
neigh_dam_vs = rv.vs_with_neight_dams[riv]
gauge_dist_thres = 5
amp_thres, rmse_thres, single_rmse_thres, itpd_method = list(rv.configs[riv].values())[3:]
buffer, corr_thres, bottom = 300, 0.75, 0.1

VS = copy.deepcopy([x for x in loaded_stations if x.id == vs_id][0])
vs_gauge_dist = abs(VS.chainage - VS.neigh_g_up_chain) / 1000
DS = sc.DensificationStation(VS, buffer, None, itpd_method)
DS.get_upstream_adjacent_vs(loaded_stations)
df_true = DS.swot_wl[['wse']].set_index(pd.to_datetime(DS.swot_wl['datetime'])).resample(
    'D').mean().dropna()
DS.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                rv.vs_with_neight_dams[riv], False)
DS.filter_stations_only_with_swot()
DS.get_slope_of_all_vs()
gauge_chain = VS.neigh_g_up_chain if VS.closest_gauge == 'up' else VS.neigh_g_dn_chain
gauge_id = VS.neigh_g_up if VS.closest_gauge == 'up' else VS.neigh_g_dn

DS.get_single_vs_interpolated_ts()
DS.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
DS.calibrate_mannings_c()
DS.densified_ts = DS.calculate_shifted_time_by_simplified_mannig(DS.densified_ts, bottom)
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

adjusted_gauge_data = DS.adjust_gauge_data_to_vs_by_regr(loaded_gauges[gauge_id].wl_df, gauge_chain)
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


DS.plot_vs_setting_with_regressions_rmse(current_river)

DS2 = sc.DensificationStation(VS, buffer, None, itpd_method)
DS2.get_upstream_adjacent_vs(loaded_stations)
DS2.filter_stations_by_corr_amp_dams_tribs_other(corr_thres, amp_thres, current_river.dams, tributary_chains,
                                                rv.vs_with_neight_dams[riv], False)
DS2.filter_stations_only_with_swot()
DS2.get_slope_of_all_vs()
DS2.get_single_vs_interpolated_ts()
DS2.get_densified_wl_by_regressions(rmse_thres=10, single_rmse_thres=single_rmse_thres)
print(1)


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


lblpad = 1


def plot_all_vs_timeseries_with_correlations(self, ax):
    # Collect all chainages to normalize them for colormap
    all_chainages = [self.chainage] + [vs.chainage for vs in self.upstream_adjacent_vs]
    min_chainage = min(all_chainages)
    max_chainage = max(all_chainages)

    # Get the viridis colormap
    cmap = plt.get_cmap('viridis')
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
    vs_to_corr1 = self.get_daily_linear_interpolated_wl_of_single_vs(False)
    color_curr_vs = get_color_from_chainage(self.chainage, min_chainage, max_chainage, cmap)
    plot_data.append({
        'series': vs_to_corr1,
        'label': f'RS',
        'color': 'red',
        'chainage': self.chainage,
        'linewidth': 2  # Make the current VS line thicker
    })

    # Process upstream adjacent VSs
    for vs in self.upstream_adjacent_vs:
        vs_to_corr2 = vs.get_daily_linear_interpolated_wl_of_single_vs(False)
        if len(vs_to_corr2) > 0 and vs.id != self.id:
            correlation = vs_to_corr1.corr(vs_to_corr2)
            color_vs = get_color_from_chainage(vs.chainage, min_chainage, max_chainage, cmap)
            plot_data.append({
                'series': vs_to_corr2,
                'label': f'R={correlation:.2f}',
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
    ax.legend(handlelength=1.0, loc='lower left', labelspacing=0.1, borderpad=0.1, fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    # Ustawienie formatu daty (np. 'YYYY-MM')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Opcjonalnie: ustawienie etykiet co miesiąc (minor locator)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(vs_to_corr2.index.min() - np.timedelta64(20, 'W'), None)
    ax.set_xlabel('Time', labelpad=lblpad)
    ax.set_ylabel('Water Level [m]', labelpad=lblpad)
    ax.grid(True, linestyle='--', alpha=0.7)
    # ax.legend(loc='center', bbox_to_anchor=(-0.35, 0.5))
    # plt.savefig(f'{fig_dir}RS_wl_corrs_at_{river_name.split(",")[0]}.png', dpi=300)


def plot_all_vs_timeseries(self, ax):
    viridis_cmap = plt.get_cmap('viridis')
    colors = [viridis_cmap(i) for i in np.linspace(0, 1, len(self.upstream_adjacent_vs))]
    # ax.plot(self.closest_in_situ_daily_wl, label=f'In situ {round((self.neigh_g_up_chain - self.chainage) / 1000)}'
    #                                              f' km from DS', color='red', linewidth=4)
    mean_ds = self.wl.wse.mean()
    curr_wl = self.wl.loc[(self.wl.index > self.wl.index.min()) & (self.wl.index < self.wl.index.max())]
    ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color='red', linewidth=3, label=f'DS {self.id}')
    for i, vs in enumerate(sorted(self.upstream_adjacent_vs, key=lambda x: x.chainage)):
        # print(vs.wl.index)
        curr_wl = vs.wl.loc[(vs.wl.index > self.wl.index.min()) & (vs.wl.index < self.wl.index.max())]
        ax.plot(curr_wl.wse - (curr_wl.wse.mean() - mean_ds), color=colors[i],
                label=f'{vs.id}, {round((vs.chainage - self.chainage) / 1000)} km from DS')
    # ax.legend(loc='center', bbox_to_anchor=(-0.35, 0.5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    # Ustawienie formatu daty (np. 'YYYY-MM')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Opcjonalnie: ustawienie etykiet co miesiąc (minor locator)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlabel('Time', labelpad=lblpad)
    ax.set_ylabel('Water level [m]', labelpad=lblpad)
    # plt.show(block=True)


def plot_map_with_vs_setting_and_vs_water_levels_v2(DS2, ax, ax2):
    swot_tiles_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/dane_SWOT/SWOT_tiles/swot_science_hr_Aug2021-v05_shapefile_swath/swot_science_hr_2.0s_4.0s_Aug2021-v5_swath.shp'
    swot_gdf = gpd.read_file(swot_tiles_file)
    x_list, y_list = map(list, zip(*[(a.x, a.y) for a in DS.upstream_adjacent_vs]))
    plot_buffer_up, plot_buffer_down, plot_buffer_left, plot_buffer_right = 0.1, 0.1, 0.1, 0.1
    x_max, x_min, y_max, y_min = max(x_list) + plot_buffer_right, min(x_list) - plot_buffer_left, max(
        y_list) + plot_buffer_up, min(
        y_list) - plot_buffer_down
    aoi = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi], crs=swot_gdf.crs)
    swot_gdf = swot_gdf.clip(aoi_gdf).to_crs(4326)
    cmap = plt.get_cmap('viridis')
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
    major_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(major_locator)
    ax.set_xlabel('Longitude', labelpad=lblpad)
    ax.set_ylabel('Latitude', labelpad=lblpad)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    y_center = (y_max + y_min) / 2
    aspect_ratio = 1 / np.cos(np.radians(y_center))
    ax.set_aspect(aspect_ratio, adjustable='box')  # <--- UŻYJ TEGO ZAMIAST 'equal'
    # Zmieniono, aby używać `ax` zamiast `plt` i usunięto drugą adnotację flow direction
    ax.text(0.082, 0.98, '(a)', transform=ax.transAxes, ha='right', va='top', fontsize=11, bbox=bbox_props)
    ax2.text(0.075, 0.98, '(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=11, bbox=bbox_props)

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
        label='SWOT swath'
    )
    handles.append(swot_patch)
    labels.append('SWOT swath')
    ax.legend(
        handles=handles,
        labels=labels,
        loc='lower left'
    )

    first_vs = DS.upstream_adjacent_vs[0]
    ax2.errorbar(x=DS.chainage / 1000, y=DS.wl['wse'].mean(), yerr=(DS.wl['wse'].max() - DS.wl['wse'].min()) / 2,
                 fmt='o', capsize=3, label='RS water level variation', color='red')
    ax2.errorbar(x=first_vs.chainage / 1000, y=first_vs.wl['wse'].mean(),
                 yerr=(first_vs.wl['wse'].max() - first_vs.wl['wse'].min()) / 2, fmt='o', capsize=3,
                 label='VS water level variation', color=colors[0])
    for i, curr_vs in enumerate(DS.upstream_adjacent_vs[1:]):
        ax2.errorbar(x=curr_vs.chainage / 1000, y=curr_vs.wl['wse'].mean(),
                     yerr=(curr_vs.wl['wse'].max() - curr_vs.wl['wse'].min()) / 2, fmt='o', capsize=3,
                     color=colors[i + 1])
    ax2.set_xlabel('Chainage [km]', labelpad=lblpad)
    ax2.set_ylabel('Water level [m]', labelpad=lblpad)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')

    # --- ZMIANA: Ręczne dostosowanie marginesów ---
    # fig.subplots_adjust(left=0.04, right=0.98, bottom=0.12, top=0.95, wspace=0.05)
    # plt.tight_layout()
    # plt.savefig(f'{fig_dir}vs_setting_at_{river_name.split(",")[0]}_with_SWOT2.png', dpi=300)
    # plt.show(block=True)


fig_setting_and_basics = False
if fig_setting_and_basics:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    plot_map_with_vs_setting_and_vs_water_levels_v2(DS2, ax1, ax2)
    plot_all_vs_timeseries_with_correlations(DS2, ax3)
    plot_all_vs_timeseries(DS2, ax4)
    t1 = ax3.text(0.015, 0.98, '(c)', transform=ax3.transAxes, ha='left', va='top', fontsize=11, bbox=bbox_props)
    t2 = ax4.text(0.015, 0.98, '(d)', transform=ax4.transAxes, ha='left', va='top', fontsize=11, bbox=bbox_props)
    # t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='none'))
    # t2.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='none'))
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    # plt.show(block=True)
    plt.savefig(f'{results_dir}fig_RS_basic_characterictics.png', dpi=300, bbox_inches='tight')


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




def get_n_colors_from_cmap(cmap_name, n):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_days_with_wl_meas(self, dt_str, ax, plot_cbar_label=True):
    df = self.svr_ts[[dt_str, 'id_vs']].loc[pd.to_datetime(self.svr_ts[dt_str]).dt.year == 2024]
    daily_counts = df.groupby(df[dt_str].dt.date).size().reset_index(name='count')
    daily_counts['dt'] = pd.to_datetime(daily_counts[dt_str])
    full_date_range = pd.date_range(start=daily_counts['dt'].min(), end=daily_counts['dt'].max(), freq='D')
    full_df = pd.DataFrame({'dt': full_date_range})
    merged_df = pd.merge(full_df, daily_counts, on='dt', how='left').fillna(0)
    merged_df['count'] = merged_df['count'].astype(int)
    merged_df['year'] = merged_df['dt'].dt.year
    merged_df['month'] = merged_df['dt'].dt.month
    merged_df['day_of_month'] = merged_df['dt'].dt.day
    heatmap_data = merged_df.pivot_table(index=['year', 'month'], columns='day_of_month', values='count',
                                         fill_value=-1).astype(int)
    selected_id_df = df[df['id_vs'] == self.id]
    selected_counts = selected_id_df.groupby(selected_id_df[dt_str].dt.date).size().reset_index(name='selected_count')
    selected_counts['dt'] = pd.to_datetime(selected_counts[dt_str])
    merged_with_selected = pd.merge(merged_df, selected_counts, on='dt', how='left').fillna(0)
    highlight_indices = merged_with_selected[merged_with_selected['selected_count'] > 0][
        ['year', 'month', 'day_of_month']]
    max_val = heatmap_data[heatmap_data > 0].max().max()
    n_colors = int(max_val) + 1
    colors = ['gray', 'white'] + [plt.cm.viridis(i) for i in np.linspace(0, 1, n_colors - 1)]
    bounds = [-1] + list(np.arange(0, n_colors)) + [n_colors]
    custom_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    htmp = sns.heatmap(heatmap_data,
                       cmap=custom_cmap,
                       norm=norm,
                       linewidths=0.25,
                       linecolor='black',
                       vmin=-1,
                       ax=ax)
    colorbar = htmp.collections[0].colorbar
    if plot_cbar_label:
        colorbar.set_label("measurements within a day")
    # 3. Rysowanie czerwonych obwódek dla wybranych komórek
    # Konwersja indeksu heatmapy do listy par (rok, miesiąc)
    heatmap_index = list(heatmap_data.index)
    for index, row in highlight_indices.iterrows():
        year = int(row['year'])
        month = int(row['month'])
        day_of_month = int(row['day_of_month'])
        try:
            # Znajdź wiersz (cycle) i kolumnę (day) dla danego dnia
            row_idx = heatmap_index.index((year, month))
            col_idx = day_of_month - 1
            ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=2))
        except ValueError:
            # Pomiń, jeśli dany miesiąc nie istnieje w heatmapie
            continue
    x_labels_full = np.arange(1, 32)
    x_labels = ['' if (i + 1) % 5 != 0 else str(i + 1) for i in range(len(x_labels_full))]
    ax.set_xticks(np.arange(len(x_labels_full)) + 0.5)
    ax.set_xticklabels(x_labels, rotation=0)
    y_labels_full = [f'{month:02d}-{year}' for year, month in heatmap_data.index]
    y_labels = ['' if (i + 1) % 3 != 0 else label for i, label in enumerate(y_labels_full)]
    ax.set_yticks(np.arange(len(y_labels_full)) + 0.5)
    ax.set_yticklabels(y_labels)


do_plot_days_with_wl_meas = False
if do_plot_days_with_wl_meas:
    vs_id, riv = '42224', 'Oder'
    ds_path = f'{results_dir}rs_stations/up_and_dn_gauges/{riv}_RS{vs_id}.pkl'
    with open(ds_path, "rb") as f:
        ds = pickle.load(f)
    DS = copy.deepcopy(ds)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    plot_days_with_wl_meas(DS, 'dt', ax1, False)
    plot_days_with_wl_meas(DS, 'shifted_time', ax2)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax2.set_xlabel("day in month")
    ax2.set_ylabel("month/year")

    t1 = ax1.text(0.012, 0.98, '(a)', transform=ax1.transAxes, ha='left', va='top', fontsize=11, bbox=bbox_props)
    t2 = ax2.text(0.012, 0.98, '(b)', transform=ax2.transAxes, ha='left', va='top', fontsize=11, bbox=bbox_props)
    # t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='none'))
    # t2.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='none'))
    fig.tight_layout()
    # plt.show(block=True)
    plt.savefig(f'{results_dir}up_and_dn_gauges/fig_RS_num_of_meas_within_days.png', dpi=300, bbox_inches='tight')

# plot_swot_obs_against_gauge_data(DS)
# plot_map_with_vs_setting_and_vs_water_levels(DS)

plot_setting_with_rmses = False
if plot_setting_with_rmses:
    plt.rcParams.update({'font.size': 12})


    def plot_vs_setting_with_regressions_rmse(self, current_river, ax, plot_swot_tiles=True):
        used_regressions = self.get_used_regressions()
        curr_riv_sect = self.clip_river_to_vs_section(current_river).set_crs(4326)
        cmap = plt.get_cmap('viridis')
        vmin = used_regressions['rmse'].min()
        vmax = used_regressions['rmse'].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        text_bbox_props = dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.3)
        all_x = []
        all_y = []
        for index, row in used_regressions.iterrows():
            vs1 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st1']][0]
            vs2 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st2']][0]
            all_x.extend([vs1.x, vs2.x])
            all_y.extend([vs1.y, vs2.y])
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)

        curr_riv_sect.to_crs(4326).plot(ax=ax, linewidth=2, color='blue', zorder=2)
        if plot_swot_tiles:
            swot_tiles_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/dane_SWOT/SWOT_tiles/swot_science_hr_Aug2021-v05_shapefile_swath/swot_science_hr_2.0s_4.0s_Aug2021-v5_swath.shp'
            swot_gdf = gpd.read_file(swot_tiles_file)
            swot_gdf.to_crs(4326).plot(
                ax=ax,
                color='gray',  # Ustawia kolor wypełnienia na szary
                alpha=0.4,  # Ustawia przezroczystość na 50% (0.0 to w pełni przezroczysty, 1.0 to w pełni kryjący)
                edgecolor='black',  # Ustawia kolor obwódki na czarny
                linewidth=0.05,  # Ustawia grubość obwódki (możesz dostosować tę wartość, aby uzyskać "cienki" efekt)
                zorder=1
            )
        for index, row in used_regressions.iterrows():
            vs1 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st1']][0]
            vs2 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st2']][0]
            line_color = cmap(norm(row['rmse']))
            if row['steps'] == 1:
                ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=3, zorder=3, marker='o',
                        markeredgecolor='black')
            elif row['steps'] == 2:
                ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=3, linestyle='dashed', zorder=3,
                        marker='o', markeredgecolor='black')
            elif row['steps'] == 3:
                ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=3, linestyle='dotted', zorder=3,
                        marker='o', markeredgecolor='black')

            for vs in [vs1, vs2]:
                if vs.id in [23407, 42224, 42222]:
                    ax.text(vs.x + 0.01, vs.y + 0.0304, str(vs.id), ha='left', fontsize=9, bbox=text_bbox_props)
                elif vs.id in [41917]:
                    ax.text(vs.x - 0.02, vs.y - 0.0544, str(vs.id), ha='left', fontsize=9, bbox=text_bbox_props)
                elif vs.id in [13659]:
                    ax.text(vs.x + 0.02, vs.y - 0.0344, str(vs.id), ha='left', fontsize=9, bbox=text_bbox_props)
                elif vs.id in [23408, 23410, 23411]:
                    ax.text(vs.x - 0.2, vs.y - 0.0504, str(vs.id), ha='left', fontsize=9, bbox=text_bbox_props)
                else:
                    ax.text(vs.x + 0.03, vs.y + 0.0018, str(vs.id), ha='left', fontsize=9, bbox=text_bbox_props)

        ax.scatter(self.x, self.y, label='RS', color='red', zorder=4, s=50, edgecolor='black')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(
            [])  # Ustawiamy pustą tablicę, ponieważ pasek kolorów nie jest bezpośrednio powiązany z wykresem plot
        cbar = fig.colorbar(sm, ax=ax, label='regression RMSE [m]', fraction=0.046, pad=0.04)
        cbar_ax = cbar.ax  # Uzyskaj dostęp do osi colorbara
        cbar_ax.axhline(0.2, color='black', linewidth=3, linestyle='solid')

        solid_line = mlines.Line2D([], [], color='darkblue', linewidth=3, linestyle='-', label='direct regressions')
        dashed_line = mlines.Line2D([], [], color='darkblue', linewidth=3, linestyle='dashed',
                                    label='bypass regressions')
        dotted_line = mlines.Line2D([], [], color='yellow', linewidth=3, linestyle='dotted',
                                    label='unused regressions')
        river_line = mlines.Line2D([], [], color='blue', linewidth=2, label='river course')
        rs_point = mlines.Line2D([], [], color='red', marker='o', markeredgecolor='black', linestyle='None',
                                 label='reference station')
        vs_point = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None', markeredgecolor='black',
                                 label='virtual station')
        swot_patch = mpatches.Patch(
            facecolor='gray',
            edgecolor='black',
            linewidth=0.05,
            alpha=0.4,
            label='SWOT swath'
        )
        # ax.legend(handles=[solid_line, dashed_line, river_line, ax.get_children()[-1]])
        ax.legend(handles=[rs_point, vs_point, solid_line, dotted_line, dashed_line, river_line, swot_patch],
                  framealpha=0.5, loc='upper right')
        # ax.legend(handles=[rs_point, vs_point, solid_line, river_line, swot_patch])

        aspect_ratio_data = y_range / x_range
        new_y_range = x_range * aspect_ratio_data
        y_center = (max(all_y) + min(all_y)) / 2
        ax.set_ylim((y_center - y_range / 2) - 0.075, (y_center + y_range / 2) + 0.075)
        ax.set_xlim(min(all_x) - 0.025, max(all_x) + 0.075)
        aspect_ratio = 1 / np.cos(np.radians(y_center))
        ax.set_aspect(aspect_ratio, adjustable='box')  # <--- UŻYJ TEGO ZAMIAST 'equal'

        # ax.set_title(f'RS{self.id} regressions')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')


    def plot_vs_setting_with_regressions_rmse_sum(self, current_river, ax, plot_swot_tiles=True):
        curr_rmse_sum_thres = 0.598
        used_regressions = self.get_used_regressions()
        curr_riv_sect = self.clip_river_to_vs_section(current_river).set_crs(4326)
        vmin, vmax = self.densified_ts['rmse_sum'].min(), self.densified_ts['rmse_sum'].max()
        cmap = plt.get_cmap('viridis')  # Możesz też użyć 'coolwarm', 'PuOr', 'BrBG' itd.
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
        curr_riv_sect.to_crs(4326).plot(ax=ax, linewidth=2, color='blue', zorder=2)
        if plot_swot_tiles:
            swot_tiles_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/dane_SWOT/SWOT_tiles/swot_science_hr_Aug2021-v05_shapefile_swath/swot_science_hr_2.0s_4.0s_Aug2021-v5_swath.shp'
            swot_gdf = gpd.read_file(swot_tiles_file)
            swot_gdf.to_crs(4326).plot(
                ax=ax,
                color='gray',  # Ustawia kolor wypełnienia na szary
                alpha=0.4,  # Ustawia przezroczystość na 50% (0.0 to w pełni przezroczysty, 1.0 to w pełni kryjący)
                edgecolor='black',  # Ustawia kolor obwódki na czarny
                linewidth=0.05,  # Ustawia grubość obwódki (możesz dostosować tę wartość, aby uzyskać "cienki" efekt)
                zorder=1
            )
        for index, row in used_regressions.iterrows():
            vs1 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st1']][0]
            vs2 = [x for x in [self] + self.upstream_adjacent_vs if x.id == row['st2']][0]
            rmse_sum = \
            self.densified_ts['rmse_sum'].loc[self.densified_ts['id_vs'] == vs1.id].iloc[0]
            line_color = cmap(norm(rmse_sum))
            ax.scatter(vs1.x, vs1.y, color=line_color, zorder=4, s=50, edgecolor='black')
            if row['steps'] == 1:
                if rmse_sum < curr_rmse_sum_thres:
                    ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=4, zorder=3)
                else:
                    ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=2, zorder=3)
            elif row['steps'] == 2:
                if rmse_sum < curr_rmse_sum_thres:
                    ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=4, zorder=3, linestyle='dashed')
                else:
                    ax.plot([vs1.x, vs2.x], [vs1.y, vs2.y], color=line_color, linewidth=2, zorder=3, linestyle='dashed')
        ax.scatter(self.x, self.y, label='RS', color='red', zorder=4, s=50, edgecolor='black')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(
            [])  # Ustawiamy pustą tablicę, ponieważ pasek kolorów nie jest bezpośrednio powiązany z wykresem plot
        cbar = fig.colorbar(sm, ax=ax, label='aggregated RMSE [m]', fraction=0.046, pad=0.04)
        cbar_ax = cbar.ax  # Uzyskaj dostęp do osi colorbara
        cbar_ax.axhline(curr_rmse_sum_thres, color='black', linewidth=3, linestyle='solid')
        # cbar_ax.text(1.05, curr_rmse_sum_thres, f'RMSE AGG THRES: {curr_rmse_sum_thres} m',
        #              va='center', ha='left', color='red', fontsize=12)

        solid_line = mlines.Line2D([], [], color='darkblue', linewidth=4, linestyle='-', label='used regressions')
        dashed_line = mlines.Line2D([], [], color='blue', linewidth=4, linestyle='dashed',
                                    label='bypass regressions')
        dotted_line = mlines.Line2D([], [], color='yellow', linewidth=3, linestyle='dotted',
                                    label='bypassed regressions')
        red_line = mlines.Line2D([], [], color='darkblue', linewidth=2,
                                 label='unused regressions')
        river_line = mlines.Line2D([], [], color='blue', linewidth=2, label='river course')
        rs_point = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='reference station')
        vs_point = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None', label='virtual station')
        swot_patch = mpatches.Patch(
            facecolor='gray',
            edgecolor='black',
            linewidth=0.05,
            alpha=0.4,
            label='SWOT swath'
        )
        # ax.legend(handles=[solid_line, dashed_line, river_line, ax.get_children()[-1]])
        # ax.legend(handles=[rs_point, vs_point, solid_line, dashed_line, dotted_line, river_line])
        ax.legend(handles=[solid_line, red_line], loc='upper right')

        aspect_ratio_data = y_range / x_range
        y_center = (max(all_y) + min(all_y)) / 2
        ax.set_ylim((y_center - y_range / 2) - 0.075, (y_center + y_range / 2) + 0.075)
        ax.set_xlim(min(all_x) - 0.025, max(all_x) + 0.075)

        aspect_ratio = 1 / np.cos(np.radians(y_center))
        ax.set_aspect(aspect_ratio, adjustable='box')  # <--- UŻYJ TEGO ZAMIAST 'equal'
        # ax.set_title(f'RS{self.id} regressions')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # ax.grid(True, linestyle='--', alpha=0.7)
        # fig.tight_layout()
        # plt.show(block=True)


    # fig_width, fig_height = 7.75, 5
    curr_rmse_sum_thres = 0.598
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    plot_vs_setting_with_regressions_rmse(DS2, current_river, ax1)
    plot_vs_setting_with_regressions_rmse_sum(DS2, current_river, ax2)
    # ax1.text(0.97, 0.95, '(a)', transform=ax1.transAxes, ha='right', va='top', fontsize=12)
    # ax2.text(0.97, 0.95, '(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=12)
    ax1.text(0.08, 0.05, '(a)', transform=ax1.transAxes, ha='right', va='top', fontsize=12, bbox=bbox_props)
    ax2.text(0.08, 0.05, '(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=12, bbox=bbox_props)
    fig.tight_layout()
    # plt.show(block=True)
    plt.savefig(f'{results_dir}fig_RS{DS2.id}_at_{current_river.name}_regressions_map.png', dpi=300, bbox_inches='tight')


plot_cval_vel_calibration = False
if plot_cval_vel_calibration:
    res_df = pd.read_csv(
        '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/scripts/velocities_cross_cal.csv',
        sep=';', decimal=',')
    cval_buff = 0.01
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    axes = [ax1, ax2]
    for i, vs_id in enumerate([46217, 42224]):
        ax = axes[i]
        curr_df = res_df.loc[res_df['id'] == vs_id]
        vels, rmse, rmse_cval = curr_df['velocity'], curr_df['rmse_reg'], curr_df['rmse_cval']
        min_index, min_cval = rmse_cval.idxmin(), rmse_cval.min()
        cval_range = rmse_cval[rmse_cval < min_cval + cval_buff]
        vels_at_cval_range = vels[cval_range.index]
        median_velocity = (vels_at_cval_range.max() + vels_at_cval_range.min()) / 2
        ax.plot(vels, rmse, label=f'RMSE (gauge data)', zorder=1, color='darkblue')
        ax.plot(vels, rmse_cval, linestyle='dashed', label='RMSE (cross-val.)', zorder=1, color='orange')
        ax.plot(vels[cval_range.index], cval_range, color='grey', linewidth=7, alpha=.4,
                label='velocity within 1 cm from\nmin. RMSE (cross-val.)', zorder=5)
        ax.scatter(vels[min_index], min_cval, marker='d', color='black', s=100, label='min. RMSE cross-val.', zorder=10)
        ax.scatter(median_velocity, rmse_cval.min() + 0.005, marker='d', s=100, color='red', label='selected velocity',
                   zorder=10)
        ax.grid(linestyle='dashed', alpha=0.6)
        ax.set_ylabel('RMSE [m]')
        ax.set_xlabel('Average velocity [m/s]')
    ax2.legend()
    ax1.text(0.09, 0.065, '(a)', transform=ax1.transAxes, ha='right', va='top', fontsize=12, bbox=bbox_props)
    ax2.text(0.09, 0.065, '(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=12, bbox=bbox_props)
    fig.tight_layout()
    # plt.show(block=True)
    plt.savefig(f'{results_dir}fig_velocity_cval_calibration.png', dpi=300, bbox_inches='tight')


def plot_accuracies_at_rivers(results_df):
    plt.rcParams.update({'font.size': 12})
    # Ustal kolory bazowe
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#b8860b', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']

    metric, metric2 = 'RMSE [m]', 'NSE'
    cols1 = 'rmse_sv'
    cols2 = 'nse_sv'
    cols1_raw = 'rmse_raw'
    cols2_raw = 'nse_raw'

    # 1. Przygotowanie danych do boxplotów (bez zmian)
    river_names = ['Oder', 'Elbe', 'Po', 'Rhine', 'Missouri', 'Mississippi', 'Ganges', 'Solimoes']
    data_rmse = []
    data_nse = []

    # Symulacja zmiennej rv (utrzymana ze względu na kontekst)

    neigh_dam_vs = []
    for x in rv.vs_with_neight_dams.values():
        for a in x:
            neigh_dam_vs.append(a)

    for river in river_names:
        curr_res = results_df.loc[results_df['river'] == river]
        if len(curr_res) == 0:
            data_rmse.extend([[], []])
            data_nse.extend([[], []])
            continue
        curr_res = curr_res.loc[~curr_res['id'].isin(neigh_dam_vs)]
        selected_data1_sv = curr_res[cols1].dropna(axis=0)
        selected_data2_sv = curr_res[cols2].dropna(axis=0)

        selected_data1_raw = curr_res[cols1_raw].dropna(axis=0)
        selected_data2_raw = curr_res[cols2_raw].dropna(axis=0)

        data_rmse.append(selected_data1_sv)
        data_rmse.append(selected_data1_raw)

        data_nse.append(selected_data2_sv)
        data_nse.append(selected_data2_raw)

    # Pozycje boxplotów (bez zmian)
    start_pos = 5
    group_spacing = 3
    num_rivers = len(river_names)
    custom_positions = []
    for i in range(num_rivers):
        pos_sv = start_pos + i * group_spacing
        pos_raw = pos_sv + 1
        custom_positions.extend([pos_sv, pos_raw])

    boxplot_widths = [0.9] * len(custom_positions)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- ZMIANA 1: Ujednolicenie Kolorów Wypełnienia i Definicja Stylów ---

    # Używamy głównego koloru dla wypełnienia obu boxplotów w parze
    colors_sv = COLORS[:len(river_names)]
    colors_to_use = []
    for c in colors_sv:
        # Ten sam kolor dla SV i RAW w danej rzece
        colors_to_use.extend([c, c])

    median_styles = {'color': 'black', 'linewidth': 1}

    # NOWA FUNKCJA: Ustawia kolory wypełnienia i krawędzi (ramki)
    def set_box_styles(bp, colors):
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)

            # Boxy Reach-Reg (SV) - indeksy 0, 2, 4, ...
            if i % 2 == 0:
                patch.set_edgecolor('black')  # Domyślna czarna krawędź
                patch.set_linewidth(1)
            # Boxy Raw - indeksy 1, 3, 5, ...
            else:
                patch.set_edgecolor('red')  # Czerwona ramka dla Raw
                patch.set_linewidth(2)  # Lekko pogrubiona ramka

            # Wiskersy, czapki, loty i mediany
            for element in ['whiskers', 'caps', 'fliers']:
                # Ustawiamy kolor tych elementów na czarny, aby ramka była jedynym wyróżnikiem
                # Musimy jednak iterować po elementach, ponieważ to nie jest w box['boxes']
                pass  # Kolory wiskersów itp. są ustawiane przez 'medianprops' w boxplot

    # Boxplot dla RMSE
    bp_rmse = ax.boxplot(
        data_rmse,
        patch_artist=True,
        medianprops=median_styles,
        widths=boxplot_widths,
        showfliers=True,
        positions=custom_positions
    )
    set_box_styles(bp_rmse, colors_to_use)  # Użycie nowej funkcji

    # Boxplot dla NSE
    bp_nse = ax2.boxplot(
        data_nse,
        patch_artist=True,
        medianprops=median_styles,
        widths=boxplot_widths,
        showfliers=True,
        positions=custom_positions
    )
    set_box_styles(bp_nse, colors_to_use)  # Użycie nowej funkcji

    # 3. Ustawienia wykresu (bez zmian)
    ax.text(0.05, 0.985, '(a)', transform=ax.transAxes, ha='right', va='top', fontsize=12, bbox=bbox_props)
    ax2.text(0.05, 0.985, '(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=12, bbox=bbox_props)

    ax.set_ylabel(metric, fontsize=12)
    ax2.set_ylabel(metric2, fontsize=12)

    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )
    ax2.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )

    separator_positions = [pos + 0.9 for pos in custom_positions[1::2]]
    separator_positions = separator_positions[:-1]

    ylim_rmse = ax.get_ylim()
    ylim_nse = ax2.get_ylim()
    xlim_rmse = ax.get_xlim()
    xlim_nse = ax2.get_xlim()

    # Rysowanie pionowych linii siatki (separatorów)
    ax.vlines(separator_positions, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
              color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.vlines(separator_positions, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1],
               color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

    # --- ZMIANA 2: Nowa Legenda dla Typu Danych (Kolor Ramki) ---
    # Używamy szarego tła wypełnienia dla pokazania ramki
    data_type_handles = [
        mpatches.Patch(facecolor='grey', edgecolor='black', linewidth=1, alpha=0.7, label='Reach-Reg'),
        mpatches.Patch(facecolor='grey', edgecolor='red', linewidth=1.5, alpha=0.7, label='Raw')
    ]

    fig.legend(
        handles=data_type_handles,
        loc='lower right',
        bbox_to_anchor=(0.999, 0.0),
        ncol=2,
        frameon=True,
        fontsize=12
    )

    # Legenda dla rzek (kolory bazowe) - bez zmian
    legend_handles = [mpatches.Patch(color=color, label=river)
                      for color, river in zip(colors_sv, river_names)]

    fig.legend(
        handles=legend_handles,
        loc='lower left',
        bbox_to_anchor=(0.03, 0.0),
        ncol=len(river_names),
        frameon=True,
        fontsize=12
    )
    ax.set_ylim(ylim_rmse)
    ax2.set_ylim(ylim_nse)
    ax.set_xlim(xlim_rmse[0] - .5, xlim_rmse[1])
    ax2.set_xlim(xlim_nse[0] - .5, xlim_nse[1])
    # 5. Dostosowanie układu
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.075)
    # plt.show(block=True)
    plt.savefig(f'{results_dir}up_and_dn_gauges/fig_accuracies_at_rivers_boxplot.png', dpi=300)


def plot_vs_accuracies(results_df):
    plt.rcParams.update({'font.size': 12})
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
    metric = 'RMSE [m]'
    col = 'rmse'
    river_names = ['Oder', 'Elbe', 'Po', 'Rhine', 'Missouri', 'Mississippi', 'Ganges']
    data_rmse = []
    for river in river_names:
        curr_res = results_df.loc[results_df['river'] == river]
        data_rmse.append(curr_res[col])
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_sv = COLORS[:len(river_names)]

    def set_box_colors(bp, colors):
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_edgecolor('black')
            patch.set_alpha(0.7)

    bp_rmse = ax.boxplot(
        data_rmse,  # TERAZ ZAWIERA 2*N ZESTAWÓW DANYCH
        patch_artist=True,
        showfliers=True,
    )
    set_box_colors(bp_rmse, colors_sv)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid()
    plt.show(block=True)


def plot_resulting_timeseries(self, axis):
    axis.plot(self.closest_in_situ_daily_wl, label='Gauge data', color='black', alpha=0.8, linewidth=4, zorder=1)
    axis.plot(self.single_VS_itpd, label='Raw interpolation', color='blue', linewidth=1.5, zorder=4)
    axis.plot(self.svr_itpd, label='Reach-Reg', color='red', linewidth=1.5, zorder=4)
    axis.scatter(self.svr_ts['shifted_time'], self.densified_ts['shifted_wl'], marker='.', s=75, color='red',
                 edgecolor='grey', linewidth=.6, label='Measurements unified with Reach-Reg', zorder=3)
    curr_df = self.densified_ts.loc[pd.to_datetime('2024-01-01'): pd.to_datetime('2025-01-01')]
    axis.set_xlim(curr_df.index.min(), curr_df.index.max())
    axis.set_ylim(curr_df['shifted_wl'].min() - .5, curr_df['shifted_wl'].max() + .5)
    axis.grid(True, linestyle='--', alpha=0.6)


do_plot_rs_timeseries = False
if do_plot_rs_timeseries:
    plt.rcParams.update({'font.size': 12})
    rs_path = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/article/results/rs_stations/'
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7.5*1.3, 9.5*1.3), sharex=True)
    axes, letters = [ax1, ax2, ax3, ax4], ['(a)', '(b)', '(c)', '(d)']
    rs_stations = [{'riv': 'Missouri', 'id': '17455'}, {'riv': 'Mississippi', 'id': '15776'}, {'riv': 'Ganges', 'id': '46395'}, {'riv': 'Solimoes', 'id': ''}]
    # rs_stations = [{'riv': 'Oder', 'id': '42214'}, {'riv': 'Elbe', 'id': '41945'}, {'riv': 'Po', 'id': '46217'}, {'riv': 'Rhine', 'id': '41903'}]

    for index, rs in enumerate(rs_stations):
        if rs['id'] != '':
            filepath = f'{rs_path}{rs["riv"]}_RS{rs["id"]}.pkl'
            with open(filepath, "rb") as f:
                DS = pickle.load(f)
            plot_resulting_timeseries(DS, axes[index])
        axes[index].set_ylabel('Water level [m]')
        axes[index].text(0.025, 0.915, f'{letters[index]} RS{rs["id"]}, {rs["riv"]} River',transform=axes[index].transAxes, bbox=bbox_props)
        axes[index].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[index].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[index].xaxis.set_minor_locator(mdates.MonthLocator())

    ax1.legend(ncols=4, loc='lower center', bbox_to_anchor=(0.5, 0.99), fontsize=12, columnspacing=0.8, handletextpad=0.4)
    ax4.set_xlabel('Time')
    plt.tight_layout()
    # plt.show(block=True)
    # plt.savefig(f'{results_dir}fig_rs_timeseries_a.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{results_dir}fig_rs_timeseries_b.png', dpi=300, bbox_inches='tight')


def plot_densified_vs_gauge_color_by_chainage(self, timeseries):
    fig = plt.figure(figsize=(11, 9))

    # 2 wiersze (ratio 1:2) i 2 kolumny (na górne ploty)
    # Dodatkowa kolumna na cbar zostanie utworzona obok dolnego wykresu
    gs = GridSpec(2, 2,
                  figure=fig,
                  height_ratios=[1, 2],
                  # Zmniejszenie domyślnych odstępów między wykresami
                  hspace=0.2,  # Odstęp w pionie
                  wspace=0.2)  # Odstęp w poziomie (jeśli jest potrzebny, można zmniejszyć)

    # Górne wykresy rozciągają się na pełną szerokość
    ax1 = fig.add_subplot(gs[0, 0])  # Górny lewy
    ax2 = fig.add_subplot(gs[0, 1])  # Górny prawy

    # Dolny wykres rozciąga się na pełną szerokość (obie kolumny gs)
    ax = fig.add_subplot(gs[1, 0:])
    timeseries['chainage_diff'] = (timeseries['vs_chain'] - self.chainage) / 1000
    min_val = timeseries['chainage_diff'].min()
    max_val = timeseries['chainage_diff'].max()
    class_range = 20
    bin_start = np.floor(min_val / class_range) * class_range
    bin_end = np.ceil(max_val / class_range) * class_range
    bins = np.arange(bin_start, bin_end + class_range, class_range)
    # n_classes = len(bins) - 1  # Liczba klas
    cmap = cm.get_cmap('RdYlGn')
    cmap_density = cm.get_cmap('viridis')  # Dobra mapa do gęstości
    norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
    scatter = ax.scatter(timeseries['shifted_time'], timeseries['shifted_wl'],
                         c=timeseries['chainage_diff'], cmap=cmap, marker='o',
                         label='WL coloured by distance', zorder=2, norm=norm, edgecolors='black', linewidths=0.5)
    # ax.errorbar(timeseries['shifted_time'], timeseries['shifted_wl'], yerr=timeseries['bias'],
    #             ecolor='red', label='VS measurement bias', zorder=1)
    ax.scatter(self.wl['datetime'], self.wl['wse'], marker='d', s=50, color='purple',
               label='WL from RS', zorder=11, edgecolors='black', linewidths=0.5)
    # ax.plot(self.densified_wl['shifted_time'], self.densified_wl['shifted_wl_gauge'], color='black',
    #         label='shifted gauge WL')
    ax.plot(self.closest_in_situ_daily_wl, color='black', label='Gauge data')

    ax.set_xlabel('Time')
    ax.set_ylabel('Water level [m]')
    ax_pos = ax.get_position()
    cbar_width = 0.02
    cbar_pad = 0.01  # Mały odstęp od dolnego wykresu (zmniejszony)
    ax_cbar = fig.add_axes([ax_pos.x1 + cbar_pad, ax_pos.y0, cbar_width, ax_pos.height])
    cbar = plt.colorbar(scatter, cax=ax_cbar, ticks=bins)
    cbar.set_label('Along-river distance from RS [km]')
    cbar.ax.set_yticklabels([f'{b:.0f}' for b in bins])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)

    ax.legend()

    df = pd.DataFrame({
        'X1': timeseries['chainage_diff'],
        'Y1': timeseries['shifted_wl_bias'],
        'X2': timeseries['bias'],
        # Y2 to to samo co Y1: timeseries['shifted_wl_bias']
    }).dropna()

    X1, Y1 = df['X1'].values, df['Y1'].values

    # Obliczenie estymacji gęstości jądrowej (KDE)
    xy1 = np.vstack([X1, Y1])
    z1 = gaussian_kde(xy1)(xy1)  # Z1 to wartość gęstości dla każdego punktu

    # Sortowanie danych wg gęstości, aby gęstsze punkty były rysowane na wierzchu
    idx1 = z1.argsort()
    X1, Y1, z1 = X1[idx1], Y1[idx1], z1[idx1]

    scatter1 = ax1.scatter(X1, Y1, c=z1, s=15, cmap=cmap_density)
    # plt.colorbar(scatter1, ax=ax1, label='Gęstość punktów (KDE)', fraction=0.046, pad=0.04)
    X2, Y2 = df['X2'].values, df['Y1'].values  # Y2 = Y1, używamy df['Y1']

    # Obliczenie estymacji gęstości jądrowej (KDE)
    xy2 = np.vstack([X2, Y2])
    z2 = gaussian_kde(xy2)(xy2)

    # Sortowanie danych wg gęstości
    idx2 = z2.argsort()
    X2, Y2, z2 = X2[idx2], Y2[idx2], z2[idx2]

    scatter2 = ax2.scatter(X2, Y2, c=z2, s=15, cmap=cmap_density)
    # plt.colorbar(scatter2, ax=ax2, label='Gęstość punktów (KDE)', fraction=0.046, pad=0.04)

    # ax1.scatter(timeseries['chainage_diff'], timeseries['shifted_wl_bias'])
    ax1.set_xlabel('Along-river distance from RS [km]')
    ax1.set_ylabel('Bias to gauge WL at RS [m]')

    # ax2.scatter(timeseries['bias'], timeseries['shifted_wl_bias'])
    ax2.set_xlabel('Bias of VS measurement [m]')
    ax2.set_ylabel('Bias to gauge WL at RS [m]')

    ax.text(0.01, 0.95, f'(c)', transform=ax.transAxes, bbox=bbox_props)
    ax1.text(0.02, 0.915, f'(a)', transform=ax1.transAxes, bbox=bbox_props)
    ax2.text(0.02, 0.915, f'(b)', transform=ax2.transAxes, bbox=bbox_props)

    fig.subplots_adjust(
        left=0.068,  # Zmniejsz margines lewy
        right=0.905,  # Upewnij się, że jest miejsce na colorbar po prawej
        bottom=0.08,
        top=0.99,
        wspace=0.2,  # Odstęp poziomy między górnymi wykresami (można regulować)
        hspace=0.2  # Odstęp pionowy
    )

    fig.tight_layout(rect=[0, 0, 1, 1])
    # plt.show(block=True)
    plt.savefig(f'{results_dir}up_and_dn_gagues/fig_rs_oder_ts.png', dpi=300, bbox_inches='tight')


do_plot_rs_oder_ts_color_by_chain = False
if do_plot_rs_oder_ts_color_by_chain:
    vs_id, riv = '42224', 'Oder'
    ds_path = f'{results_dir}rs_stations/up_and_dn_gagues/{riv}_RS{vs_id}.pkl'
    with open(ds_path, "rb") as f:
        DS = pickle.load(f)
    plot_densified_vs_gauge_color_by_chainage(DS, DS.svr_ts)



# plot_regressions_btwn_stations(23404, DS.id, 'h', colors[1])
# plot_regressions_btwn_stations(42305, 23404, 'h', colors[2])
# plot_regressions_btwn_stations(19763, 42305, 'h', colors[3])
# plot_regressions_btwn_stations(23406, 19763, 'h', colors[4])