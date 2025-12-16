import pandas as pd
import pickle
import geopandas as gpd
import os
from insituapi.InSitu import InSitu
from model import station_utils as sc
from model.Station_class import GaugeStation


def download_in_situ_data_dahiti(riv_obj, cntry, t1, res_dir):
    """
        Downloads in-situ water level data for a specified river from the DAHITI platform
        and processes it into GaugeStation objects.

        The function first checks if a processed data file (.pkl) already exists for the river.
        If it exists, it loads and returns the saved data. Otherwise, it connects to InSitu
        API (DAHITI), retrieves a list of stations for the given country, filters
        them by river proximity and observation period/frequency, downloads the time series,
        and creates a dictionary of GaugeStation objects, which is then saved to a pickle file.

        :param riv_obj: River object containing geometrical data (e.g., a GeoDataFrame in .gdf attribute).
        :param cntry: Country code or name used to filter in-situ stations.
        :param t1: Start date/time for filtering gauge data.
        :param res_dir: Directory path where the resulting pickle file will be saved/loaded from.
        :returns: A dictionary where keys are gauge IDs and values are GaugeStation objects with uploaded water levels.
        """
    
    filepath = f'{res_dir}gauge_at_{riv_obj.name}.pkl'
    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the in situ downloading.")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    insitu = InSitu()
    stations_data = sc.get_list_of_stations_from_country(cntry, insitu)
    insitu_df = pd.DataFrame(stations_data)
    insitu_gdf = gpd.GeoDataFrame(insitu_df, geometry=gpd.points_from_xy(insitu_df.longitude, insitu_df.latitude),
                                  crs="EPSG:4326")
    insitu_gdf = insitu_gdf.to_crs(riv_obj.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'longitude': 'X', 'latitude': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj)
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1)
    riv_name_for_insitu = riv_obj.name if riv_obj.name != 'Rhine' else 'Rhein'
    selected_gauges = sc.filter_gauges_by_target_name(selected_gauges, insitu, riv_name_for_insitu)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        gauges_on_river_dict[row['id']] = GaugeStation(row['X'], row['Y'], row['id'], riv_obj.name, row['chainage'],
                                                       row['unit'], row['data_sampling'])
        row_data = insitu.download(int(row['id']))
        row_df = pd.DataFrame(row_data['data']).rename(columns={'value': 'stage'})
        gauges_on_river_dict[row['id']].upload_wl(row_df)
        if row['id'] == 19140:
            gauges_on_river_dict[row['id']].wl_df.loc[gauges_on_river_dict[row['id']].wl_df['stage'] > 22.5,
                                                      'stage'] = np.nan
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_usgs_insitu_data(riv_obj, t1, res_dir):
    """
        Downloads and processes in-situ water level data for a specified river from USGS,
        loading data from local CSV files.

        It first checks for an existing processed pickle file. If not found, it loads station
        metadata and data from local CSV files, filters stations based on river proximity
        and data availability/periodality (using t1), handles timezone conversions, calculates
        data sampling frequency, and returns a dictionary of GaugeStation objects.

        :param riv_obj: River object containing geometrical data.
        :param t1: Start date/time for filtering gauge data (minimum required date).
        :param res_dir: Directory path for the resulting pickle file.
        :returns: A dictionary where keys are gauge site numbers and values are GaugeStation objects.
        """
    filepath = f'{res_dir}gauge_at_{riv_obj.name}.pkl'
    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the in situ downloading.")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    usgs_dir = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_USGS/'
    metadata_path = f'{usgs_dir}{riv_obj.name}_gauge_metadata.csv'
    data_path = f'{usgs_dir}{riv_obj.name}_gauge_data.csv'
    # t1_utc, t2_utc = t1.tz_localize('UTC'), t2.tz_localize('UTC')
    df_data = pd.read_csv(data_path, sep=';')
    insitu_df = pd.read_csv(metadata_path)
    insitu_gdf = gpd.GeoDataFrame(insitu_df, geometry=gpd.points_from_xy(insitu_df.dec_long_va, insitu_df.dec_lat_va),
                                  crs=4326)
    insitu_gdf = insitu_gdf.to_crs(riv_obj.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'dec_long_va': 'X', 'dec_lat_va': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj)
    min_dates, max_dates = [], []
    for index, row in gauges_on_river_metadata.iterrows():
        curr_data = df_data.loc[df_data['id'] == row['site_no']]
        if len(curr_data) > 0:
            curr_data['date'] = pd.to_datetime(curr_data['date'])
            # if curr_data['date'].dt.tz != timezone.utc:
            curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
            curr_data['date'] = curr_data['date'].dt.tz_localize(None)

            min_dates.append(curr_data['date'].min())
            max_dates.append(curr_data['date'].max())
        else:
            min_dates.append(None)
            max_dates.append(None)
    gauges_on_river_metadata['min_date'] = min_dates
    gauges_on_river_metadata['max_date'] = max_dates

    gauges_on_river_metadata = gauges_on_river_metadata.dropna(subset=['min_date', 'max_date'])
    gauges_on_river_metadata['type'] = 'water_level'
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)
    selected_gauges['unit'] = 'm'
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_data = df_data.loc[df_data['id'] == row['site_no']]
        curr_data['date'] = pd.to_datetime(curr_data['date'])
        curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
        curr_data['date'] = curr_data['date'].dt.tz_localize(None)

        data_sampling = pd.to_datetime(curr_data['date']).diff().mode()[0].resolution_string
        gauges_on_river_dict[row['site_no']] = GaugeStation(row['X'], row['Y'], row['site_no'], riv_obj.name,
                                                            row['chainage'], row['unit'], data_sampling)
        gauges_on_river_dict[row['site_no']].upload_wl(curr_data)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_ana_insitu_data(riv_obj, t1, res_dir):
    """
        Downloads and processes in-situ water level data for a specified river from the ANA agency
        (Brazil), loading data from local CSV files.

        The function loads station metadata and data from local files, filters data for a specific
        gauge ID (hardcoded), processes the time series by applying an index (for slicing) and
        a conversion (stage / 100), and saves the processed data to a new CSV before creating
        and returning GaugeStation objects.

        :param riv_obj: River object containing geometrical data.
        :param t1: Start date/time for data slicing and filtering (minimum required date).
        :param res_dir: Directory path for the resulting pickle file.
        :returns: A dictionary where keys are gauge station codes and values are GaugeStation objects.
        """
    filepath = f'{res_dir}gauge_at_{riv_obj.name}.pkl'
    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the in situ downloading.")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    gauge_id = 13150003
    ana_dir = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_ANA/'
    metadata_path = f'{ana_dir}RIO SOLIMÕES-AMAZONAS_gauge_metadata.csv'
    data_path = f'{ana_dir}RIO SOLIMÕES-AMAZONAS_gauge_data.csv'
    df_data = pd.read_csv(data_path, sep=';')
    insitu_df = pd.read_csv(metadata_path, sep=';')
    curr_metadata = insitu_df.loc[insitu_df['StationCode'] == gauge_id]
    insitu_gdf = gpd.GeoDataFrame(curr_metadata, geometry=gpd.points_from_xy(curr_metadata['Longitude'],
                                                                             curr_metadata['Latitude']), crs=4326)
    insitu_gdf = insitu_gdf.to_crs(riv_obj.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'Longitude': 'X', 'Latitude': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj, 5000)
    min_dates, max_dates = [], []
    for index, row in gauges_on_river_metadata.iterrows():
        curr_df = df_data.loc[df_data['id'].astype(int) == row['StationCode']]
        if len(curr_df) > 0:
            curr_df.index = pd.to_datetime(curr_df['date'])
            curr_df = curr_df[t1:]
            curr_df['stage'] = curr_df['stage'] / 100
            curr_df[['stage']].to_csv(f'{ana_dir}gauge{gauge_id}.csv', sep=';', decimal=',')

            min_dates.append(curr_df['date'].min())
            max_dates.append(curr_df['date'].max())
        else:
            min_dates.append(None)
            max_dates.append(None)
    gauges_on_river_metadata['min_date'] = min_dates
    gauges_on_river_metadata['max_date'] = max_dates

    gauges_on_river_metadata['type'] = 'water_level'
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)
    selected_gauges['unit'] = 'm'
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_df = df_data.loc[df_data['id'] == row['StationCode']]
        curr_df.index = pd.to_datetime(curr_df['date'])
        curr_df = curr_df[t1:]
        curr_df['stage'] = curr_df['stage'] / 100
        curr_df[['stage']].to_csv(f'{ana_dir}gauge{gauge_id}.csv', sep=';', decimal=',')

        data_sampling = pd.to_datetime(curr_df['date']).diff().mode()[0].resolution_string
        gauges_on_river_dict[row['StationCode']] = GaugeStation(row['X'], row['Y'], row['StationCode'], riv_obj.name,
                                                                row['chainage'], row['unit'], data_sampling)
        gauges_on_river_dict[row['StationCode']].upload_wl(curr_df)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_ganges_insitu_data(riv_obj, t1, res_dir):
    """
        Downloads and processes in-situ water level data for the Ganges river,
        loading data from local CSV files.

        The function loads metadata and data from local files, filters stations by river
        proximity (with a 2000m buffer), handles timezone conversion from 'Asia/Kolkata' to UTC,
        calculates data sampling, and creates a dictionary of GaugeStation objects.

        :param riv_obj: River object containing geometrical data.
        :param t1: Start date/time for filtering gauge data (minimum required date).
        :param res_dir: Directory path for the resulting pickle file.
        :returns: A dictionary where keys are gauge indices and values are GaugeStation objects.
        """
    filepath = f'{res_dir}gauge_at_{riv_obj.name}.pkl'
    if os.path.exists(filepath):
        print(f"File already exists '{filepath}'. Skipping the in situ downloading.")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    ganges_dir = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_GANGES/'
    metadata_path = f'{ganges_dir}{riv_obj.name}_gauge_metadata.csv'
    data_path = f'{ganges_dir}{riv_obj.name}_gauge_data.csv'
    df_data = pd.read_csv(data_path, sep=';', decimal=',')
    insitu_df = pd.read_csv(metadata_path, sep=';', decimal=',')
    insitu_gdf = gpd.GeoDataFrame(insitu_df, geometry=gpd.points_from_xy(insitu_df.y, insitu_df.x),
                                  crs=4326)
    insitu_gdf = insitu_gdf.to_crs(riv_obj.gdf.crs)
    insitu_gdf = insitu_gdf.rename(columns={'y': 'X', 'x': 'Y'})
    gauges_on_river_metadata = sc.select_gauges_from_river(insitu_gdf, riv_obj, 2000)
    min_dates, max_dates = [], []
    for index, row in gauges_on_river_metadata.iterrows():
        curr_data = df_data.loc[df_data['name'] == row['name']]
        if len(curr_data) > 0:
            curr_data['date'] = pd.to_datetime(curr_data['dt'])
            curr_data['date'] = curr_data['date'].dt.tz_localize('Asia/Kolkata')
            curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
            curr_data['date'] = curr_data['date'].dt.tz_localize(None)

            min_dates.append(curr_data['date'].min())
            max_dates.append(curr_data['date'].max())
        else:
            min_dates.append(None)
            max_dates.append(None)
    gauges_on_river_metadata['min_date'] = min_dates
    gauges_on_river_metadata['max_date'] = max_dates

    gauges_on_river_metadata = gauges_on_river_metadata.dropna(subset=['min_date', 'max_date'])
    gauges_on_river_metadata['type'] = 'water_level'
    selected_gauges = sc.filter_gauges_by_dt_freq_target(gauges_on_river_metadata, t1, False)
    selected_gauges = sc.get_chainages_for_all_gauges(selected_gauges, riv_obj)
    selected_gauges['unit'] = 'm'
    gauges_on_river_dict = {}
    for index, row in selected_gauges.iterrows():
        curr_data = df_data.loc[df_data['name'] == row['name']]
        curr_data['date'] = pd.to_datetime(curr_data['dt'])
        curr_data['date'] = curr_data['date'].dt.tz_localize('Asia/Kolkata')
        curr_data['date'] = curr_data['date'].dt.tz_convert('UTC')
        curr_data['date'] = curr_data['date'].dt.tz_localize(None)
        curr_data = curr_data[['date', 'WSE']].rename(columns={'WSE': 'stage'})
        data_sampling = pd.to_datetime(curr_data['date']).diff().mode()[0].resolution_string
        gauges_on_river_dict[index] = GaugeStation(row['X'], row['Y'], index, riv_obj.name, row['chainage'],
                                                   row['unit'], data_sampling)
        gauges_on_river_dict[index].upload_wl(curr_data)
    for gauge_station in gauges_on_river_dict.values():
        print(gauge_station)
    with open(filepath, "wb") as g_file:
        pickle.dump(gauges_on_river_dict, g_file)
    return gauges_on_river_dict


def download_in_situ_data(riv_obj, country, t_1, dir_g_data):
    """
        A routing function that selects and calls the appropriate country/source-specific
        in-situ data download function based on the river name.

        It extracts the river's main name and uses conditional logic (if/elif) to direct
        the call to one of the specialized download functions (DAHITI, USGS, ANA, Ganges).

        :param riv_obj: River object containing geometrical data.
        :param country: Country code or name (used only by the DAHITI function).
        :param t_1: Start date/time for filtering gauge data.
        :param dir_g_data: Directory path for saving/loading pickle files.
        :returns: A dictionary of GaugeStation objects for the selected river.
        """
    gauges = None
    if riv_obj.name in ['Po', 'Oder', 'Rhine', 'Elbe']:
        gauges = download_in_situ_data_dahiti(riv_obj, country, t_1, dir_g_data)
    elif riv_obj.name in ['Mississippi', 'Missouri']:
        gauges = download_usgs_insitu_data(riv_obj, t_1, dir_g_data)
    elif riv_obj.name == 'Ganges':
        gauges = download_ganges_insitu_data(riv_obj, t_1, dir_g_data)
    elif riv_obj.name.replace('õ', 'o') == 'Solimoes':
        gauges = download_ana_insitu_data(riv_obj, t_1, dir_g_data)
    return gauges
