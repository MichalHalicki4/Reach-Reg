import glob
from datetime import datetime, timedelta
import pandas as pd

files = glob.glob('/Users/michalhalicki/Documents/nauka/dane_gis/dane_IMGW/*.csv')
imgw_stations = pd.read_csv('/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro/linux/program_imgw/metadata/gauge/stacje_imgw_28.csv', sep=';', encoding='utf-8-sig')
current_file = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python/data/imgw_h_all_gauges_2022_to_2024.csv'
curr_df = pd.read_csv(current_file, sep=';')
imgw_ids = imgw_stations['id'].tolist()
imgw_ids = [str(x) for x in imgw_ids]


# sel_files = [file for file in files if file[-10:] == '024_05.csv']
# print(sel_files)
# print(1)
# df_res = pd.DataFrame()
# for file in files:
#     # date_str = file[-11:-4]
#     if file[-10:] == 'imgw_h.csv':
#         imgw_df = pd.read_csv(file, sep=';', usecols=[0, 1, 2], header=0, decimal='.')
#         imgw_df['stage'] = imgw_df["stage"].astype(float)
#         # print(imgw_df['id'])
#         imgw_df['id'] = imgw_df["id"].astype(str)
#         imgw_df = imgw_df.set_index(pd.to_datetime(imgw_df['date'], format='%Y-%m-%d %H:%M:%S'))
#     else:
#         imgw_df = pd.read_csv(file, sep=';', usecols=[0, 2, 3], names=['id', 'date', 'stage'], decimal=',')
#         try:
#             imgw_df['stage'] = imgw_df["stage"].astype(float)
#         except ValueError:
#             imgw_df = imgw_df.iloc[1:]
#             imgw_df['stage'] = imgw_df["stage"].astype(float)
#         # print(imgw_df['id'])
#         imgw_df['id'] = imgw_df["id"].astype(str)
#         imgw_df = imgw_df.set_index(pd.to_datetime(imgw_df['date']))
#
#     # imgw_df = imgw_df.loc[imgw_df['id'].isin(imgw_ids)]
#     imgw_df = imgw_df.groupby('id').resample('H').mean(numeric_only=True)
#     imgw_df.stage = imgw_df.stage.round()
#     if file[-10:] == 'imgw_h.csv':
#         df_res = imgw_df
#     else:
#         df_res = pd.concat([df_res, imgw_df])
#     print(file[-10:])
#
# print(df_res)
# resfile = 'imgw_h_all_gauges_2022_to_2024.csv'
# df_res.to_csv(resfile, sep=';')
# print(1)
# group_files = False
# if group_files:
#     for file in files:
#         # date_str = file[-11:-4]
#         print(file[-10:])
#         if file[-11:] == '2024_07.csv':
#             imgw_df = pd.read_csv(file, sep=';', usecols=[0, 2, 3], names=['id', 'date', 'stage'], decimal=',', skiprows=1)
#         else:
#             imgw_df = pd.read_csv(file, sep=';', usecols=[0, 2, 3], names=['id', 'date', 'stage'], decimal=',')
#         imgw_df['stage'] = imgw_df["stage"].astype(float)
#         # print(imgw_df['id'])
#         imgw_df['id'] = imgw_df["id"].astype(str)
#         imgw_df = imgw_df.set_index(pd.to_datetime(imgw_df['date'], format='%Y-%m-%d %H:%M'))
#         imgw_df = imgw_df.loc[imgw_df['id'].isin(imgw_ids)]
#         imgw_df = imgw_df[['id', 'stage']]
#         imgw_df = imgw_df.groupby('id').resample('H').mean()
#         imgw_df.stage = imgw_df.stage.round()
#         if files.index(file) == 0:
#             df_res = imgw_df
#         else:
#             df_res = pd.concat([df_res, imgw_df])
#
#     print(df_res)
#     resfile = 'imgw_h_2024_08_to_2024_09.csv'
#     df_res.to_csv(resfile, sep=';')


def reorganize_db_meas_to_imgw(metadata, file):
    res_df = pd.DataFrame()
    for col in file.columns:
        curr_df = file[[col]]
        code = metadata['code_gauge'].loc[metadata['id_gauge'] == int(col)].item()
        curr_df['id'] = code
        curr_df = curr_df.rename(columns={col: 'stage'})
        curr_df.index = curr_df.index.rename('date')
        curr_df = curr_df[['id', 'stage']]
        curr_df['stage'] = curr_df['stage'].astype(float) * 100
        res_df = pd.concat([res_df, curr_df])
    res_df = res_df.groupby('id').resample('H').mean()
    return res_df[['stage']]


reorganise = False
if reorganise:
    df_metadata = pd.read_csv('/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro/metadata/gauge/stacje_imgw_data_raw_utf8.csv', sep=';')
    df_meas = pd.read_csv('/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro/db_elephantsql/pred_gauge_1_2023-06-10_2023-09-05.csv', sep=';', index_col=['Unnamed: 0'], parse_dates=True)
    df_meas = pd.read_csv('/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro/db_elephantsql/pred_gauge_1_2023-06-10_2023-09-05.csv', sep=';', index_col=['Unnamed: 0'], parse_dates=True)
    df_reorganised = reorganize_db_meas_to_imgw(df_metadata, df_meas)
    print(df_reorganised)


    # print(df_res)
    df_reorganised.to_csv('imgw_h_db_srv.csv', sep=';', decimal=',')
    # df_res.to_csv('imgw_h_test.csv', sep=';', decimal=',')
    print('________________________________')

prepare_db_like_data_from_raw_imgw = False
if prepare_db_like_data_from_raw_imgw:
    imgw_h = pd.read_csv(
        '/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro/imgw_h_2023_2024.csv',
        sep=';')
    stacje_imgw = pd.read_csv(
        '/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro/linux/program_imgw/metadata/gauge/stacje_imgw_data_raw2.csv',
        sep=';')
    stacje_pairs = stacje_imgw.set_index('code_gauge')['id_gauge'].to_dict()
    pred_gauge_df = imgw_h[['date', 'stage']].loc[imgw_h['id'] == 152140050]
    pred_gauge_df = pred_gauge_df.rename(columns={'date': 'dt', 'stage': '1'})
    pred_gauge_df['dt'] = pd.to_datetime(pred_gauge_df['dt'])
    pred_gauge_df['dt'] = pred_gauge_df['dt'].dt.round('H')
    pred_gauge_df.set_index('dt', inplace=True)
    pred_gauge_df = pred_gauge_df.sort_index()
    for key, val in stacje_pairs.items():
        curr_df = imgw_h[['date', 'stage']].loc[imgw_h['id'] == key]
        curr_df = curr_df.rename(columns={'date': 'dt', 'stage': str(val)})
        curr_df['dt'] = pd.to_datetime(curr_df['dt'])
        curr_df['dt'] = curr_df['dt'].dt.round('H')
        curr_df.set_index('dt', inplace=True)
        curr_df = curr_df.sort_index()
        pred_gauge_df[str(val)] = curr_df[str(val)]
    dt1, dt2 = pred_gauge_df.index[0], pred_gauge_df.index[-1]
    pred_gauge_df = pred_gauge_df.reindex(pd.date_range(dt1, dt2, freq='H'))
    pred_gauge_df = pred_gauge_df / 100
    pred_gauge_df.to_csv(f'pred_gauge_1_{str(dt1)[:10]}_{str(dt2)[:10]}.csv', sep=';')



update_all_meas_file = True
if update_all_meas_file:
    for file in files:
        if '2025' in file:
            imgw_df = pd.read_csv(file, sep=';', usecols=[0, 2, 3], names=['id', 'date', 'stage'], decimal=',')
            try:
                imgw_df['stage'] = imgw_df["stage"].astype(float)
            except ValueError:
                imgw_df = imgw_df.iloc[1:]
                imgw_df['stage'] = imgw_df["stage"].astype(float)
            # print(imgw_df['id'])
            imgw_df['id'] = imgw_df["id"].astype(str)
            imgw_df = imgw_df.set_index(pd.to_datetime(imgw_df['date']))
            imgw_df = imgw_df.groupby('id').resample('H').mean(numeric_only=True)
            imgw_df.stage = imgw_df.stage.round()
            imgw_df['id'] = imgw_df.index.get_level_values('id')
            imgw_df['date'] = imgw_df.index.get_level_values('date')
            imgw_df['index'] = range(len(imgw_df))
            imgw_df = imgw_df.set_index(imgw_df['index'])
            imgw_df = imgw_df[['id', 'date', 'stage']]
            curr_df = pd.concat([curr_df, imgw_df])
    print(curr_df)
    resfile = 'imgw_h_all_gauges_2022_to_2025_04.csv'
    curr_df.to_csv(resfile, sep=';')


get_imgw_metadata = True
if get_imgw_metadata:
    import json
    coords_file = '/Users/michalhalicki/Documents/nauka/dane_gis/imgw_coords.csv'
    metadata_file = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_IMGW/imgw_metadata/all_imgw_metadata.csv'
    df_coords = pd.read_csv(coords_file, sep=',')
    df_metadata = pd.read_csv(metadata_file, sep=';')

    names, rivers = [], []
    for g_id in df_coords['id']:
        try:
            nm, riv = \
            df_metadata[['STATION NAME', 'RIVER/WATERBODY (CODE)']].loc[df_metadata['STATION ID'] == g_id].values[0]
            riv = riv[:riv.index('(') - 1]
        except:
            nm, riv = '', ''
        names.append(nm)
        rivers.append(riv)
    df_coords['station name'] = names
    df_coords['river name'] = rivers
    out_file = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_IMGW/imgw_metadata/all_imgw_metadata_with_coords'
    df_coords.to_csv(f'{out_file}.csv', sep=';')
    res_dict = {row['id']: row.drop('Unnamed: 0').to_dict() for index, row in df_coords.iterrows()}
    out_file_json = f'{out_file}.json'
    with open(out_file_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)