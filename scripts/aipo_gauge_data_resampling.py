import glob
import pandas as pd
import numpy as np
import json

files = glob.glob('/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/dane_hydro/AIPO_Po_River_Italy/*.csv')
metadata_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/dane_hydro/AIPO_Po_River_Italy/po_gauge_metadata.xlsx'
stations_metadata = pd.read_excel(metadata_file, decimal=',')
stations_metadata = stations_metadata.rename(columns={'Latitude': 'Y', 'Longitude': 'X'})
stations_metadata = stations_metadata[['Name', 'ID', 'X', 'Y']]
res_dict = {row['ID']: row.to_dict() for index, row in stations_metadata.iterrows()}
out_file = '/Users/michalhalicki/Library/CloudStorage/OneDrive-UniwersytetWrocławski/dane_gis/dane_hydro/AIPO_Po_River_Italy/po_gauge_metadata'
with open(f'{out_file}.json', 'w', encoding='utf-8') as f:
    json.dump(res_dict, f, ensure_ascii=False, indent=4)


res_df = pd.DataFrame()
for file in files:
    if file == metadata_file:
        continue
    df = pd.read_csv(file)
    try:
        gauge_name = df.columns[1][:df.columns[1].index(' - Idrometro')]
    except ValueError:
        gauge_name = df.columns[1][:df.columns[1].index(' - Water')]
    gauge_id = stations_metadata['ID'].loc[stations_metadata['Name'] == gauge_name].values[0]
    df = df.rename(columns={'Time': 'date', df.columns[1]: 'stage'})
    curr_df = df[['date', 'stage']]
    curr_df['id'] = gauge_id
    curr_df['stage'] = curr_df["stage"].replace('-', np.nan).astype(float)
    curr_df = curr_df.set_index(pd.to_datetime(curr_df['date']))
    curr_df = curr_df.groupby('id').resample('h').mean(numeric_only=True)
    curr_df.stage = curr_df.stage.round(decimals=2)
    curr_df['id'] = curr_df.index.get_level_values('id')
    curr_df['date'] = curr_df.index.get_level_values('date')
    curr_df['index'] = range(len(curr_df))
    curr_df = curr_df.set_index(curr_df['index'])
    curr_df = curr_df[['id', 'date', 'stage']]
    res_df = pd.concat([res_df, curr_df])

print(res_df)
resfile = 'po_water_levels_0723_to_0725.csv'
res_df.to_csv(resfile, sep=';')
