from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

gdata_dir = '/Users/michalhalicki/Documents/nauka/dane_gis/dane_GANGES/'


def filter_outliers_by_tstudent_test(df, column, window_days=3, min_periods=3, confidence_level=0.99, plot_outliers=False):
    df = df.copy(deep=True)
    alpha = 1 - confidence_level  # Poziom istotności (0.01)
    z_critical = stats.norm.ppf(1 - alpha / 2)  # Dla 99% to ok. 2.576
    df['Rolling_Mean'] = df[column].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).mean()
    df['Rolling_Std'] = df[column].rolling(window=f'{window_days}D', center=True, min_periods=min_periods).std()
    df['Lower_Bound'] = df['Rolling_Mean'] - (z_critical * df['Rolling_Std'])
    df['Upper_Bound'] = df['Rolling_Mean'] + (z_critical * df['Rolling_Std'])
    df['Is_Outlier'] = (df[column] < df['Lower_Bound']) | \
                       (df[column] > df['Upper_Bound'])

    # print(f"\nŁączna liczba zidentyfikowanych outlierów: {df['Is_Outlier'].sum()}")
    if plot_outliers:
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df[column], label='Stan wody', alpha=0.8, marker='.')
        plt.plot(df.index, df['Rolling_Mean'], label='Średnia Krocząca', color='orange', linestyle='--')
        plt.plot(df.index, df['Lower_Bound'], label='Dolny Limit Ufności (99%)', color='red', linestyle=':')
        plt.plot(df.index, df['Upper_Bound'], label='Górny Limit Ufności (99%)', color='red', linestyle=':')

        # Zaznaczanie outlierów
        outliers = df[df['Is_Outlier']]
        plt.scatter(outliers.index, outliers[column], color='purple', marker='o', s=50, zorder=5,
                    label='Zidentyfikowane Outliery')

        plt.title('Wykrywanie Outlierów w Znormalizowanym Szeregu Czasowym')
        plt.xlabel('Data')
        plt.ylabel('Znormalizowany Poziom Wody')
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
    return df.loc[(-df['Is_Outlier'])]


gauges_metadata = pd.read_excel(f'{gdata_dir}ganges_gauge_metadata.xlsx')
results_df = pd.DataFrame()
for gauge_name in gauges_metadata['name']:
    curr_df = pd.read_excel(f'{gdata_dir}data_{gauge_name}.xlsx')
    curr_df = curr_df.set_index(pd.to_datetime(curr_df['time']))
    curr_df = filter_outliers_by_tstudent_test(curr_df, 'WSE', plot_outliers=False)
    curr_df = pd.DataFrame(curr_df['WSE'].resample('h').mean())
    curr_df['name'] = gauge_name
    curr_df['dt'] = curr_df.index
    curr_df['index'] = range(len(results_df), len(results_df) + len(curr_df))
    curr_df = curr_df.set_index(curr_df['index'])
    results_df = pd.concat([results_df, curr_df])
results_df = results_df[['name', 'dt', 'WSE']]
results_df.to_csv(f'{gdata_dir}ganges_gauge_data.csv', sep=';', decimal=',')