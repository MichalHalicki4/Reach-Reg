import pandas as pd
import dataretrieval.nwis as nwis
from datetime import datetime
import matplotlib.pyplot as plt


def get_river_gage_stations_with_data(
        river_name: str,
        parameter_code: str,
        start_date: str,
        end_date: str,
        state_codes: list = None,  # Zmieniono na listę kodów stanów
        county_code: str = None,  # Pozostawiamy dla elastyczności (będzie działać w ramach jednego stanu)
        # bBox: list = None, # Usuwamy bBox jako parametr bezpośredni, by uniknąć konfliktu logiki
        site_type: str = 'ST',
        has_data_in_date_range: bool = True,
        data_type_to_check: str = 'dv'
) -> pd.DataFrame:
    """
    Znajduje stacje monitorujące stan wody na wybranej rzece, iterując po liście stanów.
    Zwraca również konkretne daty początku/końca dla DV i IV.

    Args:
        river_name (str): Nazwa rzeki (np. "Colorado River"). Używana do dopasowania w nazwie stacji.
                          Może być pusty string, jeśli nie chcesz filtrować po nazwie rzeki,
                          ale wtedy MUSISZ podać state_codes (lub rozważyć bBox poza tą funkcją).
        parameter_code (str): Kod parametru hydrologicznego (np. '00065' dla stanu wody).
                              Jest to argument obowiązkowy i musi być poprawny.
        start_date (str): Data początkowa zakresu w formacie 'YYYY-MM-DD'.
        end_date (str): Data końcowa zakresu w formacie 'YYYY-MM-DD'.
        state_codes (list, optional): Lista dwuliterowych kodów stanów USA (np. ['CO', 'AZ']).
                                      Funkcja będzie iterować po tych stanach.
        county_code (str, optional): Kod hrabstwa. Wymaga podania state_codes z jednym elementem.
        site_type (str, optional): Kod typu stacji (domyślnie 'ST' dla strumienia/rzeki).
        has_data_in_date_range (bool, optional): Czy sprawdzić, czy stacja ma dane w zakresie dat.
        data_type_to_check (str, optional): Typ danych do sprawdzenia ('dv' lub 'iv').

    Returns:
        pd.DataFrame: DataFrame z informacjami o stacjach, które spełniają kryteria,
                      lub pusty DataFrame, jeśli nie znaleziono żadnych stacji.
                      Zawiera kolumny: 'site_no', 'station_nm', 'dec_lat_va', 'dec_long_va',
                      'huc_cd', 'drain_area_va', 'agency_cd', 'site_tp_cd',
                      'begin_date', 'end_date',
                      'dv_{param_code}_begin_date', 'dv_{param_code}_end_date',
                      'iv_{param_code}_begin_date', 'iv_{param_code}_end_date'.
    """

    print(f"Szukanie stacji dla rzeki '{river_name}' monitorujących parametr '{parameter_code}'...")

    if not parameter_code:
        print("BŁĄD: 'parameter_code' jest pusty lub None. To jest wymagany parametr.")
        return pd.DataFrame()

    # Sprawdzanie, czy podano choć jeden filtr przestrzenny
    if not river_name and (not state_codes or len(state_codes) == 0):
        print("BŁĄD: Aby wyszukać stacje, musisz podać co najmniej nazwę rzeki ('river_name') "
              "LUB listę kodów stanów ('state_codes'). Zapytanie globalne jest zbyt szerokie.")
        return pd.DataFrame()

    # Lista do przechowywania DataFrame'ów z każdego stanu
    all_stations_dfs = []

    # Iteracja po każdym stanie, jeśli podano listę stanów
    states_to_iterate = state_codes if state_codes else [None]  # Jeśli state_codes jest None, iteruj raz z None

    for current_state_code in states_to_iterate:
        print(
            f"Pobieranie stacji dla stanu: {current_state_code if current_state_code else 'WSZYSTKIE STANY (ostrożnie!)'}...")

        # Parametry do search_args (tylko siteStatus)
        search_arguments = {
            'siteStatus': 'active'
        }

        try:
            # Przekazywanie parametrów bezpośrednio do nwis.get_info()
            sites_for_state, metadata_info = nwis.get_info(
                huc=None,
                stateCd=current_state_code,  # Używamy bieżącego stanu z iteracji
                countyCd=county_code,  # County code będzie działał tylko w ramach jednego state_code
                bBox=None,  # Tutaj bBox jest celowo None, bo iterujemy po stanach
                siteType=site_type,
                parameterCd=parameter_code,
                siteName=river_name,
                search_args=search_arguments
            )
            if not sites_for_state.empty:
                all_stations_dfs.append(sites_for_state)
            else:
                print(f"Brak stacji dla '{river_name}' w stanie {current_state_code}.")

        except Exception as e:
            print(f"Błąd podczas pobierania informacji o stacjach dla stanu {current_state_code}: {e}. "
                  f"Upewnij się, że podane filtry są poprawne i sensowne.")
            if "Bad Request" in str(e):
                print(
                    "Prawdopodobną przyczyną jest zbyt ogólne zapytanie (jeśli state_code był None) lub inny błąd parametru.")
            # Kontynuujemy do kolejnego stanu, nie przerywamy funkcji

    if not all_stations_dfs:
        print(
            f"Nie znaleziono żadnych stacji dla rzeki '{river_name}' z parametrem '{parameter_code}' w podanych stanach.")
        return pd.DataFrame()

    # Łączenie wszystkich znalezionych DataFrame'ów
    sites_on_river = pd.concat(all_stations_dfs).drop_duplicates(subset=['site_no']).reset_index(drop=True)

    print(
        f"Znaleziono {len(sites_on_river)} unikalnych potencjalnych stacji monitorujących stan wody na rzece '{river_name}' w podanych stanach.")

    # Reszta funkcji pozostaje bez większych zmian, odfiltrowujemy po zakresie dat
    extra_date_cols = [
        f'dv_{parameter_code}_begin_date', f'dv_{parameter_code}_end_date',
        f'iv_{parameter_code}_begin_date', f'iv_{parameter_code}_end_date'
    ]
    base_cols = ['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va',
                 'huc_cd', 'drain_area_va', 'agency_cd', 'site_tp_cd',
                 'begin_date', 'end_date']

    cols_to_return = [col for col in base_cols + extra_date_cols if col in sites_on_river.columns]

    if not has_data_in_date_range:
        return sites_on_river[cols_to_return].reset_index(drop=True)

    filtered_sites = []
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    for index, site_row in sites_on_river.iterrows():
        site_no = site_row['site_no']
        station_nm = site_row['station_nm']

        if data_type_to_check == 'dv':
            param_begin_date_col = f'dv_{parameter_code}_begin_date'
            param_end_date_col = f'dv_{parameter_code}_end_date'
        elif data_type_to_check == 'iv':
            param_begin_date_col = f'iv_{parameter_code}_begin_date'
            param_end_date_col = f'iv_{parameter_code}_end_date'
        else:
            param_begin_date_col = f'prws_{parameter_code}_begin_date'
            param_end_date_col = f'prws_{parameter_code}_end_date'

        station_begin_date = pd.to_datetime(site_row.get(param_begin_date_col, pd.NaT))
        station_end_date = pd.to_datetime(site_row.get(param_end_date_col, pd.NaT))

        if (pd.isna(station_begin_date) or station_begin_date <= end_dt) and \
                (pd.isna(station_end_date) or station_end_date >= start_dt):
            filtered_sites.append(site_row)
        else:
            pass

    if not filtered_sites:
        print(
            f"Brak stacji z danymi stanu wody dla rzeki '{river_name}' ({data_type_to_check.upper()} data) w zakresie {start_date} do {end_date}.")
        return pd.DataFrame()

    df_filtered_sites = pd.DataFrame(filtered_sites)
    print(
        f"Znaleziono {len(df_filtered_sites)} stacji z danymi ({data_type_to_check.upper()} data) w podanym zakresie.")

    return df_filtered_sites[cols_to_return].reset_index(drop=True)


download_gauge_data_for_river = True
if download_gauge_data_for_river:
    gage_height_param = "00065" # Kod dla Gage Height (stan wody)
    start_date = "2023-07-01"
    end_date = "2025-06-01"
    river_name = "Missouri River"
    msspi_river_states = ["AL", "IL", "MS", "MO"]
    mssri_river_states = ['SD', 'IA', 'NE', 'MO', 'KS']
    # red_river_states = ["TX", "OK", "AR", "LA"]
    # Wywołanie funkcji
    stations_on_river = get_river_gage_stations_with_data(
        river_name= river_name,
        parameter_code=gage_height_param,
        start_date=start_date,
        end_date=end_date,
        state_codes=mssri_river_states,
        has_data_in_date_range=True
    )
    stations_on_river.to_csv(f'/Users/michalhalicki/Documents/nauka/dane_gis/dane_USGS/{river_name}_gauge_metadata.csv')
    df_res = pd.DataFrame()
    for i in range(len(stations_on_river)):
        site_code = stations_on_river['site_no'].iloc[i]
        # Retrieve the data
        df_gage_height, metadata = nwis.get_iv(
            sites=site_code,
            start=start_date,
            end=end_date,
            parameterCd=gage_height_param
        )

        aggregation_rules = {
            'id': 'first',  # Bierzemy pierwszą wartość ID w danym interwale
            'stage': 'mean'  # Obliczamy średnią ze stanu wody w danym interwale
        }
        if len(df_gage_height) == 0:
            continue
        col1, col2, col3 = df_gage_height.columns
        df = df_gage_height.loc[
            (df_gage_height[col3] == 'P') | (df_gage_height[col3] == 'A')]
        df = df[[col1, col2]].rename(columns={col1: 'id', col2: 'stage'})
        df = df.resample('H').agg(aggregation_rules).reset_index().dropna()
        df['stage'] = round(df['stage'] * 0.3048, 2)
        df = df.rename(columns={'datetime':'date'})

        if len(df_res) == 0:
            df_res = df
        else:
            df_res = pd.concat([df_res, df])
        print(stations_on_river['site_no'].iloc[i], len(df_res))

    output_path = f'/Users/michalhalicki/Documents/nauka/dane_gis/dane_USGS/{river_name}_gauge_data.csv'
    df_res.to_csv(output_path, sep=';')
    print(1)

plot_all_gauges = True
if plot_all_gauges:
    # river_name = "Red River"
    output_path = f'/Users/michalhalicki/Documents/nauka/dane_gis/dane_USGS/{river_name}_gauge_data.csv'
    df_res = pd.read_csv(output_path, sep=';')
    fig, ax = plt.subplots()
    for g_id in df_res['id'].unique():
        curr_df = df_res.loc[df_res['id'] == g_id]
        ax.plot(pd.to_datetime(curr_df['date']), curr_df['stage'])
    plt.show(block='True')
    print(1)