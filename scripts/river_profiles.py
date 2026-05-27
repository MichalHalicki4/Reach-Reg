import os
import pickle
import numpy as np
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator


def prepare_river_data(files, riv_metadata):
    """
    Tworzy dwa DataFrame: jeden z WSE, drugi z WSE_U.
    Kolumny są posortowane po chainage.
    """
    all_wse = []
    all_wse_u = []

    for f in files:
        station_id = re.search(r'RS(\d+)', f).group(1)
        # Wczytujemy dane, upewniając się, że indeks to datetime
        df_temp = pd.read_csv(f, sep=';', parse_dates=[0], index_col=0)

        all_wse.append(df_temp['wse'].rename(station_id))
        all_wse_u.append(df_temp['wse_u'].rename(station_id))

    # Łączymy w DataFrame'y
    master_wse = pd.concat(all_wse, axis=1)
    master_wse_u = pd.concat(all_wse_u, axis=1)

    # Sortowanie na podstawie metadanych
    riv_metadata['id'] = riv_metadata['id'].astype(str)
    sorted_meta = riv_metadata.sort_values('chain')

    available_stations = [s for s in sorted_meta['id'] if s in master_wse.columns]

    master_wse = master_wse[available_stations]
    master_wse_u = master_wse_u[available_stations]

    return master_wse, master_wse_u, sorted_meta


def plot_river_profile(date_str, df_wse, df_wse_u, df_meta):
    """
    Rysuje profil podłużny z wstęgą niepewności dla danej daty.
    """
    if date_str not in df_wse.index:
        print(f"Brak danych dla daty {date_str}")
        return

    # Pobieramy dane dla konkretnego dnia
    altitudes = df_wse.loc[date_str]
    uncertainty = df_wse_u.loc[date_str]

    # Łączymy w tymczasowy DF, żeby łatwo usunąć braki danych (NaN) dla tego konkretnego dnia
    temp_df = pd.DataFrame({
        'wse': altitudes,
        'u': uncertainty,
        'chain': df_meta.set_index('id').loc[altitudes.index, 'chain']
    }).dropna(subset=['wse'])  # Usuwamy stacje, które nie miały przelotu w tym dniu

    if temp_df.empty:
        print(f"Brak dostępnych pomiarów dla daty {date_str}")
        return

    plt.figure(figsize=(12, 6))

    # Rysujemy wstęgę niepewności (WSE +/- WSE_U)
    plt.fill_between(temp_df['chain'],
                     temp_df['wse'] - temp_df['u'],
                     temp_df['wse'] + temp_df['u'],
                     color='b', alpha=0.2, label='Niepewność (wse_u)')

    # Rysujemy linię profilu
    plt.plot(temp_df['chain'], temp_df['wse'], marker='o', markersize=4,
             linestyle='-', color='b', linewidth=1.5, label=f'Profil: {date_str}')

    plt.xlabel('Kilometraż rzeki (chainage) [m]')
    plt.ylabel('WSE [m n.p.m.]')
    plt.title(f'Profil podłużny rzeki {river} - {date_str}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


def animate_river_profile(start_date, end_date, df_wse, df_wse_u, df_meta, interval=100, save_path=None):
    """
    Tworzy animację profilu podłużnego rzeki dla wybranego zakresu dat.
    """
    # Filtrowanie zakresu dat
    mask = (df_wse.index >= start_date) & (df_wse.index <= end_date)
    dates = df_wse.index[mask]

    if len(dates) == 0:
        print("Brak danych dla podanego zakresu dat.")
        return

    # Inicjalizacja figury
    fig, ax = plt.subplots(figsize=(12, 6))

    # Stałe limity osi dla lepszej percepcji zmian (z marginesem)
    y_min = df_wse.loc[dates].min().min() - 1.0
    y_max = df_wse.loc[dates].max().max() + 1.0
    x_min = df_meta['chain'].min()
    x_max = df_meta['chain'].max()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Kilometraż rzeki (chainage) [m]')
    ax.set_ylabel('WSE [m n.p.m.]')
    ax.grid(True, linestyle='--', alpha=0.5)

    title = ax.set_title('')
    line, = ax.plot([], [], marker='o', markersize=4, color='b', lw=1.5, label='WSE')

    # Przygotowanie fill_between (wymaga triku z usuwaniem starej kolekcji)
    poly_collection = [None]

    def update(frame_date):
        date_str = frame_date.strftime('%Y-%m-%d')

        # Pobieranie i czyszczenie danych dla klatki
        altitudes = df_wse.loc[frame_date]
        uncertainty = df_wse_u.loc[frame_date]

        temp_df = pd.DataFrame({
            'wse': altitudes,
            'u': uncertainty,
            'chain': df_meta.set_index('id').loc[altitudes.index, 'chain']
        }).dropna(subset=['wse'])

        # Aktualizacja linii i tytułu
        if not temp_df.empty:
            line.set_data(temp_df['chain'], temp_df['wse'])

            # Aktualizacja fill_between
            if poly_collection[0] is not None:
                poly_collection[0].remove()

            poly_collection[0] = ax.fill_between(
                temp_df['chain'],
                temp_df['wse'] - temp_df['u'],
                temp_df['wse'] + temp_df['u'],
                color='b', alpha=0.2
            )

        title.set_text(f'Profil podłużny rzeki - {date_str}')
        return line, title

    # Tworzenie animacji
    # interval = 100ms oznacza 10 klatek na sekundę
    ani = FuncAnimation(fig, update, frames=dates, interval=interval, blit=False)

    if save_path:
        print(f"Zapisywanie animacji do: {save_path}...")
        ani.save(save_path, writer='ffmpeg', fps=10)
        print("Zapis zakończony.")
    else:
        plt.legend(loc='upper right')
        plt.show(block=True)


def collect_pkl_data(river_name, data_path):
    pattern = os.path.join(data_path, f"{river_name}_RS*.pkl")
    files = glob.glob(pattern)

    all_data_frames = []

    for file in files:
        with open(file, 'rb') as f:
            rs_obj = pickle.load(f)
            # Pobieramy densified_ts
            df_vs = rs_obj.densified_ts[['id_vs', 'vs_chain', 'dt', 'vs_wl']].copy()

            # Zaokrąglamy daty do pełnych dni
            df_vs['dt'] = pd.to_datetime(df_vs['dt']).dt.normalize()
            df_vs['vs_chain'] = df_vs['vs_chain']/1000
            all_data_frames.append(df_vs)

    # Łączymy wszystko w jeden długi format (Long Format)
    full_df = pd.concat(all_data_frames, ignore_index=True)

    # Usuwamy ewentualne duplikaty (ten sam VS, ta sama data, to samo WL)
    full_df = full_df.drop_duplicates(subset=['id_vs', 'dt'])

    return full_df


def collect_pkl_data_with_coords(river_name, data_path):
    pattern = os.path.join(data_path, f"{river_name}_RS*.pkl")
    files = glob.glob(pattern)

    all_ts_frames = []
    vs_coords = {}  # Słownik: id_vs -> (x, y)

    for file in files:
        with open(file, 'rb') as f:
            rs_obj = pickle.load(f)

            # 1. Pobieramy szeregi czasowe
            df_vs = rs_obj.densified_ts[['id_vs', 'vs_chain', 'dt', 'vs_wl']].copy()
            df_vs['dt'] = pd.to_datetime(df_vs['dt']).dt.normalize()
            all_ts_frames.append(df_vs)

            # 2. Budujemy mapę współrzędnych z obiektów VS
            # Przeszukujemy listę sąsiednich VS przypisaną do tego rs
            if hasattr(rs_obj, 'upstream_adjacent_vs'):
                for vs in rs_obj.upstream_adjacent_vs:
                    if vs.id not in vs_coords:
                        vs_coords[vs.id] = (vs.x, vs.y)

    # Łączymy wszystkie szeregi
    full_df = pd.concat(all_ts_frames, ignore_index=True).drop_duplicates(subset=['id_vs', 'dt'])

    # 3. Mapujemy współrzędne na główny DataFrame
    # Tworzymy pomocniczy DF ze współrzędnymi
    coords_df = pd.DataFrame.from_dict(vs_coords, orient='index', columns=['x', 'y']).reset_index()
    coords_df.rename(columns={'index': 'id_vs'}, inplace=True)

    # Łączymy (Left Join) - teraz każdy pomiar ma swoje X i Y
    full_df = full_df.merge(coords_df, on='id_vs', how='left')

    # Zamiana chainage na kilometry (zgodnie z Twoim odkryciem!)
    full_df['vs_chain'] = full_df['vs_chain'] / 1000.0

    return full_df


def animate_sparse_river(df_long, river_name, save_path=None):
    # Unikalne daty posortowane chronologicznie
    dates = sorted(df_long['dt'].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    # Ustalenie stałych limitów dla osi Y i X
    y_min, y_max = df_long['vs_wl'].min() - 0.5, df_long['vs_wl'].max() + 0.5
    x_min, x_max = df_long['vs_chain'].min() - 1000, df_long['vs_chain'].max() + 1000

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Chainage [m]')
    ax.set_ylabel('WSE (vs_wl) [m]')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Inicjalizacja pustego wykresu kropkowego
    scatter = ax.scatter([], [], c='blue', s=30, edgecolors='black', alpha=0.7, label='VS Measurements')
    title = ax.set_title('')

    def update(frame_date):
        # Filtrujemy dane tylko dla konkretnego dnia
        day_data = df_long[df_long['dt'] == frame_date]

        if not day_data.empty:
            # Aktualizujemy pozycje kropek (X = chainage, Y = water level)
            offsets = np.column_stack((day_data['vs_chain'], day_data['vs_wl']))
            scatter.set_offsets(offsets)
        else:
            # Jeśli brak danych w danym dniu, czyścimy kropki
            scatter.set_offsets(np.empty((0, 2)))

        title.set_text(f'River: {river_name} | Date: {frame_date.strftime("%Y-%m-%d")}')
        return scatter, title

    ani = FuncAnimation(fig, update, frames=dates, interval=200, blit=False)

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=5)
    else:
        plt.legend()
        plt.show(block=True)


def animate_river_combined(start_date, end_date, df_wse, df_wse_u, df_wse_amp, df_meta, df_long, interval=200):
    # 1. Przygotowanie wspólnych dat
    mask = (df_wse.index >= start_date) & (df_wse.index <= end_date)
    dates = df_wse.index[mask]

    # 2. Inicjalizacja figury z dwoma subplotami
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    x_min, x_max = df_meta['chain'].min(), df_meta['chain'].max()

    # --- Konfiguracja AX1 (WSE Bezwzględne + VS) ---
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(df_wse.loc[dates].min().min() - 1, df_wse.loc[dates].max().max() + 1)
    ax1.set_ylabel('WSE [m n.p.m.]')
    ax1.grid(True, linestyle='--', alpha=0.5)

    line_wse, = ax1.plot([], [], 'b-', lw=1.5, label='Modeled WSE', zorder=2)
    scatter_vs = ax1.scatter([], [], c='red', s=25, edgecolors='black', label='VS Obs (Sparse)', zorder=3)
    poly_wse = [None]  # Na fill_between

    # --- Konfiguracja AX2 (Anomalie) ---
    y_amp_lim = max(abs(df_wse_amp.loc[dates].min().min()), abs(df_wse_amp.loc[dates].max().max())) + 0.5
    ax2.set_ylim(-y_amp_lim, y_amp_lim)
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.set_ylabel('Anomaly [m]')
    ax2.set_xlabel('Chainage [m]')
    ax2.grid(True, linestyle='--', alpha=0.5)

    line_amp, = ax2.plot([], [], 'g-', lw=1.5, label='WSE Anomaly')
    poly_amp = [None]

    def update(frame_date):
        frame_date = pd.Timestamp(frame_date)
        date_str = frame_date.strftime('%Y-%m-%d')

        # --- 1. Update AX1 (Bezwzględne WSE) ---
        # Musimy pobrać dane z df_wse dla tej konkretnej klatki
        data_day = pd.DataFrame({
            'wse': df_wse.loc[frame_date],
            'u': df_wse_u.loc[frame_date],
            'chain': df_meta.set_index('id').loc[df_wse.columns, 'chain']
        }).dropna(subset=['wse'])

        if not data_day.empty:
            line_wse.set_data(data_day['chain'], data_day['wse'])
            if poly_wse[0] is not None:
                poly_wse[0].remove()
            poly_wse[0] = ax1.fill_between(data_day['chain'],
                                           data_day['wse'] - data_day['u'],
                                           data_day['wse'] + data_day['u'],
                                           color='b', alpha=0.2, zorder=1)
        else:
            line_wse.set_data([], [])

        # --- 2. Update VS Data (Kropki Sparse) ---
        vs_day = df_long[df_long['dt'] == frame_date]
        if not vs_day.empty:
            x = vs_day['vs_chain'].values.astype(float)
            y = vs_day['vs_wl'].values.astype(float)
            scatter_vs.set_offsets(np.c_[x, y])
        else:
            scatter_vs.set_offsets(np.empty((0, 2)))

        # --- 3. Update AX2 (Anomalie/Amplitudy) ---
        amp_day = pd.DataFrame({
            'amp': df_wse_amp.loc[frame_date],
            'u': df_wse_u.loc[frame_date],
            'chain': df_meta.set_index('id').loc[df_wse_amp.columns, 'chain']
        }).dropna(subset=['amp'])

        if not amp_day.empty:
            line_amp.set_data(amp_day['chain'], amp_day['amp'])
            if poly_amp[0] is not None:
                poly_amp[0].remove()
            poly_amp[0] = ax2.fill_between(amp_day['chain'],
                                           amp_day['amp'] - amp_day['u'],
                                           amp_day['amp'] + amp_day['u'],
                                           color='g', alpha=0.2, zorder=1)
        else:
            line_amp.set_data([], [])

        fig.suptitle(f'River Profile Analysis: {river} | Date: {date_str}', fontsize=14)
        return line_wse, scatter_vs, line_amp

    ani = FuncAnimation(fig, update, frames=dates, interval=interval, blit=False)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

def animate_river_combined_map(start_date, end_date, df_wse, df_wse_u, df_wse_amp, df_meta, df_long, gdf_river,
                               interval=200, save_path=None):
    """
    Kompleksowa animacja:
    1. Profil podłużny WSE + kropki VS (Góra)
    2. Profil anomalii (Dół)
    3. Inset map z lokalizacją przelotów (Wewnątrz górnego plotu)
    """
    # 1. Przygotowanie zakresu dat
    mask = (df_wse.index >= start_date) & (df_wse.index <= end_date)
    dates = df_wse.index[mask]

    if len(dates) == 0:
        print("Brak danych w podanym zakresie.")
        return

    # 2. Inicjalizacja figury
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # --- Konfiguracja AX1 (Profil bezwzględny) ---
    x_min, x_max = df_meta['chain'].min(), df_meta['chain'].max()
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(df_wse.loc[dates].min().min() - 1, df_wse.loc[dates].max().max() + 1)
    ax1.set_ylabel('WSE [m n.p.m.]')
    ax1.grid(True, linestyle='--', alpha=0.5)

    line_wse, = ax1.plot([], [], 'b-', lw=1.5, label='Modeled WSE', zorder=2)
    scatter_vs = ax1.scatter([], [], c='red', s=50, edgecolors='black', label='VS Observation', zorder=3)
    poly_wse = [None]  # Kontener na fill_between

    # --- Konfiguracja INSET MAP (Mapa wewnątrz AX1) ---
    ax_map = inset_axes(ax1, width="30%", height="40%", loc='upper left', borderpad=3)
    # Rysujemy rzekę
    gdf_river.plot(ax=ax_map, color='lightblue', edgecolor='blue', linewidth=0.8, zorder=1)
    scatter_map = ax_map.scatter([], [], c='red', s=20, edgecolors='black', zorder=2)

    # --- RAMKA I WSPÓŁRZĘDNE 4326 ---
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    # Ustawiamy automatyczne szukanie "ładnych" pełnych stopni/wartości
    ax_map.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax_map.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    # Formatowanie etykiet: dodajemy stopnie i kierunki
    from matplotlib.ticker import FormatStrFormatter
    ax_map.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.0f}°E'))
    ax_map.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{y:.0f}°N'))

    ax_map.tick_params(axis='both', labelsize=8, pad=2)
    ax_map.grid(True, linestyle=':', alpha=0.4)

    # Opcjonalnie: wymuszenie proporcji geograficznych (ważne w 4326)
    ax_map.set_aspect('equal')

    # --- Konfiguracja AX2 (Anomalie) ---
    y_amp_lim = max(abs(df_wse_amp.loc[dates].min().min()), abs(df_wse_amp.loc[dates].max().max())) + 0.5
    ax2.set_ylim(-y_amp_lim, y_amp_lim)
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.set_ylabel('Anomaly [m]')
    ax2.set_xlabel('Chainage [km]')
    ax2.grid(True, linestyle='--', alpha=0.5)

    line_amp, = ax2.plot([], [], 'g-', lw=1.5, label='WSE Anomaly')
    poly_amp = [None]

    def update(frame_date):
        frame_date = pd.Timestamp(frame_date)
        date_str = frame_date.strftime('%Y-%m-%d')

        # --- Update Model (AX1 i AX2) ---
        # Przygotowujemy dane modelu dla klatki
        current_wse = df_wse.loc[frame_date]
        current_amp = df_wse_amp.loc[frame_date]
        current_u = df_wse_u.loc[frame_date]

        # Tworzymy tymczasowy DF dla poprawnego mapowania chainage
        temp_df = pd.DataFrame({
            'wse': current_wse,
            'amp': current_amp,
            'u': current_u,
            'chain': df_meta.set_index('id').loc[df_wse.columns, 'chain']
        }).dropna(subset=['wse'])

        if not temp_df.empty:
            # Góra - WSE
            line_wse.set_data(temp_df['chain'], temp_df['wse'])
            if poly_wse[0] is not None: poly_wse[0].remove()
            poly_wse[0] = ax1.fill_between(temp_df['chain'], temp_df['wse'] - temp_df['u'],
                                           temp_df['wse'] + temp_df['u'], color='b', alpha=0.15, zorder=1)
            # Dół - Amplituda
            line_amp.set_data(temp_df['chain'], temp_df['amp'])
            if poly_amp[0] is not None: poly_amp[0].remove()
            poly_amp[0] = ax2.fill_between(temp_df['chain'], temp_df['amp'] - temp_df['u'],
                                           temp_df['amp'] + temp_df['u'], color='g', alpha=0.15, zorder=1)
        else:
            line_wse.set_data([], [])
            line_amp.set_data([], [])

        # --- Update Obserwacje (Profil + Mapa) ---
        vs_day = df_long[df_long['dt'] == frame_date].dropna(subset=['vs_wl', 'vs_chain'])

        if not vs_day.empty:
            # Kropki na profilu podłużnym
            scatter_vs.set_offsets(np.column_stack((vs_day['vs_chain'], vs_day['vs_wl'])))
            # Kropki na mapce inset
            if 'x' in vs_day.columns and 'y' in vs_day.columns:
                scatter_map.set_offsets(np.column_stack((vs_day['x'], vs_day['y'])))
        else:
            scatter_vs.set_offsets(np.empty((0, 2)))
            scatter_map.set_offsets(np.empty((0, 2)))

        fig.suptitle(f'River Analysis: {river} | Date: {date_str}', fontsize=15, fontweight='bold')
        return line_wse, scatter_vs, scatter_map, line_amp

    # Tworzenie animacji
    ani = FuncAnimation(fig, update, frames=dates, interval=interval, blit=False)

    ax1.legend(loc='upper right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        print(f"Zapisywanie do {save_path}...")
        ani.save(save_path, writer='ffmpeg', fps=1000 // interval)
    else:
        plt.show(block=True)


def animate_river_full_dashboard(start_date, end_date, df_wse, df_wse_u, df_wse_amp, df_meta, df_long, gdf_river, riv,
                                 gdf_world, vs_id=0, interval=200, polish=False, save_path=None):
    # 1. Przygotowanie zakresu dat
    mask = (df_wse.index >= start_date) & (df_wse.index <= end_date)
    dates = df_wse.index[mask]
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 2. Wybór stacji referencyjnej (środek rzeki)
    if vs_id == 0:
        middle_chain = (df_meta['chain'].min() + df_meta['chain'].max()) / 2
        ref_station_id = df_meta.iloc[(df_meta['chain'] - middle_chain).abs().argsort()[:1]]['id'].values[0]
    else:
        ref_station_id = vs_id
    ref_chain = df_meta[df_meta['id'] == ref_station_id]['chain'].values[0]
    mask_ref = df_long['id_vs'].astype(str).str.contains(str(ref_station_id), na=False)
    ref_x = df_long[mask_ref]['x'].iloc[0]
    ref_y = df_long[mask_ref]['y'].iloc[0]
    ts_data = df_wse[ref_station_id].loc[dates]

    # 3. Inicjalizacja figury
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8),
                                        gridspec_kw={'height_ratios': [4, 2, 1]})

    # --- AX1 (Profil WSE) ---
    x_min, x_max = df_meta['chain'].min(), df_meta['chain'].max()
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(df_wse.loc[dates].min().min() - 0.5, df_wse.loc[dates].max().max() + 0.5)
    if polish:
        ax1.set_ylabel('Wysokość [m]')
        ax1.set_xlabel('Kilometraż rzeki [km]')  # Dodany opis
        line_wse, = ax1.plot([], [], 'b-', lw=1.5, label='Wysokość lustra wody', zorder=2)
        scatter_vs = ax1.scatter([], [], s=50, color='red', label='Pomiary satelitarne', edgecolors='none', zorder=3)
        poly_wse = [None]
        uncertainty_proxy_wse = ax1.fill_between([], [], [], color='b', alpha=0.1, label='Wysokość lustra wody ±σ')
    else:
        ax1.set_ylabel('WSE [m]')
        ax1.set_xlabel('Chainage [km]')  # Dodany opis
        line_wse, = ax1.plot([], [], 'b-', lw=1.5, label='Daily WSE', zorder=2)
        scatter_vs = ax1.scatter([], [], s=50, color='red', label='WSE points', edgecolors='none', zorder=3)
        poly_wse = [None]
        uncertainty_proxy_wse = ax1.fill_between([], [], [], color='b', alpha=0.1, label='Daily WSE ±σ')

    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- INSET MAP ---
    ax_map = inset_axes(ax1, width="25%", height="35%", loc='upper left', borderpad=4)

    b = gdf_river.total_bounds  # [minx, miny, maxx, maxy]
    dx = b[2] - b[0]
    dy = b[3] - b[1]

    # Chcemy, żeby okno miało przynajmniej 0.4 stopnia szerokości/wysokości
    min_span = 0.8
    span_x = max(dx, min_span) + 0.1  # bufor
    span_y = max(dy, min_span) + 0.1  # bufor

    center_x = (b[0] + b[2]) / 2
    center_y = (b[1] + b[3]) / 2

    ax_map.set_xlim(center_x - span_x / 2, center_x + span_x / 2)
    ax_map.set_ylim(center_y - span_y / 2, center_y + span_y / 2)

    # 2. Rysowanie warstw
    # Granice państw (szare, cienkie linie)
    gdf_world.plot(ax=ax_map, color='none', edgecolor='gray', lw=0.5, alpha=0.5, zorder=0)

    # Rzeka (niebieska)
    gdf_river.plot(ax=ax_map, color='lightblue', edgecolor='blue', linewidth=0.8, zorder=1)

    # Dynamiczne kropki
    scatter_map = ax_map.scatter([], [], s=20, edgecolors='none', zorder=5)
    marker_map, = ax_map.plot([ref_x], [ref_y], marker='v', color='m', markersize=8, zorder=3, linestyle='None')
    # 3. Estetyka ramki (Przywrócona!)
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    # Współrzędne 4326
    ax_map.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax_map.yaxis.set_major_locator(MaxNLocator(nbins=2))
    ax_map.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1f}°E'))
    ax_map.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{y:.0f}°N'))
    ax_map.tick_params(axis='both', labelsize=7, pad=2)
    ax_map.set_aspect('equal')

    # --- AX2 (Anomalie) ---
    ax2.set_xlim(x_min, x_max)
    y_amp_lim = max(abs(df_wse_amp.loc[dates].min().min()), abs(df_wse_amp.loc[dates].max().max())) + 0.3
    ax2.set_ylim(-y_amp_lim, y_amp_lim)
    ax2.axhline(0, color='black', lw=1, ls='--')
    if polish:
        ax2.set_ylabel('Anomalia stanu\nwody [m]')
        ax2.set_xlabel('Kilometraż rzeki [km]')  # Dodany opis
    else:
        ax2.set_ylabel('Anomaly [m]')
        ax2.set_xlabel('Chainage [km]')  # Dodany opis
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.axvline(x=ref_chain, linestyle='--', alpha=0.7, color='m')
    ax2.scatter([ref_chain], [-y_amp_lim + 1], marker='v', c='m', s=75)
    line_amp, = ax2.plot([], [], 'g-', label='WSE anomalies', lw=1.5)
    poly_amp = [None]

    # --- AX3 (Szereg Czasowy) ---
    if polish:
        ax3.plot(ts_data.index, ts_data.values, color='gray', lw=1, alpha=0.4,
                 label=f'Wysokość lustra wody na {ref_chain:.0f} km rzeki')
        ax3.set_ylabel('Wysokość [m]')
        ax3.set_xlabel('Data')
    else:
        ax3.plot(ts_data.index, ts_data.values, color='gray', lw=1, alpha=0.4, label=f'WSE at km {ref_chain:.0f}')
        ax3.set_ylabel('WSE [m]')
        ax3.set_xlabel('Date')
    ax3.grid(True, linestyle=':', alpha=0.5)
    marker_ts, = ax3.plot([], [], marker='v', color='m', markersize=10, zorder=5, linestyle='None')
    v_line = ax3.axvline(dates[0], color='m', lw=1.2, ls='--', alpha=0.6)

    def update(frame_date):
        frame_date = pd.Timestamp(frame_date)

        # Sprawdzamy, czy mamy dane dla tej konkretnej daty w modelu
        if frame_date in df_wse.index:
            curr_wse = df_wse.loc[frame_date]
            curr_u = df_wse_u.loc[frame_date]
            curr_amp = df_wse_amp.loc[frame_date]
        else:
            # Jeśli nie ma danych dla tego dnia, bierzemy ostatnie dostępne (reindex/ffill)
            # lub po prostu nie zmieniamy linii (zostawiamy puste/poprzednie)
            return line_wse, scatter_vs, scatter_map, line_amp, marker_ts, v_line

        date_str = frame_date.strftime('%Y-%m-%d')

        # Model Update
        curr_wse = df_wse.loc[frame_date]
        curr_u = df_wse_u.loc[frame_date]
        curr_amp = df_wse_amp.loc[frame_date]
        chains = df_meta.set_index('id').loc[df_wse.columns, 'chain']

        line_wse.set_data(chains, curr_wse)
        if poly_wse[0] is not None: poly_wse[0].remove()
        poly_wse[0] = ax1.fill_between(chains, curr_wse - curr_u, curr_wse + curr_u, color='b', alpha=0.1, zorder=1)

        line_amp.set_data(chains, curr_amp)
        if poly_amp[0] is not None: poly_amp[0].remove()
        poly_amp[0] = ax2.fill_between(chains, curr_amp - curr_u, curr_amp + curr_u, color='g', alpha=0.1, zorder=1)

        # Fading Update
        vs_window = df_long[(df_long['dt'] <= frame_date) &
                            (df_long['dt'] >= frame_date - pd.Timedelta(days=2))].copy()

        if not vs_window.empty:
            vs_window['age'] = (frame_date - vs_window['dt']).dt.days
            rgba = np.zeros((len(vs_window), 4))
            rgba[:, 0] = 1.0
            rgba[:, 3] = vs_window['age'].map({0: 1.0, 1: 0.4, 2: 0.15}).fillna(0.05)

            scatter_vs.set_offsets(np.column_stack((vs_window['vs_chain'], vs_window['vs_wl'])))
            scatter_vs.set_facecolors(rgba)
            scatter_map.set_offsets(np.column_stack((vs_window['x'], vs_window['y'])))
            scatter_map.set_facecolors(rgba)
        else:
            scatter_vs.set_offsets(np.empty((0, 2)))
            scatter_map.set_offsets(np.empty((0, 2)))

        # Time Series Update
        marker_ts.set_data([frame_date], [ts_data.loc[frame_date]])
        v_line.set_xdata([frame_date])
        if polish:
            fig.suptitle(f'Przejście fali powodziowej na Środkowej Odrze w 2024 r. | Data: {date_str}', fontsize=15,
                         fontweight='bold')
        else:
            fig.suptitle(f'{riv} River profile | Date: {date_str}', fontsize=15, fontweight='bold')
        return line_wse, scatter_vs, scatter_map, line_amp, marker_ts, v_line

    ani = FuncAnimation(fig, update, frames=all_dates, interval=interval, blit=False)
    ax1.legend(loc='lower right', fontsize=8)
    ax3.legend(loc='upper left', fontsize=8)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.08, hspace=0.3)

    if save_path:
        # Bezpieczny FPS dla wideo
        calculated_fps = 1000 // interval if interval >= 10 else 25
        print(f"Zapisywanie wideo (Koder: mpeg4, FPS: {calculated_fps})...")

        try:
            # Używamy natywnego kodera mpeg4, który jest prawie zawsze dostępny
            ani.save(
                save_path,
                writer='ffmpeg',
                fps=calculated_fps,
                dpi=120,  # Rozsądny kompromis jakości do wielkości
                codec='mpeg4',
                extra_args=['-vcodec', 'mpeg4', '-q:v', '5', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
            )
            print(f"Zapis MP4 zakończony sukcesem: {save_path}")
        except Exception as e:
            print(f"Błąd zapisu MP4: {e}")
            # Ostateczne wyjście awaryjne do GIFa
            gif_path = save_path.replace('.mp4', '.gif')
            print(f"Próbuję zapisać jako GIF: {gif_path}")
            ani.save(gif_path, writer='pillow', fps=calculated_fps)
    else:
        plt.show(block=True)


# river = 'Oder'
river = 'Missouri'
data_csv_path = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_preds/reach_reg_results/results/timeseries/'
pattern = os.path.join(data_csv_path, f"{river}_RS*.csv")
river_ts_list = glob.glob(pattern)
riv_metadata_df = pd.read_csv(f'{data_csv_path}{river}_metadata_no_gdata.csv', sep=';')
df_WSE, df_WSE_U, df_META = prepare_river_data(river_ts_list, riv_metadata_df)
station_means = df_WSE.mean(axis=0)
df_WSE_AMP = df_WSE - station_means
data_pkl_path = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_preds/reach_reg_results/results/rs_stations/'
# df_res = collect_pkl_data(river, data_pkl_path)
df_res = collect_pkl_data_with_coords(river, data_pkl_path)
riv_path = f'/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python_clean/data/rivers/{river}_object.pkl'
with open(riv_path, "rb") as f:
    curr_riv = pickle.load(f)
gdf_riv = curr_riv.gdf.copy()
gdf_w = gpd.read_file('/Users/michalhalicki/Documents/nauka/dane_gis/world-administrative-boundaries/world-administrative-boundaries.shp')

animate_river_full_dashboard('2024-04-01', '2024-09-01', df_WSE, df_WSE_U, df_WSE_AMP, df_META, df_res, gdf_riv, river, gdf_w, '46334', interval=100)
# animate_river_full_dashboard('2024-04-01', '2024-09-01', df_WSE, df_WSE_U, df_WSE_AMP, df_META, df_res, gdf_riv, river, gdf_w, interval=100, save_path=f'/Users/michalhalicki/Desktop/{river}_profile.mp4')
# animate_river_full_dashboard('2024-08-01', '2024-11-01', df_WSE, df_WSE_U, df_WSE_AMP, df_META, df_res, gdf_riv, river, gdf_w, '23404', interval=150, polish=True, save_path=f'/Users/michalhalicki/Desktop/{river}_profile_pl.mp4')

fig, ax = plt.subplots()
ax.plot(df_META['chain'], df_META['mean'])
ax.grid()
plt.show(block=True)
