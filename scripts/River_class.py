import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union, linemerge, nearest_points

from shapely import MultiLineString, LineString
import json
import requests
import pandas as pd


dahiti_river_names_and_basins = {
    'Odra': {'river': 'Oder, River', 'river_names': ['Oder, River'], 'basin': 'Oder', 'up_reach': 24222900416, 'dn_reach': 24230000176, 'country': 'poland'},
    'Warta': {'river': 'Warta, River', 'river_names': ['Warta, River'], 'basin': 'Oder', 'up_reach': 24229000296, 'dn_reach': 24223000011, 'country': 'poland'},
    'Wisla': {'river': 'Wisła, River', 'river_names': ['Wisła, River', 'Vistula, River'], 'basin': 'Wisla', 'up_reach': 24244600846, 'dn_reach': 24230000025, 'country': 'poland'},
    'Bug': {'river': 'Bug, River', 'river_names': ['Bug, River'], 'basin': 'Wisla', 'up_reach': 24249900333, 'dn_reach': 24249100011, 'country': 'poland'},
    'Narew': {'river': 'Narew, River', 'river_names': ['Narew, River'], 'basin': 'Wisla', 'up_reach': 24248000166, 'dn_reach': 24245000011, 'country': 'poland'},
    'San': {'river': 'San, River', 'river_names': ['San, River'], 'basin': 'Wisla', 'up_reach': 24244900286, 'dn_reach': 24244700011, 'country': 'poland'},
    'Niemen': {'river': 'Nemunas, River', 'river_names': ['Nemunas, River'], 'basin': 'Neman', 'up_reach': 24260900146, 'dn_reach': 24250000436, 'country': 'lithuania'},
    'Elbe': {'river': 'Elbe, River', 'river_names': ['Elbe, River'], 'basin': 'Elbe', 'up_reach': 23286000676, 'dn_reach': 23270700021, 'country': 'germany'},
    'Rhine': {'river': 'Rhine, River', 'river_names': ['Rhine, River'], 'basin': 'Rhine', 'up_reach': 23269000384, 'dn_reach': 23250900015, 'country': 'germany'},
    'Danube': {'river': 'Danube, River', 'river_names': ['Danube, River'], 'basin': 'Danube', 'up_reach': 22799700131, 'dn_reach': 22710500031, 'country': 'danube'}
}

vs_with_neight_dams = {
    'Odra': [],
    'Warta': [],
    'Wisla': [],
    'Bug': [],
    'Narew': [],
    'San': [],
    'Niemen': [],
    'Rhine': [45499, 45497],
    'Elbe': [41949, 27218, 16651, 38476],
    'Danube': []
}

""" SWORD REACHES WITH A TRIBUTARY AT THE DOWNSTREAM EDGE. LISTS ARE SORTED FROM UP TO DOWNSTREAM """
river_tributary_reaches = {
    'Odra': [24222100011],  # Warta
    'Warta': [],
    'Wisla': [24244600451, 24244600041, 24244600011, 24244300011, 24244100011],  # Dunajec, Wisłoka, San, Pilica, Narew
    'Bug': [],
    'Narew': [],
    'San': [],
    'Niemen': [],
    'Rhine': [23267000011, 23265000011, 23263000011, 23261000281],  # Neckar, Main, Mosel, Ruhr
    'Elbe': [23285000031, 23285000011, 23281000241],  # Mulde, Saale, Havel
    'Danube': []
}


class River:
    def __init__(self, gdf, metrical_crs, name):
        self.gdf = gdf
        self.metrical_crs = metrical_crs
        self.name = name
        self.simplified_river = None
        self.dams = None
        self.tributary_chains = None

    # def get_simplified_geometry(self, buffer_distance=0.01):
    #     multi_line = unary_union(self.gdf['geometry'])
    #     temp_gdf = gpd.GeoDataFrame(geometry=[multi_line], crs=self.gdf.crs)
    #     temp_gdf = temp_gdf.to_crs(self.metrical_crs)
    #     buffered_geometry = temp_gdf.geometry.buffer(buffer_distance).unary_union
    #     boundary_lines = buffered_geometry.boundary
    #     if isinstance(boundary_lines, LineString):
    #         final_merged_geometry = boundary_lines
    #     else:
    #         final_merged_geometry = linemerge(boundary_lines)
    #     self.simplified_river = MultiLineString(lines=[final_merged_geometry])

    from shapely.geometry import LineString, MultiLineString, Point, Polygon, MultiPolygon
    from shapely.ops import unary_union, linemerge, nearest_points

    def get_simplified_geometry(self, order_column='order', tolerance=0.001, buffer_distance=0.0):
        """
        Tworzy jeden LineString rzeki, łącząc odcinki zgodnie z kolumną 'order'.
        Próbuje połączyć fizycznie sąsiednie odcinki, odwracając je, jeśli to konieczne.

        Args:
            order_column (str): Nazwa kolumny z kolejnością odcinków (np. 'order').
            tolerance (float): Maksymalna odległość między końcami linii, aby uznać je za stykające się.
                               W jednostkach CRS metrycznego (np. 0.001 metra = 1 mm).
            buffer_distance (float): Opcjonalna odległość buforowania dla wygładzenia/zamknięcia małych luk.
                                     Jeśli 0, buforowanie nie jest wykonywane.
        """
        print(f"Rozpoczynam tworzenie uproszczonej rzeki dla {self.name} zgodnie z kolumną '{order_column}'...")

        if order_column not in self.gdf.columns:
            print(f"Błąd: Kolumna '{order_column}' nie istnieje w GeoDataFrame. Nie można posortować geometrii.")
            self.simplified_river = None
            return

        # 1. Posortuj GeoDataFrame według kolumny 'order'
        # Ważne: to gdf powinno zawierać tylko reaches głównego nurtu, jeśli 'order' odzwierciedla tylko je
        # Jeśli 'order' jest w całym gdf, musisz upewnić się, że to co łączysz jest faktycznie głównym nurtem
        # Ale skoro masz już `order` to zakładamy, że to jest to co chcesz łączyć
        sorted_gdf = self.gdf.sort_values(by=order_column).reset_index(drop=True)

        # 2. Ręczne budowanie LineStringa z poszczególnych segmentów, dbając o kolejność wierzchołków
        final_coords = []
        current_line_segment = None  # Będzie przechowywać ostatnio dodany LineString

        for index, row in sorted_gdf.iterrows():
            segment_geom = row.geometry

            if not isinstance(segment_geom, LineString):
                print(
                    f"Ostrzeżenie: Segment {row['reach_id']} (order: {row[order_column]}) nie jest LineStringiem ({type(segment_geom)}). Pomijam.")
                continue

            if current_line_segment is None:
                # Jeśli to pierwszy segment, po prostu dodaj jego wierzchołki
                final_coords.extend(list(segment_geom.coords))
                current_line_segment = segment_geom
            else:
                # Sprawdź odległość między końcem obecnej linii a początkiem/końcem nowego segmentu
                current_end_point = Point(current_line_segment.coords[-1])
                segment_start_point = Point(segment_geom.coords[0])
                segment_end_point = Point(segment_geom.coords[-1])

                dist_end_to_start = current_end_point.distance(segment_start_point)
                dist_end_to_end = current_end_point.distance(segment_end_point)

                if dist_end_to_start < tolerance:
                    # Segment jest w dobrej orientacji, dodaj wierzchołki
                    if current_end_point.equals(segment_start_point):  # Jeśli punkty są identyczne
                        final_coords.extend(list(segment_geom.coords)[1:])
                    else:  # Jeśli są blisko, ale nie identyczne (mała luka, np. wynik zaokrągleń)
                        # Możesz tutaj wstawić punkt łączący, albo po prostu dodać wszystkie wierzchołki
                        final_coords.extend(list(segment_geom.coords))
                    current_line_segment = segment_geom  # Zaktualizuj ostatni segment
                elif dist_end_to_end < tolerance:
                    # Segment jest odwrócony, odwróć go i dodaj wierzchołki
                    reversed_segment_geom = segment_geom.reverse()
                    if current_end_point.equals(Point(reversed_segment_geom.coords[0])):
                        final_coords.extend(list(reversed_segment_geom.coords)[1:])
                    else:
                        final_coords.extend(list(reversed_segment_geom.coords))
                    current_line_segment = reversed_segment_geom  # Zaktualizuj ostatni segment
                else:
                    # Luka przestrzenna jest zbyt duża lub brak styku. Segment zostanie dodany, ale ciągłość nie będzie idealna.
                    # W tym przypadku, shapely będzie próbować utworzyć MultiLineString, jeśli nie ma fizycznego styku.
                    print(
                        f"Ostrzeżenie: Duża luka przestrzenna (> {tolerance:.4f} m) między segmentem {sorted_gdf.loc[index - 1, 'reach_id']} a {row['reach_id']} (order: {row[order_column]}).")
                    print(
                        f"  Odległość od końca poprzedniego do początku/końca obecnego: {dist_end_to_start:.4f} / {dist_end_to_end:.4f}.")
                    final_coords.extend(list(segment_geom.coords))  # Dodaj punkty obecnego segmentu mimo luki
                    current_line_segment = segment_geom  # Zaktualizuj ostatni segment (ważne dla kolejnej iteracji)

        if not final_coords:
            print("Brak wierzchołków do utworzenia LineStringa. simplified_river ustawiony na None.")
            self.simplified_river = None
            return

        # Utwórz ostateczny LineString z uporządkowanych wierzchołków
        temp_line = LineString(final_coords)
        print(f"Utworzono LineString z {len(final_coords)} wierzchołkami przed opcjonalnym buforowaniem.")

        # 3. Opcjonalne buforowanie dla wygładzenia lub zamknięcia bardzo małych luk
        if buffer_distance > 0 and not temp_line.is_empty:
            print(f"Stosuję buforowanie z dystansem: {buffer_distance} jednostek CRS metrycznego.")
            buffered_geometry = temp_line.buffer(buffer_distance).unary_union
            boundary_lines = buffered_geometry.boundary

            if isinstance(boundary_lines, LineString):
                self.simplified_river = boundary_lines
            elif isinstance(boundary_lines, MultiLineString):
                # Jeśli po buforowaniu nadal MultiLineString, linemerge pomoże połączyć
                # ale to już wskazuje na większe problemy (np. buforowanie tworzy wiele poligonów)
                self.simplified_river = linemerge(boundary_lines)
                print(
                    f"Po buforowaniu i linemerge nadal MultiLineString z {len(list(self.simplified_river.geoms))} częściami.")
            elif isinstance(boundary_lines, (Polygon, MultiPolygon)):
                print(
                    f"Ostrzeżenie: Wynik buforowania to poligon/multi-poligon. Używam LineStringa przed buforowaniem.")
                self.simplified_river = temp_line  # Wróć do linii przed buforowaniem
            else:
                print(
                    f"Ostrzeżenie: Nieoczekiwany typ geometrii po buforowaniu: {type(boundary_lines)}. Używam LineStringa przed buforowaniem.")
                self.simplified_river = temp_line  # Wróć do linii przed buforowaniem
        else:
            self.simplified_river = temp_line  # Bez buforowania, użyj utworzonego LineStringa

        print(f"Zakończono tworzenie uproszczonej rzeki. Typ: {type(self.simplified_river)}")
        if isinstance(self.simplified_river, (LineString, MultiLineString)):
            num_components = len(list(self.simplified_river.geoms)) if isinstance(self.simplified_river,
                                                                                  MultiLineString) else 1
            print(f"Końcowa liczba komponentów: {num_components}")
            if num_components > 1:
                print("UWAGA: simplified_river jest MultiLineStringiem! project() może działać niepoprawnie.")

        self.simplified_river = gpd.GeoSeries([temp_line], crs=4326).to_crs(self.metrical_crs).geometry[0]

    def get_chainage_of_point(self, point_x, point_y):
        point = gpd.GeoDataFrame(geometry=[Point(point_x, point_y)], crs=4326)
        point = point.to_crs(self.metrical_crs)
        closest_point = nearest_points(self.simplified_river, point.geometry[0])[0]
        return self.simplified_river.project(closest_point)

    def get_dams_chainages(self):
        dams = []
        for index, row in self.gdf.loc[self.gdf['type'] == 4].iterrows():
            pt1, pt2 = row.geometry.boundary.geoms[0], row.geometry.boundary.geoms[1]
            dams.append([self.get_chainage_of_point(pt1.x, pt2.y), self.get_chainage_of_point(pt2.x, pt2.y)])
        self.dams = dams

    def upload_tributary_chains(self, reaches):
        tribtr_chains = []
        for reach_id in reaches:
            curr_reach = self.gdf.loc[self.gdf['reach_id'] == reach_id].iloc[0]
            pt1 = curr_reach.geometry.boundary.geoms[0]
            tribtr_chains.append(self.get_chainage_of_point(pt1.x, pt1.y))
        self.tributary_chains = tribtr_chains

    def upload_dam_and_tributary_chains(self, reaches):
        self.get_dams_chainages()
        self.upload_tributary_chains(reaches)


def find_main_stream_recursively(df, current_id, source_id, path=None):
    """
    Finds the list of ID's of the main stream reaches from the current ID to the source
    (recursive version handling multiple IDs in 'rch_id_up').

    Args:
        df (pd.DataFrame): DataFrame with river reach information,
                           containing columns 'reach_id' (reach ID) and
                           'rch_id_up' (string with upstream reach IDs, space-separated).
        current_id (int): ID of the currently checked reach.
        source_id (int): ID of the source reach.
        path (list, optional): List of reaches visited so far. Defaults to None.

    Returns:
        list: List of reach IDs of the main stream from the current ID to the source,
              or None if no path to the source is found.
    """
    if path is None:
        path = [current_id]

    if current_id == source_id:
        return path

    reach = df[df['reach_id'] == current_id].iloc[0]
    rch_id_up_str = reach['rch_id_up']

    if pd.isna(rch_id_up_str):
        return None  # Reached the end of a branch

    next_ids = [int(id_str.strip()) for id_str in rch_id_up_str.split()]

    for next_id in next_ids:
        if next_id not in path:  # Prevent cycles
            result = find_main_stream_recursively(df, next_id, source_id, path + [next_id])
            if result is not None and result[-1] == source_id:
                return result

    return None


def find_main_stream(df, mouth_id, source_id):
    return find_main_stream_recursively(df, mouth_id, source_id)


def prepare_river_object(riv_path, riv, metric_crs):
    river_name, river_names, basin_name, up_reach, dn_reach, country = dahiti_river_names_and_basins[riv].values()
    sword_rivers = gpd.read_file(riv_path)
    river_reaches_li = find_main_stream(sword_rivers, dn_reach, up_reach)
    selected_river = sword_rivers[sword_rivers['reach_id'].isin(river_reaches_li)]
    selected_river = selected_river.sort_values('dist_out')
    selected_river['order'] = range(len(selected_river))
    current_river = River(selected_river, metric_crs, riv)
    current_river.get_simplified_geometry()
    return current_river


# river_name, basin_name, up_reach, dn_reach = dahiti_river_names_and_basins['odra'].values()
# riv_path = '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp'
# list_url = 'https://dahiti.dgfi.tum.de/api/v2/list-targets/'
# args = {'api_key': '59B654E28B331DF19DFD0E252F4627EB723A281AC163AC648A82C841636131D6', 'basin': basin_name}
# response = requests.post(list_url, json=args)
# vs_metadata = []
# if response.status_code == 200:
#     data = json.loads(response.text)
#     for vs in data['data']:
#         if vs['target_name'] == river_name:
#             print(vs)
#             vs_metadata.append(vs)
#
# sword_rivers = gpd.read_file(riv_path)
# river_reaches_li = find_main_stream(sword_rivers, dn_reach, up_reach)
# selected_river = sword_rivers[sword_rivers['reach_id'].isin(river_reaches_li)]
#
# stations = []
# current_river = River(selected_river, '2180')
# current_river.get_simplified_geometry()
# for i in range(len(vs_metadata)):
#     vs_x, vs_y = vs_metadata[i]['longitude'], vs_metadata[i]['latitude']
#     chainage = current_river.get_chainage_of_point(vs_x, vs_y)
#     stations.append([vs_x, vs_y, chainage])
# stations_df = pd.DataFrame(stations, columns=['x', 'y', 'dist'])
# stations_gdf = gpd.GeoDataFrame(stations_df, geometry=gpd.points_from_xy(stations_df.x, stations_df.y), crs=4326)

# fig, ax = plt.subplots()
# current_river.gdf.plot(ax=ax)
# stations_gdf.plot(ax=ax, column='dist', cmap='viridis', legend=True)
# ax.set_title(f'{river_name}')
# plt.show(block='True')
# print(1)
