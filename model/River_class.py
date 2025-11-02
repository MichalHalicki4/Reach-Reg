import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import linemerge, nearest_points
from shapely import MultiLineString, LineString


class River:
    def __init__(self, gdf, metrical_crs, name):
        self.gdf = gdf
        self.metrical_crs = metrical_crs
        self.name = name
        self.simplified_river = None
        self.dams = None
        self.tributary_chains = None

    def get_simplified_geometry(self, order_column='order', tolerance=0.001, buffer_distance=0.0):
        """
        Creates a single LineString for a river by joining segments according to the “order” column.
        Attempts to join physically adjacent segments, reversing them if necessary.

        Args:
            order_column (str): Name of the column with the order of segments (e.g., “order”).
            tolerance (float): Maximum distance between line ends to consider them touching in metric CRS units (metre).
            buffer_distance (float): Optional buffering distance for smoothing/closing small gaps. 0 = no buffering.
        """
        print(f"Starting creation of a simplified river for {self.name} according to the column '{order_column}'...")

        if order_column not in self.gdf.columns:
            print(f"Error: Column '{order_column}' does not exist in the GeoDataFrame. Cannot sort geometry.")
            self.simplified_river = None
            return

        sorted_gdf = self.gdf.sort_values(by=order_column).reset_index(drop=True)
        final_coords = []
        current_line_segment = None

        for index, row in sorted_gdf.iterrows():
            segment_geom = row.geometry
            if not isinstance(segment_geom, LineString):
                print(f"Warning: Segment {row['reach_id']} (order: {row[order_column]}) is not a LineString "
                      f"({type(segment_geom)}). Skipping.")
                continue

            if current_line_segment is None:
                final_coords.extend(list(segment_geom.coords))
                current_line_segment = segment_geom
            else:
                current_end_point = Point(current_line_segment.coords[-1])
                segment_start_point = Point(segment_geom.coords[0])
                segment_end_point = Point(segment_geom.coords[-1])

                dist_end_to_start = current_end_point.distance(segment_start_point)
                dist_end_to_end = current_end_point.distance(segment_end_point)

                if dist_end_to_start < tolerance:
                    if current_end_point.equals(segment_start_point):
                        final_coords.extend(list(segment_geom.coords)[1:])
                    else:
                        final_coords.extend(list(segment_geom.coords))
                    current_line_segment = segment_geom
                elif dist_end_to_end < tolerance:
                    reversed_segment_geom = segment_geom.reverse()
                    if current_end_point.equals(Point(reversed_segment_geom.coords[0])):
                        final_coords.extend(list(reversed_segment_geom.coords)[1:])
                    else:
                        final_coords.extend(list(reversed_segment_geom.coords))
                    current_line_segment = reversed_segment_geom  # Zaktualizuj ostatni segment
                else:
                    print(
                        f"Warning: Large spatial gap (> {tolerance:.4f} m) between segment "
                        f"{sorted_gdf.loc[index - 1, 'reach_id']} and {row['reach_id']} (order: {row[order_column]}).")
                    print(
                        f"  Distance from end of previous to start/end of current: "
                        f"{dist_end_to_start:.4f} / {dist_end_to_end:.4f}.")
                    final_coords.extend(list(segment_geom.coords))
                    current_line_segment = segment_geom

        if not final_coords:
            print("No vertices to create LineString. simplified_river set to None.")
            self.simplified_river = None
            return

        temp_line = LineString(final_coords)
        print(f"Created LineString with {len(final_coords)} vertices before optional buffering.")

        if buffer_distance > 0 and not temp_line.is_empty:
            print(f"Applying buffering with distance: {buffer_distance} units of the metric CRS.")
            buffered_geometry = temp_line.buffer(buffer_distance).unary_union
            boundary_lines = buffered_geometry.boundary

            if isinstance(boundary_lines, LineString):
                self.simplified_river = boundary_lines
            elif isinstance(boundary_lines, MultiLineString):
                self.simplified_river = linemerge(boundary_lines)
                print(f"After buffering and linemerge, still a MultiLineString with "
                      f"{len(list(self.simplified_river.geoms))} parts.")
            elif isinstance(boundary_lines, (Polygon, MultiPolygon)):
                print(f"Warning: Buffering result is a Polygon/MultiPolygon. Using LineString before buffering.")
                self.simplified_river = temp_line
            else:
                print(f"Warning: Unexpected geometry type after buffering: {type(boundary_lines)}. "
                      f"Using LineString before buffering.")
                self.simplified_river = temp_line
        else:
            self.simplified_river = temp_line

        print(f"Finished creating simplified river. Type: {type(self.simplified_river)}")
        if isinstance(self.simplified_river, (LineString, MultiLineString)):
            num_components = len(list(self.simplified_river.geoms)) if isinstance(self.simplified_river,
                                                                                  MultiLineString) else 1
            print(f"Final number of components: {num_components}")
            if num_components > 1:
                print("ATTENTION: simplified_river is a MultiLineString! project() may not work correctly.")

        self.simplified_river = gpd.GeoSeries([temp_line], crs=4326).to_crs(self.metrical_crs).geometry[0]

    def get_chainage_of_point(self, point_x, point_y):
        """
        Calculates the chainage (distance from the river's start) of a given point on the simplified river geometry.
        The function projects the nearest point on the simplified river line onto the line to determine the chainage.

        Args:
            point_x (float): The X coordinate (longitude) of the point in EPSG:4326.
            point_y (float): The Y coordinate (latitude) of the point in EPSG:4326.

        Returns:
            float: The chainage of the point in the metrical CRS units (e.g., metres).
        """
        point = gpd.GeoDataFrame(geometry=[Point(point_x, point_y)], crs=4326)
        point = point.to_crs(self.metrical_crs)
        closest_point = nearest_points(self.simplified_river, point.geometry[0])[0]
        return self.simplified_river.project(closest_point)

    def get_dams_chainages(self):
        """
        Calculates and stores the chainage (distance from the river's start) for all dam structures.
        It iterates over GeoDataFrame entries where 'type' is 4, calculates the chainage for the boundary points of
            each dam, and saves the results.

        Args:
            self: Instance of the class (requires self.gdf, self.get_chainage_of_point, and expects self.dams to be
                initialized/updated).
        """
        dams = []
        for index, row in self.gdf.loc[self.gdf['type'] == 4].iterrows():
            pt1, pt2 = row.geometry.boundary.geoms[0], row.geometry.boundary.geoms[1]
            dams.append([self.get_chainage_of_point(pt1.x, pt1.y), self.get_chainage_of_point(pt2.x, pt2.y)])
        self.dams = dams

    def upload_tributary_chains(self, reaches):
        """
        Calculates the chainage (distance from the river's start) for the starting point of specified tributary reaches.
        It iterates through the provided list of reach IDs, finds the first boundary point of each reach's geometry,
            and stores its chainage.

        Args:
            self: Instance of the class (requires self.gdf, self.get_chainage_of_point, and expects
                self.tributary_chains to be initialized/updated).
            reaches (list): A list of 'reach_id' values (integers or strings) identifying the tributary segments to
                process.
        """
        tribtr_chains = []
        for reach_id in reaches:
            curr_reach = self.gdf.loc[self.gdf['reach_id'] == reach_id].iloc[0]
            pt1 = curr_reach.geometry.boundary.geoms[0]
            tribtr_chains.append(self.get_chainage_of_point(pt1.x, pt1.y))
        self.tributary_chains = tribtr_chains

    def upload_dam_and_tributary_chains(self, reaches):
        """
        Orchestrates the calculation and storage of chainages for both dam structures and specified tributary reaches.
        It calls the 'get_dams_chainages' method to process dams and the 'upload_tributary_chains' method to process
            tributaries.

            Args:
                self: Instance of the class (requires access to methods get_dams_chainages and upload_tributary_chains).
                reaches (list): A list of 'reach_id' values (integers or strings) identifying the tributary segments to
                    process.
            """
        self.get_dams_chainages()
        self.upload_tributary_chains(reaches)
