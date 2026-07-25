"""
create_city_map.py
One-time utility to download a city's road network for a given city-name
from OpenStreetMap and save it as an offline GraphML file.
"""
from pathlib import Path
import osmnx as ox
import argparse

# Configuration and defaults
#DEFAULT_CITY_NAME = "Downtown Austin, Texas, USA"
DEFAULT_CITY_NAME = "San Francisco, California, USA"
DEFAULT_NETWORK_TYPE = "drive"
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "static_city_map.graphml"

def build_city_graph(
    city_name: str = DEFAULT_CITY_NAME, 
    network_type: str = DEFAULT_NETWORK_TYPE,
    output_file: str = DEFAULT_OUTPUT_FILE,
):
    """
    Downloads a road network from OpenStreetMap and saves it
    as a GraphML file.
    Parameters
    ----------
    city_name : str, optional
        Name of the city/region to download.
        Defaults to DEFAULT_CITY_NAME.
    network_type: Defaults to drive.
    output_file: The file where to store the graph,
        defaults to DEFAULT_OUTPUT_FILE

    Returns
    -------
    networkx.MultiDiGraph
        Downloaded road network graph.
    """

    print(f"Downloading map for: {city_name} and network_type: {network_type}")

    """
    try:
        gdf = ox.geocode_to_gdf(place)
        print("Found!")
        print(gdf[["display_name"]])
    except Exception as e:
        print(f"Could not locate map for {city_name} in OpenStreetMap")
        exit(-1)
    """
    
    graph = ox.graph_from_place(
        city_name,
        network_type=network_type,
        simplify=True
    )

    # Optional but highly recommended for routing
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)

    print("Download complete...")

    print(f"Nodes : {len(graph.nodes)}")
    print(f"Edges : {len(graph.edges)}")

    ox.save_graphml(
        graph,
        output_file,
    )

    print(f"Graph saved to:\n{output_file}")

    return graph


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download an OpenStreetMap road network and save it as GraphML."
    )
    parser.add_argument(
        "--city",
        default=DEFAULT_CITY_NAME,
        help=f'City or region name (default: "{DEFAULT_CITY_NAME}")'
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output GraphML file (default: {DEFAULT_OUTPUT_FILE})"
    )
    return parser.parse_args()
    
    
def main():
    args = parse_arguments()
    build_city_graph(
        city_name=args.city,
        output_file=args.output,
    )

if __name__ == "__main__":
    main()