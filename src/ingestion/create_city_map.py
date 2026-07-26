"""
create_city_map.py
One-time utility to download a city's road network for a given city-name
from OpenStreetMap and save it as an offline GraphML file.

Usage (from project root):
    python -m src.ingestion.create_city_map
    python -m src.ingestion.create_city_map --city "Mumbai, Maharashtra, India"
"""
from pathlib import Path
import osmnx as ox
import argparse
import ssl
import urllib.request

from src.config import MAP_DIRECTORY, MAP_FILENAME

# Disable SSL verification for corporate networks that intercept HTTPS traffic.
# This affects the one-time OSM download only; all subsequent pipeline steps are offline.
ox.settings.requests_kwargs = {"verify": False}

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration and defaults
DEFAULT_CITY_NAME = "Bengaluru, Karnataka, India"
DEFAULT_NETWORK_TYPE = "drive"
DEFAULT_OUTPUT_FILE = MAP_DIRECTORY / MAP_FILENAME

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

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if Path(output_file).exists():
        print(f"Map already exists at {output_file} - skipping download.")
        return ox.load_graphml(output_file)

    print(f"Downloading map for: {city_name} and network_type: {network_type}")

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